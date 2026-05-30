
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import ConcatDataset, WeightedRandomSampler
from torchvision import transforms
import matplotlib.pyplot as plt

import numpy as np
import cv2
import os
import io
from itertools import groupby
import time

from pathlib import Path
from skimage.filters import threshold_sauvola

import editdistance
import random

def _unwrap(model):
    """Devuelve el módulo original si está compilado con torch.compile, si no
    devuelve el módulo tal cual. Evita el prefijo '_orig_mod.' en state_dict."""
    return model._orig_mod if hasattr(model, '_orig_mod') else model

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1    = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1      = nn.BatchNorm2d(out_channels)
        self.conv2    = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2      = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out  = F.relu(self.bn1(self.conv1(x)))
        out  = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class CNNFeatureExtractor(nn.Module):
    """
    CNN aligerada con pool(2,2) en layer1 (FIX 1).

    Flujo para entrada H=64, W=256:
        conv_init 3×3   →  (B,  32, 64, 256)
        layer1 pool(2,2)→  (B,  32, 32, 128)  ← W÷2: T=128 al LSTM
        layer2 pool(2,1)→  (B,  64, 16, 128)
        layer3 pool(2,1)→  (B, 128,  8, 128)
        layer4 AdaptAvg →  (B, 256,  1, 128)
        squeeze+permute →  (B, 128, 256)
    """

    def __init__(self, input_channels=1, hidden_channels=32):
        super().__init__()

        self.conv_init = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )

        c = hidden_channels  # 32

        # FIX 1: pool(2,2) — reduce W a la mitad, T=128 en lugar de T=256/320
        self.layer1 = nn.Sequential(
            ResidualBlock(c, c),
            ResidualBlock(c, c),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        # Layers 2-3: pool(2,1) — solo reducen H, W se mantiene
        self.layer2 = nn.Sequential(
            ResidualBlock(c, c * 2, stride=1),
            ResidualBlock(c * 2, c * 2),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        )
        c = c * 2  # 64

        self.layer3 = nn.Sequential(
            ResidualBlock(c, c * 2, stride=1),
            ResidualBlock(c * 2, c * 2),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        )
        c = c * 2  # 128

        # Layer 4: colapsa H a 1 con AdaptiveAvgPool
        self.layer4 = nn.Sequential(
            ResidualBlock(c, c * 2, stride=1),
            ResidualBlock(c * 2, c * 2),
            nn.AdaptiveAvgPool2d((1, None))
        )
        c = c * 2  # 256

        self.out_channels = c

    def forward(self, x):
        x = self.conv_init(x)   # (B,  32, 64, W)
        x = self.layer1(x)      # (B,  32, 32, W/2)
        x = self.layer2(x)      # (B,  64, 16, W/2)
        x = self.layer3(x)      # (B, 128,  8, W/2)
        x = self.layer4(x)      # (B, 256,  1, W/2)
        x = x.squeeze(2)        # (B, 256, W/2)
        x = x.permute(0, 2, 1)  # (B, W/2, 256)  →  T = W/2 timesteps
        return x

class BiLSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=1, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.out_features = hidden_size * 2

    def forward(self, x):
        out, _ = self.lstm(x)
        return out

class CRNN(nn.Module):
    """
    CRNN-Lite v2.
        CNN:   ~2.1M params  (32→64→128→256 ch, pool(2,2) en layer1)
        LSTM:  ~0.7M params  (input=256, hidden=128×2, 1 capa bidir.)
        Head:  ~17K  params  (Linear 256 → vocab+1)
        Total: ~3.2M params
    """

    def __init__(self, num_classes, img_height=64,
                 hidden_channels=32, lstm_hidden=128, dropout=0.3):
        super().__init__()

        self.cnn        = CNNFeatureExtractor(1, hidden_channels)
        cnn_out         = self.cnn.out_channels  # 256

        self.bilstm     = BiLSTMBlock(cnn_out, lstm_hidden, num_layers=1, dropout=dropout)
        self.dropout    = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(lstm_hidden * 2, num_classes + 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features  = self.cnn(x)                     # (B, T, 256)
        recurrent = self.bilstm(features)            # (B, T, 256)
        recurrent = self.dropout(recurrent)
        logits    = self.classifier(recurrent)       # (B, T, vocab+1)
        log_probs = F.log_softmax(logits, dim=2)
        return log_probs.permute(1, 0, 2)            # (T, B, vocab+1)

class CTCDecoder:
    def __init__(self, charset, blank_idx=0):
        self.charset   = charset
        self.blank_idx = blank_idx
        self.idx2char  = {i + 1: c for i, c in enumerate(charset)}
        self.idx2char[0] = '<blank>'

    def greedy_decode(self, log_probs):
        best_path = torch.argmax(log_probs, dim=2).permute(1, 0)
        decoded   = []
        for seq in best_path:
            chars = [self.idx2char.get(idx, '?')
                     for idx, _ in groupby(seq.tolist())
                     if idx != self.blank_idx]
            decoded.append(''.join(chars))
        return decoded

    def beam_search_decode(self, log_probs, beam_width=10):
        probs = torch.exp(log_probs[:, 0, :]).cpu().numpy()
        T, C  = probs.shape
        beams = [(0.0, [])]
        for t in range(T):
            new_beams = {}
            for score, seq in beams:
                for c in range(C):
                    p = probs[t, c]
                    if p == 0:
                        continue
                    ns = score + np.log(p + 1e-10)
                    if c == self.blank_idx:
                        key, new_seq = tuple(seq), seq
                    elif seq and c == seq[-1]:
                        key, new_seq = tuple(seq), seq
                    else:
                        new_seq = seq + [c]
                        key     = tuple(new_seq)
                    if key not in new_beams or new_beams[key][0] < ns:
                        new_beams[key] = (ns, new_seq)
            beams = sorted(new_beams.values(), key=lambda x: x[0], reverse=True)[:beam_width]
        if not beams:
            return ''
        return ''.join(self.idx2char.get(i, '?') for i in beams[0][1])

class HTRTrainer:
    def __init__(self, model, charset, device='cuda', lr=1e-3,
                 checkpoint_dir='checkpoints',
                 hidden_channels=32, lstm_hidden=128, img_height=64,
                 weight_decay=1e-4):

        self.model          = model.to(device)
        self.device         = device
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.config = {
            'hidden_channels': hidden_channels,
            'lstm_hidden':     lstm_hidden,
            'img_height':      img_height,
            'num_classes':     len(charset)
        }

        self.criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)

        # AdamW con weight_decay excluido para BatchNorm y bias.
        # Estándar moderno: aplicar wd a pesos de Conv/Linear; NO a normalizaciones
        # ni bias, porque sesga la estadística de BN y rompe equivarianzas.
        decay, no_decay = [], []
        for name, p in _unwrap(model).named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim <= 1 or name.endswith('.bias') or 'bn' in name.lower():
                no_decay.append(p)
            else:
                decay.append(p)
        self.optimizer = optim.AdamW(
            [{'params': decay,    'weight_decay': weight_decay},
             {'params': no_decay, 'weight_decay': 0.0}],
            lr=lr
        )

        # Scheduler por defecto: ReduceLROnPlateau. Se puede SUSTITUIR desde
        # el script principal por OneCycleLR (recomendado para fase 1) o por
        # un schedule más conservador en fine-tuning.
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

        self.use_amp = (device == 'cuda' and torch.cuda.is_available())
        self.scaler  = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.decoder     = CTCDecoder(charset=charset, blank_idx=0)
        self.best_cer    = float('inf')
        self.start_epoch = 1

        self.metrics_path = os.path.join(checkpoint_dir, 'metrics_history.pt')
        self.history      = self._load_history()

    def _load_history(self):
        if os.path.exists(self.metrics_path):
            data = torch.load(self.metrics_path, weights_only=False)
            print(f"  Historial cargado: {len(data['loss_train'])} épocas previas")
            return data
        return {'epoch': [], 'loss_train': [], 'loss_val': [], 'cer': [], 'wer': []}

    def _save_history(self):
        torch.save(self.history, self.metrics_path)

    def _append_metrics(self, epoch, loss_train, loss_val, cer, wer, extras=None):
        self.history['epoch'].append(epoch)
        self.history['loss_train'].append(loss_train)
        self.history['loss_val'].append(loss_val)
        self.history['cer'].append(cer)
        self.history['wer'].append(wer)
        self.history.setdefault('lr', []).append(
                self.optimizer.param_groups[0]['lr']
            )
        # Métricas extra (monitoring multi-val: val_unseen, val_real, IAM val...)
        if extras:
            extras_hist = self.history.setdefault('extras', {})
            for name, metrics in extras.items():
                slot = extras_hist.setdefault(name, {'cer': [], 'wer': [], 'loss': []})
                slot['cer'].append(metrics['cer'])
                slot['wer'].append(metrics['wer'])
                slot['loss'].append(metrics['loss'])
        self._save_history()

    def train_epoch(self, dataloader):
        self.model.train()
        running_loss = 0.0

        for batch_idx, (images, targets, target_lengths, texts, input_lengths) in enumerate(dataloader):
            images = images.to(self.device, non_blocking=True)  # non_blocking con pin_memory

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                log_probs   = self.model(images)
                current_T   = log_probs.size(0)
                batch_size  = images.size(0)
                input_lengths = torch.full((batch_size,), current_T, dtype=torch.long)
                loss = self.criterion(log_probs, targets, input_lengths, target_lengths)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0) #2.0########################

            scale_before = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            skip_lr = scale_before > self.scaler.get_scale()
            if not skip_lr and isinstance(self.scheduler,
                                          torch.optim.lr_scheduler.OneCycleLR):
                self.scheduler.step()

            running_loss += loss.item()

            if (batch_idx + 1) % 20 == 0:
                print(f"  Batch {batch_idx + 1}/{len(dataloader)} | "
                      f"Loss: {loss.item():.4f}")

        return running_loss / len(dataloader)

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        running_loss = 0.0
        all_preds    = []
        all_targets  = []

        for images, targets, target_lengths, texts, input_lengths in dataloader:
            images = images.to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                log_probs   = self.model(images)
                current_T   = log_probs.size(0)
                batch_size  = images.size(0)
                input_lengths = torch.full((batch_size,), current_T, dtype=torch.long)
                loss = self.criterion(log_probs, targets, input_lengths, target_lengths)

            running_loss += loss.item()
            all_preds.extend(self.decoder.greedy_decode(log_probs))
            all_targets.extend(self._untokenize(targets, target_lengths))

        cer = self.compute_cer(all_preds, all_targets)
        wer = self.compute_wer(all_preds, all_targets)
        return running_loss / len(dataloader), cer, wer

    def _untokenize(self, targets, target_lengths):
        texts, offset = [], 0
        for length in target_lengths.tolist():
            idx  = targets[offset: offset + length].tolist()
            texts.append(''.join(self.decoder.idx2char.get(i, '?') for i in idx))
            offset += length
        return texts

    def compute_cer(self, predictions, targets):
        total_edit, total_chars = 0, 0
        for pred, tgt in zip(predictions, targets):
            total_edit  += editdistance.eval(pred, tgt)
            total_chars += len(tgt)
        return total_edit / max(total_chars, 1)

    def compute_wer(self, predictions, targets):
        total_edit, total_words = 0, 0
        for pred, tgt in zip(predictions, targets):
            total_edit  += editdistance.eval(pred.split(), tgt.split())
            total_words += len(tgt.split())
        return total_edit / max(total_words, 1)

    def save_checkpoint(self, epoch, cer, filename=None):
        if filename is None:
            filename = f'checkpoint_epoch{epoch:03d}_cer{cer:.4f}.pt'
        path = os.path.join(self.checkpoint_dir, filename)
        # Guardamos siempre el state_dict del módulo SIN compilar para que
        # el checkpoint sea portable entre sesiones compilen o no.
        torch.save({
            'epoch':                epoch,
            'model_state_dict':     _unwrap(self.model).state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict':    self.scaler.state_dict(),
            'best_cer':             self.best_cer,
            'config':               self.config,
        }, path)
        print(f"  ✓ Checkpoint: {path}")

    def load_checkpoint(self, checkpoint_path):
        print(f"Cargando checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        # Cargamos en el modelo desenrollado; vale igual si self.model está
        # compilado o no, porque _unwrap devuelve el original en ambos casos.
        _unwrap(self.model).load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if 'scaler_state_dict' in ckpt:
            self.scaler.load_state_dict(ckpt['scaler_state_dict'])
        self.best_cer    = ckpt['best_cer']
        self.start_epoch = ckpt['epoch'] + 1
        print(f"  Reanudando desde época {self.start_epoch} | "
              f"Mejor CER: {self.best_cer:.4f} | "
              f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")

    def train(self, train_loader, val_loader, num_epochs=100,
              early_stopping_patience=10,
              extra_val_loaders=None,
              checkpoint_score_fn=None):
        """
        train_loader, val_loader: train y val PRINCIPAL (early stopping y scheduler).
        extra_val_loaders: dict {nombre: DataLoader} para monitoreo adicional.
                           Se evalúa cada época, se reporta, pero NO afecta early
                           stopping ni decisión de mejor checkpoint (a menos que
                           se use a través de checkpoint_score_fn).
        checkpoint_score_fn: callable(cer_principal, extras_dict) -> float a MINIMIZAR.
                             Si es None, se usa el CER del val_loader principal.
                             Útil en fase 2 para ponderar val_seen vs val_unseen.
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        print("=" * 65)
        print(f"  CRNN-Lite v2  —  {total_params:,} parámetros")
        print(f"  AMP: {'ON' if self.use_amp else 'OFF'}  |  "
              f"Device: {self.device}")
        if extra_val_loaders:
            print(f"  Extra val loaders (monitoring): {list(extra_val_loaders.keys())}")
        if checkpoint_score_fn is not None:
            print(f"  Score de checkpoint: función custom (ponderada)")
        print("=" * 65)

        epochs_without_improvement = 0
        best_score = float('inf')

        for epoch in range(self.start_epoch, num_epochs + 1):
            t0         = time.time()
            train_loss = self.train_epoch(train_loader)

            # Val principal (drive early stopping y scheduler)
            val_loss, cer, wer = self.evaluate(val_loader)

            # Vals extra (monitoring, sin efecto en decisiones salvo via score_fn)
            extras = {}
            if extra_val_loaders:
                for name, loader in extra_val_loaders.items():
                    ev_loss, ev_cer, ev_wer = self.evaluate(loader)
                    extras[name] = {'loss': ev_loss, 'cer': ev_cer, 'wer': ev_wer}

            if not isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                self.scheduler.step(val_loss)

            # Score para early stopping y best
            if checkpoint_score_fn is not None:
                score = checkpoint_score_fn(cer, extras)
            else:
                score = cer

            elapsed    = time.time() - t0
            current_lr = self.optimizer.param_groups[0]['lr']

            print(f"Época {epoch:3d}/{num_epochs} | "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                  f"CER: {cer:.4f} | WER: {wer:.4f} | "
                  f"LR: {current_lr:.2e} | {elapsed:.0f}s")
            for name, m in extras.items():
                print(f"           [{name}] CER: {m['cer']:.4f} | WER: {m['wer']:.4f} | Loss: {m['loss']:.4f}")
            if checkpoint_score_fn is not None:
                print(f"           score = {score:.4f}")

            self._append_metrics(epoch, train_loss, val_loss, cer, wer, extras)
            self.save_checkpoint(epoch, cer)

            if score < best_score:
                best_score = score
                self.best_cer = cer  # mantener compatibilidad para load_checkpoint
                epochs_without_improvement = 0
                self.save_checkpoint(epoch, cer, 'best_model.pt')
                print(f"  ★ Nuevo mejor score: {score:.4f}  (cer principal={cer:.4f})")
            else:
                epochs_without_improvement += 1
                print(f"  Sin mejora: {epochs_without_improvement}/{early_stopping_patience}")

            # Alarma de olvido si hay un loader de IAM en extras
            if 'iam_val' in extras and extras['iam_val']['cer'] > 0.20:
                print(f"  ⚠ CER IAM = {extras['iam_val']['cer']:.4f} — "
                      f"posible olvido, considera subir % IAM en el sampler")

            # Penalización adicional si val loss sube tres épocas seguidas
            if len(self.history['loss_val']) >= 3:
                last_three = self.history['loss_val'][-3:]
                val_loss_rising = all(last_three[i] < last_three[i+1]
                                     for i in range(len(last_three)-1))
                if val_loss_rising:
                    epochs_without_improvement += 1

            if epochs_without_improvement >= early_stopping_patience:
                print(f"\n  Early stopping en época {epoch}. "
                          f"Mejor score: {best_score:.4f}")
                break

    def plot_metrics(self, save_path=None):
        if not self.history['epoch']:
            print("Sin métricas aún.")
            return
        epochs = self.history['epoch']
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('CRNN-Lite v2 — Curvas de entrenamiento', fontsize=14)
        ax1.plot(epochs, self.history['loss_train'], label='Train', color='steelblue', lw=2)
        ax1.plot(epochs, self.history['loss_val'],   label='Val',   color='tomato',    lw=2)
        ax1.set(xlabel='Época', ylabel='CTC Loss', title='Loss')
        ax1.legend(); ax1.grid(alpha=0.3)
        ax2.plot(epochs, self.history['cer'], label='CER', color='darkorange', lw=2)
        ax2.plot(epochs, self.history['wer'], label='WER', color='purple',     lw=2)
        ax2.set(xlabel='Época', ylabel='Tasa de error', title='CER / WER', ylim=(0, 1.05))
        ax2.legend(); ax2.grid(alpha=0.3)
        # Marcar sesiones
        for i in range(1, len(epochs)):
            if epochs[i] != epochs[i - 1] + 1:
                for ax in (ax1, ax2):
                    ax.axvline(x=epochs[i] - 0.5, color='gray', ls=':', alpha=0.6)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

def collate_fn(batch):
    images, labels, texts = zip(*batch)
    widths = [img.shape[2] for img in images]
    max_w  = max(widths)

    if len(set(widths)) > 1:
        padded = []
        for img in images:
            diff = max_w - img.shape[2]
            if diff > 0:
                pad = torch.ones(1, img.shape[1], diff, dtype=img.dtype)
                img = torch.cat([img, pad], dim=2)
            padded.append(img)
        images = torch.stack(padded, 0)
    else:
        images = torch.stack(list(images), 0)

    return (images,
            torch.cat(labels, dim=0),
            torch.tensor([len(l) for l in labels], dtype=torch.long),
            texts,
            torch.tensor(widths, dtype=torch.long))

def denoise(image):
    no_salt_pepper = cv2.medianBlur(image, ksize=3)
    return cv2.fastNlMeansDenoising(
        no_salt_pepper, h=10, templateWindowSize=7, searchWindowSize=21
    )
 
def enhance_contrast(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)
 
def background_estimation_poly(image, degree=2):
    h, w = image.shape
    step = 4
    ys, xs = np.mgrid[0:h:step, 0:w:step]
    zs     = image[::step, ::step].astype(np.float64)
 
    coords = np.column_stack([xs.ravel() / w, ys.ravel() / h])
    terms  = []
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            terms.append((coords[:, 0] ** i) * (coords[:, 1] ** j))
    A = np.column_stack(terms)
    coeffs, _, _, _ = np.linalg.lstsq(A, zs.ravel(), rcond=None)
 
    ys_full, xs_full = np.mgrid[0:h, 0:w]
    coords_full      = np.column_stack([xs_full.ravel() / w, ys_full.ravel() / h])
    terms_full       = []
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            terms_full.append((coords_full[:, 0] ** i) * (coords_full[:, 1] ** j))
    A_full     = np.column_stack(terms_full)
    background = (A_full @ coeffs).reshape(h, w)
    background = np.clip(background, 1, 255).astype(np.float64)
 
    corrected = np.clip((image.astype(np.float64) / background) * 255, 0, 255)
    return corrected.astype(np.uint8)
 
def resize_height_only(image: np.ndarray, img_height: int) -> np.ndarray:
    h, w   = image.shape
    new_w  = max(int(w * img_height / h), 1)
    return cv2.resize(image, (new_w, img_height), interpolation=cv2.INTER_AREA)
 
def resize_img(image: np.ndarray, img_height: int, img_width: int) -> np.ndarray:
    h, w  = image.shape
    new_w = max(int(w * img_height / h), 1)
    img   = cv2.resize(image, (new_w, img_height), interpolation=cv2.INTER_AREA)
 
    if new_w < img_width:
        pad = np.ones((img_height, img_width - new_w), dtype=np.uint8) * 255
        img = np.hstack([img, pad])
    else:
        img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_AREA)
 
    return img
 
def _binarize(gray: np.ndarray) -> np.ndarray:
    background = background_estimation_poly(gray)
    no_noise   = denoise(background)
    enhanced   = enhance_contrast(no_noise)
    normalized = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
    thresh     = threshold_sauvola(normalized, window_size=25, k=0.2)
    binarized  = ((normalized >= thresh) * 255).astype(np.uint8)
    return binarized
 
def preprocess_array(gray: np.ndarray, img_height: int, img_width: int):
    binarized  = _binarize(gray)
    resized    = resize_img(binarized, img_height, img_width)
    background = background_estimation_poly(gray)
    no_noise   = denoise(background)
    enhanced   = enhance_contrast(no_noise)
    normalized = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
    return (gray, background, no_noise, enhanced, normalized, binarized, resized)
 
def preprocess_image(image_path: str, img_height: int, img_width: int):
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return preprocess_array(gray, img_height, img_width)
 
def preprocess_image_binarized(image_path: str, img_height: int) -> np.ndarray:
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"No se pudo leer: {image_path}")
    binarized = _binarize(gray)
    return resize_height_only(binarized, img_height)
 
def preprocess_array_binarized(gray: np.ndarray, img_height: int) -> np.ndarray:
    binarized = _binarize(gray)
    return resize_height_only(binarized, img_height)
 
def _png_encode(img: np.ndarray) -> bytes:
    encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 9]
    success, buf  = cv2.imencode('.png', img, encode_params)
    if not success:
        raise RuntimeError("Error al codificar PNG")
    return buf.tobytes()
 
def _png_decode(png_bytes: bytes) -> np.ndarray:
    buf = np.frombuffer(png_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError("Error al decodificar PNG")
    return img
 
class HTRDataset(Dataset):
 
    def __init__(self, image_paths, labels, charset,
                 img_height=32, img_width=256, augment=False):
        self.image_paths = image_paths
        self.labels      = list(labels.values()) if isinstance(labels, dict) else list(labels)
        self.img_height  = img_height
        self.img_width   = img_width
        self.char2idx    = {c: i + 1 for i, c in enumerate(charset)}
        self.charset     = charset
        self.normalize   = transforms.Normalize(mean=[0.5], std=[0.5])
        self.augment     = augment   # ← solo se aplica en train con sintético
 
        self._tensors_cache = None
        self._images_png    = None
        self._labels_cache  = None

    def _apply_augmentation(self, img: np.ndarray) -> np.ndarray:
        """Augmentations leves en uint8 (CPU, dentro del worker)."""
        h, w = img.shape

        # 1) Translation aleatoria ±3 px en x, ±2 px en y (rellena con 255=blanco)
        tx = random.randint(-3, 3)
        ty = random.randint(-2, 2)
        if tx != 0 or ty != 0:
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            img = cv2.warpAffine(img, M, (w, h),
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=255)

        # 2) Width jitter ±5% (estira/encoge horizontalmente y vuelve a img_width)
        if random.random() < 0.5:
            scale = random.uniform(0.95, 1.05)
            new_w = max(int(w * scale), 1)
            img = cv2.resize(img, (new_w, h), interpolation=cv2.INTER_LINEAR)
            if new_w < w:
                pad = np.ones((h, w - new_w), dtype=np.uint8) * 255
                img = np.hstack([img, pad])
            elif new_w > w:
                img = img[:, :w]

        # 3) RandomErasing: 0-2 parches blancos de ~4-8 × 6-14 px
        n_erase = random.choices([0, 1, 2], weights=[0.4, 0.4, 0.2])[0]
        for _ in range(n_erase):
            eh = random.randint(4, 8)
            ew = random.randint(6, 14)
            y0 = random.randint(0, max(h - eh, 0))
            x0 = random.randint(0, max(w - ew, 0))
            img[y0:y0+eh, x0:x0+ew] = 255

        return img
 
    def precompute_chunked(self, output_dir, chunk_size=5000):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
 
        total      = len(self.image_paths)
        num_chunks = (total + chunk_size - 1) // chunk_size
 
        print(f"Total imágenes : {total}")
        print(f"Tamaño de chunk: {chunk_size}")
        print(f"Chunks a generar: {num_chunks}")
        print("Formato: PNG bytes (sin resize de ancho)\n")
 
        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size
            end   = min(start + chunk_size, total)
 
            chunk_images_png = []
            chunk_labels     = []
            chunk_texts      = []
 
            print(f"[Chunk {chunk_idx + 1}/{num_chunks}] imágenes {start}–{end - 1}")
 
            for i in range(start, end):
                img_path   = self.image_paths[i]
                label_text = self.labels[i]
 
                try:
                    binarized = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    png_bytes     = _png_encode(binarized)
                    label_encoded = torch.tensor(
                        self.encode_label(label_text), dtype=torch.long
                    )
 
                    chunk_images_png.append(png_bytes)
                    chunk_labels.append(label_encoded)
                    chunk_texts.append(label_text)
 
                except Exception as e:
                    print(f"  ⚠ Error en {img_path}: {e} — omitida")
                    continue
 
                if (i - start + 1) % 500 == 0:
                    print(f"  {i - start + 1}/{end - start} procesadas")
 
            chunk_path = output_dir / f"chunk_{chunk_idx:04d}.pt"
            torch.save({
                'images_png':     chunk_images_png,
                'encoded_labels': chunk_labels,
                'texts':          chunk_texts,
                'charset':        self.charset,
                'img_height':     self.img_height,
                'format':         'png_bytes_v2',
            }, chunk_path)
 
            tam_mb = os.path.getsize(chunk_path) / (1024 * 1024)
            raw_mb = sum(len(b) for b in chunk_images_png) / (1024 * 1024)
            print(f"  ✓ {chunk_path.name}  —  {tam_mb:.1f} MB en disco "
                  f"({raw_mb:.1f} MB solo PNG)")
 
            del chunk_images_png, chunk_labels, chunk_texts
            torch.cuda.empty_cache()
 
        print(f"\n✓ Precomputación completa. {num_chunks} chunks en: {output_dir}")
 
    @classmethod
    def from_precomputed_chunked(cls, chunks_dir, charset, img_width=None, augment=False):
        chunks_dir  = Path(chunks_dir)
        chunk_files = sorted(chunks_dir.glob("chunk_*.pt"))
 
        if not chunk_files:
            raise FileNotFoundError(f"No se encontraron chunks en: {chunks_dir}")
 
        print(f"Cargando {len(chunk_files)} chunks desde: {chunks_dir}")
 
        all_images_png = []
        all_labels     = []
        all_texts      = []
        charset_saved  = None
        img_height     = None
 
        for chunk_path in chunk_files:
            print(f"  {chunk_path.name}...", end=" ", flush=True)
            data = torch.load(chunk_path, weights_only=False)
 
            fmt = data.get('format', 'legacy')
            if fmt != 'png_bytes_v2':
                raise ValueError(
                    f"{chunk_path.name} usa el formato antiguo (tensores float32). "
                    "Regenera los chunks con precompute_chunked()."
                )
 
            if charset_saved is None:
                charset_saved = data['charset']
                img_height    = data['img_height']
 
            if data['charset'] != charset_saved:
                raise ValueError(f"Charset inconsistente en {chunk_path.name}")
 
            all_images_png.extend(data['images_png'])
            all_labels.extend(data['encoded_labels'])
            all_texts.extend(data['texts'])
            print(f"{len(data['images_png'])} imágenes")
            del data
 
        if charset_saved != charset:
            raise ValueError(
                "El charset de los chunks no coincide con el charset actual.\n"
                f"  Guardado: {charset_saved[:10]}...\n"
                f"  Actual:   {charset[:10]}..."
            )
 
        instance = cls(
            image_paths=[],
            labels=all_texts,
            charset=charset,
            img_height=img_height,
            img_width=img_width or 0,
            augment=augment,
        )
        instance._images_png    = all_images_png
        instance._labels_cache  = all_labels
        instance._tensors_cache = None
 
        total_mb = sum(len(b) for b in all_images_png) / (1024 * 1024)
        print(f"\n✓ {len(all_images_png)} imágenes en RAM  ({total_mb:.1f} MB PNG)")
        if img_width is None:
            print("  Modo: ancho variable — collate_fn hará el padding por batch.")
        else:
            print(f"  Modo: ancho fijo {img_width} px — resize en __getitem__.")
        if augment:
            print(f"  Augmentation dinámica: ON (shift + width jitter + erasing)")
        return instance
 
    def precompute(self, output_path):
        print(f"Precomputando {len(self.image_paths)} imágenes...")
        tensors        = []
        encoded_labels = []
 
        for i, (img_path, label_text) in enumerate(zip(self.image_paths, self.labels)):
            *_, img = preprocess_image(img_path, self.img_height, self.img_width)
 
            img_tensor    = torch.tensor(img, dtype=torch.float32) / 255.0
            img_tensor    = img_tensor.unsqueeze(0)
            img_tensor    = self.normalize(img_tensor)
            label_encoded = torch.tensor(self.encode_label(label_text), dtype=torch.long)
 
            tensors.append(img_tensor)
            encoded_labels.append(label_encoded)
 
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(self.image_paths)}")
 
        torch.save({
            'tensors':        torch.stack(tensors),
            'encoded_labels': encoded_labels,
            'texts':          self.labels,
            'charset':        self.charset,
            'img_height':     self.img_height,
            'img_width':      self.img_width,
        }, output_path)
 
        print(f"✓ Guardado en: {output_path}")
        tam_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  Tamaño: {tam_mb:.1f} MB")
 
    @classmethod
    def from_precomputed(cls, precomputed_path, charset, augment=False):
        print(f"Cargando dataset precomputado: {precomputed_path}")
        data = torch.load(precomputed_path, weights_only=False)
 
        if data['charset'] != charset:
            raise ValueError(
                "El charset del archivo no coincide con el charset actual.\n"
                f"  Guardado:  {data['charset'][:10]}...\n"
                f"  Actual:    {charset[:10]}..."
            )
 
        instance = cls(
            image_paths=[],
            labels=data['texts'],
            charset=charset,
            img_height=data['img_height'],
            img_width=data['img_width'],
            augment=augment,
        )
        instance._tensors_cache = data['tensors']
        instance._labels_cache  = data['encoded_labels']
 
        print(f"  ✓ {len(instance._tensors_cache)} muestras cargadas en RAM")
        return instance
 
    @staticmethod
    def merge_chunks(chunks_dir, output_path):
        chunks_dir  = Path(chunks_dir)
        chunk_files = sorted(chunks_dir.glob("chunk_*.pt"))
 
        if not chunk_files:
            raise FileNotFoundError(f"No se encontraron chunks en: {chunks_dir}")
 
        print(f"Fusionando {len(chunk_files)} chunks...")
 
        all_tensors = []
        all_labels  = []
        all_texts   = []
        charset = img_height = img_width = None
 
        for chunk_path in chunk_files:
            print(f"  Cargando {chunk_path.name}...", end=" ")
            data = torch.load(chunk_path, weights_only=False)
 
            if charset is None:
                charset    = data['charset']
                img_height = data['img_height']
                img_width  = data.get('img_width', None)
            else:
                if data['charset'] != charset:
                    raise ValueError(f"Charset inconsistente en {chunk_path.name}")
 
            all_tensors.append(data['tensors'])
            all_labels.extend(data['encoded_labels'])
            all_texts.extend(data['texts'])
            print(f"{len(data['tensors'])} muestras")
            del data
 
        print("Concatenando tensores...", end=" ", flush=True)
        merged_tensors = torch.cat(all_tensors, dim=0)
        del all_tensors
        print(f"shape final: {merged_tensors.shape}")
 
        torch.save({
            'tensors':        merged_tensors,
            'encoded_labels': all_labels,
            'texts':          all_texts,
            'charset':        charset,
            'img_height':     img_height,
            'img_width':      img_width,
        }, output_path)
 
        tam_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✓ Archivo final: {output_path}  ({tam_mb:.1f} MB)")
 
    def __len__(self):
        if self._images_png    is not None: return len(self._images_png)
        if self._tensors_cache is not None: return len(self._tensors_cache)
        return len(self.image_paths)
 
    def __getitem__(self, idx):
        # Modo PNG bytes (recomendado — datos precomputados desde Kaggle)
        if self._images_png is not None:
            img = _png_decode(self._images_png[idx])
 
            use_fixed_width = self.img_width and self.img_width > 0
            if use_fixed_width:
                img = resize_img(img, self.img_height, self.img_width)

            # Augmentation dinámica leve (solo si se activó en el dataset)
            # El sintético ya tiene blur/elastic/ruido pre-aplicado en el PNG;
            # esto añade shift, width jitter y erasing por época para evitar
            # memorización de patrones fijos.
            if self.augment:
                img = self._apply_augmentation(img)
 
            img_tensor = torch.tensor(img, dtype=torch.float32) / 255.0
            img_tensor = img_tensor.unsqueeze(0)
            img_tensor = self.normalize(img_tensor)
            return img_tensor, self._labels_cache[idx], self.labels[idx]
 
        # Modo tensor cache (formato antiguo)
        if self._tensors_cache is not None:
            return (
                self._tensors_cache[idx],
                self._labels_cache[idx],
                self.labels[idx],
            )
 
        # Modo lectura desde disco
        img_path   = self.image_paths[idx]
        label_text = self.labels[idx]
 
        *_, img = preprocess_image(str(img_path), self.img_height, self.img_width)
 
        img_tensor    = torch.tensor(img, dtype=torch.float32) / 255.0
        img_tensor    = img_tensor.unsqueeze(0)
        img_tensor    = self.normalize(img_tensor)
        label_encoded = torch.tensor(self.encode_label(label_text), dtype=torch.long)
 
        return img_tensor, label_encoded, label_text
 
    def encode_label(self, text):
        return [self.char2idx[c] for c in text if c in self.char2idx]
 
CHARSET = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
               "0123456789 $#&%*.,;:¡!¿?«»-_'\"()[]áéíóúüñÁÉÍÓÚÜÑ")

IMG_HEIGHT = 64
IMG_WIDTH  = 256
BATCH_SIZE = 128
NUM_EPOCHS = 100

DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
PIN_MEMORY = (DEVICE == 'cuda')
torch.backends.cudnn.benchmark = True

# ─── RUTAS DE LOS DATASETS (ajusta a tus nombres en Kaggle Datasets) ─────────

PATH_IAM_TRAIN        = '/kaggle/input/datasets/lauramartir/iam-dataset-happy-ending/iam_chunks_train'
PATH_IAM_VAL          = '/kaggle/input/datasets/lauramartir/iam-dataset-happy-ending/iam_chunks_val'

PATH_SYNTH_TRAIN      = '/kaggle/input/datasets/lauramartir/synthetic-dataset-5/synthetic_dataset_chunks_train'
PATH_SYNTH_VAL_SEEN   = '/kaggle/input/datasets/lauramartir/synthetic-dataset-5/synthetic_dataset_chunks_val_seen'
PATH_SYNTH_VAL_UNSEEN = '/kaggle/input/datasets/lauramartir/synthetic-dataset-5/synthetic_dataset_chunks_val_unseen'

PATH_MIX_TRAIN        = '/kaggle/input/datasets/lauramartir/dataset-final-new-aug/final_dataset_chunks_train'  # real + synth pre-mezclados
PATH_REAL_VAL         = '/kaggle/input/datasets/lauramartir/dataset-final-new-aug/final_dataset_chunks_val'

# Checkpoint de la fase anterior (se usa al cargar pesos en fases 2 y 3)
#PATH_CKPT_FASE1       = '/kaggle/input/datasets/lauramartir/model-etapa1/best_model.pt'
PATH_CKPT_FASE2       = '/kaggle/input/datasets/lauramartir/model-etapa3/best_model.pt'

model = CRNN(
    num_classes=len(CHARSET),
    img_height=IMG_HEIGHT,
    hidden_channels=32,
    lstm_hidden=128,
    dropout=0.3
)
print(f"Parámetros CRNN-Lite v2: {sum(p.numel() for p in model.parameters()):,}")

import torch._dynamo
torch._dynamo.config.suppress_errors = True
model = torch.compile(model)

'''
iam_train = HTRDataset.from_precomputed_chunked(
    PATH_IAM_TRAIN, charset=CHARSET, img_width=IMG_WIDTH, augment=True
)
iam_val = HTRDataset.from_precomputed_chunked(
    PATH_IAM_VAL, charset=CHARSET, img_width=IMG_WIDTH, augment=False
)

assert iam_train._images_png is not None, "Train IAM no cargó PNG bytes precomputados"

max_label_len = max(len(t) for t in iam_train.labels)
print(f"\nMax label length: {max_label_len} | T disponible: 128")
if max_label_len > 60:
    print(f"  ⚠ max_label_len={max_label_len}: considera pool(2,1) en layer1 para T=256")

train_loader = DataLoader(iam_train, batch_size=BATCH_SIZE, shuffle=True,
                          pin_memory=PIN_MEMORY, collate_fn=collate_fn,
                          num_workers=4, persistent_workers=True, prefetch_factor=4,
                          drop_last=True)
val_loader = DataLoader(iam_val, batch_size=BATCH_SIZE, shuffle=False,
                        pin_memory=PIN_MEMORY, collate_fn=collate_fn,
                        num_workers=4, persistent_workers=False, prefetch_factor=4,
                        drop_last=False)

trainer = HTRTrainer(model=model, charset=CHARSET, lr=1e-3, weight_decay=1e-4,
                     hidden_channels=32, lstm_hidden=128, img_height=IMG_HEIGHT,
                     device=DEVICE, checkpoint_dir='checkpoints_fase1_iam')

trainer.scheduler = torch.optim.lr_scheduler.OneCycleLR(
    trainer.optimizer,
    max_lr=1e-3, epochs=NUM_EPOCHS, steps_per_epoch=len(train_loader),
    pct_start=0.1, anneal_strategy='cos', div_factor=25, final_div_factor=1e4
)

# Fase 1: SIN extra_val_loaders, SIN score custom — IAM val decide solo
trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=NUM_EPOCHS,
    early_stopping_patience=7,
)
'''

'''
TARGET_SYNTH = 0.85
TARGET_IAM   = 0.15

synth_train = HTRDataset.from_precomputed_chunked(
    PATH_SYNTH_TRAIN, charset=CHARSET, img_width=IMG_WIDTH, augment=True
)
iam_train = HTRDataset.from_precomputed_chunked(
    PATH_IAM_TRAIN, charset=CHARSET, img_width=IMG_WIDTH, augment=True
)

n_synth = len(synth_train)
n_iam   = len(iam_train)
weights = [TARGET_SYNTH / n_synth] * n_synth + [TARGET_IAM / n_iam] * n_iam
num_samples_per_epoch = int(n_synth / TARGET_SYNTH)
sampler = WeightedRandomSampler(weights, num_samples=num_samples_per_epoch,
                                replacement=True)

train_dataset = ConcatDataset([synth_train, iam_train])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                          pin_memory=PIN_MEMORY, collate_fn=collate_fn,
                          num_workers=4, persistent_workers=True, prefetch_factor=4,
                          drop_last=True)

synth_val_seen = HTRDataset.from_precomputed_chunked(
    PATH_SYNTH_VAL_SEEN, charset=CHARSET, img_width=IMG_WIDTH, augment=False
)
synth_val_unseen = HTRDataset.from_precomputed_chunked(
    PATH_SYNTH_VAL_UNSEEN, charset=CHARSET, img_width=IMG_WIDTH, augment=False
)

assert synth_train._images_png is not None, "Train sintético no cargó PNG"

max_label_len = max(len(t) for t in synth_train.labels)
print(f"\nMax label length: {max_label_len} | T disponible: 128")
if max_label_len > 60:
    print(f"  ⚠ max_label_len={max_label_len}: considera pool(2,1) en layer1")

val_seen_loader = DataLoader(synth_val_seen, batch_size=BATCH_SIZE, shuffle=False,
                             pin_memory=PIN_MEMORY, collate_fn=collate_fn,
                             num_workers=4, persistent_workers=False, prefetch_factor=4,
                             drop_last=False)
val_unseen_loader = DataLoader(synth_val_unseen, batch_size=BATCH_SIZE, shuffle=False,
                               pin_memory=PIN_MEMORY, collate_fn=collate_fn,
                               num_workers=4, persistent_workers=False, prefetch_factor=4,
                               drop_last=False)

trainer = HTRTrainer(model=model, charset=CHARSET, lr=2e-4, weight_decay=1e-4,
                     hidden_channels=32, lstm_hidden=128, img_height=IMG_HEIGHT,
                     device=DEVICE, checkpoint_dir='checkpoints_fase2_synth')

# Cargar pesos de fase 1 (NO optimizer ni scheduler)
ckpt = torch.load(PATH_CKPT_FASE1, map_location=DEVICE, weights_only=False)
_unwrap(trainer.model).load_state_dict(ckpt['model_state_dict'])
print(f"\n✓ Pesos cargados de fase 1 | CER previo en IAM: {ckpt['best_cer']:.4f}")

trainer.scheduler = torch.optim.lr_scheduler.OneCycleLR(
    trainer.optimizer,
    max_lr=2e-4, epochs=NUM_EPOCHS, steps_per_epoch=len(train_loader),
    pct_start=0.1, anneal_strategy='cos', div_factor=20, final_div_factor=1e3
)

# Score ponderado: prioriza generalización a fuentes no vistas
def score_fn_fase2(cer_seen, extras):
    return 0.4 * cer_seen + 0.6 * extras['val_unseen']['cer']

# val principal = val_seen (early stopping); val_unseen y opcionalmente IAM
# como monitoring. Quita 'iam_val' si no te interesa monitorear olvido del inglés.
iam_val_dataset = HTRDataset.from_precomputed_chunked(
    PATH_IAM_VAL, charset=CHARSET, img_width=IMG_WIDTH, augment=False
)
iam_val_loader = DataLoader(iam_val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            pin_memory=PIN_MEMORY, collate_fn=collate_fn,
                            num_workers=2, persistent_workers=False, prefetch_factor=2,
                            drop_last=False)

trainer.train(
    train_loader=train_loader,
    val_loader=val_seen_loader,
    extra_val_loaders={
        'val_unseen': val_unseen_loader,
        'iam_val':    iam_val_loader,
    },
    checkpoint_score_fn=score_fn_fase2,
    num_epochs=NUM_EPOCHS,
    early_stopping_patience=10,
)

'''

TARGET_MIX = 0.85   # del batch viene del conjunto mezclado real+sint
TARGET_IAM = 0.15
assert abs(TARGET_MIX + TARGET_IAM - 1.0) < 1e-6

mix_train = HTRDataset.from_precomputed_chunked(
    PATH_MIX_TRAIN, charset=CHARSET, img_width=IMG_WIDTH, augment=True
)
iam_train_mix = HTRDataset.from_precomputed_chunked(
    PATH_IAM_TRAIN, charset=CHARSET, img_width=IMG_WIDTH, augment=True
)

real_val = HTRDataset.from_precomputed_chunked(
    PATH_REAL_VAL, charset=CHARSET, img_width=IMG_WIDTH, augment=False
)

# Monitoring opcional: val seen/unseen sintético + IAM val
synth_val_seen   = HTRDataset.from_precomputed_chunked(
    PATH_SYNTH_VAL_SEEN, charset=CHARSET, img_width=IMG_WIDTH, augment=False
)
synth_val_unseen = HTRDataset.from_precomputed_chunked(
    PATH_SYNTH_VAL_UNSEEN, charset=CHARSET, img_width=IMG_WIDTH, augment=False
)
iam_val = HTRDataset.from_precomputed_chunked(
    PATH_IAM_VAL, charset=CHARSET, img_width=IMG_WIDTH, augment=False
)

# WeightedRandomSampler entre dos grupos: mix vs IAM
n_mix = len(mix_train)
n_iam = len(iam_train_mix)

print(f"\nMezcla fase 3:")
print(f"  Mix (real+sint): {n_mix:>6}  ({TARGET_MIX*100:.0f}% del batch)")
print(f"  IAM:             {n_iam:>6}  ({TARGET_IAM*100:.0f}% del batch)")

weights = (
    [TARGET_MIX / n_mix] * n_mix +
    [TARGET_IAM / n_iam] * n_iam
)

# Tamaño de época calibrado al mix: cada época ve ~n_mix / TARGET_MIX muestras
# (≈ mismo número de imágenes de mix que en un epoch lineal del conjunto).
num_samples_per_epoch = int(n_mix / TARGET_MIX)
sampler = WeightedRandomSampler(weights=weights,
                                num_samples=num_samples_per_epoch,
                                replacement=True)

train_dataset = ConcatDataset([mix_train, iam_train_mix])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                          pin_memory=PIN_MEMORY, collate_fn=collate_fn,
                          num_workers=4, persistent_workers=True, prefetch_factor=4,
                          drop_last=True)
real_val_loader = DataLoader(real_val, batch_size=BATCH_SIZE, shuffle=False,
                             pin_memory=PIN_MEMORY, collate_fn=collate_fn,
                             num_workers=2, persistent_workers=False, prefetch_factor=2,
                             drop_last=False)
synth_val_seen_loader   = DataLoader(synth_val_seen, batch_size=BATCH_SIZE, shuffle=False,
                                     pin_memory=PIN_MEMORY, collate_fn=collate_fn,
                                     num_workers=2, persistent_workers=False, prefetch_factor=2,
                                     drop_last=False)
synth_val_unseen_loader = DataLoader(synth_val_unseen, batch_size=BATCH_SIZE, shuffle=False,
                                     pin_memory=PIN_MEMORY, collate_fn=collate_fn,
                                     num_workers=2, persistent_workers=False, prefetch_factor=2,
                                     drop_last=False)
iam_val_loader          = DataLoader(iam_val, batch_size=BATCH_SIZE, shuffle=False,
                                     pin_memory=PIN_MEMORY, collate_fn=collate_fn,
                                     num_workers=2, persistent_workers=False, prefetch_factor=2,
                                     drop_last=False)

trainer = HTRTrainer(model=model, charset=CHARSET, lr=5e-5, weight_decay=1e-4,
                     hidden_channels=32, lstm_hidden=128, img_height=IMG_HEIGHT,
                     device=DEVICE, checkpoint_dir='checkpoints_fase3_real')

# Cargar pesos de fase 2
ckpt = torch.load(PATH_CKPT_FASE2, map_location=DEVICE, weights_only=False)
_unwrap(trainer.model).load_state_dict(ckpt['model_state_dict'])
print(f"\n✓ Pesos cargados de fase 2 | CER previo en val seen: {ckpt['best_cer']:.4f}")

trainer.scheduler = torch.optim.lr_scheduler.OneCycleLR(
    trainer.optimizer,
    max_lr=5e-5, epochs=NUM_EPOCHS, steps_per_epoch=len(train_loader),
    pct_start=0.05, anneal_strategy='cos', div_factor=20, final_div_factor=50
)

# val principal = REAL (decide early stopping y best). Resto es monitoring.
trainer.train(
    train_loader=train_loader,
    val_loader=real_val_loader,
    extra_val_loaders={
        'val_seen_synth':   synth_val_seen_loader,
        'val_unseen_synth': synth_val_unseen_loader,
        'iam_val':          iam_val_loader,
    },
    checkpoint_score_fn=None,   # mejor checkpoint = mínimo CER real
    num_epochs=NUM_EPOCHS,
    early_stopping_patience=15,
)
