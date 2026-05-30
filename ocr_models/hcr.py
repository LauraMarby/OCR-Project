import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from preprocessing_utils import preprocess_image, preprocess_array, preprocess_image_binarized, resize_img, preprocess_array_binarized, preprocess_image_binarized
import matplotlib.pyplot as plt

import numpy as np
import cv2
import os
import json
from itertools import groupby
import time

from pathlib import Path

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
                 hidden_channels=32, lstm_hidden=128):
        super().__init__()

        self.cnn        = CNNFeatureExtractor(1, hidden_channels)
        cnn_out         = self.cnn.out_channels  # 256

        self.bilstm     = BiLSTMBlock(cnn_out, lstm_hidden, num_layers=1, dropout=0.3)
        self.dropout    = nn.Dropout(p=0.3)
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
    """
    Decodificador CTC para convertir las predicciones en texto.

    Soporta dos estrategias:
    - greedy:     rápido, toma el caracter más probable en cada paso
    - beam_search: más preciso, explora múltiples hipótesis en paralelo
    """

    def __init__(self, charset, blank_idx=0):
        """
        charset: lista de caracteres del vocabulario (sin blank)
        blank_idx: índice reservado para el token blank (por convención: 0)
        """
        self.charset = charset
        self.blank_idx = blank_idx
        # idx2char: mapeamos índice → caracter
        # El índice 0 es blank, los demás son caracteres del charset
        self.idx2char = {i + 1: c for i, c in enumerate(charset)}
        self.idx2char[0] = '<blank>'

    def greedy_decode(self, log_probs):
        """
        Decodificación greedy: argmax por timestep + colapso CTC.

        Args:
            log_probs: tensor (T, Batch, num_classes+1)
        Returns:
            lista de strings decodificadas, una por muestra del batch
        """
        # Argmax por timestep: mejor clase en cada momento
        # (T, B, C) → argmax sobre C → (T, B)
        best_path = torch.argmax(log_probs, dim=2)  # → (T, B)
        best_path = best_path.permute(1, 0)          # → (B, T)

        decoded_batch = []
        for sequence in best_path:
            # sequence: (T,) — índices de clase para cada timestep

            # Paso 1: Eliminar blanks y caracteres repetidos consecutivos
            # groupby agrupa consecutivos iguales
            chars = []
            for idx, group in groupby(sequence.tolist()):
                if idx != self.blank_idx:
                    chars.append(self.idx2char.get(idx, '?'))

            decoded_batch.append(''.join(chars))

        return decoded_batch

    def beam_search_decode(self, log_probs, beam_width=10):
        """
        Decodificación beam search: mantiene las top-K hipótesis.

        Más lento que greedy pero produce mejores resultados, especialmente
        en secuencias con ambigüedad.

        Args:
            log_probs: tensor (T, 1, num_classes+1) — procesa 1 muestra
            beam_width: número de hipótesis a mantener
        Returns:
            string decodificada (mejor hipótesis)
        """
        # Extraer probabilidades (T, C)
        probs = torch.exp(log_probs[:, 0, :]).cpu().numpy()
        T, C = probs.shape

        # Beam: lista de (score_log, secuencia_sin_blank)
        # Inicializar con secuencia vacía
        beams = [(0.0, [])]

        for t in range(T):
            new_beams = {}

            for score, seq in beams:
                for c in range(C):
                    p = probs[t, c]
                    if p == 0:
                        continue
                    new_score = score + np.log(p + 1e-10)

                    if c == self.blank_idx:
                        # Blank: la secuencia no cambia
                        key = tuple(seq)
                        new_seq = seq
                    elif len(seq) > 0 and c == seq[-1]:
                        # Repetido: solo se añade si hay blank intermedio
                        # En beam search simplificado: ignoramos duplicados
                        key = tuple(seq)
                        new_seq = seq
                    else:
                        new_seq = seq + [c]
                        key = tuple(new_seq)

                    if key not in new_beams or new_beams[key][0] < new_score:
                        new_beams[key] = (new_score, new_seq)

            # Ordenar y mantener top beam_width hipótesis
            sorted_beams = sorted(new_beams.values(),
                                  key=lambda x: x[0], reverse=True)
            # beams = [(s, list(k)) for s, k in
            #          [(b[0], b[1]) for b in sorted_beams[:beam_width]]]
            beams = [(b[0], b[1]) for b in sorted_beams[:beam_width]]

        if not beams:
            return ''

        best_seq = beams[0][1]
        return ''.join(self.idx2char.get(i, '?') for i in best_seq)
    
class HTRTrainer:
    """
    Clase de entrenamiento con todas las buenas prácticas:
    - Learning rate scheduling (reduce on plateau)
    - Gradient clipping (evita explosión de gradientes en LSTM)
    - Checkpointing (guarda el mejor modelo)
    - Métricas: CER (Character Error Rate) y WER (Word Error Rate)
    """
    
    def __init__(self,model, charset, device='cuda', lr=1e-3, checkpoint_dir='checkpoints', 
                 hidden_channels=32, lstm_hidden=256, img_height=32):
        
        self.model = model.to(device)
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Guardar config para poder reconstruir el modelo al reanudar
        self.config = {
            'hidden_channels': hidden_channels,
            'lstm_hidden': lstm_hidden,
            'img_height': img_height,
            'num_classes': len(charset)
        }

        # --- CTC Loss ---
        # blank=0: el índice 0 corresponde al token blank
        # zero_infinity=True: ignora pérdidas infinitas (secuencias inválidas)
        self.criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)

        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5) # L2 regularización
    
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', # reducir cuando la métrica deja de bajar
            factor=0.5, # nuevo_lr = lr * 0.5
            patience=5  # esperar 5 épocas sin mejora
        )
     
        self.use_amp = (device == 'cuda' and torch.cuda.is_available())
        self.scaler  = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.decoder = CTCDecoder(charset=charset, blank_idx=0)
        self.best_cer = float('inf')
        self.start_epoch = 1 # ← desde qué época empezar

        self.metrics_path = os.path.join(checkpoint_dir, 'metrics_history.pt')

        # Intenta cargar historial previo, si no existe empieza vacío
        self.history = self._load_history()

    def _load_history(self):
        """Carga el historial de métricas si existe, si no devuelve uno vacío."""
        if os.path.exists(self.metrics_path):
            data = torch.load(self.metrics_path, weights_only=False)
            print(f"  Historial cargado: {len(data['loss_train'])} épocas previas")
            return data
        return {
            'epoch':      [],
            'loss_train': [],
            'loss_val':   [],
            'cer':        [],
            'wer':        [],
        }

    def _save_history(self):
        """Guarda el historial actualizado en disco."""
        torch.save(self.history, self.metrics_path)

    def _append_metrics(self, epoch, loss_train, loss_val, cer, wer):
        """Añade una fila de métricas al historial y lo persiste."""
        self.history['epoch'].append(epoch)
        self.history['loss_train'].append(loss_train)
        self.history['loss_val'].append(loss_val)
        self.history['cer'].append(cer)
        self.history['wer'].append(wer)
        self._save_history()

    def train_epoch(self, dataloader):
        """Una época de entrenamiento completa."""
        self.model.train()
        total_loss  = 0.0
        num_batches = 0
        t0 = time.time()
 
        for batch in dataloader:
            # collate_fn devuelve 5 elementos; input_lengths = anchos reales
            images, targets, target_lengths, _, input_lengths = batch
 
            images         = images.to(self.device, non_blocking=True)
            targets        = targets.to(self.device, non_blocking=True)
            target_lengths = target_lengths.to(self.device, non_blocking=True)
            input_lengths  = input_lengths.to(self.device)
 
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                log_probs = self.model(images)          # (T, B, C)
                # input_lengths ya contiene los anchos reales de cada imagen;
                # para ancho fijo todos son iguales a T (sin overhead).
                loss = self.criterion(
                    log_probs, targets, input_lengths, target_lengths
                )
 
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
 
            total_loss  += loss.item()
            num_batches += 1
 
            if num_batches % 200 == 0:
                elapsed = time.time() - t0
                print(f"    batch {num_batches}/{len(dataloader)} | "
                      f"loss={total_loss/num_batches:.4f} | "
                      f"{num_batches/elapsed:.1f} batches/s")
 
        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def evaluate(self, dataloader):
        """Evaluación en el conjunto de validación."""
        self.model.eval()
        total_loss = 0.0
        all_preds  = []
        all_targets = []
 
        for batch in dataloader:
            images, targets, target_lengths, texts, input_lengths = batch
 
            images         = images.to(self.device, non_blocking=True)
            targets_dev    = targets.to(self.device, non_blocking=True)
            target_lengths = target_lengths.to(self.device, non_blocking=True)
            input_lengths  = input_lengths.to(self.device)
 
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                log_probs = self.model(images)
                loss      = self.criterion(
                    log_probs, targets_dev, input_lengths, target_lengths
                )
 
            total_loss += loss.item()
            preds = self.decoder.greedy_decode(log_probs)
            all_preds.extend(preds)
            all_targets.extend(texts)
 
        avg_loss = total_loss / max(len(dataloader), 1)
        cer = self.compute_cer(all_preds, all_targets)
        wer = self.compute_wer(all_preds, all_targets)
        return avg_loss, cer, wer

    def compute_cer(self, predictions, targets):
        """
        CER = Character Error Rate
        CER = edit_distance(pred, target) / len(target)

        Mide el % de caracteres que hay que insertar/borrar/sustituir
        para transformar la predicción en el texto correcto.
        """
        total_edit = 0
        total_chars = 0

        for pred, target in zip(predictions, targets):
            edit_dist = self._edit_distance(pred, target)
            total_edit += edit_dist
            total_chars += len(target)

        return total_edit / max(total_chars, 1)

    def compute_wer(self, predictions, targets):
        """
        WER = Word Error Rate
        Similar al CER pero a nivel de palabras.
        """
        total_edit = 0
        total_words = 0

        for pred, target in zip(predictions, targets):
            pred_words = pred.split()
            target_words = target.split()
            edit_dist = self._edit_distance(pred_words, target_words)
            total_edit += edit_dist
            total_words += len(target_words)

        return total_edit / max(total_words, 1)

    def _edit_distance(self, s1, s2):
        """Distancia de Levenshtein entre dos secuencias."""
        m, n = len(s1), len(s2)
        dp = np.zeros((m + 1, n + 1), dtype=int)

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j],    # borrar
                                       dp[i][j-1],    # insertar
                                       dp[i-1][j-1])  # sustituir
        return dp[m][n]

    def save_checkpoint(self, epoch, cer, filename=None):
        if filename is None:
            filename = f'checkpoint_epoch{epoch:03d}_cer{cer:.4f}.pt'
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            'epoch':               epoch,
            'model_state_dict':    self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict':   self.scaler.state_dict(),
            'best_cer':            self.best_cer,
            'config':              self.config,
        }, path)
        print(f"  ✓ Checkpoint guardado: {path}")
 
    def load_checkpoint(self, checkpoint_path):
        print(f"Cargando checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.best_cer    = checkpoint['best_cer']
        self.start_epoch = checkpoint['epoch'] + 1
        print(f"  Reanudando desde época {self.start_epoch}")
        print(f"  Mejor CER: {self.best_cer:.4f}")
        print(f"  LR actual: {self.optimizer.param_groups[0]['lr']:.6f}")

    def load_checkpoint(self, checkpoint_path):
        """
        Carga un checkpoint y restaura el estado completo del entrenamiento.
        Llama a esto ANTES de train() para reanudar.
        """
        print(f"Cargando checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device,
                                weights_only=False)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.best_cer = checkpoint['best_cer']
        self.start_epoch = checkpoint['epoch'] + 1  # ← continuar desde la siguiente

        # El historial se cargó automáticamente en __init__,
        # solo confirmamos que está sincronizado
        epocas_en_historial = len(self.history['epoch'])

        print(f"  Reanudando desde época {self.start_epoch}")
        print(f"  Mejor CER hasta ahora: {self.best_cer:.4f}")
        print(f"  LR actual: {self.optimizer.param_groups[0]['lr']:.6f}")
        print(f"  Épocas en historial:   {epocas_en_historial}")

    def train(self, train_loader, val_loader, num_epochs=100):
        print("=" * 60)
        print("  ENTRENAMIENTO CRNN para HTR  (mixed precision)" if self.use_amp
              else "  ENTRENAMIENTO CRNN para HTR  (fp32)")
        print("=" * 60)
 
        for epoch in range(self.start_epoch, num_epochs + 1):
            t0 = time.time()
            train_loss           = self.train_epoch(train_loader)
            val_loss, cer, wer   = self.evaluate(val_loader)
            self.scheduler.step(val_loss)
 
            elapsed    = time.time() - t0
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Época {epoch:3d}/{num_epochs} | "
                  f"Loss train: {train_loss:.4f} | Loss val: {val_loss:.4f} | "
                  f"CER: {cer:.4f} | WER: {wer:.4f} | "
                  f"LR: {current_lr:.6f} | {elapsed:.0f}s")
 
            self._append_metrics(epoch, train_loss, val_loss, cer, wer)
            self.save_checkpoint(epoch, cer)
 
            if cer < self.best_cer:
                self.best_cer = cer
                self.save_checkpoint(epoch, cer, 'best_model.pt')
                print(f"  ★ Nuevo mejor CER: {cer:.4f}")

    def plot_metrics(self, save_path=None):
        """
        Grafica las curvas de loss y CER/WER usando el historial acumulado
        de todas las sesiones de entrenamiento.
        """
        if not self.history['epoch']:
            print("No hay métricas registradas todavía.")
            return

        epochs     = self.history['epoch']
        loss_train = self.history['loss_train']
        loss_val   = self.history['loss_val']
        cer        = self.history['cer']
        wer        = self.history['wer']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Curvas de entrenamiento', fontsize=14, fontweight='bold')

        ax1.plot(epochs, loss_train, label='Loss train', color='steelblue',   linewidth=2)
        ax1.plot(epochs, loss_val,   label='Loss val',   color='tomato',      linewidth=2)
        ax1.set_xlabel('Época')
        ax1.set_ylabel('CTC Loss')
        ax1.set_title('Loss train vs validación')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Marcar mínimo de val loss
        min_val_idx = int(np.argmin(loss_val))
        ax1.axvline(x=epochs[min_val_idx], color='tomato', linestyle='--',
                    alpha=0.5, label=f'Min val época {epochs[min_val_idx]}')
        ax1.scatter([epochs[min_val_idx]], [loss_val[min_val_idx]],
                    color='tomato', zorder=5, s=80)

        ax2.plot(epochs, cer, label='CER', color='darkorange', linewidth=2)
        ax2.plot(epochs, wer, label='WER', color='purple',     linewidth=2)
        ax2.set_xlabel('Época')
        ax2.set_ylabel('Tasa de error')
        ax2.set_title('CER y WER')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.05)

        # Marcar mínimo de CER
        min_cer_idx = int(np.argmin(cer))
        ax2.axvline(x=epochs[min_cer_idx], color='darkorange', linestyle='--',
                    alpha=0.5)
        ax2.scatter([epochs[min_cer_idx]], [cer[min_cer_idx]],
                    color='darkorange', zorder=5, s=80,
                    label=f'Min CER={cer[min_cer_idx]:.4f} (época {epochs[min_cer_idx]})')
        ax2.legend()

        # Separar sesiones con línea vertical si hay saltos en las épocas
        for i in range(1, len(epochs)):
            if epochs[i] != epochs[i-1] + 1:
                ax1.axvline(x=epochs[i] - 0.5, color='gray',
                            linestyle=':', alpha=0.6, linewidth=1.5)
                ax2.axvline(x=epochs[i] - 0.5, color='gray',
                            linestyle=':', alpha=0.6, linewidth=1.5)
                ax1.text(epochs[i] - 0.5, ax1.get_ylim()[1] * 0.95,
                         'nueva\nsesión', fontsize=7, color='gray', ha='center')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Gráfica guardada en: {save_path}")

        plt.show()          

def collate_fn(batch):
    """
    Función de collate personalizada para DataLoader.
 
    Admite imágenes de ANCHO VARIABLE dentro del mismo batch:
    hace padding blanco a la derecha hasta el máximo ancho del batch.
 
    Devuelve 5 elementos (antes eran 4):
        images        (B, 1, H, max_W)  — float32 normalizado
        targets       tensor 1D concatenado con todas las etiquetas
        target_lengths longitudes de cada etiqueta
        texts          tupla de strings originales
        input_lengths  anchos reales ANTES del padding — pasar directamente
                       a CTCLoss como input_lengths (T = W para esta CNN).
 
    Para ancho FIJO (todas las imágenes ya tienen el mismo W):
        input_lengths son todas iguales → comportamiento idéntico al anterior.
    """
    images, labels, texts = zip(*batch)
 
    # Anchos reales de cada imagen antes de cualquier padding
    widths = [img.shape[2] for img in images]   # img.shape = (1, H, W)
    max_w  = max(widths)
 
    if len(set(widths)) > 1:
        # Hay imágenes de distinto ancho → pad blanco a la derecha
        padded = []
        for img in images:
            diff = max_w - img.shape[2]
            if diff > 0:
                # Con Normalize(mean=0.5, std=0.5):
                #   píxel blanco (1.0) → (1.0 − 0.5) / 0.5 = 1.0 en espacio norm.
                white_pad = torch.ones(1, img.shape[1], diff, dtype=img.dtype)
                img = torch.cat([img, white_pad], dim=2)
            padded.append(img)
        images = torch.stack(padded, 0)
    else:
        images = torch.stack(list(images), 0)
 
    input_lengths  = torch.tensor(widths, dtype=torch.long)
    target_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    targets        = torch.cat(labels, dim=0)
 
    return images, targets, target_lengths, texts, input_lengths

class HTRInference:
    """Clase de inferencia para CRNN-Lite v2."""

    def __init__(self, model_path, charset, device='cpu',
                 img_height=64, img_width=256, hidden_channels=32, lstm_hidden=128):
        self.device     = device
        self.charset    = charset
        self.img_height = img_height
        # img_width NO se guarda en el config del checkpoint (HTRTrainer no lo incluye),
        # así que siempre se usa el valor del parámetro (default: 256 = IMG_WIDTH del script).
        self.img_width  = img_width

        # Cargar checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        cfg = checkpoint.get('config', {})

        self.model = CRNN(
            num_classes=cfg.get('num_classes', len(charset)),
            img_height=cfg.get('img_height', img_height),          # guardado: 64
            hidden_channels=cfg.get('hidden_channels', hidden_channels),  # guardado: 32
            lstm_hidden=cfg.get('lstm_hidden', lstm_hidden),        # guardado: 128
        )

        # Eliminar prefijo "_orig_mod." generado por torch.compile()
        state_dict = checkpoint['model_state_dict']
        if any(k.startswith('_orig_mod.') for k in state_dict):
            state_dict = {k.replace('_orig_mod.', '', 1): v for k, v in state_dict.items()}

        self.model.load_state_dict(state_dict)
        self.model.to(device).eval()

        self.decoder   = CTCDecoder(charset, blank_idx=0)
        self.normalize = transforms.Normalize(mean=[0.5], std=[0.5])

    def preprocess(self, img_path_or_array):
        """
        Preprocesa una imagen para inferencia.

        Pipeline idéntico al de HTRDataset.__getitem__ (modo disco):
          1. Lectura / conversión a escala de grises
          2. _binarize()  →  resize_img()   (vía preprocess_image / preprocess_array)
          3. float32 / 255  →  unsqueeze(canal)  →  Normalize(0.5, 0.5)
          4. unsqueeze(batch)  para obtener (1, 1, H, W)
        """

        img = cv2.imread(str(img_path_or_array), cv2.IMREAD_GRAYSCALE)

        # Igual que HTRDataset.__getitem__: divide por 255, añade canal, normaliza
        tensor = torch.tensor(img, dtype=torch.float32) / 255.0
        tensor = tensor.unsqueeze(0)       # (1, H, W)  — dimensión de canal
        tensor = self.normalize(tensor)    # Normalize(mean=0.5, std=0.5)
        tensor = tensor.unsqueeze(0)       # (1, 1, H, W) — dimensión de batch
        return tensor

    @torch.no_grad()
    def predict(self, img_path_or_array, use_beam_search=False, beam_width=10):
        tensor    = self.preprocess(img_path_or_array).to(self.device)
        log_probs = self.model(tensor)
        if use_beam_search:
            return self.decoder.beam_search_decode(log_probs, beam_width)
        return self.decoder.greedy_decode(log_probs)[0]

def _encode(img_uint8: np.ndarray, img_path: str) -> bytes:
    """
    Codifica un array uint8 como PNG o JPEG en memoria según la extensión del path.
    PNG usa compresión máxima (ideal para imágenes binarizadas).
    JPEG usa calidad 95 (balance entre tamaño y fidelidad).
    """
    ext = Path(img_path).suffix.lower()

    if ext == '.png':
        params = [cv2.IMWRITE_PNG_COMPRESSION, 9]
    elif ext in ('.jpg', '.jpeg'):
        params = [cv2.IMWRITE_JPEG_QUALITY, 95]
    else:
        raise ValueError(f"Extensión no soportada: '{ext}'. Usa .png o .jpg/.jpeg")

    ok, buf = cv2.imencode(ext, img_uint8, params)
    if not ok:
        raise RuntimeError(f"cv2.imencode falló para '{img_path}'")
    return buf.tobytes()

def _decode(img_bytes: bytes) -> np.ndarray:
    """Decodifica bytes PNG o JPEG a array uint8 en escala de grises."""
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError("cv2.imdecode falló: bytes corruptos o formato no reconocido")
    return img
 
class HTRDataset(Dataset):
 
    def __init__(self, image_paths, labels, charset,
                 img_height=32, img_width=256):
        self.image_paths = image_paths
        self.labels      = list(labels.values()) if isinstance(labels, dict) else list(labels)
        self.img_height  = img_height
        self.img_width   = img_width
        self.char2idx    = {c: i + 1 for i, c in enumerate(charset)}
        self.charset     = charset
        self.normalize   = transforms.Normalize(mean=[0.5], std=[0.5])
 
        self._tensors_cache = None
        self._images_png    = None
        self._labels_cache  = None
 
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
                    # binarized = preprocess_image_binarized(
                    #     str(img_path), self.img_height
                    # )
                    binarized = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

                    png_bytes     = _encode(binarized, img_path)
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
    def from_precomputed_chunked(cls, chunks_dir, charset, img_width=None):
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
    def from_precomputed(cls, precomputed_path, charset):
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
        # Modo PNG bytes (recomendado)
        if self._images_png is not None:
            img = _decode(self._images_png[idx])
 
            use_fixed_width = self.img_width and self.img_width > 0
            if use_fixed_width:
                img = resize_img(img, self.img_height, self.img_width)
 
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
