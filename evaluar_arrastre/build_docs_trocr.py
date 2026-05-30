#!/usr/bin/env python3
"""
build_docs_trocr.py — Genera un .txt por documento con la transcripción de TrOCR.

PIPELINE
════════
Entrada:
  - Una carpeta de imágenes de SEGMENTOS (líneas). Cada imagen se llama
        {hash}_linea_NN.jpg
    donde {hash} es un prefijo hexadecimal que identifica el DOCUMENTO al
    que pertenece la línea, y NN es el número de línea dentro del documento.
  - Un JSON cuya LLAVE es el nombre de la imagen del segmento y cuyo VALOR
    es la transcripción ground-truth de ese segmento (formato test.json).
    El JSON se usa para (a) saber qué segmentos pertenecen a qué documento
    y en qué orden van, y (b) opcionalmente calcular el CER por segmento /
    por documento (arrastre del error del OCR).

Salida:
  - docs/{hash}.txt           → transcripción de TrOCR de todo el documento,
                                una línea por segmento, en orden de línea.
  - docs_gt/{hash}.txt        → (opcional) ground-truth del documento, mismo
                                orden, para alinear con la evaluación del SRI.
  - trocr_segments.csv        → CER por segmento (pred vs gt, latencia).
  - trocr_docs_cer.json       → CER agregado por documento + global, para el
                                análisis estadístico del arrastre de error.

Las funciones load_model() y predict_one() se reutilizan TAL CUAL de
test_trocr_standalone.py (mismo modelo, misma config de generación,
mismo cómputo de CER por Levenshtein), de modo que el CER reportado aquí
es idéntico al del tester standalone.

USO
═══
    python build_docs_trocr.py \
        --model   /ruta/a/trocr_es_finetuned \
        --images  /ruta/a/segmentos/ \
        --gt      /ruta/a/test.json \
        --out     ./salida_ocr

    # Sin modelo, modo "oráculo" (copia el GT como si el OCR fuera perfecto;
    # útil para verificar el pipeline y como baseline CER=0 del SRI):
    python build_docs_trocr.py --gt test.json --out ./salida_ocr --oracle
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from pathlib import Path

# ── Reutilizamos las funciones del tester standalone ──────────────────────
# Se importa el módulo si está disponible en el PYTHONPATH; si no, se
# definen copias idénticas para que el script sea autocontenido.
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_trocr_standalone import (  # type: ignore
        load_model, predict_one, compute_cer, levenshtein,
    )
    _IMPORTED = True
except Exception:
    _IMPORTED = False

    import time

    def load_model(model_path: str, num_beams: int, max_length: int):
        """Copia idéntica de test_trocr_standalone.load_model."""
        print(f'Cargando modelo desde {model_path}...')
        t0 = time.perf_counter()
        try:
            import torch  # noqa: F401
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        except ImportError as e:
            print(f'\nERROR: falta dependencia: {e}')
            print('Instala con: pip install transformers sentencepiece torch Pillow')
            sys.exit(1)

        model_path = Path(model_path)
        if not model_path.is_dir():
            print(f'\nERROR: la carpeta {model_path} no existe.')
            sys.exit(1)

        has_weights = (
            (model_path / 'pytorch_model.bin').is_file() or
            (model_path / 'model.safetensors').is_file()
        )
        has_tokenizer = (
            (model_path / 'sentencepiece.bpe.model').is_file() or
            (model_path / 'tokenizer.json').is_file()
        )
        missing = []
        if not (model_path / 'config.json').is_file():
            missing.append('config.json')
        if not has_weights:
            missing.append('pytorch_model.bin  o  model.safetensors')
        if not has_tokenizer:
            missing.append('sentencepiece.bpe.model  o  tokenizer.json')
        if missing:
            print(f'\nERROR: archivos faltantes en el modelo: {missing}')
            sys.exit(1)

        use_fast = (model_path / 'tokenizer.json').is_file()
        processor = TrOCRProcessor.from_pretrained(str(model_path), use_fast=use_fast)
        model = VisionEncoderDecoderModel.from_pretrained(str(model_path))
        model.eval()

        tok = processor.tokenizer
        gc = model.generation_config
        gc.decoder_start_token_id = tok.bos_token_id
        gc.pad_token_id           = tok.pad_token_id
        gc.eos_token_id           = tok.eos_token_id
        gc.bos_token_id           = tok.bos_token_id
        gc.max_length             = max_length
        gc.num_beams              = num_beams
        gc.early_stopping         = True
        gc.no_repeat_ngram_size   = 3
        gc.length_penalty         = 2.0

        elapsed = time.perf_counter() - t0
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f'  Cargado en {elapsed:.1f}s — {n_params:.1f}M params')
        print(f'  Tokenizer: {type(tok).__name__}, vocab={tok.vocab_size}')
        print(f'  Beams: {num_beams}, max_length: {max_length}')
        return processor, model

    def predict_one(processor, model, image_path: Path):
        """Copia idéntica de test_trocr_standalone.predict_one."""
        import torch
        from PIL import Image
        img = Image.open(image_path).convert('RGB')
        t0 = time.perf_counter()
        pixel_values = processor(images=img, return_tensors='pt').pixel_values
        with torch.no_grad():
            ids = model.generate(pixel_values)
        text = processor.batch_decode(ids, skip_special_tokens=True)[0]
        dt_ms = (time.perf_counter() - t0) * 1000
        return text.strip(), dt_ms

    def levenshtein(s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            s1, s2 = s2, s1
        if not s2:
            return len(s1)
        previous = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current = [i + 1]
            for j, c2 in enumerate(s2):
                insert     = previous[j + 1] + 1
                delete     = current[j] + 1
                substitute = previous[j] + (c1 != c2)
                current.append(min(insert, delete, substitute))
            previous = current
        return previous[-1]

    def compute_cer(pred: str, gt: str) -> float:
        return levenshtein(pred, gt) / max(len(gt), 1)

_SEG_RE = re.compile(r'^(?P<hash>[0-9a-fA-F]+)_linea_(?P<line>\d+)\.', re.IGNORECASE)

def parse_segment_key(key: str):
    """Devuelve (doc_hash, line_number) o (None, None) si no matchea."""
    m = _SEG_RE.match(key)
    if not m:
        return None, None
    return m.group('hash'), int(m.group('line'))

def load_gt(gt_path: Path) -> dict[str, str]:
    """Carga el JSON {filename: text} tolerando los formatos del tester."""
    with open(gt_path, encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict):
        v = next(iter(data.values()))
        if isinstance(v, str):
            return data
        if isinstance(v, dict) and 'text' in v:
            return {k: x['text'] for k, x in data.items()}
    if isinstance(data, list):
        return {x['file']: x['text'] for x in data}
    raise ValueError('Formato de GT no reconocido.')

def group_by_document(gt: dict[str, str]) -> dict[str, list[tuple[int, str]]]:
    """
    Agrupa las llaves del GT por hash de documento.
    Devuelve {hash -> [(line_no, segment_filename), ...]} ordenado por line_no.
    """
    groups: dict[str, list[tuple[int, str]]] = {}
    for key in gt:
        h, line = parse_segment_key(key)
        if h is None:
            print(f'  ⚠ Llave ignorada (no matchea patrón): {key!r}')
            continue
        groups.setdefault(h, []).append((line, key))
    for h in groups:
        groups[h].sort(key=lambda t: t[0])  # ordenar por número de línea
    return groups

def main() -> int:
    ap = argparse.ArgumentParser(
        description='Genera un .txt por documento con la transcripción de TrOCR.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument('--gt', required=True,
                    help='JSON {segmento.jpg: transcripción}. Define documentos y orden.')
    ap.add_argument('--images', default=None,
                    help='Carpeta con las imágenes de segmentos. Obligatorio salvo --oracle.')
    ap.add_argument('--model', default=None,
                    help='Carpeta del modelo TrOCR. Obligatorio salvo --oracle.')
    ap.add_argument('--out', default='./salida_ocr',
                    help='Carpeta de salida (default ./salida_ocr).')
    ap.add_argument('--oracle', action='store_true',
                    help='No corre TrOCR: usa el GT como "predicción perfecta" '
                         '(CER=0). Sirve de baseline y para validar el pipeline.')
    ap.add_argument('--num-beams', type=int, default=4)
    ap.add_argument('--max-length', type=int, default=128)
    ap.add_argument('--no-gt-docs', action='store_true',
                    help='No escribir los .txt de ground-truth en docs_gt/.')
    args = ap.parse_args()

    gt_path = Path(args.gt)
    if not gt_path.is_file():
        print(f'ERROR: {gt_path} no existe.')
        return 1
    gt = load_gt(gt_path)
    groups = group_by_document(gt)
    print(f'GT: {len(gt)} segmentos → {len(groups)} documentos')

    out_dir = Path(args.out)
    docs_dir = out_dir / 'docs'
    docs_gt_dir = out_dir / 'docs_gt'
    docs_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_gt_docs:
        docs_gt_dir.mkdir(parents=True, exist_ok=True)

    processor = model = None
    if not args.oracle:
        if not args.model or not args.images:
            print('ERROR: en modo real hacen falta --model y --images '
                  '(o usa --oracle para validar sin modelo).')
            return 1
        images_dir = Path(args.images)
        if not images_dir.is_dir():
            print(f'ERROR: la carpeta de imágenes {images_dir} no existe.')
            return 1
        processor, model = load_model(args.model, args.num_beams, args.max_length)
    else:
        images_dir = Path(args.images) if args.images else None
        print('Modo ORÁCULO: las predicciones son el ground-truth (CER=0).')

    seg_rows: list[dict] = []          # filas para trocr_segments.csv
    doc_cer: dict[str, dict] = {}      # CER agregado por documento
    all_seg_cers: list[float] = []     # micro-promedio global

    print('\n' + '=' * 70)
    print(' GENERANDO TXT POR DOCUMENTO')
    print('=' * 70)

    for h, items in sorted(groups.items()):
        pred_lines: list[str] = []
        gt_lines: list[str] = []
        per_doc_cers: list[float] = []
        # CER ponderado por caracteres (a nivel documento, no promedio de líneas)
        doc_edits = 0
        doc_gt_chars = 0

        print(f'\n  📄 {h}  ({len(items)} segmentos)')

        for line_no, seg_name in items:
            gt_text = gt[seg_name]

            if args.oracle:
                pred_text = gt_text
                dt_ms = 0.0
            else:
                img_path = images_dir / seg_name
                if not img_path.is_file():
                    # Tolerar extensiones alternativas
                    alt = None
                    for ext in ('.jpg', '.jpeg', '.png'):
                        cand = images_dir / (Path(seg_name).stem + ext)
                        if cand.is_file():
                            alt = cand
                            break
                    if alt is None:
                        print(f'     ⚠ Imagen no encontrada: {seg_name} (línea omitida)')
                        continue
                    img_path = alt
                pred_text, dt_ms = predict_one(processor, model, img_path)

            cer = compute_cer(pred_text, gt_text)
            per_doc_cers.append(cer)
            all_seg_cers.append(cer)
            doc_edits += levenshtein(pred_text, gt_text)
            doc_gt_chars += max(len(gt_text), 1)

            pred_lines.append(pred_text)
            gt_lines.append(gt_text)

            seg_rows.append({
                'doc': h,
                'line': line_no,
                'file': seg_name,
                'pred': pred_text,
                'gt': gt_text,
                'cer': f'{cer:.6f}',
                'time_ms': f'{dt_ms:.1f}',
            })

        # Escribir TXT del documento (predicción TrOCR)
        doc_txt = docs_dir / f'{h}.txt'
        doc_txt.write_text('\n'.join(pred_lines) + '\n', encoding='utf-8')

        # Escribir TXT del ground-truth (para alinear con el SRI)
        if not args.no_gt_docs:
            (docs_gt_dir / f'{h}.txt').write_text(
                '\n'.join(gt_lines) + '\n', encoding='utf-8')

        # CER del documento: dos definiciones
        cer_macro = statistics.mean(per_doc_cers) if per_doc_cers else 0.0
        cer_micro = doc_edits / max(doc_gt_chars, 1)
        doc_cer[h] = {
            'n_segments': len(per_doc_cers),
            'cer_macro_avg_lines': cer_macro,    # promedio de CER por línea
            'cer_micro_chars': cer_micro,        # ediciones totales / chars totales
            'total_edits': doc_edits,
            'total_gt_chars': doc_gt_chars,
        }
        print(f'     CER doc (micro, por chars): {cer_micro:.4f}  |  '
              f'(macro, por línea): {cer_macro:.4f}')

    print('\n' + '=' * 70)
    print(' RESUMEN GLOBAL')
    print('=' * 70)
    n_docs = len(doc_cer)
    micro_global = (sum(d['total_edits'] for d in doc_cer.values()) /
                    max(sum(d['total_gt_chars'] for d in doc_cer.values()), 1))
    macro_seg = statistics.mean(all_seg_cers) if all_seg_cers else 0.0
    macro_doc = (statistics.mean(d['cer_micro_chars'] for d in doc_cer.values())
                 if doc_cer else 0.0)
    print(f'  Documentos generados:        {n_docs}')
    print(f'  Segmentos procesados:        {len(all_seg_cers)}')
    print(f'  CER global (micro, chars):   {micro_global:.4f}')
    print(f'  CER macro (prom. segmentos): {macro_seg:.4f}')
    print(f'  CER macro (prom. documentos):{macro_doc:.4f}')
    if all_seg_cers:
        print(f'  CER segmento mejor / peor:   '
              f'{min(all_seg_cers):.4f} / {max(all_seg_cers):.4f}')
        n_perfect = sum(1 for c in all_seg_cers if c == 0)
        print(f'  Segmentos perfectos:         {n_perfect}/{len(all_seg_cers)}')

    seg_csv = out_dir / 'trocr_segments.csv'
    with open(seg_csv, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['doc', 'line', 'file', 'pred',
                                          'gt', 'cer', 'time_ms'])
        w.writeheader()
        w.writerows(seg_rows)

    cer_json = out_dir / 'trocr_docs_cer.json'
    payload = {
        'mode': 'oracle' if args.oracle else 'trocr',
        'n_documents': n_docs,
        'n_segments': len(all_seg_cers),
        'cer_global_micro_chars': micro_global,
        'cer_macro_avg_segments': macro_seg,
        'cer_macro_avg_documents': macro_doc,
        'per_document': doc_cer,
        'per_segment_cer': all_seg_cers,
    }
    with open(cer_json, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f'\n  TXT por documento → {docs_dir}/')
    if not args.no_gt_docs:
        print(f'  TXT ground-truth  → {docs_gt_dir}/')
    print(f'  CER por segmento  → {seg_csv}')
    print(f'  CER por documento → {cer_json}')
    print()
    return 0

if __name__ == '__main__':
    sys.exit(main())
