#!/usr/bin/env python3

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

def load_model(model_path: str, num_beams: int, max_length: int):
    """Carga TrOCR processor + model con la config correcta."""
    print(f'Cargando modelo desde {model_path}...')
    t0 = time.perf_counter()

    try:
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    except ImportError as e:
        print(f'\nERROR: falta dependencia: {e}')
        print('Instala con: pip install transformers sentencepiece torch Pillow')
        print('(en Python 3.13 Windows, asegura sentencepiece>=0.2.1)')
        sys.exit(1)

    model_path = Path(model_path)
    if not model_path.is_dir():
        print(f'\nERROR: la carpeta {model_path} no existe.')
        sys.exit(1)

    # Acepta pytorch_model.bin (formato antiguo) o model.safetensors (formato moderno)
    has_weights = (
        (model_path / 'pytorch_model.bin').is_file() or
        (model_path / 'model.safetensors').is_file()
    )
    # Acepta sentencepiece.bpe.model (SPM) o tokenizer.json (fast tokenizer)
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
        print(f'Asegúrate de descomprimir final_model.zip completo.')
        sys.exit(1)

    # use_fast=True cuando hay tokenizer.json; False solo si solo hay SPM
    use_fast = (model_path / 'tokenizer.json').is_file()
    processor = TrOCRProcessor.from_pretrained(str(model_path), use_fast=use_fast)
    model = VisionEncoderDecoderModel.from_pretrained(str(model_path))
    model.eval()

    # Configuración de generación (transformers ≥4.50)
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

def predict_one(processor, model, image_path: Path) -> tuple[str, float]:
    """Devuelve (texto, latencia_ms)."""
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
    """Distancia de edición de Levenshtein en Python puro.

    Implementación O(n*m) tiempo, O(min(n,m)) memoria. Para líneas
    de hasta ~200 caracteres es instantáneo (<1 ms).
    """
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
    """CER (Character Error Rate). Devuelve valor en [0, ∞), típicamente [0, 1]."""
    return levenshtein(pred, gt) / max(len(gt), 1)

def main():
    parser = argparse.ArgumentParser(
        description='Standalone tester para TrOCR fine-tuneado',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--model', required=True,
                        help='Carpeta con el modelo TrOCR')
    parser.add_argument('--image', required=True,
                        help='Imagen o carpeta de imágenes')
    parser.add_argument('--gt', default=None,
                        help='JSON {filename: text} para comparar (opcional)')
    parser.add_argument('--num-beams', type=int, default=4)
    parser.add_argument('--max-length', type=int, default=128)
    parser.add_argument('--benchmark', type=int, default=0,
                        help='Repite cada inferencia N veces')
    parser.add_argument('--output', default=None,
                        help='Guarda resultados a CSV')

    args = parser.parse_args()

    # Resolver lista de imágenes
    img_arg = Path(args.image)
    if img_arg.is_dir():
        images = sorted(
            list(img_arg.glob('*.jpg')) +
            list(img_arg.glob('*.jpeg')) +
            list(img_arg.glob('*.png'))
        )
        print(f'Carpeta detectada: {len(images)} imágenes')
    elif img_arg.is_file():
        images = [img_arg]
    else:
        print(f'ERROR: --image debe ser imagen o carpeta. Recibí: {img_arg}')
        sys.exit(1)

    if not images:
        print('No se encontró ninguna imagen.')
        sys.exit(1)

    # Cargar ground truth si se proporcionó
    gt_dict = None
    if args.gt:
        gt_path = Path(args.gt)
        if not gt_path.is_file():
            print(f'ERROR: {gt_path} no existe.')
            sys.exit(1)
        with open(gt_path) as f:
            data = json.load(f)
        # Tolerar varios formatos
        if isinstance(data, dict):
            v = next(iter(data.values()))
            if isinstance(v, str):
                gt_dict = data
            elif isinstance(v, dict) and 'text' in v:
                gt_dict = {k: x['text'] for k, x in data.items()}
        elif isinstance(data, list):
            gt_dict = {x['file']: x['text'] for x in data}
        print(f'Ground truth: {len(gt_dict)} entradas')

    # Cargar modelo
    processor, model = load_model(args.model, args.num_beams, args.max_length)

    # Inferencia
    print('\n' + '=' * 70)
    print(f' INFERENCIA SOBRE {len(images)} IMAGEN(ES)')
    print('=' * 70)

    results = []
    cer_values = []
    all_times = []

    for img_path in images:
        print(f'\n  📄 {img_path.name}')

        if args.benchmark > 0:
            times = []
            for i in range(args.benchmark):
                text, dt = predict_one(processor, model, img_path)
                times.append(dt)
            avg_text = text  # asumimos determinismo
            all_times.extend(times)
            print(f'     → {avg_text}')
            print(f'     latencia: min={min(times):.0f}ms, '
                  f'mediana={statistics.median(times):.0f}ms, '
                  f'max={max(times):.0f}ms')
        else:
            text, dt = predict_one(processor, model, img_path)
            all_times.append(dt)
            print(f'     → {text}')
            print(f'     ({dt:.0f} ms)')

        row = {'file': img_path.name, 'pred': text, 'time_ms': all_times[-1]}

        if gt_dict and img_path.name in gt_dict:
            gt = gt_dict[img_path.name]
            cer = compute_cer(text, gt)
            cer_values.append(cer)
            row['gt'] = gt
            row['cer'] = cer
            print(f'     gt:  {gt}')
            print(f'     CER: {cer:.4f}')

        results.append(row)

    # Resumen
    print('\n' + '=' * 70)
    print(' RESUMEN')
    print('=' * 70)
    print(f'  Imágenes procesadas:     {len(results)}')
    print(f'  Latencia media:          {statistics.mean(all_times):.0f} ms/línea')
    print(f'  Latencia mediana:        {statistics.median(all_times):.0f} ms/línea')

    if cer_values:
        print(f'\n  CER global (micro avg):  {sum(cer_values)/len(cer_values):.4f}')
        print(f'  CER mejor:               {min(cer_values):.4f}')
        print(f'  CER peor:                {max(cer_values):.4f}')
        n_perfect = sum(1 for c in cer_values if c == 0)
        print(f'  Líneas perfectas:        {n_perfect}/{len(cer_values)}')

    # Guardar CSV si se pidió
    if args.output:
        import csv
        keys = ['file', 'pred', 'gt', 'cer', 'time_ms']
        with open(args.output, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        print(f'\n  CSV guardado en: {args.output}')

    print()

if __name__ == '__main__':
    main()