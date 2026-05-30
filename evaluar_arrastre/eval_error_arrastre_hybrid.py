#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

def _import_sri(sri_path: Path):
    sys.path.insert(0, str(sri_path))
    from sri import HybridRetriever            # noqa
    from metrics import (evaluate_run, compare_runs,            # noqa
                         format_comparison_table,
                         shapiro_wilk_normality)
    return (HybridRetriever, evaluate_run, compare_runs,
            format_comparison_table, shapiro_wilk_normality)

# ── Lectores (compatibles con eval_sri_on_docs.py) ─────────────────────────
def read_docs(docs_dir: Path):
    files = sorted(docs_dir.glob('*.txt'))
    if not files:
        raise FileNotFoundError(f'No hay .txt en {docs_dir}')
    doc_ids, texts = [], []
    for fp in files:
        doc_ids.append(fp.stem)
        texts.append(fp.read_text(encoding='utf-8').replace('\n', ' ').strip())
    return doc_ids, texts

def read_queries(path: Path):
    queries, auto = {}, 0
    for raw in path.read_text(encoding='utf-8').splitlines():
        line = raw.rstrip('\n')
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        if '\t' in line:
            qid, text = line.split('\t', 1)
            queries[qid.strip()] = text.strip()
        else:
            auto += 1
            queries[f'q{auto}'] = line.strip()
    if not queries:
        raise ValueError(f'No se leyeron queries de {path}')
    return queries

def read_qrels(path: Path, binary_threshold: int = 1):
    qrels_bin, qrels_grd = {}, {}
    for raw in path.read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) == 4:
            qid, _it, doc_id, rel = parts
        elif len(parts) == 3:
            qid, doc_id, rel = parts
        else:
            print(f'  ⚠ qrels: línea ignorada: {raw!r}')
            continue
        rel = int(rel)
        qrels_grd.setdefault(qid, {})[doc_id] = max(rel, 0)
        if rel >= binary_threshold:
            qrels_bin.setdefault(qid, {})[doc_id] = 1
    return qrels_bin, qrels_grd

def read_cer(path: Path) -> dict[str, float]:
    """Lee doc_id,cer. Autodetecta fracción vs porcentaje. Devuelve fracción 0..1."""
    rows = []
    text = path.read_text(encoding='utf-8')
    delim = '\t' if (text.count('\t') >= text.count(',')) else ','
    for r in csv.reader(text.splitlines(), delimiter=delim):
        if not r or len(r) < 2:
            continue
        a, b = r[0].strip(), r[1].strip()
        if not a or not b:
            continue
        try:
            val = float(b)
        except ValueError:
            # probablemente la cabecera (doc_id,cer)
            continue
        rows.append((a, val))
    if not rows:
        raise ValueError(f'No se leyó CER de {path}')
    # Autodetección: si algún valor > 1.0 asumimos que TODO viene en %.
    as_percent = any(v > 1.0 for _, v in rows)
    cer = {d: (v / 100.0 if as_percent else v) for d, v in rows}
    unit = 'porcentaje (0..100)' if as_percent else 'fracción (0..1)'
    print(f'  CER leído para {len(cer)} docs (interpretado como {unit}).')
    return cer

# ── Núcleo: indexar + correr SOLO híbrido + evaluar ────────────────────────
def index_and_eval_hybrid(label, docs_dir, queries, qrels_bin, qrels_grd,
                          ks, top_k, HybridRetriever, evaluate_run,
                          model_path, k1, b, rrf_k, device):
    print(f'\n  Indexando [{label}] desde {docs_dir} ...')
    doc_ids, texts = read_docs(docs_dir)
    print(f'    {len(doc_ids)} documentos.')
    retr = HybridRetriever(model_path=model_path, k1=k1, b=b,
                           rrf_k=rrf_k, device=device)
    retr.index(doc_ids, texts, show_progress=False)
    run = retr.search_hybrid_batch(queries, top_k=top_k,
                                   top_k_per_method=top_k, show_progress=False)
    doc_lists = {qid: [d for d, _ in items] for qid, items in run.runs.items()}
    ev = evaluate_run(doc_lists, qrels_binary=qrels_bin,
                      qrels_graded=qrels_grd, ks=ks)
    return ev

def relative_degradation(agg_gt, agg_ocr):
    out = {}
    for met, vgt in agg_gt.items():
        vocr = agg_ocr.get(met)
        if vgt is None or vocr is None:
            continue
        rel = ((vgt - vocr) / vgt * 100.0) if vgt != 0 else (
            0.0 if vocr == 0 else float('inf'))
        out[met] = {'gt': vgt, 'ocr': vocr, 'abs_drop': vgt - vocr, 'rel_pct': rel}
    return out

def _pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return None
    return sxy / math.sqrt(sxx * syy)

def _spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        rk = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                rk[order[k]] = avg
            i = j + 1
        return rk
    return _pearson(ranks(xs), ranks(ys))

def _ols_slope(xs, ys):
    """pérdida ≈ a + b·CER ; devuelve (b, a)."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None, None
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return b, my - b * mx

def correlate_with_backing(xs, ys, shapiro_fn, scipy_stats):
    """
    Correlaciona CER (xs) con pérdida de recuperación (ys) CON respaldo:
      - Shapiro-Wilk sobre AMBAS variables (no se asume normalidad).
      - Pearson y Spearman, cada uno con su p-value (exacto si hay scipy).
      - Elección justificada del coeficiente PRINCIPAL:
          · Pearson sólo si ambas variables pasan normalidad (p>0.05) y n>=10;
            Pearson asume linealidad + normalidad bivariada para su inferencia.
          · En caso contrario Spearman (monótona, robusto a outliers y
            no exige normalidad) — situación esperable con CER sesgado.
      - OLS para la pendiente interpretable (puntos de pérdida por unidad CER).
    Devuelve un dict autoexplicativo (incluye la razón de la elección).
    """
    n = len(xs)
    out = {'n': n}
    if n < 3:
        out['error'] = f'n={n}<3: insuficiente para correlación'
        return out

    norm_x = shapiro_fn(xs)
    norm_y = shapiro_fn(ys)
    out['normality_cer'] = norm_x
    out['normality_loss'] = norm_y

    # Coeficientes + p-values
    if scipy_stats is not None:
        pr = scipy_stats.pearsonr(xs, ys)
        sr = scipy_stats.spearmanr(xs, ys)
        pear_r, pear_p = float(pr[0]), float(pr[1])
        spear_r, spear_p = float(sr[0]), float(sr[1])
    else:
        pear_r = _pearson(xs, ys)
        spear_r = _spearman(xs, ys)
        pear_p = spear_p = None  # sin scipy no damos p exacto

    out['pearson'] = {'r': pear_r, 'p_value': pear_p}
    out['spearman'] = {'rho': spear_r, 'p_value': spear_p}

    # Decisión justificada del coeficiente principal (espeja metrics.py)
    x_normal = bool(norm_x.get('is_normal'))
    y_normal = bool(norm_y.get('is_normal'))
    if x_normal and y_normal and n >= 10:
        primary, reason = 'pearson', (
            f'CER y pérdida pasan Shapiro (p_x={norm_x.get("p_value")}, '
            f'p_y={norm_y.get("p_value")}) y n={n}>=10: '
            f'Pearson válido (normalidad + linealidad)')
    else:
        why = []
        if not x_normal:
            why.append(f'CER no normal (Shapiro p={norm_x.get("p_value")})')
        if not y_normal:
            why.append(f'pérdida no normal (Shapiro p={norm_y.get("p_value")})')
        if n < 10:
            why.append(f'n={n}<10')
        primary, reason = 'spearman', (
            'se prefiere Spearman (monótona, robusta, sin supuesto de '
            'normalidad): ' + '; '.join(why))
    out['primary'] = primary
    out['primary_reason'] = reason

    # OLS para interpretabilidad (pendiente)
    slope, intercept = _ols_slope(xs, ys)
    out['ols'] = {'slope_per_unit_cer': slope, 'intercept': intercept}
    return out

def query_cer(qid, qrels_grd, cer_by_doc, mode='mean_relevant'):
    """CER representativo de una consulta = promedio del CER de sus docs relevantes."""
    rels = list(qrels_grd.get(qid, {}).keys())
    vals = [cer_by_doc[d] for d in rels if d in cer_by_doc]
    if not vals:
        return None
    if mode == 'max_relevant':
        return max(vals)
    return sum(vals) / len(vals)

def main() -> int:
    ap = argparse.ArgumentParser(
        description='Arrastre de error CER(TrOCR) → SRI híbrido.',
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument('--docs', required=True, help='Carpeta .txt salida TrOCR.')
    ap.add_argument('--docs-gt', required=True, help='Carpeta .txt ground-truth.')
    ap.add_argument('--queries', required=True)
    ap.add_argument('--qrels', required=True)
    ap.add_argument('--cer', required=True, help='CSV/TSV doc_id,cer.')
    ap.add_argument('--model', type=Path, required=True,
                    help='Modelo E5 local (obligatorio: el híbrido lo necesita).')
    ap.add_argument('--sri-path', type=Path,
                    default=Path(__file__).resolve().parent / 'sri')
    ap.add_argument('--ks', default='1,3,5,10')
    ap.add_argument('--top-k', type=int, default=1000)
    ap.add_argument('--binary-threshold', type=int, default=1)
    ap.add_argument('--bm25-k1', type=float, default=1.5)
    ap.add_argument('--bm25-b', type=float, default=0.75)
    ap.add_argument('--rrf-k', type=int, default=60)
    ap.add_argument('--device', default=None)
    ap.add_argument('--cer-query-mode', default='mean_relevant',
                    choices=['mean_relevant', 'max_relevant'],
                    help='Cómo resumir el CER por consulta (default: media de relevantes).')
    ap.add_argument('--corr-metric', default='ap',
                    help='Métrica per-query para la correlación CER↔pérdida '
                         '(default ap; p.ej. ap, rr, ndcg@10).')
    ap.add_argument('--out', default='./salida_arrastre')
    args = ap.parse_args()

    (HybridRetriever, evaluate_run, compare_runs,
     format_comparison_table, shapiro_wilk_normality) = _import_sri(args.sri_path)

    # scipy para p-values exactos de las correlaciones (opcional).
    try:
        from scipy import stats as _scipy_stats
    except Exception:
        _scipy_stats = None
        print('  ⚠ scipy no disponible: las correlaciones se reportan sin '
              'p-value exacto (sólo coeficiente).')

    ks = sorted({int(x) for x in args.ks.split(',') if x.strip()})
    queries = read_queries(Path(args.queries))
    qrels_bin, qrels_grd = read_qrels(Path(args.qrels), args.binary_threshold)
    cer_by_doc = read_cer(Path(args.cer))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_judged = sum(1 for q in queries if qrels_bin.get(q))
    print(f'  Queries: {len(queries)} | con juicios: {n_judged}')
    cer_global = (sum(cer_by_doc.values()) / len(cer_by_doc)) if cer_by_doc else float('nan')
    print(f'  CER medio del corpus (TrOCR): {cer_global * 100:.2f}%')

    # Evaluar el HÍBRIDO en ambos corpus
    ev_gt = index_and_eval_hybrid(
        'GT', Path(args.docs_gt), queries, qrels_bin, qrels_grd, ks,
        args.top_k, HybridRetriever, evaluate_run, args.model,
        args.bm25_k1, args.bm25_b, args.rrf_k, args.device)
    ev_ocr = index_and_eval_hybrid(
        'TrOCR', Path(args.docs), queries, qrels_bin, qrels_grd, ks,
        args.top_k, HybridRetriever, evaluate_run, args.model,
        args.bm25_k1, args.bm25_b, args.rrf_k, args.device)

    agg_gt, agg_ocr = ev_gt['aggregated'], ev_ocr['aggregated']

    # (1) Degradación relativa
    rel = relative_degradation(agg_gt, agg_ocr)

    print('\n  ══ (1) ARRASTRE DE ERROR — HÍBRIDO: GT vs TrOCR ══')
    print(f'    {"métrica":<10}{"GT":>10}{"TrOCR":>10}{"Δ abs":>10}{"Δ rel %":>10}')
    order = ['map', 'mrr', 'r_prec'] + \
            [f'ndcg@{k}' for k in ks] + [f'p@{k}' for k in ks] + [f'r@{k}' for k in ks]
    for met in order:
        if met not in rel:
            continue
        d = rel[met]
        print(f'    {met:<10}{d["gt"]:>10.4f}{d["ocr"]:>10.4f}'
              f'{d["abs_drop"]:>10.4f}{d["rel_pct"]:>9.2f}%')

    # (2) Caída absoluta con significancia (test pareado del propio SRI)
    print('\n  ══ (2) CAÍDA ABSOLUTA + SIGNIFICANCIA (pareado por consulta) ══')
    print('     (delta = GT − TrOCR; >0 ⇒ el OCR degradó la recuperación)')
    test_metrics = tuple(['ap', 'rr', 'r_prec'] +
                         [f'ndcg@{k}' for k in ks] + [f'p@{k}' for k in ks])
    cmp = compare_runs(ev_gt, ev_ocr, name_a='hybrid_GT', name_b='hybrid_TrOCR',
                       metrics_to_test=test_metrics)
    try:
        print(format_comparison_table(cmp))
    except Exception as e:
        print(f'    (no se pudo formatear: {e})')

    # (3) Correlación CER ↔ pérdida de recuperación, por consulta
    print('\n  ══ (3) CORRELACIÓN CER ↔ PÉRDIDA DE RECUPERACIÓN ══')
    metric = args.corr_metric
    pq_gt = ev_gt.get('per_query', {})
    pq_ocr = ev_ocr.get('per_query', {})
    xs, ys, table = [], [], []
    for qid in queries:
        if qid not in qrels_bin:
            continue
        c = query_cer(qid, qrels_grd, cer_by_doc, args.cer_query_mode)
        gtv = (pq_gt.get(qid) or {}).get(metric)
        ocv = (pq_ocr.get(qid) or {}).get(metric)
        if c is None or gtv is None or ocv is None:
            continue
        loss = gtv - ocv          # pérdida de calidad en esa consulta
        xs.append(c)
        ys.append(loss)
        table.append((qid, c, gtv, ocv, loss))

    corr_payload = {}
    if len(xs) >= 3:
        res = correlate_with_backing(xs, ys, shapiro_wilk_normality, _scipy_stats)
        print(f'    Métrica de calidad por consulta: {metric}')
        print(f'    Resumen CER por consulta: {args.cer_query_mode}')
        print(f'    n consultas: {res["n"]}')

        nx, ny = res['normality_cer'], res['normality_loss']
        print(f'    Shapiro-Wilk CER:     p={nx.get("p_value")}  '
              f'→ {"normal" if nx.get("is_normal") else "NO normal"}')
        print(f'    Shapiro-Wilk pérdida: p={ny.get("p_value")}  '
              f'→ {"normal" if ny.get("is_normal") else "NO normal"}')

        pe, sp = res['pearson'], res['spearman']
        pe_p = f'{pe["p_value"]:.4f}' if pe['p_value'] is not None else '—'
        sp_p = f'{sp["p_value"]:.4f}' if sp['p_value'] is not None else '—'
        rv = f'{pe["r"]:+.3f}' if pe['r'] is not None else 'indef.'
        rhov = f'{sp["rho"]:+.3f}' if sp['rho'] is not None else 'indef.'
        print(f'    Pearson  r   = {rv}   (p={pe_p})')
        print(f'    Spearman rho = {rhov}   (p={sp_p})')
        print(f'    → Coef. PRINCIPAL: {res["primary"].upper()}')
        print(f'      razón: {res["primary_reason"]}')

        slope = res['ols']['slope_per_unit_cer']
        intercept = res['ols']['intercept']
        if slope is not None:
            print(f'    OLS: pérdida ≈ {intercept:+.4f} {slope:+.4f}·CER')
            print(f'         ⇒ por cada +1 punto de CER (0.01), la pérdida en '
                  f'{metric} cambia ~{slope * 0.01:+.5f}.')

        print(f'\n    {"qid":<6}{"CER%":>8}{metric+"_GT":>12}'
              f'{metric+"_OCR":>12}{"pérdida":>10}')
        for qid, c, gtv, ocv, loss in sorted(table, key=lambda t: -t[1]):
            print(f'    {qid:<6}{c * 100:>7.2f}%{gtv:>12.4f}{ocv:>12.4f}{loss:>10.4f}')

        corr_payload = {
            'metric': metric, 'cer_query_mode': args.cer_query_mode,
            **res,
            'points': [{'qid': q, 'cer': c, f'{metric}_gt': g,
                        f'{metric}_ocr': o, 'loss': l}
                       for q, c, g, o, l in table],
        }
    else:
        print('    No hay suficientes consultas (n>=3) con CER+juicios '
              'para una correlación con respaldo.')

    # Volcado JSON
    payload = {
        'system': 'hybrid (BM25+E5+RRF)',
        'config': {
            'docs': str(args.docs), 'docs_gt': str(args.docs_gt),
            'queries': str(args.queries), 'qrels': str(args.qrels),
            'cer': str(args.cer), 'model': str(args.model),
            'ks': ks, 'top_k': args.top_k,
            'bm25_k1': args.bm25_k1, 'bm25_b': args.bm25_b, 'rrf_k': args.rrf_k,
            'binary_threshold': args.binary_threshold,
        },
        'n_queries': len(queries), 'n_judged': n_judged,
        'cer_corpus_mean': cer_global,
        'aggregated_gt': agg_gt,
        'aggregated_trocr': agg_ocr,
        'relative_degradation': rel,           # (1)
        'significance_tests': cmp,             # (2)
        'cer_loss_correlation': corr_payload,  # (3)
    }
    out_json = out_dir / 'arrastre_resultados.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    print(f'\n  Resultados → {out_json}')
    return 0

if __name__ == '__main__':
    sys.exit(main())
