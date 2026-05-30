#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Importamos el SRI y las métricas SIN MODIFICARLOS. Asumimos que la
# carpeta sri/ (con sri.py, metrics.py) está junto a este script o se pasa
# vía --sri-path.
def _import_sri(sri_path: Path):
    sys.path.insert(0, str(sri_path))
    from sri import HybridRetriever, RetrievalRun       # noqa
    from metrics import evaluate_run, compare_runs, format_comparison_table  # noqa
    return HybridRetriever, RetrievalRun, evaluate_run, compare_runs, format_comparison_table

def read_docs(docs_dir: Path) -> tuple[list[str], list[str]]:
    """Lee docs/*.txt → (doc_ids, texts). doc_id = nombre sin extensión."""
    files = sorted(docs_dir.glob('*.txt'))
    if not files:
        raise FileNotFoundError(f'No hay .txt en {docs_dir}')
    doc_ids, texts = [], []
    for fp in files:
        doc_ids.append(fp.stem)
        # Unimos las líneas del documento en un solo texto a indexar.
        texts.append(fp.read_text(encoding='utf-8').replace('\n', ' ').strip())
    return doc_ids, texts

def read_queries(path: Path) -> dict[str, str]:
    """TXT: 'qid<TAB>texto' por línea, o sólo texto (autonumera q1,q2,...)."""
    queries: dict[str, str] = {}
    auto = 0
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
    """
    TREC qrels: 'qid iter doc_id rel' (o corto 'qid doc_id rel').
    Devuelve (qrels_binary, qrels_graded).
    """
    qrels_bin: dict[str, dict[str, int]] = {}
    qrels_grd: dict[str, dict[str, int]] = {}
    for raw in path.read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) == 4:
            qid, _iter, doc_id, rel = parts
        elif len(parts) == 3:
            qid, doc_id, rel = parts
        else:
            print(f'  ⚠ Línea de qrels ignorada (formato inesperado): {raw!r}')
            continue
        rel = int(rel)
        qrels_grd.setdefault(qid, {})[doc_id] = max(rel, 0)
        if rel >= binary_threshold:
            qrels_bin.setdefault(qid, {})[doc_id] = 1
    if not qrels_bin:
        print('  ⚠ Ningún juicio supera el umbral binario; las métricas '
              'binarias serán 0. Revisa --binary-threshold.')
    return qrels_bin, qrels_grd

def run_methods(retriever, queries, methods, top_k):
    runs = {}
    for m in methods:
        if m == 'bm25':
            runs[m] = retriever.search_bm25_batch(queries, top_k=top_k)
        elif m == 'semantic':
            runs[m] = retriever.search_semantic_batch(queries, top_k=top_k,
                                                      show_progress=False)
        elif m == 'hybrid':
            runs[m] = retriever.search_hybrid_batch(queries, top_k=top_k,
                                                    top_k_per_method=top_k,
                                                    show_progress=False)
    return runs

def evaluate_corpus(label, docs_dir, queries, qrels_bin, qrels_grd,
                    model_path, methods, ks, top_k, evaluate_run,
                    HybridRetriever, k1, b, rrf_k, device):
    """Indexa un corpus y evalúa todos los métodos. Devuelve {method: eval}."""
    print(f'\n  Indexando corpus [{label}] desde {docs_dir} ...')
    doc_ids, texts = read_docs(docs_dir)
    print(f'    {len(doc_ids)} documentos.')
    retriever = HybridRetriever(model_path=model_path, k1=k1, b=b,
                                rrf_k=rrf_k, device=device)
    retriever.index(doc_ids, texts, show_progress=False)

    runs = run_methods(retriever, queries, methods, top_k)
    evals = {}
    for m, r in runs.items():
        doc_lists = {qid: [d for d, _ in items] for qid, items in r.runs.items()}
        evals[m] = evaluate_run(doc_lists, qrels_binary=qrels_bin,
                                qrels_graded=qrels_grd, ks=ks)
    return evals

def print_table(label, evals, ks):
    print(f'\n  ── Métricas [{label}] ' + '─' * 40)
    metrics = ['map', 'mrr', 'r_prec'] + \
              [f'p@{k}' for k in ks] + [f'r@{k}' for k in ks] + \
              [f'ndcg@{k}' for k in ks]
    methods = list(evals.keys())
    print(f'    {"métrica":<12}' + ''.join(f'{m:>12}' for m in methods))
    for met in metrics:
        if not any(met in evals[m]['aggregated'] for m in methods):
            continue
        row = f'    {met:<12}'
        for m in methods:
            v = evals[m]['aggregated'].get(met)
            row += f'{v:>12.4f}' if v is not None else f'{"—":>12}'
        print(row)

def main() -> int:
    ap = argparse.ArgumentParser(
        description='Evalúa el SRI híbrido sobre los .txt de docs con queries/qrels.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument('--docs', required=True,
                    help='Carpeta con los .txt (salida TrOCR). doc_id = nombre sin .txt')
    ap.add_argument('--docs-gt', default=None,
                    help='Carpeta con los .txt de ground-truth para comparar '
                         'el arrastre de error (opcional).')
    ap.add_argument('--queries', required=True, help='TXT de queries (qid<TAB>texto).')
    ap.add_argument('--qrels', required=True, help='TXT de qrels (TREC).')
    ap.add_argument('--model', type=Path, default=None,
                    help='Modelo E5 local. Si se omite, sólo BM25.')
    ap.add_argument('--sri-path', type=Path,
                    default=Path(__file__).resolve().parent / 'sri',
                    help='Carpeta con sri.py y metrics.py (default ./sri).')
    ap.add_argument('--methods', default='bm25,semantic,hybrid',
                    help='Coma-sep: bm25,semantic,hybrid (default todos).')
    ap.add_argument('--ks', default='1,3,5,10',
                    help='Cutoffs coma-sep para P@k/R@k/nDCG@k (default 1,3,5,10).')
    ap.add_argument('--top-k', type=int, default=1000)
    ap.add_argument('--binary-threshold', type=int, default=1)
    ap.add_argument('--bm25-k1', type=float, default=1.5)
    ap.add_argument('--bm25-b', type=float, default=0.75)
    ap.add_argument('--rrf-k', type=int, default=60)
    ap.add_argument('--device', default=None)
    ap.add_argument('--out', default='./salida_sri')
    args = ap.parse_args()

    # Importar SRI sin tocarlo
    (HybridRetriever, RetrievalRun, evaluate_run,
     compare_runs, format_comparison_table) = _import_sri(args.sri_path)

    methods = [m.strip() for m in args.methods.split(',') if m.strip()]
    ks = sorted({int(x) for x in args.ks.split(',') if x.strip()})

    # Si pidieron semantic/hybrid sin modelo, degradar a BM25
    if any(m in methods for m in ('semantic', 'hybrid')) and args.model is None:
        print('  ⚠ Sin --model: semantic/hybrid no disponibles, sólo BM25.')
        methods = [m for m in methods if m == 'bm25'] or ['bm25']

    queries = read_queries(Path(args.queries))
    qrels_bin, qrels_grd = read_qrels(Path(args.qrels), args.binary_threshold)
    print(f'  Queries: {len(queries)} | Queries con juicios: '
          f'{sum(1 for q in queries if qrels_bin.get(q))}')

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    evals_ocr = evaluate_corpus(
        'TrOCR', Path(args.docs), queries, qrels_bin, qrels_grd,
        args.model, methods, ks, args.top_k, evaluate_run,
        HybridRetriever, args.bm25_k1, args.bm25_b, args.rrf_k, args.device)
    print_table('TrOCR', evals_ocr, ks)

    payload = {
        'config': {
            'docs': str(args.docs), 'queries': str(args.queries),
            'qrels': str(args.qrels), 'model': str(args.model) if args.model else None,
            'methods': methods, 'ks': ks, 'top_k': args.top_k,
            'binary_threshold': args.binary_threshold,
            'bm25_k1': args.bm25_k1, 'bm25_b': args.bm25_b, 'rrf_k': args.rrf_k,
        },
        'n_queries': len(queries),
        'aggregated_trocr': {m: e['aggregated'] for m, e in evals_ocr.items()},
    }

    # ── Corpus de referencia: ground-truth (arrastre de error) ────────
    if args.docs_gt:
        evals_gt = evaluate_corpus(
            'GroundTruth', Path(args.docs_gt), queries, qrels_bin, qrels_grd,
            args.model, methods, ks, args.top_k, evaluate_run,
            HybridRetriever, args.bm25_k1, args.bm25_b, args.rrf_k, args.device)
        print_table('GroundTruth', evals_gt, ks)
        payload['aggregated_groundtruth'] = {
            m: e['aggregated'] for m, e in evals_gt.items()}

        # Tests de significancia: GT (A) vs TrOCR (B) por método.
        # delta>0 ⇒ el OCR degradó la métrica → eso es el arrastre de error.
        print('\n  ══ ARRASTRE DE ERROR: GroundTruth vs TrOCR (por método) ══')
        comparisons = {}
        for m in methods:
            if m not in evals_gt or m not in evals_ocr:
                continue
            cmp = compare_runs(
                evals_gt[m], evals_ocr[m],
                name_a=f'{m}_GT', name_b=f'{m}_TrOCR',
                metrics_to_test=tuple(
                    x for x in ('ap', 'rr', 'r_prec',
                                *[f'ndcg@{k}' for k in ks],
                                *[f'p@{k}' for k in ks])),
            )
            comparisons[m] = cmp
            print(f'\n  [{m}]  (delta = GT − TrOCR; >0 ⇒ el OCR degradó)')
            try:
                print(format_comparison_table(cmp))
            except Exception as e:
                print(f'    (no se pudo formatear tabla: {e})')
        payload['error_propagation_tests'] = comparisons

    out_json = out_dir / 'sri_results.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # Runs TREC y per-query CSV de la corrida TrOCR
    print(f'\n  Resultados → {out_json}')
    return 0

if __name__ == '__main__':
    sys.exit(main())
