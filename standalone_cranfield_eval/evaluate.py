#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Imports locales
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sri import HybridRetriever, RetrievalRun  # noqa: E402
from dataset import load_dataset, Collection  # noqa: E402
from metrics import evaluate_run, compare_runs  # noqa: E402

logger = logging.getLogger(__name__)

def _supports_color() -> bool:
    return sys.stdout.isatty()
_C = _supports_color()
def _c(s, code): return f"\033[{code}m{s}\033[0m" if _C else s
def bold(s):   return _c(s, "1")
def dim(s):    return _c(s, "2")
def green(s):  return _c(s, "32")
def yellow(s): return _c(s, "33")
def cyan(s):   return _c(s, "36")
def red(s):    return _c(s, "31")

# Orden estándar de presentación, agrupado lógicamente
METRIC_GROUPS = [
    ("Agregadas (binarias)", ["map", "mrr", "r_prec"]),
    ("P@k",                  ["p@5", "p@10", "p@20", "p@100"]),
    ("R@k",                  ["r@5", "r@10", "r@20", "r@100", "r@1000"]),
    ("F1@k",                 ["f1@5", "f1@10", "f1@20"]),
    ("nDCG@k",               ["ndcg@5", "ndcg@10", "ndcg@20", "ndcg@100"]),
]

def print_metrics_table(method_evals: dict[str, dict]) -> None:
    methods = list(method_evals.keys())
    if not methods:
        return
    header = f"  {'Métrica':<14}" + "".join(f"{m:>14}" for m in methods)
    print()
    print(bold(header))
    print(dim("  " + "─" * (14 + 14 * len(methods))))
    for group_name, metric_list in METRIC_GROUPS:
        any_present = any(any(m in method_evals[mt]["aggregated"]
                              for m in metric_list)
                          for mt in methods)
        if not any_present:
            continue
        print(cyan(f"  {group_name}"))
        for metric in metric_list:
            if not any(metric in method_evals[mt]["aggregated"] for mt in methods):
                continue
            row = f"    {metric:<12}"
            vals = {mt: method_evals[mt]["aggregated"].get(metric) for mt in methods}
            best = max((v for v in vals.values() if v is not None), default=None)
            for mt in methods:
                v = vals.get(mt)
                if v is None:
                    row += f"{'—':>14}"
                else:
                    s = f"{v:.4f}"
                    if best is not None and abs(v - best) < 1e-9:
                        s = bold(green(s))
                    row += f"{s:>{14 + (len(s) - len(f'{v:.4f}'))}}"
            print(row)
    print(dim("  " + "─" * (14 + 14 * len(methods))))
    print(dim(f"  Queries evaluadas: " +
              " / ".join(f"{m}: {method_evals[m]['n_queries_evaluated']}"
                         for m in methods)))

def print_comparison(cmp: dict) -> None:
    if "error" in cmp:
        print(red(f"  ✗ {cmp['error']}"))
        return
    a, b = cmp["name_a"], cmp["name_b"]
    n = cmp["n_common_queries"]
    print()
    print(bold(f"  Comparación pareada {a} vs {b}  (n={n} queries comunes)"))
    print(dim("  " + "─" * 76))
    print(dim(f"  {'Métrica':<10} {a:>10} {b:>10} {'Δ':>10} {'rel%':>8}  "
              f"{'t':>8} {'p (t)':>10} {'p (Wilc.)':>10}"))
    for metric, data in cmp["tests"].items():
        rel = data["rel_delta"]
        rel_s = f"{rel*100:+.1f}%" if rel is not None else "—"
        t_p = data["t_test"].get("p_value")
        w_p = data["wilcoxon"].get("p_value")
        t_p_s = f"{t_p:.4f}" if t_p is not None else "n/a"
        w_p_s = f"{w_p:.4f}" if w_p is not None else "n/a"
        def sigmark(p):
            if p is None: return ""
            if p < 0.01:  return "**"
            if p < 0.05:  return "*"
            return ""
        row = (f"  {metric:<10} "
               f"{data['mean_a']:>10.4f} "
               f"{data['mean_b']:>10.4f} "
               f"{data['delta']:>+10.4f} "
               f"{rel_s:>8}  "
               f"{data['t_test']['t']:>8.2f} "
               f"{t_p_s:>8}{sigmark(t_p):<2} "
               f"{w_p_s:>8}{sigmark(w_p):<2}")
        print(row)
    print(dim("  " + "─" * 76))
    print(dim("  Significancia: * p<0.05, ** p<0.01 (dos colas)"))

def inspect_query(qid: str,
                  coll: Collection,
                  runs: dict[str, RetrievalRun],
                  top_k_show: int = 10) -> None:
    if qid not in coll.queries:
        print(red(f"  Query {qid!r} no existe."))
        return
    rel_set = set(coll.qrels.get(qid, {}).keys())
    graded  = coll.qrels_graded.get(qid, {})

    print()
    print(bold(f"━━━ Query {qid} ━━━"))
    print(f"  {coll.queries[qid][:300]}")
    print(dim(f"  ({len(rel_set)} docs marcados como relevantes)"))

    for method, run in runs.items():
        print()
        print(bold(f"  Resultados [{method}]:"))
        topk = run.top_k(qid, top_k_show)
        if not topk:
            print(dim("    (vacío)"))
            continue
        for i, (doc_id, score) in enumerate(topk, start=1):
            is_rel = doc_id in rel_set
            grade  = graded.get(doc_id, 0)
            mark = green("✓") if is_rel else dim("·")
            grade_s = f"rel={grade}" if grade > 0 else ""
            doc = coll.docs.get(doc_id)
            title = (doc.title or doc.text[:80]) if doc else ""
            title = title.replace("\n", " ")[:80]
            print(f"    {mark} {i:>2}. doc={doc_id:>6} score={score:>9.4f}  "
                  f"{dim(grade_s):>8}  {dim(title)}")

def export_trec_run(run: RetrievalRun, output_path: Path,
                    run_name: str | None = None) -> None:
    name = run_name or run.name
    with open(output_path, "w", encoding="utf-8") as f:
        for qid, ranking in run.runs.items():
            for rank, (doc_id, score) in enumerate(ranking, start=1):
                f.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} {name}\n")
    logger.info("Run TREC escrito: %s", output_path)

def export_metrics_csv(method_evals: dict[str, dict], output_path: Path) -> None:
    methods = list(method_evals.keys())
    if not methods:
        return
    all_metrics = sorted({m for mt in methods
                          for m in method_evals[mt]["aggregated"]})
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("method," + ",".join(all_metrics) + "\n")
        for mt in methods:
            agg = method_evals[mt]["aggregated"]
            row = [mt] + [f"{agg[m]:.6f}" if m in agg else ""
                          for m in all_metrics]
            f.write(",".join(row) + "\n")
    logger.info("CSV escrito: %s", output_path)

def export_per_query_csv(method_evals: dict[str, dict], output_path: Path) -> None:
    methods = list(method_evals.keys())
    all_metrics: list[str] = []
    seen: set[str] = set()
    for mt in methods:
        for q, ms in method_evals[mt]["per_query"].items():
            for m in ms:
                if m not in seen:
                    seen.add(m)
                    all_metrics.append(m)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("method,qid," + ",".join(all_metrics) + "\n")
        for mt in methods:
            for qid, ms in method_evals[mt]["per_query"].items():
                row = [mt, qid] + [f"{ms[m]:.6f}" if m in ms else ""
                                   for m in all_metrics]
                f.write(",".join(row) + "\n")
    logger.info("CSV per-query escrito: %s", output_path)

def export_json(payload: dict, output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info("JSON escrito: %s", output_path)

def parse_ks(s: str) -> list[int]:
    return sorted(set(int(x.strip()) for x in s.split(",") if x.strip()))

def parse_methods(s: str) -> list[str]:
    valid = {"bm25", "semantic", "hybrid"}
    out = [m.strip() for m in s.split(",")]
    for m in out:
        if m not in valid:
            raise argparse.ArgumentTypeError(
                f"Método desconocido: {m!r}. Válidos: {sorted(valid)}"
            )
    return out

def parse_text_fields(s: str) -> tuple[str, ...]:
    return tuple(f.strip() for f in s.split(",") if f.strip())

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Dataset
    parser.add_argument("--dataset", default="cranfield",
                        help="Identificador de ir_datasets. Default: cranfield. "
                             "Catálogo: https://ir-datasets.com/")
    parser.add_argument("--text-fields", type=parse_text_fields, default=None,
                        help="Campos a concatenar para indexar (coma-sep). "
                             "Ejemplo: title,text. Si se omite se auto-detecta.")
    parser.add_argument("--qrel-binary-threshold", type=int, default=1,
                        help="Umbral de relevancia binaria (default 1: cualquier "
                             "juicio positivo cuenta como relevante).")
    parser.add_argument("--cranfield-original-scale", action="store_true",
                        help="SOLO para 'cranfield': invertir la escala (1↔4, 2↔3) "
                             "para ser fiel al archivo Glasgow original donde 1=mejor. "
                             "Default: dejar los valores como ir_datasets los devuelve "
                             "(la convención dominante en literatura moderna). "
                             "Sólo afecta a nDCG; métricas binarias son invariantes.")
    parser.add_argument("--max-docs", type=int, default=None,
                        help="Limitar cantidad de docs cargados (debugging).")

    # Modelo
    parser.add_argument("--model", type=Path, default=None,
                        help="Directorio del modelo E5 local. Si se omite, "
                             "sólo se evalúa BM25.")
    parser.add_argument("--device", default=None,
                        help="Dispositivo torch: cpu, cuda, cuda:0, mps... "
                             "Default: autodetect.")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size para encoding (default 32).")
    parser.add_argument("--max-seq-length", type=int, default=512)

    # Hiperparámetros del retriever
    parser.add_argument("--bm25-k1", type=float, default=1.5)
    parser.add_argument("--bm25-b",  type=float, default=0.75)
    parser.add_argument("--rrf-k",   type=int,   default=60)

    # Métodos a correr
    parser.add_argument("--methods", type=parse_methods,
                        default=["bm25", "semantic", "hybrid"],
                        help="Lista coma-sep: bm25,semantic,hybrid.")
    parser.add_argument("--ks", type=parse_ks, default=[5, 10, 20, 100, 1000],
                        help="Cutoffs para P@k, R@k, F1@k, nDCG@k.")
    parser.add_argument("--top-k", type=int, default=1000,
                        help="Profundidad del run (default 1000).")

    # Salida
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Directorio de salida. Si se da, se escriben "
                             "TREC runs, CSV de métricas y JSON con todo.")
    parser.add_argument("--inspect-query", default=None,
                        help="Imprimir top-k de una query (qid) marcando "
                             "relevancias. Implica que NO se generan reportes.")
    parser.add_argument("--top-k-show", type=int, default=10,
                        help="Cuántos resultados mostrar en --inspect-query.")

    parser.add_argument("-v", "--verbose", action="count", default=0)

    args = parser.parse_args()

    level = logging.WARNING - (10 * args.verbose)
    logging.basicConfig(
        level=max(logging.DEBUG, level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    print(bold("═" * 76))
    print(bold(f"  Evaluación SRI sobre {args.dataset!r}"))
    print(bold("═" * 76))

    t0 = time.time()
    try:
        coll = load_dataset(
            args.dataset,
            text_fields=args.text_fields,
            binary_threshold=args.qrel_binary_threshold,
            max_docs=args.max_docs,
            cranfield_invert_original_scale=args.cranfield_original_scale,
        )
    except ImportError as exc:
        print(red(f"  ✗ {exc}"))
        return 2
    print(f"  {coll}")
    print(dim(f"  Carga: {time.time() - t0:.2f}s"))

    if not coll.queries:
        print(red("  ✗ El dataset no tiene queries — no se puede evaluar."))
        return 1
    if not coll.qrels:
        print(yellow("  ⚠ El dataset no tiene qrels — se ejecutarán las queries "
                     "pero no se calcularán métricas."))

    methods = list(args.methods)
    if any(m in methods for m in ("semantic", "hybrid")) and args.model is None:
        print(yellow("\n  ⚠ --model no especificado; se eliminan 'semantic'/'hybrid' "
                     "de la lista de métodos. Sólo BM25."))
        methods = [m for m in methods if m == "bm25"] or ["bm25"]

    print()
    print(bold(f"  Construyendo índice…"))
    t0 = time.time()
    retriever = HybridRetriever(
        model_path=args.model,
        k1=args.bm25_k1, b=args.bm25_b,
        rrf_k=args.rrf_k,
        batch_size=args.batch_size,
        device=args.device,
        max_seq_length=args.max_seq_length,
    )

    # Orden de inserción de ir_datasets — estable y reproducible
    doc_ids = list(coll.docs.keys())
    texts = [coll.docs[d].text for d in doc_ids]
    retriever.index(doc_ids, texts, show_progress=True)
    print(dim(f"  Indexación: {time.time() - t0:.2f}s"))

    print()
    print(bold("  Ejecutando consultas…"))
    runs: dict[str, RetrievalRun] = {}
    for method in methods:
        t0 = time.time()
        if method == "bm25":
            r = retriever.search_bm25_batch(coll.queries, top_k=args.top_k)
        elif method == "semantic":
            r = retriever.search_semantic_batch(coll.queries, top_k=args.top_k)
        elif method == "hybrid":
            r = retriever.search_hybrid_batch(coll.queries, top_k=args.top_k,
                                              top_k_per_method=args.top_k)
        else:
            continue
        runs[method] = r
        print(dim(f"  [{method}] {len(r.runs)} queries · "
                  f"{time.time() - t0:.2f}s"))

    if args.inspect_query:
        inspect_query(args.inspect_query, coll, runs,
                      top_k_show=args.top_k_show)
        return 0

    if not coll.qrels:
        print(yellow("\n  No hay qrels; saltando evaluación."))
        return 0

    print()
    print(bold("  Evaluando…"))
    method_evals: dict[str, dict] = {}
    for method, r in runs.items():
        run_doc_lists = {qid: [d for d, _ in items]
                         for qid, items in r.runs.items()}
        ev = evaluate_run(
            run_doc_lists,
            qrels_binary=coll.qrels,
            qrels_graded=coll.qrels_graded,
            ks=args.ks,
        )
        method_evals[method] = ev

    print_metrics_table(method_evals)

    if len(method_evals) >= 2:
        print()
        print(bold("  Tests de significancia estadística"))
        method_names = list(method_evals.keys())
        for i in range(len(method_names)):
            for j in range(i + 1, len(method_names)):
                a, b = method_names[i], method_names[j]
                cmp_metrics = ("ap", "rr", "ndcg@10", "p@10", "r@100")
                cmp = compare_runs(
                    method_evals[a], method_evals[b],
                    name_a=a, name_b=b,
                    metrics_to_test=cmp_metrics,
                )
                print_comparison(cmp)

    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        print()
        print(bold(f"  Exportando artefactos a {args.output}/ ..."))
        for method, r in runs.items():
            export_trec_run(r, args.output / f"run_{method}.trec",
                            run_name=method)
        export_metrics_csv(method_evals, args.output / "metrics_aggregated.csv")
        export_per_query_csv(method_evals, args.output / "metrics_per_query.csv")

        json_payload = {
            "config": {
                "dataset":        args.dataset,
                "text_fields":    list(args.text_fields) if args.text_fields else "auto-detected",
                "model":          str(args.model) if args.model else None,
                "bm25_k1":        args.bm25_k1,
                "bm25_b":         args.bm25_b,
                "rrf_k":          args.rrf_k,
                "binary_threshold": args.qrel_binary_threshold,
                "methods":        methods,
                "ks":             args.ks,
                "top_k":          args.top_k,
            },
            "collection_stats": {
                "name":        coll.name,
                "num_docs":    coll.num_docs,
                "num_queries": coll.num_queries,
                "num_qrels":   coll.num_qrels,
            },
            "aggregated_metrics": {m: ev["aggregated"]
                                   for m, ev in method_evals.items()},
        }
        comparisons = []
        method_names = list(method_evals.keys())
        for i in range(len(method_names)):
            for j in range(i + 1, len(method_names)):
                a, b = method_names[i], method_names[j]
                cmp = compare_runs(
                    method_evals[a], method_evals[b],
                    name_a=a, name_b=b,
                )
                comparisons.append(cmp)
        json_payload["comparisons"] = comparisons
        export_json(json_payload, args.output / "results.json")

        per_query_payload = {
            m: ev["per_query"] for m, ev in method_evals.items()
        }
        export_json(per_query_payload, args.output / "per_query.json")

        print(green(f"  ✓ Listo. {len(list(args.output.iterdir()))} archivos en {args.output}/"))

    return 0

if __name__ == "__main__":
    sys.exit(main())
