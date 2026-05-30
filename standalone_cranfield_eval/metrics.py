"""
metrics.py — Métricas estándar de IR + tests de significancia estadística.

Métricas binarias (sobre qrels binarios — relevante o no):
  • Precision@k     P@k = |relevantes en top k| / k
  • Recall@k        R@k = |relevantes en top k| / |total relevantes|
  • F1@k            media armónica de P@k y R@k
  • Average Precision  AP = (1/R) * sum_{k:rel@k=1} P@k
                       MAP = media de AP sobre queries
  • Reciprocal Rank RR = 1 / rank(primer relevante); MRR = media
  • R-Precision     P@R donde R = |relevantes de esa query|

Métricas graded (sobre qrels con relevancia ordinal, p.e. Cranfield 1..4):
  • DCG@k    sum_{i=1..k} (2^rel_i - 1) / log2(i + 1)         (gain exponencial)
  • nDCG@k   DCG@k / IDCG@k

Significancia estadística (compare_runs):
  • Shapiro-Wilk de las diferencias pareadas (normalidad).
  • Paired t-test (paramétrico, si las diferencias son normales).
  • Wilcoxon signed-rank (no paramétrico, baseline tradicional).
  • Test de permutación pareado (gold standard en IR — Smucker et al. 2007).
  • Cohen's d_z (tamaño del efecto pareado).
  • Bootstrap percentile CI del delta.
  • Corrección Holm-Bonferroni para múltiples métricas.
  • Selección automática del test recomendado según n + normalidad +
    tipo de métrica.

Referencias:
  Manning, C. D., Raghavan, P. & Schütze, H. (2008). "Introduction to
    Information Retrieval", Cambridge UP, capítulo 8.
  Järvelin, K. & Kekäläinen, J. (2002). "Cumulated gain-based evaluation
    of IR techniques". ACM TOIS 20(4).
  Sakai, T. (2014). "Statistical Reform in Information Retrieval?".
    SIGIR Forum 48(1).
  Smucker, M. D., Allan, J. & Carterette, B. (2007). "A Comparison of
    Statistical Significance Tests for Information Retrieval Evaluation".
    CIKM 2007.
  Cohen, J. (1988). "Statistical Power Analysis for the Behavioral
    Sciences", 2nd ed.
"""

from __future__ import annotations

import math
from statistics import mean
from typing import Iterable, Literal, Optional

import numpy as np

# scipy es necesaria para Shapiro-Wilk y para los p-values exactos del
# t-test y Wilcoxon. Está en requirements.txt; si no se instala, los
# tests devolverán p_value=None con un mensaje claro.
try:
    from scipy import stats as _scipy_stats  # type: ignore
    _HAS_SCIPY = True
except ImportError:
    _scipy_stats = None
    _HAS_SCIPY = False

def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    topk = retrieved[:k]
    if not topk:
        return 0.0
    hits = sum(1 for d in topk if d in relevant)
    return hits / k

def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for d in retrieved[:k] if d in relevant)
    return hits / len(relevant)

def f1_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    p = precision_at_k(retrieved, relevant, k)
    r = recall_at_k(retrieved, relevant, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)

def average_precision(retrieved: list[str], relevant: set[str]) -> float:
    """AP = (1/R) * sum_{i=1..N} [rel(i) * P@i]"""
    if not relevant:
        return 0.0
    hits = 0
    sum_precisions = 0.0
    for i, d in enumerate(retrieved, start=1):
        if d in relevant:
            hits += 1
            sum_precisions += hits / i
    return sum_precisions / len(relevant)

def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for i, d in enumerate(retrieved, start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0

def r_precision(retrieved: list[str], relevant: set[str]) -> float:
    R = len(relevant)
    if R == 0:
        return 0.0
    return precision_at_k(retrieved, relevant, R)

def _dcg(gains: list[float], k: int,
         gain_type: Literal["exp", "linear"] = "exp") -> float:
    total = 0.0
    for i, g in enumerate(gains[:k], start=1):
        num = (2 ** g) - 1 if gain_type == "exp" else g
        total += num / math.log2(i + 1)
    return total

def ndcg_at_k(retrieved: list[str],
              gains_by_doc: dict[str, int],
              k: int,
              gain_type: Literal["exp", "linear"] = "exp") -> float:
    if not gains_by_doc:
        return 0.0
    gains_ret = [gains_by_doc.get(d, 0) for d in retrieved[:k]]
    ideal = sorted(gains_by_doc.values(), reverse=True)[:k]
    if not any(ideal):
        return 0.0
    dcg = _dcg(gains_ret, k, gain_type)
    idcg = _dcg(ideal, k, gain_type)
    return dcg / idcg if idcg > 0 else 0.0

def evaluate_run(run: dict[str, list[str]],
                 qrels_binary: dict[str, dict[str, int]],
                 qrels_graded: dict[str, dict[str, int]] | None = None,
                 ks: Iterable[int] = (5, 10, 20, 100, 1000),
                 skip_queries_without_relevant: bool = True
                 ) -> dict:
    """
    Evalúa un run sobre un set de qrels. Devuelve un dict con métricas
    agregadas y per-query.
    """
    ks = list(ks)
    per_query: dict[str, dict[str, float]] = {}
    n_skipped = 0

    for qid, retrieved in run.items():
        rel_set = set(qrels_binary.get(qid, {}).keys())
        if not rel_set and skip_queries_without_relevant:
            n_skipped += 1
            continue

        q_metrics: dict[str, float] = {
            "ap":      average_precision(retrieved, rel_set),
            "rr":      reciprocal_rank(retrieved, rel_set),
            "r_prec":  r_precision(retrieved, rel_set),
            "num_rel": float(len(rel_set)),
        }
        for k in ks:
            q_metrics[f"p@{k}"]  = precision_at_k(retrieved, rel_set, k)
            q_metrics[f"r@{k}"]  = recall_at_k(retrieved, rel_set, k)
            q_metrics[f"f1@{k}"] = f1_at_k(retrieved, rel_set, k)

        if qrels_graded is not None:
            gains = qrels_graded.get(qid, {})
            for k in ks:
                q_metrics[f"ndcg@{k}"] = ndcg_at_k(retrieved, gains, k,
                                                   gain_type="exp")

        per_query[qid] = q_metrics

    aggregated: dict[str, float] = {}
    if per_query:
        for key in ("ap", "rr", "r_prec"):
            aggregated[key] = mean(q[key] for q in per_query.values())
        aggregated["map"] = aggregated.pop("ap")
        aggregated["mrr"] = aggregated.pop("rr")
        for k in ks:
            for prefix in ("p@", "r@", "f1@"):
                key = f"{prefix}{k}"
                aggregated[key] = mean(q[key] for q in per_query.values())
        if qrels_graded is not None:
            for k in ks:
                key = f"ndcg@{k}"
                aggregated[key] = mean(q[key] for q in per_query.values())

    return {
        "aggregated": aggregated,
        "per_query":  per_query,
        "n_queries_evaluated": len(per_query),
        "n_queries_skipped":   n_skipped,
    }

def shapiro_wilk_normality(diffs: list[float]) -> dict:
    """
    Test de Shapiro-Wilk sobre las diferencias pareadas.

    H0: las diferencias provienen de una distribución normal.
    p < 0.05 → se rechaza H0 → NO usar t-test, preferir no paramétrico.

    Válido para 3 ≤ n ≤ 5000. Si n > 5000, Shapiro es hipersensible
    (rechaza H0 ante desviaciones triviales); ese caso lo marcamos
    pero no abortamos.
    """
    n = len(diffs)
    if not _HAS_SCIPY:
        return {"statistic": None, "p_value": None, "is_normal": None,
                "error": "scipy no instalado"}
    if n < 3:
        return {"statistic": None, "p_value": None, "is_normal": None,
                "error": f"Shapiro requiere n>=3, recibido n={n}"}
    if len(set(diffs)) == 1:
        # Todas las diferencias iguales (típicamente todas 0):
        # la distribución es degenerada, no tiene sentido testear.
        return {"statistic": None, "p_value": None, "is_normal": False,
                "error": "Todas las diferencias son idénticas"}
    res = _scipy_stats.shapiro(diffs)
    out = {"statistic": float(res.statistic),
           "p_value":   float(res.pvalue),
           "is_normal": float(res.pvalue) > 0.05}
    if n > 5000:
        out["warning"] = "n>5000: Shapiro hipersensible, usar con cautela"
    return out

def paired_t_test(a: list[float], b: list[float]) -> dict:
    """
    Paired Student's t-test de dos colas. H0: media(a - b) = 0.
    Asume que las diferencias son aprox. normales — verificar con
    shapiro_wilk_normality antes de confiar en este p-value.
    """
    if len(a) != len(b):
        raise ValueError(f"Long. distintas: {len(a)} vs {len(b)}")
    n = len(a)
    if n < 2:
        return {"t": 0.0, "p_value": None, "mean_diff": 0.0, "n": n}

    diffs = [x - y for x, y in zip(a, b)]
    mean_d = sum(diffs) / n
    var_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)
    se = math.sqrt(var_d / n) if var_d > 0 else 0.0
    t = mean_d / se if se > 0 else 0.0

    if _HAS_SCIPY:
        p = float(_scipy_stats.t.sf(abs(t), df=n - 1) * 2)
    else:
        p = None
    return {"t": t, "p_value": p, "mean_diff": mean_d, "n": n}

def wilcoxon_signed_rank(a: list[float], b: list[float]) -> dict:
    """
    Wilcoxon signed-rank test. No paramétrico clásico. Atención: con
    métricas discretas con muchos empates (P@5, P@10) puede perder
    potencia. En ese caso preferir permutation_test_paired.
    """
    if len(a) != len(b):
        raise ValueError(f"Long. distintas: {len(a)} vs {len(b)}")
    if not _HAS_SCIPY:
        return {"statistic": None, "p_value": None, "n_nonzero": None,
                "error": "scipy no instalado — pip install scipy"}
    diffs = [x - y for x, y in zip(a, b) if x != y]
    if len(diffs) < 2:
        return {"statistic": 0.0, "p_value": 1.0, "n_nonzero": len(diffs)}
    res = _scipy_stats.wilcoxon(diffs)
    return {
        "statistic": float(res.statistic),
        "p_value":   float(res.pvalue),
        "n_nonzero": len(diffs),
    }

def permutation_test_paired(a: list[float], b: list[float],
                             n_resamples: int = 10000,
                             seed: int | None = 42) -> dict:
    """
    Test de permutación pareado (randomization test) sobre la diferencia
    de medias. Para cada par (a_i, b_i), decide aleatoriamente si los
    intercambia. Bajo H0 (no hay diferencia sistemática), intercambiar
    es equivalente a no hacerlo, así que la distribución de la media
    de las diferencias permutadas es la distribución bajo H0.

    p_value = proporción de permutaciones cuya |diff| es >= |observada|.
    Se suma +1 al numerador y denominador (Phipson & Smyth 2010) para
    evitar p=0 con n_resamples finito.

    Recomendado en IR sobre Wilcoxon cuando hay muchos empates
    (métricas discretas como P@k pequeño). Ver Smucker et al. (2007).
    """
    if len(a) != len(b):
        raise ValueError(f"Long. distintas: {len(a)} vs {len(b)}")
    n = len(a)
    if n < 2:
        return {"observed_mean_diff": 0.0, "p_value": None,
                "n_resamples": 0, "n": n}

    rng = np.random.default_rng(seed)
    diffs = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    observed = float(diffs.mean())
    abs_obs = abs(observed)

    # Para cada remuestreo, cada diff se mantiene (+1) o se invierte (-1).
    # Vectorizado: matriz (n_resamples, n) de signos.
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_resamples, n))
    permuted_means = (signs * diffs).mean(axis=1)
    count = int(np.sum(np.abs(permuted_means) >= abs_obs))
    p_value = (count + 1) / (n_resamples + 1)

    return {"observed_mean_diff": observed,
            "p_value":            p_value,
            "n_resamples":        n_resamples,
            "n":                  n}

def cohens_d_paired(a: list[float], b: list[float]) -> dict:
    """
    Tamaño del efecto para muestras pareadas (Cohen's d_z).

        d_z = mean(diff) / sd(diff)

    Interpretación clásica (Cohen 1988):
      |d| < 0.2  → trivial
      0.2 ≤ |d| < 0.5  → pequeño
      0.5 ≤ |d| < 0.8  → mediano
      0.8 ≤ |d|        → grande

    En IR las mejoras suelen ser "trivial" o "pequeño" aún cuando son
    estadísticamente significativas. Reportar d JUNTO al p-value es la
    mejor práctica (Sakai 2014).
    """
    if len(a) != len(b):
        raise ValueError(f"Long. distintas: {len(a)} vs {len(b)}")
    n = len(a)
    if n < 2:
        return {"d": 0.0, "magnitude": "n_insuficiente"}

    diffs = [x - y for x, y in zip(a, b)]
    mean_d = sum(diffs) / n
    var_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)
    sd = math.sqrt(var_d) if var_d > 0 else 0.0
    if sd == 0:
        return {"d": 0.0, "magnitude": "sin_variabilidad"}

    d = mean_d / sd
    abs_d = abs(d)
    if abs_d < 0.2:
        mag = "trivial"
    elif abs_d < 0.5:
        mag = "pequeño"
    elif abs_d < 0.8:
        mag = "mediano"
    else:
        mag = "grande"
    return {"d": d, "magnitude": mag}

def bootstrap_ci_paired(a: list[float], b: list[float],
                         confidence: float = 0.95,
                         n_resamples: int = 10000,
                         seed: int | None = 42) -> dict:
    """
    Intervalo de confianza bootstrap (percentile) para mean(a - b).

    Más informativo que un p-value solo: muestra magnitud y precisión.
    Si el CI cruza 0, la diferencia no es significativa al nivel
    correspondiente (≈ test bilateral).
    """
    if len(a) != len(b):
        raise ValueError(f"Long. distintas: {len(a)} vs {len(b)}")
    n = len(a)
    if n < 2:
        return {"ci_low": None, "ci_high": None,
                "confidence": confidence, "n_resamples": 0}

    rng = np.random.default_rng(seed)
    diffs = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    # Muestras con reposición: matriz (n_resamples, n) de índices.
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot_means = diffs[idx].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low  = float(np.quantile(boot_means, alpha))
    high = float(np.quantile(boot_means, 1.0 - alpha))
    return {"ci_low":      low,
            "ci_high":     high,
            "confidence":  confidence,
            "n_resamples": n_resamples}

def _select_recommended_test(n: int,
                              normality_p: Optional[float],
                              metric_name: str) -> tuple[str, str]:
    """
    Decide qué test usar como referencia y explica por qué.

    Reglas:
      - Métricas P@k con k pequeño (≤10) tienen muchos empates →
        permutación (Wilcoxon es subóptimo con empates).
      - Si n ≥ 30 y Shapiro NO rechaza normalidad → t-test (más
        potente cuando el supuesto se cumple).
      - Si n ≥ 20 → permutación (gold standard en IR, robusto).
      - n < 20 → Wilcoxon (más estable con muestras chicas, simple).

    Retorna (nombre_test, razón).
    """
    if any(metric_name.startswith(p) for p in ("p@5", "p@10")):
        return ("permutation",
                f"métrica discreta con muchos empates posibles")
    if n >= 30 and normality_p is not None and normality_p > 0.05:
        return ("t_test",
                f"n={n}≥30 y Shapiro p={normality_p:.3f}>0.05 (normalidad ok)")
    if n >= 20:
        if normality_p is not None and normality_p <= 0.05:
            return ("permutation",
                    f"Shapiro p={normality_p:.3f}≤0.05 (no normal); "
                    f"permutación es robusta y precisa en IR")
        return ("permutation",
                f"n={n}≥20: permutación es el test recomendado en IR "
                f"(Smucker et al. 2007)")
    return ("wilcoxon",
            f"n={n}<20: muestra pequeña, Wilcoxon es la opción estándar")

def _holm_bonferroni(pvalues: list[tuple[str, float]]
                      ) -> dict[str, float]:
    """
    Holm-Bonferroni paso a paso. Más potente que Bonferroni puro
    (rechaza al menos los mismos H0 al mismo α).

    Algoritmo:
      1. Ordenar p_(1) ≤ p_(2) ≤ ... ≤ p_(k).
      2. Ajustar: p_adj_(i) = max((k - i + 1) * p_(i), p_adj_(i-1)).
      3. Cap en 1.0.
    """
    sorted_p = sorted(pvalues, key=lambda x: x[1])
    k = len(sorted_p)
    adjusted: dict[str, float] = {}
    prev = 0.0
    for i, (name, p) in enumerate(sorted_p):
        adj = min(p * (k - i), 1.0)
        adj = max(adj, prev)   # monotonía
        adjusted[name] = adj
        prev = adj
    return adjusted

def _bonferroni(pvalues: list[tuple[str, float]]) -> dict[str, float]:
    k = len(pvalues)
    return {name: min(p * k, 1.0) for name, p in pvalues}

def compare_runs(eval_a: dict, eval_b: dict,
                 name_a: str = "A", name_b: str = "B",
                 metrics_to_test: tuple[str, ...] = ("ap", "rr", "ndcg@10",
                                                      "p@10", "r@100"),
                 *,
                 alpha: float = 0.05,
                 multiple_comparison: Literal["none", "bonferroni", "holm"] = "holm",
                 n_resamples: int = 10000,
                 bootstrap_confidence: float = 0.95,
                 seed: int | None = 42,
                 ) -> dict:
    """
    Compara dos runs evaluados con evaluate_run sobre el set de queries
    en común.

    Para cada métrica reporta:
      - Medias, delta absoluto, delta relativo.
      - Shapiro-Wilk de las diferencias.
      - Los tres tests: t-test, Wilcoxon, permutación.
      - Test recomendado + razón.
      - Cohen's d_z (tamaño del efecto).
      - Bootstrap percentile CI del delta.
      - p-value ajustado por comparaciones múltiples (Holm por defecto).
      - Marcador `significant_after_correction` (bool).

    Args:
        alpha: nivel de significancia para `significant_after_correction`.
        multiple_comparison: "holm" (default), "bonferroni" o "none".
        n_resamples: nº de remuestreos para permutación y bootstrap.
        bootstrap_confidence: nivel del CI bootstrap (default 0.95).
        seed: semilla para reproducibilidad de permutación y bootstrap.

    Backward compat: el dict resultante mantiene todas las claves del
    código viejo (`name_a`, `name_b`, `n_common_queries`, `tests`,
    `tests[m].t_test`, `tests[m].wilcoxon`, etc.) — sólo agrega
    información nueva.
    """
    pq_a = eval_a["per_query"]
    pq_b = eval_b["per_query"]
    common = sorted(set(pq_a) & set(pq_b))
    if not common:
        return {"error": "Sin queries en común"}

    out: dict = {
        "name_a": name_a, "name_b": name_b,
        "n_common_queries":     len(common),
        "alpha":                alpha,
        "multiple_comparison":  multiple_comparison,
        "n_resamples":          n_resamples,
        "seed":                 seed,
        "tests": {},
    }

    # Recolectar p-values del test recomendado para corrección posterior.
    recommended_pvalues: list[tuple[str, float]] = []

    for m in metrics_to_test:
        a_vals = [pq_a[q].get(m, 0.0) for q in common]
        b_vals = [pq_b[q].get(m, 0.0) for q in common]
        diffs  = [x - y for x, y in zip(a_vals, b_vals)]
        mean_a = sum(a_vals) / len(a_vals)
        mean_b = sum(b_vals) / len(b_vals)
        delta  = mean_a - mean_b

        normality = shapiro_wilk_normality(diffs)
        rec_name, rec_reason = _select_recommended_test(
            len(common), normality.get("p_value"), m
        )

        t_res = paired_t_test(a_vals, b_vals)
        w_res = wilcoxon_signed_rank(a_vals, b_vals)
        p_res = permutation_test_paired(a_vals, b_vals,
                                         n_resamples=n_resamples, seed=seed)

        # P-value del test recomendado (para corrección + flag).
        if rec_name == "t_test":
            rec_p = t_res.get("p_value")
        elif rec_name == "wilcoxon":
            rec_p = w_res.get("p_value")
        else:
            rec_p = p_res.get("p_value")

        if rec_p is not None:
            recommended_pvalues.append((m, rec_p))

        out["tests"][m] = {
            "mean_a":              mean_a,
            "mean_b":              mean_b,
            "delta":               delta,
            "rel_delta":           delta / mean_b if mean_b > 0 else None,
            "normality":           normality,
            "recommended_test":    rec_name,
            "recommended_reason":  rec_reason,
            "recommended_p_value": rec_p,
            "t_test":              t_res,
            "wilcoxon":            w_res,
            "permutation":         p_res,
            "cohens_d":            cohens_d_paired(a_vals, b_vals),
            "bootstrap_ci":        bootstrap_ci_paired(
                a_vals, b_vals,
                confidence=bootstrap_confidence,
                n_resamples=n_resamples,
                seed=seed,
            ),
            # Se completa abajo, una vez aplicada la corrección.
            "adjusted_p_value":            None,
            "significant_after_correction": None,
        }

    # ── Corrección por comparaciones múltiples ────────────────────
    if multiple_comparison != "none" and recommended_pvalues:
        if multiple_comparison == "bonferroni":
            adjusted = _bonferroni(recommended_pvalues)
        else:  # "holm"
            adjusted = _holm_bonferroni(recommended_pvalues)
        for m, adj_p in adjusted.items():
            out["tests"][m]["adjusted_p_value"] = adj_p
            out["tests"][m]["significant_after_correction"] = adj_p < alpha
    else:
        # Sin corrección: marcamos significancia con el p crudo.
        for m, p in recommended_pvalues:
            out["tests"][m]["adjusted_p_value"] = p
            out["tests"][m]["significant_after_correction"] = p < alpha

    return out

# ── Utilidad: formato bonito de los resultados (para CLI/tesis) ──────

def format_comparison_table(comparison: dict, *, latex: bool = False) -> str:
    """
    Convierte el dict de compare_runs en una tabla legible. Útil tanto
    para imprimir en consola como para pegar en el manuscrito.

    Columnas: métrica, mean_A, mean_B, Δ, Δ%, Cohen's d, CI 95%,
    test recomendado, p crudo, p ajustado, significancia.
    Significancia: '' si p≥0.05, '*' si <0.05, '**' si <0.01, '***' <0.001.
    """
    name_a = comparison.get("name_a", "A")
    name_b = comparison.get("name_b", "B")
    mc     = comparison.get("multiple_comparison", "none")
    n      = comparison.get("n_common_queries", 0)

    sep = " & " if latex else "  "
    eol = " \\\\\n" if latex else "\n"

    lines: list[str] = []
    header = sep.join(["Métrica",
                       f"{name_a}",
                       f"{name_b}",
                       "Δ", "Δ%", "d", "CI 95%",
                       "Test", "p", f"p_adj ({mc})", "sig"])
    lines.append(header + eol.rstrip())

    def stars(p: Optional[float]) -> str:
        if p is None:
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    for m, t in comparison.get("tests", {}).items():
        rel = t.get("rel_delta")
        rel_str = f"{rel*100:+.2f}%" if rel is not None else "—"
        ci = t.get("bootstrap_ci") or {}
        ci_lo, ci_hi = ci.get("ci_low"), ci.get("ci_high")
        ci_str = (f"[{ci_lo:+.3f}, {ci_hi:+.3f}]"
                  if ci_lo is not None else "—")
        rec_p     = t.get("recommended_p_value")
        adj_p     = t.get("adjusted_p_value")
        d         = (t.get("cohens_d") or {}).get("d")
        sig_after = t.get("significant_after_correction")

        row = sep.join([
            m,
            f"{t['mean_a']:.4f}",
            f"{t['mean_b']:.4f}",
            f"{t['delta']:+.4f}",
            rel_str,
            f"{d:+.2f}" if d is not None else "—",
            ci_str,
            t.get("recommended_test", "?"),
            f"{rec_p:.4f}" if rec_p is not None else "—",
            f"{adj_p:.4f}{stars(adj_p)}" if adj_p is not None else "—",
            "✓" if sig_after else "",
        ])
        lines.append(row + eol.rstrip())

    footer = f"\nn queries comparadas: {n}. Sig: {mc}, α=0.05."
    return "\n".join(lines) + footer
