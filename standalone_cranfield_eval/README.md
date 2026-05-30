# Evaluación standalone del SRI sobre colecciones IR

Herramienta autocontenida para medir la calidad del sistema de
recuperación de información (BM25 + E5 + RRF) sobre **Cranfield 1400**
y, de paso, cualquier otra colección del catálogo de
[`ir_datasets`](https://ir-datasets.com/) (BEIR, MS MARCO, TREC,
NFCorpus, Antique, Vaswani, etc.).

**No depende del proyecto OCR**. Replica la arquitectura del SRI en
producción pero con su propia implementación.

---

## 1. Estructura

```
standalone_cranfield_eval/
├── sri.py             # HybridRetriever (BM25 + E5 + RRF)
├── dataset.py         # Carga via ir_datasets (cranfield, beir, msmarco, ...)
├── metrics.py         # P@k, R@k, F1@k, MAP, MRR, R-Prec, nDCG@k, tests
├── evaluate.py        # CLI principal
├── requirements.txt
└── README.md
```

Las cuatro piezas son independientes — `metrics.py` se puede usar
contra cualquier run en formato `{qid -> [doc_id, ...]}` y `dataset.py`
contra cualquier dataset que esté en `ir_datasets`.

---

## 2. Instalación

```bash
cd standalone_cranfield_eval
pip install -r requirements.txt
```

Las cinco deps son:
- `numpy`, `rank_bm25` — el mínimo absoluto para BM25
- `ir_datasets` — para descargar/parsear las colecciones
- `scipy` — para Wilcoxon signed-rank y t-test exacto
- `sentence-transformers` — para E5 (sólo si vas a usar `--methods semantic` o `hybrid`)

Si vas a evaluar **sólo BM25**, podés saltarte `sentence-transformers`
(unos 2 GB de descarga entre torch y deps). Si vas a evaluar **sólo
sobre Cranfield offline** y ya tenés los archivos localmente, podés
saltarte la red por completo (ver §3).

---

## 3. Cargar Cranfield: tres opciones

`ir_datasets` cachea los datasets en `~/.ir_datasets/` por defecto
(overridable con la env var `IR_DATASETS_HOME`).

### Opción A — descarga automática (lo más simple)

```bash
python evaluate.py --dataset cranfield --methods bm25
```

La primera corrida descarga `cran.tar.gz` (~22 KB) desde el mirror
oficial de Glasgow. A partir de ahí, está cacheado para siempre.

### Opción B — ya tenés Cranfield localmente

Si ya tenés los tres archivos sueltos (`cran.all.1400`, `cran.qry`,
`cranqrel`) en `/ruta/a/cranfield/`, los empaquetás una vez en el lugar
donde `ir_datasets` los espera:

```bash
mkdir -p ~/.ir_datasets/cranfield
cd /ruta/a/cranfield
tar czf ~/.ir_datasets/cranfield/cran.tar.gz \
    cran.all.1400 cran.qry cranqrel
```

A partir de ahí, ningún `evaluate.py` vuelve a tocar la red.

### Opción C — cache portable

Si vas a llevar el proyecto a otra máquina sin red, copiá la carpeta
`~/.ir_datasets/` entera. O apuntá `IR_DATASETS_HOME` a un directorio
versionado con git:

```bash
export IR_DATASETS_HOME=/proyecto/datasets_cache
```

---

## 4. Modelo E5

Para `--methods semantic` o `hybrid` hace falta un directorio con el
modelo (típicamente `multilingual-e5-small`). Estructura esperada:

```
e5/
├── config.json
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
├── sentence_bert_config.json
├── modules.json
├── 1_Pooling/
└── model.safetensors      (o pytorch_model.bin)
```

Y pasalo con `--model /ruta/a/e5`. Si no lo pasás, el script se queda
en modo sólo-BM25 sin levantar errores.

---

## 5. Uso

### 5.1. Evaluación completa sobre Cranfield (recomendado para tesis)

```bash
python evaluate.py \
    --dataset cranfield \
    --model   /ruta/a/e5 \
    --output  resultados_cranfield/
```

1. Carga la colección vía `ir_datasets`.
2. Indexa los 1.400 documentos con BM25 y E5 (CPU: 1-3 min; GPU: ~10 s).
3. Corre las 225 consultas con tres métodos: BM25, E5 puro, e Híbrido (RRF).
4. Reporta métricas agregadas, comparaciones pareadas y tests de significancia.
5. Vuelca en `resultados_cranfield/`:
   - `run_{bm25,semantic,hybrid}.trec` — runs en formato TREC, compatibles
     con `trec_eval` de NIST para validación cruzada.
   - `metrics_aggregated.csv` — una fila por método, todas las métricas.
   - `metrics_per_query.csv` — métricas por (método, query).
   - `results.json` — config + métricas + comparaciones.
   - `per_query.json` — todas las métricas por query.

### 5.2. Otras colecciones (bonus para la tesis)

El mismo `evaluate.py` corre sobre cualquier dataset del catálogo:

```bash
# BEIR/scifact: corpus científico, 5K docs, 300 queries de test
python evaluate.py --dataset beir/scifact/test --model /ruta/a/e5

# NFCorpus: queries biomédicas de NutritionFacts, 3.6K docs
python evaluate.py --dataset nfcorpus/test --model /ruta/a/e5

# Vaswani: 11K abstracts de IR/CS, 93 queries (más viejo)
python evaluate.py --dataset vaswani

# Antique: preguntas tipo "Yahoo Answers", 400K docs
python evaluate.py --dataset antique/test --model /ruta/a/e5
```

Los campos de texto se detectan automáticamente (`title + text`,
`title + abstract`, etc.). Si necesitás un set específico:

```bash
python evaluate.py --dataset nfcorpus/test --text-fields title,abstract
```

### 5.3. Sólo BM25 (rápido, sin modelo)

```bash
python evaluate.py --methods bm25
```

### 5.4. Inspeccionar una consulta

```bash
python evaluate.py --dataset cranfield --model /ruta/a/e5 \
                   --inspect-query 1 --top-k-show 20
```

Muestra el texto de la query, los 20 primeros resultados de cada
método, marcando con `✓` los relevantes y mostrando el grado.

### 5.5. Tuning de hiperparámetros (para sección de ablación)

```bash
# Probar valores no-default de BM25
python evaluate.py --bm25-k1 1.2 --bm25-b 0.5

# Otro k de RRF
python evaluate.py --model /ruta/a/e5 --rrf-k 30 --methods hybrid

# Threshold de relevancia más estricto (sólo grados 2 o más)
python evaluate.py --qrel-binary-threshold 2
```

---

## 6. ⚠ La escala de relevancia de Cranfield: nota crítica para tesis

Este punto **importa** si vas a reportar nDCG y compararlo con
literatura histórica:

- El archivo original `cranqrel` de Cleverdon (Glasgow, 1968) usa
  la convención **1 = "respuesta completa" (más relevante), 4 =
  "interés mínimo" (menos relevante), -1 = no relevante**.
- `ir_datasets` lee el archivo *literalmente* (sin invertir),
  pero documenta los labels con la convención **opuesta** (1 = mínimo,
  4 = respuesta completa). Esa interpretación invertida es la
  dominante en literatura post-2020 que usa ir_datasets, PyTerrier, etc.

Por **defecto este loader sigue la convención de ir_datasets** (no
invierte): los valores se pasan tal cual y la "ganancia" para nDCG es
mayor para los rel=4 que para los rel=1.

Si querés ser fiel al paper original de Cleverdon, pasá
`--cranfield-original-scale`. La transformación interna es 1↔4, 2↔3.

**Esto SOLO afecta a nDCG**. Las métricas binarias (MAP, MRR, P@k,
R@k, R-Prec) son invariantes porque {1, 2, 3, 4} todos cuentan como
"relevantes" con `--qrel-binary-threshold 1`. En la tesis, lo más
correcto es:

1. Mencionar explícitamente cuál convención usás (cita esto).
2. Reportar las binarias y nDCG por separado.
3. Si comparás contra papers concretos, comprobar qué convención
   usaron ellos.

---

## 7. Métricas implementadas

| Métrica   | Fórmula                                                      | Sirve para                                |
|-----------|--------------------------------------------------------------|-------------------------------------------|
| **P@k**   | \|relevantes en top *k*\| / *k*                              | Calidad en la parte alta del ranking      |
| **R@k**   | \|relevantes en top *k*\| / \|total relevantes\|             | Cobertura                                 |
| **F1@k**  | Media armónica de P@k y R@k                                  | Equilibrio entre las dos                  |
| **MAP**   | Media de AP sobre queries; AP = (1/R) · Σ\_{rel@k} P@k       | Métrica binaria de referencia en IR       |
| **MRR**   | Media de 1/rank(primer relevante)                            | Cuán arriba aparece el primer hit         |
| **R-Prec**| P@R donde R = \|relevantes de esa query\|                    | Comparable entre queries con R distintos  |
| **nDCG@k**| DCG@k / IDCG@k, gain exponencial (2^rel - 1) / log₂(i+1)     | Métrica graded — usa la escala 1..4       |

Para tests pareados (bm25 vs híbrido, etc.):

- **Paired t-test** (asume normalidad de diferencias).
- **Wilcoxon signed-rank** (no paramétrico; *el recomendado en IR*,
  ver Sakai 2014, "Statistical Reform in Information Retrieval?").

Significancia marcada con `*` (p < 0.05) y `**` (p < 0.01).

---

## 8. Qué reportar en la tesis

### 8.1. Tabla principal de métricas

Una tabla con métodos en filas y métricas en columnas:

| Método | MAP | MRR | P@5 | P@10 | R@100 | nDCG@10 | R-Prec |
|--------|-----|-----|-----|------|-------|---------|--------|
| BM25 (baseline) | … | … | … | … | … | … | … |
| E5 (denso)      | … | … | … | … | … | … | … |
| Híbrido RRF     | … | … | … | … | … | … | … |

`metrics_aggregated.csv` → `pandas.DataFrame.to_latex()`.

### 8.2. Tabla de significancia

| Comparación | ΔMAP | ΔnDCG@10 | p (Wilcoxon) |
|---|---|---|---|
| Híbrido vs BM25 | +0.05 | +0.06 | 0.003 ** |
| Híbrido vs E5  | +0.02 | +0.03 | 0.041 *  |

Estos números salen directo del bloque "Tests de significancia" o de
`results.json → comparisons`.

### 8.3. Análisis cualitativo

Para 2-3 consultas representativas, mostrá el top-k con
`--inspect-query` y discutí: ¿qué falla BM25 que el híbrido arregla?
¿hay queries donde E5 ayuda más?

### 8.4. Ablación de hiperparámetros

Variar `--bm25-k1`, `--bm25-b`, `--rrf-k` y reportar cómo cambia MAP.

### 8.5. Limitaciones a mencionar honestamente

- **Cranfield es chico** (1.400 docs) — techo MAP ~0.40-0.50 con BM25
  vainilla. Diferencias pequeñas pueden no ser significativas.
- **Sin stemming.** Replica producción. Ablación con Porter sube ~0.02-0.05.
- **Sin expansión de consulta** (PRF/RM3). Mejora estándar no implementada.
- **Cranfield está en inglés**, el SRI en producción atiende español.
  `multilingual-e5-small` es multilingüe pero Cranfield no mide eso.
- **Escala de qrels** (ver §6 arriba).

---

## 9. Validación cruzada con `trec_eval`

Los `run_*.trec` son compatibles con la herramienta de NIST. Tener los
mismos números reportados por las dos implementaciones aumenta credibilidad:

```bash
# Exportar qrels en formato TREC desde ir_datasets:
python -c "
import ir_datasets
ds = ir_datasets.load('cranfield')
with open('qrels.trec', 'w') as f:
    for q in ds.qrels_iter():
        f.write(f'{q.query_id} 0 {q.doc_id} {q.relevance}\n')
"

# Correr trec_eval:
trec_eval -m map -m P.5,10,20 -m ndcg_cut.10 qrels.trec run_bm25.trec
```

Diferencias menores en nDCG son normales según la convención de gain
(usamos exponencial igual que `trec_eval`) y manejo de queries sin
relevantes.

---

## 10. Reproducibilidad

Todos los componentes son deterministas. Para reproducción exacta,
fijar:

- Hash del modelo E5 (e.g. `sha256sum model.safetensors`).
- Versiones de `rank_bm25`, `sentence-transformers`, `torch`, `numpy`,
  `ir_datasets`.
- Dispositivo (`--device cpu` si te importa la última cifra; CUDA vs
  CPU pueden divergir).

Todo eso se guarda en `results.json → config` automáticamente.

---

## 11. Citas mínimas para la sección de metodología

```
Robertson, S. E., & Walker, S. (1994). Some simple effective approximations
to the 2-Poisson model for probabilistic weighted retrieval. SIGIR '94.

Cormack, G. V., Clarke, C. L. A., & Büttcher, S. (2009). Reciprocal rank
fusion outperforms condorcet and individual rank learning methods. SIGIR '09.

Wang, L., Yang, N., Huang, X., Yang, L., Majumder, R., & Wei, F. (2024).
Multilingual E5 Text Embeddings: A Technical Report. arXiv:2402.05672.

Järvelin, K., & Kekäläinen, J. (2002). Cumulated gain-based evaluation
of IR techniques. ACM TOIS, 20(4).

Sakai, T. (2014). Statistical Reform in Information Retrieval?
SIGIR Forum, 48(1).

Cleverdon, C. (1967). The Cranfield tests on index language devices.
Aslib Proceedings, 19(6).

MacAvaney, S., Yates, A., Feldman, S., Downey, D., Cohan, A., &
Goharian, N. (2021). Simplified Data Wrangling with ir_datasets. SIGIR '21.
```
