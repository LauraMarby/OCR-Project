"""
dataset.py — Carga de colecciones IR vía la biblioteca `ir_datasets`.

Usa el catálogo oficial de ir_datasets (https://ir-datasets.com/) en
lugar de parsers a mano. Beneficios para tesis:

  - Los doc_id, query_id y qrels son los CANÓNICOS de la literatura,
    así que tus números son directamente comparables con los papers
    que reportan sobre las mismas colecciones.
  - Una sola línea cambia entre Cranfield y otras colecciones más
    modernas (BEIR, MS MARCO, TREC DL, NFCorpus, etc.).
  - ir_datasets maneja descarga, validación de hashes, caching y
    normalización de escalas de relevancia automáticamente.

Soporte multi-dataset: la función `load_dataset(name)` acepta
cualquier identificador del catálogo. La selección de campos de texto
se hace por heurística (probando combinaciones típicas como
`title + text`, `title + abstract`, etc.) o explícitamente vía el
parámetro `text_fields`.

Convención de relevancia (post-ir_datasets):
  - Mayor = más relevante.
  - 0 (o negativo) = no relevante.
  - Cranfield específicamente: 4=respuesta completa, 3=alta relevancia,
    2=fondo útil, 1=interés mínimo, -1=no relevante.

Para usar Cranfield offline si ya tenés los archivos cran.all.1400,
cran.qry, cranqrel localmente, basta empaquetarlos en el lugar donde
ir_datasets espera la tarball (ver README sección "Uso offline").
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

@dataclass
class Doc:
    """
    Documento normalizado para el retriever. `text` es lo que se indexa,
    `title` se reserva para mostrar en --inspect-query, `raw` conserva
    el namedtuple original de ir_datasets por si hace falta acceder a
    autores, URLs, etc.
    """
    doc_id: str
    text:   str
    title:  str = ""
    raw:    object = None       # ir_datasets.formats.* namedtuple

    def __repr__(self) -> str:
        snip = self.text[:60].replace("\n", " ")
        return f"Doc(id={self.doc_id!r}, title={self.title[:40]!r}, text={snip!r}...)"

@dataclass
class Collection:
    """
    Colección lista para evaluación. `qrels` y `qrels_graded` cubren
    métricas binarias y graded respectivamente (ver metrics.py).
    """
    name: str
    docs:         dict[str, Doc]            = field(default_factory=dict)
    queries:      dict[str, str]            = field(default_factory=dict)
    qrels:        dict[str, dict[str, int]] = field(default_factory=dict)
    qrels_graded: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def num_docs(self) -> int:    return len(self.docs)
    @property
    def num_queries(self) -> int: return len(self.queries)
    @property
    def num_qrels(self) -> int:   return sum(len(v) for v in self.qrels_graded.values())

    def __repr__(self) -> str:
        n_judged = sum(1 for q in self.queries if self.qrels.get(q))
        return (f"Collection(name={self.name!r}, docs={self.num_docs}, "
                f"queries={self.num_queries}, qrels={self.num_qrels}, "
                f"queries_with_judgments={n_judged})")

_DEFAULT_FIELD_CANDIDATES: list[tuple[str, ...]] = [
    ("title", "text"),        # Cranfield, BEIR/{scifact,nfcorpus,fiqa,...}, scidocs
    ("title", "body"),        # TREC Robust, TREC DL passage
    ("title", "abstract"),    # NFCorpus, algunos TREC bio
    ("title", "contents"),    # WikIR
    ("text",),                # Antique, Vaswani, BEIR/dbpedia
    ("body",),                # MS MARCO passage
    ("contents",),
]

def _detect_text_fields(sample_doc) -> tuple[str, ...]:
    """Elige la primera combinación que existe Y tiene contenido en sample_doc."""
    fields = set(sample_doc._fields)
    for combo in _DEFAULT_FIELD_CANDIDATES:
        if all(f in fields for f in combo):
            # Verificar que al menos uno tiene texto en el doc de muestra
            if any(getattr(sample_doc, f, "") for f in combo):
                return combo
    # Fallback: todos los campos string excepto doc_id/url
    skip = {"doc_id", "url", "iteration"}
    str_fields = tuple(
        f for f in sample_doc._fields
        if f not in skip and isinstance(getattr(sample_doc, f, None), str)
    )
    return str_fields or ("text",)

def _extract_text(doc, text_fields: tuple[str, ...]) -> tuple[str, str]:
    """Devuelve (texto_a_indexar, titulo_para_mostrar)."""
    parts = []
    for f in text_fields:
        v = getattr(doc, f, "")
        if v:
            parts.append(str(v))
    text = "\n\n".join(parts).strip()
    title = getattr(doc, "title", "") if "title" in doc._fields else ""
    return text, str(title)

def load_dataset(name: str = "cranfield",
                 *,
                 text_fields: Optional[Iterable[str]] = None,
                 binary_threshold: int = 1,
                 max_docs: Optional[int] = None,
                 cranfield_invert_original_scale: bool = False,
                 verbose: bool = True) -> Collection:
    """
    Carga una colección de ir_datasets en una Collection.

    Args:
        name: identificador de ir_datasets (e.g. "cranfield",
            "vaswani", "beir/scifact/test", "msmarco-passage/dev/small",
            "antique/test"). Catálogo completo: https://ir-datasets.com/
        text_fields: campos a concatenar para formar el texto a indexar.
            Si es None, se detecta automáticamente con una heurística.
            Ejemplos: ("title", "text"), ("title", "abstract"), ("text",).
        binary_threshold: una relevancia >= este valor cuenta como
            "relevante" para métricas binarias (P, R, MAP, MRR, R-Prec).
            Default 1: cualquier juicio positivo cuenta. Para Cranfield,
            poner 2 excluye el grado más bajo.
        max_docs: si se da, limita la cantidad de docs cargados (útil
            para pruebas rápidas sobre colecciones enormes).
        cranfield_invert_original_scale: aplica SOLO a "cranfield".
            El archivo original de Glasgow usa la convención 1=mejor,
            4=peor, -1=no rel. La biblioteca `ir_datasets` lee los
            valores TAL CUAL del archivo, pero documenta los labels
            como si fuera la escala invertida (4=mejor). Esa
            interpretación es la dominante en la literatura post-2020
            que usa ir_datasets/PyTerrier — y es la que aplica este
            loader por defecto (sin invertir).

            Si querés reproducir números de papers PRE-ir_datasets que
            reportan con la escala original Glasgow (1=mejor), o
            simplemente ser fiel al paper de Cleverdon de 1967,
            pasá True aquí: se aplica la transformación 1→4, 2→3,
            3→2, 4→1, -1→0.

            ⚠ Esto SOLO afecta a nDCG. Las métricas binarias (MAP,
            MRR, P@k, etc.) son invariantes porque {1,2,3,4} todas
            cuentan como relevantes con binary_threshold=1.
        verbose: logs INFO sobre lo que va cargando.

    Returns:
        Collection con docs, queries y dos vistas de qrels (binary y
        graded). Si la colección de ir_datasets no trae alguno de los
        tres componentes (p.e. una colección sólo de docs sin qrels),
        el atributo correspondiente queda vacío.

    Lanza:
        ImportError si ir_datasets no está instalado.
        ir_datasets exception si el dataset no existe o falla descarga.
    """
    try:
        import ir_datasets  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "ir_datasets no está instalado. pip install ir_datasets"
        ) from exc

    ds = ir_datasets.load(name)
    coll = Collection(name=name)

    if not ds.has_docs():
        raise ValueError(f"Dataset {name!r} no tiene docs.")

    # Tomamos un sample para auto-detectar campos
    first_doc = next(ds.docs_iter())
    if text_fields is None:
        text_fields_t = _detect_text_fields(first_doc)
        if verbose:
            logger.info("Campos auto-detectados para texto: %s", text_fields_t)
    else:
        text_fields_t = tuple(text_fields)
        # Verificar que existen
        unknown = [f for f in text_fields_t if f not in first_doc._fields]
        if unknown:
            raise ValueError(
                f"Campos {unknown!r} no existen en el dataset {name!r}. "
                f"Disponibles: {list(first_doc._fields)}"
            )

    if verbose:
        try:
            total = ds.docs_count()
            logger.info("Cargando %d docs de %s ...", total, name)
        except Exception:
            logger.info("Cargando docs de %s ...", name)

    n = 0
    for doc in ds.docs_iter():
        text, title = _extract_text(doc, text_fields_t)
        coll.docs[str(doc.doc_id)] = Doc(
            doc_id=str(doc.doc_id),
            text=text,
            title=title,
            raw=doc,
        )
        n += 1
        if max_docs and n >= max_docs:
            break

    if verbose:
        logger.info("✓ %d docs cargados.", coll.num_docs)

    if ds.has_queries():
        for q in ds.queries_iter():
            coll.queries[str(q.query_id)] = q.text
        if verbose:
            logger.info("✓ %d queries cargadas.", coll.num_queries)
    else:
        if verbose:
            logger.warning("Dataset %r no tiene queries.", name)

    if ds.has_qrels():
        do_invert = (cranfield_invert_original_scale
                     and name.split("/")[0] == "cranfield")
        if do_invert and verbose:
            logger.info("Aplicando inversión de escala original Glasgow "
                        "(1↔4, 2↔3) para Cranfield.")
        for qr in ds.qrels_iter():
            qid = str(qr.query_id)
            did = str(qr.doc_id)
            rel = int(qr.relevance)
            # Inversión opcional (solo Cranfield)
            if do_invert and 1 <= rel <= 4:
                rel = 5 - rel
            # graded: clipeamos a [0, +inf) para nDCG (los -1 → 0)
            coll.qrels_graded.setdefault(qid, {})[did] = max(rel, 0)
            # binary: aplicamos threshold
            if rel >= binary_threshold:
                coll.qrels.setdefault(qid, {})[did] = 1
        if verbose:
            logger.info("✓ %d juicios de relevancia cargados (%d queries con juicios).",
                        coll.num_qrels,
                        sum(1 for q in coll.queries if coll.qrels.get(q)))
    else:
        if verbose:
            logger.warning("Dataset %r no tiene qrels.", name)

    return coll

def list_available_text_field_combos() -> list[tuple[str, ...]]:
    """Para que la CLI pueda imprimirlas si el usuario lo pide."""
    return list(_DEFAULT_FIELD_CANDIDATES)
