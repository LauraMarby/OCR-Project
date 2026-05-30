"""
Visualizador del pipeline de preprocesado etapa por etapa.

Ejecuta `preprocess_line` paso a paso capturando la imagen intermedia
tras cada etapa, y genera un PNG con todas las etapas dispuestas en
grid para inspección visual.

Las 12 etapas capturadas:

  1.  Original (RGB)
  2.  Escala de grises — canal con máximo contraste
  3.  Filtrado bilateral — denoising preservando bordes
  4.  Estimación de fondo — clausura morfológica del papel
  5.  Normalización de iluminación — división por el fondo
  6.  Deskew — rotación inversa por ajuste lineal del baseline
  7.  Binarización Sauvola — umbralización local con auto-ajuste de k
  8.  Enderezado polinomial — corrección de curvatura del baseline
  9.  Sin rayas de pauta — eliminación por Hough probabilístico
  10. Limpieza de ruido — descarte de componentes pequeños
  11. Recorte vertical — bbox a las filas con tinta + margen
  12. Resize final — altura fija para entrada al modelo OCR

Configuración por constantes al inicio del archivo, como en visualize.py.
"""

import sys
import traceback
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

sys.path.insert(0, str(Path(__file__).parent))

from preprocessing.binarization    import normalize_illumination
from preprocessing.line_processing import straighten_line
from preprocessing.line_preprocess import (
    LineConfig,
    _to_gray_best,
    _estimate_text_height,
    _deskew_grayscale,
    _binarize_with_retry,
    _remove_horizontal_rule_lines,
    _clean_speckles,
    _vertical_trim_only,
    _resize_to_height,
)

INPUT_PATH = Path("new_prepr/223cc42a598df928b4446086d3b5b59b_linea_06.jpg")               # archivo .jpg/.png o carpeta
OUTPUT_DIR = Path("new_prepr")        # destino de los PNG generados

GRID_COLS  = 1         # columnas del grid (12 etapas → 4 × 3 filas)
FIG_WIDTH  = 20        # ancho de la figura en pulgadas (cabe en A4)
DPI        = 150       # resolución del PNG de salida

_IMG_EXTS = {".jpg", ".jpeg", ".png"}

def preprocess_with_snapshots(
    img_bgr: np.ndarray,
    cfg:     LineConfig = None,
) -> list:
    """
    Reproduce `preprocess_line` etapa por etapa y devuelve una lista de
    tuplas (título, imagen, modo) donde:

      · título: descripción de la etapa.
      · imagen: numpy array en el formato producido por esa etapa.
      · modo:   'rgb' o 'gray', usado por el visualizador para elegir cmap.
    """
    if cfg is None:
        cfg = LineConfig()

    snaps = []

    # ── 1. Original (RGB)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if img_bgr.ndim == 3 else img_bgr
    snaps.append(("1. Original (RGB)", rgb, "rgb" if img_bgr.ndim == 3 else "gray"))

    # ── 2. Escala de grises — canal con máximo contraste
    gray = _to_gray_best(img_bgr, pick_best=cfg.pick_best_channel)
    snaps.append(("2. Escala de grises", gray, "gray"))

    # ── 3. Filtrado bilateral
    if cfg.use_bilateral:
        gray_bilat = cv2.bilateralFilter(
            gray,
            d           = cfg.bilateral_d,
            sigmaColor  = cfg.bilateral_sigma,
            sigmaSpace  = cfg.bilateral_sigma,
        )
    else:
        gray_bilat = gray.copy()
    snaps.append(("3. Filtrado bilateral", gray_bilat, "gray"))

    H, W = gray_bilat.shape
    ksize = max(25, min(H, W) // 15)
    if ksize % 2 == 0:
        ksize += 1
    kernel     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    background = cv2.morphologyEx(gray_bilat, cv2.MORPH_CLOSE, kernel)
    snaps.append(("4. Estimación de fondo", background, "gray"))

    # ── 5. Normalización de iluminación
    if cfg.normalize_bg:
        gray_norm = normalize_illumination(gray_bilat)
    else:
        gray_norm = gray_bilat.copy()
    snaps.append(("5. Normalización iluminación", gray_norm, "gray"))

    # ── 6. Deskew (rotación inversa por baseline)
    angle = 0.0
    if cfg.deskew:
        gray_desk, angle = _deskew_grayscale(gray_norm, max_angle=cfg.max_skew_angle)
    else:
        gray_desk = gray_norm.copy()
    snaps.append((f"6. Deskew ({angle:+.2f}°)", gray_desk, "gray"))

    # ── 7. Binarización Sauvola con auto-ajuste de k
    text_h = _estimate_text_height(gray_desk)
    win    = cfg.sauvola_window if cfg.sauvola_window > 0 else max(15, int(text_h * 1.2))
    if win % 2 == 0:
        win += 1
    binary, k_used = _binarize_with_retry(
        gray_desk,
        window     = win,
        k_init     = cfg.sauvola_k,
        pre_blur   = cfg.sauvola_pre_blur,
        target_ink = cfg.target_ink_range,
    )
    snaps.append((f"7. Binarización Sauvola (k={k_used:.2f})", binary, "gray"))

    # ── 8. Enderezado polinomial
    binary_str = straighten_line(binary, poly_degree=2) if cfg.straighten else binary.copy()
    snaps.append(("8. Enderezado polinomial", binary_str, "gray"))

    # ── 9. Eliminación de rayas de pauta
    if cfg.remove_rule_lines:
        binary_rule = _remove_horizontal_rule_lines(
            binary_str,
            max_h             = cfg.rule_line_max_h,
            min_coverage_frac = cfg.rule_min_coverage,
            max_angle_deg     = cfg.rule_max_angle_deg,
        )
    else:
        binary_rule = binary_str.copy()
    snaps.append(("9. Sin rayas de pauta", binary_rule, "gray"))

    # ── 10. Limpieza de componentes pequeños
    if cfg.remove_noise:
        binary_clean = _clean_speckles(
            binary_rule,
            max_area       = cfg.noise_max_area,
            preserve_edges = cfg.preserve_edges,
        )
    else:
        binary_clean = binary_rule.copy()
    snaps.append(("10. Limpieza de ruido", binary_clean, "gray"))

    # ── 11. Recorte vertical
    if cfg.vertical_trim:
        binary_trim = _vertical_trim_only(binary_clean, margin=cfg.vertical_margin)
    else:
        binary_trim = binary_clean.copy()
    snaps.append(("11. Recorte vertical", binary_trim, "gray"))

    # ── 12. Resize a altura objetivo
    if (binary_trim < 128).sum() == 0:
        # Sin tinta detectada: devolver un lienzo en blanco para no romper batch.
        out = np.full((cfg.target_height, cfg.min_output_width), 255, dtype=np.uint8)
    else:
        out = _resize_to_height(binary_trim, target_h=cfg.target_height, min_w=cfg.min_output_width)
    snaps.append((f"12. Resize ({out.shape[1]}×{out.shape[0]} px)", out, "gray"))

    return snaps

def plot_stages(
    snaps:     list,
    save_path: Path,
    cols:      int   = GRID_COLS,
    fig_width: float = FIG_WIDTH,
    title:     str   = "",
) -> None:
    """
    Genera una figura con todas las etapas en grid `rows × cols` y la
    guarda como PNG. La altura de la figura se calcula a partir del
    aspect ratio promedio de las imágenes para que las celdas no
    queden ni aplastadas ni alargadas.
    """
    n    = len(snaps)
    rows = (n + cols - 1) // cols

    cell_w    = fig_width / cols
    # Altura mínima por celda en pulgadas, independiente del aspect ratio.
    # Con imágenes muy apaisadas (texto manuscrito) un valor fijo de 1.2–2.0
    # da legibilidad sin que la figura sea enorme.
    cell_h    = 2.0

    title_pad  = 0.7 if title else 0.0          # pulgadas reservadas al suptitle
    fig_height = cell_h * rows + 0.4 + title_pad

    # Conversión a fracciones (matplotlib usa fracciones del figsize total).
    top_frac          = 1.0 - title_pad / fig_height
    suptitle_y_frac   = 1.0 - 0.30 / fig_height

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=DPI)
    gs  = gridspec.GridSpec(
        rows, cols,
        figure  = fig,
        hspace  = 0.7,
        wspace  = 0.10,
        left    = 0.02,
        right   = 0.98,
        top     = top_frac,
        bottom  = 0.03,
    )

    for i, (label, img, mode) in enumerate(snaps):
        r, c = divmod(i, cols)
        ax   = fig.add_subplot(gs[r, c])
        if mode == "rgb":
            ax.imshow(img, aspect="auto")
        else:
            ax.imshow(img, cmap="gray", vmin=0, vmax=255, aspect="auto")
        ax.set_title(label, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Celdas vacías en la última fila (si las hay)
    for k in range(n, rows * cols):
        r, c = divmod(k, cols)
        ax   = fig.add_subplot(gs[r, c])
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=13, y=suptitle_y_frac)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=DPI, facecolor="white")
    plt.close(fig)

def process_one(img_path: Path, out_dir: Path) -> str:
    """Procesa una imagen y guarda el PNG con las etapas."""
    img_bgr = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None:
        print(f"  ERROR: no se pudo leer '{img_path}'", file=sys.stderr)
        return "error"

    snaps    = preprocess_with_snapshots(img_bgr)
    out_path = out_dir / f"{img_path.stem}_stages.png"
    plot_stages(snaps, save_path=out_path, title=f"Pipeline: {img_path.name}")
    print(f"  [OK]   {img_path.name} → {out_path}")
    return "ok"

def main() -> None:
    inp = INPUT_PATH.resolve()
    out = OUTPUT_DIR.resolve()
    out.mkdir(parents=True, exist_ok=True)

    if inp.is_file() and inp.suffix.lower() in _IMG_EXTS:
        print(f"\nVisualizando etapas de: {inp}")
        print(f"Salida en:              {out}\n")
        process_one(inp, out)
        return

    if inp.is_dir():
        images = sorted(p for p in inp.iterdir()
                        if p.suffix.lower() in _IMG_EXTS and p.is_file())
        if not images:
            print(f"No se encontraron .jpg/.jpeg/.png en '{inp}'.", file=sys.stderr)
            sys.exit(0)

        print(f"\n{'='*60}")
        print(f"  Visualización por etapas — {len(images)} imágenes")
        print(f"  Entrada: {inp}")
        print(f"  Salida:  {out}")
        print(f"{'='*60}\n")

        ok = err = 0
        for i, img in enumerate(images, 1):
            print(f"  [{i:>3}/{len(images)}] ", end="")
            try:
                status = process_one(img, out)
            except Exception:
                print(f"ERROR procesando {img.name}:")
                traceback.print_exc()
                status = "error"
            if status == "ok":
                ok += 1
            else:
                err += 1

        print(f"\n{'='*60}")
        print(f"  Completado.  OK: {ok}   ERROR: {err}")
        print(f"{'='*60}\n")
        return

    print(f"ERROR: INPUT_PATH='{inp}' no es archivo válido ni directorio.",
          file=sys.stderr)
    sys.exit(1)

if __name__ == "__main__":
    main()