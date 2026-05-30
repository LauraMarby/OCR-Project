"""
remove_line.py
--------------
Elimina líneas de color o renglones de libreta de imágenes de documentos
manuscritos usando detección HSV + inpainting (colores) o morfología
horizontal + inpainting (libreta).

Uso:
    # Una imagen con renglón naranja (cuaderno de color)
    python remove_line.py --input foto.jpg --output limpia.png

    # Carpeta completa con renglones de libreta
    python remove_line.py --input carpeta/ --output limpia/ --color libreta

    # Otros colores disponibles
    python remove_line.py --input carpeta/ --output limpia/ --color rojo
    python remove_line.py --input carpeta/ --output limpia/ --color azul

Colores disponibles: naranja, amarillo, rojo, azul, verde, libreta

Dependencias:
    pip install opencv-python numpy
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

SUPPORTED = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

COLOR_RANGES = {
    'naranja': [
        (np.array([10,  80, 120]), np.array([35, 255, 255])),
    ],
    'amarillo': [
        (np.array([20,  80, 120]), np.array([40, 255, 255])),
    ],
    'rojo': [
        (np.array([0,   80, 80]),  np.array([10, 255, 255])),
        (np.array([165, 80, 80]),  np.array([179, 255, 255])),
    ],
    'azul': [
        (np.array([90, 80, 80]),   np.array([130, 255, 255])),
    ],
    'verde': [
        (np.array([40, 80, 80]),   np.array([90, 255, 255])),
    ],
}

def build_mask_color(img_bgr, color, dilate_px=3):
    """Detecta píxeles de un color específico en HSV."""
    hsv    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    ranges = COLOR_RANGES.get(color)
    if ranges is None:
        raise ValueError(f"Color '{color}' no reconocido. "
                         f"Opciones: {list(COLOR_RANGES.keys()) + ['libreta']}")

    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        mask |= cv2.inRange(hsv, lower, upper)

    if dilate_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_px, dilate_px))
        mask   = cv2.dilate(mask, kernel, iterations=2)

    return mask

def build_mask_libreta(img_bgr, min_line_width=80, dilate_v=3):
    """
    Detecta renglones de libreta por morfología horizontal.
    Busca estructuras muy largas y delgadas (renglones) que no son letras.

    min_line_width : ancho mínimo en píxeles para considerar algo un renglón.
                     Ajustar si la imagen es muy pequeña (bajar a 40-50).
    dilate_v       : píxeles de dilatación vertical para cubrir bien el renglón.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Abre morfológicamente con kernel horizontal largo:
    # conserva solo lo que es tan ancho como min_line_width → renglones
    kernel_h  = cv2.getStructuringElement(cv2.MORPH_RECT, (min_line_width, 1))
    lineas    = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_h)

    # Los renglones son zonas oscuras sobre fondo claro → invertir
    _, mask = cv2.threshold(lineas, 30, 255, cv2.THRESH_BINARY_INV)

    # Dilatar verticalmente para asegurar cobertura completa del renglón
    if dilate_v > 0:
        kernel_d = cv2.getStructuringElement(cv2.MORPH_RECT, (1, dilate_v))
        mask     = cv2.dilate(mask, kernel_d, iterations=1)

    return mask

def remove_colored_line(img_bgr, color='naranja', inpaint_radius=7, dilate_px=3):
    if color == 'libreta':
        mask = build_mask_libreta(img_bgr)
    else:
        mask = build_mask_color(img_bgr, color, dilate_px)

    n_px   = int(mask.sum() // 255)
    result = cv2.inpaint(img_bgr, mask, inpaintRadius=inpaint_radius,
                         flags=cv2.INPAINT_TELEA)
    return result, n_px

def process_single(input_path, output_path, color, inpaint_radius):
    img = cv2.imread(str(input_path))
    if img is None:
        print(f"  No se pudo leer: {input_path.name}")
        return False

    result, n_px = remove_colored_line(img, color, inpaint_radius)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), result)
    print(f"  OK  {input_path.name}  —  {n_px} px eliminados")
    return True

def process_folder(input_dir, output_dir, color, inpaint_radius, max_html):
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted([
        p for p in input_dir.rglob('*')
        if p.suffix.lower() in SUPPORTED
    ])

    if not images:
        print(f"No se encontraron imágenes en: {input_dir}")
        sys.exit(1)

    print(f"Imágenes  : {len(images)}")
    print(f"Color     : {color}")
    print(f"Salida    : {output_dir}\n")

    resultados = []
    for i, img_path in enumerate(images):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  No se pudo leer: {img_path.name}")
            resultados.append((img_path, 0, False))
            continue

        result, n_px = remove_colored_line(img, color, inpaint_radius)

        rel      = img_path.relative_to(input_dir)
        out_path = output_dir / rel.with_suffix('.png')
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), result)
        resultados.append((img_path, n_px, True))

        if (i + 1) % 50 == 0 or (i + 1) == len(images):
            print(f"  [{i+1}/{len(images)}] {img_path.name}  —  {n_px} px")

    ok        = sum(ok for *_, ok in resultados)
    sin_linea = sum(1 for _, n, ok in resultados if ok and n == 0)
    print(f"\n── Resumen ──")
    print(f"  Procesadas          : {ok}/{len(images)}")
    print(f"  Sin línea detectada : {sin_linea}  (pueden no tener renglón de ese color)")

    html_path = output_dir / "_comparativa_remove.html"
    _generate_html(input_dir, output_dir, images, resultados, html_path, max_html)
    print(f"\n  HTML de comparativa: {html_path}")
    print(f"Listo.")

def _generate_html(input_dir, output_dir, images, resultados, html_path, max_html):
    items = []
    for img_path, (_, n_px, ok) in zip(images[:max_html], resultados[:max_html]):
        rel      = img_path.relative_to(input_dir)
        out_p    = output_dir / rel.with_suffix('.png')
        orig_uri = img_path.resolve().as_uri()
        out_uri  = out_p.resolve().as_uri() if ok else ""

        warn    = '<span class="caution"> ningun px detectado</span>' if (ok and n_px == 0) else ""
        out_tag = f'<img src="{out_uri}">' if ok else '<p class="error">ERROR</p>'

        items.append(f"""
        <div class="card">
          <p class="fname">{img_path.name} &nbsp; {n_px} px eliminados {warn}</p>
          <div class="pair">
            <div><p>Original</p><img src="{orig_uri}"></div>
            <div><p>Sin renglón</p>{out_tag}</div>
          </div>
        </div>""")

    note = (f"<p style='color:#888'>Mostrando {max_html} de {len(images)}.</p>"
            if len(images) > max_html else "")

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <title>Eliminación de renglón</title>
  <style>
    body  {{ font-family: sans-serif; background: #1a1a1a; color: #eee; padding: 20px; }}
    h1    {{ color: #fff; }}
    .card {{ border: 1px solid #444; border-radius: 8px; padding: 12px;
             margin-bottom: 20px; background: #2a2a2a; }}
    .fname{{ font-size: 0.85em; color: #aaa; margin: 0 0 8px; }}
    .pair {{ display: flex; gap: 16px; flex-wrap: wrap; }}
    .pair > div {{ flex: 1; min-width: 200px; }}
    .pair p {{ font-size: 0.8em; color: #888; margin: 0 0 4px; }}
    img   {{ max-width: 100%; border: 1px solid #555; border-radius: 4px; background: white; }}
    .caution {{ color: #ffa94d; }}
    .error   {{ color: #ff6b6b; }}
  </style>
</head>
<body>
  <h1>Eliminación de renglón de color</h1>
  {note}
  {"".join(items)}
</body>
</html>"""
    html_path.write_text(html, encoding='utf-8')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Elimina líneas de color de imágenes de documentos.'
    )
    parser.add_argument('--input',          required=True,  help='Imagen o carpeta')
    parser.add_argument('--output',         required=True,  help='Destino imagen o carpeta')
    parser.add_argument('--color',          default='naranja',
                        choices=list(COLOR_RANGES.keys()) + ['libreta'],
                        help='Color a eliminar (default: naranja). Usar libreta para renglones de cuaderno.')
    parser.add_argument('--inpaint-radius', type=int, default=7,
                        help='Radio de inpainting (default: 7)')
    parser.add_argument('--max-html',       type=int, default=200,
                        help='Max imágenes en el HTML (default: 200)')
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output)

    if inp.is_dir():
        process_folder(inp, out, args.color, args.inpaint_radius, args.max_html)
    elif inp.is_file():
        if not process_single(inp, out, args.color, args.inpaint_radius):
            sys.exit(1)
    else:
        print(f"Error: {inp} no existe.")
        sys.exit(1)