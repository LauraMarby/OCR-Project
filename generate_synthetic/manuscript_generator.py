
import os
import re
import cv2
import json
import time
import random
import hashlib
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count

from PIL import Image, ImageDraw, ImageFont, ImageFilter
import albumentations as A

CORPUS_DIR     = "./Corpus/"              # carpeta con los .txt del corpus
FONT_DIR       = "./fonts/"               # directorio de fuentes
OUTPUT_DIR     = "./synthetic_manuscript" # directorio de salida
BACKGROUND_DIR = "./backgrounds/"         # fondos externos (opcional)

TOTAL_IMAGES   = 24  # objetivo total; ajusta según tu capacidad de cómputo
MIN_PER_FONT   = 1      # mínimo garantizado por fuente

N_WORKERS = None

MIN_TOKENS = 5
MAX_TOKENS = 20
MAX_CHARS  = 90

FONT_SIZE_RANGE       = (32, 60)
FONT_SIZE_BREAKPOINTS = [40, 48]
FONT_SIZE_WEIGHTS     = [0.00, 0.30, 0.70]
MARGIN_X_RANGE  = (25, 60)
MARGIN_Y_RANGE  = (14, 36)

SUPERSAMPLE = 2

P_ELASTIC    = 1.00
P_GRID       = 1.00
P_MORPH      = 1.00
P_WAVE       = 1.00
P_PERSPECTIVE= 1.00
P_INK_BLEED  = 1.00
P_NOISE      = 1.00
P_RULED_LINES= 0.18

FRECUENCIAS_OBJETIVO = {
    "a": 10.2, "e": 11.5, "i":  5.3, "o":  7.5, "u":  3.1,
    "á":  2.5, "é":  2.5, "í":  2.5, "ó":  2.5, "ú":  2.5, "ü":  1.5,
    "s":  6.8, "r":  5.9, "n":  5.7, "l":  5.0, "d":  4.2,
    "t":  3.9, "c":  2.9, "m":  2.5, "p":  2.3,
    "b":  2.5, "h":  2.5, "q":  2.5, "v":  2.5,
    "f":  2.5, "g":  2.5, "y":  2.5, "z":  2.5,
    "j":  2.0, "ñ":  2.0, "k":  2.0, "w":  2.0, "x":  2.0,
    "A":  4.5, "E":  5.0, "I":  2.5, "O":  3.5, "U":  1.5,
    "Á":  1.5, "É":  1.5, "Í":  1.5, "Ó":  1.5, "Ú":  1.5, "Ü":  1.0,
    "S":  3.0, "R":  2.5, "N":  2.5, "L":  2.5, "D":  2.0,
    "T":  2.0, "C":  2.0, "M":  2.0, "P":  2.0,
    "B":  1.5, "H":  1.5, "Q":  1.5, "V":  1.5,
    "F":  1.5, "G":  1.5, "Y":  1.5, "Z":  1.5,
    "J":  1.5, "Ñ":  1.5, "K":  1.0, "W":  1.0, "X":  1.0,
    "0":  1.5, "1":  1.5, "2":  1.5, "3":  1.5, "4":  1.5,
    "5":  1.5, "6":  1.5, "7":  1.5, "8":  1.5, "9":  1.5,
    ".":  3.0, ",":  2.5, ";":  1.5, ":":  1.5,
    "!":  1.5, "?":  1.5, "¡":  2.0, "¿":  2.0,
    "(":  1.0, ")":  1.0, '"':  1.0, "'":  1.0, "-": 2.0,
    "—":  3.0, "«":  2.0, "»":  2.0,
    "%":  1.5, "&":  1.5, "$":  1.5, "#":  0.8,
    "*":  1.0, "[":  0.8, "]":  0.8,
}

def cargar_corpus(corpus_dir: str) -> list:
    p = Path(corpus_dir)
    if not p.exists():
        raise FileNotFoundError(
            f"No se encontró el directorio de corpus: '{corpus_dir}'\n"
            f"Crea la carpeta y pon dentro tus archivos .txt."
        )

    archivos = sorted(p.glob("*.txt"))
    if not archivos:
        raise ValueError(f"No hay archivos .txt en '{corpus_dir}'")

    regex = r"[a-zA-ZáéíóúÁÉÍÓÚüÜñÑ]+|\d|[—«»]|[^\w\s]"
    todos_tokens = []

    print(f"  Leyendo corpus desde '{corpus_dir}':")
    for archivo in archivos:
        try:
            texto  = archivo.read_text(encoding="utf-8", errors="replace")
            tokens = re.findall(regex, texto)
            todos_tokens.extend(tokens)
            print(f"    {archivo.name:<40} {len(tokens):>10,} tokens")
        except Exception as e:
            print(f"    {archivo.name:<40} ERROR: {e}")

    if not todos_tokens:
        raise ValueError("El corpus está vacío tras leer todos los archivos.")

    print(f"\n  Total tokens extraídos: {len(todos_tokens):,}")
    return todos_tokens

def construir_indice_posiciones(todos_tokens: list) -> dict:
    indice = defaultdict(list)
    for i, tok in enumerate(todos_tokens):
        for ch in set(tok):
            indice[ch].append(i)
    chars_idx = len(indice)
    print(f"  Índice posicional: {len(todos_tokens):,} tokens | {chars_idx} caracteres indexados")
    return indice

def _deficit(contador: Counter, obj: dict, total: int) -> dict:
    total_peso = sum(obj.values())
    return {
        ch: (p / total_peso) * total - contador.get(ch, 0)
        for ch, p in obj.items()
    }

def _elegir_inicio(indice_pos: dict, deficit: dict, max_start: int) -> int:
    candidatos = sorted(deficit.items(), key=lambda x: x[1], reverse=True)[:12]
    random.shuffle(candidatos)
    for ch, d in candidatos:
        if d > 0 and ch in indice_pos:
            pos = random.choice(indice_pos[ch])
            retroceso = random.randint(0, min(4, pos))
            start = pos - retroceso
            if 0 <= start <= max_start:
                return start
    return random.randint(0, max_start)

def generar_frases_corpus(todos_tokens: list, indice_pos: dict, n: int) -> list:
    if len(todos_tokens) < MIN_TOKENS:
        raise ValueError("Corpus demasiado pequeño para generar frases.")

    max_start = len(todos_tokens) - MIN_TOKENS
    frases: list  = []
    usadas:  set  = set()
    contador = Counter()
    intentos = 0
    max_intentos = n * 40

    while len(frases) < n and intentos < max_intentos:
        intentos += 1

        total = sum(contador.values())
        dfct  = (_deficit(contador, FRECUENCIAS_OBJETIVO, total)
                 if total > 0 else {k: 1.0 for k in FRECUENCIAS_OBJETIVO})

        start = _elegir_inicio(indice_pos, dfct, max_start)
        n_tok = random.randint(MIN_TOKENS, MAX_TOKENS)
        tokens_candidatos = todos_tokens[start: start + n_tok]

        tokens_ok: list = []
        chars = 0
        for tok in tokens_candidatos:
            extra = len(tok) + (1 if tokens_ok else 0)
            if chars + extra > MAX_CHARS:
                break
            tokens_ok.append(tok)
            chars += extra

        if len(tokens_ok) < MIN_TOKENS:
            continue

        frase = " ".join(tokens_ok)

        if frase not in usadas:
            frases.append(frase)
            usadas.add(frase)
            contador.update(c for c in frase if c != " ")

    print(f"  {len(frases):,} frases generadas desde corpus ({intentos:,} intentos)")
    if len(frases) < n:
        print(f"  ⚠ Solo {len(frases):,} frases únicas (pedidas: {n:,}). "
              f"Ampliá el corpus o reducí TOTAL_IMAGES.")
    return frases

def _papel_color() -> tuple:
    modo = random.choice(["blanco", "crema", "crema", "amarillo"])
    if modo == "blanco":
        v = random.randint(248, 255)
        return (v, v, v)
    elif modo == "crema":
        r = random.randint(240, 252)
        return (r, r - random.randint(4, 12), r - random.randint(10, 22))
    else:
        r = random.randint(245, 255)
        return (r, r - random.randint(2, 8), r - random.randint(30, 50))

def generar_fondo_papel(w: int, h: int, font_size: int = 48) -> Image.Image:
    """Fondo sintético realista. Optimización: ruido en float32 explícito."""
    img = Image.new("RGB", (w, h), _papel_color())
    arr = np.asarray(img, dtype=np.float32).copy()

    # Textura de grano — float32 explícito (era float64 por defecto)
    noise = np.random.normal(
        0, random.uniform(1.0, 3.5), arr.shape
    ).astype(np.float32)
    arr += noise
    np.clip(arr, 0, 255, out=arr)
    img = Image.fromarray(arr.astype(np.uint8))

    if random.random() < P_RULED_LINES:
        draw   = ImageDraw.Draw(img)
        min_spacing = int(font_size * 1.5)
        spacing = random.randint(max(min_spacing, 38), max(min_spacing + 12, 52))
        offset  = random.randint(0, spacing)
        estilo  = random.choice(["gris", "azul"])

        papel_color = _papel_color()
        if estilo == "azul":
            color_linea = (
                min(255, papel_color[0] - random.randint(15, 30)),
                min(255, papel_color[1] - random.randint(5,  15)),
                min(255, papel_color[2] + random.randint(10, 25)),
            )
        else:
            v = min(255, sum(papel_color) // 3 - random.randint(18, 35))
            color_linea = (v, v, v)

        for y in range(offset, h, spacing):
            draw.line([(0, y), (w, y)], fill=color_linea, width=1)

    return img

_BG_CACHE: list = []

def obtener_fondo_externo(w: int, h: int):
    """
    Recorte aleatorio de un fondo externo con rotación leve, brillo/contraste
    y ruido. Optimización: en lugar de rotar la imagen completa (potencialmente
    millones de píxeles) y luego recortar, recortamos primero un parche
    ligeramente mayor que el target, lo rotamos y centramos al tamaño final.
    Equivalente visual.
    """
    if not _BG_CACHE:
        return None

    bg_arr = random.choice(_BG_CACHE)   # uint8 (H, W, 3)
    bh, bw = bg_arr.shape[:2]

    angle = random.uniform(-5, 5)

    # Padding suficiente para que la rotación no introduzca esquinas vacías
    # en el área central. Con ±5°, ~10% de pad alcanza con margen.
    pad = int(max(w, h) * 0.10) + 8
    cw, ch = w + 2 * pad, h + 2 * pad

    if bw < cw or bh < ch:
        # Fondo más pequeño que lo necesario: resize completo (raro).
        crop = cv2.resize(bg_arr, (w, h), interpolation=cv2.INTER_LANCZOS4)
    else:
        x = random.randint(0, bw - cw)
        y = random.randint(0, bh - ch)
        crop = bg_arr[y:y + ch, x:x + cw]

        # PIL Image.rotate() sin resample usa NEAREST → cv2.INTER_NEAREST
        M = cv2.getRotationMatrix2D((cw / 2.0, ch / 2.0), angle, 1.0)
        crop = cv2.warpAffine(
            crop, M, (cw, ch),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(245, 245, 240),
        )
        # Crop central al tamaño final
        crop = crop[pad:pad + h, pad:pad + w]

    # Brillo / contraste / ruido (mismo rango que el original, float32 explícito)
    arr = crop.astype(np.float32)
    brillo    = random.uniform(-12, 12)
    contraste = random.uniform(0.88, 1.12)
    arr = (arr + brillo) * contraste

    sigma = random.uniform(1.5, 4.0)
    noise = np.random.normal(0, sigma, arr.shape).astype(np.float32)
    arr += noise
    np.clip(arr, 0, 255, out=arr)

    return Image.fromarray(arr.astype(np.uint8))

def _elastic_transform(arr: np.ndarray) -> np.ndarray:
    """
    Deformación elástica + grid con los parámetros originales:
      alpha 30–65, sigma 4–7, num_steps 2–5, distort_limit 0.15–0.32.
    Sin cambios respecto a v2.0.
    """
    transform = A.Compose([
        A.ElasticTransform(
            alpha=random.uniform(30, 65),
            sigma=random.uniform(4, 7),
            border_mode=cv2.BORDER_CONSTANT,
            fill=(255, 255, 255),
            p=P_ELASTIC,
        ),
        A.GridDistortion(
            num_steps=random.randint(2, 5),
            distort_limit=random.uniform(0.15, 0.32),
            border_mode=cv2.BORDER_CONSTANT,
            fill=(255, 255, 255),
            p=P_GRID,
        ),
    ])
    return transform(image=arr)["image"]

def _slant(img: Image.Image) -> Image.Image:
    """Inclinación horizontal variable (PIL BICUBIC). Sin cambios."""
    w, h = img.size
    shear = random.uniform(-0.05, 0.05)
    return img.transform(
        (w, h), Image.Transform.AFFINE,
        (1, shear, 0, 0, 1, 0),
        resample=Image.Resampling.BICUBIC,
        fillcolor="white"
    )

def _paper_wave(img: Image.Image) -> Image.Image:
    """
    Curvatura del renglón. Misma fórmula y mismos parámetros que v2.0
    (intensidad 2 px, período 100 px), pero implementado con cv2.remap
    en lugar de scipy.ndimage.map_coordinates.

    Equivalencias:
      scipy mode='nearest'    ↔  cv2.BORDER_REPLICATE
      scipy order=3 (default) ↔  cv2.INTER_CUBIC
                                 (cubic spline vs bicubic difieren ~0.1%
                                 por píxel para desplazamientos de 2 px,
                                 visualmente indistinguible)

    Speedup: ~6-10× en esta función, sobre todo a 2× supersample.
    """
    arr = np.asarray(img)
    h, w = arr.shape[:2]
    x, y = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
    )
    intensity = 2.0
    map_x = x + intensity * np.sin(2.0 * np.pi * y / 100.0)
    map_y = y + intensity * np.cos(2.0 * np.pi * x / 100.0)
    out = cv2.remap(
        arr,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return Image.fromarray(out)

def _tremor(img: Image.Image) -> Image.Image:
    """Micro-desplazamiento global (PIL BICUBIC). Sin cambios."""
    w, h = img.size
    max_shift = random.randint(0, 2)
    sx = random.randint(-max_shift, max_shift)
    sy = random.randint(-max_shift, max_shift)
    return img.transform(
        (w, h), Image.Transform.AFFINE,
        (1, 0, sx, 0, 1, sy),
        resample=Image.Resampling.BICUBIC,
        fillcolor="white"
    )

def aplicar_augmentaciones(img: Image.Image) -> Image.Image:
    """Pipeline idéntico a v2.0: elastic+grid → slant → paper_wave → tremor."""
    arr = _elastic_transform(np.array(img))
    img = Image.fromarray(arr)
    img = _slant(img)
    img = _paper_wave(img)
    img = _tremor(img)
    return img

_FONT_CACHE: dict = {}

def _get_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    key = (path, size)
    if key not in _FONT_CACHE:
        _FONT_CACHE[key] = ImageFont.truetype(path, size)
    return _FONT_CACHE[key]

def _color_tinta() -> tuple:
    paleta = [
        (random.randint(0, 15),  random.randint(0, 15),  random.randint(0, 15)),
        (random.randint(0, 15),  random.randint(0, 15),  random.randint(0, 15)),
        (0,  0,  random.randint(60, 130)),
        (random.randint(10, 40), random.randint(10, 40), random.randint(80, 140)),
        (random.randint(25, 65), random.randint(10, 25), 0),
    ]
    return random.choice(paleta)

def generar_imagen(texto: str, font_path: str) -> Image.Image:
    """Renderiza la frase con fondo + texto + augmentación. Misma lógica que v2.0."""
    S         = SUPERSAMPLE
    font_size = random.randint(*FONT_SIZE_RANGE)
    margin_x  = random.randint(*MARGIN_X_RANGE)
    margin_y  = random.randint(*MARGIN_Y_RANGE)

    try:
        font_1x = _get_font(font_path, font_size)
        font_Sx = _get_font(font_path, font_size * S)
    except Exception:
        font_1x = ImageFont.load_default()
        font_Sx = font_1x

    dummy = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    bx0, by0, bx1, by1 = dummy.textbbox((0, 0), texto, font=font_1x)
    w_1x = (bx1 - bx0) + 2 * margin_x
    h_1x = (by1 - by0) + 2 * margin_y

    w_hi, h_hi = w_1x * S, h_1x * S
    fondo = obtener_fondo_externo(w_hi, h_hi)
    if fondo is None or random.random() < 0.40:
        fondo = generar_fondo_papel(w_hi, h_hi)

    bx0s, by0s, _, _ = dummy.textbbox((0, 0), texto, font=font_Sx)
    ImageDraw.Draw(fondo).text(
        (margin_x * S - bx0s, margin_y * S - by0s),
        texto, font=font_Sx, fill=_color_tinta()
    )

    img_hi = aplicar_augmentaciones(fondo)
    return img_hi.resize((w_1x, h_1x), Image.Resampling.LANCZOS)

_OUTPUT_DIR_WORKER: Path = None

def _worker_init(bg_files_paths: list, output_dir: str):
    """
    Inicializador del Pool. Se ejecuta UNA VEZ por proceso worker.
    Carga todos los backgrounds en RAM y guarda el directorio de salida.
    También re-siembra el RNG para que cada worker genere secuencias
    diferentes (evita que workers gemelos produzcan augmentaciones idénticas).
    """
    global _BG_CACHE, _OUTPUT_DIR_WORKER

    _BG_CACHE = []
    for p in bg_files_paths:
        try:
            _BG_CACHE.append(np.array(Image.open(p).convert("RGB")))
        except Exception:
            pass

    _OUTPUT_DIR_WORKER = Path(output_dir)

    # Cada worker con su propia semilla (PID + tiempo del sistema)
    seed = (os.getpid() * 100003) ^ int(time.time() * 1000) & 0xFFFFFFFF
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)

def _generate_one(args):
    """
    Tarea worker: genera y guarda una imagen.
    Devuelve (font_name, img_name, texto, error_str_o_None).
    """
    font_path, font_name, idx, texto = args
    uid      = hashlib.md5(f"{texto}{font_name}{idx}".encode()).hexdigest()[:8]
    img_name = f"{font_name}_{idx:05d}_{uid}.png"
    out_path = _OUTPUT_DIR_WORKER / img_name

    try:
        img = generar_imagen(texto, font_path)
        # Compresión PNG por defecto (compress_level=6) — no se modifica.
        img.save(str(out_path), "PNG")
        return (font_name, img_name, texto, None)
    except Exception as e:
        return (font_name, img_name, texto, str(e))

def main():
    print("═" * 70)
    print("  GENERADOR OCR MANUSCRITOS EN ESPAÑOL — v2.1 (paralelizado)")
    print("═" * 70)

    print("\n[1/4] Cargando corpus...")
    todos_tokens = cargar_corpus(CORPUS_DIR)
    indice_pos   = construir_indice_posiciones(todos_tokens)

    print("\n[2/4] Cargando fuentes...")
    fonts = (
        sorted(Path(FONT_DIR).glob("*.ttf")) +
        sorted(Path(FONT_DIR).glob("*.otf"))
    )
    fonts = [str(f) for f in fonts]
    if not fonts:
        print(f"ERROR: No se encontraron fuentes en '{FONT_DIR}'")
        return

    # Verificar que cada fuente carga (filtra las rotas antes de paralelizar)
    valid_fonts = []
    for f in fonts:
        try:
            ImageFont.truetype(f, 32)
            valid_fonts.append(f)
            print(f"  ✓ {Path(f).name}")
        except Exception as e:
            print(f"  ✗ {Path(f).name}: {e}")

    if not valid_fonts:
        print("ERROR: Ninguna fuente cargó correctamente.")
        return
    fonts = valid_fonts

    print("\n[3/4] Generando frases desde el corpus (ventanas contiguas)...")
    imgs_por_fuente = max(MIN_PER_FONT, TOTAL_IMAGES // len(fonts))
    total_real      = imgs_por_fuente * len(fonts)
    print(f"  {total_real:,} imágenes en total  ({imgs_por_fuente:,} por fuente × {len(fonts)} fuentes)")

    frases_necesarias = total_real
    print(f"  Generando {frases_necesarias:,} frases únicas...")
    pool_frases = generar_frases_corpus(todos_tokens, indice_pos, n=frases_necesarias)

    if len(pool_frases) < frases_necesarias:
        print(f"  ⚠ Solo se generaron {len(pool_frases):,} frases únicas "
              f"(pedidas: {frases_necesarias:,}). "
              f"Considerá ampliar el corpus o reducir TOTAL_IMAGES.")

    random.shuffle(pool_frases)
    print(f"  Pool total: {len(pool_frases):,} frases únicas disponibles")

    frases_por_fuente = {}
    for i, font_path in enumerate(fonts):
        inicio = i * imgs_por_fuente
        frases_por_fuente[font_path] = pool_frases[inicio: inicio + imgs_por_fuente]

    tasks = []
    for font_path in fonts:
        font_name = Path(font_path).stem
        for idx, texto in enumerate(frases_por_fuente[font_path]):
            tasks.append((font_path, font_name, idx, texto))

    out_path = Path(OUTPUT_DIR)
    out_path.mkdir(parents=True, exist_ok=True)

    bg_files = []
    bgp = Path(BACKGROUND_DIR)
    if bgp.exists():
        bg_files = (
            list(bgp.glob("*.png")) +
            list(bgp.glob("*.jpg")) +
            list(bgp.glob("*.jpeg"))
        )
    bg_files = [str(p) for p in bg_files]

    n_workers = N_WORKERS if N_WORKERS else max(1, cpu_count() - 1)
    print(f"\n[4/4] Generando imágenes — {n_workers} procesos paralelos")
    if bg_files:
        print(f"  Backgrounds disponibles: {len(bg_files)} archivos (precargados en RAM por worker)")
        print(f"  Estrategia: 60 % fondos externos + 40 % sintéticos")
    else:
        print(f"  Sin fondos externos — generación sintética 100 %")

    total_tasks = len(tasks)
    print_interval = max(1, total_tasks // 50)

    labels: dict   = {}
    stats          = defaultdict(int)
    errors_per_fnt = defaultdict(list)
    err_examples   = defaultdict(str)
    completed      = 0
    t0             = time.time()

    # Chunksize 25 → buen balance entre overhead IPC y load balancing
    with Pool(
        processes=n_workers,
        initializer=_worker_init,
        initargs=(bg_files, str(out_path)),
    ) as pool_workers:
        for result in pool_workers.imap_unordered(_generate_one, tasks, chunksize=25):
            font_name, img_name, texto, err = result
            completed += 1

            if err is None:
                labels[img_name] = texto
                stats[font_name] += 1
            else:
                errors_per_fnt[font_name].append(err)
                err_examples[font_name] = err

            if completed % print_interval == 0 or completed == total_tasks:
                elapsed = time.time() - t0
                rate    = completed / elapsed if elapsed > 0 else 0
                eta_s   = (total_tasks - completed) / rate if rate > 0 else 0
                eta_min = eta_s / 60
                print(f"  {completed:>7,} / {total_tasks:,}  "
                      f"({100*completed/total_tasks:5.1f}%)  "
                      f"{rate:>5.1f} img/s  "
                      f"ETA {eta_min:>5.1f} min")

    elapsed_total = time.time() - t0
    total_ok      = sum(stats.values())

    labels_path = out_path / "labels.json"
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)

    print("\n" + "═" * 70)
    print(f"  DATASET LISTO: {total_ok:,} imágenes en {elapsed_total/60:.1f} min "
          f"({total_ok/elapsed_total:.1f} img/s)")
    print(f"  Salida: {out_path.resolve()}")
    print("─" * 70)
    print("  Por fuente:")
    for fname, cnt in sorted(stats.items()):
        bar_w = (cnt // (imgs_por_fuente // 20)) if imgs_por_fuente >= 20 else 0
        bar   = "█" * bar_w
        err_n = len(errors_per_fnt.get(fname, []))
        err_s = f"  ({err_n} err)" if err_n else ""
        print(f"    {fname:<38} {cnt:>6,}  {bar}{err_s}")

    if errors_per_fnt:
        print("\n  Errores (1ª ocurrencia por fuente):")
        for fname, e in err_examples.items():
            print(f"    {fname}: {e}")

    counter = Counter(c for t in labels.values() for c in t if c != " ")
    total_chars = sum(counter.values())
    print(f"\n  Caracteres totales en labels: {total_chars:,}")
    print(f"  Caracteres distintos:         {len(counter)}")
    raros = [(c, counter.get(c, 0)) for c in "ñÑáéíóúÁÉÍÓÚüÜ¿¡—«»"]
    print("\n  Caracteres especiales del español:")
    for ch, cnt in raros:
        estado = "✓" if cnt >= 200 else "⚠ BAJO"
        print(f"    '{ch}'  {cnt:>6,} apariciones  {estado}")

    print("═" * 70)

if __name__ == "__main__":
    main()
