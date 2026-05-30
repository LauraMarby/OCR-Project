import json
import sys
from pathlib import Path


def rename_png_keys_to_jpg(input_path: str, output_path: str = None):
    input_file = Path(input_path)

    if not input_file.exists():
        print(f"Error: No se encontró el archivo '{input_path}'")
        sys.exit(1)

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    renamed = {
        (key[:-4] + ".jpg" if key.endswith(".png") else key): value
        for key, value in data.items()
    }

    changed = sum(1 for k in data if k.endswith(".png"))

    if output_path is None:
        output_path = input_file.stem + "_renamed.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(renamed, f, ensure_ascii=False, indent=2)

    print(f"✓ {changed} clave(s) renombrada(s) de .png a .jpg")
    print(f"✓ Resultado guardado en: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python rename_png_to_jpg.py <archivo.json> [salida.json]")
        sys.exit(1)

    input_arg = sys.argv[1]
    output_arg = sys.argv[2] if len(sys.argv) >= 3 else None

    rename_png_keys_to_jpg(input_arg, output_arg)