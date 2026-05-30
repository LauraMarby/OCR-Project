# OCR de manuscritos en español + recuperación de información

Pipeline completo para reconocimiento de texto manuscrito (HTR) sobre
documentos en español, desde la generación de datos sintéticos y el
preprocesado de imágenes hasta el entrenamiento de modelos OCR y la
evaluación del impacto del error de OCR sobre un sistema de recuperación
de información (SRI) híbrido.

El proyecto cubre dos familias de modelos OCR:

- **CRNN** propio (CNN ResNet-slim + BiLSTM x2 + CTC), entrenado desde cero.
- **TrOCR** (Vision Encoder-Decoder de Hugging Face), tanto el modelo base
  como una versión fine-tuneada para español manuscrito.

Sobre las transcripciones se monta un **SRI híbrido** (BM25 + E5 + RRF) y se
mide cuánto se degrada la recuperación al pasar del texto correcto al texto
reconocido por OCR ("arrastre de error").

## Estructura del repositorio

```
.
├── binarize/                 # Preprocesado y binarización de imágenes de línea
│   ├── preprocessing/        #   pipeline, binarización, normalización de líneas
│   ├── plotting.py           #   visualiza el pipeline etapa por etapa
│   └── visualize.py
├── generate_synthetic/       # Generación de dataset sintético de manuscritos
│   └── manuscript_generator.py
├── ocr_models/               # Modelos OCR
│   ├── hcr.py                #   CRNN: modelo, dataset, entrenador e inferencia
│   ├── crnn_lite_v2_train.py #   script de entrenamiento CRNN-Lite v2
│   ├── original_trocr.py     #   carga/predicción con TrOCR base
│   └── test_trocr_standalone.py  # tester CLI del TrOCR fine-tuneado
├── data_utils/               # Utilidades de preparación de datos
│   ├── remove_line.py        #   elimina renglones de libreta / líneas de color
│   └── rename.py             #   renombra claves .png → .jpg en los JSON de GT
├── evaluar_arrastre/         # Evaluación del arrastre de error OCR → SRI
│   ├── build_docs_trocr.py   #   transcribe segmentos a un .txt por documento
│   ├── eval_sri_on_docs.py   #   evalúa el SRI sobre los docs transcritos
│   ├── eval_error_arrastre_hybrid.py  # cuantifica la degradación GT vs OCR
│   ├── queries.txt, qrels.txt
│   └── docs/, resultados*/   #   entradas y salidas del experimento
├── standalone_cranfield_eval/  # Evaluación autocontenida del SRI sobre Cranfield
│   ├── sri.py                #   HybridRetriever (BM25 + E5 + RRF)
│   ├── dataset.py            #   carga vía ir_datasets
│   ├── metrics.py            #   P@k, R@k, MAP, MRR, nDCG@k, tests estadísticos
│   └── evaluate.py           #   CLI principal
├── notebooks/                # Experimentación y análisis
│   ├── hcr_work.ipynb        #   entrenamiento/uso del CRNN
│   ├── data_augmentation.ipynb
│   ├── data_analysis.ipynb
│   ├── splitting.ipynb       #   partición val/test del dataset
│   ├── compare.ipynb         #   comparación CRNN vs TrOCR
│   └── testing.ipynb
├── results/                  # Métricas e información de referencia
│   ├── results.txt, results2.txt
│   └── test_set_groundtruth.json
├── pagina_web/               # Código de la página web 
├── requirements.txt
└── README.md
```

## Instalación

Requiere Python 3.13 (probado en esa versión; debería funcionar en 3.10+).

```bash
python -m venv .venv
source .venv/bin/activate        # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Para GPU, instala primero `torch`/`torchvision` con la versión de CUDA
correspondiente siguiendo https://pytorch.org/get-started/ y luego el resto
del `requirements.txt`.

## Dependencia faltante: `preprocessing_utils`

`ocr_models/hcr.py` y los notebooks `hcr_work.ipynb` / `testing.ipynb`
importan un módulo `preprocessing_utils` (`preprocess_image`,
`preprocess_array`, `preprocess_image_binarized`, `resize_img`, ...) que
**no está incluido** en este repositorio. La lógica equivalente vive en
`binarize/preprocessing/`. Antes de ejecutar el CRNN hay que proporcionar
ese módulo, por ejemplo creando un `preprocessing_utils.py` que reexporte
las funciones del paquete `binarize/preprocessing`, o ajustando los imports.

## Flujo de trabajo típico

1. **Generar datos sintéticos** con `generate_synthetic/manuscript_generator.py`
   (combina corpus + tipografías manuscritas + fondos de papel).
2. **Preprocesar** las imágenes de línea con el pipeline de `binarize/`
   (binarización, normalización de altura, limpieza de renglones con
   `data_utils/remove_line.py`).
3. **Particionar** el dataset en validación y test con `notebooks/splitting.ipynb`.
4. **Entrenar** el CRNN (`ocr_models/hcr.py` / `crnn_lite_v2_train.py`) o
   usar/fine-tunear TrOCR.
5. **Comparar** modelos sobre el test real (`notebooks/compare.ipynb`); los
   resultados de referencia están en `results/`.
6. **Evaluar el SRI** de forma aislada sobre Cranfield (`standalone_cranfield_eval/`)
   y medir el **arrastre de error** OCR → recuperación (`evaluar_arrastre/`).

## Resultados de referencia

Sobre el test set real (706 muestras), el TrOCR fine-tuneado supera con
claridad tanto al CRNN como al TrOCR base:

| Modelo            | CER global | WER global |
|-------------------|:----------:|:----------:|
| CRNN              |   0.3365   |   0.7035   |
| TrOCR base        |   0.3447   |   0.7929   |
| TrOCR fine-tuneado|   0.0919   |   0.2375   |

Detalle completo en `results/results.txt` y `results/results2.txt`.

## Notas

- Los modelos entrenados (checkpoints `.pt`, `final_model.zip`, TrOCR
  fine-tuneado) y los datasets de imágenes no se incluyen por tamaño; las
  rutas se configuran al inicio de cada script/notebook.
- `standalone_cranfield_eval/` es independiente del resto: solo necesita
  `numpy`, `rank_bm25`, `ir_datasets` y, opcionalmente,
  `sentence_transformers` para la parte semántica.
