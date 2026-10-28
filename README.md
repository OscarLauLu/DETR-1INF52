# ¿Se satura la predicción de conjunto de DETR en escenas aéreas densas?
Efecto del número de object queries bajo shift de dominio (COCO → VisDrone). Curso 1INF52 Deep Learning, PUCP, 2026-2 (Grupo 3).

## Contenido
- `notebooks/01_data_prep.ipynb`: VisDrone (Kaggle, YOLO) → COCO JSON, subconjuntos, clases, densidad y techo de recall.
- `notebooks/02_teardown.ipynb`: **Fase A**: parámetros/FLOPs por bloque, latencia/memoria, features intermedias, atención del decoder, campo receptivo (teórico vs efectivo), línea base en COCO y costo en función de N (H3).
- `notebooks/03_queries_ablation.ipynb`: **Fase B**: zero-shot de referencia, ablación del número de queries (100 vs 300, +500 si hay tiempo), recall por imagen vs techo min(1, N/n).
- `src/common.py`: conversión de datos, modelo con N queries, entrenamiento y evaluación (COCOeval con `maxDets` hasta 500).
- `DESIGN_LOG.md`: cuaderno de decisiones (decisiones, hipótesis previas, registro de corridas, uso de IA).
- `data/`: JSON en formato COCO generados por el notebook 01 (se versionan los pequeños).
- `results/`: tablas (CSV/JSON) y figuras generadas.

## Reproducir
1. GPU (Colab o Kaggle, T4 o superior). La primera celda de cada notebook clona el repo e instala `requirements.txt`; solo hay que ajustar `REPO_URL` y las rutas del dataset (`<...>`).
2. **Antes de entrenar:** commitear `DESIGN_LOG.md` con las hipótesis H1–H3.
3. Ejecutar en orden:
   | Notebook | Qué produce | Tiempo aprox. (T4) |
   |---|---|---|
   | 01 | `data/*.json`, histograma, techo de recall | minutos (lee solo cabeceras de imagen) |
   | 02 | tablas y figuras de la Fase A | ~15–20 min (incluye descargar COCO val2017, ~1 GB) |
   | 03 | zero-shot + ablación N=100 y N=300 | depende de la GPU: medir la 1.ª época y anotarla en el log |
4. Semilla fija: 42 (`set_seed`). Entorno exacto: `results/entorno.txt` (lo genera la primera celda de cada notebook).

## Datos, modelos y licencias
- DETR: Carion et al., *End-to-end object detection with transformers*, ECCV 2020. Código oficial (facebookresearch/detr) bajo licencia Apache 2.0; pesos `facebook/detr-resnet-50` en Hugging Face.
- VisDrone-DET: Zhu et al., *Detection and tracking meet drones challenge*, IEEE TPAMI 2021. Uso académico/no comercial según los términos del repositorio oficial (VisDrone/VisDrone-Dataset). Se usó una copia de Kaggle en formato YOLO: `<enlace a la copia de Kaggle>` (verificar su licencia en la página del dataset).
- COCO: Lin et al., ECCV 2014. Anotaciones bajo CC BY 4.0; imágenes sujetas a los términos de Flickr.
- Bibliotecas: PyTorch (BSD), Hugging Face Transformers (Apache 2.0), timm (Apache 2.0), pycocotools (BSD).

## Uso responsable de datos
VisDrone contiene imágenes aéreas urbanas con personas y vehículos. Se usa solo con fines académicos, sin intentar identificar individuos ni placas, y no se redistribuyen las imágenes (el repo solo contiene anotaciones derivadas y métricas). Posible sesgo: escenas de ciudades chinas, por lo que los resultados no se generalizan directamente a otros contextos urbanos (p. ej., Lima).

## Uso de asistentes de IA
Registrado decisión por decisión en `DESIGN_LOG.md`. Se usó un asistente de IA (Claude) como herramienta de trabajo: planificación, revisión del diseño experimental y código base (`src/common.py`, celdas de los notebooks). El equipo revisó, ejecutó y es responsable de cada decisión.
