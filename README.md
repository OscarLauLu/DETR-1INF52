# ¿Se satura la predicción de conjunto de DETR en escenas aéreas densas?
Efecto del número de object queries bajo shift de dominio (COCO → VisDrone). Curso 1INF52 Deep Learning, PUCP, 2026-2 (Grupo 3).

## Contenido
- `notebooks/01_data_prep.ipynb`: VisDrone (Kaggle, YOLO) → COCO JSON, subconjuntos, densidad y techo de recall.
- `notebooks/02_teardown.ipynb`: Fase A: parámetros/FLOPs por bloque, latencia/memoria, features, atención, campo receptivo, línea base en COCO.
- `notebooks/03_queries_ablation.ipynb`: Fase B: ablación del número de queries (100 vs 300, +500 si hay tiempo).
- `src/common.py`: conversión de datos, modelo con N queries, entrenamiento y evaluación (COCOeval con `maxDets` hasta 500).
- `DESIGN_LOG.md`: cuaderno de decisiones (decisiones, hipótesis previas, uso de IA).
- `results/`: tablas y figuras generadas.

## Reproducir
1. GPU (Colab/Kaggle). `pip install -r requirements.txt`.
2. Ejecutar los notebooks en orden; ajustar las rutas marcadas con `<...>`.
3. Semilla fija: 42. Entorno: ver `requirements.txt` (anotar versiones exactas de `torch` y `transformers` usadas).

## Datos, modelos y licencias
- DETR: Carion et al., ECCV 2020 (pesos `facebook/detr-resnet-50`, Hugging Face).
- VisDrone: Zhu et al., IEEE TPAMI (copia en Kaggle; uso académico, respetar términos del repositorio oficial).
- COCO: Lin et al., ECCV 2014.
- Uso responsable: imágenes aéreas con personas y vehículos; solo uso académico, sin identificar individuos.

## Uso de asistentes de IA
Registrado en `DESIGN_LOG.md`. Se usó un asistente de IA como herramienta de trabajo (planificación y código base); el equipo revisa y es responsable de cada decisión.
