# ¿Se satura la predicción de conjunto de DETR en escenas aéreas densas?
Efecto del número de object queries bajo shift de dominio (COCO → VisDrone). Curso 1INF52 Deep Learning, PUCP, 2026-2 (Grupo 3).

## Contenido
- `notebooks/01_data_prep.ipynb`: VisDrone (Kaggle, YOLO) → COCO JSON, subconjuntos, clases, densidad y techo de recall.
- `notebooks/02_teardown.ipynb`: **Fase A**: parámetros/FLOPs por bloque, latencia/memoria, features intermedias, atención del decoder, campo receptivo (teórico vs efectivo), línea base en COCO y costo en función de N (H3).
- `notebooks/03_queries_ablation.ipynb`: **Fase B**: zero-shot de referencia, ablación del número de queries (100 vs 300, +500 si hay tiempo), recall por imagen vs techo min(1, N/n).
- `src/common.py`: conversión de datos, modelo con N queries, entrenamiento y evaluación (COCOeval con `maxDets` hasta 500).
- `DESIGN_LOG.md`: cuaderno de decisiones (decisiones, hipótesis previas, registro de corridas, uso de IA).
- `data/`: JSON en formato COCO generados por el notebook 01 (subconjunto de train y val completo).
- `results/`: tablas (CSV/JSON), figuras y detecciones (`dets_N100.json`, `dets_N300.json`) generadas por los notebooks.

## Reproducir (Kaggle)
1. **Notebook de Kaggle** por cada notebook del repo (File → Import Notebook con la URL del `.ipynb`). Opciones de sesión: Internet activado, *Pin to original environment*, acelerador **GPU T4 x2** para 02 y 03 (el 01 corre en CPU). No usar P100: el PyTorch de la imagen de Kaggle no trae kernels para esa GPU.
2. **Dataset:** Add Input → `banuprasadb/visdrone-dataset`. Se monta en `/kaggle/input/datasets/banuprasadb/visdrone-dataset/VisDrone_Dataset`, ruta ya fijada en los notebooks.
3. **Entorno:** la primera celda clona el repo (`BRANCH = 'rodrigo_dev'`, la rama usada en las corridas; cambiar a `main` tras la integración) e instala `requirements.txt`. `transformers` está fijado en 4.57.6 porque la versión 5 de la imagen de Kaggle rompe el código (D16). Versiones exactas: `results/entorno.txt`.
4. **Antes de entrenar:** las hipótesis H1 a H3 deben estar commiteadas en `DESIGN_LOG.md` (el notebook 03 imprime ese commit como evidencia).
5. **Orden de ejecución** (Save Version → Save & Run All; los resultados se descargan desde la pestaña *Output* de la versión guardada):
   | Notebook | Qué produce | Tiempo medido (T4) |
   |---|---|---|
   | 01 | `data/*.json`, distribución por clase, histograma y techo de recall | conversión en segundos (precarga en paralelo de 2296 archivos: 7 s) |
   | 02 | Fase A: tablas y figuras en `results/` | incluye descargar COCO val2017 (~1 GB) |
   | 03 | Zero-shot y ablación N = 100 y N = 300 | precarga 31 s; 39 a 44 s por época; 7,8 y 9,3 min por variante |
6. Semilla fija: 42 (`set_seed`). Dos ejecuciones del notebook 03 dieron pérdidas y métricas idénticas.

## Resultados del E2 (resumen)
- **Fase A:** 41,5 M de parámetros (56,5 % en el backbone) y 192 GFLOPs a 750 × 1333 (85,8 % en el backbone); campo receptivo teórico 427 px y efectivo cerca de 200 px; línea base en 500 imágenes de COCO: AP 45,6 (42,0 reportado). Detalle en `DESIGN_LOG.md`.
- **Fase B:** con N = 100 el techo de recall en val es 89,0 % (19,3 % de las imágenes tienen más de 100 objetos). Tras 8 épocas de fine-tuning, el recall queda muy por debajo del techo y pasar a N = 300 no lo cambia: H1 no se puede confirmar y H2 no se sostiene con este presupuesto de entrenamiento. El modelo ajustado rinde menos que el zero-shot y colapsa en las clases mayoritarias.

## Datos, modelos y licencias
- DETR: Carion et al., *End-to-end object detection with transformers*, ECCV 2020. Código oficial (facebookresearch/detr) bajo licencia Apache 2.0; pesos `facebook/detr-resnet-50` en Hugging Face.
- VisDrone-DET: Zhu et al., *Detection and tracking meet drones challenge*, IEEE TPAMI 2021. Uso académico/no comercial según los términos del repositorio oficial (VisDrone/VisDrone-Dataset). Se usó la copia en formato YOLO publicada en Kaggle (https://www.kaggle.com/datasets/banuprasadb/visdrone-dataset), distribuida bajo CC BY-NC-SA 3.0 IGO (https://creativecommons.org/licenses/by-nc-sa/3.0/igo/). Las anotaciones derivadas en `data/*.json` y las detecciones en `results/dets_*.json` se distribuyen bajo la misma licencia (ver `data/LICENSE.md`); las imágenes no se redistribuyen.
- COCO: Lin et al., ECCV 2014. Anotaciones bajo CC BY 4.0; imágenes sujetas a los términos de Flickr.
- Bibliotecas: PyTorch (BSD), Hugging Face Transformers (Apache 2.0), timm (Apache 2.0), pycocotools (BSD).

## Uso responsable de datos
VisDrone contiene imágenes aéreas urbanas con personas y vehículos. Se usa solo con fines académicos, sin intentar identificar individuos ni placas, y no se redistribuyen las imágenes (el repo solo contiene anotaciones derivadas y métricas). Posible sesgo: escenas de ciudades chinas, por lo que los resultados no se generalizan directamente a otros contextos urbanos (p. ej., Lima).

## Uso de asistentes de IA
Registrado decisión por decisión en `DESIGN_LOG.md`. Se usó un asistente de IA (Claude) como herramienta de trabajo: planificación, revisión del diseño experimental y código base (`src/common.py`, celdas de los notebooks). El equipo revisó, ejecutó y es responsable de cada decisión.
