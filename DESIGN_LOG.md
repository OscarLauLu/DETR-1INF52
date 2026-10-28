# Cuaderno de decisiones (design log)

Curso 1INF52 Deep Learning · 2026-2 · Grupo 3. Cada entrada: decisión, justificación, alternativas descartadas y **dónde se usó asistencia de IA**.
Regla del equipo: las hipótesis se escriben y se **commitean antes** de ejecutar cada ablación (el notebook 03 imprime ese commit como evidencia).

## Pregunta de investigación
¿Se satura la predicción de conjunto de DETR en escenas aéreas densas? Efecto del número de object queries (100 / 300 / 500) bajo shift de dominio (COCO → VisDrone).

## Decisiones

| # | Fecha | Decisión | Justificación | Alternativas descartadas | Uso de IA |
|---|---|---|---|---|---|
| D1 | 2026-10-07 | Modelo base: DETR-R50 (`facebook/detr-resnet-50`) | Línea base fiel y documentada; RF-DETR solo como referencia opcional en Fase C | Backbone swap a DINOv2 (se decidió no usar DINOv2) | Claude ayudó a comparar opciones; decisión del equipo |
| D2 | 2026-10-27 | VisDrone-DET, copia YOLO de Kaggle `banuprasadb/visdrone-dataset` (CC BY-NC-SA 3.0 IGO; 10 clases, orden verificado contra `visdrone.yaml`) | Shift real respecto a COCO: vista aérea, objetos pequeños, alta densidad | EuroSAT / otros | Claude propuso la opción; el equipo la validó |
| D3 | 2026-10-07 | Variable de la ablación: número de queries N ∈ {100 (control), 300, 500} | Un solo cambio medible; toca el núcleo (decoder + emparejamiento húngaro) | Cambiar loss, patch size | Claude propuso; el equipo defiende la decisión |
| D4 | 2026-10-27 | Subconjunto: **600** imágenes de train elegidas con `random.Random(42).sample` sobre la lista ordenada de archivos; se convierten solo esas (no se genera `train.json` completo) / **548** de val (completo); resolución lado corto 512 (lado largo ≤ 853). Resultado: 32 924 cajas en train (54,9 por imagen) y 38 732 en val (70,7 por imagen) | Cómputo limitado (GPU gratuita). La inferencia es barata, así que se usa todo val para tener más imágenes densas por grupo. Convertir solo el subconjunto evita leer las 6471 imágenes de train desde el dataset montado, cuya lectura en frío es lenta | Resolución completa (800); val de 200 | Claude sugirió usar val completo; el equipo lo aprobó |
| D5 | 2026-10-21 | Backbone congelado en E2 (solo se entrenan input_projection, encoder, decoder, queries y cabezas) | Reduce memoria y tiempo; aísla el efecto del decoder | Fine-tuning completo (posible en E3) | |
| D6 | 2026-10-21 | Inicialización de queries, **igual para todo N**: q_i = Q_pre[i mod 100] + 0.1·std(Q_pre)·ruido (`build_model`) | Evita partir de cero. Si N=100 conservara las queries limpias y N>100 se reiniciaran, se mezclaría el efecto de N con el de la inicialización → el control de 100 usa el mismo protocolo | Inicialización aleatoria pura; copiar solo para N>100 | Claude detectó el confusor y escribió la función; el equipo revisó |
| D7 | 2026-10-21 | Evaluación con `maxDets=[1,10,100,500]`; se reportan AP (@100, comparable con la literatura), AP@500, AR@100 y AR@500 | Con el límite por defecto (100) las variantes grandes quedarían truncadas | `maxDets` por defecto | Código de `coco_eval` escrito con Claude |
| D8 | 2026-10-21 | Conversión YOLO→COCO con el tamaño real de cada imagen; cajas recortadas al borde; se descartan cajas con ancho o alto < 2 px. Resultado: 27 cajas descartadas en val (de 38 759) y 1 en el subconjunto de train; 0 clases inválidas; 0 imágenes sin etiqueta | Tamaños variables; cajas degeneradas rompen el procesador | | Código escrito con Claude, probado con datos sintéticos |
| D9 | 2026-10-21 | Semilla 42 en todas las corridas; una sola semilla | El enunciado no exige múltiples semillas | Varias semillas (suma, no se exige) | |
| D10 | 2026-10-21 | Una detección por query (clase argmax sin "no object"), umbral de score 0 | Respeta el diseño de DETR: una imagen nunca recibe más de N detecciones, así que el techo min(1, N/n) se mide de verdad | Top-k sobre queries × clases (estilo Deformable DETR), que esquivaría el techo | Claude señaló el riesgo |
| D11 | 2026-10-21 | Zero-shot COCO → VisDrone solo en 6 clases con equivalente: person→pedestrian, bicycle, car, motorcycle→motor, bus, truck | van, tricycle, awning-tricycle y people no existen en COCO; evaluarlas penalizaría al modelo por clases que no conoce. Las 6 clases mapeadas cubren el 77,6 % de las cajas de val; people (13,2 %) es la clase sin equivalente con más cajas | Evaluar las 10 clases | Mapeo propuesto por Claude; revisar ambigüedad person/people |
| D12 | 2026-10-21 | Grupos de densidad por terciles de objetos/imagen (low/medium/high) + grupo `gt100` (n > 100) | Los terciles dan grupos balanceados; `gt100` aísla justo las imágenes donde N=100 tiene techo activo | Cortes fijos arbitrarios | |
| D13 | 2026-10-21 | Entrenamiento: AdamW, lr 1e-4, weight decay 1e-4, gradient clipping 0.1 (como DETR), 8 épocas, batch 2, fp32, sin aumentación | Hiperparámetros del artículo original; mismo protocolo para todo N | Mixed precision (riesgo de inestabilidad en el matching) | |
| D14 | 2026-10-21 | Verificar el orden de clases del `data.yaml` de la copia de Kaggle antes de convertir. Resultado (2026-10-27): `visdrone.yaml` declara `nc: 10` en el orden de `VISDRONE_CLASSES`; las etiquetas de val solo contienen los índices 0 a 9 | El mapeo D11 depende del orden de clases | Asumir el orden estándar | |
| D15 | 2026-10-21 | Campo receptivo teórico de referencia: **427 px** (no 483) | timm `resnet50` es ResNet v1.5 (stride en la conv 3×3); 483 px corresponde a ResNet v1 (stride en la 1×1). Verificado con r ← r + (k−1)·j | — | Claude hizo el cálculo; verificado en el notebook 02 (A6) |
| D16 | 2026-10-27 | Transformers fijado en 4.57.6 (`requirements.txt`) | La imagen de Kaggle trae 5.16.1, que rompe `processor.pad`, `backbone.conv_encoder` y `output_attentions` (SDPA por defecto); con 4.57.6 el código corre sin cambios | Portar el código a la API 5.x | Claude detectó la incompatibilidad en una prueba local; el equipo fijó la versión |
| D17 | 2026-10-27 | Prioridad de variantes: N=100 y N=300; N=500 solo si sobra cómputo | En val, el techo de recall es 89,0 % con N=100, 99,9 % con N=300 y 100 % con N=500: N=500 no cambia el techo, solo probaría el efecto de exceso de queries (H2) | Correr las tres variantes | Claude propuso la prioridad a partir del techo medido; el equipo decidió |

## Hipótesis (escribir y commitear ANTES de entrenar)
- **H1.** Con N = 100, el recall cae en escenas muy densas, porque no se pueden emitir más detecciones que queries. *Esperado:* AR@500 del grupo "high" (y sobre todo `gt100`) claramente menor que en "low"; en la figura recall vs techo, los puntos de N=100 con n > 100 quedan en o bajo la curva min(1, 100/n).
  *Dato previo (notebook 01):* 106 de 548 imágenes de val (19,3 %) tienen más de 100 objetos (máximo: 317); techo de recall del conjunto con N=100: 89,0 %. *Criterio:* si el recall medido de N=100 en `gt100` queda muy por debajo de min(1, 100/n), la restricción activa no es el número de queries sino la detección de objetos pequeños, y H1 no se puede confirmar con este experimento.
- **H2.** Subir N a 300–500 mejora sobre todo las escenas densas y casi nada las escasas; demasiadas queries pueden dificultar el entrenamiento. *Esperado:* ΔAR@500 (300 − 100) mayor en `high`/`gt100` que en `low`; con el mismo número de épocas, la pérdida de N=300 baja más lento.
- **H3.** El costo en FLOPs y latencia de más queries es pequeño (solo crece el decoder). *Esperado:* ΔGFLOPs (500 vs 100) de pocos puntos porcentuales; el backbone domina el costo.

*Fecha y hora del commit de estas hipótesis:* ______ (copiar de `git log -1 -- DESIGN_LOG.md`)

## Registro de corridas
| Corrida | Fecha | N | GPU | Minutos | Loss final | AR@500 overall | AR@500 `gt100` | AP overall | Notas |
|---|---|---|---|---|---|---|---|---|---|
| Zero-shot (COCO, 6 clases) | | 100 | | — | — | | | | punto de partida Fase C |
| B-100 | | 100 | | | | | | | control |
| B-300 | | 300 | | | | | | | |
| B-500 | | 500 | | | | | | | si hay tiempo |

## Resultados Fase A (resumen)
| Medición | Valor | Archivo |
|---|---|---|
| Parámetros totales / % backbone / % decoder | | `results/params_por_bloque.csv` |
| GFLOPs totales / % backbone / % encoder / % decoder | | `results/flops_por_bloque.csv` |
| Latencia (ms) y memoria pico (MiB), entrada | | `results/latencia_memoria.csv` |
| Campo receptivo teórico vs efectivo | 427 px vs ___ px | `results/campo_receptivo.png` |
| AP línea base COCO (500 imgs) vs 42.0 reportado | | `results/linea_base_coco.csv` |
| ΔGFLOPs y Δlatencia N=500 vs N=100 | | `results/costo_vs_N.csv` |

## Resultados vs. expectativas (completar tras ejecutar)
| Hipótesis | Qué pasó | ¿Se cumplió? | Por qué (explicación) |
|---|---|---|---|
| H1 | | | |
| H2 | | | |
| H3 | | | |

## Lo que no funcionó / limitaciones
- Entrenamiento corto (8 épocas) en un subconjunto de 600 imágenes: interpretar diferencias relativas, no el AP absoluto.
- Una sola semilla: diferencias pequeñas entre variantes pueden ser ruido.
- Resolución reducida (512): perjudica justo a los objetos pequeños; puede confundir el efecto de N con el de la resolución.
- La copia YOLO de Kaggle ya no incluye las "ignored regions" ni las marcas de oclusión/truncamiento de VisDrone: los detectores pueden ser penalizados por detectar objetos en zonas ignoradas en la anotación original.
- Zero-shot solo cubre 6 de 10 clases; person→pedestrian es ambiguo (VisDrone separa pedestrian y people).
- Val es más denso que el subconjunto de train (70,7 frente a 54,9 objetos por imagen en promedio). La proporción de imágenes de train con n > 100 aún no se midió; es necesaria para interpretar H2.
- La primera lectura de cada archivo del dataset montado en Kaggle es lenta: en una sesión nueva, la primera época tarda más que las siguientes. Para comparar tiempos entre variantes se usan las épocas 2 en adelante.
- (agregar aquí lo que falle al ejecutar: OOM, divergencia, tiempos, etc.)

## Preparación para la defensa oral (cada integrante)
1. **¿Por qué el recall de DETR tiene un techo igual a N por imagen?** Cada query produce exactamente una predicción y el emparejamiento húngaro es 1 a 1; con n > N objetos, al menos n − N quedan sin pareja → recall ≤ N/n.
2. **¿Qué hace el emparejamiento húngaro y qué pasa con las queries sin objeto?** Encuentra la asignación query↔objeto de costo mínimo (clase + L1 + GIoU); las queries no asignadas se entrenan para predecir "no object" (con peso 0.1 en la pérdida de clase).
3. **¿Por qué el control de 100 queries se entrena con el mismo protocolo que las variantes?** Para que lo único distinto sea N: misma inicialización con ruido (D6), mismas épocas, resolución y semilla.
4. **¿Por qué se sube `maxDets` a 500 en la evaluación?** COCOeval corta en 100 detecciones por imagen por defecto; sin el cambio, N=300/500 no podrían mostrar su ventaja.
5. **¿Qué parte del costo de DETR está en el backbone, el encoder y el decoder?** Ver tablas A1/A2. Clave: el decoder escala con N (self-attention N×N y cross-attention N×HW), pero el backbone domina los FLOPs.
6. **¿Por qué el campo receptivo es 427 y no 483 px?** Ver D15.
