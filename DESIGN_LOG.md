# Cuaderno de decisiones (design log)

Curso 1INF52 Deep Learning · 2026-2 · Grupo 3. Cada entrada: decisión, justificación, alternativas descartadas y **dónde se usó asistencia de IA**.
Regla del equipo: las hipótesis se escriben y se **commitean antes** de ejecutar cada ablación.

## Pregunta de investigación
¿Se satura la predicción de conjunto de DETR en escenas aéreas densas? Efecto del número de object queries (100 / 300 / 500) bajo shift de dominio (COCO → VisDrone).

## Decisiones

| # | Fecha | Decisión | Justificación | Alternativas descartadas | Uso de IA |
|---|---|---|---|---|---|
| D1 | 2026-10-07 | Modelo base: DETR-R50 (`facebook/detr-resnet-50`) | Línea base fiel y documentada; RF-DETR solo como referencia opcional en Fase C | Backbone swap a DINOv2 (se decidió no usar DINOv2) | Claude ayudó a comparar opciones; decisión del equipo |
| D2 | 2026-10-07 | Dataset: VisDrone-DET (copia de Kaggle) | Shift real respecto a COCO: vista aérea, objetos pequeños, alta densidad | EuroSAT / otros | Claude propuso la opción; el equipo la validó |
| D3 | 2026-10-07 | Variable de la ablación: número de queries N ∈ {100 (control), 300, 500} | Un solo cambio medible; toca el núcleo (decoder + emparejamiento húngaro) | Cambiar loss, patch size | Claude propuso; el equipo defiende la decisión |
| D4 | | Subconjunto: ___ imágenes train / ___ val; resolución lado corto 512 | Cómputo limitado (GPU gratuita) | Resolución completa (800) | |
| D5 | | Backbone congelado en E2 | Reduce memoria y tiempo; aísla el efecto del decoder | Fine-tuning completo | |
| D6 | | Inicialización de queries nuevas (N>100): copia de queries preentrenadas + ruido (0.1·std) | Evita partir de cero; **puede confundir el efecto de N** → el control de 100 usa el mismo protocolo | Inicialización aleatoria pura | |
| D7 | | Evaluación con `maxDets=[1,10,100,500]` y AR@500 | Con el límite por defecto (100) las variantes grandes quedarían truncadas | `maxDets` por defecto | |
| D8 | | Conversión YOLO→COCO con el tamaño real de cada imagen; se descartan cajas < 2 px | Tamaños variables; cajas inválidas rompen el procesador | | |
| D9 | | Semilla 42 en todas las corridas; una sola semilla | El enunciado no exige múltiples semillas | Varias semillas (suma, no se exige) | |

## Hipótesis (escribir y commitear ANTES de entrenar)
- **H1.** Con N = 100, el recall cae en escenas muy densas, porque no se pueden emitir más detecciones que queries. *Esperado:* AR@500 del grupo "high" claramente menor que en "low".
- **H2.** Subir N a 300–500 mejora sobre todo las escenas densas y casi nada las escasas; demasiadas queries pueden dificultar el entrenamiento.
- **H3.** El costo en FLOPs y latencia de más queries es pequeño (solo crece el decoder).

*Fecha y hora del commit de estas hipótesis:* ______

## Resultados vs. expectativas (completar tras ejecutar)
| Hipótesis | Qué pasó | ¿Se cumplió? | Por qué (explicación) |
|---|---|---|---|
| H1 | | | |
| H2 | | | |
| H3 | | | |

## Lo que no funcionó / limitaciones
- (entrenamiento corto en subconjunto: interpretar diferencias relativas; una sola semilla; resolución reducida puede confundir objetos pequeños)

## Preparación para la defensa oral (cada integrante)
1. ¿Por qué el recall de DETR tiene un techo igual a N por imagen?
2. ¿Qué hace el emparejamiento húngaro y qué pasa con las queries sin objeto?
3. ¿Por qué el control de 100 queries se entrena con el mismo protocolo que las variantes?
4. ¿Por qué se sube `maxDets` a 500 en la evaluación?
5. ¿Qué parte del costo de DETR está en el backbone, el encoder y el decoder (ver tablas de la Fase A)?
