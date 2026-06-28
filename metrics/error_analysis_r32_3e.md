# Reporte de análisis de errores — LoRA r=32 / alpha=64 / 3 epochs

> Reporte histórico: esta evaluación fue producida con los artefactos previos de Fase 1 sobre 350 filas. El dataset canónico actual es `entrenamiento.jsonl` con 317 filas; reentrenar/evaluar antes de comparar métricas actuales.

**Modelo evaluado:** `Qwen/Qwen2.5-7B-Instruct` + LoRA `qwen-protesta-v1-r32` (rank=32, alpha=64, alpha/r=2.0, 3 epochs).
**Adapter:** `checkpoints/qwen-protesta-v1-r32` (sha1 `6804aeb4d7f85b7d1b94574b1cab816017debbf7`).
**Inferencia:** vLLM 0.23.0, `enable_lora=True`, `max_loras=1`, `SamplingParams(structured_outputs=StructuredOutputsParams(json=cleaned_schema))` contra `esquema_eventos_protesta_entrenamiento_MVS.json`.
**Eval set histórico:** 35 ejemplos del split previo de `data/chat_formatted/eval.jsonl` (gold = GPT-5.4-mini + validación humana de Nico, weight 1.0; `metadatos_extraccion.estado_validacion_humana: "No validado"` se ignora porque las 350 filas históricas del training data estaban validadas).

## A. Resumen ejecutivo

- **Mejor modelo histórico dentro de la corrida 350-era:** r=32 / alpha=64 / 3 epochs (`checkpoints/qwen-protesta-v1-r32`). Es el único modelo que cumple PLAN §6 schema_validity ≥ 0.95 y supera simultáneamente al baseline y al r16 en todas las métricas de contenido; r32 5e y r32 4e fueron controlados contra este y rinden peor.
- **Lo que funciona:** schema válido al 100% (35/35), parse válido al 100%, `field_recall.non_empty` cruza 0.7496 (vs 0.1692 del baseline), y la detección del booleano `tiene_eventos_protesta` salta de 0.2857 → 0.7714. El modelo ya no alucina sistemáticamente `nota_id="S/D"` ni fechas con `day=19`.
- **Lo que falla:** las **categorías/enum** siguen por debajo del 0.80 del PLAN §6 (aggregate 0.4189; mejor path individual: `extraccion.tiene_eventos_protesta` 0.7714, peor: `contra_quien[].nivel_institucional` 0.1856). El **f1_global** (0.5350) y la **f1_recall** (0.5752) están por debajo del 0.70 del criterio MVP, y persisten **7 falsos positivos** en `tiene_eventos_protesta` (notas sin protesta real a las que el modelo les inventa 1-4 eventos).
- **Conclusión:** el límite no es hiperparámetro. r=16 rindió 0.4637 y r=32/e=5 rindió 0.5002 (más epochs **empeoran**: recall sube pero precision cae, síntoma clásico de sobreajuste al codebook sin nueva evidencia). Iterar Fase 6 con más epochs/rank no cierra la brecha. El plan de anotación dirigida (sección F) ataca los errores estructurales observados.

## B. Métricas globales — comparación de runs

Baseline = Qwen2.5-7B-Instruct sin fine-tuning. r16_3e y r32_3e son LoRA rank 16 / alpha 32 y rank 32 / alpha 64 sobre el mismo training set. r32_e5 = mismo checkpoint que r32_3e extendido a 5 epochs (ver `reports/phase6_r32_e5_eval.json`).

| métrica | baseline | r16_3e | **r32_3e (mejor)** | r32_e5 | Δ r32_3e vs baseline |
|---|---:|---:|---:|---:|---:|
| schema_validity | 1.0000 | 1.0000 | 1.0000 | 1.0000 | +0.0000 |
| f1_global (f1) | 0.0971 | 0.4637 | 0.5350 | 0.5002 | +0.4379 |
| f1_precision | 0.2155 | 0.4664 | 0.5001 | 0.4857 | +0.2846 |
| f1_recall | 0.0627 | 0.4609 | 0.5752 | 0.5156 | +0.5125 |
| tiene_eventos_protesta (acc) | 0.2857 | 0.7143 | 0.7714 | 0.7429 | +0.4857 |
| categorical_accuracy aggregate | 0.0384 | 0.3400 | 0.4189 | 0.3728 | +0.3805 |
| field_recall exact | 0.0540 | 0.4134 | 0.5165 | 0.4631 | +0.4625 |
| field_recall non_empty | 0.1692 | 0.6335 | 0.7496 | 0.6848 | +0.5804 |

Δ = r32_3e − baseline. El **mejor modelo es r32_3e** en todas las métricas; r32_e5 (5 epochs) cae respecto a r32_3e en f1, categorical y field recall — más epochs sobreajustan sin aportar nueva evidencia.

### B.1 Confusión `extraccion.tiene_eventos_protesta` (r32_3e)

|              | pred=False | pred=True |
|---|---:|---:|
| **gold=True (27)**  | 1 FN | 26 TP |
| **gold=False (8)**  | 1 TN | 7 FP |

Accuracy: **0.7714** sobre 35 notas. El error dominante es **FP** (modelo inventa protesta en 7 notas que el gold marca como no-protesta). FN = 1 sólo: el modelo casi nunca deja de detectar una protesta real.

## C. Taxonomía de errores

### C.1 Falsos positivos / falsos negativos en `tiene_eventos_protesta`

**Falsos positivos (gold=False, pred=True)** — el modelo inventa protesta:

| nota_id | gold_total | pred_total | f1_vs_gold |
|---|---:|---:|---:|
| `IMG_20240913_094108_1989-06-14_011_nota.txt` | 0 | 1 | 0.0833 |
| `IMG_20240916_103907_1989-09-13_004_nota.txt` | 0 | 1 | 0.0833 |
| `IMG_20241031_113558_1990-05-19_013_nota.txt` | 0 | 2 | 0.0296 |
| `IMG_20241107_105320_1990_09_03_012_nota.txt` | 0 | 1 | 0.0556 |
| `IMG_20241108_102859_1990-10-01_006_nota.txt` | 0 | 1 | 0.0833 |
| `IMG_20241115_112301_1991-01-06_007_nota.txt` | 0 | 1 | 0.0833 |
| `IMG_20241115_113003_1991-01-09_011_nota.txt` | 0 | 4 | 0.0231 |

**Falsos negativos (gold=True, pred=False)** — el modelo se pierde un evento:

| nota_id | gold_total | pred_total | f1_vs_gold |
|---|---:|---:|---:|
| `IMG_20251020_102437_1994_12_05_p1_010_nota.txt` | 1 | 0 | 0.0833 |

### C.2 Errores de conteo de eventos (`total_eventos_protesta`)

Casos con mayor discrepancia |gold − pred| (top 10). Las notas con pred_total > gold_total suelen ser **FP de eventos** dentro de notas que sí tienen protesta (el modelo fragmenta un evento único en varios). Las notas con pred_total < gold_total son **FN de eventos** (sub-segmentación).

| nota_id | gold_total | pred_total | delta | tipo | f1 |
|---|---:|---:|---:|---|---:|
| `IMG_20241115_113003_1991-01-09_011_nota.txt` | 0 | 4 | +4 | extra_event | 0.0231 |
| `IMG_20241031_113558_1990-05-19_013_nota.txt` | 0 | 2 | +2 | extra_event | 0.0296 |
| `IMG_20241121_105644_1991-03-08_004_nota.txt` | 6 | 4 | -2 | missing_event | 0.5571 |
| `IMG_20250919_112322_1992-08-19_p2_015_nota.txt` | 2 | 4 | +2 | extra_event | 0.4242 |
| `IMG_20240906_105503_1989-03-05_016_nota.txt` | 2 | 1 | -1 | missing_event | 0.4688 |
| `IMG_20240913_094108_1989-06-14_011_nota.txt` | 0 | 1 | +1 | extra_event | 0.0833 |
| `IMG_20240916_103907_1989-09-13_004_nota.txt` | 0 | 1 | +1 | extra_event | 0.0833 |
| `IMG_20240919_120800_1989-10-25_006_nota.txt` | 3 | 2 | -1 | missing_event | 0.4742 |
| `IMG_20240926_115717_1989-11-28_009_nota.txt` | 2 | 1 | -1 | missing_event | 0.5000 |
| `IMG_20241004_103347_1990-02-14_012_nota.txt` | 2 | 1 | -1 | missing_event | 0.3900 |

### C.3 Paths categóricos con mayor tasa de error

Tabla histórica ordenada por error_rate (1 − accuracy) descendente. Soporte = total de comparaciones alineadas por índice sobre los 35 ejemplos del split previo.

| path | tp | tn | fp | fn | support | accuracy | error_rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| `contra_quien[].nivel_institucional` | 18 | 0 | 40 | 39 | 97 | 0.1856 | 0.8144 |
| `delimitacion.criterio_delimitacion` | 17 | 0 | 39 | 30 | 86 | 0.1977 | 0.8023 |
| `demandas[].categoria` | 21 | 0 | 44 | 35 | 100 | 0.2100 | 0.7900 |
| `accion.formato_principal.categoria` | 18 | 0 | 38 | 29 | 85 | 0.2118 | 0.7882 |
| `contra_quien[].categoria` | 25 | 0 | 33 | 32 | 90 | 0.2778 | 0.7222 |
| `lugares[].categoria` | 24 | 0 | 33 | 25 | 82 | 0.2927 | 0.7073 |
| `sujetos[].organizaciones[].categoria` | 38 | 0 | 43 | 37 | 118 | 0.3220 | 0.6780 |
| `lugares[].rol_en_evento` | 26 | 0 | 31 | 23 | 80 | 0.3250 | 0.6750 |
| `temporalidad.tipo_temporal` | 26 | 0 | 30 | 21 | 77 | 0.3377 | 0.6623 |
| `temporalidad.tempo_verbal` | 27 | 0 | 29 | 20 | 76 | 0.3553 | 0.6447 |
| `alcance.categoria` | 27 | 0 | 29 | 20 | 76 | 0.3553 | 0.6447 |
| `sujetos[].categoria` | 30 | 0 | 30 | 20 | 80 | 0.3750 | 0.6250 |
| `delimitacion.es_accion_principal_con_complementarias` | 1 | 28 | 27 | 18 | 74 | 0.3919 | 0.6081 |
| `evento.es_evento_protesta` | 30 | 4 | 22 | 13 | 69 | 0.4928 | 0.5072 |
| `cantidad_participantes.hay_cantidad_mencionada` | 37 | 0 | 19 | 10 | 66 | 0.5606 | 0.4394 |
| `cantidad_participantes.es_aproximada` | 37 | 0 | 19 | 10 | 66 | 0.5606 | 0.4394 |
| `incidentes.enfrentamiento.presencia` | 39 | 0 | 17 | 8 | 64 | 0.6094 | 0.3906 |
| `temporalidad.fecha_inicio.certeza` | 40 | 0 | 16 | 7 | 63 | 0.6349 | 0.3651 |
| `temporalidad.fecha_fin.certeza` | 40 | 0 | 16 | 7 | 63 | 0.6349 | 0.3651 |
| `incidentes.represion.presencia` | 40 | 0 | 16 | 7 | 63 | 0.6349 | 0.3651 |
| `incidentes.detenidos.presencia` | 40 | 0 | 16 | 7 | 63 | 0.6349 | 0.3651 |
| `incidentes.heridos.presencia` | 40 | 0 | 16 | 7 | 63 | 0.6349 | 0.3651 |
| `incidentes.muertos.presencia` | 40 | 0 | 16 | 7 | 63 | 0.6349 | 0.3651 |
| `incidentes.danios_materiales.presencia` | 40 | 0 | 16 | 7 | 63 | 0.6349 | 0.3651 |

## D. Tablas comparativas gold vs pred

Para cada path categórico priorizado por codebook/enums se listan las 10 comparaciones gold-vs-pred con mayor divergencia (mismatch o missing/extra). `tipo_error` ∈ {`match`, `mismatch`, `missing`, `extra`}.

### D.1 `extraccion.tiene_eventos_protesta`

| nota_id | correcto_gold | prediccion_modelo | tipo_error | por_que_importa | accion_recomendada |
|---|---|---|---|---|---|
| `IMG_20240913_094108_1989-06-14_011_nota.txt` | `false` | `true` | mismatch | Modelo inventa protesta en nota sin protesta real (FP). | Anotar más notas gold=False con conceptos conflict-adjacent (paro, huelga, amenaza) que NO constituyan protesta para enseñar el borde. |
| `IMG_20240916_103907_1989-09-13_004_nota.txt` | `false` | `true` | mismatch | Modelo inventa protesta en nota sin protesta real (FP). | Anotar más notas gold=False con conceptos conflict-adjacent (paro, huelga, amenaza) que NO constituyan protesta para enseñar el borde. |
| `IMG_20241031_113558_1990-05-19_013_nota.txt` | `false` | `true` | mismatch | Modelo inventa protesta en nota sin protesta real (FP). | Anotar más notas gold=False con conceptos conflict-adjacent (paro, huelga, amenaza) que NO constituyan protesta para enseñar el borde. |
| `IMG_20241107_105320_1990_09_03_012_nota.txt` | `false` | `true` | mismatch | Modelo inventa protesta en nota sin protesta real (FP). | Anotar más notas gold=False con conceptos conflict-adjacent (paro, huelga, amenaza) que NO constituyan protesta para enseñar el borde. |
| `IMG_20241108_102859_1990-10-01_006_nota.txt` | `false` | `true` | mismatch | Modelo inventa protesta en nota sin protesta real (FP). | Anotar más notas gold=False con conceptos conflict-adjacent (paro, huelga, amenaza) que NO constituyan protesta para enseñar el borde. |
| `IMG_20241115_112301_1991-01-06_007_nota.txt` | `false` | `true` | mismatch | Modelo inventa protesta en nota sin protesta real (FP). | Anotar más notas gold=False con conceptos conflict-adjacent (paro, huelga, amenaza) que NO constituyan protesta para enseñar el borde. |
| `IMG_20241115_113003_1991-01-09_011_nota.txt` | `false` | `true` | mismatch | Modelo inventa protesta en nota sin protesta real (FP). | Anotar más notas gold=False con conceptos conflict-adjacent (paro, huelga, amenaza) que NO constituyan protesta para enseñar el borde. |
| `IMG_20251020_102437_1994_12_05_p1_010_nota.txt` | `true` | `false` | mismatch | Modelo pierde una protesta real (FN). | Anotar más notas gold=True con eventos sutiles (asamblea, estado de alerta, reclamo administrativo) como positivos débiles. |

### D.2 `extraccion.total_eventos_protesta`

| nota_id | correcto_gold | prediccion_modelo | tipo_error | por_que_importa | accion_recomendada |
|---|---|---|---|---|---|
| `IMG_20240906_105503_1989-03-05_016_nota.txt` | `2` | `1` | mismatch | Conteo de eventos difiere del gold. | Anotar ejemplos donde múltiples acciones compartan delimitación (acción principal + complementarias) para enseñar la regla 'no fragmentes'. |
| `IMG_20240913_094108_1989-06-14_011_nota.txt` | `0` | `1` | mismatch | Conteo de eventos difiere del gold. | Anotar ejemplos donde múltiples acciones compartan delimitación (acción principal + complementarias) para enseñar la regla 'no fragmentes'. |
| `IMG_20240916_103907_1989-09-13_004_nota.txt` | `0` | `1` | mismatch | Conteo de eventos difiere del gold. | Anotar ejemplos donde múltiples acciones compartan delimitación (acción principal + complementarias) para enseñar la regla 'no fragmentes'. |
| `IMG_20240919_120800_1989-10-25_006_nota.txt` | `3` | `2` | mismatch | Conteo de eventos difiere del gold. | Anotar ejemplos donde múltiples acciones compartan delimitación (acción principal + complementarias) para enseñar la regla 'no fragmentes'. |
| `IMG_20240926_115717_1989-11-28_009_nota.txt` | `2` | `1` | mismatch | Conteo de eventos difiere del gold. | Anotar ejemplos donde múltiples acciones compartan delimitación (acción principal + complementarias) para enseñar la regla 'no fragmentes'. |
| `IMG_20241004_103347_1990-02-14_012_nota.txt` | `2` | `1` | mismatch | Conteo de eventos difiere del gold. | Anotar ejemplos donde múltiples acciones compartan delimitación (acción principal + complementarias) para enseñar la regla 'no fragmentes'. |
| `IMG_20241031_113558_1990-05-19_013_nota.txt` | `0` | `2` | mismatch | Conteo de eventos difiere del gold. | Anotar ejemplos donde múltiples acciones compartan delimitación (acción principal + complementarias) para enseñar la regla 'no fragmentes'. |
| `IMG_20241107_105320_1990_09_03_012_nota.txt` | `0` | `1` | mismatch | Conteo de eventos difiere del gold. | Anotar ejemplos donde múltiples acciones compartan delimitación (acción principal + complementarias) para enseñar la regla 'no fragmentes'. |
| `IMG_20241108_102859_1990-10-01_006_nota.txt` | `0` | `1` | mismatch | Conteo de eventos difiere del gold. | Anotar ejemplos donde múltiples acciones compartan delimitación (acción principal + complementarias) para enseñar la regla 'no fragmentes'. |
| `IMG_20241115_112301_1991-01-06_007_nota.txt` | `0` | `1` | mismatch | Conteo de eventos difiere del gold. | Anotar ejemplos donde múltiples acciones compartan delimitación (acción principal + complementarias) para enseñar la regla 'no fragmentes'. |

### D.3 `eventos_protesta[].temporalidad.tipo_temporal`

| nota_id | slot | correcto_gold | prediccion_modelo | tipo_error | por_que_importa | accion_recomendada |
|---|---:|---|---|---|---|---|
| `IMG_20240906_105926_1989-03-08_009_nota.txt` | 1 | `Anuncio` | `Hecho` | mismatch | Hecho vs Anuncio vs S/D cambia el tratamiento del evento (¿ocurrió o se anunció?). | Anotar ejemplos donde el verbo esté en futuro o haya cita tipo 'se resolvió/convocó' para reforzar Hecho vs Anuncio. |
| `IMG_20240906_105926_1989-03-08_009_nota.txt` | 2 | `Hecho` | `Anuncio` | mismatch | Hecho vs Anuncio vs S/D cambia el tratamiento del evento (¿ocurrió o se anunció?). | Anotar ejemplos donde el verbo esté en futuro o haya cita tipo 'se resolvió/convocó' para reforzar Hecho vs Anuncio. |
| `IMG_20240916_095514_1989-08-19_008_nota.txt` | 1 | `Hecho` | `Anuncio` | mismatch | Hecho vs Anuncio vs S/D cambia el tratamiento del evento (¿ocurrió o se anunció?). | Anotar ejemplos donde el verbo esté en futuro o haya cita tipo 'se resolvió/convocó' para reforzar Hecho vs Anuncio. |
| `IMG_20240919_120800_1989-10-25_006_nota.txt` | 1 | `Hecho` | `Anuncio` | mismatch | Hecho vs Anuncio vs S/D cambia el tratamiento del evento (¿ocurrió o se anunció?). | Anotar ejemplos donde el verbo esté en futuro o haya cita tipo 'se resolvió/convocó' para reforzar Hecho vs Anuncio. |
| `IMG_20241004_113909_1990-03-17_012_nota.txt` | 2 | `Hecho` | `Anuncio` | mismatch | Hecho vs Anuncio vs S/D cambia el tratamiento del evento (¿ocurrió o se anunció?). | Anotar ejemplos donde el verbo esté en futuro o haya cita tipo 'se resolvió/convocó' para reforzar Hecho vs Anuncio. |
| `IMG_20241004_113909_1990-03-17_012_nota.txt` | 3 | `Hecho` | `Anuncio` | mismatch | Hecho vs Anuncio vs S/D cambia el tratamiento del evento (¿ocurrió o se anunció?). | Anotar ejemplos donde el verbo esté en futuro o haya cita tipo 'se resolvió/convocó' para reforzar Hecho vs Anuncio. |
| `IMG_20241108_115422_1990-11-26_011_nota.txt` | 0 | `Hecho` | `S/D` | mismatch | Hecho vs Anuncio vs S/D cambia el tratamiento del evento (¿ocurrió o se anunció?). | Anotar ejemplos donde el verbo esté en futuro o haya cita tipo 'se resolvió/convocó' para reforzar Hecho vs Anuncio. |
| `IMG_20250919_112322_1992-08-19_p2_015_nota.txt` | 0 | `Hecho` | `Anuncio` | mismatch | Hecho vs Anuncio vs S/D cambia el tratamiento del evento (¿ocurrió o se anunció?). | Anotar ejemplos donde el verbo esté en futuro o haya cita tipo 'se resolvió/convocó' para reforzar Hecho vs Anuncio. |
| `IMG_20250919_112322_1992-08-19_p2_015_nota.txt` | 1 | `Hecho` | `Anuncio` | mismatch | Hecho vs Anuncio vs S/D cambia el tratamiento del evento (¿ocurrió o se anunció?). | Anotar ejemplos donde el verbo esté en futuro o haya cita tipo 'se resolvió/convocó' para reforzar Hecho vs Anuncio. |
| `IMG_20251006_102342_1994_01_08_p2_005_nota.txt` | 0 | `S/D` | `Hecho` | mismatch | Hecho vs Anuncio vs S/D cambia el tratamiento del evento (¿ocurrió o se anunció?). | Anotar ejemplos donde el verbo esté en futuro o haya cita tipo 'se resolvió/convocó' para reforzar Hecho vs Anuncio. |

### D.4 `eventos_protesta[].temporalidad.tempo_verbal`

| nota_id | slot | correcto_gold | prediccion_modelo | tipo_error | por_que_importa | accion_recomendada |
|---|---:|---|---|---|---|---|
| `IMG_20240906_105926_1989-03-08_009_nota.txt` | 1 | `Futuro` | `Pasado` | mismatch | Pasado/Presente/Futuro/S/D es ortogonal a tipo_temporal; mal clasificado afecta análisis de series temporales. | Anotar ejemplos con verbos en presente de noticias ('se declara', 'continúa') para enseñar Presente vs Pasado en notas del día. |
| `IMG_20240906_105926_1989-03-08_009_nota.txt` | 2 | `Pasado` | `Futuro` | mismatch | Pasado/Presente/Futuro/S/D es ortogonal a tipo_temporal; mal clasificado afecta análisis de series temporales. | Anotar ejemplos con verbos en presente de noticias ('se declara', 'continúa') para enseñar Presente vs Pasado en notas del día. |
| `IMG_20240919_120800_1989-10-25_006_nota.txt` | 1 | `Pasado` | `Futuro` | mismatch | Pasado/Presente/Futuro/S/D es ortogonal a tipo_temporal; mal clasificado afecta análisis de series temporales. | Anotar ejemplos con verbos en presente de noticias ('se declara', 'continúa') para enseñar Presente vs Pasado en notas del día. |
| `IMG_20241004_113909_1990-03-17_012_nota.txt` | 2 | `Pasado` | `Futuro` | mismatch | Pasado/Presente/Futuro/S/D es ortogonal a tipo_temporal; mal clasificado afecta análisis de series temporales. | Anotar ejemplos con verbos en presente de noticias ('se declara', 'continúa') para enseñar Presente vs Pasado en notas del día. |
| `IMG_20241004_113909_1990-03-17_012_nota.txt` | 3 | `Pasado` | `Futuro` | mismatch | Pasado/Presente/Futuro/S/D es ortogonal a tipo_temporal; mal clasificado afecta análisis de series temporales. | Anotar ejemplos con verbos en presente de noticias ('se declara', 'continúa') para enseñar Presente vs Pasado en notas del día. |
| `IMG_20241108_115422_1990-11-26_011_nota.txt` | 0 | `Pasado` | `S/D` | mismatch | Pasado/Presente/Futuro/S/D es ortogonal a tipo_temporal; mal clasificado afecta análisis de series temporales. | Anotar ejemplos con verbos en presente de noticias ('se declara', 'continúa') para enseñar Presente vs Pasado en notas del día. |
| `IMG_20250919_112322_1992-08-19_p2_015_nota.txt` | 0 | `Pasado` | `Futuro` | mismatch | Pasado/Presente/Futuro/S/D es ortogonal a tipo_temporal; mal clasificado afecta análisis de series temporales. | Anotar ejemplos con verbos en presente de noticias ('se declara', 'continúa') para enseñar Presente vs Pasado en notas del día. |
| `IMG_20250919_112322_1992-08-19_p2_015_nota.txt` | 1 | `Pasado` | `Futuro` | mismatch | Pasado/Presente/Futuro/S/D es ortogonal a tipo_temporal; mal clasificado afecta análisis de series temporales. | Anotar ejemplos con verbos en presente de noticias ('se declara', 'continúa') para enseñar Presente vs Pasado en notas del día. |
| `IMG_20251006_102342_1994_01_08_p2_005_nota.txt` | 0 | `S/D` | `Pasado` | mismatch | Pasado/Presente/Futuro/S/D es ortogonal a tipo_temporal; mal clasificado afecta análisis de series temporales. | Anotar ejemplos con verbos en presente de noticias ('se declara', 'continúa') para enseñar Presente vs Pasado en notas del día. |
| `IMG_20251009_104152_1994_05_05_p1_010_nota.txt` | 0 | `S/D` | `Pasado` | mismatch | Pasado/Presente/Futuro/S/D es ortogonal a tipo_temporal; mal clasificado afecta análisis de series temporales. | Anotar ejemplos con verbos en presente de noticias ('se declara', 'continúa') para enseñar Presente vs Pasado en notas del día. |

### D.5 `eventos_protesta[].delimitacion_evento.criterio_delimitacion`

| nota_id | slot | correcto_gold | prediccion_modelo | tipo_error | por_que_importa | accion_recomendada |
|---|---:|---|---|---|---|---|
| `IMG_20240906_105503_1989-03-05_016_nota.txt` | 0 | `Accion principal con acciones complementarias` | `Evento unico en la nota` | mismatch | Criterio de delimitación es la regla que evita fragmentar eventos; alto FP/FN aquí explica parte del drift en total_eventos. | Anotar ejemplos con 'acción principal + complementarias' vs 'temporal' vs 'evento único' para reforzar la distinción del codebook §3. |
| `IMG_20240913_093817_1989-06-12_013_nota.txt` | 0 | `Evento unico en la nota` | `S/D` | mismatch | Criterio de delimitación es la regla que evita fragmentar eventos; alto FP/FN aquí explica parte del drift en total_eventos. | Anotar ejemplos con 'acción principal + complementarias' vs 'temporal' vs 'evento único' para reforzar la distinción del codebook §3. |
| `IMG_20240916_095514_1989-08-19_008_nota.txt` | 0 | `Temporal y espacial` | `Evento unico en la nota` | mismatch | Criterio de delimitación es la regla que evita fragmentar eventos; alto FP/FN aquí explica parte del drift en total_eventos. | Anotar ejemplos con 'acción principal + complementarias' vs 'temporal' vs 'evento único' para reforzar la distinción del codebook §3. |
| `IMG_20240916_095514_1989-08-19_008_nota.txt` | 1 | `S/D` | `Evento unico en la nota` | mismatch | Criterio de delimitación es la regla que evita fragmentar eventos; alto FP/FN aquí explica parte del drift en total_eventos. | Anotar ejemplos con 'acción principal + complementarias' vs 'temporal' vs 'evento único' para reforzar la distinción del codebook §3. |
| `IMG_20240919_120800_1989-10-25_006_nota.txt` | 0 | `Accion principal con acciones complementarias` | `Temporal` | mismatch | Criterio de delimitación es la regla que evita fragmentar eventos; alto FP/FN aquí explica parte del drift en total_eventos. | Anotar ejemplos con 'acción principal + complementarias' vs 'temporal' vs 'evento único' para reforzar la distinción del codebook §3. |
| `IMG_20240919_120800_1989-10-25_006_nota.txt` | 1 | `Temporal` | `S/D` | mismatch | Criterio de delimitación es la regla que evita fragmentar eventos; alto FP/FN aquí explica parte del drift en total_eventos. | Anotar ejemplos con 'acción principal + complementarias' vs 'temporal' vs 'evento único' para reforzar la distinción del codebook §3. |
| `IMG_20240926_115717_1989-11-28_009_nota.txt` | 0 | `Accion principal con acciones complementarias` | `Temporal` | mismatch | Criterio de delimitación es la regla que evita fragmentar eventos; alto FP/FN aquí explica parte del drift en total_eventos. | Anotar ejemplos con 'acción principal + complementarias' vs 'temporal' vs 'evento único' para reforzar la distinción del codebook §3. |
| `IMG_20241004_103347_1990-02-14_012_nota.txt` | 0 | `Temporal` | `Accion principal con acciones complementarias` | mismatch | Criterio de delimitación es la regla que evita fragmentar eventos; alto FP/FN aquí explica parte del drift en total_eventos. | Anotar ejemplos con 'acción principal + complementarias' vs 'temporal' vs 'evento único' para reforzar la distinción del codebook §3. |
| `IMG_20241004_113909_1990-03-17_012_nota.txt` | 1 | `Evento unico en la nota` | `Temporal` | mismatch | Criterio de delimitación es la regla que evita fragmentar eventos; alto FP/FN aquí explica parte del drift en total_eventos. | Anotar ejemplos con 'acción principal + complementarias' vs 'temporal' vs 'evento único' para reforzar la distinción del codebook §3. |
| `IMG_20241004_113909_1990-03-17_012_nota.txt` | 2 | `Evento unico en la nota` | `Temporal` | mismatch | Criterio de delimitación es la regla que evita fragmentar eventos; alto FP/FN aquí explica parte del drift en total_eventos. | Anotar ejemplos con 'acción principal + complementarias' vs 'temporal' vs 'evento único' para reforzar la distinción del codebook §3. |

### D.6 `eventos_protesta[].accion.formato_principal.categoria`

| nota_id | slot | correcto_gold | prediccion_modelo | tipo_error | por_que_importa | accion_recomendada |
|---|---:|---|---|---|---|---|
| `IMG_20240906_105503_1989-03-05_016_nota.txt` | 0 | `Manifestaciones de baja intensidad` | `Asambleas` | mismatch | Categoría de formato principal es el primer nivel taxonómico de la acción. | Anotar ejemplos de cada categoría (Huelgas / Cortes / Asambleas / Manifestaciones / Manifestaciones de baja intensidad / Ocupaciones / Ataques / Acciones judiciales / Reuniones / Residuales) para balancear. |
| `IMG_20240906_105926_1989-03-08_009_nota.txt` | 1 | `Asambleas` | `Manifestaciones de baja intensidad` | mismatch | Categoría de formato principal es el primer nivel taxonómico de la acción. | Anotar ejemplos de cada categoría (Huelgas / Cortes / Asambleas / Manifestaciones / Manifestaciones de baja intensidad / Ocupaciones / Ataques / Acciones judiciales / Reuniones / Residuales) para balancear. |
| `IMG_20240906_105926_1989-03-08_009_nota.txt` | 2 | `Huelgas` | `Asambleas` | mismatch | Categoría de formato principal es el primer nivel taxonómico de la acción. | Anotar ejemplos de cada categoría (Huelgas / Cortes / Asambleas / Manifestaciones / Manifestaciones de baja intensidad / Ocupaciones / Ataques / Acciones judiciales / Reuniones / Residuales) para balancear. |
| `IMG_20240916_095514_1989-08-19_008_nota.txt` | 0 | `Manifestaciones` | `Manifestaciones de baja intensidad` | mismatch | Categoría de formato principal es el primer nivel taxonómico de la acción. | Anotar ejemplos de cada categoría (Huelgas / Cortes / Asambleas / Manifestaciones / Manifestaciones de baja intensidad / Ocupaciones / Ataques / Acciones judiciales / Reuniones / Residuales) para balancear. |
| `IMG_20240919_120800_1989-10-25_006_nota.txt` | 0 | `Asambleas` | `Huelgas` | mismatch | Categoría de formato principal es el primer nivel taxonómico de la acción. | Anotar ejemplos de cada categoría (Huelgas / Cortes / Asambleas / Manifestaciones / Manifestaciones de baja intensidad / Ocupaciones / Ataques / Acciones judiciales / Reuniones / Residuales) para balancear. |
| `IMG_20240919_120800_1989-10-25_006_nota.txt` | 1 | `Manifestaciones de baja intensidad` | `S/D` | mismatch | Categoría de formato principal es el primer nivel taxonómico de la acción. | Anotar ejemplos de cada categoría (Huelgas / Cortes / Asambleas / Manifestaciones / Manifestaciones de baja intensidad / Ocupaciones / Ataques / Acciones judiciales / Reuniones / Residuales) para balancear. |
| `IMG_20241004_103347_1990-02-14_012_nota.txt` | 0 | `Ataques` | `Manifestaciones de baja intensidad` | mismatch | Categoría de formato principal es el primer nivel taxonómico de la acción. | Anotar ejemplos de cada categoría (Huelgas / Cortes / Asambleas / Manifestaciones / Manifestaciones de baja intensidad / Ocupaciones / Ataques / Acciones judiciales / Reuniones / Residuales) para balancear. |
| `IMG_20241004_113909_1990-03-17_012_nota.txt` | 0 | `Huelgas` | `Asambleas` | mismatch | Categoría de formato principal es el primer nivel taxonómico de la acción. | Anotar ejemplos de cada categoría (Huelgas / Cortes / Asambleas / Manifestaciones / Manifestaciones de baja intensidad / Ocupaciones / Ataques / Acciones judiciales / Reuniones / Residuales) para balancear. |
| `IMG_20241004_113909_1990-03-17_012_nota.txt` | 2 | `Reuniones entre las partes litigantes` | `Asambleas` | mismatch | Categoría de formato principal es el primer nivel taxonómico de la acción. | Anotar ejemplos de cada categoría (Huelgas / Cortes / Asambleas / Manifestaciones / Manifestaciones de baja intensidad / Ocupaciones / Ataques / Acciones judiciales / Reuniones / Residuales) para balancear. |
| `IMG_20241004_113909_1990-03-17_012_nota.txt` | 3 | `Manifestaciones de baja intensidad` | `Huelgas` | mismatch | Categoría de formato principal es el primer nivel taxonómico de la acción. | Anotar ejemplos de cada categoría (Huelgas / Cortes / Asambleas / Manifestaciones / Manifestaciones de baja intensidad / Ocupaciones / Ataques / Acciones judiciales / Reuniones / Residuales) para balancear. |

### D.7 `eventos_protesta[].sujetos[].categoria`

| nota_id | slot | correcto_gold | prediccion_modelo | tipo_error | por_que_importa | accion_recomendada |
|---|---:|---|---|---|---|---|
| `IMG_20240913_093817_1989-06-12_013_nota.txt` | 0 | `Asalariados` | `Vecinos` | mismatch | Categoría del sujeto es ambigua en notas con sujetos colectivos (sindicato vs asalariados vs militantes). | Anotar ejemplos donde el sujeto sea un sindicato con personería gremial (Asalariados), una organización política (Militantes) o un grupo vecinal (Vecinos). |
| `IMG_20240916_095514_1989-08-19_008_nota.txt` | 1 | `Residual` | `Empresarios / Gerentes / Directivos` | mismatch | Categoría del sujeto es ambigua en notas con sujetos colectivos (sindicato vs asalariados vs militantes). | Anotar ejemplos donde el sujeto sea un sindicato con personería gremial (Asalariados), una organización política (Militantes) o un grupo vecinal (Vecinos). |
| `IMG_20240919_120800_1989-10-25_006_nota.txt` | 1 | `Asalariados` | `S/D` | mismatch | Categoría del sujeto es ambigua en notas con sujetos colectivos (sindicato vs asalariados vs militantes). | Anotar ejemplos donde el sujeto sea un sindicato con personería gremial (Asalariados), una organización política (Militantes) o un grupo vecinal (Vecinos). |
| `IMG_20241004_103347_1990-02-14_012_nota.txt` | 0 | `Profesionales` | `Residual` | mismatch | Categoría del sujeto es ambigua en notas con sujetos colectivos (sindicato vs asalariados vs militantes). | Anotar ejemplos donde el sujeto sea un sindicato con personería gremial (Asalariados), una organización política (Militantes) o un grupo vecinal (Vecinos). |
| `IMG_20241004_103347_1990-02-14_012_nota.txt` | 1 | `Profesionales` | `Residual` | mismatch | Categoría del sujeto es ambigua en notas con sujetos colectivos (sindicato vs asalariados vs militantes). | Anotar ejemplos donde el sujeto sea un sindicato con personería gremial (Asalariados), una organización política (Militantes) o un grupo vecinal (Vecinos). |
| `IMG_20241108_115422_1990-11-26_011_nota.txt` | 0 | `Asalariados` | `S/D` | mismatch | Categoría del sujeto es ambigua en notas con sujetos colectivos (sindicato vs asalariados vs militantes). | Anotar ejemplos donde el sujeto sea un sindicato con personería gremial (Asalariados), una organización política (Militantes) o un grupo vecinal (Vecinos). |
| `IMG_20241121_105644_1991-03-08_004_nota.txt` | 2 | `Militantes` | `Asalariados` | mismatch | Categoría del sujeto es ambigua en notas con sujetos colectivos (sindicato vs asalariados vs militantes). | Anotar ejemplos donde el sujeto sea un sindicato con personería gremial (Asalariados), una organización política (Militantes) o un grupo vecinal (Vecinos). |
| `IMG_20250918_110753_1992-05-21_p2_008_nota.txt` | 0 | `Profesionales` | `Asalariados` | mismatch | Categoría del sujeto es ambigua en notas con sujetos colectivos (sindicato vs asalariados vs militantes). | Anotar ejemplos donde el sujeto sea un sindicato con personería gremial (Asalariados), una organización política (Militantes) o un grupo vecinal (Vecinos). |
| `IMG_20251006_102342_1994_01_08_p2_005_nota.txt` | 0 | `S/D` | `Asalariados` | mismatch | Categoría del sujeto es ambigua en notas con sujetos colectivos (sindicato vs asalariados vs militantes). | Anotar ejemplos donde el sujeto sea un sindicato con personería gremial (Asalariados), una organización política (Militantes) o un grupo vecinal (Vecinos). |
| `IMG_20251009_104152_1994_05_05_p1_010_nota.txt` | 0 | `S/D` | `Asalariados` | mismatch | Categoría del sujeto es ambigua en notas con sujetos colectivos (sindicato vs asalariados vs militantes). | Anotar ejemplos donde el sujeto sea un sindicato con personería gremial (Asalariados), una organización política (Militantes) o un grupo vecinal (Vecinos). |

### D.8 `eventos_protesta[].demandas[].categoria`

| nota_id | slot | correcto_gold | prediccion_modelo | tipo_error | por_que_importa | accion_recomendada |
|---|---:|---|---|---|---|---|
| `IMG_20240906_105926_1989-03-08_009_nota.txt` | 1 | `Residual` | `S/D` | mismatch | Categorías de demanda (Salarial / Laboral / Política / Gremial / Económica / Seguridad / etc.) tienen solapamiento semántico. | Anotar ejemplos donde la demanda sea salarial vs laboral vs gremial (intra-sindical) para reforzar la distinción. |
| `IMG_20240906_105926_1989-03-08_009_nota.txt` | 2 | `Salarial` | `S/D` | mismatch | Categorías de demanda (Salarial / Laboral / Política / Gremial / Económica / Seguridad / etc.) tienen solapamiento semántico. | Anotar ejemplos donde la demanda sea salarial vs laboral vs gremial (intra-sindical) para reforzar la distinción. |
| `IMG_20240916_095514_1989-08-19_008_nota.txt` | 0 | `Económica` | `Política` | mismatch | Categorías de demanda (Salarial / Laboral / Política / Gremial / Económica / Seguridad / etc.) tienen solapamiento semántico. | Anotar ejemplos donde la demanda sea salarial vs laboral vs gremial (intra-sindical) para reforzar la distinción. |
| `IMG_20240916_095514_1989-08-19_008_nota.txt` | 1 | `Económica` | `Política` | mismatch | Categorías de demanda (Salarial / Laboral / Política / Gremial / Económica / Seguridad / etc.) tienen solapamiento semántico. | Anotar ejemplos donde la demanda sea salarial vs laboral vs gremial (intra-sindical) para reforzar la distinción. |
| `IMG_20240919_120800_1989-10-25_006_nota.txt` | 0 | `Laboral` | `Salarial` | mismatch | Categorías de demanda (Salarial / Laboral / Política / Gremial / Económica / Seguridad / etc.) tienen solapamiento semántico. | Anotar ejemplos donde la demanda sea salarial vs laboral vs gremial (intra-sindical) para reforzar la distinción. |
| `IMG_20240919_120800_1989-10-25_006_nota.txt` | 1 | `Salarial` | `S/D` | mismatch | Categorías de demanda (Salarial / Laboral / Política / Gremial / Económica / Seguridad / etc.) tienen solapamiento semántico. | Anotar ejemplos donde la demanda sea salarial vs laboral vs gremial (intra-sindical) para reforzar la distinción. |
| `IMG_20241004_113909_1990-03-17_012_nota.txt` | 1 | `Laboral` | `Salarial` | mismatch | Categorías de demanda (Salarial / Laboral / Política / Gremial / Económica / Seguridad / etc.) tienen solapamiento semántico. | Anotar ejemplos donde la demanda sea salarial vs laboral vs gremial (intra-sindical) para reforzar la distinción. |
| `IMG_20241004_113909_1990-03-17_012_nota.txt` | 2 | `Residual` | `Salarial` | mismatch | Categorías de demanda (Salarial / Laboral / Política / Gremial / Económica / Seguridad / etc.) tienen solapamiento semántico. | Anotar ejemplos donde la demanda sea salarial vs laboral vs gremial (intra-sindical) para reforzar la distinción. |
| `IMG_20241107_113507_1990_09_26_004_nota.txt` | 0 | `Gremial (Inter-intra sindical)` | `S/D` | mismatch | Categorías de demanda (Salarial / Laboral / Política / Gremial / Económica / Seguridad / etc.) tienen solapamiento semántico. | Anotar ejemplos donde la demanda sea salarial vs laboral vs gremial (intra-sindical) para reforzar la distinción. |
| `IMG_20241108_115422_1990-11-26_011_nota.txt` | 0 | `Política` | `S/D` | mismatch | Categorías de demanda (Salarial / Laboral / Política / Gremial / Económica / Seguridad / etc.) tienen solapamiento semántico. | Anotar ejemplos donde la demanda sea salarial vs laboral vs gremial (intra-sindical) para reforzar la distinción. |

### D.9 `eventos_protesta[].contra_quien[].categoria`

| nota_id | slot | correcto_gold | prediccion_modelo | tipo_error | por_que_importa | accion_recomendada |
|---|---:|---|---|---|---|---|
| `IMG_20240906_105503_1989-03-05_016_nota.txt` | 0 | `Estado/Gobierno` | `S/D` | mismatch | Destinatario 'Sindicatos' vs 'Estado/Gobierno' vs 'Patronal' es estructuralmente ambiguo en paros. | Anotar ejemplos donde el 'contra_quien' sea otro sindicato (paros de solidaridad), el Estado como empleador, o la patronal privada. |
| `IMG_20240906_105926_1989-03-08_009_nota.txt` | 1 | `Estado/Gobierno` | `S/D` | mismatch | Destinatario 'Sindicatos' vs 'Estado/Gobierno' vs 'Patronal' es estructuralmente ambiguo en paros. | Anotar ejemplos donde el 'contra_quien' sea otro sindicato (paros de solidaridad), el Estado como empleador, o la patronal privada. |
| `IMG_20240906_105926_1989-03-08_009_nota.txt` | 2 | `Estado/Gobierno` | `S/D` | mismatch | Destinatario 'Sindicatos' vs 'Estado/Gobierno' vs 'Patronal' es estructuralmente ambiguo en paros. | Anotar ejemplos donde el 'contra_quien' sea otro sindicato (paros de solidaridad), el Estado como empleador, o la patronal privada. |
| `IMG_20240919_120800_1989-10-25_006_nota.txt` | 1 | `Estado/Gobierno` | `S/D` | mismatch | Destinatario 'Sindicatos' vs 'Estado/Gobierno' vs 'Patronal' es estructuralmente ambiguo en paros. | Anotar ejemplos donde el 'contra_quien' sea otro sindicato (paros de solidaridad), el Estado como empleador, o la patronal privada. |
| `IMG_20241004_103347_1990-02-14_012_nota.txt` | 0 | `Sindicatos` | `Residual` | mismatch | Destinatario 'Sindicatos' vs 'Estado/Gobierno' vs 'Patronal' es estructuralmente ambiguo en paros. | Anotar ejemplos donde el 'contra_quien' sea otro sindicato (paros de solidaridad), el Estado como empleador, o la patronal privada. |
| `IMG_20241004_113909_1990-03-17_012_nota.txt` | 2 | `S/D` | `Estado/Gobierno` | mismatch | Destinatario 'Sindicatos' vs 'Estado/Gobierno' vs 'Patronal' es estructuralmente ambiguo en paros. | Anotar ejemplos donde el 'contra_quien' sea otro sindicato (paros de solidaridad), el Estado como empleador, o la patronal privada. |
| `IMG_20241107_113507_1990_09_26_004_nota.txt` | 0 | `Sindicatos` | `S/D` | mismatch | Destinatario 'Sindicatos' vs 'Estado/Gobierno' vs 'Patronal' es estructuralmente ambiguo en paros. | Anotar ejemplos donde el 'contra_quien' sea otro sindicato (paros de solidaridad), el Estado como empleador, o la patronal privada. |
| `IMG_20241108_115422_1990-11-26_011_nota.txt` | 0 | `Residual` | `S/D` | mismatch | Destinatario 'Sindicatos' vs 'Estado/Gobierno' vs 'Patronal' es estructuralmente ambiguo en paros. | Anotar ejemplos donde el 'contra_quien' sea otro sindicato (paros de solidaridad), el Estado como empleador, o la patronal privada. |
| `IMG_20241121_105644_1991-03-08_004_nota.txt` | 0 | `Estado/Gobierno` | `S/D` | mismatch | Destinatario 'Sindicatos' vs 'Estado/Gobierno' vs 'Patronal' es estructuralmente ambiguo en paros. | Anotar ejemplos donde el 'contra_quien' sea otro sindicato (paros de solidaridad), el Estado como empleador, o la patronal privada. |
| `IMG_20241121_105644_1991-03-08_004_nota.txt` | 1 | `Estado/Gobierno` | `S/D` | mismatch | Destinatario 'Sindicatos' vs 'Estado/Gobierno' vs 'Patronal' es estructuralmente ambiguo en paros. | Anotar ejemplos donde el 'contra_quien' sea otro sindicato (paros de solidaridad), el Estado como empleador, o la patronal privada. |

### D.10 `eventos_protesta[].contra_quien[].nivel_institucional`

| nota_id | slot | correcto_gold | prediccion_modelo | tipo_error | por_que_importa | accion_recomendada |
|---|---:|---|---|---|---|---|
| `IMG_20240906_105503_1989-03-05_016_nota.txt` | 0 | `Nacional` | `S/D` | mismatch | Nivel institucional (Municipal/Provincial/Nacional/Internacional/Privado/No aplica) es el path con peor accuracy (0.1856). | Anotar explícitamente el nivel cuando el destinatario sea el Estado (Nacional vs Provincial vs Municipal) o una empresa privada (Privado). |
| `IMG_20240906_105926_1989-03-08_009_nota.txt` | 1 | `Provincial` | `S/D` | mismatch | Nivel institucional (Municipal/Provincial/Nacional/Internacional/Privado/No aplica) es el path con peor accuracy (0.1856). | Anotar explícitamente el nivel cuando el destinatario sea el Estado (Nacional vs Provincial vs Municipal) o una empresa privada (Privado). |
| `IMG_20240906_105926_1989-03-08_009_nota.txt` | 2 | `Nacional` | `S/D` | mismatch | Nivel institucional (Municipal/Provincial/Nacional/Internacional/Privado/No aplica) es el path con peor accuracy (0.1856). | Anotar explícitamente el nivel cuando el destinatario sea el Estado (Nacional vs Provincial vs Municipal) o una empresa privada (Privado). |
| `IMG_20240916_095514_1989-08-19_008_nota.txt` | 1 | `Nacional` | `Provincial` | mismatch | Nivel institucional (Municipal/Provincial/Nacional/Internacional/Privado/No aplica) es el path con peor accuracy (0.1856). | Anotar explícitamente el nivel cuando el destinatario sea el Estado (Nacional vs Provincial vs Municipal) o una empresa privada (Privado). |
| `IMG_20240919_120800_1989-10-25_006_nota.txt` | 0 | `Municipal` | `Provincial` | mismatch | Nivel institucional (Municipal/Provincial/Nacional/Internacional/Privado/No aplica) es el path con peor accuracy (0.1856). | Anotar explícitamente el nivel cuando el destinatario sea el Estado (Nacional vs Provincial vs Municipal) o una empresa privada (Privado). |
| `IMG_20240919_120800_1989-10-25_006_nota.txt` | 1 | `Provincial` | `S/D` | mismatch | Nivel institucional (Municipal/Provincial/Nacional/Internacional/Privado/No aplica) es el path con peor accuracy (0.1856). | Anotar explícitamente el nivel cuando el destinatario sea el Estado (Nacional vs Provincial vs Municipal) o una empresa privada (Privado). |
| `IMG_20241004_103347_1990-02-14_012_nota.txt` | 0 | `S/D` | `No aplica` | mismatch | Nivel institucional (Municipal/Provincial/Nacional/Internacional/Privado/No aplica) es el path con peor accuracy (0.1856). | Anotar explícitamente el nivel cuando el destinatario sea el Estado (Nacional vs Provincial vs Municipal) o una empresa privada (Privado). |
| `IMG_20241004_113909_1990-03-17_012_nota.txt` | 2 | `No aplica` | `Provincial` | mismatch | Nivel institucional (Municipal/Provincial/Nacional/Internacional/Privado/No aplica) es el path con peor accuracy (0.1856). | Anotar explícitamente el nivel cuando el destinatario sea el Estado (Nacional vs Provincial vs Municipal) o una empresa privada (Privado). |
| `IMG_20241108_115422_1990-11-26_011_nota.txt` | 0 | `No aplica` | `S/D` | mismatch | Nivel institucional (Municipal/Provincial/Nacional/Internacional/Privado/No aplica) es el path con peor accuracy (0.1856). | Anotar explícitamente el nivel cuando el destinatario sea el Estado (Nacional vs Provincial vs Municipal) o una empresa privada (Privado). |
| `IMG_20241121_105644_1991-03-08_004_nota.txt` | 0 | `Nacional` | `S/D` | mismatch | Nivel institucional (Municipal/Provincial/Nacional/Internacional/Privado/No aplica) es el path con peor accuracy (0.1856). | Anotar explícitamente el nivel cuando el destinatario sea el Estado (Nacional vs Provincial vs Municipal) o una empresa privada (Privado). |

### D.11 `eventos_protesta[].lugares[].categoria`

| nota_id | slot | correcto_gold | prediccion_modelo | tipo_error | por_que_importa | accion_recomendada |
|---|---:|---|---|---|---|---|
| `IMG_20240906_105503_1989-03-05_016_nota.txt` | 0 | `S/D` | `Lugar de Trabajo` | mismatch | Categoría de lugar (Vía pública / Instituciones públicas / Sede patronal / Lugar de Trabajo / Sede sindical) se confunde en notas con múltiples espacios. | Anotar ejemplos donde el evento ocurra en sede sindical, patronal o en la vía pública para reforzar la distinción. |
| `IMG_20240906_105926_1989-03-08_009_nota.txt` | 0 | `Lugar de Trabajo` | `S/D` | mismatch | Categoría de lugar (Vía pública / Instituciones públicas / Sede patronal / Lugar de Trabajo / Sede sindical) se confunde en notas con múltiples espacios. | Anotar ejemplos donde el evento ocurra en sede sindical, patronal o en la vía pública para reforzar la distinción. |
| `IMG_20240906_105926_1989-03-08_009_nota.txt` | 1 | `Instituciones públicas` | `S/D` | mismatch | Categoría de lugar (Vía pública / Instituciones públicas / Sede patronal / Lugar de Trabajo / Sede sindical) se confunde en notas con múltiples espacios. | Anotar ejemplos donde el evento ocurra en sede sindical, patronal o en la vía pública para reforzar la distinción. |
| `IMG_20240913_093817_1989-06-12_013_nota.txt` | 0 | `Vía pública` | `Sede sindical` | mismatch | Categoría de lugar (Vía pública / Instituciones públicas / Sede patronal / Lugar de Trabajo / Sede sindical) se confunde en notas con múltiples espacios. | Anotar ejemplos donde el evento ocurra en sede sindical, patronal o en la vía pública para reforzar la distinción. |
| `IMG_20240916_095514_1989-08-19_008_nota.txt` | 0 | `S/D` | `Instituciones públicas` | mismatch | Categoría de lugar (Vía pública / Instituciones públicas / Sede patronal / Lugar de Trabajo / Sede sindical) se confunde en notas con múltiples espacios. | Anotar ejemplos donde el evento ocurra en sede sindical, patronal o en la vía pública para reforzar la distinción. |
| `IMG_20240919_120800_1989-10-25_006_nota.txt` | 0 | `S/D` | `Instituciones públicas` | mismatch | Categoría de lugar (Vía pública / Instituciones públicas / Sede patronal / Lugar de Trabajo / Sede sindical) se confunde en notas con múltiples espacios. | Anotar ejemplos donde el evento ocurra en sede sindical, patronal o en la vía pública para reforzar la distinción. |
| `IMG_20241004_103347_1990-02-14_012_nota.txt` | 0 | `Lugar de Trabajo` | `Instituciones públicas` | mismatch | Categoría de lugar (Vía pública / Instituciones públicas / Sede patronal / Lugar de Trabajo / Sede sindical) se confunde en notas con múltiples espacios. | Anotar ejemplos donde el evento ocurra en sede sindical, patronal o en la vía pública para reforzar la distinción. |
| `IMG_20241004_113909_1990-03-17_012_nota.txt` | 0 | `Lugar de Trabajo` | `S/D` | mismatch | Categoría de lugar (Vía pública / Instituciones públicas / Sede patronal / Lugar de Trabajo / Sede sindical) se confunde en notas con múltiples espacios. | Anotar ejemplos donde el evento ocurra en sede sindical, patronal o en la vía pública para reforzar la distinción. |
| `IMG_20241004_113909_1990-03-17_012_nota.txt` | 1 | `Vía pública` | `S/D` | mismatch | Categoría de lugar (Vía pública / Instituciones públicas / Sede patronal / Lugar de Trabajo / Sede sindical) se confunde en notas con múltiples espacios. | Anotar ejemplos donde el evento ocurra en sede sindical, patronal o en la vía pública para reforzar la distinción. |
| `IMG_20241004_113909_1990-03-17_012_nota.txt` | 2 | `Instituciones públicas` | `S/D` | mismatch | Categoría de lugar (Vía pública / Instituciones públicas / Sede patronal / Lugar de Trabajo / Sede sindical) se confunde en notas con múltiples espacios. | Anotar ejemplos donde el evento ocurra en sede sindical, patronal o en la vía pública para reforzar la distinción. |

### D.12 `eventos_protesta[].incidentes.*.presencia (representativo)`

| nota_id | slot | correcto_gold | prediccion_modelo | tipo_error | por_que_importa | accion_recomendada |
|---|---:|---|---|---|---|---|
| `IMG_20240913_094108_1989-06-14_011_nota.txt` | 0 | `—` | `No` | extra | Entre los incidentes con error (non-match), 138/139 = 99.3% son missing/extra por alineación (downstream del conteo de eventos); sólo 1/139 = 0.7% es mismatch directo del booleano presencia. Sobre el total de 378 comparisons, downstream = 36.5%. Cuando ambos gold y pred son no-None, los pares son mayormente (No, No) y (S/D, S/D) — no hay un patrón 'pred=S/D, gold=No'. | Priorizar anotación de delimitación de evento y no-event hard negatives (F.1 + F.4/F.5). La accuracy de incidente subirá como side-effect; no hace falta anotar incidente adicional. |
| `IMG_20240916_103907_1989-09-13_004_nota.txt` | 0 | `—` | `No` | extra | Entre los incidentes con error (non-match), 138/139 = 99.3% son missing/extra por alineación (downstream del conteo de eventos); sólo 1/139 = 0.7% es mismatch directo del booleano presencia. Sobre el total de 378 comparisons, downstream = 36.5%. Cuando ambos gold y pred son no-None, los pares son mayormente (No, No) y (S/D, S/D) — no hay un patrón 'pred=S/D, gold=No'. | Priorizar anotación de delimitación de evento y no-event hard negatives (F.1 + F.4/F.5). La accuracy de incidente subirá como side-effect; no hace falta anotar incidente adicional. |
| `IMG_20241031_113558_1990-05-19_013_nota.txt` | 0 | `—` | `No` | extra | Entre los incidentes con error (non-match), 138/139 = 99.3% son missing/extra por alineación (downstream del conteo de eventos); sólo 1/139 = 0.7% es mismatch directo del booleano presencia. Sobre el total de 378 comparisons, downstream = 36.5%. Cuando ambos gold y pred son no-None, los pares son mayormente (No, No) y (S/D, S/D) — no hay un patrón 'pred=S/D, gold=No'. | Priorizar anotación de delimitación de evento y no-event hard negatives (F.1 + F.4/F.5). La accuracy de incidente subirá como side-effect; no hace falta anotar incidente adicional. |
| `IMG_20241031_113558_1990-05-19_013_nota.txt` | 1 | `—` | `No` | extra | Entre los incidentes con error (non-match), 138/139 = 99.3% son missing/extra por alineación (downstream del conteo de eventos); sólo 1/139 = 0.7% es mismatch directo del booleano presencia. Sobre el total de 378 comparisons, downstream = 36.5%. Cuando ambos gold y pred son no-None, los pares son mayormente (No, No) y (S/D, S/D) — no hay un patrón 'pred=S/D, gold=No'. | Priorizar anotación de delimitación de evento y no-event hard negatives (F.1 + F.4/F.5). La accuracy de incidente subirá como side-effect; no hace falta anotar incidente adicional. |
| `IMG_20241107_105320_1990_09_03_012_nota.txt` | 0 | `—` | `No` | extra | Entre los incidentes con error (non-match), 138/139 = 99.3% son missing/extra por alineación (downstream del conteo de eventos); sólo 1/139 = 0.7% es mismatch directo del booleano presencia. Sobre el total de 378 comparisons, downstream = 36.5%. Cuando ambos gold y pred son no-None, los pares son mayormente (No, No) y (S/D, S/D) — no hay un patrón 'pred=S/D, gold=No'. | Priorizar anotación de delimitación de evento y no-event hard negatives (F.1 + F.4/F.5). La accuracy de incidente subirá como side-effect; no hace falta anotar incidente adicional. |
| `IMG_20241108_102859_1990-10-01_006_nota.txt` | 0 | `—` | `No` | extra | Entre los incidentes con error (non-match), 138/139 = 99.3% son missing/extra por alineación (downstream del conteo de eventos); sólo 1/139 = 0.7% es mismatch directo del booleano presencia. Sobre el total de 378 comparisons, downstream = 36.5%. Cuando ambos gold y pred son no-None, los pares son mayormente (No, No) y (S/D, S/D) — no hay un patrón 'pred=S/D, gold=No'. | Priorizar anotación de delimitación de evento y no-event hard negatives (F.1 + F.4/F.5). La accuracy de incidente subirá como side-effect; no hace falta anotar incidente adicional. |
| `IMG_20241115_112301_1991-01-06_007_nota.txt` | 0 | `—` | `No` | extra | Entre los incidentes con error (non-match), 138/139 = 99.3% son missing/extra por alineación (downstream del conteo de eventos); sólo 1/139 = 0.7% es mismatch directo del booleano presencia. Sobre el total de 378 comparisons, downstream = 36.5%. Cuando ambos gold y pred son no-None, los pares son mayormente (No, No) y (S/D, S/D) — no hay un patrón 'pred=S/D, gold=No'. | Priorizar anotación de delimitación de evento y no-event hard negatives (F.1 + F.4/F.5). La accuracy de incidente subirá como side-effect; no hace falta anotar incidente adicional. |
| `IMG_20241115_113003_1991-01-09_011_nota.txt` | 0 | `—` | `No` | extra | Entre los incidentes con error (non-match), 138/139 = 99.3% son missing/extra por alineación (downstream del conteo de eventos); sólo 1/139 = 0.7% es mismatch directo del booleano presencia. Sobre el total de 378 comparisons, downstream = 36.5%. Cuando ambos gold y pred son no-None, los pares son mayormente (No, No) y (S/D, S/D) — no hay un patrón 'pred=S/D, gold=No'. | Priorizar anotación de delimitación de evento y no-event hard negatives (F.1 + F.4/F.5). La accuracy de incidente subirá como side-effect; no hace falta anotar incidente adicional. |
| `IMG_20241115_113003_1991-01-09_011_nota.txt` | 1 | `—` | `No` | extra | Entre los incidentes con error (non-match), 138/139 = 99.3% son missing/extra por alineación (downstream del conteo de eventos); sólo 1/139 = 0.7% es mismatch directo del booleano presencia. Sobre el total de 378 comparisons, downstream = 36.5%. Cuando ambos gold y pred son no-None, los pares son mayormente (No, No) y (S/D, S/D) — no hay un patrón 'pred=S/D, gold=No'. | Priorizar anotación de delimitación de evento y no-event hard negatives (F.1 + F.4/F.5). La accuracy de incidente subirá como side-effect; no hace falta anotar incidente adicional. |
| `IMG_20241115_113003_1991-01-09_011_nota.txt` | 2 | `—` | `No` | extra | Entre los incidentes con error (non-match), 138/139 = 99.3% son missing/extra por alineación (downstream del conteo de eventos); sólo 1/139 = 0.7% es mismatch directo del booleano presencia. Sobre el total de 378 comparisons, downstream = 36.5%. Cuando ambos gold y pred son no-None, los pares son mayormente (No, No) y (S/D, S/D) — no hay un patrón 'pred=S/D, gold=No'. | Priorizar anotación de delimitación de evento y no-event hard negatives (F.1 + F.4/F.5). La accuracy de incidente subirá como side-effect; no hace falta anotar incidente adicional. |

### D.13 Aggregate `incidentes.*.presencia` (todos los 6 paths)

Las 6 paths `incidentes.*.presencia` (represion / enfrentamiento / detenidos / heridos / muertos / danios_materiales) se agregaron para cuantificar qué parte del error es **downstream del conteo de eventos** (extra/missing por desalineación por índice) y qué parte es **true mismatch** del booleano presencia. Esto es la evidencia que sustenta el F.3 (ver §F).

| path | match | missing (downstream) | extra (downstream) | mismatch (directo) |
|---|---:|---:|---:|---:|
| `incidentes.represion.presencia` | 40 | 7 | 16 | 0 |
| `incidentes.enfrentamiento.presencia` | 39 | 7 | 16 | 1 |
| `incidentes.detenidos.presencia` | 40 | 7 | 16 | 0 |
| `incidentes.heridos.presencia` | 40 | 7 | 16 | 0 |
| `incidentes.muertos.presencia` | 40 | 7 | 16 | 0 |
| `incidentes.danios_materiales.presencia` | 40 | 7 | 16 | 0 |
| **TOTAL** | **239** | **42** | **96** | **1** |

**Lectura:** sobre 378 comparaciones agregadas, 138/378 (36.5%) son downstream del conteo de eventos (extra cuando el modelo inventa un evento cuyo `incidentes.* = No`, missing cuando el modelo trunca un evento que el gold tenía). Sólo 1/378 (0.3%) son **true mismatches** del booleano presencia (gold y pred ambos no-None con valores distintos).

**Pares (gold, pred) cuando ambos son no-None:**

| gold | pred | count |
|---|---|---:|
| `No` | `No` | 198 |
| `S/D` | `S/D` | 40 |
| `Sí` | `No` | 1 |
| `Sí` | `Sí` | 1 |

No se observa el patrón 'pred=S/D cuando gold=No' que la versión previa del reporte afirmaba: los pares no-None son mayormente matches (No→No, S/D→S/D, Sí→Sí); los mismatches directos son puntuales y se reportan explícitamente arriba.

## E. Peores ejemplos (micro-f1)

Orden por micro-f1 ascendente sobre las hojas aplanadas (`f1_vs_gold.tp/fp/fn`). Los casos con `gold_total=0, pred_total≥1` son **FP puros** (notas sin protesta a las que el modelo les inventa eventos). Los casos con `pred_total << gold_total` pero `pred_leaves >> gold_leaves` indican fragmentación + drift.

| rank | nota_id | f1 | tp/fp/fn | gold_leaves | pred_leaves | gold_total | pred_total | finish | out_tokens |
|---:|---|---:|---|---:|---:|---:|---:|---|---:|
| 1 | `IMG_20241115_113003_1991-01-09_011_nota.txt` | 0.0231 | 3/251/3 | 6 | 254 | 0 | 4 | stop | 2891 |
| 2 | `IMG_20241031_113558_1990-05-19_013_nota.txt` | 0.0296 | 2/127/4 | 6 | 129 | 0 | 2 | stop | 1369 |
| 3 | `IMG_20241107_105320_1990_09_03_012_nota.txt` | 0.0556 | 2/64/4 | 6 | 66 | 0 | 1 | stop | 729 |
| 4 | `IMG_20240913_094108_1989-06-14_011_nota.txt` | 0.0833 | 3/63/3 | 6 | 66 | 0 | 1 | stop | 664 |
| 5 | `IMG_20240916_103907_1989-09-13_004_nota.txt` | 0.0833 | 3/63/3 | 6 | 66 | 0 | 1 | stop | 686 |

### E.1 Diff compacto por peor ejemplo

Para cada uno de los 5 peores se listan las divergencias gold/pred en los paths del codebook más sensibles. Esto sirve para auditoría humana y para guiar la anotación dirigida.

#### E.1.1 `IMG_20241115_113003_1991-01-09_011_nota.txt` — f1=0.0231

- gold_total=0, pred_total=4, gold_leaves=6, pred_leaves=254, tp/fp/fn = 3/251/3.

| campo/path | slot | correcto_gold | prediccion_modelo | tipo_error |
|---|---:|---|---|---|
| `extraccion.tiene_eventos_protesta` | 0 | `false` | `true` | mismatch |
| `extraccion.total_eventos_protesta` | 0 | `0` | `4` | mismatch |
| `eventos_protesta[].temporalidad.tipo_temporal` | 0 | `—` | `Hecho` | extra |
| `eventos_protesta[].temporalidad.tipo_temporal` | 1 | `—` | `Hecho` | extra |
| `eventos_protesta[].temporalidad.tipo_temporal` | 2 | `—` | `Hecho` | extra |
| `eventos_protesta[].temporalidad.tipo_temporal` | 3 | `—` | `Hecho` | extra |
| `eventos_protesta[].temporalidad.tempo_verbal` | 0 | `—` | `Pasado` | extra |
| `eventos_protesta[].temporalidad.tempo_verbal` | 1 | `—` | `Pasado` | extra |
| `eventos_protesta[].temporalidad.tempo_verbal` | 2 | `—` | `Pasado` | extra |
| `eventos_protesta[].temporalidad.tempo_verbal` | 3 | `—` | `Pasado` | extra |
| `eventos_protesta[].delimitacion_evento.criterio_delimitacion` | 0 | `—` | `Temporal` | extra |
| `eventos_protesta[].delimitacion_evento.criterio_delimitacion` | 1 | `—` | `Temporal` | extra |
| `eventos_protesta[].delimitacion_evento.criterio_delimitacion` | 2 | `—` | `Temporal` | extra |
| `eventos_protesta[].delimitacion_evento.criterio_delimitacion` | 3 | `—` | `Temporal` | extra |
| `eventos_protesta[].accion.formato_principal.categoria` | 0 | `—` | `Reuniones entre las partes litigantes` | extra |
| `eventos_protesta[].accion.formato_principal.categoria` | 1 | `—` | `Manifestaciones` | extra |
| `eventos_protesta[].accion.formato_principal.categoria` | 2 | `—` | `Manifestaciones` | extra |
| `eventos_protesta[].accion.formato_principal.categoria` | 3 | `—` | `Manifestaciones de baja intensidad` | extra |
| `eventos_protesta[].sujetos[].categoria` | 0 | `—` | `Asalariados` | extra |
| `eventos_protesta[].sujetos[].categoria` | 1 | `—` | `Asalariados` | extra |
| `eventos_protesta[].sujetos[].categoria` | 2 | `—` | `Asalariados` | extra |
| `eventos_protesta[].sujetos[].categoria` | 3 | `—` | `Asalariados` | extra |
| `eventos_protesta[].demandas[].categoria` | 0 | `—` | `Salarial` | extra |
| `eventos_protesta[].demandas[].categoria` | 1 | `—` | `Salarial` | extra |
| `eventos_protesta[].demandas[].categoria` | 2 | `—` | `Salarial` | extra |
| `eventos_protesta[].demandas[].categoria` | 3 | `—` | `Salarial` | extra |
| `eventos_protesta[].contra_quien[].categoria` | 0 | `—` | `Estado/Gobierno` | extra |
| `eventos_protesta[].contra_quien[].categoria` | 1 | `—` | `Estado/Gobierno` | extra |
| `eventos_protesta[].contra_quien[].categoria` | 2 | `—` | `Estado/Gobierno` | extra |
| `eventos_protesta[].contra_quien[].categoria` | 3 | `—` | `Estado/Gobierno` | extra |
| `eventos_protesta[].contra_quien[].nivel_institucional` | 0 | `—` | `Provincial` | extra |
| `eventos_protesta[].contra_quien[].nivel_institucional` | 1 | `—` | `Provincial` | extra |
| `eventos_protesta[].contra_quien[].nivel_institucional` | 2 | `—` | `Provincial` | extra |
| `eventos_protesta[].contra_quien[].nivel_institucional` | 3 | `—` | `Provincial` | extra |
| `eventos_protesta[].lugares[].categoria` | 0 | `—` | `Instituciones públicas` | extra |
| `eventos_protesta[].lugares[].categoria` | 1 | `—` | `Instituciones públicas` | extra |
| `eventos_protesta[].lugares[].categoria` | 2 | `—` | `Vía pública` | extra |
| `eventos_protesta[].lugares[].categoria` | 3 | `—` | `Vía pública` | extra |
| `eventos_protesta[].incidentes.*.presencia (representativo)` | 0 | `—` | `No` | extra |
| `eventos_protesta[].incidentes.*.presencia (representativo)` | 1 | `—` | `No` | extra |
| `eventos_protesta[].incidentes.*.presencia (representativo)` | 2 | `—` | `No` | extra |
| `eventos_protesta[].incidentes.*.presencia (representativo)` | 3 | `—` | `No` | extra |

> **Nota:** el gold marca la nota como **sin protesta**; el modelo **alucina 4 evento(s)** completos (con lugares, demandas, incidentes, etc.). En este ejemplo el desbalance es **tp=3 / fp=251 / fn=3** sobre 6 hojas gold vs 254 hojas predichas.

#### E.1.2 `IMG_20241031_113558_1990-05-19_013_nota.txt` — f1=0.0296

- gold_total=0, pred_total=2, gold_leaves=6, pred_leaves=129, tp/fp/fn = 2/127/4.

| campo/path | slot | correcto_gold | prediccion_modelo | tipo_error |
|---|---:|---|---|---|
| `extraccion.tiene_eventos_protesta` | 0 | `false` | `true` | mismatch |
| `extraccion.total_eventos_protesta` | 0 | `0` | `2` | mismatch |
| `eventos_protesta[].temporalidad.tipo_temporal` | 0 | `—` | `Anuncio` | extra |
| `eventos_protesta[].temporalidad.tipo_temporal` | 1 | `—` | `Hecho` | extra |
| `eventos_protesta[].temporalidad.tempo_verbal` | 0 | `—` | `Futuro` | extra |
| `eventos_protesta[].temporalidad.tempo_verbal` | 1 | `—` | `Presente` | extra |
| `eventos_protesta[].delimitacion_evento.criterio_delimitacion` | 0 | `—` | `Evento unico en la nota` | extra |
| `eventos_protesta[].delimitacion_evento.criterio_delimitacion` | 1 | `—` | `Temporal` | extra |
| `eventos_protesta[].accion.formato_principal.categoria` | 0 | `—` | `Manifestaciones de baja intensidad` | extra |
| `eventos_protesta[].accion.formato_principal.categoria` | 1 | `—` | `Huelgas` | extra |
| `eventos_protesta[].sujetos[].categoria` | 0 | `—` | `Asalariados` | extra |
| `eventos_protesta[].sujetos[].categoria` | 1 | `—` | `Asalariados` | extra |
| `eventos_protesta[].demandas[].categoria` | 0 | `—` | `Salarial` | extra |
| `eventos_protesta[].demandas[].categoria` | 1 | `—` | `Salarial` | extra |
| `eventos_protesta[].contra_quien[].categoria` | 0 | `—` | `Estado/Gobierno` | extra |
| `eventos_protesta[].contra_quien[].categoria` | 1 | `—` | `Estado/Gobierno` | extra |
| `eventos_protesta[].contra_quien[].nivel_institucional` | 0 | `—` | `Nacional` | extra |
| `eventos_protesta[].contra_quien[].nivel_institucional` | 1 | `—` | `Nacional` | extra |
| `eventos_protesta[].lugares[].categoria` | 0 | `—` | `S/D` | extra |
| `eventos_protesta[].lugares[].categoria` | 1 | `—` | `Instituciones públicas` | extra |
| `eventos_protesta[].incidentes.*.presencia (representativo)` | 0 | `—` | `No` | extra |
| `eventos_protesta[].incidentes.*.presencia (representativo)` | 1 | `—` | `No` | extra |

> **Nota:** el gold marca la nota como **sin protesta**; el modelo **alucina 2 evento(s)** completos (con lugares, demandas, incidentes, etc.). En este ejemplo el desbalance es **tp=2 / fp=127 / fn=4** sobre 6 hojas gold vs 129 hojas predichas.

#### E.1.3 `IMG_20241107_105320_1990_09_03_012_nota.txt` — f1=0.0556

- gold_total=0, pred_total=1, gold_leaves=6, pred_leaves=66, tp/fp/fn = 2/64/4.

| campo/path | slot | correcto_gold | prediccion_modelo | tipo_error |
|---|---:|---|---|---|
| `extraccion.tiene_eventos_protesta` | 0 | `false` | `true` | mismatch |
| `extraccion.total_eventos_protesta` | 0 | `0` | `1` | mismatch |
| `eventos_protesta[].temporalidad.tipo_temporal` | 0 | `—` | `Hecho` | extra |
| `eventos_protesta[].temporalidad.tempo_verbal` | 0 | `—` | `Presente` | extra |
| `eventos_protesta[].delimitacion_evento.criterio_delimitacion` | 0 | `—` | `Evento unico en la nota` | extra |
| `eventos_protesta[].accion.formato_principal.categoria` | 0 | `—` | `Manifestaciones` | extra |
| `eventos_protesta[].sujetos[].categoria` | 0 | `—` | `Asalariados` | extra |
| `eventos_protesta[].demandas[].categoria` | 0 | `—` | `Política` | extra |
| `eventos_protesta[].contra_quien[].categoria` | 0 | `—` | `Estado/Gobierno` | extra |
| `eventos_protesta[].contra_quien[].nivel_institucional` | 0 | `—` | `Nacional` | extra |
| `eventos_protesta[].lugares[].categoria` | 0 | `—` | `Sede sindical` | extra |
| `eventos_protesta[].incidentes.*.presencia (representativo)` | 0 | `—` | `No` | extra |

> **Nota:** el gold marca la nota como **sin protesta**; el modelo **alucina 1 evento(s)** completos (con lugares, demandas, incidentes, etc.). En este ejemplo el desbalance es **tp=2 / fp=64 / fn=4** sobre 6 hojas gold vs 66 hojas predichas.

#### E.1.4 `IMG_20240913_094108_1989-06-14_011_nota.txt` — f1=0.0833

- gold_total=0, pred_total=1, gold_leaves=6, pred_leaves=66, tp/fp/fn = 3/63/3.

| campo/path | slot | correcto_gold | prediccion_modelo | tipo_error |
|---|---:|---|---|---|
| `extraccion.tiene_eventos_protesta` | 0 | `false` | `true` | mismatch |
| `extraccion.total_eventos_protesta` | 0 | `0` | `1` | mismatch |
| `eventos_protesta[].temporalidad.tipo_temporal` | 0 | `—` | `S/D` | extra |
| `eventos_protesta[].temporalidad.tempo_verbal` | 0 | `—` | `S/D` | extra |
| `eventos_protesta[].delimitacion_evento.criterio_delimitacion` | 0 | `—` | `S/D` | extra |
| `eventos_protesta[].accion.formato_principal.categoria` | 0 | `—` | `S/D` | extra |
| `eventos_protesta[].sujetos[].categoria` | 0 | `—` | `S/D` | extra |
| `eventos_protesta[].demandas[].categoria` | 0 | `—` | `S/D` | extra |
| `eventos_protesta[].contra_quien[].categoria` | 0 | `—` | `S/D` | extra |
| `eventos_protesta[].contra_quien[].nivel_institucional` | 0 | `—` | `S/D` | extra |
| `eventos_protesta[].lugares[].categoria` | 0 | `—` | `S/D` | extra |
| `eventos_protesta[].incidentes.*.presencia (representativo)` | 0 | `—` | `No` | extra |

> **Nota:** el gold marca la nota como **sin protesta**; el modelo **alucina 1 evento(s)** completos (con lugares, demandas, incidentes, etc.). En este ejemplo el desbalance es **tp=3 / fp=63 / fn=3** sobre 6 hojas gold vs 66 hojas predichas.

#### E.1.5 `IMG_20240916_103907_1989-09-13_004_nota.txt` — f1=0.0833

- gold_total=0, pred_total=1, gold_leaves=6, pred_leaves=66, tp/fp/fn = 3/63/3.

| campo/path | slot | correcto_gold | prediccion_modelo | tipo_error |
|---|---:|---|---|---|
| `extraccion.tiene_eventos_protesta` | 0 | `false` | `true` | mismatch |
| `extraccion.total_eventos_protesta` | 0 | `0` | `1` | mismatch |
| `eventos_protesta[].temporalidad.tipo_temporal` | 0 | `—` | `S/D` | extra |
| `eventos_protesta[].temporalidad.tempo_verbal` | 0 | `—` | `S/D` | extra |
| `eventos_protesta[].delimitacion_evento.criterio_delimitacion` | 0 | `—` | `S/D` | extra |
| `eventos_protesta[].accion.formato_principal.categoria` | 0 | `—` | `S/D` | extra |
| `eventos_protesta[].sujetos[].categoria` | 0 | `—` | `S/D` | extra |
| `eventos_protesta[].demandas[].categoria` | 0 | `—` | `S/D` | extra |
| `eventos_protesta[].contra_quien[].categoria` | 0 | `—` | `S/D` | extra |
| `eventos_protesta[].contra_quien[].nivel_institucional` | 0 | `—` | `S/D` | extra |
| `eventos_protesta[].lugares[].categoria` | 0 | `—` | `S/D` | extra |
| `eventos_protesta[].incidentes.*.presencia (representativo)` | 0 | `—` | `No` | extra |

> **Nota:** el gold marca la nota como **sin protesta**; el modelo **alucina 1 evento(s)** completos (con lugares, demandas, incidentes, etc.). En este ejemplo el desbalance es **tp=3 / fp=63 / fn=3** sobre 6 hojas gold vs 66 hojas predichas.

## F. Plan de anotación dirigida (siguiente iteración)

Las recomendaciones están ordenadas por **impacto esperado** sobre las métricas actuales. La evidencia concreta está en las secciones C/D/E. **No se recomienda** más datos genéricos: el análisis muestra errores estructurales por tipo de nota y por categoría del codebook, no por tamaño del set.

### F.1 Anotar 14 notas **gold=False** con léxico conflict-adjacent (paro, huelga, advertencia, conflicto, tensión, medida de fuerza) que NO constituyan protesta real.

**Evidencia observada:** 7/35 notas tienen FP en `tiene_eventos_protesta`; 5/5 de los peores ejemplos son FP puros (gold_total=0, pred_total≥1). El modelo sobre-reacciona a palabras como 'paro', 'huelga', 'estado de alerta' aunque la nota no reporte protesta efectiva.

**Acción concreta:** Buscar y anotar 8-15 notas del CSV `muestra_350_conflictos_1989_1995.csv` donde el texto mencione estos términos pero el gold correcto sea `tiene_eventos_protesta=false`. Usar como negativos difíciles para enseñar el borde del concepto.

**Métrica objetivo a mover:** Reducir `extraccion.tiene_eventos_protesta` FP (hoy 7) y bajar `evento.es_evento_protesta` FP.

**Criterio de éxito:** FP ≤ 3 sobre 35 en `tiene_eventos_protesta` y accuracy ≥ 0.85; recall no cae más de 0.02 respecto a r32_3e.

### F.2 Anotar 10-20 ejemplos de **`contra_quien[].nivel_institucional`** explícitamente etiquetados (Municipal / Provincial / Nacional / Privado / Internacional / No aplica) cuando el destinatario sea el Estado o una empresa privada.

**Evidencia observada:** `contra_quien[].nivel_institucional` es el path con **menor accuracy (0.1856)** sobre 97 comparaciones; muchos casos predicen 'S/D' cuando el gold es 'Nacional' o 'Privado'.

**Acción concreta:** Re-pasar manualmente las notas del set de entrenamiento con `contra_quien` presente y **forzar** el nivel_institucional cuando el texto mencione 'Nación', 'Provincia', 'Municipalidad', 'empresa', 'firma', etc.

**Métrica objetivo a mover:** `contra_quien[].nivel_institucional` accuracy ≥ 0.40 (hoy 0.1856).

**Criterio de éxito:** Accuracy ≥ 0.40 sobre el eval set y descenso del FN en ≥ 30% sin subir FP global por encima de 0.55.

### F.3 NO priorizar anotación adicional de `incidentes.*.presencia`: los errores son downstream del conteo de eventos (extra/missing), no del booleano presencia.

**Evidencia observada:** Aggregate across the 6 `incidentes.*.presencia` paths (378 comparaciones): 239 matches, 42 missings, 96 extras, 1 true mismatch (direct gold-vs-pred boolean error). Los `missing` y `extra` son artefactos de desalineación por índice cuando `pred_total != gold_total` — son errores de event boundary, no del booleano presencia. Cuando ambos gold y pred son no-None, los pares son abrumadoramente (No, No) y (S/D, S/D); no hay un patrón 'pred=S/D, gold=No'.

**Acción concreta:** Re-pasar la métrica de `incidentes.*.presencia` después de aplicar F.1 (no-event hard negatives) y F.4/F.5 (delimitación de evento). Si la accuracy sube a ≥ 0.80 sin anotar incidente adicional, cerrar este F.3 como 'side-effect confirmado'. Si tras F.1+F.4/F.5 la accuracy sigue < 0.70, entonces sí abrir un target nuevo con la evidencia cuantificada de mismatch directo (no antes).

**Métrica objetivo a mover:** `incidentes.*.presencia` aggregate accuracy ≥ 0.80 después de F.1 + F.4/F.5, sin anotación adicional de incidente.

**Criterio de éxito:** Aggregate accuracy de los 6 paths `incidentes.*.presencia` ≥ 0.80, con `true_direct_mismatch` ≤ 2 en el eval set (hoy 1).

### F.4 Anotar 10-20 ejemplos de **`delimitacion.criterio_delimitacion`** donde la categoría sea ambigua en el codebook.

**Evidencia observada:** `delimitacion.criterio_delimitacion` tiene accuracy 0.1977 sobre 86 comparaciones; está en el top-2 de error_rate.

**Acción concreta:** Anotar notas cuya regla de delimitación sea Accion principal con acciones complementarias vs Temporal vs Espacial vs Temporal y espacial vs Evento unico en la nota, y revisar manualmente la consistencia con la regla del codebook §3 (delimitación).

**Métrica objetivo a mover:** `delimitacion.criterio_delimitacion` accuracy ≥ 0.45.

**Criterio de éxito:** Accuracy del path ≥ 0.45 sin deteriorar aggregate categorical > 0.05.

### F.5 Anotar **eventos correctamente delimitados** cuando hay acciones complementarias (corte + volanteada, paro + marcha, etc.) para reforzar `criterio_delimitacion = 'Accion principal con acciones complementarias'`.

**Evidencia observada:** `delimitacion.criterio_delimitacion` accuracy 0.1977 (peor entre los delimitadores). Hay 11 ejemplos con extra_event y 6 con missing_event — ambos son síntomas de fragmentación.

**Acción concreta:** Anotar 10-15 notas donde la regla 'acción principal + complementarias' aplique explícitamente, y verificar que `es_accion_principal_con_complementarias=true` se use correctamente.

**Métrica objetivo a mover:** `delimitacion.criterio_delimitacion` accuracy ≥ 0.40 y `delimitacion.es_accion_principal_con_complementarias` accuracy ≥ 0.55.

**Criterio de éxito:** Reducir extra_event / missing_event en ≥ 50% sobre el eval set y subir f1_global a ≥ 0.60.

## G. Caveats

- **Eval set histórico tiene sólo 35 ejemplos.** Las métricas son evidencia direccional, no intervalos de confianza. Errores agregados (1862 comparaciones categóricas) son robustos a este tamaño, pero cada bin individual (ej. `Sujetos[].categoria`) tiene pocos ejemplos y un par de correcciones pueden moverlo mucho.
- **Gold histórico es gold dentro de esa corrida.** Las 35 notas son parte del set histórico de 350 producidas por GPT-5.4-mini + validación humana de Nico (weight 1.0). El campo `metadatos_extraccion.estado_validacion_humana: "No validado"` se ignora: es un placeholder engañoso heredado del pipeline. Si una nota histórica de las 35 parece mal anotada, se discute antes de cambiar el training data.
- **Alineación por índice.** Las comparaciones categóricas alinean `eventos_protesta[].*` por índice, no por contenido. Cuando pred_total ≠ gold_total, los slots extra/missing pueden arrastrar falsos FP/FN de categorías estructurales (no sólo semánticas). Por eso este reporte separa `C.2 errores de conteo` de `C.3 errores categóricos`.
- **Origen del baseline correcto.** El baseline de PLAN §6 es **Qwen2.5-7B-Instruct sin LoRA**, NO GPT-5.5. Cualquier mención histórica a `gpt-5.5` en artefactos es ruido y no se usa como baseline ni como origen del training data.
- **No se corrió entrenamiento ni inferencia para producir este reporte histórico.** Los números vienen literal de `metrics/finetuned_qwen-protesta-v1-r32.json` y de `data/chat_formatted/eval.jsonl` (gold histórico). El script sólo agrega y compara.

## Sources (read-only)

- `metrics/finetuned_qwen-protesta-v1-r32.json` — métricas completas del modelo evaluado
- `metrics/finetuned_qwen-protesta-v1-r32_outputs.jsonl` — raw outputs del run
- `metrics/finetuned_qwen-protesta-v1.json` — r16_3e (para comparación)
- `metrics/finetuned_qwen-protesta-v1-r32-e5.json` — r32_e5 (control de epochs)
- `metrics/baseline_qwen2.5-7b.json` — Phase 2 baseline
- `data/chat_formatted/eval.jsonl` — gold assistant JSON histórico (35 ejemplos)
- `esquema_eventos_protesta_entrenamiento_MVS.json` — MVS schema con enums
- `reports/phase6_r32_eval.json`, `reports/phase6_r32_e5_eval.json` — readiness reports
- `scripts/baseline_qwen_full.py` — `EVENT_CATEGORICAL_PATHS` y `_resolve_path` (helpers puros)
- `scripts/analyze_r32_errors.py` — generador de este reporte
