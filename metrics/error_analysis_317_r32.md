# Error analysis — clean 317-r32 model

Este reporte analiza **solo** el modelo clean `317-r32` sobre el eval actual de 32 filas. No compara deltas contra artefactos históricos 350-era, porque son no comparables.

## Executive summary

- Schema validity: **1.0000**.
- f1_global: **0.4910** (precision=0.4569, recall=0.5307).
- Categorical accuracy aggregate: **0.3670**.
- `extraccion.tiene_eventos_protesta` accuracy: **0.8438** (TP=27, TN=0, FP=5, FN=0).
- Lectura clave: el modelo detecta todos los casos positivos del eval, pero convierte los 5 negativos en protesta; el cuello de botella pasa por hard negatives, conteo/delimitación de eventos y categorías de demanda/acción/sujeto/destinatario.

## Headline metrics — current 317-r32 only

| metric | value |
|---|---|
| schema_validity | 1.0000 |
| f1_global | 0.4910 |
| categorical_accuracy_aggregate | 0.3670 |
| tiene_eventos_protesta accuracy | 0.8438 |
| eval examples | 32 |

> Historical 350-era metrics are intentionally omitted from the headline table. They may be useful as provenance, but are not a valid direct delta against this 317-row split.

## Error taxonomy

### `extraccion.tiene_eventos_protesta`

| bucket | count | nota_id examples |
|---|---|---|
| TP | 27 | IMG_20240906_104231_1989-02-25_012_nota.txt, IMG_20240912_114446_1989-04-29_010_nota.txt, IMG_20240916_093733_1989-08-11_012_nota.txt |
| TN | 0 | — |
| FP | 5 | IMG_20241031_111940_1990-05-09_013_nota.txt, IMG_20241108_114930_1990-11-23_013_nota.txt, IMG_20241115_105327_1990-12-19_007_nota.txt |
| FN | 0 | — |

### Event count errors

| gold_total_events | pred_total_events | delta | exact | extra | missing |
|---|---|---|---|---|---|
| 46 | 45 | -1 | 15 | 9 | 8 |

### Top categorical paths by error rate

| path | accuracy | error_rate | support | fp | fn |
|---|---|---|---|---|---|
| demandas[].categoria | 0.1720 | 0.8280 | 93 | 45 | 32 |
| accion.formato_principal.categoria | 0.1857 | 0.8143 | 70 | 32 | 25 |
| delimitacion.criterio_delimitacion | 0.2029 | 0.7971 | 69 | 31 | 24 |
| sujetos[].organizaciones[].categoria | 0.2366 | 0.7634 | 93 | 42 | 29 |
| contra_quien[].categoria | 0.2658 | 0.7342 | 79 | 29 | 29 |
| contra_quien[].nivel_institucional | 0.2658 | 0.7342 | 79 | 29 | 29 |
| sujetos[].categoria | 0.2683 | 0.7317 | 82 | 35 | 25 |
| lugares[].categoria | 0.3143 | 0.6857 | 70 | 28 | 20 |
| delimitacion.es_accion_principal_con_complementarias | 0.3387 | 0.6613 | 62 | 24 | 17 |
| temporalidad.tipo_temporal | 0.3387 | 0.6613 | 62 | 24 | 17 |

### False-event null contract

Status: **pass**. Violations=0; `S/D` violations=0.

## Gold-vs-pred high-value examples

| nota_id | path/campo | correcto_gold | prediccion_modelo | tipo_error | por_que_importa | accion_recomendada |
|---|---|---|---|---|---|---|
| IMG_20241031_111940_1990-05-09_013_nota.txt | extraccion.tiene_eventos_protesta | False | True | mismatch | Abre o cierra toda la extracción; un FP crea eventos inexistentes y contamina hojas downstream. | Agregar hard negatives de notas no-protesta y explicitar criterios de exclusión. |
| IMG_20241115_105327_1990-12-19_007_nota.txt | extraccion.tiene_eventos_protesta | False | True | mismatch | Abre o cierra toda la extracción; un FP crea eventos inexistentes y contamina hojas downstream. | Agregar hard negatives de notas no-protesta y explicitar criterios de exclusión. |
| IMG_20241108_114930_1990-11-23_013_nota.txt | extraccion.tiene_eventos_protesta | False | True | mismatch | Abre o cierra toda la extracción; un FP crea eventos inexistentes y contamina hojas downstream. | Agregar hard negatives de notas no-protesta y explicitar criterios de exclusión. |
| IMG_20241115_114308_1991-01-19_012_nota.txt | extraccion.tiene_eventos_protesta | False | True | mismatch | Abre o cierra toda la extracción; un FP crea eventos inexistentes y contamina hojas downstream. | Agregar hard negatives de notas no-protesta y explicitar criterios de exclusión. |
| IMG_20241125_103409_1991-04-12_014_nota.txt | extraccion.tiene_eventos_protesta | False | True | mismatch | Abre o cierra toda la extracción; un FP crea eventos inexistentes y contamina hojas downstream. | Agregar hard negatives de notas no-protesta y explicitar criterios de exclusión. |
| IMG_20241031_111940_1990-05-09_013_nota.txt | extraccion.total_eventos_protesta | 0 | 1 | mismatch | El alineamiento por índice hace que un conteo errado multiplique FP/FN en casi todos los campos. | Revisar reglas de segmentación y conteo de eventos por nota. |
| IMG_20241115_105327_1990-12-19_007_nota.txt | extraccion.total_eventos_protesta | 0 | 1 | mismatch | El alineamiento por índice hace que un conteo errado multiplique FP/FN en casi todos los campos. | Revisar reglas de segmentación y conteo de eventos por nota. |
| IMG_20241108_114930_1990-11-23_013_nota.txt | extraccion.total_eventos_protesta | 0 | 1 | mismatch | El alineamiento por índice hace que un conteo errado multiplique FP/FN en casi todos los campos. | Revisar reglas de segmentación y conteo de eventos por nota. |
| IMG_20241115_114308_1991-01-19_012_nota.txt | extraccion.total_eventos_protesta | 0 | 1 | mismatch | El alineamiento por índice hace que un conteo errado multiplique FP/FN en casi todos los campos. | Revisar reglas de segmentación y conteo de eventos por nota. |
| IMG_20241125_103409_1991-04-12_014_nota.txt | extraccion.total_eventos_protesta | 0 | 1 | mismatch | El alineamiento por índice hace que un conteo errado multiplique FP/FN en casi todos los campos. | Revisar reglas de segmentación y conteo de eventos por nota. |

## Worst examples — top 5 by per-example F1

| rank | nota_id | f1 | tp | fp | fn | gold_total | pred_total |
|---|---|---|---|---|---|---|---|
| 1 | IMG_20241031_111940_1990-05-09_013_nota.txt | 0.0513 | 2 | 70 | 4 | 0 | 1 |
| 2 | IMG_20241115_105327_1990-12-19_007_nota.txt | 0.0556 | 2 | 64 | 4 | 0 | 1 |
| 3 | IMG_20241108_114930_1990-11-23_013_nota.txt | 0.0833 | 3 | 63 | 3 | 0 | 1 |
| 4 | IMG_20241115_114308_1991-01-19_012_nota.txt | 0.0833 | 3 | 63 | 3 | 0 | 1 |
| 5 | IMG_20241125_103409_1991-04-12_014_nota.txt | 0.0833 | 3 | 63 | 3 | 0 | 1 |

### IMG_20241031_111940_1990-05-09_013_nota.txt — f1=0.0513

| path | gold | pred | tipo_error |
|---|---|---|---|
| extraccion.tiene_eventos_protesta | False | True | mismatch |
| extraccion.total_eventos_protesta | 0 | 1 | mismatch |
| delimitacion.criterio_delimitacion | null | Evento unico en la nota | fp_extra_pred |
| delimitacion.es_accion_principal_con_complementarias | null | False | fp_extra_pred |
| accion.formato_principal.categoria | null | Manifestaciones de baja intensidad | fp_extra_pred |
| demandas[].categoria | null | S/D | fp_extra_pred |
| sujetos[].categoria | null | Militantes | fp_extra_pred |
| sujetos[].categoria | null | Empresarios / Gerentes / Directivos | fp_extra_pred |

### IMG_20241115_105327_1990-12-19_007_nota.txt — f1=0.0556

| path | gold | pred | tipo_error |
|---|---|---|---|
| extraccion.tiene_eventos_protesta | False | True | mismatch |
| extraccion.total_eventos_protesta | 0 | 1 | mismatch |
| delimitacion.criterio_delimitacion | null | S/D | fp_extra_pred |
| delimitacion.es_accion_principal_con_complementarias | null | False | fp_extra_pred |
| accion.formato_principal.categoria | null | S/D | fp_extra_pred |
| demandas[].categoria | null | S/D | fp_extra_pred |
| sujetos[].categoria | null | S/D | fp_extra_pred |
| sujetos[].organizaciones[].categoria | null | S/D | fp_extra_pred |

### IMG_20241108_114930_1990-11-23_013_nota.txt — f1=0.0833

| path | gold | pred | tipo_error |
|---|---|---|---|
| extraccion.tiene_eventos_protesta | False | True | mismatch |
| extraccion.total_eventos_protesta | 0 | 1 | mismatch |
| delimitacion.criterio_delimitacion | null | Evento unico en la nota | fp_extra_pred |
| delimitacion.es_accion_principal_con_complementarias | null | False | fp_extra_pred |
| accion.formato_principal.categoria | null | Manifestaciones de baja intensidad | fp_extra_pred |
| demandas[].categoria | null | Política | fp_extra_pred |
| sujetos[].categoria | null | Asalariados | fp_extra_pred |
| sujetos[].organizaciones[].categoria | null | Patronal | fp_extra_pred |

### IMG_20241115_114308_1991-01-19_012_nota.txt — f1=0.0833

| path | gold | pred | tipo_error |
|---|---|---|---|
| extraccion.tiene_eventos_protesta | False | True | mismatch |
| extraccion.total_eventos_protesta | 0 | 1 | mismatch |
| delimitacion.criterio_delimitacion | null | Evento unico en la nota | fp_extra_pred |
| delimitacion.es_accion_principal_con_complementarias | null | False | fp_extra_pred |
| accion.formato_principal.categoria | null | Manifestaciones de baja intensidad | fp_extra_pred |
| demandas[].categoria | null | S/D | fp_extra_pred |
| sujetos[].categoria | null | Militares | fp_extra_pred |
| sujetos[].organizaciones[].categoria | null | S/D | fp_extra_pred |

### IMG_20241125_103409_1991-04-12_014_nota.txt — f1=0.0833

| path | gold | pred | tipo_error |
|---|---|---|---|
| extraccion.tiene_eventos_protesta | False | True | mismatch |
| extraccion.total_eventos_protesta | 0 | 1 | mismatch |
| delimitacion.criterio_delimitacion | null | Evento unico en la nota | fp_extra_pred |
| delimitacion.es_accion_principal_con_complementarias | null | False | fp_extra_pred |
| accion.formato_principal.categoria | null | Asambleas | fp_extra_pred |
| demandas[].categoria | null | S/D | fp_extra_pred |
| sujetos[].categoria | null | Asalariados | fp_extra_pred |
| sujetos[].organizaciones[].categoria | null | Sindical | fp_extra_pred |

## Annotation/action plan

| target | evidence | action |
|---|---|---|
| Hard negatives para `tiene_eventos_protesta` | Hay 5 FP y 0 FN en 32 filas; el modelo no produjo TN. | Agregar/curar notas no-protesta parecidas a protesta y exigir detalles null cuando `tiene=false`. |
| Delimitación y conteo de eventos | Gold total=46, pred total=45, delta=-1; 9 notas tienen eventos de más y 8 eventos de menos. | Anotar ejemplos contrastivos de evento único vs acciones complementarias y revisar `total_eventos_protesta`. |
| Codebook/examples para `demandas[].categoria` | accuracy=0.1720, error_rate=0.8280, support=93, fp=45, fn=32. | Priorizar ejemplos contrastivos de demanda salarial/laboral/gremial y demandas múltiples. |
| Codebook/examples para `accion.formato_principal.categoria` | accuracy=0.1857, error_rate=0.8143, support=70, fp=32, fn=25. | Reforzar pares confusos del formato de acción con ejemplos contrastivos. |
| Codebook/examples para `delimitacion.criterio_delimitacion` | accuracy=0.2029, error_rate=0.7971, support=69, fp=31, fn=24. | Anotar casos límite de acción principal + complementarias vs evento único. |
| Codebook/examples para `sujetos[].organizaciones[].categoria` | accuracy=0.2366, error_rate=0.7634, support=93, fp=42, fn=29. | Separar con ejemplos la categoría del sujeto de la categoría de su organización. |
| Codebook/examples para `contra_quien[].categoria` | accuracy=0.2658, error_rate=0.7342, support=79, fp=29, fn=29. | Anotar destinatarios Estado/patronal/sindicato con evidencia textual explícita. |

## Caveats

- Eval actual pequeño: 32 filas; usar como diagnóstico, no como conclusión estadística definitiva.
- Artefactos 350-era son históricos y no comparables contra este split 317.
- El alineamiento de eventos es por índice; si `pred_total != gold_total`, muchos FP/FN downstream reflejan desalineación, no necesariamente error semántico independiente del campo.
- Contrato vigente: en falsos eventos los detalles deben ser `null`; `S/D` solo corresponde dentro de eventos reales con valor textual/categorial desconocido.

## Sources

- `/home/agusnieto77/train_pea/metrics/finetuned_qwen-protesta-317-r32.json`
- `/home/agusnieto77/train_pea/metrics/finetuned_qwen-protesta-317-r32_outputs.jsonl`
- `/home/agusnieto77/train_pea/data/chat_formatted/eval.jsonl`
