# Guía para agentes OpenCode

Repositorio de **fine-tuning de Qwen2.5-7B-Instruct como extractor experto de eventos de protesta** a partir de notas periodísticas históricas argentinas. El training data canónico actual es `entrenamiento.jsonl` con **317 ejemplos**, originados en el pipeline OpenAI Batch API con **GPT-5.4-mini** y validados por Nico.

## Estado del repo (2026-06-26)

Pre-proceso consolidado en la raíz (no hay subcarpetas operativas). **Fase 1 — Data prep** fue regenerada desde las 317 filas canónicas; los artefactos previos de training/eval sobre 350 filas son históricos hasta reentrenar/evaluar otra vez. Plan completo en `PLAN_ENTRENAMIENTO_QWEN.md`.

## Fuentes de verdad

| Archivo | Rol |
|---|---|
| `PLAN_ENTRENAMIENTO_QWEN.md` | Plan de fine-tuning (7 fases, hiperparámetros, criterios de éxito). **Empezar por acá.** |
| `entrenamiento.jsonl` | 317 ejemplos canónicos del training data (todos validados por Nico, weight 1.0). |
| `SYSTEM_PROMPT_GPT5_USADO.md` | System prompt EXACTO (1873 chars) extraído del batch histórico. **Usar literal** en training e inference; no editar el bloque literal. |
| `USER_MESSAGE_TEMPLATE_GPT5.md` | Reconstrucción operativa del user message. **Validar contra `entrenamiento.jsonl`/script antes de formatear ChatML; no duplicar fecha/título.** |
| `esquema_eventos_protesta_entrenamiento_MVS.json` | Schema MVS pruned (-42% campos) usado en el fine-tuning. |
| `esquema_eventos_protesta_v1_1_0.json` | Schema v1.1.0 original con el que se generaron las extracciones (referencia y fuente de verdad del schema que produjo el training data). |
| `Codebook.md` | Definiciones operativas de las categorías categóricas. |
| `Ejemplos_Codebook.md` | Casos de aplicación del codebook. |
| `extraer_eventos_protesta_batch.py` | Script CLI OpenAI Batch (para Fase 6: regenerar batch con más datos). |
| `muestra_350_conflictos_1989_1995.csv` | CSV crudo de las 350 notas fuente. |
| `prompt_eventos_protesta_batch_v1_1_0.md` | Versión extendida del codebook operativo (NO es el system prompt que se envió). |
| `PREPROCESSING_NOTES.md` | Notas operativas del pipeline pre-proceso (re-ejecución en Fase 6). |
| `requirements.txt` | Dependencias Python del script OpenAI Batch. |

## ⚠️ Aclaraciones metodológicas antes de Fase 1

1. **Origen formal del training data**:
   - Los 317 ejemplos canónicos de `entrenamiento.jsonl` fueron producidos por **GPT-5.4-mini** y validados humanamente por Nico.
   - Cualquier mención histórica a `gpt-5.5` en artefactos de requests/scripts debe tratarse como ruido documental o referencia operativa para regeneraciones futuras, **no** como origen ni baseline de los 317 ejemplos actuales.
   - Baseline correcto para el plan: **GPT-5.4-mini + validación humana**.

2. ~~Schema `completo.json` vs `v1_1_0.json`~~ **RESUELTO**: `esquema_eventos_protesta_completo.json` ahora es byte-idéntico a `esquema_eventos_protesta_v1_1_0.json` (SHA-256 `D83D4108...`). El archivo `completo.json` fue eliminado del repo por ser duplicado puro; queda solo `v1_1_0.json` como fuente de verdad del schema que produjo el training data.

3. **`metadatos_extraccion.estado_validacion_humana: "No validado"`** es un placeholder engañoso. **Las 317 filas canónicas están validadas por Nico**. La presencia del top-level `validacion_humana` solo discrimina si Nico editó; **NO** discrimina calidad. Tratarlo como gold/silver es incorrecto — todas son gold (weight 1.0).

4. **Formato real del user message**: `extraer_eventos_protesta_batch.py` construye `user` como `RowInput.input_text`: `FECHA DE EDICIÓN DE LA NOTA: {fecha}\n\n{texto}`. En `muestra_350_conflictos_1989_1995.csv`, la columna `texto` ya incluye título + cuerpo. En `entrenamiento.jsonl`, `nota.texto_original` ya contiene ese bloque completo. Para Fase 1, **no concatenar de nuevo `fecha + titulo + texto_original`** porque duplicaría contexto.

## Comandos útiles

- Smoke test del script batch (sin gastar API):
  ```bash
  python extraer_eventos_protesta_batch.py prepare \
    --csv muestra_350_conflictos_1989_1995.csv \
    --schema esquema_eventos_protesta_v1_1_0.json \
    --batch-input smoke_batch_requests.jsonl \
    --manifest smoke_manifest.jsonl \
    --prepare-meta smoke_prepare.json \
    --dry-run
  ```
- Preparar batch completo desde CSV (gasta API al hacer submit; solo para nuevas regeneraciones, no describe el origen de los 317 canónicos actuales):
  ```bash
  python extraer_eventos_protesta_batch.py prepare \
    --csv muestra_350_conflictos_1989_1995.csv \
    --schema esquema_eventos_protesta_v1_1_0.json \
    --batch-input batch_requests_v3.jsonl \
    --manifest batch_manifest_v3.jsonl \
    --prepare-meta batch_prepare_v3.json \
    --manifest-store-text \
    --model gpt-5.4-mini \
    --reasoning-effort medium \
    --verbosity low \
    --max-output-tokens 16000
  ```
- **No ejecutes `submit` ni `download` salvo pedido explícito** — gastan crédito de OpenAI.

## Convenciones del pipeline batch

- El input operativo del script actual es exactamente `FECHA DE EDICIÓN DE LA NOTA: {fecha}\n\n{texto}`. La columna `texto` del CSV ya empieza con el título y luego el cuerpo. `nota.texto_original` en `entrenamiento.jsonl` ya guarda ese bloque completo.
- `custom_id` del batch = `txt_file` (= `nota_id`).
- El script actual puede usar un modelo por defecto para nuevas regeneraciones; eso no cambia que los 317 ejemplos canónicos actuales se tratan formalmente como **GPT-5.4-mini + validación humana**.
- El script fuerza metadatos deterministas al parsear: `nota.nota_id = txt_file`, `nota.archivo_fuente = txt_file`, `nota.fecha_publicacion` en `DD/MM/AAAA`.
- Por defecto `nota.texto_original` queda como `"S/D"`; para conservar texto completo usar `--manifest-store-text` y `--store-full-text`.
- El schema para OpenAI se prepara quitando `$schema`, `$id`, `title` y convirtiendo `const` en `enum`.

## Validaciones que importan

- `prepare` carga el schema JSON y exige raíz `type=object`.
- `assert_strict_schema_shape` exige que todo objeto con `properties` tenga todas sus propiedades en `required` y `additionalProperties: false`.
- El parseo valida cada respuesta contra `jsonschema.Draft202012Validator`; errores van a `*.errors.jsonl` y el comando devuelve código `2`.

## Gotchas del repo

- Las 317 filas canónicas de `entrenamiento.jsonl` están todas validadas por Nico. NO es gold/silver; es un único set de 317 con weight 1.0 universal.
- Los reportes/modelos/checkpoints derivados de la versión previa de 350 filas son históricos y no comparables con los artefactos 317 hasta regenerar entrenamiento y evaluación.
- `metadatos_extraccion.estado_validacion_humana` siempre dice "No validado" — IGNORAR ese campo. La presencia de top-level `validacion_humana` indica edición humana, no mayor/menor calidad.
- Convención crítica `S/D` vs `null`: en el gold v1.1.0, cuando un registro tiene `es_evento_protesta=false`, los campos de detalle del evento deben ser `null`, no `S/D`. Usar `S/D` solo para valores textuales/categoriales desconocidos dentro de eventos reales (`es_evento_protesta=true`).
- Para construir ChatML desde `entrenamiento.jsonl`, usar `nota.texto_original` como user content base o reconstruir desde CSV, pero **no mezclar ambos caminos**.
- Si se modifica cualquier schema, mantener compatibilidad con Structured Outputs estrictos (objetos cerrados, todas las propiedades en `required`).

## Próximo paso inmediato

**Próximo paso después de Fase 1 regenerada**: entrenar/evaluar nuevamente desde los artefactos 317 (`data/mvs_projected.jsonl`, `data/train_validated.jsonl`, `data/eval_set.jsonl`, `data/chat_formatted/*`) antes de comparar métricas. Ver `PLAN_ENTRENAMIENTO_QWEN.md` §5.
