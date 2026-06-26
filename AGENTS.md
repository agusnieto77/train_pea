# Guía para agentes OpenCode

Repositorio de **fine-tuning de Qwen2.5-7B-Instruct como extractor experto de eventos de protesta** a partir de notas periodísticas históricas argentinas. El training data (350 ejemplos) fue producido por un pipeline previo de OpenAI Batch API y validado por Nico.

## Estado del repo (2026-06-26)

Pre-proceso consolidado en la raíz (no hay subcarpetas operativas). El repo está listo para arrancar **Fase 1 — Data prep**; Fase 0 quedó absorbida por los artefactos ya presentes (`v1_1_0` + MVS). Plan completo en `PLAN_ENTRENAMIENTO_QWEN.md`.

## Fuentes de verdad

| Archivo | Rol |
|---|---|
| `PLAN_ENTRENAMIENTO_QWEN.md` | Plan de fine-tuning (7 fases, hiperparámetros, criterios de éxito). **Empezar por acá.** |
| `entrenamiento.jsonl` | 350 ejemplos del training data (todos validados por Nico, weight 1.0). |
| `SYSTEM_PROMPT_GPT5_USADO.md` | System prompt EXACTO (1873 chars) extraído de las 350 requests. **Usar literal** en training e inference. |
| `USER_MESSAGE_TEMPLATE_GPT5.md` | Reconstrucción operativa del user message. **Validar contra `entrenamiento.jsonl`/script antes de formatear ChatML; no duplicar fecha/título.** |
| `esquema_eventos_protesta_entrenamiento_MVS.json` | Schema MVS pruned (-42% campos) usado en el fine-tuning. |
| `esquema_eventos_protesta_v1_1_0.json` | Schema v1.1.0 original con el que se generaron las 350 extracciones (referencia y fuente de verdad del schema que produjo el training data). |
| `Codebook.md` | Definiciones operativas de las categorías categóricas. |
| `Ejemplos_Codebook.md` | Casos de aplicación del codebook. |
| `extraer_eventos_protesta_batch.py` | Script CLI OpenAI Batch (para Fase 6: regenerar batch con más datos). |
| `muestra_350_conflictos_1989_1995.csv` | CSV crudo de las 350 notas fuente. |
| `prompt_eventos_protesta_batch_v1_1_0.md` | Versión extendida del codebook operativo (NO es el system prompt que se envió). |
| `PREPROCESSING_NOTES.md` | Notas operativas del pipeline pre-proceso (re-ejecución en Fase 6). |
| `requirements.txt` | Dependencias Python del script OpenAI Batch. |

## ⚠️ Discrepancias a flagear antes de Fase 1

1. **Modelo en `body.model` vs `metadatos_extraccion.modelo`**:
   - `batch_requests_eventos_protesta.jsonl` (las 350 requests enviadas): `gpt-5.5`
   - `entrenamiento.jsonl` metadata: `gpt-5.4-mini`
   - Si son modelos distintos (no aliases), auditoría queda bajo duda.

2. ~~Schema `completo.json` vs `v1_1_0.json`~~ **RESUELTO**: `esquema_eventos_protesta_completo.json` ahora es byte-idéntico a `esquema_eventos_protesta_v1_1_0.json` (SHA-256 `D83D4108...`). El archivo `completo.json` fue eliminado del repo por ser duplicado puro; queda solo `v1_1_0.json` como fuente de verdad del schema que produjo el training data.

3. **`metadatos_extraccion.estado_validacion_humana: "No validado"`** es un placeholder engañoso en las 350 filas. **Las 350 están validadas por Nico** (162 con `validacion_humana.modificada: true`, 188 aprobadas sin edición y sin bloque `validacion_humana`). La presencia del top-level `validacion_humana` solo discrimina si Nico editó; **NO** discrimina calidad. Tratarlo como gold/silver es incorrecto — todas son gold (weight 1.0).

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
- Preparar batch completo desde CSV (gasta API al hacer submit):
  ```bash
  python extraer_eventos_protesta_batch.py prepare \
    --csv muestra_350_conflictos_1989_1995.csv \
    --schema esquema_eventos_protesta_v1_1_0.json \
    --batch-input batch_requests_v3.jsonl \
    --manifest batch_manifest_v3.jsonl \
    --prepare-meta batch_prepare_v3.json \
    --manifest-store-text \
    --model gpt-5.5 \
    --reasoning-effort medium \
    --verbosity low \
    --max-output-tokens 16000
  ```
- **No ejecutes `submit` ni `download` salvo pedido explícito** — gastan crédito de OpenAI.

## Convenciones del pipeline batch

- El input operativo del script actual es exactamente `FECHA DE EDICIÓN DE LA NOTA: {fecha}\n\n{texto}`. La columna `texto` del CSV ya empieza con el título y luego el cuerpo. `nota.texto_original` en `entrenamiento.jsonl` ya guarda ese bloque completo.
- `custom_id` del batch = `txt_file` (= `nota_id`).
- El script actual usa `DEFAULT_MODEL = "gpt-5.5"` y prompt embebido por defecto (`DEFAULT_PROMPT_FILE = ""`) para que `prepare --dry-run` funcione sin carpetas externas.
- El script fuerza metadatos deterministas al parsear: `nota.nota_id = txt_file`, `nota.archivo_fuente = txt_file`, `nota.fecha_publicacion` en `DD/MM/AAAA`.
- Por defecto `nota.texto_original` queda como `"S/D"`; para conservar texto completo usar `--manifest-store-text` y `--store-full-text`.
- El schema para OpenAI se prepara quitando `$schema`, `$id`, `title` y convirtiendo `const` en `enum`.

## Validaciones que importan

- `prepare` carga el schema JSON y exige raíz `type=object`.
- `assert_strict_schema_shape` exige que todo objeto con `properties` tenga todas sus propiedades en `required` y `additionalProperties: false`.
- El parseo valida cada respuesta contra `jsonschema.Draft202012Validator`; errores van a `*.errors.jsonl` y el comando devuelve código `2`.

## Gotchas del repo

- Las 350 filas de `entrenamiento.jsonl` están todas validadas por Nico. NO es gold/silver; es un único set de 350 con weight 1.0 universal.
- `metadatos_extraccion.estado_validacion_humana` siempre dice "No validado" — IGNORAR ese campo. La presencia de top-level `validacion_humana` indica edición humana, no mayor/menor calidad.
- Para construir ChatML desde `entrenamiento.jsonl`, usar `nota.texto_original` como user content base o reconstruir desde CSV, pero **no mezclar ambos caminos**.
- Si se modifica cualquier schema, mantener compatibilidad con Structured Outputs estrictos (objetos cerrados, todas las propiedades en `required`).

## Próximo paso inmediato

**Fase 1 — Data prep**: proyectar las 350 filas de `entrenamiento.jsonl` al MVS schema (`esquema_eventos_protesta_entrenamiento_MVS.json`), split 90/10 estratificado, formatear ChatML con `SYSTEM_PROMPT_GPT5_USADO.md` + `USER_MESSAGE_TEMPLATE_GPT5.md`. Ver `PLAN_ENTRENAMIENTO_QWEN.md` §5.
