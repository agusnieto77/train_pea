# Pre-proceso data — extracción de eventos de protesta v1.1.0

Pipeline original que produjo el training data con el que se va a fine-tunear
Qwen2.5-7B-Instruct. Esta carpeta conserva los artefactos de **referencia y
re-ejecución**, no los artefactos operativos ya consumidos.

---

## ¿Qué se borró el 2026-06-26?

Antes de Fase 1, se eliminaron tres archivos cuyo valor era puramente operativo
del run pasado:

| Archivo borrado | Por qué se borró |
|---|---|
| `batch_manifest_eventos_protesta.jsonl` (1.3 MB) | Lo reemplaza `train_pea/entrenamiento.jsonl`, que tiene los mismos `txt_file`+`fecha`+`texto`+`extraccion`+metadata por nota. |
| `batch_meta_eventos_protesta.json` (1.4 KB) | Metadata de la API de OpenAI (`batch_id`, `input_file_id`, `output_file_id`). Ya consumida; no aporta al training. |
| `batch_prepare_eventos_protesta.json` (536 B) | Output local del comando `prepare` del script. Reproducible corriendo `prepare` de nuevo. |

Lo que SÍ se mantiene está abajo.

---

## ¿Qué hay en esta carpeta y por qué?

### Datos fuente

- **`muestra_350_conflictos_1989_1995.csv`** (1.3 MB, 350 filas) — CSV crudo con
  las 350 notas periodísticas usadas para producir el training data. Columnas
  usadas: `txt_file`, `fecha`, `texto`. Es la fuente de verdad de las notas.

### Codebook y esquema

- **`Codebook.md`** (35 KB) — Definiciones operativas de cada categoría
  categórica (formato de acción, sujeto, organización, demanda, contra-quién,
  lugar). **Referencia obligatoria** para entender las salidas del modelo y
  para auditar nuevos lotes.
- **`Ejemplos_Codebook.md`** (24 KB) — Casos de aplicación del codebook con
  notas reales anotadas. Sirve para Few-Shot prompting y para validar que el
  fine-tuning no se desvíe del codebook.
- **`esquema_eventos_protesta_v1_1_0.json`** (38 KB) — JSON Schema v1.1.0
  **completo** (con `razonamiento_*`, `calidad_extraccion`, `voces_protagonistas`,
  `personas_mencionadas`, etc.). Es el schema contra el que se generaron las 350
  extracciones. **No es** el schema que se va a usar en el fine-tuning (eso es
  el MVS en `train_pea/esquema_eventos_protesta_entrenamiento_MVS.json`); se
  conserva como referencia para entender qué información se descartó al podar.

### Prompt y script

- **`prompt_eventos_protesta_batch_v1_1_0.md`** (53 KB) — Versión extendida
  del codebook operativo con instrucciones detalladas para el modelo. NO es
  el system prompt exacto que se envió al batch (esa versión corta está en
  `train_pea/SYSTEM_PROMPT_GPT5_USADO.md`, 1873 chars). Se conserva como
  referencia de las reglas de extracción que motivaron la curación de Nico.
- **`extraer_eventos_protesta_batch.py`** (33 KB) — Script CLI para regenerar
  el batch (`prepare` / `submit` / `status` / `download` / `parse`). Útil para
  Fase 6 cuando se quieran agregar más notas validadas (ver
  `PLAN_ENTRENAMIENTO_QWEN.md` §6).

### Config

- **`requirements.txt`** — Dependencias Python del script
  (`openai>=1.99.0`, `pandas>=2.0.0`, `jsonschema>=4.22.0`).

---

## Archivos que viven AHORA en `train_pea/` (no en esta carpeta)

Estos son los deliverables del fine-tuning, derivados del pre-proceso:

- `train_pea/SYSTEM_PROMPT_GPT5_USADO.md` — System prompt EXACTO (1873 chars)
  extraído de las 350 filas de `batch_requests_eventos_protesta.jsonl`.
  Verificado único (hash SHA-256). **Usar literal** en training e inference.
- `train_pea/USER_MESSAGE_TEMPLATE_GPT5.md` — Plantilla del user message
  operativo (`FECHA DE EDICIÓN...` + texto CSV, donde el texto ya incluye título + cuerpo).
- `train_pea/entrenamiento.jsonl` — 350 ejemplos del training data
  (notas + extracciones validadas por Nico).
- ~~`train_pea/esquema_eventos_protesta_completo.json`~~ — Eliminado el 2026-06-26 por ser byte-idéntico a `esquema_eventos_protesta_v1_1_0.json` (SHA-256 `D83D4108...`). Usar solo `v1_1_0.json`.
- `train_pea/esquema_eventos_protesta_entrenamiento_MVS.json` — Schema MVS
  pruned (-42% campos) que se va a usar en el fine-tuning.
- `train_pea/PLAN_ENTRENAMIENTO_QWEN.md` — Plan de fine-tuning (7 fases,
  hiperparámetros QLoRA locked, criterios de éxito, cronograma).

---

## ⚠️ Origen formal del modelo base de los datos

| Fuente | Criterio metodológico |
|---|---|
| `train_pea/entrenamiento.jsonl` | Origen formal: `gpt-5.4-mini` + validación humana Nico |
| Menciones históricas a `gpt-5.5` en requests/scripts | Ruido documental o configuración para regeneraciones futuras; no baseline de los 350 actuales |

Decisión metodológica: para Fase 1, los 350 ejemplos actuales se tratan como
**GPT-5.4-mini + validación humana Nico**. No usar `gpt-5.5` como origen ni
baseline del training set actual.

---

## Validación de los datos

Las 350 filas de `entrenamiento.jsonl` están **todas validadas por Nico**:

- 162 tienen `validacion_humana.modificada: true` (Nico abrió el editor y
  modificó la salida del modelo).
- 188 NO tienen `validacion_humana` (Nico las revisó y aprobó como-estaban,
  sin necesidad de editar).

El campo `metadatos_extraccion.estado_validacion_humana` dice "No validado" en
las 350 filas — es un placeholder del post-procesador que miente y NO debe
usarse como discriminador. El discriminador real es la **presencia** del
top-level `validacion_humana`.

**Implicancia para fine-tuning:** weight 1.0 para las 350 (no gold/silver).
Ver `PLAN_ENTRENAMIENTO_QWEN.md` §4 actualizado.

---

## Reproducir el pre-proceso (Fase 6 — más datos)

```bash
# 1. Preparar batch con notas nuevas
python extraer_eventos_protesta_batch.py prepare \
  --csv <csv_nuevas_notas>.csv \
  --schema esquema_eventos_protesta_v1_1_0.json \
  --batch-input batch_requests_v3.jsonl \
  --manifest batch_manifest_v3.jsonl \
  --prepare-meta batch_prepare_v3.json \
  --manifest-store-text \
  --model gpt-5.4-mini \
  --reasoning-effort medium \
  --verbosity low \
  --max-output-tokens 16000

# 2. Subir (gasta API)
python extraer_eventos_protesta_batch.py submit \
  --batch-input batch_requests_v3.jsonl \
  --batch-meta batch_meta_v3.json

# 3. Monitorear
python extraer_eventos_protesta_batch.py status --batch-meta batch_meta_v3.json

# 4. Descargar y parsear
python extraer_eventos_protesta_batch.py download \
  --batch-id <batch_id> \
  --manifest batch_manifest_v3.jsonl \
  --schema esquema_eventos_protesta_v1_1_0.json \
  --out respuestas_v3.jsonl \
  --raw-out respuestas_v3_raw.jsonl \
  --errors-out respuestas_v3_errors.jsonl
```

Después: Nico valida cada nueva nota (revisión + edición opcional) y se
   agregan a `entrenamiento.jsonl` antes de la próxima ronda de fine-tuning.

**Gotcha:** el script actual arma el user message como
`FECHA DE EDICIÓN DE LA NOTA: {fecha}\n\n{texto}`. En el CSV, `texto` ya contiene
título + cuerpo; no reconstruirlo como `fecha + titulo + texto_original` si se parte
de `entrenamiento.jsonl`.

---

## Instalación

```bash
pip install -r requirements.txt
```

---

**Última actualización:** 2026-06-26. Cleanup previo a Fase 1.
