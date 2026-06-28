# Plan de fine-tuning — Qwen2.5-7B-Instruct extractor de eventos de protesta

**Repo:** `G:\PROTESTA_EXTRACT\train_pea`
**Fecha:** 2026-06-26
**Status:** Plan actualizado al dataset canónico de 317 filas. Fase 1 — Data prep fue regenerada desde `entrenamiento.jsonl`; Fase 0 quedó materializada en los schemas ya presentes en la raíz. Los artefactos previos de training/eval sobre 350 filas son históricos hasta reentrenar/evaluar nuevamente.

---

## 1. Objetivo

Reemplazar la dependencia de OpenAI Batch API por un modelo local open-source (**Qwen2.5-7B-Instruct**) fine-tuneado como extractor experto de eventos de protesta, capaz de producir JSON estructurado contra un esquema podado (MVS) que preserva las 5W del codebook. El training data formal actual son **317 ejemplos canónicos de `entrenamiento.jsonl`, producidos por GPT-5.4-mini y validados humanamente por Nico**.

**Por qué ahora:**
- Costo recurrente de OpenAI Batch API; los 317 ejemplos canónicos ya están validados por Nico y no deben tratarse como gold/silver.
- Necesidad de reentrenar rápido cuando entren nuevas validaciones humanas de Nico.
- Latencia, reproducibilidad y soberanía de datos.

---

## 2. Decisiones locked (no se reabren sin experimentación)

| Eje | Decisión | Rationale |
|---|---|---|
| Modelo base | **Qwen2.5-7B-Instruct** (no Qwen3, no 14B) | Cabe cómodo en QLoRA sobre RTX 5090 32GB; instruct tuning maduro; buen seguimiento de schema en JSON. |
| Datos | **317 canónicos validados por Nico** | TODOS los ejemplos canónicos del jsonl fueron visados y aprobados por Nico. La presencia de `validacion_humana` NO es gold/silver — refleja si Nico editó, no calidad. **Weight 1.0 para los 317.** |
| Hardware MVP | **RTX 5090 32GB local**, QLoRA, sin nube | $0 compute; el usuario aprobó gastar pero el MVP cabe local |
| Método | **SFT con LoRA** en chat-format (no DPO en MVP) | Más simple, suficiente para MVP; DPO es fase 6 |
| Schema | **MVS pruned** (-42% campos, -40% tokens output) | Reduce overfitting en campos ruidosos, mejora batch efectivo |
| Sampling | `completion_only_loss=True`, `packing=False` | Estándar para SFT |
| Optimizer | `paged_adamw_8bit`, lr 2e-4 cosine, warmup 5% | Cabe en 32GB sin OOM |

---

## 3. Esquema MVS — qué se queda, qué se va

Preserva las **5W** (qué, cuándo, cómo, dónde, por qué) y poda ruido de validación humana.

### KEEP (campos retenidos)

**Top-level**
- `schema_version` (const "1.0.0")
- `nota`: `nota_id`, `fecha_publicacion` (DD/MM/AAAA), `titulo`
- `extraccion`: `tiene_eventos_protesta`, `total_eventos_protesta`, `eventos_protesta[]`

**Por evento (`EventoProtesta`)**
- `evento_id`, `evento_numero`, `es_evento_protesta`
- `delimitacion_evento`: `descripcion_sintetica`, `criterio_delimitacion`, `es_accion_principal_con_complementarias`, `cita_textual_evento`
- `temporalidad`: `tipo_temporal`, `tempo_verbal`, `fecha_inicio` (valor+certeza+cita), `fecha_fin` (valor+certeza+cita), `expresion_temporal_textual`, `fecha_publicacion_usada_como_referencia`
- `accion`: `descripcion_textual`, `formato_principal` (cita+valor_textual+categoria), `formatos_complementarios[]` (valor_textual+categoria)
- `sujetos[]`: `sujeto_id`, `cita_textual`, `nombre_textual`, `categoria`, `organizaciones[]` (organizacion_id+nombre_textual+categoria)
- `demandas[]`: `demanda_id`, `cita_textual`, `descripcion_textual`, `categoria`, `dirigida_a_contra_ids[]`
- `contra_quien[]`: `contra_id`, `nombre_textual`, `categoria`, `nivel_institucional`
- `lugares[]`: `lugar_id`, `nombre_textual`, `categoria`, `rol_en_evento`, `localidad`, `provincia`, `pais`, `direccion_o_referencia`
- `alcance`: `categoria` (único)
- `cantidad_participantes`: `hay_cantidad_mencionada`, `valor`, `valor_textual`, `es_aproximada`, `cita_textual`
- `incidentes`: `represion` (presencia+valor_booleano+descripcion), `enfrentamiento` (idem), `danios_materiales` (idem), `detenidos` (presencia+valor+valor_textual), `heridos` (idem), `muertos` (idem)

### DROP (campos eliminados en MVS)

- **Todos los `razonamiento_*`** (justificaciones en prosa): baja señal de training, inflar tokens output, riesgo de hallucination
- **`calidad_extraccion.*`** completa: `confianza_global`, `campos_inferidos`, `ambiguedades`, `advertencias_atomicidad`
- **`voces_protagonistas[]`**, **`personas_mencionadas[]`**: no son centrales para las 5W
- **`observaciones_extraccion`** (top-level): ruido
- **`metadatos_extraccion`** y **`validacion_humana`**: metadatos de proceso, no de evento
- **`fuente_de_la_cifra`** (en `cantidad_participantes`): opcional, baja cobertura
- **`cita_textual`/`razonamiento`** específicos de: demanda, lugar, contra_quien, alcance, incidentes.*
- **Input `texto_original`, `subtitulo`, `fuente`, `archivo_fuente`, `nota_id`**: los inyecta el script de extracción (no los debe predecir el modelo)

### Reducción estimada

- **Campos:** 115 → ~70 (-39%)
- **Tokens output por nota:** -40% (medido sobre 50 ejemplos de muestra)
- **Batch efectivo:** +50% por salida MVS más corta; el contexto de entrenamiento queda en 20,480 tokens para no excluir notas largas.

---

## 4. Hiperparámetros QLoRA locked

```yaml
# config.yaml — training
model:
  base: "Qwen/Qwen2.5-7B-Instruct"
  dtype: "bfloat16"
  quantize: "4bit"             # NF4 double quant
  max_seq_length: 20480          # Qwen2.5-7B-Instruct soporta 32,768 en config actual; 20,480 cubre el audit sin YaRN

lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: [q, k, v, o, gate, up, down]
  bias: "none"
  task_type: "CAUSAL_LM"

training:
  method: "SFT"
  epochs: 3
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 24    # batch efectivo = 24; conserva batch efectivo al subir contexto
  learning_rate: 2.0e-4
  lr_scheduler: "cosine"
  warmup_ratio: 0.05
  weight_decay: 0.0
  optimizer: "paged_adamw_8bit"
  gradient_checkpointing: true
  bf16: true
  completion_only_loss: true
  packing: false

data:
  train_validated: "data/train_validated.jsonl"  # 285 ejemplos — TODOS validados por Nico
  eval_set: "data/eval_set.jsonl"                # 32 ejemplos (stratified 90/10 sobre train_validated)
  weighting:
    validated: 1.0                                # todos los ejemplos son gold; no hay silver
  split_strategy: "estratificado por tiene_eventos_protesta + buckets de total_eventos_protesta (0/1/2/3+/evento)"
  chat_format: "ChatML"
  system_prompt_source: "SYSTEM_PROMPT_GPT5_USADO.md (prompt histórico literal del batch; no editar bloque literal)"
  user_template: "USER_MESSAGE_TEMPLATE_GPT5.md"

eval:
  metrics:
    - schema_validity         # JSON parse + Draft202012Validator
    - categorical_accuracy    # enums (formato_accion, categoria_sujeto, etc.)
    - f1_global               # F1 sobre campos validados vs pred
    - field_recall            # recall por campo (qué predice, qué omite)
  baseline_reference: "OpenAI extraction baseline: GPT-5.4-mini + validación humana Nico"
```

**Soporte de contexto verificado:** el model card oficial de `Qwen/Qwen2.5-7B-Instruct` declara contexto completo hasta 131,072 tokens y `config.json` actual hasta 32,768 tokens sin YaRN. Como `chatml_audit.json` detectó ejemplos largos hasta ~16.4k tokens por proxy, el plan sube `max_seq_length` a **20,480** para no dejar ejemplos afuera y sin activar YaRN.

**Memoria estimada (RTX 5090 32GB):**
- Modelo 4-bit: ~5 GB
- LoRA + grads + optimizer: ~3 GB
- Activaciones con gradient checkpointing @ seq 20480 batch 1: comparable o algo mayor que el plan original; smoke test obligatorio antes del run largo.
- Headroom: dependiente de implementación/kernel; si OOM, mantener `max_seq_length=20480` y bajar microbatch/activar optimizaciones, NO recortar ejemplos.

---

## 5. Fases

### Fase 0 — Audit schema + MVS (completada / referencia)

**Objetivo:** conservar el schema original v1.1.0 y producir el schema MVS de entrenamiento validado.

**Estado actual:** los archivos operativos están en la raíz:
1. `esquema_eventos_protesta_v1_1_0.json` — schema completo usado como referencia del training data.
2. `esquema_eventos_protesta_entrenamiento_MVS.json` — schema MVS pruned para fine-tuning.

**Deliverables:** `esquema_eventos_protesta_v1_1_0.json`, `esquema_eventos_protesta_entrenamiento_MVS.json`.

**Done cuando:** ambos archivos pasan `Draft202012Validator` con 0 errores.

---

### Fase 1 — Data prep (1 día)

**Objetivo:** proyectar los 317 ejemplos canónicos al esquema MVS y formatear ChatML.

**Regla crítica de prompt MVS:** el `system` debe conservar el prompt histórico literal de `SYSTEM_PROMPT_GPT5_USADO.md`, pero agregar inmediatamente después un override explícito para el target MVS:

```text
IMPORTANTE — TARGET MVS PARA ESTE ENTRENAMIENTO:
El output debe limitarse estrictamente al esquema MVS provisto.
Si el prompt histórico menciona campos ausentes del MVS, como
calidad_extraccion.*, observaciones_extraccion, metadatos_extraccion,
validacion_humana, voces_protagonistas o personas_mencionadas,
NO los generes. Representá la ambigüedad usando únicamente los campos
disponibles en MVS y, si no hay evidencia textual suficiente, usá "S/D"
o null según corresponda. Regla crítica: cuando `es_evento_protesta=false`,
los campos de detalle del evento van en `null`, no en `S/D`; `S/D` queda
reservado para valores textuales/categoriales desconocidos dentro de eventos
de protesta reales (`es_evento_protesta=true`).
```

Esto evita enseñar una instrucción imposible: el prompt histórico fue creado para v1.1.0 completo, pero el fine-tuning predice solo MVS.

**Tareas:**
1. `scripts/audit_schema.py` — conteo de campos y enums en el dataset, detección de enums con cobertura <5 ejemplos (alerta para augment).
2. `scripts/proyectar_a_MVS.py` — mapear cada campo del schema original al MVS:
   - DROP: razonamiento_*, calidad_extraccion.*, voces_protagonistas, personas_mencionadas, observaciones_extraccion, metadatos_extraccion, validacion_humana, fuente_de_la_cifra, citatextual/razonamiento de demanda/lugar/contra_quien/alcance/incidentes
   - KEEP: resto
   - Rewrite `nota.texto_original` para que NO esté en el output del modelo (lo inyecta el script al parsear).
3. `scripts/split_train_eval.py` — split 90/10 estratificado por `tiene_eventos_protesta` y buckets de `total_eventos_protesta` (0/1/2/3+/evento).
4. `data/chat_formatter.py` — convierte cada ejemplo proyectado a ChatML:
   - `system`: texto literal de `SYSTEM_PROMPT_GPT5_USADO.md` + override MVS + anexo con paths MVS (JSON pointer list de los campos requeridos).
   - `user`: usar `nota.texto_original` de `entrenamiento.jsonl` como bloque completo de input base; no concatenar de nuevo fecha/título/texto.
   - `assistant`: JSON del evento extraído proyectado a MVS (stringificado).
   - Regla obligatoria: si `es_evento_protesta=false`, los campos de detalle del evento permanecen en `null`; `S/D` solo se usa para desconocidos textuales/categoriales dentro de eventos reales.
5. `scripts/validate_jsonl.py` — valida cada línea contra MVS.

**Gates obligatorios de Fase 1:**

1. `reports/projection_report.json`
   - 317/317 ejemplos procesados.
   - 317/317 outputs proyectados válidos contra `esquema_eventos_protesta_entrenamiento_MVS.json`.
   - 0 violaciones del contrato false-event null en source y output proyectado.
   - 0 campos podados sobrevivientes (`razonamiento_*`, `calidad_extraccion`, `observaciones_extraccion`, `metadatos_extraccion`, `validacion_humana`, etc.).
   - Invariantes: `total_eventos_protesta == len(eventos_protesta)` y `tiene_eventos_protesta == (total_eventos_protesta > 0)`.
2. `reports/split_manifest.json`
   - Seed fijo.
   - Lista explícita de `nota_id` para train/eval.
   - Distribución comparada por `tiene_eventos_protesta`, buckets `total_eventos_protesta` 0/1/2/3+ y, cuando haya eventos, `accion.formato_principal.categoria`.
3. `reports/chatml_audit.json`
   - Hash del system prompt completo usado en ChatML (prompt histórico + override MVS + anexo MVS).
   - Verificación de que `user` usa `nota.texto_original` sin duplicar fecha/título/texto.
    - Verificación de que `assistant` no contiene `nota.texto_original` ni campos fuera del MVS.
    - Verificación de que `assistant` mantiene `null` en detalles de eventos con `es_evento_protesta=false`.
   - Conteo de tokens input/output y alerta para ejemplos que superen `max_seq_length=20480`.

**Gate de longitud:** si `chatml_audit.json` reporta ejemplos sobre `max_seq_length=20480`, no iniciar Fase 3 hasta resolverlo con una auditoría usando el tokenizer real de Qwen. Opciones aceptables: subir hasta el límite nativo de 32,768 si la memoria lo permite, reducir el anexo MVS, o definir una política explícita para ejemplos largos. No truncar silenciosamente.

**Deliverables:** `data/{train_validated,eval_set}.jsonl`, `data/chat_formatted/{train,eval}.jsonl`, `reports/{projection_report,split_manifest,chatml_audit}.json`.

**Done cuando:** todos los JSONL pasan validación MVS, los tres reportes obligatorios existen sin errores críticos y los splits están balanceados según el manifiesto.

---

### Fase 2 — Baseline Qwen2.5-7B-Instruct (½ día)

**Objetivo:** medir el comportamiento del modelo base (sin fine-tuning) para saber dónde estamos parados.

**Tareas:**
1. Cargar `Qwen/Qwen2.5-7B-Instruct` con vLLM, `guided_json=esquema_eventos_protesta_entrenamiento_MVS.json`.
2. Correr sobre `data/eval_set.jsonl` (32 ejemplos validados).
3. Métricas: schema_validity, categorical_accuracy, F1 global, field_recall.
4. Análisis cualitativo de 5-10 outputs para detectar patrones de falla.

**Hipótesis a validar:**
- schema_validity ≥ 50% (si <50%, replantear antes de Fase 3; guided_json debería garantizar ~100% si está bien armado).
- categorical_accuracy ≥ 30% en formato_accion (prioridad).

**Deliverable:** `metrics/baseline_qwen2.5-7b.json`.

---

### Fase 3 — SFT QLoLA training (½-1 día wall clock)

**Objetivo:** entrenar el modelo fine-tuneado.

**Tareas:**
1. `training/train_sft.py` — script TRL `SFTTrainer` con QLoRA.
2. Cargar `train_validated.jsonl` (285 ejemplos, weight 1.0). NO hay interleave gold/silver — todos los ejemplos son validados por Nico.
3. Checkpoint cada epoch: `checkpoints/qwen-protesta-v1/epoch-{n}/`.
4. Eval loss cada 200 steps. Log en W&B o TensorBoard.
5. Early stop si eval loss sube en epoch 2 (más de 0.05).

**Wall clock estimado (RTX 5090):**
- 285 ejemplos × 3 epochs = 855 forward+backward passes
- ~40 steps/epoch con batch efectivo 24 → ~120 steps total
- ~1-2s/step → **3-5 minutos wall clock** (sorprendentemente rápido)

**Deliverables:** `checkpoints/qwen-protesta-v1/epoch-{1,2,3}/`, `training/logs/`.

**Done cuando:** eval loss decrece monótonamente en train + el mejor checkpoint supera baseline en eval_set.

---

### Fase 4 — Eval post-training (½ día)

**Objetivo:** comparar el modelo fine-tuneado contra baseline y GPT-5.4-mini.

**Tareas:**
1. Cargar el mejor checkpoint en vLLM, `guided_json=MVS`.
2. Correr sobre `data/eval_set.jsonl`.
3. Calcular: schema_validity, categorical_accuracy, F1 global, field_recall.
4. Comparar contra `metrics/baseline_qwen2.5-7b.json` y la referencia OpenAI formal: **GPT-5.4-mini + validación humana Nico**.
5. Análisis cualitativo: 5-10 outputs por categoría de evento (corte, huelga, marcha, etc.) para detectar drift.

**Deliverable:** `metrics/finetuned_qwen-protesta-v1.json` + reporte cualitativo en `metrics/qualitative_report.md`.

---

### Fase 5 — Deploy local (½ día)

**Objetivo:** dejar el modelo utilizable como reemplazo del batch OpenAI.

**Tareas:**
1. Merge LoRA → modelo completo: `scripts/merge_lora.py` → `models/qwen-protesta-v1-merged/`.
2. (Opcional) AWQ int4 quant para inferencia más rápida: `models/qwen-protesta-v1-awq/`.
3. `extraer_eventos_protesta_local.py` — CLI paralela a `extraer_eventos_protesta_batch.py`:
   - `extract --csv notas.csv --output-dir ./out --model models/qwen-protesta-v1-awq --guided-json esquema_eventos_protesta_entrenamiento_MVS.json --max-seq 20480`
   - Mantiene misma estructura de output (manifest, batch_input, responses) para auditoría.
4. vLLM serve: `vllm serve models/qwen-protesta-v1-awq --guided-decoding-backend outlines`.
5. `README_TRAINING.md` — guía operativa: cómo arrancar vLLM, cómo correr extracción, cómo reentrenar.

**Deliverables:** `models/qwen-protesta-v1-{merged,awq}/`, `extraer_eventos_protesta_local.py`, `README_TRAINING.md`.

---

### Fase 6 — Iteración condicional

**Solo si Fase 4 no llega a los criterios de éxito.**

Escalera de mejoras (probar en orden, parar cuando se cumplan los criterios):
1. **Más rank LoRA** (r=32, alpha=64) → más capacidad.
2. **Más epochs** (4-5) → más exposición.
3. **Más datos validados** — correr `extraer_eventos_protesta_batch.py prepare` con el mismo prompt verificado sobre nuevas notas, hacer que Nico valide 50-100 más, weight 1.0 (mismo criterio que los 317 canónicos actuales).
4. **H100 full FT** (si RTX 5090 no alcanza): renting spot.
5. **DPO** sobre pares (validado Nico, pred del modelo base declarado correctamente) con `preference_data.jsonl`.

**Nunca reintroducir campos podados en el schema MVS.** Si hace falta más detalle, abrir schema MVS-extendido (MVS+) como nuevo target.

---

## 6. Criterios de éxito (todos deben cumplirse)

| Métrica | Target | Por qué |
|---|---|---|
| Schema validity | ≥ 95% | Si baja, los JSON no parsean y rompe el pipeline |
| Categorical accuracy (enums) | ≥ 80% | El codebook categórico es lo más caro de validar humano |
| F1 global vs datos validados Nico | ≥ 0.70 | Comparabilidad con el baseline formal GPT-5.4-mini + validación humana |
| Latencia por nota | ≤ 3s en RTX 5090 | Comparable a API para procesamiento en lotes |

Si schema_validity ≥ 95% Y categorical_accuracy ≥ 80% Y F1 ≥ 0.70 → **MVP aceptado**.

Si 2 de 3 → iterar Fase 6 antes de promover a producción.

---

## 7. Riesgos y mitigaciones

| Riesgo | Probabilidad | Impacto | Mitigación |
|---|---|---|---|
| Unsloth no compila en Blackwell sm_120 | Media | Bajo | Fallback a TRL `SFTTrainer` (sin Unsloth). Smoke test 5 min al inicio de Fase 3. |
| Baseline Qwen2.5-7B no produce JSON válido | Baja | Alto | guided_json de vLLM con MVS debería garantizar ~100%. Validar Fase 2 antes de Fase 3. |
| 317 validados no cubren uniformemente los enums | Media | Medio | Conteo en Fase 1; augment solo con política explícita y sin usar GPT-5.5 como origen/baseline del dataset canónico. |
| Overfitting sobre los 317 validados | Baja | Bajo | Weight 1.0 + early stop en Fase 3; si sube eval loss en epoch 2, parar. |
| Mismatch entre output training y output inference (chat template) | Baja | Alto | Usar mismo `tokenizer.apply_chat_template` en train y en vLLM (verificar Fase 5). |
| Categorical drift: modelo colapsa a clase mayoritaria | Media | Alto | Métrica categorical_accuracy por clase en Fase 4; reportar confusion matrix. |

---

## 8. Cronograma

| Fase | Horas | Wall clock | Acumulado |
|---|---|---|---|
| 0 — Audit + MVS | 6h | mismo día | 6h |
| 1 — Data prep | 8h | 1 día | 1.5 días |
| 2 — Baseline | 4h | ½ día | 2 días |
| 3 — Training | 4h setup + 30 min training | ½ día | 2.5 días |
| 4 — Eval | 4h | ½ día | 3 días |
| 5 — Deploy | 4h | ½ día | 3.5 días |
| **Total MVP** | | **~3.5 días** | |

---

## 9. Open items

- [ ] Validar que vLLM `guided_json` soporta todas las features del MVS (arrays con `minItems`, `$ref` recursivo si lo hay).
- [ ] Decidir si el system prompt incluye ejemplos few-shot o solo spec del schema (preferir solo spec para no inflar contexto).
- [ ] Definir política de versionado del modelo: `qwen-protesta-v1` → reentrenar → `v2` cuando cambien enums del codebook o lleguen >50 gold nuevos.

---

## 10. Referencias

- `AGENTS.md` — guía operativa del repo.
- `SYSTEM_PROMPT_GPT5_USADO.md` — system prompt EXACTO (1873 chars) extraído del batch histórico. **Usar literal en training e inference; no editar el bloque literal.**
- `USER_MESSAGE_TEMPLATE_GPT5.md` — plantilla del user message con custom_id + INPUT_NOTA block.
- `entrenamiento.jsonl` — 317 ejemplos canónicos, **todos validados por Nico**. Weight 1.0 para todos.
- `extraer_eventos_protesta_batch.py` — script CLI para regenerar batch (referencia).
- Engram obs `obs-344ce577cb98a800` (`topic_key: train_pea/qwen-finetune-plan`) — fuente de verdad original de este plan.
- Engram memory — corrección del dataset canónico: 317 filas, todas gold.

---

**Aprobado:** 2026-06-26. Última corrección 2026-06-27: dataset canónico 317/317 → weight 1.0 universal; Fase 1 regenerada con split 285/32.
