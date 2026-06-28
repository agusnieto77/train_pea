# Continuar entrenamiento Qwen en la PC de training

Este documento es el handoff operativo para seguir el proyecto en la máquina donde se va a entrenar el modelo. La idea es que no haya que reconstruir decisiones ni contexto.

## Estado actual

La **Fase 1 — Data prep** quedó implementada y regenerada localmente desde el dataset canónico actual de **317 filas**.

Ya existen scripts y artefactos para:

1. Proyectar `entrenamiento.jsonl` desde schema v1.1.0 al schema MVS.
2. Crear split reproducible train/eval.
3. Formatear ejemplos ChatML para SFT.
4. Auditar proyección, split y ChatML.

## Decisiones metodológicas cerradas

| Tema | Decisión |
|---|---|
| Origen formal del dataset | **GPT-5.4-mini + validación humana de Nico** |
| Estado de las 317 filas canónicas | Todas son gold, weight `1.0` |
| Artefactos previos sobre 350 filas | Históricos/superseded; no comparar hasta reentrenar/evaluar con 317 |
| `gpt-5.5` | Ruido documental / no baseline / no origen de los 317 canónicos actuales |
| Target schema | `esquema_eventos_protesta_entrenamiento_MVS.json` |
| User message | Usar `nota.texto_original` tal como está; no reconstruir ni duplicar fecha/título/texto |
| Assistant output | JSON MVS compacto, sin `nota.texto_original` ni campos podados |
| `S/D` vs `null` | En gold v1.1.0, si `es_evento_protesta=false`, los campos de detalle del evento van en `null`; `S/D` se reserva para valores textuales/categoriales desconocidos dentro de eventos reales (`es_evento_protesta=true`) |
| Context length | `max_seq_length = 20480`; no truncar ejemplos silenciosamente |
| Modelo base | `Qwen/Qwen2.5-7B-Instruct` |

## Archivos fuente importantes

| Archivo | Uso |
|---|---|
| `PLAN_ENTRENAMIENTO_QWEN.md` | Plan maestro actualizado |
| `entrenamiento.jsonl` | 317 ejemplos canónicos validados |
| `esquema_eventos_protesta_entrenamiento_MVS.json` | Schema de entrenamiento |
| `SYSTEM_PROMPT_GPT5_USADO.md` | Prompt histórico exacto |
| `USER_MESSAGE_TEMPLATE_GPT5.md` | Regla para el user message |
| `AGENTS.md` | Gotchas del repo |

## Scripts agregados

| Script | Qué hace |
|---|---|
| `scripts/proyectar_a_MVS.py` | Proyecta v1.1.0 → MVS, valida contra schema y genera reporte |
| `scripts/split_train_eval.py` | Crea split 285/32 reproducible con seed 42 |
| `data/chat_formatter.py` | Genera ChatML con prompt histórico + override MVS + anexo MVS |

## Artefactos generados

| Artefacto | Estado |
|---|---|
| `data/mvs_projected.jsonl` | 317/317 válidos contra MVS; false-event details null |
| `data/train_validated.jsonl` | 285 ejemplos |
| `data/eval_set.jsonl` | 32 ejemplos |
| `data/chat_formatted/train.jsonl` | 285 ejemplos ChatML |
| `data/chat_formatted/eval.jsonl` | 32 ejemplos ChatML |
| `reports/projection_report.json` | Proyección OK |
| `reports/split_manifest.json` | Split reproducible OK |
| `reports/chatml_audit.json` | ChatML OK, sin over-limit con `max_seq_length=20480` |

## Reproducir Fase 1 completa

Desde la raíz del repo:

```bash
python scripts/proyectar_a_MVS.py
python scripts/split_train_eval.py
python data/chat_formatter.py
```

Resultado esperado:

```text
Projection OK: 317/317 valid.
Split OK: 285 train / 32 eval (seed=42).
ChatML OK.
```

## Verificaciones mínimas antes de entrenar

```bash
python -m py_compile scripts/proyectar_a_MVS.py scripts/split_train_eval.py data/chat_formatter.py
```

Revisar:

```bash
cat reports/projection_report.json
cat reports/split_manifest.json
cat reports/chatml_audit.json
```

Puntos críticos:

- `projection_report.json` debe decir:
  - `processed: 317`
  - `valid: 317`
  - `invalid: 0`
  - `all_valid: true`
  - `false_event_null_contract.pass: true`
  - `projected_rows_with_non_null_details: 0`
- `split_manifest.json` debe decir:
  - train: 285
  - eval: 32
  - seed: 42
- `chatml_audit.json` debe decir:
  - train written: 285
  - eval written: 32
  - `assistant_schema_invalid: []`
  - `assistant_forbidden_paths: []`
  - `assistant_false_event_detail_violations: []`
  - `over_limit: []`

## Context length

Se subió `max_seq_length` a **20480** para no dejar ejemplos afuera.

Verificación usada:

- Model card oficial de `Qwen/Qwen2.5-7B-Instruct`:
  - config actual hasta **32,768 tokens**
  - full context hasta **131,072 tokens**
- Nuestro audit real con tokenizer Qwen reportó:
  - train max real: 18,465 (285 ejemplos)
  - eval max real: 6,383 (32 ejemplos)
  - overall max real: 18,465 (317 ejemplos)
  - over-limit: 0

Por eso **20480 alcanza sin YaRN**.

Regla: si hay OOM, bajar microbatch o activar optimizaciones. **No truncar ejemplos silenciosamente.**

## Tokenizer audit real

La auditoría real de tokenizer ya fue corrida con `scripts/audit_qwen_tokens.py` sobre los artefactos 317 y quedó en `reports/qwen_tokenizer_audit.json`.

### Entorno recomendado

Usar Python 3.11 o 3.12, no Python 3.13 salvo que todo el stack CUDA/TRL esté probado.

Instalación base sugerida:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux
# .venv\Scripts\activate   # Windows PowerShell

pip install -U pip
pip install "transformers>=4.37.0" "huggingface-hub<1.0" datasets accelerate peft trl bitsandbytes jsonschema
```

Si se usa vLLM para evaluación/inferencia:

```bash
pip install vllm
```

El gate vigente es:

```text
max real tokens <= 20480
```

Si supera 20480, subir hasta 32768 si memoria lo permite. No recortar.

## Config de entrenamiento actualizada

El plan quedó con:

```yaml
model:
  base: "Qwen/Qwen2.5-7B-Instruct"
  dtype: "bfloat16"
  quantize: "4bit"
  max_seq_length: 20480

training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 24
  completion_only_loss: true
  packing: false
```

Rationale: al subir contexto, bajamos microbatch a 1 y conservamos batch efectivo 24 con accumulation.

## Orden recomendado al retomar

1. Crear entorno limpio.
2. Re-ejecutar Fase 1 completa si cambia `entrenamiento.jsonl` o el schema.
3. Auditar tokens reales con tokenizer Qwen.
4. Si tokens reales <= 20480, seguir con Fase 2 baseline Qwen sin fine-tuning.
5. Recién después correr Fase 3 SFT/QLoRA.

## No hacer

- No mezclar `nota.texto_original` con reconstrucción desde CSV.
- No duplicar fecha/título/texto en el user message.
- No usar `gpt-5.5` como baseline de los 317 ejemplos canónicos.
- No bajar weights a ejemplos sin `validacion_humana`.
- No truncar ejemplos largos silenciosamente.
- No reintroducir campos podados del MVS.
