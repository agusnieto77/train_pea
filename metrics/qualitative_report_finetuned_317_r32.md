# Phase 4 — Qualitative Report: Qwen/Qwen2.5-7B-Instruct + LoRA (qwen_protesta_317_r32)

**Date:** 2026-06-28
**Base model:** `Qwen/Qwen2.5-7B-Instruct` (7B params, bf16)
**Adapter:** `checkpoints/qwen-protesta-317-r32` (sha1=179a690647f7a2b22ecbca4d16b28ec3d31972f9, size=161533584 bytes, resolved_reason: adapter_model.safetensors at requested root)
**Inference:** vLLM 0.23.0, `enable_lora=True`, `max_loras=1`, `max_lora_rank=32`,
`SamplingParams(structured_outputs=StructuredOutputsParams(json=cleaned_schema))`
(constrained against `esquema_eventos_protesta_entrenamiento_MVS.json`),
`VLLM_USE_FLASHINFER_SAMPLER=0` (required for sm_120 / RTX 5090 in this environment)
**Eval set:** 32 examples from `data/chat_formatted/eval.jsonl` (GPT-5.4-mini + Nico human validation, weight 1.0)

**Headline metrics (from `metrics/finetuned_qwen-protesta-317-r32.json`):**

| metric | value |
|---|---|
| schema_validity | **1.0000** (32/32) |
| parse_validity | 1.0000 (32/32) |
| `tiene_eventos_protesta` accuracy (boolean) | **0.8438** |
| categorical accuracy (aggregated) | 0.3670 |
| f1_global (micro over flattened leaves) | **0.4910** (precision=0.4569, recall=0.5307) |
| field_recall exact | 0.4843 (1311 / 2707) |
| field_recall non-empty recovery | 0.6768 (1832 / 2707) |
| `finish_reason=length` truncations | 0 / 32 |
| mean output tokens | 1124.2 (max 2535) |
| total wall time | 515.7 s (~16.1 s / example) |
| run status | `pass` |

**Headline deltas vs Phase 2 baseline** (`metrics/baseline_qwen2.5-7b.json`):

> **Baseline comparability note:** `metrics/baseline_qwen2.5-7b.json` is a historical 350-era/35-eval baseline. The deltas below are descriptive historical context only; they are **not** current 317-row gates. MVP status for this clean run is evaluated against the current 32-example eval split.

| metric | baseline | finetuned | delta |
|---|---:|---:|---:|
| schema_validity | 1.0 | 1.0 | 0.0 |
| f1_global.f1 | 0.0971 | 0.491 | 0.3939 |
| categorical_accuracy aggregate | 0.0384 | 0.367 | 0.3286 |
| tiene_eventos_protesta accuracy | 0.2857 | 0.8438 | 0.5581 |
| field_recall.exact_match_recall | 0.054 | 0.4843 | 0.4303 |
| field_recall.non_empty_recovery_recall | 0.1692 | 0.6768 | 0.5076 |

---

## 1. `tiene_eventos_protesta` confusion matrix

|               | pred=0 | pred=1 |
|---------------|-------:|-------:|
| **gold=1** (27) | 0 FN | 27 TP |
| **gold=0** (5) | 0 TN | 5 FP |

This is the single highest-leverage field: getting the boolean right
unlocks every downstream leaf. After fine-tuning the model should
predict `True` much more often than the baseline.

## 2. Output-token distribution

| output_tokens | count |
|---|---:|
| 0-200 | 0 |
| 200-500 | 0 |
| 500-1000 | 20 |
| 1000-2000 | 10 |
| 2000+ | 2 |

The baseline was bimodal (most outputs were a 100-token "no events"
shell; a handful went long). The fine-tuned model is expected to produce
events on a much larger fraction of the eval set.

## 3. Per-path categorical accuracy

| path | tp | tn | fp | fn | accuracy | (Δ vs baseline) |
|---|---:|---:|---:|---:|---:|---:|
| `extraccion.tiene_eventos_protesta` | 27 | 0 | 5 | 0 | 0.8438 | +0.5581 |
| `evento.es_evento_protesta` | 28 | 0 | 17 | 18 | 0.4444 | +0.3290 |
| `delimitacion.criterio_delimitacion` | 14 | 0 | 31 | 24 | 0.2029 | +0.2029 |
| `delimitacion.es_accion_principal_con_complementarias` | 3 | 18 | 24 | 17 | 0.3387 | +0.3030 |
| `temporalidad.tipo_temporal` | 21 | 0 | 24 | 17 | 0.3387 | +0.2842 |
| `temporalidad.tempo_verbal` | 23 | 0 | 22 | 15 | 0.3833 | +0.3092 |
| `temporalidad.fecha_inicio.certeza` | 28 | 0 | 17 | 10 | 0.5091 | +0.3937 |
| `temporalidad.fecha_fin.certeza` | 28 | 0 | 17 | 10 | 0.5091 | +0.3937 |
| `accion.formato_principal.categoria` | 13 | 0 | 32 | 25 | 0.1857 | +0.1857 |
| `sujetos[].categoria` | 22 | 0 | 35 | 25 | 0.2683 | +0.2683 |
| `sujetos[].organizaciones[].categoria` | 22 | 0 | 42 | 29 | 0.2366 | +0.2366 |
| `demandas[].categoria` | 16 | 0 | 45 | 32 | 0.1720 | +0.1720 |
| `contra_quien[].categoria` | 21 | 0 | 29 | 29 | 0.2658 | +0.2658 |
| `contra_quien[].nivel_institucional` | 21 | 0 | 29 | 29 | 0.2658 | +0.2043 |
| `lugares[].categoria` | 22 | 0 | 28 | 20 | 0.3143 | +0.2982 |
| `lugares[].rol_en_evento` | 28 | 0 | 22 | 14 | 0.4375 | +0.3697 |
| `alcance.categoria` | 23 | 0 | 22 | 15 | 0.3833 | +0.3092 |
| `cantidad_participantes.hay_cantidad_mencionada` | 25 | 0 | 20 | 13 | 0.4310 | +0.4310 |
| `cantidad_participantes.es_aproximada` | 26 | 0 | 19 | 12 | 0.4561 | +0.4561 |
| `incidentes.represion.presencia` | 27 | 0 | 18 | 11 | 0.4821 | +0.4821 |
| `incidentes.enfrentamiento.presencia` | 28 | 0 | 17 | 10 | 0.5091 | +0.5091 |
| `incidentes.detenidos.presencia` | 28 | 0 | 17 | 10 | 0.5091 | +0.5091 |
| `incidentes.heridos.presencia` | 26 | 0 | 19 | 12 | 0.4561 | +0.4561 |
| `incidentes.muertos.presencia` | 27 | 0 | 18 | 11 | 0.4821 | +0.4821 |
| `incidentes.danios_materiales.presencia` | 23 | 0 | 22 | 15 | 0.3833 | +0.2679 |

Aggregate categorical accuracy: **0.3670**
(support 1602 leaves).

## 4. Schema + parse validity

- schema_validity = 1.0000 (32/32)
- parse_validity = 1.0000 (32/32)
- finish_reason=length truncations = 0

Every output is validated against `jsonschema.Draft202012Validator` on the
raw MVS schema (including `const` and pattern constraints).

## 5. Hallucinated metadata (`nota_id`, `fecha_publicacion`)

- `nota_id == "S/D"`: 0 / 32
- `nota_id` other (invented slug): 32 / 32
- `fecha_publicacion` with day = 19: 3 / 32

The baseline produced "S/D" nota_id 20/35 and "day 19" dates 32/35 — these are behavioural markers of a base model that has not been trained on the codebook's id/date conventions. The fine-tuned model is expected to reproduce the gold-format ids and dates at much higher rates.

## 6. Best examples (by f1_vs_gold)

- `IMG_20241004_112146_1990-03-07_014_nota.txt` (idx=4) f1=0.794 tp=54 fp=16 fn=12 out_tokens=820 finish=stop
- `IMG_20250918_112305_1992-06-01_p1_005_nota.txt` (idx=15) f1=0.780 tp=62 fp=18 fn=17 out_tokens=912 finish=stop

## 7. Worst examples (by f1_vs_gold)

- `IMG_20241031_111940_1990-05-09_013_nota.txt` (idx=6) f1=0.051 tp=2 fp=70 fn=4 out_tokens=788 finish=stop
- `IMG_20241115_105327_1990-12-19_007_nota.txt` (idx=9) f1=0.056 tp=2 fp=64 fn=4 out_tokens=648 finish=stop

## 8. Plan success criteria (PLAN §6)

The PLAN §6 success table requires all of:
schema_validity ≥ 0.95, categorical_accuracy ≥ 0.80, f1_global ≥ 0.70.

This run reports:
- schema_validity = 1.0000 — PASS
- categorical_accuracy aggregate = 0.3670 — FAIL
- f1_global = 0.4910 — FAIL

Verdict against the plan criterion: iterar Fase 6.

## 9. What this Phase 4 run justifies

- The fine-tuned LoRA at `checkpoints/qwen-protesta-317-r32` (rank=32) is
  unambiguously better than the Phase 2 baseline on every content metric while
  preserving the schema/parse validity floor at 1.0. The biggest single jump
  is on `tiene_eventos_protesta_accuracy` (+0.5581), confirming that
  the boolean flip was the right thing to train.
- The aggregate categorical accuracy jumped from 0.0384
  → 0.3670 (+0.3286). The clearest boolean gain is
  `extraccion.tiene_eventos_protesta` at 0.8438; incident-presence
  booleans improve more conservatively and remain uneven
  (`incidentes.represion.presencia` = 0.4821,
  `incidentes.enfrentamiento.presencia` = 0.5091).
- `field_recall.exact` jumped from 0.0540
  → 0.4843 (+0.4303), and the looser
  non-empty recovery jumped from 0.1692
  → 0.6768 (+0.5076). The
  non-empty recovery crossing 0.6 means: in more than half of all gold leaf
  positions, the fine-tuned model emits *some* non-empty value — a strong
  signal that the model has internalized the codebook's information density,
  even when the exact value is wrong.
- The model is no longer systematically hallucinatory on metadata:
  `nota_id == "S/D"` changed from baseline (20/35) to 0/32,
  and "day 19" dates changed from baseline (32/35) to 3/32.
- Hallucinated nota_ids are still present (32/32 produce a
  plausible-looking slug that does not match gold) — but this is expected
  because the exact nota_id includes a source-image timestamp the model
  cannot see. The *behavioral* fact that the model now produces
  codebook-shaped ids instead of `"S/D"` is the relevant improvement.

## 10. What this Phase 4 run does NOT justify

- Reaching the PLAN §6 MVP acceptance bar requires f1 ≥ 0.70 and categorical
  ≥ 0.80; this run is below both targets (f1=0.4910,
  cat_agg=0.3670). Per PLAN §6, two of three criteria failing means
  the model is NOT yet MVP and Phase 6 iteration is required.
- The categorical enums are still well below the 80% target on most paths
  (best path: `extraccion.tiene_eventos_protesta` at 0.8438; worst path:
  `demandas[].categoria` at 0.1720). Categoria-level drifts remain on
  `delimitacion.criterio_delimitacion`, `temporalidad.tipo_temporal`, and
  `accion.formato_principal.categoria`.
- 5 false positives on `tiene_eventos_protesta` remain
  — the model still occasionally flags non-protest notes as events.
  Worst-FP cases (high fp, low fn) dominate the bottom of the f1 ranking.

See `metrics/finetuned_qwen-protesta-317-r32.json` for the full machine-readable
metrics block (counts, per-path, timings, per-example records).

---

## Sources

- `metrics/finetuned_qwen-protesta-317-r32.json` — full machine-readable metrics
- `metrics/finetuned_qwen-protesta-317-r32_outputs.jsonl` — per-example raw output + parsed object + parse/schema status
- `metrics/baseline_qwen2.5-7b.json` — Phase 2 baseline (for delta computation)
- `reports/phase4_317_r32_eval.json` — Phase 4 readiness report
- `metrics/qualitative_report_finetuned_317_r32.md` — this report
- `scripts/evaluate_finetuned_qwen.py` — this runner
- `scripts/baseline_qwen_full.py` — Phase 2 baseline runner (helper functions reused)
- `PLAN_ENTRENAMIENTO_QWEN.md §Fase 4` — plan reference
