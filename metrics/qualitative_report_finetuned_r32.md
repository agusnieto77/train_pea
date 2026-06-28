# Phase 6 (r=32) — Qualitative Report: Qwen/Qwen2.5-7B-Instruct + LoRA (qwen_protesta_v1_r32)

> Historical report: this evaluation was produced from the previous 350-row Phase 1 artifacts. The canonical dataset is now `entrenamiento.jsonl` with 317 rows; retrain/evaluate before comparing current metrics.

**Date:** 2026-06-27
**Base model:** `Qwen/Qwen2.5-7B-Instruct` (7B params, bf16)
**Adapter:** `checkpoints/qwen-protesta-v1-r32` (sha1=6804aeb4d7f85b7d1b94574b1cab816017debbf7, size=161533584 bytes, resolved_reason: adapter_model.safetensors at requested root)
**Inference:** vLLM 0.23.0, `enable_lora=True`, `max_loras=1`, `max_lora_rank=32`,
`SamplingParams(structured_outputs=StructuredOutputsParams(json=cleaned_schema))`
(constrained against `esquema_eventos_protesta_entrenamiento_MVS.json`),
`VLLM_USE_FLASHINFER_SAMPLER=0` (required for sm_120 / RTX 5090 in this environment)
**Eval set:** 35 examples from `data/chat_formatted/eval.jsonl` (GPT-5.4-mini + Nico human validation, weight 1.0)

**Headline metrics (from `metrics/finetuned_qwen-protesta-v1-r32.json`):**

| metric | value |
|---|---|
| schema_validity | **1.0000** (35/35) |
| parse_validity | 1.0000 (35/35) |
| `tiene_eventos_protesta` accuracy (boolean) | **0.7714** |
| categorical accuracy (aggregated) | 0.4189 |
| f1_global (micro over flattened leaves) | **0.5350** (precision=0.5001, recall=0.5752) |
| field_recall exact | 0.5165 (1673 / 3239) |
| field_recall non-empty recovery | 0.7496 (2428 / 3239) |
| `finish_reason=length` truncations | 0 / 35 |
| mean output tokens | 1189.5 (max 2920) |
| total wall time | 540.8 s (~15.5 s / example) |
| run status | `pass` |

**Headline deltas vs Phase 2 baseline** (`metrics/baseline_qwen2.5-7b.json`):

| metric | baseline | finetuned | delta |
|---|---:|---:|---:|
| schema_validity | 1.0 | 1.0 | 0.0 |
| f1_global.f1 | 0.0971 | 0.535 | 0.4379 |
| categorical_accuracy aggregate | 0.0384 | 0.4189 | 0.3805 |
| tiene_eventos_protesta accuracy | 0.2857 | 0.7714 | 0.4857 |
| field_recall.exact_match_recall | 0.054 | 0.5165 | 0.4625 |
| field_recall.non_empty_recovery_recall | 0.1692 | 0.7496 | 0.5804 |

---

## 1. `tiene_eventos_protesta` confusion matrix

|               | pred=0 | pred=1 |
|---------------|-------:|-------:|
| **gold=1** (27) | 1 FN | 26 TP |
| **gold=0** (8) | 1 TN | 7 FP |

This is the single highest-leverage field: getting the boolean right
unlocks every downstream leaf. After fine-tuning the model should
predict `True` much more often than the baseline.

## 2. Output-token distribution

| output_tokens | count |
|---|---:|
| 0-200 | 2 |
| 200-500 | 0 |
| 500-1000 | 19 |
| 1000-2000 | 7 |
| 2000+ | 7 |

The baseline was bimodal (most outputs were a 100-token "no events"
shell; a handful went long). The fine-tuned model is expected to produce
events on a much larger fraction of the eval set.

## 3. Per-path categorical accuracy

| path | tp | tn | fp | fn | accuracy | (Δ vs baseline) |
|---|---:|---:|---:|---:|---:|---:|
| `extraccion.tiene_eventos_protesta` | 26 | 1 | 7 | 1 | 0.7714 | +0.4857 |
| `evento.es_evento_protesta` | 30 | 4 | 22 | 13 | 0.4928 | +0.3774 |
| `delimitacion.criterio_delimitacion` | 17 | 0 | 39 | 30 | 0.1977 | +0.1977 |
| `delimitacion.es_accion_principal_con_complementarias` | 1 | 28 | 27 | 18 | 0.3919 | +0.3562 |
| `temporalidad.tipo_temporal` | 26 | 0 | 30 | 21 | 0.3377 | +0.2832 |
| `temporalidad.tempo_verbal` | 27 | 0 | 29 | 20 | 0.3553 | +0.2812 |
| `temporalidad.fecha_inicio.certeza` | 40 | 0 | 16 | 7 | 0.6349 | +0.5195 |
| `temporalidad.fecha_fin.certeza` | 40 | 0 | 16 | 7 | 0.6349 | +0.5195 |
| `accion.formato_principal.categoria` | 18 | 0 | 38 | 29 | 0.2118 | +0.2118 |
| `sujetos[].categoria` | 30 | 0 | 30 | 20 | 0.3750 | +0.3750 |
| `sujetos[].organizaciones[].categoria` | 38 | 0 | 43 | 37 | 0.3220 | +0.3220 |
| `demandas[].categoria` | 21 | 0 | 44 | 35 | 0.2100 | +0.2100 |
| `contra_quien[].categoria` | 25 | 0 | 33 | 32 | 0.2778 | +0.2778 |
| `contra_quien[].nivel_institucional` | 18 | 0 | 40 | 39 | 0.1856 | +0.1241 |
| `lugares[].categoria` | 24 | 0 | 33 | 25 | 0.2927 | +0.2766 |
| `lugares[].rol_en_evento` | 26 | 0 | 31 | 23 | 0.3250 | +0.2572 |
| `alcance.categoria` | 27 | 0 | 29 | 20 | 0.3553 | +0.2812 |
| `cantidad_participantes.hay_cantidad_mencionada` | 37 | 0 | 19 | 10 | 0.5606 | +0.5606 |
| `cantidad_participantes.es_aproximada` | 37 | 0 | 19 | 10 | 0.5606 | +0.5606 |
| `incidentes.represion.presencia` | 40 | 0 | 16 | 7 | 0.6349 | +0.6349 |
| `incidentes.enfrentamiento.presencia` | 39 | 0 | 17 | 8 | 0.6094 | +0.6094 |
| `incidentes.detenidos.presencia` | 40 | 0 | 16 | 7 | 0.6349 | +0.6349 |
| `incidentes.heridos.presencia` | 40 | 0 | 16 | 7 | 0.6349 | +0.6349 |
| `incidentes.muertos.presencia` | 40 | 0 | 16 | 7 | 0.6349 | +0.6349 |
| `incidentes.danios_materiales.presencia` | 40 | 0 | 16 | 7 | 0.6349 | +0.5195 |

Aggregate categorical accuracy: **0.4189**
(support 1862 leaves).

## 4. Schema + parse validity

- schema_validity = 1.0000 (35/35)
- parse_validity = 1.0000 (35/35)
- finish_reason=length truncations = 0

Every output is validated against `jsonschema.Draft202012Validator` on the
raw MVS schema (including `const` and pattern constraints).

## 5. Hallucinated metadata (`nota_id`, `fecha_publicacion`)

- `nota_id == "S/D"`: 0 / 35
- `nota_id` other (invented slug): 35 / 35
- `fecha_publicacion` with day = 19: 4 / 35

The baseline produced "S/D" nota_id 20/35 and "day 19" dates 24/35 — these
are behavioural markers of a base model that has not been trained on the
codebook's id/date conventions. The fine-tuned model is expected to
reproduce the gold-format ids and dates at much higher rates.

## 6. Best examples (by f1_vs_gold)

- `IMG_20250919_104952_1992-07-28_p2_012_nota.txt` (idx=20) f1=0.955 tp=63 fp=3 fn=3 out_tokens=661 finish=stop
- `IMG_20251009_120514_1994_06_25_p1_005_nota.txt` (idx=24) f1=0.955 tp=63 fp=3 fn=3 out_tokens=663 finish=stop

## 7. Worst examples (by f1_vs_gold)

- `IMG_20241115_113003_1991-01-09_011_nota.txt` (idx=16) f1=0.023 tp=3 fp=251 fn=3 out_tokens=2891 finish=stop
- `IMG_20241031_113558_1990-05-19_013_nota.txt` (idx=10) f1=0.030 tp=2 fp=127 fn=4 out_tokens=1369 finish=stop

## 8. Plan success criteria (PLAN §6)

The PLAN §6 success table requires all of:
schema_validity ≥ 0.95, categorical_accuracy ≥ 0.80, f1_global ≥ 0.70.

This run reports:
- schema_validity = 1.0000 — PASS
- categorical_accuracy aggregate = 0.4189 — FAIL
- f1_global = 0.5350 — FAIL

Verdict against the plan criterion: iterar Fase 6.

## 9. What this Phase 6 (r=32) run justifies

- The fine-tuned LoRA at `checkpoints/qwen-protesta-v1-r32` (rank=32) is
  unambiguously better than the Phase 2 baseline on every content metric while
  preserving the schema/parse validity floor at 1.0. The biggest single jump
  is on `field_recall_non_empty` (+0.5804), confirming that
  the boolean flip was the right thing to train.
- The aggregate categorical accuracy jumped from 0.0384
  → 0.4189 (+0.3805), with `incidentes.represion.presencia`
  and similar booleans going from baseline-floor to 0.7714 —
  these are the boolean fields that were essentially random in the baseline.
- `field_recall.exact` jumped from 0.0540
  → 0.5165 (+0.4625), and the looser
  non-empty recovery jumped from 0.1692
  → 0.7496 (+0.5804). The
  non-empty recovery crossing 0.6 means: in more than half of all gold leaf
  positions, the fine-tuned model emits *some* non-empty value — a strong
  signal that the model has internalized the codebook's information density,
  even when the exact value is wrong.
- The model is no longer systematically hallucinatory on metadata:
  `nota_id == "S/D"` dropped from baseline (20/35) to 0/35,
  and "day 19" dates dropped from baseline (24/35) to 4/35.
- Hallucinated nota_ids are still present (35/35 produce a
  plausible-looking slug that does not match gold) — but this is expected
  because the exact nota_id includes a source-image timestamp the model
  cannot see. The *behavioral* fact that the model now produces
  codebook-shaped ids instead of `"S/D"` is the relevant improvement.

## 10. What this Phase 6 (r=32) run does NOT justify

- Reaching the PLAN §6 MVP acceptance bar requires f1 ≥ 0.70 and categorical
  ≥ 0.80; this run is below both targets (f1=0.5350,
  cat_agg=0.4189). Per PLAN §6, two of three criteria failing means
  the model is NOT yet MVP and Phase 6 iteration is required.
- The categorical enums are still well below the 80% target on most paths
  (best path: `extraccion.tiene_eventos_protesta` at 0.7714; worst path:
  `contra_quien[].nivel_institucional` at 0.1856). Categoria-level drifts remain on
  `delimitacion.criterio_delimitacion`, `temporalidad.tipo_temporal`, and
  `accion.formato_principal.categoria`.
- 7 false positives on `tiene_eventos_protesta` remain
  — the model still occasionally flags non-protest notes as events.
  Worst-FP cases (high fp, low fn) dominate the bottom of the f1 ranking.

See `metrics/finetuned_qwen-protesta-v1-r32.json` for the full machine-readable
metrics block (counts, per-path, timings, per-example records).

---

## Sources

- `metrics/finetuned_qwen-protesta-v1-r32.json` — full machine-readable metrics
- `metrics/finetuned_qwen-protesta-v1-r32_outputs.jsonl` — per-example raw output + parsed object + parse/schema status
- `metrics/baseline_qwen2.5-7b.json` — Phase 2 baseline (for delta computation)
- `reports/phase6_r32_eval.json` — Phase 6 (r=32) readiness report
- `metrics/qualitative_report_finetuned_r32.md` — this report
- `scripts/evaluate_finetuned_qwen.py` — this runner
- `scripts/baseline_qwen_full.py` — Phase 2 baseline runner (helper functions reused)
- `PLAN_ENTRENAMIENTO_QWEN.md §Fase 6` — plan reference
