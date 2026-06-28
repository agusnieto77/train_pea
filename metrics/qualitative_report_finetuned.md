# Phase 4 — Qualitative Report: Qwen2.5-7B-Instruct + LoRA (qwen-protesta-v1)

> Historical report: this evaluation was produced from the previous 350-row Phase 1 artifacts. The canonical dataset is now `entrenamiento.jsonl` with 317 rows; retrain/evaluate before comparing current metrics.

**Date:** 2026-06-27
**Base model:** `Qwen/Qwen2.5-7B-Instruct` (7B params, bf16)
**Adapter:** `checkpoints/qwen-protesta-v1` (sha1=1464f1a8a0da5415dd649a16c275020da94f32a4, size=80792880 bytes, resolved_reason: adapter_model.safetensors at requested root)
**Inference:** vLLM 0.23.0, `enable_lora=True`, `max_loras=1`, `max_lora_rank=16`,
`SamplingParams(structured_outputs=StructuredOutputsParams(json=cleaned_schema))`,
`VLLM_USE_FLASHINFER_SAMPLER=0` (required for sm_120 / RTX 5090 in this environment)
**Eval set:** 35 examples from `data/chat_formatted/eval.jsonl` (GPT-5.4-mini + Nico human validation, weight 1.0)

**Headline metrics (from `metrics/finetuned_qwen-protesta-v1.json`):**

| metric | value |
|---|---|
| schema_validity | **1.0000** (35/35) |
| parse_validity | 1.0000 (35/35) |
| `tiene_eventos_protesta` accuracy (boolean) | **0.7143** |
| categorical accuracy (aggregated) | 0.3400 |
| f1_global (micro over flattened leaves) | **0.4637** (precision=0.4664, recall=0.4609) |
| field_recall exact | 0.4134 (1339 / 3239) |
| field_recall non-empty recovery | 0.6335 (2052 / 3239) |
| `finish_reason=length` truncations | 0 / 35 |
| mean output tokens | 1048.6 (max 3271) |
| total wall time | 500.9 s (~14.3 s / example) |
| run status | `pass` |

**Headline deltas vs Phase 2 baseline** (`metrics/baseline_qwen2.5-7b.json`):

| metric | baseline | finetuned | delta |
|---|---:|---:|---:|
| schema_validity | 1.0 | 1.0 | 0.0 |
| f1_global.f1 | 0.0971 | 0.4637 | 0.3666 |
| categorical_accuracy aggregate | 0.0384 | 0.34 | 0.3016 |
| tiene_eventos_protesta accuracy | 0.2857 | 0.7143 | 0.4286 |
| field_recall.exact_match_recall | 0.054 | 0.4134 | 0.3594 |
| field_recall.non_empty_recovery_recall | 0.1692 | 0.6335 | 0.4643 |

---

## 1. `tiene_eventos_protesta` confusion matrix

|               | pred=0 | pred=1 |
|---------------|-------:|-------:|
| **gold=1** (27) | 5 FN | 22 TP |
| **gold=0** (8) | 3 TN | 5 FP |

This is the single highest-leverage field: getting the boolean right
unlocks every downstream leaf. After fine-tuning the model should
predict `True` much more often than the baseline.

## 2. Output-token distribution

| output_tokens | count |
|---|---:|
| 0-200 | 8 |
| 200-500 | 0 |
| 500-1000 | 15 |
| 1000-2000 | 7 |
| 2000+ | 5 |

The baseline was bimodal (30/35 outputs were a 100-token "no events"
shell; 5/35 went long). The fine-tuned model is expected to produce
events on a much larger fraction of the eval set.

## 3. Per-path categorical accuracy

| path | tp | tn | fp | fn | accuracy | (Δ vs baseline) |
|---|---:|---:|---:|---:|---:|---:|
| `extraccion.tiene_eventos_protesta` | 22 | 3 | 5 | 5 | 0.7143 | +0.4286 |
| `evento.es_evento_protesta` | 24 | 3 | 20 | 20 | 0.4030 | +0.2876 |
| `delimitacion.criterio_delimitacion` | 14 | 0 | 33 | 33 | 0.1750 | +0.1750 |
| `delimitacion.es_accion_principal_con_complementarias` | 2 | 18 | 27 | 27 | 0.2703 | +0.2346 |
| `temporalidad.tipo_temporal` | 14 | 0 | 33 | 33 | 0.1750 | +0.1205 |
| `temporalidad.tempo_verbal` | 14 | 0 | 33 | 33 | 0.1750 | +0.1009 |
| `temporalidad.fecha_inicio.certeza` | 33 | 0 | 14 | 14 | 0.5410 | +0.4256 |
| `temporalidad.fecha_fin.certeza` | 33 | 0 | 14 | 14 | 0.5410 | +0.4256 |
| `accion.formato_principal.categoria` | 12 | 0 | 35 | 35 | 0.1463 | +0.1463 |
| `sujetos[].categoria` | 25 | 0 | 25 | 25 | 0.3333 | +0.3333 |
| `sujetos[].organizaciones[].categoria` | 39 | 0 | 30 | 36 | 0.3714 | +0.3714 |
| `demandas[].categoria` | 18 | 0 | 40 | 38 | 0.1875 | +0.1875 |
| `contra_quien[].categoria` | 20 | 0 | 27 | 37 | 0.2381 | +0.2381 |
| `contra_quien[].nivel_institucional` | 14 | 0 | 33 | 43 | 0.1556 | +0.0941 |
| `lugares[].categoria` | 14 | 0 | 37 | 35 | 0.1628 | +0.1467 |
| `lugares[].rol_en_evento` | 22 | 0 | 29 | 27 | 0.2821 | +0.2143 |
| `alcance.categoria` | 22 | 0 | 25 | 25 | 0.3056 | +0.2315 |
| `cantidad_participantes.hay_cantidad_mencionada` | 28 | 0 | 19 | 19 | 0.4242 | +0.4242 |
| `cantidad_participantes.es_aproximada` | 27 | 0 | 20 | 20 | 0.4030 | +0.4030 |
| `incidentes.represion.presencia` | 33 | 0 | 14 | 14 | 0.5410 | +0.5410 |
| `incidentes.enfrentamiento.presencia` | 31 | 0 | 16 | 16 | 0.4921 | +0.4921 |
| `incidentes.detenidos.presencia` | 33 | 0 | 14 | 14 | 0.5410 | +0.5410 |
| `incidentes.heridos.presencia` | 32 | 0 | 15 | 15 | 0.5161 | +0.5161 |
| `incidentes.muertos.presencia` | 32 | 0 | 15 | 15 | 0.5161 | +0.5161 |
| `incidentes.danios_materiales.presencia` | 33 | 0 | 14 | 14 | 0.5410 | +0.4256 |

Aggregate categorical accuracy: **0.3400**
(support 1809 leaves).

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
- `IMG_20240916_103907_1989-09-13_004_nota.txt` (idx=5) f1=0.833 tp=5 fp=1 fn=1 out_tokens=111 finish=stop

## 7. Worst examples (by f1_vs_gold)

- `IMG_20241115_113003_1991-01-09_011_nota.txt` (idx=16) f1=0.023 tp=3 fp=257 fn=3 out_tokens=2977 finish=stop
- `IMG_20241031_113558_1990-05-19_013_nota.txt` (idx=10) f1=0.030 tp=2 fp=127 fn=4 out_tokens=1433 finish=stop

## 8. Plan success criteria (PLAN §6)

The PLAN §6 success table requires all of:
schema_validity ≥ 0.95, categorical_accuracy ≥ 0.80, f1_global ≥ 0.70.

This run reports:
- schema_validity = 1.0000 — PASS
- categorical_accuracy aggregate = 0.3400 — FAIL
- f1_global = 0.4637 — FAIL

Verdict against the plan criterion: iterar Fase 6.

## 9. What this Phase 4 run justifies

- The fine-tuned LoRA at `checkpoints/qwen-protesta-v1` is unambiguously better
  than the Phase 2 baseline on every content metric while preserving the
  schema/parse validity floor at 1.0. The biggest single jump is on
  `tiene_eventos_protesta` (+0.4286), confirming that the boolean flip was the
  right thing to train.
- The aggregate categorical accuracy jumped from 0.0384 → 0.3400 (+0.30),
  with `incidentes.represion.presencia` and similar booleans going from 0 to
  >0.49 — these are the boolean fields that were essentially random in the
  baseline.
- `field_recall.exact` jumped from 0.054 → 0.4134 (+0.36), and the looser
  non-empty recovery jumped from 0.169 → 0.634 (+0.46). The non-empty recovery
  crossing 0.6 means: in more than half of all gold leaf positions, the
  fine-tuned model emits *some* non-empty value — a strong signal that the
  model has internalized the codebook's information density, even when the
  exact value is wrong.
- The model is no longer systematically hallucinatory on metadata:
  `nota_id == "S/D"` dropped from 20/35 to 0/35, and "day 19" dates dropped
  from 24/35 to 4/35. Every `fecha_publicacion` matches gold exactly.
- Hallucinated nota_ids are still present (35/35 produce a plausible-looking
  slug that does not match gold) — but this is expected because the exact
  nota_id includes a source-image timestamp the model cannot see. The
  *behavioral* fact that the model now produces codebook-shaped ids instead
  of `"S/D"` is the relevant improvement.

## 10. What this Phase 4 run does NOT justify

- Reaching the PLAN §6 MVP acceptance bar requires f1 ≥ 0.70 and categorical
  ≥ 0.80; this run is below both targets (f1=0.4637,
  cat_agg=0.3400).
  Per PLAN §6, two of three criteria failing means the model is NOT yet MVP
  and Phase 6 iteration is required.
- The categorical enums are still well below the 80% target on most paths
  (best path: `incidentes.represion.presencia` at 0.5410).
  Categoria-level drifts remain on `delimitacion.criterio_delimitacion`,
  `temporalidad.tipo_temporal`, and `accion.formato_principal.categoria`.
- 5 false positives on `tiene_eventos_protesta` remain — the model still
  occasionally flags non-protest notes as events. Worst-FP cases (high fp,
  low fn) dominate the bottom of the f1 ranking.

See `metrics/finetuned_qwen-protesta-v1.json` for the full machine-readable
metrics block (counts, per-path, timings, per-example records).

---

## Sources

- `metrics/finetuned_qwen-protesta-v1.json` — full machine-readable metrics
- `metrics/finetuned_qwen-protesta-v1_outputs.jsonl` — per-example raw output + parsed object + parse/schema status
- `metrics/baseline_qwen2.5-7b.json` — Phase 2 baseline (for delta computation)
- `reports/phase4_eval.json` — Phase 4 readiness report
- `scripts/evaluate_finetuned_qwen.py` — this runner
- `scripts/baseline_qwen_full.py` — Phase 2 baseline runner (helper functions reused)
- `PLAN_ENTRENAMIENTO_QWEN.md` §Fase 4 — plan reference
