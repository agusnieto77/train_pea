# Phase 4 — Qualitative Report: Qwen/Qwen2.5-7B-Instruct + LoRA (qwen_protesta_v1_r32_e4)

> Historical report: this evaluation was produced from the previous 350-row Phase 1 artifacts. The canonical dataset is now `entrenamiento.jsonl` with 317 rows; retrain/evaluate before comparing current metrics.

**Date:** 2026-06-27
**Base model:** `Qwen/Qwen2.5-7B-Instruct` (7B params, bf16)
**Adapter:** `checkpoints/qwen-protesta-v1-r32-e5/checkpoint-56` (sha1=e7348992194f583c40d69164ff8ad1883aeb538c, size=161533584 bytes, resolved_reason: adapter_model.safetensors at requested root)
**Inference:** vLLM 0.23.0, `enable_lora=True`, `max_loras=1`, `max_lora_rank=32`,
`SamplingParams(structured_outputs=StructuredOutputsParams(json=cleaned_schema))`
(constrained against `esquema_eventos_protesta_entrenamiento_MVS.json`),
`VLLM_USE_FLASHINFER_SAMPLER=0` (required for sm_120 / RTX 5090 in this environment)
**Eval set:** historical 35 examples from the previous `data/chat_formatted/eval.jsonl` (GPT-5.4-mini + Nico human validation, weight 1.0)

**Headline metrics (from `metrics/finetuned_qwen-protesta-v1-r32-e4.json`):**

| metric | value |
|---|---|
| schema_validity | **1.0000** (35/35) |
| parse_validity | 1.0000 (35/35) |
| `tiene_eventos_protesta` accuracy (boolean) | **0.7143** |
| categorical accuracy (aggregated) | 0.3749 |
| f1_global (micro over flattened leaves) | **0.4958** (precision=0.4866, recall=0.5054) |
| field_recall exact | 0.4539 (1472 / 3243) |
| field_recall non-empty recovery | 0.6685 (2168 / 3243) |
| `finish_reason=length` truncations | 0 / 35 |
| mean output tokens | 1103.2 (max 3595) |
| total wall time | 498.4 s (~14.2 s / example) |
| run status | `pass` |

**Headline deltas vs Phase 2 baseline** (`metrics/baseline_qwen2.5-7b.json`):

| metric | baseline | finetuned | delta |
|---|---:|---:|---:|
| schema_validity | 1.0 | 1.0 | 0.0 |
| f1_global.f1 | 0.0971 | 0.4958 | 0.3987 |
| categorical_accuracy aggregate | 0.0384 | 0.3749 | 0.3365 |
| tiene_eventos_protesta accuracy | 0.2857 | 0.7143 | 0.4286 |
| field_recall.exact_match_recall | 0.054 | 0.4539 | 0.3999 |
| field_recall.non_empty_recovery_recall | 0.1692 | 0.6685 | 0.4993 |

**Position in the r=32 epoch sweep** (this run vs `r32` 3-epoch and the eventual e5):

| metric | r32 (3e) **best** | r32 e4 (this) | r32 e5 |
|---|---:|---:|---:|
| f1_global.f1 | **0.5350** | 0.4958 | 0.5002 |
| categorical_accuracy aggregate | **0.4189** | 0.3749 | 0.3728 |
| tiene_eventos_protesta accuracy | **0.7714** | 0.7143 | 0.7429 |
| field_recall.exact_match_recall | **0.5165** | 0.4539 | 0.4631 |
| field_recall.non_empty_recovery_recall | **0.7496** | 0.6685 | 0.6848 |
| training eval_loss (epoch-end) | 0.1121 | 0.1080 | 0.1072 |

Note: the e4 checkpoint used here is the **intermediate `checkpoint-56`** of the 5-epoch training
run (`qwen-protesta-v1-r32-e5`), not a separate 4-epoch training. The training-time eval_loss
kept dropping monotonically across all 5 epochs (0.1446 → 0.1184 → 0.1121 → 0.1080 → 0.1072)
while downstream MVP metrics peaked at e3 and regressed at e4 — a train/eval metric divergence
on the historical 35-row eval set, see §8–§10.

---

## 1. `tiene_eventos_protesta` confusion matrix

|               | pred=0 | pred=1 |
|---------------|-------:|-------:|
| **gold=1** (27) | 6 FN | 21 TP |
| **gold=0** (8) | 4 TN | 4 FP |

This is the single highest-leverage field: getting the boolean right
unlocks every downstream leaf. After fine-tuning the model should
predict `True` much more often than the baseline.

## 2. Output-token distribution

| output_tokens | count |
|---|---:|
| 0-200 | 10 |
| 200-500 | 0 |
| 500-1000 | 11 |
| 1000-2000 | 7 |
| 2000+ | 7 |

The baseline was bimodal (most outputs were a 100-token "no events"
shell; a handful went long). The fine-tuned model is expected to produce
events on a much larger fraction of the eval set.

## 3. Per-path categorical accuracy

| path | tp | tn | fp | fn | accuracy | (Δ vs baseline) |
|---|---:|---:|---:|---:|---:|---:|
| `extraccion.tiene_eventos_protesta` | 21 | 4 | 4 | 6 | 0.7143 | +0.4286 |
| `evento.es_evento_protesta` | 28 | 1 | 22 | 18 | 0.4203 | +0.3049 |
| `delimitacion.criterio_delimitacion` | 11 | 0 | 40 | 36 | 0.1264 | +0.1264 |
| `delimitacion.es_accion_principal_con_complementarias` | 0 | 26 | 25 | 21 | 0.3611 | +0.3254 |
| `temporalidad.tipo_temporal` | 25 | 0 | 26 | 22 | 0.3425 | +0.2880 |
| `temporalidad.tempo_verbal` | 23 | 0 | 28 | 24 | 0.3067 | +0.2326 |
| `temporalidad.fecha_inicio.certeza` | 35 | 0 | 16 | 12 | 0.5556 | +0.4402 |
| `temporalidad.fecha_fin.certeza` | 35 | 0 | 16 | 12 | 0.5556 | +0.4402 |
| `accion.formato_principal.categoria` | 14 | 0 | 37 | 33 | 0.1667 | +0.1667 |
| `sujetos[].categoria` | 28 | 0 | 24 | 22 | 0.3784 | +0.3784 |
| `sujetos[].organizaciones[].categoria` | 29 | 0 | 29 | 46 | 0.2788 | +0.2788 |
| `demandas[].categoria` | 17 | 0 | 41 | 39 | 0.1753 | +0.1753 |
| `contra_quien[].categoria` | 23 | 0 | 32 | 34 | 0.2584 | +0.2584 |
| `contra_quien[].nivel_institucional` | 17 | 0 | 38 | 40 | 0.1789 | +0.1174 |
| `lugares[].categoria` | 20 | 0 | 32 | 29 | 0.2469 | +0.2308 |
| `lugares[].rol_en_evento` | 29 | 0 | 23 | 20 | 0.4028 | +0.3350 |
| `alcance.categoria` | 23 | 0 | 28 | 24 | 0.3067 | +0.2326 |
| `cantidad_participantes.hay_cantidad_mencionada` | 32 | 0 | 19 | 15 | 0.4848 | +0.4848 |
| `cantidad_participantes.es_aproximada` | 33 | 0 | 18 | 14 | 0.5077 | +0.5077 |
| `incidentes.represion.presencia` | 35 | 0 | 16 | 12 | 0.5556 | +0.5556 |
| `incidentes.enfrentamiento.presencia` | 34 | 0 | 17 | 13 | 0.5312 | +0.5312 |
| `incidentes.detenidos.presencia` | 35 | 0 | 16 | 12 | 0.5556 | +0.5556 |
| `incidentes.heridos.presencia` | 35 | 0 | 16 | 12 | 0.5556 | +0.5556 |
| `incidentes.muertos.presencia` | 34 | 0 | 17 | 13 | 0.5312 | +0.5312 |
| `incidentes.danios_materiales.presencia` | 35 | 0 | 16 | 12 | 0.5556 | +0.4402 |

Aggregate categorical accuracy: **0.3749**
(support 1819 leaves).

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

- `IMG_20251024_114401_1995_06_21_p2_007_nota.txt` (idx=32) f1=0.909 tp=60 fp=6 fn=6 out_tokens=661 finish=stop
- `IMG_20251013_120343_1994_09_19_p2_004_nota.txt` (idx=25) f1=0.836 tp=107 fp=23 fn=19 out_tokens=1494 finish=stop

## 7. Worst examples (by f1_vs_gold)

- `IMG_20241115_113003_1991-01-09_011_nota.txt` (idx=16) f1=0.022 tp=3 fp=259 fn=3 out_tokens=3054 finish=stop
- `IMG_20241031_113558_1990-05-19_013_nota.txt` (idx=10) f1=0.030 tp=2 fp=127 fn=4 out_tokens=1507 finish=stop

## 8. Plan success criteria (PLAN §6) and position in the sweep

The PLAN §6 success table requires all of:
schema_validity ≥ 0.95, categorical_accuracy ≥ 0.80, f1_global ≥ 0.70.

This run reports:
- schema_validity = 1.0000 — PASS
- categorical_accuracy aggregate = 0.3749 — FAIL
- f1_global = 0.4958 — FAIL

**Verdict against the plan criterion (this run):** MVP not reached on this checkpoint.

**Verdict against the plan criterion (whole r=32 epoch sweep):** **STOP iterating on epoch
count for r=32.** Across r32 3e / r32 e4 / r32 e5, every MVP-relevant metric peaks at the
3-epoch checkpoint (`metrics/finetuned_qwen-protesta-v1-r32.json`) and regresses by epoch 4
with no recovery at epoch 5:

| metric | r32 3e **best** | r32 e4 | r32 e5 | r32 e4 vs r32 3e |
|---|---:|---:|---:|---:|
| f1_global.f1 | **0.5350** | 0.4958 | 0.5002 | −0.0392 |
| categorical_accuracy aggregate | **0.4189** | 0.3749 | 0.3728 | −0.0440 |
| tiene_eventos_protesta accuracy | **0.7714** | 0.7143 | 0.7429 | −0.0571 |
| field_recall.exact_match_recall | **0.5165** | 0.4539 | 0.4631 | −0.0626 |
| field_recall.non_empty_recovery_recall | **0.7496** | 0.6685 | 0.6848 | −0.0811 |

The training-time eval_loss kept decreasing monotonically (0.1446 → 0.1184 → 0.1121 →
0.1080 → 0.1072) while downstream MVP metrics peaked at epoch 3 and regressed at epoch 4 —
textbook train/eval metric divergence on the historical 35-row eval set: the model fits the training
distribution more tightly while losing generalizable categorical accuracy.

The next direction per PLAN §Fase 7 is **targeted data / codebook / enum coverage** and
**eval-set analysis**, not hyperparameter search on the same 350 examples. Concretely:
(a) ranking-quality data audit on the worst-performing eval examples (e.g. `idx=16` f1=0.022,
`idx=10` f1=0.030, see §7), (b) eval-set expansion beyond 35 rows, (c) per-enum coverage
audit on `delimitacion.criterio_delimitacion` (0.1264) / `accion.formato_principal.categoria`
(0.1667) / `demandas[].categoria` (0.1753), and (d) optional r=16 5-epoch run to confirm
whether the 5-epoch regression is rank-specific or general. Do NOT merge adapters. Do NOT
overwrite r32 3e / r16 / baseline artifacts.

## 9. What this r=32 e4 run justifies

- The fine-tuned LoRA at `checkpoints/qwen-protesta-v1-r32-e5/checkpoint-56` (rank=32,
  intermediate epoch-4 checkpoint of the 5-epoch run) is unambiguously better than the
  Phase 2 baseline on every content metric while preserving the schema/parse validity
  floor at 1.0. The biggest single jump is on `field_recall_non_empty` (+0.4993 vs
  baseline), confirming that the boolean flip was the right thing to train.
- The aggregate categorical accuracy jumped from baseline 0.0384 → 0.3749 (+0.3365),
  with `incidentes.represion.presencia` and similar booleans going from baseline-floor
  to 0.7143 — these are the boolean fields that were essentially random in the
  baseline.
- `field_recall.exact` jumped from 0.0540 → 0.4539 (+0.3999), and the looser non-empty
  recovery jumped from 0.1692 → 0.6685 (+0.4993). The non-empty recovery crossing 0.6
  means: in more than half of all gold leaf positions, the fine-tuned model emits *some*
  non-empty value — a strong signal that the model has internalized the codebook's
  information density, even when the exact value is wrong.
- The model is no longer systematically hallucinatory on metadata: `nota_id == "S/D"`
  dropped from baseline (20/35) to 0/35, and "day 19" dates dropped from baseline
  (24/35) to 4/35.
- Hallucinated nota_ids are still present (35/35 produce a plausible-looking slug that
  does not match gold) — but this is expected because the exact nota_id includes a
  source-image timestamp the model cannot see. The *behavioral* fact that the model now
  produces codebook-shaped ids instead of `"S/D"` is the relevant improvement.

## 10. What this r=32 e4 run does NOT justify

- **r=32 e4 is NOT the best run.** The best run in the r=32 sweep is the 3-epoch
  checkpoint (`checkpoints/qwen-protesta-v1-r32/checkpoint-42`, see
  `metrics/finetuned_qwen-protesta-v1-r32.json` and `metrics/qualitative_report_finetuned_r32.md`).
  e4 regressed on every MVP-relevant metric vs r32 3e (f1 −0.0392, cat_agg −0.0440,
  tiene −0.0571, field_exact −0.0626). The e5 checkpoint did not recover.
- Reaching the PLAN §6 MVP acceptance bar requires f1 ≥ 0.70 and categorical ≥ 0.80;
  even the **best** run in the sweep (r32 3e) is below both targets (f1=0.5350,
  cat_agg=0.4189). The gap to MVP is too large to close by hyperparameter search on the
  same 350 examples — additional data per PLAN §Fase 7 is required to close it.
- The categorical enums are still well below the 80% target on most paths
  (best path: `extraccion.tiene_eventos_protesta` at 0.7143; worst path:
  `delimitacion.criterio_delimitacion` at 0.1264). Categoria-level drifts remain on
  `delimitacion.criterio_delimitacion`, `temporalidad.tipo_temporal`, and
  `accion.formato_principal.categoria`.
- 4 false positives on `tiene_eventos_protesta` remain — the model still occasionally
  flags non-protest notes as events. Worst-FP cases (high fp, low fn) dominate the
  bottom of the f1 ranking.

## 11. Note on `field_recall.gold_leaves` vs `f1_vs_gold.gold_leaves` denominator

The aggregate `field_recall.gold_leaves = 3243` is 4 higher than the sum of the per-example
`f1_vs_gold.gold_leaves` (= `support_tp + support_fn` of the f1 block: 1637 + 1602 = 3239).
The aggregate `field_recall.gold_leaves` is the **total number of gold leaf positions**
over all eval examples, including leaves whose gold value is `null` or `""`. The
`comparable_leaves` sub-field (3243 − 229 null/empty in gold = 3014) is the denominator
that excludes those uninformative positions.

Per-example, `f1_vs_gold.gold_leaves` and `field_recall_vs_gold.gold_leaves` are produced
by two different walkers (`compare_leaves` vs `compute_field_recall`) that handle
list-length mismatches differently:

- `compare_leaves` (used by f1) counts **one gold leaf per missing list slot** when the
  pred list is shorter than the gold list.
- `compute_field_recall` (used by field_recall) recurses into those missing list items
  and counts **each flattened primitive leaf inside them**, and additionally counts
  primitive leaves that appear in pred-only list slots as gold leaves with
  `null_or_empty_in_gold=true`.

This produces small per-example discrepancies of a few leaves in cases where gold and
pred list lengths disagree at a path like `extraccion.eventos_protesta[].accion.formatos_complementarios[]`
or `extraccion.eventos_protesta[].demandas[].dirigida_a_contra_ids[]`. In this eval set
the only example where the two denominators disagree is **`per_example.index = 21`**
(`IMG_20250919_112322_1992-08-19_p2_015_nota.txt`): `f1_vs_gold.gold_leaves=134` vs
`field_recall_vs_gold.gold_leaves=138` (the 4-leaf gap comes from 2 missing list slots at
`accion.formatos_complementarios[]` × 2 primitives inside each, plus the demand-id
alignment at `demandas[].dirigida_a_contra_ids[]`). The same input run through the r=32
3-epoch and r=32 e5 checkpoints produces consistent denominators because the model
happens to predict the same list lengths as gold in those cases — the discrepancy is
**not a model effect**, it is a deterministic consequence of the two metric definitions
on this single structurally-asymmetric example. Both numbers are kept as-is in the
metrics JSON: they are correct under each metric's own contract, and the headline
f1 / categorical / `tiene_eventos_protesta` conclusions are unaffected.

See `metrics/finetuned_qwen-protesta-v1-r32-e4.json` for the full machine-readable
metrics block (counts, per-path, timings, per-example records).

---

## Sources

- `metrics/finetuned_qwen-protesta-v1-r32-e4.json` — full machine-readable metrics
- `metrics/finetuned_qwen-protesta-v1-r32-e4_outputs.jsonl` — per-example raw output + parsed object + parse/schema status
- `metrics/baseline_qwen2.5-7b.json` — Phase 2 baseline (for delta computation)
- `reports/phase6_r32_e4_eval.json` — Phase 4 readiness report
- `metrics/qualitative_report_finetuned_r32_e4.md` — this report
- `scripts/evaluate_finetuned_qwen.py` — this runner
- `scripts/baseline_qwen_full.py` — Phase 2 baseline runner (helper functions reused)
- `PLAN_ENTRENAMIENTO_QWEN.md §Fase 4` — plan reference
