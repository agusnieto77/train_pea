# Phase 2 — Qualitative Report: Qwen2.5-7B-Instruct Baseline

**Date:** 2026-06-27
**Model:** `Qwen/Qwen2.5-7B-Instruct` (no fine-tuning), 7B parameters, bf16
**Inference:** vLLM 0.23.0, `SamplingParams(structured_outputs=StructuredOutputsParams(json=cleaned_schema))`, `VLLM_USE_FLASHINFER_SAMPLER=0` (required for sm_120 / RTX 5090 in this environment)
**Eval set:** 35 examples from `data/chat_formatted/eval.jsonl` (GPT-5.4-mini + Nico human validation, weight 1.0)
**Headline metrics (from `metrics/baseline_qwen2.5-7b.json`):**

| metric | value |
|---|---|
| schema_validity | **1.00** (35/35 parse + schema-valid) |
| parse_validity | 1.00 (35/35) |
| `tiene_eventos_protesta` accuracy (boolean) | **0.2857** (10/35) |
| categorical accuracy (aggregated, 1460 leaves) | 0.0384 |
| f1_global (micro over flattened leaves) | **0.0971** |
| field_recall exact | 0.0540 (175 / 3239 gold leaves) |
| field_recall non-empty recovery | 0.1692 (548 / 3239) |
| `finish_reason=length` truncations | 0 / 35 |
| mean output tokens | 465.9 (median 115, max 3061) |
| total wall time | 176 s (~5 s / example) |

> **Headline read:** Schema conformity is perfect because vLLM's structured-output
> constraint enforces it. The model is a competent JSON shape complier and
> otherwise a poor event extractor on this domain. It massively under-extracts
> (predicts "no events" on 30/35 notes), fabricates low-information fields, and
> rarely picks the right categorical enum. This is exactly the gap Phase 3
> (SFT/QLoRA) is meant to close.

---

## 1. Massive false-negative bias on `tiene_eventos_protesta`

Confusion matrix for the boolean `extraccion.tiene_eventos_protesta` over 35 examples:

|               | pred=0 | pred=1 |
|---------------|-------:|-------:|
| **gold=1** (27) |  **23 FN**  |   4 TP |
| **gold=0** (8)  |   6 TN  |   2 FP |

- 23 / 27 notes that *do* describe a protest event are classified as "no events" by the model (recall ≈ 15%).
- Only 4 / 27 real events are picked up at all; even when picked up, the event count and event-level fields are wrong (see §3).
- 2 / 8 non-event notes are flagged as events, suggesting the model is not just being conservative — it has a real distribution mismatch with the codebook's notion of "protest event."

This single field drives the entire f1_global collapse, because the F1 micro-aggregates over every gold leaf: missing one event produces ~50-180 FN leaves depending on event richness.

## 2. The model defaults to a 100-token "no events" shell

30 / 35 outputs are between 104 and 121 output tokens — essentially `schema_version + nota + empty extraccion {tiene:false, total:0, eventos:[]}`. The remaining 5 / 35 outputs go long (1182 – 3061 output tokens) and are the only ones that attempt to populate `eventos_protesta`.

The distribution is bimodal:

| output_tokens | count |
|---|---|
| 100-200 (no events shell) | 30 |
| 1000-3500 (extracted events) | 5 |

The model is not generating *bad* structured events — it is generating *no* structured events most of the time. The structured-output JSON grammar is being followed; the model simply does not engage with the domain-specific extraction.

## 3. When the model does extract, the categorical fields are mostly wrong

The 4 notes where the model produced at least one event (and gold also has events) all have mismatched category predictions:

| nota_id | gold cat (1st event) | pred cat (1st event) | gold alcance | pred alcance | gold criterio | pred criterio |
|---|---|---|---|---|---|---|
| `…1989-11-28_009` | Manifestaciones | Cortes | Local | Local | Accion principal con acciones complementarias | Espacial |
| `…1990-03-17_012` | Huelgas | Acciones judiciales | Provincial | Local | Evento unico en la nota | Temporal y espacial |
| `…1991-03-08_004` | Huelgas | Cortes | Nacional | Nacional | Temporal | Temporal y espacial |
| `…1994_12_27_p2_010` | Manifestaciones | Acciones judiciales | Local | Local | Evento unico en la nota | Espacial |

Categorical accuracy per path is below 12% for every measured enum field (best path: `temporalidad.fecha_inicio.certeza` and four other paths tied at 11.5%). The model knows *what kind of field* a category should be — it produces a valid enum value — but the value is essentially uncorrelated with the gold label. This is consistent with a base model that has not learned the MVS codebook.

## 4. Fabricated `fecha_publicacion` and `nota_id`

The model hallucinates the note metadata in a systematic way:

- **`fecha_publicacion`**: of 35 outputs, **24 / 35** produce a date with **day = 19** (`19/03/1989`, `19/06/1989`, `19/01/1991`, etc.). The actual gold dates are spread across days 1-31. This is a strong, uniform bias — the model has internalized a "19th of the month" default for Argentine press dates and rarely overrides it.
- **`nota_id`**: 20 / 35 outputs are the literal string `"S/D"`; the other 15 are invented slugs (e.g. `19890305_1`, `19900926_Dafnos`). **None** of the 35 outputs reproduce the real gold `nota_id` of the form `IMG_YYYYMMDD_HHMMSS_YYYY-MM-DD_NNN_nota.txt` that the system prompt explicitly tells the model to use.

These metadata hallucinations are a small contribution to the leaf-level F1, but they are a high-signal **behavioural** finding: a fine-tuned model that follows the codebook's "use the actual id injected by the script" instruction is concretely distinguishable from this baseline.

## 5. Hallucinated event contents (2 FP cases)

The two false positives both describe a single extracted event with `categoria="Acciones judiciales"` and `alcance="Local"`. Their descriptions:

- `…1990-05-19_013`: *"Conflicto de empleados del Ministerio de Trabajo reclamando equiparación salarial"*
- `…1991-01-09_011`: *"Concentración de empleados públicos frente a la Casa de Gobierno provincial"*

The descriptions are plausible protest-shaped text, but the gold labels for these notes are "no events." The model is not extracting from a true event mention — it is over-interpreting routine labor/political news as protest. This is the classic base-model failure mode: a tendency to *see* a protest in any unionized/labor context.

## 6. Output token distribution vs. prompt budget

The 35 prompts range from 2440 to 18206 tokens (consistent with the Phase 1 token audit's `eval.max=18910`). The smallest per-example `max_tokens` budget is `20480 − 18206 − 16 = 2258`, still 8× the 256 minimum output budget — so **no example was blocked for prompt-budget reasons** and **no example was truncated** (all 35 `finish_reason="stop"`). The Phase 1 gate's "all eval examples fit" claim is preserved under Phase 2.

## 7. No example hit the 8192-output cap

The 5 long outputs (1182 – 3061 tokens) are well under the 8192 cap; the cap is not the limiting factor. For the worst-case 18206-token prompt the output budget is 2258 tokens, which is what really constrained the model. If Phase 3 produces richer outputs, this 2258-token cap on the longest prompt could become a real risk; for the 7B baseline it is not yet binding.

## 8. Structured-output constraint is doing exactly its job

`schema_validity = 1.0` for all 35 examples. Every output parses as JSON and validates against `Draft202012Validator` on the raw MVS schema (with `const` and pattern constraints, not just the cleaned enum form). This is the "did the model at least obey the shape" gate, and it is at 100%. Every other metric in this report is a measurement of *content* quality, not shape quality, and content quality is what Phase 3 SFT is intended to lift.

## 9. The 4 "TP" cases are barely TPs

Even in the 4 cases where the model correctly answered `tiene_eventos_protesta=true`:
- The model produces 1-2 events where gold has 2-6 events (count under-prediction).
- The first event's `categoria`, `criterio_delimitacion`, and `alcance.categoria` differ from gold in all 4 cases.
- Field-level exact-match recovery on these examples is below 20% of gold leaves.

So the four "TP" cases are not evidence of a working extractor — they are evidence that the model *sometimes* decides to engage with the schema. The actual categorical extraction skill is essentially absent.

## 10. What this baseline justifies (and does not)

**Justifies:**
- The Phase 2 → Phase 3 gate: a 7B base model that follows the MVS shape but does not understand the MVS content is exactly the right starting point for SFT.
- Treating `tiene_eventos_protesta` recall as the highest-leverage training signal (most FNs come from this single boolean, which is the smallest possible supervision target).
- Treating the categorical enum fields as the next-highest leverage target (after the boolean flip, every other miss is a wrong enum value).

**Does not justify:**
- Any claim that the baseline is "useful for real extraction." It is not. It is a *measurement artifact* for Phase 3.
- Any claim that the structured-output constraint is broken — it is not. It is the only thing the model is doing correctly.

---

## Sources

- `metrics/baseline_qwen2.5-7b.json` — full machine-readable metrics
- `metrics/baseline_qwen2.5-7b_outputs.jsonl` — per-example raw output + parsed object + parse/schema status
- `reports/phase2_readiness.json` — readiness report updated with full-baseline status
- `scripts/baseline_qwen_full.py` — the runner that produced these results
- `PLAN_ENTRENAMIENTO_QWEN.md` §Fase 2 — plan reference
