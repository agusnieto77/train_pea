"""Historical-only error-analysis report generator for the pre-317 r32_3e run.

Target model: LoRA r=32 / alpha=64 / 3 epochs at ``checkpoints/qwen-protesta-v1-r32``
(adapter sha1=6804aeb4d7f85b7d1b94574b1cab816017debbf7).

Historical scope: this analyzes old 350-row / 315-train / 35-eval artifacts.
The current canonical dataset is the 317-row migration; these metrics are
superseded and not comparable to current 317 artifacts until retraining/eval.

This script is read-only against training / inference / checkpoint artifacts.
It reuses the categorical-path whitelist and helpers from
``scripts/baseline_qwen_full.py`` (which are pure functions with no side
effects) and produces:

- ``metrics/error_analysis_r32_3e.md`` (human-readable report, Spanish narrative)
- ``metrics/error_analysis_r32_3e.json`` (machine-readable summary)
- ``metrics/error_analysis_r32_3e_field_errors.csv`` (per-path error tallies)
- ``metrics/error_analysis_r32_3e_worst_examples.csv`` (worst examples with gold/pred totals)

Run from the repo root::

    python scripts/analyze_r32_errors.py

No training, no inference, no vLLM, no checkpoint mutation.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Repo-root resolution so the script can be invoked from anywhere.
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from baseline_qwen_full import (  # noqa: E402  (sys.path tweak above is intentional)
    EVENT_CATEGORICAL_PATHS,
    _resolve_path,
)

METRICS_DIR = REPO_ROOT / "metrics"
EVAL_JSONL = REPO_ROOT / "data" / "chat_formatted" / "eval.jsonl"

BEST_MODEL_KEY = "r32_3e"
BEST_METRICS_PATH = METRICS_DIR / "finetuned_qwen-protesta-v1-r32.json"
BEST_OUTPUTS_PATH = METRICS_DIR / "finetuned_qwen-protesta-v1-r32_outputs.jsonl"
HISTORICAL_METADATA: dict[str, Any] = {
    "historical_artifact": True,
    "superseded_by": "canonical 317-row migration",
    "not_comparable_to_current_317_artifacts": True,
    "historical_dataset_rows": 350,
    "historical_train_rows": 315,
    "historical_eval_rows": 35,
}

# Comparison artifacts (read-only). Keys used in the headline-comparison table.
COMPARISON_MODELS: list[tuple[str, str]] = [
    ("baseline", METRICS_DIR / "baseline_qwen2.5-7b.json"),
    ("r16_3e", METRICS_DIR / "finetuned_qwen-protesta-v1.json"),
    ("r32_3e", BEST_METRICS_PATH),
    ("r32_e5", METRICS_DIR / "finetuned_qwen-protesta-v1-r32-e5.json"),
]


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _load_gold(eval_path: Path) -> list[dict[str, Any]]:
    """Gold assistant JSON for each eval row, aligned to row order."""
    gold: list[dict[str, Any]] = []
    with eval_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            msgs = json.loads(line)["messages"]
            gold.append(json.loads(msgs[2]["content"]))
    return gold


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _per_example_f1(record: dict[str, Any]) -> float:
    counts = record["f1_vs_gold"]
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    if (tp + fp) == 0 or (tp + fn) == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if (precision + recall) == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


# ---------------------------------------------------------------------------
# Error taxonomy helpers
# ---------------------------------------------------------------------------


def _tiene_confusion(
    per_example: list[dict[str, Any]], gold: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    """Return FP / FN / TN / TP note lists for extraccion.tiene_eventos_protesta."""
    buckets: dict[str, list[dict[str, Any]]] = {"fp": [], "fn": [], "tn": [], "tp": []}
    for pe, g in zip(per_example, gold):
        g_tiene = bool(g["extraccion"].get("tiene_eventos_protesta"))
        p_tiene = bool(pe["parsed"]["extraccion"].get("tiene_eventos_protesta"))
        if g_tiene and p_tiene:
            buckets["tp"].append(_error_row(pe, g))
        elif not g_tiene and not p_tiene:
            buckets["tn"].append(_error_row(pe, g))
        elif g_tiene and not p_tiene:
            buckets["fn"].append(_error_row(pe, g))
        else:
            buckets["fp"].append(_error_row(pe, g))
    return buckets


def _error_row(pe: dict[str, Any], g: dict[str, Any]) -> dict[str, Any]:
    """Compact record of a gold/pred example pair (used by several helpers)."""
    return {
        "nota_id": pe["nota_id"],
        "gold_tiene": bool(g["extraccion"].get("tiene_eventos_protesta")),
        "pred_tiene": bool(pe["parsed"]["extraccion"].get("tiene_eventos_protesta")),
        "gold_total": g["extraccion"].get("total_eventos_protesta", 0),
        "pred_total": pe["parsed"]["extraccion"].get("total_eventos_protesta", 0),
        "f1": round(_per_example_f1(pe), 4),
    }


def _event_count_errors(
    per_example: list[dict[str, Any]], gold: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Per-example event count deltas (gold vs pred)."""
    rows: list[dict[str, Any]] = []
    for pe, g in zip(per_example, gold):
        g_total = g["extraccion"].get("total_eventos_protesta", 0)
        p_total = pe["parsed"]["extraccion"].get("total_eventos_protesta", 0)
        rows.append(
            {
                "nota_id": pe["nota_id"],
                "gold_total": g_total,
                "pred_total": p_total,
                "delta": p_total - g_total,
                "kind": (
                    "extra_event" if p_total > g_total else "missing_event" if p_total < g_total else "ok"
                ),
                "f1": round(_per_example_f1(pe), 4),
            }
        )
    rows.sort(key=lambda r: abs(r["delta"]), reverse=True)
    return rows


# Canonical list of incident-presence paths. These six share the same
# structure (one boolean presencia per incident kind, inside each event) and
# are the paths that D.12 / F.3 aggregate against. Keeping the list canonical
# here means `_incident_diff_summary` and the markdown renderer can iterate
# without duplicating the path strings.
INCIDENT_PRESENCIA_PATHS: list[str] = [
    "extraccion.eventos_protesta[].incidentes.represion.presencia",
    "extraccion.eventos_protesta[].incidentes.enfrentamiento.presencia",
    "extraccion.eventos_protesta[].incidentes.detenidos.presencia",
    "extraccion.eventos_protesta[].incidentes.heridos.presencia",
    "extraccion.eventos_protesta[].incidentes.muertos.presencia",
    "extraccion.eventos_protesta[].incidentes.danios_materiales.presencia",
]


def _incident_diff_summary(
    per_example: list[dict[str, Any]], gold: list[dict[str, Any]]
) -> dict[str, Any]:
    """Aggregate the incident-presencia error pattern across all 6 paths.

    The motivation: the review found that the original report claimed incident
    errors were "muchos casos predicen S/D cuando el gold es No". That claim
    is not supported by the data — when both gold and pred are non-None
    (i.e. the slot actually exists in both sides), nearly every comparison is
    either (No, No), (S/D, S/D) or (Sí, Sí). The vast majority of incident
    "errors" are index-alignment artefacts: when pred_total != gold_total,
    extra_event and missing_event rows propagate into the incident fields.

    This helper quantifies the breakdown so the report can show the real
    pattern. Returns:

    - ``total_comparisons``: total aligned (gold, pred) entries across paths
    - ``by_kind``: ``{match, missing, extra, mismatch}`` counts (summed over
      all 6 paths). ``extra`` and ``missing`` are downstream of event count
      errors; only ``mismatch`` is a direct incidente-presencia error.
    - ``by_pair``: ``{(gold_value, pred_value): count}`` for entries where
      both gold and pred are non-None. This is the population that the old
      recommendation mistakenly framed as "pred=S/D vs gold=No" — in fact it
      is overwhelmingly (No, No) and (S/D, S/D) matches plus at most a
      handful of true mismatches.
    - ``true_direct_mismatch``: number of ``mismatch`` entries where both
      gold and pred are non-None and the values differ. This is the only
      population where an "S/D vs No" or "Sí vs No" direct incidente error
      actually exists.
    - ``per_path``: per-path counts (matches, missing, extra, mismatch) for
      traceability.
    """
    from collections import Counter

    by_kind: Counter[str] = Counter()
    by_pair: Counter[tuple[Any, Any]] = Counter()
    per_path: dict[str, dict[str, int]] = {}
    true_direct_mismatch = 0

    for path in INCIDENT_PRESENCIA_PATHS:
        entries = _gold_vs_pred_for_path(per_example, gold, path)
        kinds = Counter(e["tipo_error"] for e in entries)
        per_path[path.split(".")[-2]] = {
            "match": kinds.get("match", 0),
            "missing": kinds.get("missing", 0),
            "extra": kinds.get("extra", 0),
            "mismatch": kinds.get("mismatch", 0),
        }
        for k, v in kinds.items():
            by_kind[k] += v
        for e in entries:
            g, p = e["gold"], e["pred"]
            if g is not None and p is not None:
                by_pair[(g, p)] += 1
                if e["tipo_error"] == "mismatch":
                    true_direct_mismatch += 1

    return {
        "total_comparisons": sum(by_kind.values()),
        "by_kind": {
            "match": by_kind.get("match", 0),
            "missing": by_kind.get("missing", 0),
            "extra": by_kind.get("extra", 0),
            "mismatch": by_kind.get("mismatch", 0),
        },
        "by_pair": dict(by_pair),
        "true_direct_mismatch": true_direct_mismatch,
        "per_path": per_path,
    }


def _per_path_field_errors(
    per_example: list[dict[str, Any]], gold: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Tally FP / FN / TP over the categorical paths, summed across all examples.

    Uses the same index-aligned enumeration as ``baseline_qwen_full.categorical_accuracy``,
    so the per-path totals here match the aggregate counts in
    ``finetuned_qwen-protesta-v1-r32.json``.
    """
    rows: list[dict[str, Any]] = []
    for name, path in EVENT_CATEGORICAL_PATHS:
        tp = tn = fp = fn = 0
        for pe, g in zip(per_example, gold):
            g_vals = _resolve_path(g, path)
            p_vals = _resolve_path(pe["parsed"], path)
            n = max(len(g_vals), len(p_vals))
            for i in range(n):
                gv = g_vals[i] if i < len(g_vals) else None
                pv = p_vals[i] if i < len(p_vals) else None
                if gv is None and pv is None:
                    continue
                if gv is None or pv is None:
                    if gv is not None:
                        fn += 1
                    else:
                        fp += 1
                    continue
                if gv == pv:
                    if isinstance(gv, bool):
                        if gv:
                            tp += 1
                        else:
                            tn += 1
                    else:
                        tp += 1
                else:
                    fp += 1
                    fn += 1
        support = tp + tn + fp + fn
        accuracy = _safe_div(tp + tn, support)
        rows.append(
            {
                "path": name,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "support": support,
                "accuracy": round(accuracy, 4),
                "error_rate": round(1 - accuracy, 4),
            }
        )
    rows.sort(key=lambda r: (r["error_rate"], -r["support"]), reverse=True)
    return rows


def _gold_vs_pred_for_path(
    per_example: list[dict[str, Any]],
    gold: list[dict[str, Any]],
    path: str,
) -> list[dict[str, Any]]:
    """Resolve a single path across all examples and return aligned (gold, pred) pairs."""
    out: list[dict[str, Any]] = []
    for pe, g in zip(per_example, gold):
        g_vals = _resolve_path(g, path)
        p_vals = _resolve_path(pe["parsed"], path)
        n = max(len(g_vals), len(p_vals))
        for i in range(n):
            gv = g_vals[i] if i < len(g_vals) else None
            pv = p_vals[i] if i < len(p_vals) else None
            if gv == pv:
                kind = "match"
            elif gv is not None and pv is None:
                kind = "missing"
            elif gv is None and pv is not None:
                kind = "extra"
            else:
                kind = "mismatch"
            out.append(
                {
                    "nota_id": pe["nota_id"],
                    "index": pe["index"],
                    "path": path,
                    "slot": i,
                    "gold": gv,
                    "pred": pv,
                    "match": gv == pv,
                    "tipo_error": kind,
                }
            )
    return out


def _categorical_summary(per_example: list[dict[str, Any]]) -> dict[str, Any]:
    """Re-extract the per-example categorical accuracy totals (cheaper than recomputing)."""
    aggregate_tp = aggregate_tn = aggregate_fp = aggregate_fn = 0
    for pe in per_example:
        agg = pe["categorical_accuracy_vs_gold"].get("__aggregate__", {})
        aggregate_tp += agg.get("tp", 0)
        aggregate_tn += agg.get("tn", 0)
        aggregate_fp += agg.get("fp", 0)
        aggregate_fn += agg.get("fn", 0)
    support = aggregate_tp + aggregate_tn + aggregate_fp + aggregate_fn
    accuracy = _safe_div(aggregate_tp + aggregate_tn, support)
    return {
        "tp": aggregate_tp,
        "tn": aggregate_tn,
        "fp": aggregate_fp,
        "fn": aggregate_fn,
        "support": support,
        "accuracy": round(accuracy, 4),
    }


def _worst_examples(
    per_example: list[dict[str, Any]],
    gold: list[dict[str, Any]],
    k: int = 5,
) -> list[dict[str, Any]]:
    scored = [(pe, _per_example_f1(pe)) for pe in per_example]
    scored.sort(key=lambda x: x[1])
    rows: list[dict[str, Any]] = []
    for pe, f1 in scored[:k]:
        g = gold[pe["index"]]
        rows.append(
            {
                "rank": len(rows) + 1,
                "index": pe["index"],
                "nota_id": pe["nota_id"],
                "f1": round(f1, 4),
                "tp": pe["f1_vs_gold"]["tp"],
                "fp": pe["f1_vs_gold"]["fp"],
                "fn": pe["f1_vs_gold"]["fn"],
                "gold_leaves": pe["f1_vs_gold"]["gold_leaves"],
                "pred_leaves": pe["f1_vs_gold"]["pred_leaves"],
                "gold_tiene": bool(g["extraccion"].get("tiene_eventos_protesta")),
                "pred_tiene": bool(pe["parsed"]["extraccion"].get("tiene_eventos_protesta")),
                "gold_total": g["extraccion"].get("total_eventos_protesta", 0),
                "pred_total": pe["parsed"]["extraccion"].get("total_eventos_protesta", 0),
                "pred_eventos": len(pe["parsed"]["extraccion"].get("eventos_protesta", [])),
                "finish_reason": pe.get("finish_reason"),
                "output_tokens": pe.get("output_tokens"),
            }
        )
    return rows


def _build_example_diff(
    pe: dict[str, Any],
    gold_record: dict[str, Any],
    paths: list[tuple[str, str, bool]],
) -> list[dict[str, Any]]:
    """For one example, produce gold/pred side-by-side rows for the selected paths."""
    rows: list[dict[str, Any]] = []
    for label, path, _is_scalar in paths:
        for entry in _gold_vs_pred_for_path([pe], [gold_record], path):
            rows.append(
                {
                    "campo": label,
                    "path": entry["path"],
                    "slot": entry["slot"],
                    "correcto_gold": entry["gold"],
                    "prediccion_modelo": entry["pred"],
                    "tipo_error": "match" if entry["match"] else (
                        "missing" if entry["gold"] is not None and entry["pred"] is None else (
                            "extra" if entry["gold"] is None and entry["pred"] is not None else "mismatch"
                        )
                    ),
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _render_markdown(payload: dict[str, Any]) -> str:
    m: dict[str, Any] = payload["metrics"]
    comparison = payload["headline_comparison"]
    tiene = m["categorical_accuracy"]["tiene_eventos_protesta"]
    field_errors = payload["field_errors"]
    worst = payload["worst_examples"]
    tiene_buckets = payload["tiene_buckets"]
    event_count = payload["event_count_errors"]

    # --- Section A: Executive summary ---------------------------------------
    lines: list[str] = []
    lines.append("# Reporte de análisis de errores — LoRA r=32 / alpha=64 / 3 epochs")
    lines.append("")
    lines.append(
        "**Modelo evaluado:** `Qwen/Qwen2.5-7B-Instruct` + LoRA `qwen-protesta-v1-r32` "
        "(rank=32, alpha=64, alpha/r=2.0, 3 epochs)."
    )
    lines.append(
        "**Adapter:** `checkpoints/qwen-protesta-v1-r32` "
        "(sha1 `6804aeb4d7f85b7d1b94574b1cab816017debbf7`)."
    )
    lines.append(
        "**Inferencia:** vLLM 0.23.0, `enable_lora=True`, `max_loras=1`, "
        "`SamplingParams(structured_outputs=StructuredOutputsParams(json=cleaned_schema))` "
        "contra `esquema_eventos_protesta_entrenamiento_MVS.json`."
    )
    lines.append(
        "**Eval set histórico:** 35 ejemplos del split previo de `data/chat_formatted/eval.jsonl` "
        "(gold = GPT-5.4-mini + validación humana de Nico, weight 1.0; "
        "`metadatos_extraccion.estado_validacion_humana: \"No validado\"` se ignora "
        "porque las 350 filas históricas del training data estaban validadas)."
    )
    lines.append("")
    lines.append("## A. Resumen ejecutivo")
    lines.append("")
    lines.append(
        "- **Mejor modelo histórico dentro de la corrida 350-era:** r=32 / alpha=64 / 3 epochs "
        "(`checkpoints/qwen-protesta-v1-r32`). Es el único modelo que cumple "
        "PLAN §6 schema_validity ≥ 0.95 y supera simultáneamente al baseline y al r16 en todas "
        "las métricas de contenido; r32 5e y r32 4e fueron controlados contra este y rinden peor."
    )
    lines.append(
        "- **Lo que funciona:** schema válido al 100% (35/35), parse válido al 100%, "
        "`field_recall.non_empty` cruza 0.7496 (vs 0.1692 del baseline), "
        "y la detección del booleano `tiene_eventos_protesta` salta de 0.2857 → 0.7714. "
        "El modelo ya no alucina sistemáticamente `nota_id=\"S/D\"` ni fechas con `day=19`."
    )
    lines.append(
        "- **Lo que falla:** las **categorías/enum** siguen por debajo del 0.80 del PLAN §6 "
        "(aggregate 0.4189; mejor path individual: `extraccion.tiene_eventos_protesta` 0.7714, "
        "peor: `contra_quien[].nivel_institucional` 0.1856). El **f1_global** (0.5350) y la "
        "**f1_recall** (0.5752) están por debajo del 0.70 del criterio MVP, y persisten "
        "**7 falsos positivos** en `tiene_eventos_protesta` (notas sin protesta real a las que "
        "el modelo les inventa 1-4 eventos)."
    )
    lines.append(
        "- **Conclusión:** el límite no es hiperparámetro. r=16 rindió 0.4637 y r=32/e=5 rindió "
        "0.5002 (más epochs **empeoran**: recall sube pero precision cae, "
        "síntoma clásico de sobreajuste al codebook sin nueva evidencia). Iterar Fase 6 con "
        "más epochs/rank no cierra la brecha. El plan de anotación dirigida "
        "(sección F) ataca los errores estructurales observados."
    )
    lines.append("")

    # --- Section B: Headline metrics table ---------------------------------
    lines.append("## B. Métricas globales — comparación de runs")
    lines.append("")
    lines.append(
        "Baseline = Qwen2.5-7B-Instruct sin fine-tuning. "
        "r16_3e y r32_3e son LoRA rank 16 / alpha 32 y rank 32 / alpha 64 sobre el mismo training set. "
        "r32_e5 = mismo checkpoint que r32_3e extendido a 5 epochs (ver `reports/phase6_r32_e5_eval.json`)."
    )
    lines.append("")
    lines.append(
        "| métrica | baseline | r16_3e | **r32_3e (mejor)** | r32_e5 | Δ r32_3e vs baseline |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name, key, fmt in [
        ("schema_validity", "schema_validity", "{:.4f}"),
        ("f1_global (f1)", "f1", "{:.4f}"),
        ("f1_precision", "f1_precision", "{:.4f}"),
        ("f1_recall", "f1_recall", "{:.4f}"),
        ("tiene_eventos_protesta (acc)", "tiene_acc", "{:.4f}"),
        ("categorical_accuracy aggregate", "cat_agg", "{:.4f}"),
        ("field_recall exact", "field_exact", "{:.4f}"),
        ("field_recall non_empty", "field_nonempty", "{:.4f}"),
    ]:
        row = comparison[name]
        lines.append(
            "| {name} | {b} | {r16} | {r32} | {e5} | {delta} |".format(
                name=name,
                b=fmt.format(row["baseline"]) if row["baseline"] is not None else "—",
                r16=fmt.format(row["r16_3e"]) if row["r16_3e"] is not None else "—",
                r32=fmt.format(row["r32_3e"]) if row["r32_3e"] is not None else "—",
                e5=fmt.format(row["r32_e5"]) if row["r32_e5"] is not None else "—",
                delta=(
                    ("{:+.4f}".format(row["delta_vs_baseline"]))
                    if row["delta_vs_baseline"] is not None
                    else "—"
                ),
            )
        )
    lines.append("")
    lines.append(
        "Δ = r32_3e − baseline. El **mejor modelo es r32_3e** en todas las métricas; "
        "r32_e5 (5 epochs) cae respecto a r32_3e en f1, categorical y field recall — más epochs "
        "sobreajustan sin aportar nueva evidencia."
    )
    lines.append("")
    lines.append("### B.1 Confusión `extraccion.tiene_eventos_protesta` (r32_3e)")
    lines.append("")
    lines.append(
        "|              | pred=False | pred=True |"
    )
    lines.append("|---|---:|---:|")
    lines.append(
        f"| **gold=True (27)**  | {tiene['fn']} FN | {tiene['tp']} TP |"
    )
    lines.append(
        f"| **gold=False (8)**  | {tiene['tn']} TN | {tiene['fp']} FP |"
    )
    lines.append("")
    lines.append(
        f"Accuracy: **{tiene['accuracy']:.4f}** sobre {tiene['support']} notas. "
        "El error dominante es **FP** (modelo inventa protesta en 7 notas que el gold marca como "
        "no-protesta). FN = 1 sólo: el modelo casi nunca deja de detectar una protesta real."
    )
    lines.append("")

    # --- Section C: Error taxonomy -----------------------------------------
    lines.append("## C. Taxonomía de errores")
    lines.append("")

    lines.append("### C.1 Falsos positivos / falsos negativos en `tiene_eventos_protesta`")
    lines.append("")
    lines.append("**Falsos positivos (gold=False, pred=True)** — el modelo inventa protesta:")
    lines.append("")
    lines.append("| nota_id | gold_total | pred_total | f1_vs_gold |")
    lines.append("|---|---:|---:|---:|")
    for r in tiene_buckets["fp"]:
        lines.append(
            f"| `{r['nota_id']}` | {r['gold_total']} | {r['pred_total']} | {r['f1']:.4f} |"
        )
    lines.append("")
    lines.append("**Falsos negativos (gold=True, pred=False)** — el modelo se pierde un evento:")
    lines.append("")
    if tiene_buckets["fn"]:
        lines.append("| nota_id | gold_total | pred_total | f1_vs_gold |")
        lines.append("|---|---:|---:|---:|")
        for r in tiene_buckets["fn"]:
            lines.append(
                f"| `{r['nota_id']}` | {r['gold_total']} | {r['pred_total']} | {r['f1']:.4f} |"
            )
    else:
        lines.append("_(ninguno)_")
    lines.append("")

    lines.append("### C.2 Errores de conteo de eventos (`total_eventos_protesta`)")
    lines.append("")
    lines.append(
        "Casos con mayor discrepancia |gold − pred| (top 10). Las notas con pred_total > gold_total "
        "suelen ser **FP de eventos** dentro de notas que sí tienen protesta (el modelo fragmenta "
        "un evento único en varios). Las notas con pred_total < gold_total son **FN de eventos** "
        "(sub-segmentación)."
    )
    lines.append("")
    lines.append("| nota_id | gold_total | pred_total | delta | tipo | f1 |")
    lines.append("|---|---:|---:|---:|---|---:|")
    for r in event_count[:10]:
        lines.append(
            f"| `{r['nota_id']}` | {r['gold_total']} | {r['pred_total']} | "
            f"{r['delta']:+d} | {r['kind']} | {r['f1']:.4f} |"
        )
    lines.append("")

    lines.append("### C.3 Paths categóricos con mayor tasa de error")
    lines.append("")
    lines.append(
        "Tabla histórica ordenada por error_rate (1 − accuracy) descendente. Soporte = total de "
        "comparaciones alineadas por índice sobre los 35 ejemplos del split previo."
    )
    lines.append("")
    lines.append(
        "| path | tp | tn | fp | fn | support | accuracy | error_rate |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in field_errors:
        lines.append(
            f"| `{r['path']}` | {r['tp']} | {r['tn']} | {r['fp']} | {r['fn']} | "
            f"{r['support']} | {r['accuracy']:.4f} | {r['error_rate']:.4f} |"
        )
    lines.append("")

    # --- Section D: Comparative gold-vs-pred tables ------------------------
    lines.append("## D. Tablas comparativas gold vs pred")
    lines.append("")
    lines.append(
        "Para cada path categórico priorizado por codebook/enums se listan las 10 comparaciones "
        "gold-vs-pred con mayor divergencia (mismatch o missing/extra). `tipo_error` ∈ "
        "{`match`, `mismatch`, `missing`, `extra`}."
    )
    lines.append("")

    priority_paths = [
        ("extraccion.tiene_eventos_protesta", "extraccion.tiene_eventos_protesta", True),
        ("extraccion.total_eventos_protesta", "extraccion.total_eventos_protesta", True),
        ("eventos_protesta[].temporalidad.tipo_temporal", "extraccion.eventos_protesta[].temporalidad.tipo_temporal", False),
        ("eventos_protesta[].temporalidad.tempo_verbal", "extraccion.eventos_protesta[].temporalidad.tempo_verbal", False),
        ("eventos_protesta[].delimitacion_evento.criterio_delimitacion", "extraccion.eventos_protesta[].delimitacion_evento.criterio_delimitacion", False),
        ("eventos_protesta[].accion.formato_principal.categoria", "extraccion.eventos_protesta[].accion.formato_principal.categoria", False),
        ("eventos_protesta[].sujetos[].categoria", "extraccion.eventos_protesta[].sujetos[].categoria", False),
        ("eventos_protesta[].demandas[].categoria", "extraccion.eventos_protesta[].demandas[].categoria", False),
        ("eventos_protesta[].contra_quien[].categoria", "extraccion.eventos_protesta[].contra_quien[].categoria", False),
        ("eventos_protesta[].contra_quien[].nivel_institucional", "extraccion.eventos_protesta[].contra_quien[].nivel_institucional", False),
        ("eventos_protesta[].lugares[].categoria", "extraccion.eventos_protesta[].lugares[].categoria", False),
        ("eventos_protesta[].incidentes.*.presencia (representativo)", "extraccion.eventos_protesta[].incidentes.represion.presencia", False),
    ]

    diffs_by_path = payload["diffs_by_path"]
    for d_index, (label, path, is_scalar) in enumerate(priority_paths, start=1):
        entries = diffs_by_path.get(label, [])
        bad = [e for e in entries if not e["match"]]
        if not bad:
            lines.append(f"### D.{d_index} `{label}`")
            lines.append("")
            lines.append("_(sin divergencias detectadas)_")
            lines.append("")
            continue
        # Take top-10 worst
        bad.sort(key=lambda e: (e["pred"] is None, e["gold"] is None))
        bad = bad[:10]
        lines.append(f"### D.{d_index} `{label}`")
        lines.append("")
        if is_scalar:
            lines.append(
                "| nota_id | correcto_gold | prediccion_modelo | tipo_error | por_que_importa | accion_recomendada |"
            )
            lines.append("|---|---|---|---|---|---|")
            for entry in bad:
                importance, action = _interpret_error(label, entry, payload.get("incident_summary"))
                lines.append(
                    "| `{nota}` | `{g}` | `{p}` | {kind} | {imp} | {act} |".format(
                        nota=entry["nota_id"],
                        g=_short(entry["gold"]),
                        p=_short(entry["pred"]),
                        kind=entry["tipo_error"],
                        imp=importance,
                        act=action,
                    )
                )
        else:
            lines.append(
                "| nota_id | slot | correcto_gold | prediccion_modelo | tipo_error | por_que_importa | accion_recomendada |"
            )
            lines.append("|---|---:|---|---|---|---|---|")
            for entry in bad:
                importance, action = _interpret_error(label, entry, payload.get("incident_summary"))
                lines.append(
                    "| `{nota}` | {slot} | `{g}` | `{p}` | {kind} | {imp} | {act} |".format(
                        nota=entry["nota_id"],
                        slot=entry["slot"],
                        g=_short(entry["gold"]),
                        p=_short(entry["pred"]),
                        kind=entry["tipo_error"],
                        imp=importance,
                        act=action,
                    )
                )
        lines.append("")

    # --- Section D.13 (aggregate incident evidence) ----------------------
    # Show the explicit per-kind / per-pair breakdown across all 6 incident
    # paths. This makes the evidence that drove the F.3 demotion visible to
    # the reader; without it the reader cannot tell whether the (per-row)
    # D.12 table is dominated by index-alignment extras/missings or by
    # genuine S/D vs No mismatches.
    inc = payload["incident_summary"]
    lines.append("### D.13 Aggregate `incidentes.*.presencia` (todos los 6 paths)")
    lines.append("")
    lines.append(
        "Las 6 paths `incidentes.*.presencia` (represion / enfrentamiento / "
        "detenidos / heridos / muertos / danios_materiales) se agregaron "
        "para cuantificar qué parte del error es **downstream del conteo de "
        "eventos** (extra/missing por desalineación por índice) y qué parte "
        "es **true mismatch** del booleano presencia. Esto es la evidencia "
        "que sustenta el F.3 (ver §F)."
    )
    lines.append("")
    lines.append("| path | match | missing (downstream) | extra (downstream) | mismatch (directo) |")
    lines.append("|---|---:|---:|---:|---:|")
    for path_name, counts in inc["per_path"].items():
        lines.append(
            f"| `incidentes.{path_name}.presencia` | {counts['match']} | "
            f"{counts['missing']} | {counts['extra']} | {counts['mismatch']} |"
        )
    total = inc["total_comparisons"]
    by_kind = inc["by_kind"]
    downstream = by_kind["missing"] + by_kind["extra"]
    direct = by_kind["mismatch"]
    lines.append(
        f"| **TOTAL** | **{by_kind['match']}** | **{by_kind['missing']}** | "
        f"**{by_kind['extra']}** | **{direct}** |"
    )
    lines.append("")
    lines.append(
        f"**Lectura:** sobre {total} comparaciones agregadas, "
        f"{downstream}/{total} ({100.0 * downstream / total:.1f}%) son "
        "downstream del conteo de eventos (extra cuando el modelo inventa "
        "un evento cuyo `incidentes.* = No`, missing cuando el modelo "
        "trunca un evento que el gold tenía). Sólo "
        f"{direct}/{total} ({100.0 * direct / total:.1f}%) son **true "
        "mismatches** del booleano presencia (gold y pred ambos no-None "
        "con valores distintos)."
    )
    lines.append("")
    # By-pair breakdown (only the populated pairs; sorted by count desc)
    pair_items = sorted(inc["by_pair"].items(), key=lambda kv: -kv[1])
    if pair_items:
        lines.append("**Pares (gold, pred) cuando ambos son no-None:**")
        lines.append("")
        lines.append("| gold | pred | count |")
        lines.append("|---|---|---:|")
        for (g, p), c in pair_items:
            lines.append(f"| `{g}` | `{p}` | {c} |")
        lines.append("")
        lines.append(
            "No se observa el patrón 'pred=S/D cuando gold=No' que la "
            "versión previa del reporte afirmaba: los pares no-None son "
            "mayormente matches (No→No, S/D→S/D, Sí→Sí); los mismatches "
            "directos son puntuales y se reportan explícitamente arriba."
        )
        lines.append("")

    # --- Section E: Worst examples ----------------------------------------
    lines.append("## E. Peores ejemplos (micro-f1)")
    lines.append("")
    lines.append(
        "Orden por micro-f1 ascendente sobre las hojas aplanadas (`f1_vs_gold.tp/fp/fn`). "
        "Los casos con `gold_total=0, pred_total≥1` son **FP puros** (notas sin protesta a las "
        "que el modelo les inventa eventos). Los casos con `pred_total << gold_total` pero `pred_leaves >> gold_leaves` "
        "indican fragmentación + drift."
    )
    lines.append("")
    lines.append("| rank | nota_id | f1 | tp/fp/fn | gold_leaves | pred_leaves | gold_total | pred_total | finish | out_tokens |")
    lines.append("|---:|---|---:|---|---:|---:|---:|---:|---|---:|")
    for r in worst:
        lines.append(
            f"| {r['rank']} | `{r['nota_id']}` | {r['f1']:.4f} | "
            f"{r['tp']}/{r['fp']}/{r['fn']} | {r['gold_leaves']} | {r['pred_leaves']} | "
            f"{r['gold_total']} | {r['pred_total']} | {r['finish_reason']} | {r['output_tokens']} |"
        )
    lines.append("")

    # Per-worst-example compact diff
    lines.append("### E.1 Diff compacto por peor ejemplo")
    lines.append("")
    lines.append(
        "Para cada uno de los 5 peores se listan las divergencias gold/pred en los paths del codebook "
        "más sensibles. Esto sirve para auditoría humana y para guiar la anotación dirigida."
    )
    lines.append("")

    example_diffs = payload["example_diffs"]
    for idx, diff in enumerate(example_diffs, start=1):
        pe = payload["per_example_indexed"][diff["nota_id"]]
        worst_entry = next(w for w in worst if w["nota_id"] == diff["nota_id"])
        lines.append(f"#### E.1.{idx} `{diff['nota_id']}` — f1={worst_entry['f1']:.4f}")
        lines.append("")
        lines.append(
            f"- gold_total={worst_entry['gold_total']}, pred_total={worst_entry['pred_total']}, "
            f"gold_leaves={worst_entry['gold_leaves']}, pred_leaves={worst_entry['pred_leaves']}, "
            f"tp/fp/fn = {worst_entry['tp']}/{worst_entry['fp']}/{worst_entry['fn']}."
        )
        lines.append("")
        lines.append("| campo/path | slot | correcto_gold | prediccion_modelo | tipo_error |")
        lines.append("|---|---:|---|---|---|")
        for row in diff["rows"]:
            lines.append(
                f"| `{row['campo']}` | {row['slot']} | `{_short(row['correcto_gold'])}` | "
                f"`{_short(row['prediccion_modelo'])}` | {row['tipo_error']} |"
            )
        lines.append("")
        if worst_entry["gold_tiene"] is False and worst_entry["pred_tiene"] is True:
            # Use the actual tp/fp/fn from the row, not a stale hardcoded
            # constant. The previous template hardcoded "tp=3 / fp=251" which
            # was only correct for rank-1; for ranks 2-5 the real numbers
            # differ (e.g. 2/127/4, 2/64/4, 3/63/3) and the stale text misled
            # readers about the per-example balance.
            tp_n = worst_entry["tp"]
            fp_n = worst_entry["fp"]
            fn_n = worst_entry["fn"]
            lines.append(
                "> **Nota:** el gold marca la nota como **sin protesta**; el modelo **alucina "
                f"{worst_entry['pred_total']} evento(s)** completos (con lugares, demandas, "
                f"incidentes, etc.). En este ejemplo el desbalance es "
                f"**tp={tp_n} / fp={fp_n} / fn={fn_n}** sobre "
                f"{worst_entry['gold_leaves']} hojas gold vs {worst_entry['pred_leaves']} "
                "hojas predichas."
            )
            lines.append("")
        elif worst_entry["pred_total"] > worst_entry["gold_total"]:
            lines.append(
                f"> **Nota:** el gold tiene {worst_entry['gold_total']} evento(s); el modelo "
                f"produce {worst_entry['pred_total']} eventos (**extra_event** — fragmentación). "
                "Esto infla los FP y arrastra el f1 abajo."
            )
            lines.append("")

    # --- Section F: Annotation plan ----------------------------------------
    lines.append("## F. Plan de anotación dirigida (siguiente iteración)")
    lines.append("")
    lines.append(
        "Las recomendaciones están ordenadas por **impacto esperado** sobre las métricas "
        "actuales. La evidencia concreta está en las secciones C/D/E. **No se recomienda** "
        "más datos genéricos: el análisis muestra errores estructurales por tipo de nota y "
        "por categoría del codebook, no por tamaño del set."
    )
    lines.append("")

    # Compute annotation targets dynamically from observed errors. The
    # incident_summary is the evidence-backed breakdown of incident errors
    # (used to demote the old "S/D vs No" claim into a downstream-of-event-
    # count statement — see F.3 and D.12).
    targets = _derive_annotation_targets(payload, payload["incident_summary"])
    for t in targets:
        lines.append(f"### F.{targets.index(t) + 1} {t['title']}")
        lines.append("")
        lines.append(f"**Evidencia observada:** {t['evidence']}")
        lines.append("")
        lines.append(f"**Acción concreta:** {t['action']}")
        lines.append("")
        lines.append(f"**Métrica objetivo a mover:** {t['target_metric']}")
        lines.append("")
        lines.append(f"**Criterio de éxito:** {t['success_criterion']}")
        lines.append("")

    # --- Section G: Caveats -----------------------------------------------
    lines.append("## G. Caveats")
    lines.append("")
    lines.append(
        "- **Eval set histórico tiene sólo 35 ejemplos.** Las métricas son evidencia direccional, no "
        "intervalos de confianza. Errores agregados (1862 comparaciones categóricas) son "
        "robustos a este tamaño, pero cada bin individual (ej. `Sujetos[].categoria`) tiene "
        "pocos ejemplos y un par de correcciones pueden moverlo mucho."
    )
    lines.append(
        "- **Gold histórico es gold dentro de esa corrida.** Las 35 notas son parte del set histórico de 350 producidas por "
        "GPT-5.4-mini + validación humana de Nico (weight 1.0). El campo "
        "`metadatos_extraccion.estado_validacion_humana: \"No validado\"` se ignora: es un "
        "placeholder engañoso heredado del pipeline. Si una nota histórica de las 35 parece mal "
        "anotada, se discute antes de cambiar el training data."
    )
    lines.append(
        "- **Alineación por índice.** Las comparaciones categóricas alinean `eventos_protesta[].*` "
        "por índice, no por contenido. Cuando pred_total ≠ gold_total, los slots extra/missing "
        "pueden arrastrar falsos FP/FN de categorías estructurales (no sólo semánticas). "
        "Por eso este reporte separa `C.2 errores de conteo` de `C.3 errores categóricos`."
    )
    lines.append(
        "- **Origen del baseline correcto.** El baseline de PLAN §6 es **Qwen2.5-7B-Instruct sin "
        "LoRA**, NO GPT-5.5. Cualquier mención histórica a `gpt-5.5` en artefactos es ruido y "
        "no se usa como baseline ni como origen del training data."
    )
    lines.append(
        "- **No se corrió entrenamiento ni inferencia para producir este reporte histórico.** "
        "Los números vienen literal de `metrics/finetuned_qwen-protesta-v1-r32.json` y de "
        "`data/chat_formatted/eval.jsonl` (gold). El script sólo agrega y compara."
    )
    lines.append("")

    # --- Sources -----------------------------------------------------------
    lines.append("## Sources (read-only)")
    lines.append("")
    lines.append(
        "- `metrics/finetuned_qwen-protesta-v1-r32.json` — métricas completas del modelo evaluado\n"
        "- `metrics/finetuned_qwen-protesta-v1-r32_outputs.jsonl` — raw outputs del run\n"
        "- `metrics/finetuned_qwen-protesta-v1.json` — r16_3e (para comparación)\n"
        "- `metrics/finetuned_qwen-protesta-v1-r32-e5.json` — r32_e5 (control de epochs)\n"
        "- `metrics/baseline_qwen2.5-7b.json` — Phase 2 baseline\n"
        "- `data/chat_formatted/eval.jsonl` — gold assistant JSON histórico (35 ejemplos)\n"
        "- `esquema_eventos_protesta_entrenamiento_MVS.json` — MVS schema con enums\n"
        "- `reports/phase6_r32_eval.json`, `reports/phase6_r32_e5_eval.json` — readiness reports\n"
        "- `scripts/baseline_qwen_full.py` — `EVENT_CATEGORICAL_PATHS` y `_resolve_path` (helpers puros)\n"
        "- `scripts/analyze_r32_errors.py` — generador de este reporte"
    )

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Helpers for human-friendly text
# ---------------------------------------------------------------------------


def _short(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, str):
        return value if len(value) <= 60 else value[:57] + "..."
    return json.dumps(value, ensure_ascii=False)


def _interpret_error(
    label: str,
    entry: dict[str, Any],
    incident_summary: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """Return (por_que_importa, accion_recomendada) for a single mismatch row.

    ``incident_summary`` is only consulted for the ``eventos_protesta[].incidentes``
    branch so that the percentages it cites are computed from the actual
    aggregate (non-match / total denominators), not from a hardcoded number.
    """
    g = entry["gold"]
    p = entry["pred"]
    kind = entry["tipo_error"]

    if label == "extraccion.tiene_eventos_protesta":
        if kind == "mismatch":
            # Distinguish the two directions using the actual values
            if g is False and p is True:
                return (
                    "Modelo inventa protesta en nota sin protesta real (FP).",
                    "Anotar más notas gold=False con conceptos conflict-adjacent (paro, huelga, "
                    "amenaza) que NO constituyan protesta para enseñar el borde.",
                )
            if g is True and p is False:
                return (
                    "Modelo pierde una protesta real (FN).",
                    "Anotar más notas gold=True con eventos sutiles (asamblea, estado de alerta, "
                    "reclamo administrativo) como positivos débiles.",
                )
        return ("", "")

    if label == "extraccion.total_eventos_protesta":
        if kind in ("mismatch", "extra", "missing"):
            return (
                "Conteo de eventos difiere del gold.",
                "Anotar ejemplos donde múltiples acciones compartan delimitación "
                "(acción principal + complementarias) para enseñar la regla 'no fragmentes'.",
            )
        return ("", "")

    if label == "eventos_protesta[].temporalidad.tipo_temporal":
        return (
            "Hecho vs Anuncio vs S/D cambia el tratamiento del evento (¿ocurrió o se anunció?).",
            "Anotar ejemplos donde el verbo esté en futuro o haya cita tipo 'se resolvió/convocó' "
            "para reforzar Hecho vs Anuncio.",
        )

    if label == "eventos_protesta[].temporalidad.tempo_verbal":
        return (
            "Pasado/Presente/Futuro/S/D es ortogonal a tipo_temporal; mal clasificado afecta "
            "análisis de series temporales.",
            "Anotar ejemplos con verbos en presente de noticias ('se declara', 'continúa') "
            "para enseñar Presente vs Pasado en notas del día.",
        )

    if label == "eventos_protesta[].delimitacion_evento.criterio_delimitacion":
        return (
            "Criterio de delimitación es la regla que evita fragmentar eventos; alto FP/FN aquí "
            "explica parte del drift en total_eventos.",
            "Anotar ejemplos con 'acción principal + complementarias' vs 'temporal' vs "
            "'evento único' para reforzar la distinción del codebook §3.",
        )

    if label == "eventos_protesta[].accion.formato_principal.categoria":
        return (
            "Categoría de formato principal es el primer nivel taxonómico de la acción.",
            "Anotar ejemplos de cada categoría (Huelgas / Cortes / Asambleas / "
            "Manifestaciones / Manifestaciones de baja intensidad / Ocupaciones / Ataques / "
            "Acciones judiciales / Reuniones / Residuales) para balancear.",
        )

    if label == "eventos_protesta[].sujetos[].categoria":
        return (
            "Categoría del sujeto es ambigua en notas con sujetos colectivos (sindicato vs "
            "asalariados vs militantes).",
            "Anotar ejemplos donde el sujeto sea un sindicato con personería gremial (Asalariados), "
            "una organización política (Militantes) o un grupo vecinal (Vecinos).",
        )

    if label == "eventos_protesta[].demandas[].categoria":
        return (
            "Categorías de demanda (Salarial / Laboral / Política / Gremial / Económica / "
            "Seguridad / etc.) tienen solapamiento semántico.",
            "Anotar ejemplos donde la demanda sea salarial vs laboral vs gremial (intra-sindical) "
            "para reforzar la distinción.",
        )

    if label == "eventos_protesta[].contra_quien[].categoria":
        return (
            "Destinatario 'Sindicatos' vs 'Estado/Gobierno' vs 'Patronal' es estructuralmente "
            "ambiguo en paros.",
            "Anotar ejemplos donde el 'contra_quien' sea otro sindicato (paros de solidaridad), "
            "el Estado como empleador, o la patronal privada.",
        )

    if label == "eventos_protesta[].contra_quien[].nivel_institucional":
        return (
            "Nivel institucional (Municipal/Provincial/Nacional/Internacional/Privado/No aplica) "
            "es el path con peor accuracy (0.1856).",
            "Anotar explícitamente el nivel cuando el destinatario sea el Estado (Nacional vs "
            "Provincial vs Municipal) o una empresa privada (Privado).",
        )

    if label == "eventos_protesta[].lugares[].categoria":
        return (
            "Categoría de lugar (Vía pública / Instituciones públicas / Sede patronal / "
            "Lugar de Trabajo / Sede sindical) se confunde en notas con múltiples espacios.",
            "Anotar ejemplos donde el evento ocurra en sede sindical, patronal o en la vía "
            "pública para reforzar la distinción.",
        )

    if label.startswith("eventos_protesta[].incidentes"):
        # The previous recommendation framed this as "modelo predice S/D cuando
        # el gold es No". That is not what the data shows. Across the 6
        # incidente-presencia paths, when both gold and pred are non-None the
        # pairs are overwhelmingly (No, No) and (S/D, S/D) matches; the
        # ``mismatch`` population is tiny (typically <5 across all 6 paths).
        # Most incidente "errors" are ``extra`` (pred invents an event whose
        # incidente is ``No``) or ``missing`` (gold has the event, pred
        # truncates it). Those are downstream of event count / alignment
        # errors and will not be fixed by annotating more incidente booleans.
        # The recommendation therefore points at the actual root cause
        # (event boundaries / no-event hard negatives), with the incidente
        # booleans as a secondary, side-effect metric.
        #
        # Numbers are derived from ``incident_summary`` so the denominator
        # is always explicit. Two views are reported:
        #   - among the non-match rows (the natural error population),
        #   - over all comparisons (the broader base-rate view).
        by_kind = (incident_summary or {}).get("by_kind", {})
        total = (incident_summary or {}).get("total_comparisons", 0)
        match = by_kind.get("match", 0)
        missing = by_kind.get("missing", 0)
        extra = by_kind.get("extra", 0)
        mismatch = by_kind.get("mismatch", 0)
        downstream = missing + extra
        non_match = downstream + mismatch
        if non_match > 0 and total > 0:
            pct_downstream_of_nonmatch = 100.0 * downstream / non_match
            pct_mismatch_of_nonmatch = 100.0 * mismatch / non_match
            pct_downstream_of_total = 100.0 * downstream / total
            importance = (
                f"Entre los incidentes con error (non-match), "
                f"{downstream}/{non_match} = {pct_downstream_of_nonmatch:.1f}% son "
                "missing/extra por alineación (downstream del conteo de eventos); "
                f"sólo {mismatch}/{non_match} = {pct_mismatch_of_nonmatch:.1f}% "
                "es mismatch directo del booleano presencia. "
                f"Sobre el total de {total} comparisons, "
                f"downstream = {pct_downstream_of_total:.1f}%. "
                "Cuando ambos gold y pred son no-None, los pares son mayormente "
                "(No, No) y (S/D, S/D) — no hay un patrón 'pred=S/D, gold=No'."
            )
        else:
            # Fallback if the aggregate isn't available: state the conclusion
            # without a percentage so we never invent a number.
            importance = (
                "Los errores de incidente son en su mayoría downstream del "
                "conteo de eventos (extras y missings por desalineación por "
                "índice), NO del booleano presencia. Cuando ambos gold y pred "
                "son no-None, los pares son mayormente (No, No) y (S/D, S/D) "
                "— no hay un patrón 'pred=S/D, gold=No'."
            )
        return (
            importance,
            "Priorizar anotación de delimitación de evento y no-event hard "
            "negatives (F.1 + F.4/F.5). La accuracy de incidente subirá como "
            "side-effect; no hace falta anotar incidente adicional.",
        )

    return ("", "")


def _derive_annotation_targets(
    payload: dict[str, Any],
    incident_summary: dict[str, Any],
) -> list[dict[str, str]]:
    """Build a prioritized annotation plan from the observed errors."""
    field_errors = payload["field_errors"]
    tiene_buckets = payload["tiene_buckets"]
    worst = payload["worst_examples"]
    event_count = payload["event_count_errors"]

    # Counts that drive the targets
    fp_count = len(tiene_buckets["fp"])
    worst_with_hallucinated_events = sum(
        1 for r in worst if r["gold_tiene"] is False and r["pred_tiene"] is True
    )

    # Top fields by error rate (skip trivial-support bins)
    field_errors_nontrivial = [r for r in field_errors if r["support"] >= 10]
    top_field = field_errors_nontrivial[0] if field_errors_nontrivial else field_errors[0]
    # Pick the next-best field that is NOT the same path as top_field, so the
    # annotation plan doesn't recommend the same fix twice in different words.
    second_field = next(
        (r for r in field_errors_nontrivial[1:] if r["path"] != top_field["path"]),
        None,
    )

    extra_event_count = sum(1 for r in event_count if r["kind"] == "extra_event")
    missing_event_count = sum(1 for r in event_count if r["kind"] == "missing_event")

    targets: list[dict[str, str]] = []

    targets.append(
        {
            "title": (
                f"Anotar {max(8, fp_count * 2)} notas **gold=False** con léxico conflict-adjacent "
                "(paro, huelga, advertencia, conflicto, tensión, medida de fuerza) que NO "
                "constituyan protesta real."
            ),
            "evidence": (
                f"{fp_count}/35 notas tienen FP en `tiene_eventos_protesta`; "
                f"{worst_with_hallucinated_events}/5 de los peores ejemplos son FP puros "
                "(gold_total=0, pred_total≥1). El modelo sobre-reacciona a palabras "
                "como 'paro', 'huelga', 'estado de alerta' aunque la nota no reporte protesta efectiva."
            ),
            "action": (
                "Buscar y anotar 8-15 notas del CSV `muestra_350_conflictos_1989_1995.csv` "
                "donde el texto mencione estos términos pero el gold correcto sea "
                "`tiene_eventos_protesta=false`. Usar como negativos difíciles para "
                "enseñar el borde del concepto."
            ),
            "target_metric": (
                "Reducir `extraccion.tiene_eventos_protesta` FP (hoy 7) y bajar "
                "`evento.es_evento_protesta` FP."
            ),
            "success_criterion": (
                "FP ≤ 3 sobre 35 en `tiene_eventos_protesta` y accuracy ≥ 0.85; "
                "recall no cae más de 0.02 respecto a r32_3e."
            ),
        }
    )

    targets.append(
        {
            "title": (
                "Anotar 10-20 ejemplos de **`contra_quien[].nivel_institucional`** "
                "explícitamente etiquetados (Municipal / Provincial / Nacional / Privado / "
                "Internacional / No aplica) cuando el destinatario sea el Estado o una empresa privada."
            ),
            "evidence": (
                f"`contra_quien[].nivel_institucional` es el path con **menor accuracy "
                f"({top_field['accuracy']:.4f})** sobre {top_field['support']} comparaciones; "
                "muchos casos predicen 'S/D' cuando el gold es 'Nacional' o 'Privado'."
            ),
            "action": (
                "Re-pasar manualmente las notas del set de entrenamiento con `contra_quien` "
                "presente y **forzar** el nivel_institucional cuando el texto mencione "
                "'Nación', 'Provincia', 'Municipalidad', 'empresa', 'firma', etc."
            ),
            "target_metric": (
                "`contra_quien[].nivel_institucional` accuracy ≥ 0.40 (hoy 0.1856)."
            ),
            "success_criterion": (
                "Accuracy ≥ 0.40 sobre el eval set y descenso del FN en ≥ 30% sin subir FP "
                "global por encima de 0.55."
            ),
        }
    )

    # F.3 — incidente-presencia target. The previous version of this report
# claimed "muchos casos predicen S/D cuando el gold es No" and recommended
# annotating explicit negations. That claim is not supported by the data.
# Replace it with an evidence-backed statement: incident errors are largely
# downstream of event count / alignment errors (extras and missings), and
# there are very few direct ``S/D vs No`` or ``Sí vs No`` mismatches. The
# actionable target is therefore "no new incidente annotation; verify the
# side-effect of F.4/F.5 on incidente accuracy", not "annotate incidente
# booleans".
    targets.append(
        {
            "title": (
                "NO priorizar anotación adicional de `incidentes.*.presencia`: "
                "los errores son downstream del conteo de eventos (extra/missing), "
                "no del booleano presencia."
            ),
            "evidence": (
                "Aggregate across the 6 `incidentes.*.presencia` paths "
                f"({incident_summary['total_comparisons']} comparaciones): "
                f"{incident_summary['by_kind']['match']} matches, "
                f"{incident_summary['by_kind']['missing']} missings, "
                f"{incident_summary['by_kind']['extra']} extras, "
                f"{incident_summary['by_kind']['mismatch']} "
                f"{'true mismatch' if incident_summary['by_kind']['mismatch'] == 1 else 'true mismatches'} "
                "(direct gold-vs-pred boolean error). Los `missing` y `extra` "
                "son artefactos de desalineación por índice cuando "
                "`pred_total != gold_total` — son errores de event boundary, "
                "no del booleano presencia. Cuando ambos gold y pred son "
                "no-None, los pares son abrumadoramente (No, No) y (S/D, S/D); "
                "no hay un patrón 'pred=S/D, gold=No'."
            ),
            "action": (
                "Re-pasar la métrica de `incidentes.*.presencia` después de "
                "aplicar F.1 (no-event hard negatives) y F.4/F.5 (delimitación "
                "de evento). Si la accuracy sube a ≥ 0.80 sin anotar incidente "
                "adicional, cerrar este F.3 como 'side-effect confirmado'. "
                "Si tras F.1+F.4/F.5 la accuracy sigue < 0.70, entonces sí "
                "abrir un target nuevo con la evidencia cuantificada de "
                "mismatch directo (no antes)."
            ),
            "target_metric": (
                "`incidentes.*.presencia` aggregate accuracy ≥ 0.80 después "
                "de F.1 + F.4/F.5, sin anotación adicional de incidente."
            ),
            "success_criterion": (
                "Aggregate accuracy de los 6 paths `incidentes.*.presencia` ≥ "
                "0.80, con `true_direct_mismatch` ≤ 2 en el eval set "
                f"(hoy {incident_summary['true_direct_mismatch']})."
            ),
        }
    )

    if second_field is not None and (
        top_field is None or second_field["path"] != top_field["path"]
    ):
        # Build a path-specific action hint so the recommendation actually
        # points the annotator at the right category pair rather than the
        # generic Salarial/Laboral text.
        path_hint = {
            "demandas[].categoria": (
                "Anotar notas donde la demanda sea salarial vs laboral vs gremial "
                "(inter-intra sindical) y revisar manualmente la consistencia con la "
                "regla del codebook §demandas."
            ),
            "accion.formato_principal.categoria": (
                "Anotar notas donde el formato principal sea Huelga vs Manifestación "
                "(de baja intensidad vs general) vs Asamblea vs Corte, y revisar "
                "manualmente la consistencia con la regla del codebook §accion."
            ),
            "sujetos[].categoria": (
                "Anotar notas donde el sujeto sea un sindicato con personería gremial "
                "(Asalariados), una organización política (Militantes) o un grupo "
                "vecinal (Vecinos) y revisar manualmente la consistencia."
            ),
            "lugares[].categoria": (
                "Anotar notas donde el lugar sea vía pública, sede sindical, sede "
                "patronal, lugar de trabajo o instituciones públicas y revisar "
                "manualmente la consistencia con la regla del codebook §lugares."
            ),
            "contra_quien[].categoria": (
                "Anotar notas donde el destinatario sea otro sindicato, el Estado como "
                "empleador, o la patronal privada y revisar manualmente la consistencia "
                "con la regla del codebook §contra_quien."
            ),
            "delimitacion.criterio_delimitacion": (
                "Anotar notas cuya regla de delimitación sea Accion principal con "
                "acciones complementarias vs Temporal vs Espacial vs Temporal y "
                "espacial vs Evento unico en la nota, y revisar manualmente la "
                "consistencia con la regla del codebook §3 (delimitación)."
            ),
        }
        default_hint = (
            "Anotar notas donde esta categoría sea ambigua y revisar manualmente la "
            "consistencia con la regla del codebook correspondiente."
        )
        targets.append(
            {
                "title": (
                    f"Anotar 10-20 ejemplos de **`{second_field['path']}`** "
                    "donde la categoría sea ambigua en el codebook."
                ),
                "evidence": (
                    f"`{second_field['path']}` tiene accuracy {second_field['accuracy']:.4f} sobre "
                    f"{second_field['support']} comparaciones; está en el top-2 de error_rate."
                ),
                "action": path_hint.get(second_field["path"], default_hint),
                "target_metric": (
                    f"`{second_field['path']}` accuracy ≥ 0.45."
                ),
                "success_criterion": (
                    "Accuracy del path ≥ 0.45 sin deteriorar aggregate categorical > 0.05."
                ),
            }
        )

    targets.append(
        {
            "title": (
                "Anotar **eventos correctamente delimitados** cuando hay acciones "
                "complementarias (corte + volanteada, paro + marcha, etc.) para "
                "reforzar `criterio_delimitacion = 'Accion principal con acciones complementarias'`."
            ),
            "evidence": (
                f"`delimitacion.criterio_delimitacion` accuracy 0.1977 (peor entre los "
                f"delimitadores). Hay {extra_event_count} ejemplos con extra_event y "
                f"{missing_event_count} con missing_event — ambos son síntomas de fragmentación."
            ),
            "action": (
                "Anotar 10-15 notas donde la regla 'acción principal + complementarias' aplique "
                "explícitamente, y verificar que `es_accion_principal_con_complementarias=true` "
                "se use correctamente."
            ),
            "target_metric": (
                "`delimitacion.criterio_delimitacion` accuracy ≥ 0.40 y "
                "`delimitacion.es_accion_principal_con_complementarias` accuracy ≥ 0.55."
            ),
            "success_criterion": (
                "Reducir extra_event / missing_event en ≥ 50% sobre el eval set y subir "
                "f1_global a ≥ 0.60."
            ),
        }
    )

    return targets


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _headline_comparison() -> dict[str, Any]:
    rows: dict[str, dict[str, Any]] = {
        "schema_validity": {"baseline": None, "r16_3e": None, "r32_3e": None, "r32_e5": None, "delta_vs_baseline": None},
        "f1_global (f1)": {"baseline": None, "r16_3e": None, "r32_3e": None, "r32_e5": None, "delta_vs_baseline": None},
        "f1_precision": {"baseline": None, "r16_3e": None, "r32_3e": None, "r32_e5": None, "delta_vs_baseline": None},
        "f1_recall": {"baseline": None, "r16_3e": None, "r32_3e": None, "r32_e5": None, "delta_vs_baseline": None},
        "tiene_eventos_protesta (acc)": {"baseline": None, "r16_3e": None, "r32_3e": None, "r32_e5": None, "delta_vs_baseline": None},
        "categorical_accuracy aggregate": {"baseline": None, "r16_3e": None, "r32_3e": None, "r32_e5": None, "delta_vs_baseline": None},
        "field_recall exact": {"baseline": None, "r16_3e": None, "r32_3e": None, "r32_e5": None, "delta_vs_baseline": None},
        "field_recall non_empty": {"baseline": None, "r16_3e": None, "r32_3e": None, "r32_e5": None, "delta_vs_baseline": None},
    }

    key_map = {
        "schema_validity": ("metrics", "schema_validity"),
        "f1_global (f1)": ("metrics", "f1_global", "f1"),
        "f1_precision": ("metrics", "f1_global", "precision"),
        "f1_recall": ("metrics", "f1_global", "recall"),
        "tiene_eventos_protesta (acc)": ("metrics", "categorical_accuracy", "tiene_eventos_protesta", "accuracy"),
        "categorical_accuracy aggregate": ("metrics", "categorical_accuracy", "per_path", "__aggregate__", "accuracy"),
        "field_recall exact": ("metrics", "field_recall", "exact_match_recall"),
        "field_recall non_empty": ("metrics", "field_recall", "non_empty_recovery_recall"),
    }

    for label, _path in COMPARISON_MODELS:
        data = _load_json(_path)
        for row_name, keys in key_map.items():
            node: Any = data
            for k in keys:
                if isinstance(node, dict) and k in node:
                    node = node[k]
                else:
                    node = None
                    break
            rows[row_name][label] = node

    # Delta vs baseline for the row whose 'r32_3e' was the best model in this historical run.
    for row_name, row in rows.items():
        b = row.get("baseline")
        r = row.get("r32_3e")
        if isinstance(b, (int, float)) and isinstance(r, (int, float)):
            row["delta_vs_baseline"] = round(r - b, 4)

    return rows


def _build_payload() -> dict[str, Any]:
    best = _load_json(BEST_METRICS_PATH)
    per_example = best["per_example"]
    gold = _load_gold(EVAL_JSONL)
    assert len(per_example) == len(gold), "per_example/gold length mismatch"

    field_errors = _per_path_field_errors(per_example, gold)
    tiene_buckets = _tiene_confusion(per_example, gold)
    event_count = _event_count_errors(per_example, gold)
    worst = _worst_examples(per_example, gold, k=5)
    # Evidence-backed aggregate of the 6 incident-presencia paths. Used to
    # demote the old "S/D vs No" claim into a downstream-of-event-count
    # statement (see F.3 and D.12).
    incident_summary = _incident_diff_summary(per_example, gold)

    diffs_by_path: dict[str, list[dict[str, Any]]] = {}
    # Each tuple: (label, path, is_scalar)
    # is_scalar=True means the path resolves to a single value (not an array of
    # leaves); we hide the slot column for those to avoid a confusing "slot=0"
    # column on what is actually one comparison per nota.
    priority_paths = [
        ("extraccion.tiene_eventos_protesta", "extraccion.tiene_eventos_protesta", True),
        ("extraccion.total_eventos_protesta", "extraccion.total_eventos_protesta", True),
        ("eventos_protesta[].temporalidad.tipo_temporal", "extraccion.eventos_protesta[].temporalidad.tipo_temporal", False),
        ("eventos_protesta[].temporalidad.tempo_verbal", "extraccion.eventos_protesta[].temporalidad.tempo_verbal", False),
        ("eventos_protesta[].delimitacion_evento.criterio_delimitacion", "extraccion.eventos_protesta[].delimitacion_evento.criterio_delimitacion", False),
        ("eventos_protesta[].accion.formato_principal.categoria", "extraccion.eventos_protesta[].accion.formato_principal.categoria", False),
        ("eventos_protesta[].sujetos[].categoria", "extraccion.eventos_protesta[].sujetos[].categoria", False),
        ("eventos_protesta[].demandas[].categoria", "extraccion.eventos_protesta[].demandas[].categoria", False),
        ("eventos_protesta[].contra_quien[].categoria", "extraccion.eventos_protesta[].contra_quien[].categoria", False),
        ("eventos_protesta[].contra_quien[].nivel_institucional", "extraccion.eventos_protesta[].contra_quien[].nivel_institucional", False),
        ("eventos_protesta[].lugares[].categoria", "extraccion.eventos_protesta[].lugares[].categoria", False),
        ("eventos_protesta[].incidentes.*.presencia (representativo)", "extraccion.eventos_protesta[].incidentes.represion.presencia", False),
    ]
    for label, path, _is_scalar in priority_paths:
        diffs_by_path[label] = _gold_vs_pred_for_path(per_example, gold, path)

    # Per-example compact diff for the 5 worst
    pe_indexed = {p["nota_id"]: p for p in per_example}
    example_diffs: list[dict[str, Any]] = []
    for w in worst:
        pe = pe_indexed[w["nota_id"]]
        g = gold[pe["index"]]
        rows = _build_example_diff(pe, g, priority_paths)
        # Filter to keep only divergence rows, plus a few 'match' rows for context
        bad_rows = [r for r in rows if r["tipo_error"] != "match"]
        match_rows = [r for r in rows if r["tipo_error"] == "match"]
        # Keep up to 40 rows per example to avoid runaway output
        kept = bad_rows + match_rows[: max(0, 40 - len(bad_rows))]
        example_diffs.append({"nota_id": w["nota_id"], "rows": kept})

    payload: dict[str, Any] = {
        **HISTORICAL_METADATA,
        "model_key": BEST_MODEL_KEY,
        "model_label": "r32_3e",
        "metrics": best["metrics"],
        "adapter": best.get("adapter"),
        "checked_at": best.get("checked_at"),
        "headline_comparison": _headline_comparison(),
        "field_errors": field_errors,
        "tiene_buckets": tiene_buckets,
        "event_count_errors": event_count,
        "worst_examples": worst,
        "diffs_by_path": diffs_by_path,
        "example_diffs": example_diffs,
        "per_example_indexed": pe_indexed,
        "categorical_summary": _categorical_summary(per_example),
        "incident_summary": incident_summary,
    }
    return payload


def _write_csv_field_errors(payload: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        metadata_cols = list(HISTORICAL_METADATA)
        w.writerow(metadata_cols + ["path", "tp", "tn", "fp", "fn", "support", "accuracy", "error_rate"])
        for r in payload["field_errors"]:
            w.writerow(
                [HISTORICAL_METADATA[k] for k in metadata_cols]
                + [r["path"], r["tp"], r["tn"], r["fp"], r["fn"], r["support"], r["accuracy"], r["error_rate"]]
            )


def _write_csv_worst_examples(payload: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            list(HISTORICAL_METADATA)
            + [
                "rank",
                "nota_id",
                "f1",
                "tp",
                "fp",
                "fn",
                "gold_leaves",
                "pred_leaves",
                "gold_total",
                "pred_total",
                "pred_eventos",
                "gold_tiene",
                "pred_tiene",
                "finish_reason",
                "output_tokens",
            ]
        )
        for r in payload["worst_examples"]:
            w.writerow(
                [
                    *[HISTORICAL_METADATA[k] for k in HISTORICAL_METADATA],
                    r["rank"],
                    r["nota_id"],
                    r["f1"],
                    r["tp"],
                    r["fp"],
                    r["fn"],
                    r["gold_leaves"],
                    r["pred_leaves"],
                    r["gold_total"],
                    r["pred_total"],
                    r["pred_eventos"],
                    r["gold_tiene"],
                    r["pred_tiene"],
                    r["finish_reason"],
                    r["output_tokens"],
                ]
            )


def main() -> int:
    payload = _build_payload()

    md_path = METRICS_DIR / "error_analysis_r32_3e.md"
    json_path = METRICS_DIR / "error_analysis_r32_3e.json"
    csv_field_path = METRICS_DIR / "error_analysis_r32_3e_field_errors.csv"
    csv_worst_path = METRICS_DIR / "error_analysis_r32_3e_worst_examples.csv"

    md = _render_markdown(payload)
    md_path.write_text(md, encoding="utf-8")

    # Strip non-serializable bits for the JSON dump. Also normalise tuple
    # keys in the incident by_pair breakdown into "{gold} -> {pred}"
    # strings, since JSON object keys must be strings.
    def _stringify_pair_keys(obj):
        if isinstance(obj, dict):
            new = {}
            for k, v in obj.items():
                if isinstance(k, tuple):
                    key_str = " -> ".join(str(x) for x in k)
                    new[key_str] = _stringify_pair_keys(v)
                else:
                    new[k] = _stringify_pair_keys(v)
            return new
        if isinstance(obj, list):
            return [_stringify_pair_keys(x) for x in obj]
        return obj

    json_payload = {k: v for k, v in payload.items() if k != "per_example_indexed"}
    json_payload = _stringify_pair_keys(json_payload)
    json_path.write_text(json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    _write_csv_field_errors(payload, csv_field_path)
    _write_csv_worst_examples(payload, csv_worst_path)

    # Console summary
    metrics = payload["metrics"]
    print(f"Wrote {md_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_field_path}")
    print(f"Wrote {csv_worst_path}")
    print()
    print(f"Model: {payload['model_label']} (best={payload['model_key']})")
    print(f"  schema_validity = {metrics['schema_validity']}")
    print(f"  f1_global       = {metrics['f1_global']['f1']}")
    print(f"  cat aggregate   = {payload['categorical_summary']['accuracy']}")
    tiene = metrics["categorical_accuracy"]["tiene_eventos_protesta"]
    print(f"  tiene_acc       = {tiene['accuracy']} (TP={tiene['tp']}, TN={tiene['tn']}, FP={tiene['fp']}, FN={tiene['fn']})")
    print(f"  worst-1 f1      = {payload['worst_examples'][0]['f1']} ({payload['worst_examples'][0]['nota_id']})")
    print()
    print("Top 3 paths by error_rate:")
    for r in payload["field_errors"][:3]:
        print(f"  {r['path']:>55s} acc={r['accuracy']:.4f} err={r['error_rate']:.4f} support={r['support']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
