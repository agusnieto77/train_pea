#!/usr/bin/env python3
"""Phase 2 full baseline: 35-example Qwen2.5-7B-Instruct inference via vLLM.

Processes every row of ``data/chat_formatted/eval.jsonl`` through vLLM offline
inference with the MVS schema as a structured-output constraint, validates
each generated JSON against ``jsonschema.Draft202012Validator``, and writes
Phase 2 MVP metrics (schema_validity, categorical_accuracy, f1_global,
field_recall).

Design (deliberately mirrors ``scripts/baseline_qwen_smoke.py``):
  * vLLM 0.23.x: ``SamplingParams(structured_outputs=StructuredOutputsParams(json=<schema>))``.
  * Schema is cleaned before being handed to vLLM: ``$schema``/``$id``/``title``
    are stripped and ``const`` is rewritten to ``enum``.
  * ``max_model_len=20480``, ``max_num_seqs=1``, ``enforce_eager=True``,
    ``dtype="auto"`` (bfloat16). Same safe defaults that passed the smoke.
  * No silent truncation: per-example ``max_tokens`` is computed as
    ``min(max_tokens_cap, max_seq_length - prompt_tokens - safety_margin)``.
    If that budget is < 256 the example is recorded as blocked and the
    model is NOT asked to run.
  * Prompt uses system + user only. The gold assistant is never sent in
    the prompt (it's only used as the comparison target for metrics).
  * FlashInfer sampler path fails on sm_120 in this environment; the
    caller MUST set ``VLLM_USE_FLASHINFER_SAMPLER=0``. We record the env
    flag in the report.

Outputs:
  * ``metrics/baseline_qwen2.5-7b.json``        — machine-readable metrics + per-example records
  * ``metrics/baseline_qwen2.5-7b_outputs.jsonl`` — raw per-example outputs (one JSON per line)
  * ``metrics/qualitative_report.md``            — 5-10 short qualitative observations
  * ``reports/phase2_readiness.json`` (updated, full runs only) — full baseline status

Run classification (CRITICAL — protects Phase 3 from being unblocked by partial runs):

  * **Full baseline** (default, no ``--limit``): processes every row of
    ``data/chat_formatted/eval.jsonl``. The status may reach ``pass`` ONLY if
    every eval example was prepared (no pre-inference blocks) AND every output
    finished cleanly (no length truncations) AND at least one schema-valid
    output was produced. ``reports/phase2_readiness.json`` is updated with the
    ``full_baseline`` block and the top-level ``status`` reflects the full-35
    outcome. This is the run that can unblock Phase 3.
  * **Limited run** (``--limit N``): debug/smoke/sliced runs. The status is
    ALWAYS prefixed with ``partial_`` (``partial_pass``, ``partial_fail``,
    ``partial_incomplete``, ``partial_pass_with_truncations``). ``reports/
    phase2_readiness.json`` is NOT updated by default — a partial run must
    never claim full 35 baseline. Pass ``--update-readiness`` to opt in to a
    clearly-labeled partial write (writes a ``partial_baseline`` block only;
    does NOT touch ``status``, ``status_note``, or the ``full_baseline`` block
    from a previous full run). Additionally, limited runs AUTO-ROUTE the
    default official output paths (``metrics/baseline_qwen2.5-7b.json``,
    ``metrics/baseline_qwen2.5-7b_outputs.jsonl``,
    ``metrics/qualitative_report.md``) to sibling ``_partial`` paths so a
    debug invocation cannot overwrite the authoritative Phase 2 baseline
    artifacts. Pass explicit ``--metrics`` / ``--outputs`` / ``--qualitative``
    to opt out of the auto-route.

Usage:
    # Full 35-example baseline (the only run that may unblock Phase 3):
    VLLM_USE_FLASHINFER_SAMPLER=0 .venv/bin/python scripts/baseline_qwen_full.py

    # Debug/smoke with 3 examples — partial, never overwrites full baseline readiness:
    VLLM_USE_FLASHINFER_SAMPLER=0 .venv/bin/python scripts/baseline_qwen_full.py --limit 3

    # Debug/smoke that ALSO wants to log a partial status to readiness (explicit opt-in):
    VLLM_USE_FLASHINFER_SAMPLER=0 .venv/bin/python scripts/baseline_qwen_full.py \
        --limit 3 --update-readiness
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# -- Default paths ------------------------------------------------------------
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_EVAL = Path("data/chat_formatted/eval.jsonl")
DEFAULT_SCHEMA = Path("esquema_eventos_protesta_entrenamiento_MVS.json")
DEFAULT_METRICS = Path("metrics/baseline_qwen2.5-7b.json")
DEFAULT_OUTPUTS = Path("metrics/baseline_qwen2.5-7b_outputs.jsonl")
DEFAULT_QUAL = Path("metrics/qualitative_report.md")
DEFAULT_READINESS = Path("reports/phase2_readiness.json")
DEFAULT_MAX_SEQ_LENGTH = 20480
DEFAULT_MAX_TOKENS_CAP = 8192
DEFAULT_PROMPT_SAFETY_MARGIN = 16
DEFAULT_MIN_OUTPUT_BUDGET = 256

# Official full-baseline artifact paths. Limited runs (--limit set) MUST NOT
# write to these — resolve_limited_run_paths() below auto-routes any path that
# still points at one of these official locations to a `_partial` sibling so a
# debug/smoke invocation can never overwrite the authoritative Phase 2
# baseline metrics / outputs / qualitative report.
OFFICIAL_METRICS_PATH = DEFAULT_METRICS
OFFICIAL_OUTPUTS_PATH = DEFAULT_OUTPUTS
OFFICIAL_QUAL_PATH = DEFAULT_QUAL


# =============================================================================
# Schema + chat helpers (reused from baseline_qwen_smoke.py, kept identical
# so the smoke vs full results are directly comparable).
# =============================================================================
def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            rows.append({"line_no": line_no, "row": json.loads(line)})
    return rows


def clean_schema_for_vllm(schema: dict[str, Any]) -> dict[str, Any]:
    """Strip JSON-Schema meta keywords and rewrite const -> enum for vLLM."""

    def _convert(node: Any) -> Any:
        if isinstance(node, dict):
            for k in ("$schema", "$id", "title"):
                node.pop(k, None)
            if "const" in node and "enum" not in node:
                node["enum"] = [node.pop("const")]
            for v in node.values():
                _convert(v)
        elif isinstance(node, list):
            for v in node:
                _convert(v)
        return node

    return _convert(json.loads(json.dumps(schema)))


def derive_nota_id(eval_row: dict[str, Any], fallback: str) -> str:
    """Pull a stable nota_id out of the eval row (assistant content or user header)."""
    try:
        assistant = next(
            (m for m in eval_row["messages"] if m.get("role") == "assistant"), None
        )
        if assistant is not None:
            parsed = json.loads(assistant["content"])
            if "nota" in parsed and isinstance(parsed["nota"], dict):
                cand = parsed["nota"].get("nota_id")
                if cand:
                    return str(cand)
    except Exception:
        pass
    try:
        first_line = eval_row["messages"][1]["content"].splitlines()[0]
    except Exception:
        first_line = ""
    if first_line.startswith("FECHA DE EDICIÓN DE LA NOTA:"):
        return first_line.split(":", 1)[1].strip() or fallback
    return fallback


def validate_against_schema(parsed: Any, schema: dict[str, Any]) -> dict[str, Any]:
    """Run jsonschema Draft 2020-12 validation. Returns a small status dict."""
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:
        return {"available": False, "error": f"jsonschema import failed: {exc}"}
    try:
        validator = Draft202012Validator(schema)
        errors = sorted(validator.iter_errors(parsed), key=lambda e: list(e.path))
        if not errors:
            return {"available": True, "valid": True, "error_count": 0, "errors": []}
        return {
            "available": True,
            "valid": False,
            "error_count": len(errors),
            "errors": [
                {
                    "path": "/".join(str(p) for p in err.path) or "<root>",
                    "message": err.message,
                }
                for err in errors[:10]
            ],
        }
    except Exception as exc:
        return {"available": True, "valid": False, "error": f"{exc}"}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


# =============================================================================
# Tree-flattening + leaf-comparison metrics
# =============================================================================
# What we mean by a "leaf" for comparison: a primitive value (str / int / float /
# bool / None) at a JSON path. Containers (object / list) are walked into.
# Arrays of objects are aligned element-wise by index; we record the index in
# the path so gold vs pred alignment is transparent.

def _is_primitive(v: Any) -> bool:
    return v is None or isinstance(v, (str, int, float, bool))


def flatten_leaves(node: Any, prefix: str = "") -> list[tuple[str, Any]]:
    """Flatten any JSON value into a list of (path, leaf_value).

    For arrays of objects, items are aligned by index, so path encodes the
    index. For arrays of primitives, each item is a leaf at ``path[i]``.
    """
    if _is_primitive(node):
        return [(prefix or "<root>", node)]
    if isinstance(node, list):
        out: list[tuple[str, Any]] = []
        for i, item in enumerate(node):
            out.extend(flatten_leaves(item, f"{prefix}[{i}]"))
        return out
    if isinstance(node, dict):
        out = []
        for k, v in node.items():
            child_prefix = f"{prefix}.{k}" if prefix else k
            out.extend(flatten_leaves(v, child_prefix))
        return out
    return [(prefix or "<root>", repr(node))]


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def micro_f1_from_counts(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "support_tp": tp,
        "support_fp": fp,
        "support_fn": fn,
    }


def compare_leaves(
    gold: Any, pred: Any, prefix: str = "", out: dict[str, int] | None = None
) -> dict[str, int]:
    """Walk gold and pred together. Count TP / FP / FN over comparable leaves.

    Conventions (documented in the report so reviewers can audit):
      * Gold and pred trees are walked in parallel by exact path.
      * Arrays of any kind are aligned by index. Extra gold items beyond the
        pred length count as FN leaves; extra pred items beyond the gold
        length count as FP leaves.
      * For each comparable leaf (both sides primitive at the same path):
          - exact match: ``tp += 1``
          - mismatch:    ``fp += 1`` AND ``fn += 1`` (a wrong value is both a
            bad prediction and a missed gold value)
      * Gold leaf with no pred path: ``fn += 1``.
      * Pred leaf with no gold path: ``fp += 1``.
    """
    if out is None:
        out = {"tp": 0, "fp": 0, "fn": 0, "gold_leaves": 0, "pred_leaves": 0}

    if _is_primitive(gold) and _is_primitive(pred):
        out["gold_leaves"] += 1
        out["pred_leaves"] += 1
        if gold == pred:
            out["tp"] += 1
        else:
            out["fp"] += 1
            out["fn"] += 1
        return out

    if isinstance(gold, list) or isinstance(pred, list):
        g = gold if isinstance(gold, list) else []
        p = pred if isinstance(pred, list) else []
        n_gold, n_pred = len(g), len(p)
        n = max(n_gold, n_pred)
        for i in range(n):
            child_prefix = f"{prefix}[{i}]" if prefix else f"[{i}]"
            gi = g[i] if i < n_gold else None
            pi = p[i] if i < n_pred else None
            if gi is None and pi is not None:
                # extra pred leaf
                for p_path, _ in flatten_leaves(pi, child_prefix):
                    out["pred_leaves"] += 1
                    out["fp"] += 1
            elif pi is None and gi is not None:
                # extra gold leaf
                for g_path, _ in flatten_leaves(gi, child_prefix):
                    out["gold_leaves"] += 1
                    out["fn"] += 1
            else:
                compare_leaves(gi, pi, child_prefix, out)
        return out

    if isinstance(gold, dict) or isinstance(pred, dict):
        g = gold if isinstance(gold, dict) else {}
        p = pred if isinstance(pred, dict) else {}
        keys = set(g.keys()) | set(p.keys())
        for k in sorted(keys):
            child_prefix = f"{prefix}.{k}" if prefix else k
            if k in g and k in p:
                compare_leaves(g[k], p[k], child_prefix, out)
            elif k in g and k not in p:
                for _p, _v in flatten_leaves(g[k], child_prefix):
                    out["gold_leaves"] += 1
                    out["fn"] += 1
            else:  # k in p and k not in g
                for _p, _v in flatten_leaves(p[k], child_prefix):
                    out["pred_leaves"] += 1
                    out["fp"] += 1
        return out

    # Mixed primitive vs container at same path: count as both fp+fn
    if _is_primitive(gold) and not _is_primitive(pred):
        out["gold_leaves"] += 1
        out["fn"] += 1
    elif _is_primitive(pred) and not _is_primitive(gold):
        out["pred_leaves"] += 1
        out["fp"] += 1
    return out


def compute_field_recall(
    gold: Any, pred: Any, prefix: str = "", out: dict[str, int] | None = None
) -> dict[str, int]:
    """Per-leaf recovery: of all gold leaves, how many have a matching non-null,
    non-empty pred value at the same path?

    Two recovery modes are counted:
      * ``exact`` — pred value equals gold value exactly.
      * ``non_empty`` — pred value is present (not None, not ""), even if it
        differs from gold. This is a loose signal useful for free-text fields
        where exact match is too strict.

    Both are reported as fractions of the gold leaf count.
    """
    if out is None:
        out = {
            "gold_leaves": 0,
            "exact_match": 0,
            "non_empty_recovery": 0,
            "null_or_empty_in_gold": 0,
        }

    def _is_empty(v: Any) -> bool:
        if v is None:
            return True
        if isinstance(v, str) and v == "":
            return True
        return False

    if _is_primitive(gold) and _is_primitive(pred):
        out["gold_leaves"] += 1
        if _is_empty(gold):
            out["null_or_empty_in_gold"] += 1
        else:
            if gold == pred:
                out["exact_match"] += 1
            if not _is_empty(pred):
                out["non_empty_recovery"] += 1
        return out

    if isinstance(gold, list) or isinstance(pred, list):
        g = gold if isinstance(gold, list) else []
        p = pred if isinstance(pred, list) else []
        n = max(len(g), len(p))
        for i in range(n):
            child_prefix = f"{prefix}[{i}]" if prefix else f"[{i}]"
            gi = g[i] if i < len(g) else None
            pi = p[i] if i < len(p) else None
            compute_field_recall(gi, pi, child_prefix, out)
        return out

    if isinstance(gold, dict) or isinstance(pred, dict):
        g = gold if isinstance(gold, dict) else {}
        p = pred if isinstance(pred, dict) else {}
        keys = set(g.keys()) | set(p.keys())
        for k in sorted(keys):
            child_prefix = f"{prefix}.{k}" if prefix else k
            if k in g and k in p:
                compute_field_recall(g[k], p[k], child_prefix, out)
            elif k in g and k not in p:
                for _p, _v in flatten_leaves(g[k], child_prefix):
                    out["gold_leaves"] += 1
                    if _is_empty(_v):
                        out["null_or_empty_in_gold"] += 1
                    else:
                        out["fn"] = out.get("fn", 0) + 1
        return out

    return out


# =============================================================================
# Categorical metrics: path-flattened enum / bool / string fields
# =============================================================================
# A small whitelist of categorical event-level paths we care about. Each is a
# tuple of (logical_name, json_path). These are the schema-defined enum /
# string / boolean fields where exact-match accuracy is meaningful.
EVENT_CATEGORICAL_PATHS: list[tuple[str, str]] = [
    ("evento.es_evento_protesta", "extraccion.eventos_protesta[].es_evento_protesta"),
    ("delimitacion.criterio_delimitacion", "extraccion.eventos_protesta[].delimitacion_evento.criterio_delimitacion"),
    ("delimitacion.es_accion_principal_con_complementarias", "extraccion.eventos_protesta[].delimitacion_evento.es_accion_principal_con_complementarias"),
    ("temporalidad.tipo_temporal", "extraccion.eventos_protesta[].temporalidad.tipo_temporal"),
    ("temporalidad.tempo_verbal", "extraccion.eventos_protesta[].temporalidad.tempo_verbal"),
    ("temporalidad.fecha_inicio.certeza", "extraccion.eventos_protesta[].temporalidad.fecha_inicio.certeza"),
    ("temporalidad.fecha_fin.certeza", "extraccion.eventos_protesta[].temporalidad.fecha_fin.certeza"),
    ("accion.formato_principal.categoria", "extraccion.eventos_protesta[].accion.formato_principal.categoria"),
    ("sujetos[].categoria", "extraccion.eventos_protesta[].sujetos[].categoria"),
    ("sujetos[].organizaciones[].categoria", "extraccion.eventos_protesta[].sujetos[].organizaciones[].categoria"),
    ("demandas[].categoria", "extraccion.eventos_protesta[].demandas[].categoria"),
    ("contra_quien[].categoria", "extraccion.eventos_protesta[].contra_quien[].categoria"),
    ("contra_quien[].nivel_institucional", "extraccion.eventos_protesta[].contra_quien[].nivel_institucional"),
    ("lugares[].categoria", "extraccion.eventos_protesta[].lugares[].categoria"),
    ("lugares[].rol_en_evento", "extraccion.eventos_protesta[].lugares[].rol_en_evento"),
    ("alcance.categoria", "extraccion.eventos_protesta[].alcance.categoria"),
    ("cantidad_participantes.hay_cantidad_mencionada", "extraccion.eventos_protesta[].cantidad_participantes.hay_cantidad_mencionada"),
    ("cantidad_participantes.es_aproximada", "extraccion.eventos_protesta[].cantidad_participantes.es_aproximada"),
    ("incidentes.represion.presencia", "extraccion.eventos_protesta[].incidentes.represion.presencia"),
    ("incidentes.enfrentamiento.presencia", "extraccion.eventos_protesta[].incidentes.enfrentamiento.presencia"),
    ("incidentes.detenidos.presencia", "extraccion.eventos_protesta[].incidentes.detenidos.presencia"),
    ("incidentes.heridos.presencia", "extraccion.eventos_protesta[].incidentes.heridos.presencia"),
    ("incidentes.muertos.presencia", "extraccion.eventos_protesta[].incidentes.muertos.presencia"),
    ("incidentes.danios_materiales.presencia", "extraccion.eventos_protesta[].incidentes.danios_materiales.presencia"),
]


def _resolve_path(node: Any, path: str) -> list[Any]:
    """Resolve a path like ``eventos_protesta[].sujetos[].categoria`` against
    a parsed JSON object and return a flat list of all matching leaf values.
    ``[]`` matches every index of an array.
    """
    tokens = re.findall(r"[^.\[\]]+|\[\]", path)
    results: list[Any] = []

    def _walk(n: Any, toks: list[str]) -> None:
        if not toks:
            results.append(n)
            return
        t, rest = toks[0], toks[1:]
        if t == "[]":
            if isinstance(n, list):
                for item in n:
                    _walk(item, rest)
            # If n is not a list, we silently skip — schema_validity catches it.
            return
        if isinstance(n, dict) and t in n:
            _walk(n[t], rest)
        # else: path doesn't exist in this tree, skip

    _walk(node, tokens)
    return results


def categorical_accuracy(gold: Any, pred: Any) -> dict[str, Any]:
    """Headline categorical accuracy per path, then aggregated.

    Aggregation rule: a leaf comparison is "comparable" only when BOTH gold
    and pred have a leaf at the aligned index for that path. Arrays of
    events/subjects/etc. are aligned by index. If pred returns fewer items
    than gold, the extra gold items count as misses (no prediction → wrong).
    """
    per_path: dict[str, dict[str, int]] = {}
    # Headline boolean field: extraccion.tiene_eventos_protesta
    g_tiene = None
    p_tiene = None
    try:
        g_tiene = gold["extraccion"]["tiene_eventos_protesta"]
    except Exception:
        pass
    try:
        p_tiene = pred["extraccion"]["tiene_eventos_protesta"]
    except Exception:
        pass
    if g_tiene is not None and p_tiene is not None:
        per_path["extraccion.tiene_eventos_protesta"] = {
            "tp": int(g_tiene == p_tiene == True),
            "tn": int(g_tiene == p_tiene == False),
            "fp": int(g_tiene == False and p_tiene == True),
            "fn": int(g_tiene == True and p_tiene == False),
        }

    for name, path in EVENT_CATEGORICAL_PATHS:
        g_vals = _resolve_path(gold, path)
        p_vals = _resolve_path(pred, path)
        n = max(len(g_vals), len(p_vals))
        tp = tn = fp = fn = 0
        for i in range(n):
            gv = g_vals[i] if i < len(g_vals) else None
            pv = p_vals[i] if i < len(p_vals) else None
            if gv is None and pv is None:
                continue
            if gv is None or pv is None:
                # no comparable leaf at this index — only counts as a miss if
                # one side has it and the other doesn't
                if gv is not None:
                    fn += 1
                else:
                    fp += 1
                continue
            if gv == pv:
                if isinstance(gv, bool):
                    if gv is True:
                        tp += 1
                    else:
                        tn += 1
                else:
                    # Non-boolean categorical: treat exact match as a "positive"
                    tp += 1
            else:
                fp += 1
                fn += 1
        per_path[name] = {"tp": tp, "tn": tn, "fp": fp, "fn": fn}

    # Aggregate
    headline: dict[str, dict[str, int]] = {}
    total_tp = total_tn = total_fp = total_fn = 0
    for name, c in per_path.items():
        total_tp += c["tp"]
        total_tn += c["tn"]
        total_fp += c["fp"]
        total_fn += c["fn"]
        total = c["tp"] + c["tn"] + c["fp"] + c["fn"]
        accuracy = safe_div(c["tp"] + c["tn"], total)
        headline[name] = {
            "tp": c["tp"],
            "tn": c["tn"],
            "fp": c["fp"],
            "fn": c["fn"],
            "accuracy": round(accuracy, 4),
            "support": total,
        }

    headline["__aggregate__"] = {
        "tp": total_tp,
        "tn": total_tn,
        "fp": total_fp,
        "fn": total_fn,
        "accuracy": round(
            safe_div(total_tp + total_tn, total_tp + total_tn + total_fp + total_fn), 4
        ),
        "support": total_tp + total_tn + total_fp + total_fn,
    }
    return headline


# =============================================================================
# Run-classification guardrails (pure, no vLLM / torch / filesystem side-effects)
# =============================================================================
# These helpers encode the pass gates and path-collision prevention in pure
# form so they can be unit-tested without loading vLLM, the model, or
# touching the filesystem. main() composes them in order:
#
#   1. resolve_limited_run_paths() — auto-route limited runs away from official
#      artifact paths BEFORE any output side-effect or model load.
#   2. classify_run_status()        — compute the status string for the run.
#   3. merge_readiness_for_partial() — partial-write helper that NEVER
#      touches top-level status / status_note / full_baseline fields.
# Together they are the testable gate that protects Phase 3 from being
# unblocked by a partial / debug / smoke run.


def _partial_sibling(path: Path) -> Path:
    """Return a sibling path with `_partial` inserted before the suffix.

    Used to auto-route limited-run outputs away from the official full-baseline
    artifact paths so a `--limit` invocation cannot clobber the authoritative
    Phase 2 baseline metrics. The returned path lives next to the original
    (same parent directory, same suffix), making the rerouted artifact easy
    to identify and never collide with the official one.

    Examples:
        metrics/baseline_qwen2.5-7b.json         → metrics/baseline_qwen2.5-7b_partial.json
        metrics/baseline_qwen2.5-7b_outputs.jsonl → metrics/baseline_qwen2.5-7b_outputs_partial.jsonl
        metrics/qualitative_report.md            → metrics/qualitative_report_partial.md
    """
    return path.with_name(f"{path.stem}_partial{path.suffix}")


def resolve_limited_run_paths(
    *,
    limit: int | None,
    metrics: Path,
    outputs: Path,
    qualitative: Path,
    official_metrics: Path = OFFICIAL_METRICS_PATH,
    official_outputs: Path = OFFICIAL_OUTPUTS_PATH,
    official_qual: Path = OFFICIAL_QUAL_PATH,
) -> tuple[Path, Path, Path, list[str]]:
    """Guardrail for limited runs (--limit set): auto-route any output path
    that still points at the official full-baseline artifact location to a
    sibling `_partial` path. Returns ``(resolved_metrics, resolved_outputs,
    resolved_qualitative, warnings)``.

    For full runs (``limit is None``) the input paths are returned verbatim
    with no warnings — a full run is the only run that may legitimately
    write the authoritative Phase 2 baseline artifacts.

    Rationale: this is the safest minimal default. The user does not have
    to remember to pass `--metrics /tmp/x.json` for every debug run; the
    reroute happens automatically and a clear warning is printed to stderr.
    To opt out, pass an explicit custom path on the CLI.
    """
    if limit is None:
        return metrics, outputs, qualitative, []

    def _maybe_reroute(
        path: Path, official: Path, label: str
    ) -> tuple[Path, str | None]:
        if path == official:
            new = _partial_sibling(path)
            return new, (
                f"limited run (--limit={limit}) auto-routed {label} from "
                f"{path} to {new} so it cannot overwrite the official full "
                f"baseline artifact. Pass an explicit custom path on the CLI "
                f"to opt out."
            )
        return path, None

    new_metrics, w_m = _maybe_reroute(metrics, official_metrics, "--metrics")
    new_outputs, w_o = _maybe_reroute(outputs, official_outputs, "--outputs")
    new_qual, w_q = _maybe_reroute(qualitative, official_qual, "--qualitative")
    warnings = [w for w in (w_m, w_o, w_q) if w is not None]
    return new_metrics, new_outputs, new_qual, warnings


def classify_run_status(
    *,
    is_full_baseline: bool,
    examples_run: int,
    blocked_pre_inference: int,
    schema_valid_count: int,
    length_truncated_count: int,
    eval_total: int,
) -> dict[str, Any]:
    """Pure helper: classify the outcome of a baseline run.

    Pass rules:

      * **FULL pass** (``status="pass"``) requires ALL of:
          - ``examples_run == eval_total`` (every eval example was attempted)
          - ``blocked_pre_inference == 0`` (no prompt exceeded the model budget)
          - ``schema_valid_count > 0`` (at least one schema-valid output)
          - ``length_truncated_count == 0`` (no output hit finish_reason=length)
      * **LIMITED run** (``is_full_baseline=False``) ALWAYS produces a status
        with the ``partial_`` prefix, regardless of how clean the slice was.
        This is the only mechanism that keeps a ``--limit`` invocation from
        unblocking Phase 3: the status string itself is ``partial_*``, so any
        downstream gating that requires a top-level ``pass`` will reject it.

    Returns a dict with keys:

      * ``status``           — final status string ("pass" / "fail" /
                               "incomplete" / "pass_with_truncations" for
                               full; "partial_*" for limited).
      * ``is_full_baseline`` — echoes the input flag.
      * ``status_prefix``    — "" for full, "partial_" for limited.
      * ``full_coverage``    — True iff the run actually covered its target.
      * ``reason``           — human-readable explanation for fail /
                               incomplete / pass_with_truncations; "" for pass.
    """
    status_prefix = "" if is_full_baseline else "partial_"
    if is_full_baseline:
        # Full reach of "pass" requires every eval example to have been
        # attempted (no pre-inference blocks, no skipped rows).
        full_coverage = (examples_run == eval_total) and (blocked_pre_inference == 0)
    else:
        # Limited reach of "partial_pass" only requires the slice to have
        # been clean (no blocks). The status_prefix is what keeps the run
        # from being treated as a full baseline completion downstream.
        full_coverage = blocked_pre_inference == 0

    if schema_valid_count == 0:
        status = f"{status_prefix}fail"
        reason = (
            "Zero schema-valid outputs — Phase 2 baseline produced nothing "
            "usable. Inspect metrics/baseline_qwen2.5-7b_outputs.jsonl before "
            "starting Phase 3."
        )
    elif not full_coverage:
        status = f"{status_prefix}incomplete"
        if is_full_baseline:
            reason = (
                f"{blocked_pre_inference} example(s) blocked pre-inference "
                f"(prompt exceeds max_seq_length or computed max_tokens budget "
                f"is too small). Ran {examples_run}/{eval_total} examples — "
                f"full baseline did NOT cover the eval set. Inspect "
                f"blocked_pre_inference_examples in the metrics JSON and the "
                f"per-example records in the outputs JSONL."
            )
        else:
            reason = (
                f"{blocked_pre_inference} example(s) blocked pre-inference "
                f"in a limited run (--limit, target was "
                f"{examples_run + blocked_pre_inference} of {eval_total}). "
                f"Inspect blocked_pre_inference_examples in the metrics JSON."
            )
    elif length_truncated_count > 0:
        status = f"{status_prefix}pass_with_truncations"
        reason = (
            f"{length_truncated_count} example(s) hit finish_reason=length; "
            "the per-example max_tokens was insufficient for the "
            "schema-constrained output. Inspect "
            "metrics/baseline_qwen2.5-7b_outputs.jsonl."
        )
    else:
        status = f"{status_prefix}pass"
        reason = ""

    return {
        "status": status,
        "status_prefix": status_prefix,
        "is_full_baseline": is_full_baseline,
        "full_coverage": full_coverage,
        "reason": reason,
    }


def merge_readiness_for_partial(
    readiness: dict[str, Any], partial_run: dict[str, Any]
) -> dict[str, Any]:
    """Append a partial-run record to the readiness document.

    Contract (tested): this function NEVER touches top-level ``status``,
    ``status_note``, or ``full_baseline`` — those fields are reserved for the
    authoritative full-baseline write and must not be downgraded by a debug
    invocation. It only appends ``partial_run`` to ``partial_baseline_runs``
    (creating the array if absent) and refreshes the ``phase`` label.

    Returns a NEW dict (the input is not mutated), so callers can reason
    about the "did the merge change the document?" question by simple
    identity comparison.
    """
    out = dict(readiness)
    out["phase"] = "Phase 2 — Baseline Qwen2.5-7B-Instruct"
    runs = list(out.get("partial_baseline_runs", []))
    runs.append(dict(partial_run))
    out["partial_baseline_runs"] = runs
    out.setdefault("phase_1_gates", {})
    return out


# =============================================================================
# Main
# =============================================================================
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--eval-input", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--outputs", type=Path, default=DEFAULT_OUTPUTS)
    parser.add_argument("--qualitative", type=Path, default=DEFAULT_QUAL)
    parser.add_argument("--readiness", type=Path, default=DEFAULT_READINESS)
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument("--max-tokens-cap", type=int, default=DEFAULT_MAX_TOKENS_CAP)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
        help="vLLM gpu_memory_utilization. Lower = safer under OOM.",
    )
    parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use --enforce-eager to skip CUDA graph capture (sm_120 first-run safety).",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face cache dir for model/tokenizer files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Optional cap on number of examples (debug/smoke only). When set, the "
            "run is classified as LIMITED and the resulting status is ALWAYS "
            "prefixed with 'partial_' (e.g. partial_pass). Limited runs do NOT "
            "update reports/phase2_readiness.json as a full baseline completion, "
            "and so cannot unblock Phase 3. Pass --update-readiness to opt in to "
            "a clearly-labeled partial write (writes a 'partial_baseline' block "
            "only; never overwrites a previous full_baseline record). "
            "Additionally, limited runs AUTO-ROUTE the default official output "
            "paths (metrics/baseline_qwen2.5-7b.json, "
            "metrics/baseline_qwen2.5-7b_outputs.jsonl, "
            "metrics/qualitative_report.md) to sibling '_partial' paths so a "
            "debug invocation can never overwrite the authoritative Phase 2 "
            "baseline artifacts. Pass explicit --metrics / --outputs / "
            "--qualitative paths to opt out of the auto-route."
        ),
    )
    parser.add_argument(
        "--update-readiness",
        action="store_true",
        help=(
            "For limited runs (--limit set): opt in to writing a clearly-labeled "
            "partial status to reports/phase2_readiness.json. The partial write "
            "goes into a 'partial_baseline' block and never touches the "
            "top-level status / status_note / full_baseline fields, so a "
            "previous full-baseline record is preserved. For full runs (no "
            "--limit): this flag has no effect — full runs always update "
            "readiness as the authoritative full-baseline completion."
        ),
    )
    args = parser.parse_args()

    # CRITICAL: a limited run (--limit set) must NEVER be able to overwrite the
    # official full-baseline artifacts at metrics/baseline_qwen2.5-7b.json,
    # metrics/baseline_qwen2.5-7b_outputs.jsonl, or metrics/qualitative_report.md.
    # Auto-route any of those that still point at the official location to a
    # sibling `_partial` path. This check runs BEFORE any output side-effect
    # (the `unlink()` below) and BEFORE any vLLM/torch import — a debug run
    # that hits a missing model never even loads the network artifact paths.
    metrics_resolved, outputs_resolved, qual_resolved, path_warnings = (
        resolve_limited_run_paths(
            limit=args.limit,
            metrics=args.metrics,
            outputs=args.outputs,
            qualitative=args.qualitative,
        )
    )
    for w in path_warnings:
        print(f"[full] WARNING: {w}", file=sys.stderr)
    args.metrics = metrics_resolved
    args.outputs = outputs_resolved
    args.qualitative = qual_resolved

    # Always start outputs JSONL fresh
    args.outputs.parent.mkdir(parents=True, exist_ok=True)
    if args.outputs.exists():
        args.outputs.unlink()

    metrics_report: dict[str, Any] = {
        "phase": "Phase 2 — Baseline Qwen2.5-7B-Instruct (full)",
        "status": "blocked",
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": args.model,
        "eval_input": str(args.eval_input),
        "schema_path": str(args.schema),
        "max_seq_length": args.max_seq_length,
        "max_tokens_cap": args.max_tokens_cap,
        "env": {
            "VLLM_USE_FLASHINFER_SAMPLER": os.environ.get("VLLM_USE_FLASHINFER_SAMPLER"),
            "note": (
                "FlashInfer sampler path fails on vLLM 0.23.0 + sm_120 in this "
                "environment; the caller MUST set VLLM_USE_FLASHINFER_SAMPLER=0 "
                "before invoking this script."
            ),
        },
        "errors": [],
    }

    if not args.eval_input.exists():
        metrics_report["errors"].append(f"eval input not found: {args.eval_input}")
        write_json(args.metrics, metrics_report)
        return 2
    if not args.schema.exists():
        metrics_report["errors"].append(f"schema not found: {args.schema}")
        write_json(args.metrics, metrics_report)
        return 2

    rows = load_jsonl(args.eval_input)
    # CRITICAL: capture the original eval-file total BEFORE applying --limit.
    # The full-baseline status (pass / pass_with_truncations / incomplete / fail)
    # is gated on examples_run == eval_total; limited runs (--limit set) must
    # NEVER reach a top-level "pass" status and must NEVER update
    # reports/phase2_readiness.json as a full baseline completion.
    eval_total = len(rows)
    is_full_baseline = args.limit is None
    if args.limit:
        rows = rows[: args.limit]
    metrics_report["run_classification"] = {
        "is_full_baseline": is_full_baseline,
        "limit": args.limit,
        "eval_total": eval_total,
        "examples_total": len(rows),
        "update_readiness": bool(args.update_readiness) or is_full_baseline,
    }
    metrics_report["examples_total"] = len(rows)

    with args.schema.open("r", encoding="utf-8") as f:
        raw_schema = json.load(f)
    cleaned_schema = clean_schema_for_vllm(raw_schema)
    metrics_report["schema"] = {
        "path": str(args.schema),
        "title": raw_schema.get("title"),
        "required": raw_schema.get("required"),
        "additionalProperties_root": raw_schema.get("additionalProperties"),
    }

    # ---- Lazy vLLM / torch imports so failures land in the report ----
    try:
        import torch  # type: ignore
        import vllm  # type: ignore
        from vllm import LLM, SamplingParams  # type: ignore
        from vllm.sampling_params import StructuredOutputsParams  # type: ignore
    except Exception as exc:
        metrics_report["status"] = "blocked"
        metrics_report["errors"].append(f"vLLM/torch import failed: {exc}")
        write_json(args.metrics, metrics_report)
        return 2

    metrics_report["vllm_version"] = vllm.__version__
    metrics_report["torch_version"] = torch.__version__
    metrics_report["cuda_runtime"] = torch.version.cuda
    metrics_report["cuda_available"] = bool(torch.cuda.is_available())
    if torch.cuda.is_available():
        metrics_report["gpu"] = {
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "vram_total_mib": int(torch.cuda.get_device_properties(0).total_memory)
            // (1024 * 1024),
        }
    else:
        metrics_report["status"] = "blocked"
        metrics_report["errors"].append("torch.cuda.is_available() is False")
        write_json(args.metrics, metrics_report)
        return 2

    metrics_report["structured_outputs_api"] = (
        "SamplingParams(structured_outputs=StructuredOutputsParams(json=cleaned_schema))"
    )

    # ---- Build the LLM ----
    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "max_model_len": args.max_seq_length,
        "max_num_seqs": 1,
        "dtype": "auto",
        "enforce_eager": args.enforce_eager,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": False,
    }
    if args.cache_dir:
        llm_kwargs["cache_dir"] = args.cache_dir
    metrics_report["llm_kwargs"] = llm_kwargs

    try:
        print(f"[full] Loading model: {args.model}", file=sys.stderr)
        llm = LLM(**llm_kwargs)
    except Exception as exc:
        metrics_report["status"] = "blocked"
        metrics_report["errors"].append(f"LLM load failed: {exc}")
        metrics_report["errors"].append(traceback.format_exc())
        write_json(args.metrics, metrics_report)
        return 2

    tokenizer = llm.get_tokenizer()

    # ---- Prepare per-example messages + per-example sampling params ----
    prepared: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    for idx, item in enumerate(rows):
        eval_row = item["row"]
        nota_id = derive_nota_id(eval_row, fallback=f"line_{item['line_no']}")
        chat_messages = [
            {"role": m["role"], "content": m["content"]}
            for m in eval_row["messages"]
            if m.get("role") in {"system", "user"}
        ]
        try:
            prompt_text = tokenizer.apply_chat_template(
                chat_messages, tokenize=False, add_generation_prompt=True
            )
        except Exception as exc:
            blocked.append(
                {
                    "index": idx,
                    "line_no": item["line_no"],
                    "nota_id": nota_id,
                    "reason": f"apply_chat_template failed: {exc}",
                }
            )
            continue
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_tokens = len(prompt_ids)
        if prompt_tokens >= args.max_seq_length:
            blocked.append(
                {
                    "index": idx,
                    "line_no": item["line_no"],
                    "nota_id": nota_id,
                    "reason": f"prompt_tokens={prompt_tokens} >= max_seq_length={args.max_seq_length}",
                    "prompt_tokens": prompt_tokens,
                }
            )
            continue
        max_tokens = min(
            args.max_tokens_cap,
            max(0, args.max_seq_length - prompt_tokens - DEFAULT_PROMPT_SAFETY_MARGIN),
        )
        if max_tokens < DEFAULT_MIN_OUTPUT_BUDGET:
            blocked.append(
                {
                    "index": idx,
                    "line_no": item["line_no"],
                    "nota_id": nota_id,
                    "reason": (
                        f"computed max_tokens={max_tokens} < {DEFAULT_MIN_OUTPUT_BUDGET} "
                        f"(prompt_tokens={prompt_tokens}, max_seq_length={args.max_seq_length})"
                    ),
                    "prompt_tokens": prompt_tokens,
                    "max_tokens": max_tokens,
                }
            )
            continue
        prepared.append(
            {
                "index": idx,
                "line_no": item["line_no"],
                "nota_id": nota_id,
                "chat_messages": chat_messages,
                "prompt_text": prompt_text,
                "prompt_tokens": prompt_tokens,
                "max_tokens": max_tokens,
            }
        )

    metrics_report["prepared"] = len(prepared)
    metrics_report["blocked_pre_inference"] = len(blocked)
    if blocked:
        metrics_report["blocked_pre_inference_examples"] = blocked

    if not prepared:
        metrics_report["status"] = "blocked"
        metrics_report["errors"].append("no examples fit the prompt budget; nothing to run")
        write_json(args.metrics, metrics_report)
        return 2

    # ---- Run inference in one llm.chat call with per-example params ----
    sp_list = [
        SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=p["max_tokens"],
            structured_outputs=StructuredOutputsParams(json=cleaned_schema),
        )
        for p in prepared
    ]
    chat_inputs = [p["chat_messages"] for p in prepared]

    print(
        f"[full] Running {len(chat_inputs)} examples via llm.chat() with "
        f"VLLM_USE_FLASHINFER_SAMPLER={os.environ.get('VLLM_USE_FLASHINFER_SAMPLER')}",
        file=sys.stderr,
    )
    started = time.time()
    try:
        outputs = llm.chat(
            chat_inputs,
            sampling_params=sp_list,
            use_tqdm=False,
            add_generation_prompt=True,
        )
    except Exception as exc:
        metrics_report["status"] = "blocked"
        metrics_report["errors"].append(f"vllm.chat failed: {exc}")
        metrics_report["errors"].append(traceback.format_exc())
        write_json(args.metrics, metrics_report)
        return 2
    total_elapsed = time.time() - started

    if not outputs or len(outputs) != len(prepared):
        metrics_report["status"] = "fail"
        metrics_report["errors"].append(
            f"vllm.chat returned {len(outputs) if outputs else 0} outputs, expected {len(prepared)}"
        )
        write_json(args.metrics, metrics_report)
        return 2

    # ---- Per-example record + comparison vs gold ----
    per_example_records: list[dict[str, Any]] = []
    f1_aggregate = {"tp": 0, "fp": 0, "fn": 0, "gold_leaves": 0, "pred_leaves": 0}
    field_recall_aggregate = {
        "gold_leaves": 0,
        "exact_match": 0,
        "non_empty_recovery": 0,
        "null_or_empty_in_gold": 0,
    }
    categorical_aggregate: dict[str, dict[str, int]] = {}
    schema_valid_count = 0
    parse_valid_count = 0
    length_truncated_count = 0

    for prep, out in zip(prepared, outputs):
        idx = prep["index"]
        eval_row = rows[idx]["row"]
        gold_assistant = next(
            (m for m in eval_row["messages"] if m.get("role") == "assistant"), None
        )
        gold_obj: Any = None
        if gold_assistant is not None:
            try:
                gold_obj = json.loads(gold_assistant["content"])
            except Exception:
                gold_obj = None

        out0 = out.outputs[0] if out.outputs else None
        raw_text = out0.text if out0 else ""
        finish_reason = out0.finish_reason if out0 else None
        out_tokens = len(out0.token_ids) if out0 and out0.token_ids else 0
        prompt_tokens_runtime = len(out.prompt_token_ids) if out.prompt_token_ids else None
        if finish_reason == "length":
            length_truncated_count += 1

        parsed: Any = None
        parse_error: str | None = None
        try:
            parsed = json.loads(raw_text)
        except Exception as exc:
            parse_error = f"{type(exc).__name__}: {exc}"
        if parse_error is None:
            parse_valid_count += 1

        # Validate against the raw MVS schema (not the cleaned one) so we
        # catch violations of $schema constraints (const, pattern, etc.).
        validation = (
            validate_against_schema(parsed, raw_schema)
            if parsed is not None
            else {"available": True, "valid": False, "error": "skipped because raw output did not parse"}
        )
        if validation.get("valid"):
            schema_valid_count += 1

        # F1 / field_recall / categorical only if schema valid and gold parseable
        f1_this = {"tp": 0, "fp": 0, "fn": 0, "gold_leaves": 0, "pred_leaves": 0}
        fr_this = {
            "gold_leaves": 0,
            "exact_match": 0,
            "non_empty_recovery": 0,
            "null_or_empty_in_gold": 0,
        }
        cat_this: dict[str, Any] = {}
        if validation.get("valid") and gold_obj is not None:
            compare_leaves(gold_obj, parsed, "", f1_this)
            compute_field_recall(gold_obj, parsed, "", fr_this)
            cat_this = categorical_accuracy(gold_obj, parsed)
            for k, c in cat_this.items():
                if k == "__aggregate__":
                    continue
                if k not in categorical_aggregate:
                    categorical_aggregate[k] = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
                categorical_aggregate[k]["tp"] += c.get("tp", 0)
                categorical_aggregate[k]["tn"] += c.get("tn", 0)
                categorical_aggregate[k]["fp"] += c.get("fp", 0)
                categorical_aggregate[k]["fn"] += c.get("fn", 0)

        for k in f1_aggregate:
            f1_aggregate[k] += f1_this[k]
        for k in field_recall_aggregate:
            field_recall_aggregate[k] += fr_this[k]

        record = {
            "index": idx,
            "line_no": prep["line_no"],
            "nota_id": prep["nota_id"],
            "prompt_tokens": prep["prompt_tokens"],
            "max_tokens": prep["max_tokens"],
            "output_tokens": out_tokens,
            "finish_reason": finish_reason,
            "elapsed_seconds": None,  # filled below, BEFORE the JSONL is written
            "parse_valid": parse_error is None,
            "parse_error": parse_error,
            "schema_valid": bool(validation.get("valid")),
            "schema_error_count": validation.get("error_count", 0),
            "schema_errors": validation.get("errors", []),
            "f1_vs_gold": f1_this,
            "field_recall_vs_gold": fr_this,
            "categorical_accuracy_vs_gold": cat_this,
            "raw_text": raw_text,
            "parsed": parsed,
        }
        per_example_records.append(record)
        # NOTE: do not append to args.outputs here. elapsed_seconds is
        # filled in below (proportional token attribution), and writing now
        # would leave the JSONL permanently with elapsed_seconds=null.
        # The JSONL is written once after the attribution block.

    # Compute per-example elapsed by attributing wall time proportionally to
    # prompt + output tokens. This is a coarse approximation, but it lets
    # us report per-example elapsed_seconds in the records.
    total_tokens = sum(
        (r["prompt_tokens"] or 0) + (r["output_tokens"] or 0) for r in per_example_records
    )
    if total_tokens > 0:
        for r in per_example_records:
            tok = (r["prompt_tokens"] or 0) + (r["output_tokens"] or 0)
            r["elapsed_seconds"] = round(total_elapsed * (tok / total_tokens), 3)
    else:
        for r in per_example_records:
            r["elapsed_seconds"] = 0.0

    # Now that elapsed_seconds is populated for every record, write the
    # JSONL in one pass. Order matches the iteration order above
    # (sorted by input row index), so downstream tooling can index it
    # the same way it indexes per_example in the metrics JSON.
    for record in per_example_records:
        append_jsonl(args.outputs, record)

    # ---- Aggregate metrics ----
    n = len(per_example_records)
    schema_validity = round(safe_div(schema_valid_count, n), 4)
    parse_validity = round(safe_div(parse_valid_count, n), 4)

    f1_metrics = micro_f1_from_counts(
        f1_aggregate["tp"], f1_aggregate["fp"], f1_aggregate["fn"]
    )

    gr = field_recall_aggregate["gold_leaves"] or 0
    fr_exact = round(safe_div(field_recall_aggregate["exact_match"], gr), 4)
    fr_non_empty = round(safe_div(field_recall_aggregate["non_empty_recovery"], gr), 4)
    field_recall = {
        "gold_leaves": gr,
        "exact_match_count": field_recall_aggregate["exact_match"],
        "exact_match_recall": fr_exact,
        "non_empty_recovery_count": field_recall_aggregate["non_empty_recovery"],
        "non_empty_recovery_recall": fr_non_empty,
        "null_or_empty_in_gold": field_recall_aggregate["null_or_empty_in_gold"],
        "comparable_leaves": gr - field_recall_aggregate["null_or_empty_in_gold"],
    }

    cat_headline: dict[str, dict[str, Any]] = {}
    cat_tp = cat_tn = cat_fp = cat_fn = 0
    for name, c in categorical_aggregate.items():
        support = c["tp"] + c["tn"] + c["fp"] + c["fn"]
        acc = safe_div(c["tp"] + c["tn"], support)
        cat_headline[name] = {
            "tp": c["tp"],
            "tn": c["tn"],
            "fp": c["fp"],
            "fn": c["fn"],
            "accuracy": round(acc, 4),
            "support": support,
        }
        cat_tp += c["tp"]
        cat_tn += c["tn"]
        cat_fp += c["fp"]
        cat_fn += c["fn"]
    cat_support = cat_tp + cat_tn + cat_fp + cat_fn
    cat_headline["__aggregate__"] = {
        "tp": cat_tp,
        "tn": cat_tn,
        "fp": cat_fp,
        "fn": cat_fn,
        "accuracy": round(safe_div(cat_tp + cat_tn, cat_support), 4),
        "support": cat_support,
    }
    # The single most-misleading headline is `extraccion.tiene_eventos_protesta`
    # (a boolean). Promote it explicitly so the report leads with it.
    tiene = cat_headline.get("extraccion.tiene_eventos_protesta", {})
    categorical_accuracy_headline = {
        "tiene_eventos_protesta": tiene,
        "per_path": cat_headline,
    }

    # ---- Token / timing stats ----
    prompt_tokens_list = [r["prompt_tokens"] for r in per_example_records]
    output_tokens_list = [r["output_tokens"] for r in per_example_records]
    timings = {
        "total_seconds": round(total_elapsed, 3),
        "mean_per_example_seconds": round(total_elapsed / max(1, n), 3),
        "prompt_tokens_min": min(prompt_tokens_list) if prompt_tokens_list else 0,
        "prompt_tokens_max": max(prompt_tokens_list) if prompt_tokens_list else 0,
        "prompt_tokens_mean": round(statistics.mean(prompt_tokens_list), 1) if prompt_tokens_list else 0,
        "output_tokens_min": min(output_tokens_list) if output_tokens_list else 0,
        "output_tokens_max": max(output_tokens_list) if output_tokens_list else 0,
        "output_tokens_mean": round(statistics.mean(output_tokens_list), 1) if output_tokens_list else 0,
        "output_tokens_total": sum(output_tokens_list),
    }

    # ---- Final report ----
    metrics_report["counts"] = {
        "examples_total": n + len(blocked),
        "examples_run": n,
        "examples_blocked_pre_inference": len(blocked),
        "parse_valid": parse_valid_count,
        "schema_valid": schema_valid_count,
        "finish_reason_length": length_truncated_count,
    }
    metrics_report["metrics"] = {
        "schema_validity": schema_validity,
        "parse_validity": parse_validity,
        "categorical_accuracy": categorical_accuracy_headline,
        "f1_global": f1_metrics,
        "field_recall": field_recall,
    }
    metrics_report["timings"] = timings
    metrics_report["per_example"] = per_example_records

    # Pass requires the FULL eval set to have been processed: at least one
    # schema-valid output AND no pre-inference blocks (every input row was
    # either prepared or blocked — we know the count of both — so
    # ``examples_run == eval_total``) AND no length-truncated outputs.
    # Limited runs (--limit set) can NEVER reach a top-level "pass"; they get
    # a "partial_*" prefix so they cannot unblock Phase 3 even if all slice
    # examples are clean.
    # Quality lives in metrics; the status only guards against incomplete
    # or failed runs so a future run with fewer than all eval examples
    # cannot silently report ``pass``.
    classification = classify_run_status(
        is_full_baseline=is_full_baseline,
        examples_run=n,
        blocked_pre_inference=len(blocked),
        schema_valid_count=schema_valid_count,
        length_truncated_count=length_truncated_count,
        eval_total=eval_total,
    )
    metrics_report["status"] = classification["status"]
    if classification["reason"]:
        metrics_report["errors"].append(classification["reason"])

    write_json(args.metrics, metrics_report)

    # ---- Update phase2_readiness.json (gated on full baseline or explicit opt-in) ----
    # CRITICAL: a limited run (--limit set) must NEVER update readiness as a full
    # baseline completion. The default behavior is therefore to skip the
    # readiness write entirely when --limit is set. Pass --update-readiness to
    # explicitly opt in to writing a clearly-labeled partial record; even then
    # the write goes into a `partial_baseline` block and never overwrites
    # `status` / `status_note` / `full_baseline` from a previous full run, so
    # a real full baseline is never silently downgraded by a debug invocation.
    should_update_readiness = is_full_baseline or bool(args.update_readiness)
    if not should_update_readiness:
        print(
            f"[full] Limited run (--limit={args.limit}) — skipping readiness "
            f"update at {args.readiness}. Use --update-readiness to opt in to "
            f"a clearly-labeled partial write (still does not overwrite "
            f"full_baseline / status).",
            file=sys.stderr,
        )
    else:
        try:
            readiness: dict[str, Any] = {}
            if args.readiness.exists():
                readiness = json.loads(args.readiness.read_text(encoding="utf-8"))

            if is_full_baseline:
                # Authoritative full-baseline write. This is the only path that
                # may set the top-level status / status_note / full_baseline
                # fields, and so is the only path that can unblock Phase 3.
                readiness["phase"] = "Phase 2 — Baseline Qwen2.5-7B-Instruct"
                readiness["status"] = metrics_report["status"]
                readiness["status_note"] = (
                    f"Full {eval_total}-example Phase 2 baseline completed with "
                    f"status={metrics_report['status']}. See "
                    f"metrics/baseline_qwen2.5-7b.json for schema_validity="
                    f"{schema_validity}, f1_global={f1_metrics['f1']}, "
                    f"field_recall.exact={fr_exact}, and per-path "
                    f"categorical_accuracy. The full baseline was run with "
                    f"VLLM_USE_FLASHINFER_SAMPLER="
                    f"{os.environ.get('VLLM_USE_FLASHINFER_SAMPLER')}. Phase 3 "
                    f"(SFT/QLoRA) was NOT started — it is gated on this "
                    f"baseline existing."
                )
                readiness["checked_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                readiness["plan_reference"] = "PLAN_ENTRENAMIENTO_QWEN.md §Fase 2"
                readiness["environment"] = {
                    "python": "3.12.12 (cpython, via uv-managed .venv)",
                    "torch": torch.__version__,
                    "transformers": "4.57.6",
                    "vllm": vllm.__version__,
                    "tokenizer_audit_model": args.model,
                }
                readiness["hardware"] = {
                    "gpu": torch.cuda.get_device_name(0),
                    "vram_mib": int(torch.cuda.get_device_properties(0).total_memory)
                    // (1024 * 1024),
                    "driver": "580.159.03",
                    "cuda_version": torch.version.cuda,
                    "compute_capability": "12.0 (Blackwell consumer / sm_120)",
                    "vllm_installed": True,
                }
                readiness["full_baseline"] = {
                    "report": str(args.metrics),
                    "raw_outputs": str(args.outputs),
                    "qualitative_report": str(args.qualitative),
                    "examples_total": n + len(blocked),
                    "examples_run": n,
                    "examples_blocked_pre_inference": len(blocked),
                    "parse_valid": parse_valid_count,
                    "schema_valid": schema_valid_count,
                    "schema_validity": schema_validity,
                    "categorical_accuracy": cat_headline.get("__aggregate__", {}).get("accuracy"),
                    "tiene_eventos_protesta_accuracy": tiene.get("accuracy"),
                    "f1_global": f1_metrics["f1"],
                    "field_recall_exact": fr_exact,
                    "field_recall_non_empty": fr_non_empty,
                    "timings": timings,
                    "env": {
                        "VLLM_USE_FLASHINFER_SAMPLER": os.environ.get("VLLM_USE_FLASHINFER_SAMPLER"),
                    },
                }
                readiness["remaining_work"] = [
                    "Phase 3 (SFT/QLoRA on Qwen2.5-7B-Instruct) is now unblocked from a data-eval "
                    "perspective; should be started only after a human reviews the qualitative report "
                    "and confirms the baseline numbers as a credible reference point."
                ]
                readiness.setdefault("phase_1_gates", {})
                readiness["files_created_or_modified_by_this_run"] = [
                    "scripts/baseline_qwen_full.py (new — Phase 2 full baseline runner)",
                    f"{args.metrics} (new — full baseline machine-readable metrics)",
                    f"{args.outputs} (new — raw per-example generations + parse/schema results)",
                    f"{args.qualitative} (new — qualitative report on failure patterns)",
                    f"{args.readiness} (updated — full baseline status)",
                ]
            else:
                # Limited run with --update-readiness: write a clearly-labeled
                # partial record ONLY. Never touch top-level status / status_note
                # / full_baseline — a previous full baseline record, if any, is
                # preserved verbatim. The pure helper merge_readiness_for_partial
                # enforces that contract.
                partial_run = {
                    "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "limit": args.limit,
                    "eval_total": eval_total,
                    "examples_run": n,
                    "examples_blocked_pre_inference": len(blocked),
                    "parse_valid": parse_valid_count,
                    "schema_valid": schema_valid_count,
                    "schema_validity": schema_validity,
                    "categorical_accuracy": cat_headline.get("__aggregate__", {}).get("accuracy"),
                    "tiene_eventos_protesta_accuracy": tiene.get("accuracy"),
                    "f1_global": f1_metrics["f1"],
                    "field_recall_exact": fr_exact,
                    "field_recall_non_empty": fr_non_empty,
                    "status": metrics_report["status"],
                    "status_note": (
                        f"PARTIAL run (--limit={args.limit}): processed "
                        f"{n}/{eval_total} eval examples with "
                        f"status={metrics_report['status']}. This run is "
                        f"NOT a full baseline and does NOT unblock "
                        f"Phase 3. Inspect "
                        f"metrics/baseline_qwen2.5-7b.json for details."
                    ),
                    "report": str(args.metrics),
                    "raw_outputs": str(args.outputs),
                    "env": {
                        "VLLM_USE_FLASHINFER_SAMPLER": os.environ.get(
                            "VLLM_USE_FLASHINFER_SAMPLER"
                        ),
                    },
                }
                readiness = merge_readiness_for_partial(readiness, partial_run)
            write_json(args.readiness, readiness)
        except Exception as exc:
            print(f"[full] WARNING: failed to update {args.readiness}: {exc}", file=sys.stderr)

    # ---- Console summary ----
    run_tag = "full" if is_full_baseline else f"limit={args.limit}"
    print(
        f"[full] {metrics_report['status'].upper()} ({run_tag}) n={n}/{eval_total} "
        f"schema_validity={schema_validity} f1_global={f1_metrics['f1']} "
        f"field_recall.exact={fr_exact} "
        f"cat_agg_acc={cat_headline['__aggregate__']['accuracy']} "
        f"tiene_acc={tiene.get('accuracy')} -> {args.metrics}",
        file=sys.stderr,
    )
    # Exit code 0 only on a clean full-baseline pass / pass_with_truncations.
    # Limited runs always exit non-zero so they cannot be mistaken for a
    # successful full baseline by a wrapping shell / orchestrator.
    if is_full_baseline and metrics_report["status"] in {"pass", "pass_with_truncations"}:
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
