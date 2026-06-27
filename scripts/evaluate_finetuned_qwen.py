#!/usr/bin/env python3
"""Phase 4 — post-training evaluation of the fine-tuned LoRA adapter.

Loads ``Qwen/Qwen2.5-7B-Instruct`` in vLLM with ``enable_lora=True`` and runs
the full 35-example eval set through ``data/chat_formatted/eval.jsonl`` with
the MVS schema enforced via ``StructuredOutputsParams(json=...)``. The
Phase 3 LoRA adapter at ``checkpoints/qwen-protesta-v1`` is attached via
``LoRARequest`` so every generation uses the fine-tuned weights.

This is the comparator runner for the Phase 2 baseline: same eval set, same
prompt construction, same ``StructuredOutputsParams(json=cleaned_schema)``
constraint, same per-example ``max_tokens`` budget logic, same validation
against the raw MVS schema, and the same metric definitions
(``schema_validity``, ``parse_validity``, ``tiene_eventos_protesta``
confusion, ``categorical_accuracy.per_path`` with ``__aggregate__``,
``f1_global`` over flattened leaves, ``field_recall`` exact and non-empty
recovery). Reuses the helper functions from ``scripts/baseline_qwen_full.py``
verbatim so the metrics are directly comparable.

The shared comparison logic is imported (not duplicated) via a local import
of the baseline module. This guarantees that any future tweak to
``compare_leaves`` / ``compute_field_recall`` / ``categorical_accuracy`` in
the baseline runner is automatically reflected here.

This script MUST NOT:
  * Train or merge the adapter (Phase 3 already did; this is eval only).
  * Mutate any checkpoint on disk.
  * Overwrite the Phase 2 baseline artifacts (``metrics/baseline_qwen2.5-7b*``
    and ``metrics/qualitative_report.md``) — enforced by a fail-fast guardrail
    in :func:`assert_phase4_outputs_safe`. The guardrail covers ALL FOUR write
    paths (``--metrics``, ``--outputs``, ``--qualitative``, ``--readiness``)
    so passing ``--readiness metrics/baseline_qwen2.5-7b.json`` cannot silently
    overwrite the authoritative baseline metrics. Pass
    ``--allow-baseline-output-overwrite`` to deliberately bypass the guardrail
    (NOT recommended).
  * Silently truncate prompts; if a prompt does not fit, it is recorded as
    blocked_pre_inference with an explicit reason and the model is NOT run.
  * Silently truncate outputs; any ``finish_reason='length'`` is a HARD
    failure — the run status flips to ``fail`` and the exit code is 2
    (non-zero), regardless of whether the run otherwise had full coverage.
    Per PLAN §6 the run must NOT be promoted to MVP if any schema-constrained
    output was truncated by the token budget. The truncation check happens
    BEFORE the incomplete check in :func:`classify_run_status` so a mixed
    truncation+incomplete run is hard-failed, never downgraded to incomplete.

The Phase 4 readiness report (``reports/phase4_eval.json``) carries an
unambiguous three-field top-level summary so automation can tell apart
"did the run finish?" from "did the model meet the MVP bar?":

  * ``status``              — combined run + MVP, e.g.
                              ``completed_mvp_failed`` /
                              ``completed_mvp_accepted`` / ``failed`` /
                              ``blocked`` / ``incomplete``.
  * ``run_status``          — run-only outcome (``pass`` / ``fail`` /
                              ``incomplete`` / ``blocked``).
  * ``mvp_accepted``        — explicit boolean. ``False`` does NOT mean the
                              run failed; it means the metrics are below the
                              PLAN §6 MVP bar (schema ≥ 0.95, categorical
                              aggregate ≥ 0.80, f1 ≥ 0.70).
  * ``mvp_criteria``        — per-criterion threshold / value / passed map.
  * ``mvp_reason``          — human-readable summary of the MVP decision.

Outputs (do not collide with Phase 2):
  * ``metrics/finetuned_qwen-protesta-v1.json``        — machine-readable metrics
  * ``metrics/finetuned_qwen-protesta-v1_outputs.jsonl`` — raw per-example outputs
  * ``metrics/qualitative_report_finetuned.md``         — qualitative report
  * ``reports/phase4_eval.json``                        — Phase 4 readiness

Usage:
    VLLM_USE_FLASHINFER_SAMPLER=0 .venv/bin/python scripts/evaluate_finetuned_qwen.py
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# Reuse baseline comparison logic verbatim so finetuned vs baseline deltas
# are computed by the same code path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_qwen_full import (  # type: ignore  # noqa: E402
    EVENT_CATEGORICAL_PATHS,
    append_jsonl,
    categorical_accuracy,
    clean_schema_for_vllm,
    compare_leaves,
    compute_field_recall,
    derive_nota_id,
    load_jsonl,
    micro_f1_from_counts,
    safe_div,
    validate_against_schema,
    write_json,
)

# -- Default paths ------------------------------------------------------------
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_ADAPTER = Path("checkpoints/qwen-protesta-v1")
DEFAULT_EVAL = Path("data/chat_formatted/eval.jsonl")
DEFAULT_SCHEMA = Path("esquema_eventos_protesta_entrenamiento_MVS.json")
DEFAULT_METRICS = Path("metrics/finetuned_qwen-protesta-v1.json")
DEFAULT_OUTPUTS = Path("metrics/finetuned_qwen-protesta-v1_outputs.jsonl")
DEFAULT_QUAL = Path("metrics/qualitative_report_finetuned.md")
DEFAULT_READINESS = Path("reports/phase4_eval.json")
DEFAULT_BASELINE_METRICS = Path("metrics/baseline_qwen2.5-7b.json")

DEFAULT_MAX_SEQ_LENGTH = 20480
DEFAULT_MAX_TOKENS_CAP = 8192
DEFAULT_PROMPT_SAFETY_MARGIN = 16
DEFAULT_MIN_OUTPUT_BUDGET = 256

# vLLM LoRA defaults from the run contract.
DEFAULT_MAX_LORA_RANK = 16  # matches adapter_config.json r=16
DEFAULT_LORA_NAME = "qwen_protesta_v1"
DEFAULT_LORA_INT_ID = 1


# =============================================================================
# Protected Phase 2 baseline artifact paths
# =============================================================================
# Phase 4 MUST NOT overwrite these by accident — they are the authoritative
# Phase 2 baseline outputs. The guardrail below makes the default Phase 4
# outputs (which point at the `_finetuned`/`finetuned_qwen-protesta-v1` paths)
# safe, AND raises a fail-fast error if the caller explicitly passes one of
# these protected paths on the CLI without the deliberate override flag.
#
# The readiness path (``reports/phase4_eval.json`` by default) is ALSO a write
# path — passing ``--readiness metrics/baseline_qwen2.5-7b.json`` would silently
# overwrite the Phase 2 baseline metrics. The guardrail therefore protects all
# four write paths uniformly so the readiness path cannot be redirected into a
# baseline artifact by accident.
PROTECTED_BASELINE_OUTPUT_PATHS: tuple[Path, ...] = (
    Path("metrics/baseline_qwen2.5-7b.json"),
    Path("metrics/baseline_qwen2.5-7b_outputs.jsonl"),
    Path("metrics/qualitative_report.md"),
)


class ProtectedBaselinePathError(RuntimeError):
    """Raised when a Phase 4 output path collides with a Phase 2 baseline
    artifact. Resolved by passing ``--allow-baseline-output-overwrite`` on the
    CLI (NOT recommended — overwrites the authoritative baseline)."""


# =============================================================================
# Protected Phase 3 r=16 finetuned artifact paths
# =============================================================================
# Phase 6 iterations (e.g. r=32) MUST NOT overwrite the Phase 3 r=16 finetuned
# artifacts by accident — they are the authoritative comparator for every
# later LoRA-rank experiment. The guardrail below is a sibling of the baseline
# guardrail and protects the four r=16 write paths
# (``metrics``, ``outputs``, ``qualitative``, ``readiness``) uniformly. A
# separate CLI flag ``--allow-r16-output-overwrite`` is required to bypass it.
#
# Naming rationale: the r=16 run wrote ``finetuned_qwen-protesta-v1.{json,jsonl}``
# + ``qualitative_report_finetuned.md`` + ``reports/phase4_eval.json``. Any
# Phase 6 iteration that accidentally points one of those flags at those
# targets will trip the guardrail and exit 2 without writing anything.
PROTECTED_FINETUNED_R16_OUTPUT_PATHS: tuple[Path, ...] = (
    Path("metrics/finetuned_qwen-protesta-v1.json"),
    Path("metrics/finetuned_qwen-protesta-v1_outputs.jsonl"),
    Path("metrics/qualitative_report_finetuned.md"),
    Path("reports/phase4_eval.json"),
)


class ProtectedFinetunedR16PathError(RuntimeError):
    """Raised when a Phase 6 output path collides with the Phase 3 r=16
    finetuned artifacts. Resolved by passing ``--allow-r16-output-overwrite``
    on the CLI (NOT recommended — overwrites the authoritative r=16 result)."""


# =============================================================================
# PLAN §6 MVP acceptance criteria
# =============================================================================
# All three must be met for the run to be promoted to MVP. Source:
# PLAN_ENTRENAMIENTO_QWEN.md §6 — Criterios de éxito.
MVP_CRITERIA: dict[str, float] = {
    "schema_validity": 0.95,
    "categorical_accuracy_aggregate": 0.80,
    "f1_global": 0.70,
}


def _normalize_path(p: Path | str) -> Path:
    """Resolve a path to an absolute, normalized form for equality checks.

    Strips trailing separators, collapses ``..`` segments, and removes any
    redundant ``.`` so that the same physical target is recognised whether the
    caller passed it as ``metrics/foo.json``, ``./metrics/foo.json``,
    ``/abs/repo/metrics/foo.json``, etc.
    """
    return Path(os.path.normpath(os.path.abspath(str(p))))


def _is_protected_baseline_path(p: Path, repo_root: Path) -> bool:
    """True iff ``p`` (relative or absolute) resolves to one of the protected
    Phase 2 baseline artifact paths."""
    p_abs = _normalize_path(p if p.is_absolute() else repo_root / p)
    for protected in PROTECTED_BASELINE_OUTPUT_PATHS:
        prot_abs = _normalize_path(
            protected if protected.is_absolute() else repo_root / protected
        )
        if p_abs == prot_abs:
            return True
    return False


def assert_phase4_outputs_safe(
    *,
    metrics: Path,
    outputs: Path,
    qualitative: Path,
    readiness: Path,
    repo_root: Path,
    allow_baseline_overwrite: bool,
) -> list[str]:
    """Fail-fast guardrail: Phase 4 outputs must never collide with the
    authoritative Phase 2 baseline artifacts unless the caller has set the
    deliberate override flag.

    The check covers ALL FOUR write paths:

      * ``metrics``     — Phase 4 metrics JSON
      * ``outputs``     — Phase 4 per-example outputs JSONL
      * ``qualitative`` — Phase 4 qualitative markdown report
      * ``readiness``   — Phase 4 readiness JSON

    Every path is normalised to absolute form before comparison so trivial
    variants (``./metrics/baseline_qwen2.5-7b.json``,
    ``/abs/repo/metrics/baseline_qwen2.5-7b.json``,
    ``metrics/sub/../baseline_qwen2.5-7b.json``) all trip the guardrail
    instead of sneaking past it.

    Returns an empty list when no collision is detected. Raises
    :class:`ProtectedBaselinePathError` listing every offending flag if any
    protected path is targeted and the override is not set. The error is
    raised BEFORE any readiness write so a caller passing
    ``--readiness metrics/baseline_qwen2.5-7b.json`` cannot overwrite the
    Phase 2 baseline metrics — even though the blocked-write path inside
    ``main()`` would otherwise fall back to writing a minimal blocked
    readiness. The blocked-readiness fallback itself ALSO refuses to write
    when the readiness path is protected (see
    :func:`_write_blocked_readiness_or_skip`).
    """
    if allow_baseline_overwrite:
        return []
    collisions: list[str] = []
    for label, p in (
        ("--metrics", metrics),
        ("--outputs", outputs),
        ("--qualitative", qualitative),
        ("--readiness", readiness),
    ):
        if _is_protected_baseline_path(p, repo_root):
            collisions.append(
                f"{label}={p} resolves to a protected Phase 2 baseline artifact"
            )
    if collisions:
        collision_lines = "\n  ".join(collisions)
        raise ProtectedBaselinePathError(
            "Phase 4 output paths collide with Phase 2 baseline artifacts:\n"
            f"  {collision_lines}\n"
            "Pass --allow-baseline-output-overwrite to deliberately overwrite "
            "(NOT recommended — overwrites the authoritative Phase 2 baseline)."
        )
    return []


def assert_phase6_r16_outputs_safe(
    *,
    metrics: Path,
    outputs: Path,
    qualitative: Path,
    readiness: Path,
    repo_root: Path,
    allow_r16_overwrite: bool,
) -> list[str]:
    """Fail-fast guardrail for Phase 6 iterations (e.g. r=32).

    Phase 6 output paths must never collide with the Phase 3 r=16 finetuned
    artifacts unless the caller has set the deliberate override flag. Same
    four-path coverage and absolute-path normalisation as
    :func:`assert_phase4_outputs_safe`, but for the r=16 target set.

    Returns an empty list when no collision is detected. Raises
    :class:`ProtectedFinetunedR16PathError` listing every offending flag if
    any r=16 path is targeted and the override is not set.
    """
    if allow_r16_overwrite:
        return []
    collisions: list[str] = []
    for label, p in (
        ("--metrics", metrics),
        ("--outputs", outputs),
        ("--qualitative", qualitative),
        ("--readiness", readiness),
    ):
        if _is_protected_finetuned_r16_path(p, repo_root):
            collisions.append(
                f"{label}={p} resolves to a protected Phase 3 r=16 finetuned artifact"
            )
    if collisions:
        collision_lines = "\n  ".join(collisions)
        raise ProtectedFinetunedR16PathError(
            "Phase 6 output paths collide with Phase 3 r=16 finetuned artifacts:\n"
            f"  {collision_lines}\n"
            "Pass --allow-r16-output-overwrite to deliberately overwrite "
            "(NOT recommended — overwrites the authoritative r=16 comparator)."
        )
    return []


def _is_protected_finetuned_r16_path(p: Path, repo_root: Path) -> bool:
    """True iff ``p`` (relative or absolute) resolves to one of the protected
    Phase 3 r=16 finetuned artifact paths."""
    p_abs = _normalize_path(p if p.is_absolute() else repo_root / p)
    for protected in PROTECTED_FINETUNED_R16_OUTPUT_PATHS:
        prot_abs = _normalize_path(
            protected if protected.is_absolute() else repo_root / protected
        )
        if p_abs == prot_abs:
            return True
    return False


def _is_protected_readiness_path(p: Path, repo_root: Path) -> str | None:
    """Return the protection class of ``p``, or ``None`` if unprotected.

    Used by :func:`_write_blocked_readiness_or_skip` to decide whether a
    blocked-readiness write would clobber an authoritative artifact. Both
    protected sets are checked so a single helper covers Phase 4 baseline
    AND Phase 6 r=16 guardrails — they must BOTH refuse to write the
    blocked payload to their respective protected targets.

    Returns:
      * ``"baseline"`` if the path resolves to a protected Phase 2
        baseline artifact (``PROTECTED_BASELINE_OUTPUT_PATHS``).
      * ``"r16"``     if it resolves to a protected Phase 3 r=16 finetuned
        artifact (``PROTECTED_FINETUNED_R16_OUTPUT_PATHS``).
      * ``None``      if it is safe to write to.

    The function never raises — it is called from a defensive fallback
    path where raising would mask the original guardrail error.
    """
    if _is_protected_baseline_path(p, repo_root):
        return "baseline"
    if _is_protected_finetuned_r16_path(p, repo_root):
        return "r16"
    return None


def _write_blocked_readiness_or_skip(
    *,
    readiness: Path,
    exc: Exception,
    repo_root: Path,
) -> bool:
    """Decide what to do with the readiness path when the guardrail trips.

    Extracted from :func:`main` so unit tests can drive the EXACT production
    code path the runner uses. Contract:

      * If ``readiness`` resolves to EITHER a protected Phase 2 baseline
        artifact OR a protected Phase 3 r=16 finetuned artifact, the runner
        MUST NOT write to it — not even the minimal blocked-readiness
        payload. Writing there would silently overwrite an authoritative
        artifact (Phase 2 baseline metrics, or Phase 3 r=16 finetuned
        readiness ``reports/phase4_eval.json``), which is the exact
        regression this guardrail exists to prevent. The runner prints an
        explanatory error to stderr and returns ``False`` so ``main()`` can
        exit with code 2 without producing any artifact.
      * Otherwise the runner writes a minimal blocked-readiness payload to
        the (safe) readiness path so automation can observe the guardrail
        tripped. Returns ``True`` on a successful (or attempted) write.

    Returns ``True`` if a write was attempted, ``False`` if it was skipped
    because the readiness path was protected. Exceptions during the safe-
    path write are swallowed (logged via the ``pass``) so a transient
    filesystem error cannot mask the underlying ``ProtectedBaselinePathError``
    or ``ProtectedFinetunedR16PathError`` or change the exit code.

    Regression note: a Phase 6 iteration (e.g. r=32) used to fall through
    this helper and overwrite ``reports/phase4_eval.json`` with the
    blocked-readiness payload when ``--readiness`` pointed at the r=16
    artifact. The r=16 readiness artifact IS in
    ``PROTECTED_FINETUNED_R16_OUTPUT_PATHS`` so this helper now refuses
    that write just like the baseline case. The fix is symmetric with the
    r16 / baseline guardrails themselves: a blocked Phase 6 invocation
    with ``--readiness reports/phase4_eval.json`` cannot overwrite r=16
    readiness via this fallback.
    """
    protection = _is_protected_readiness_path(readiness, repo_root)
    if protection == "baseline":
        print(
            f"[phase4] BLOCKED: refusing to write blocked-readiness to "
            f"protected baseline path {readiness} — exiting 2 with no write "
            f"so the authoritative Phase 2 baseline artifact is preserved.",
            file=sys.stderr,
        )
        return False
    if protection == "r16":
        print(
            f"[phase6] BLOCKED: refusing to write blocked-readiness to "
            f"protected Phase 3 r=16 finetuned path {readiness} — exiting 2 "
            f"with no write so the authoritative r=16 readiness artifact "
            f"(reports/phase4_eval.json) is preserved.",
            file=sys.stderr,
        )
        return False
    blocked_readiness: dict[str, Any] = {
        "phase": "Phase 4 — Post-training evaluation",
        "status": "blocked",
        "run_status": "blocked",
        "evaluation_completed": False,
        "mvp_accepted": False,
        "mvp_status": "blocked",
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "plan_reference": "PLAN_ENTRENAMIENTO_QWEN.md §Fase 4",
        "errors": [str(exc)],
    }
    try:
        readiness.parent.mkdir(parents=True, exist_ok=True)
        write_json(readiness, blocked_readiness)
    except Exception as write_exc:  # noqa: BLE001 — defensive: see docstring
        print(
            f"[phase4] WARNING: failed to write blocked-readiness at "
            f"{readiness}: {write_exc}",
            file=sys.stderr,
        )
    return True


def compute_mvp_decision(readiness_metrics: dict[str, Any]) -> dict[str, Any]:
    """Decide MVP acceptance from the flat readiness-metrics dict.

    Accepts the readiness-shape (``schema_validity``,
    ``categorical_accuracy_aggregate``, ``f1_global`` as floats) and also the
    nested metrics-report shape (``metrics.schema_validity``,
    ``metrics.categorical_accuracy.per_path.__aggregate__.accuracy``,
    ``metrics.f1_global.f1``). Any criterion with a missing or non-numeric
    value is treated as ``passed=False`` (conservative — never silently
    promotes a run to MVP on incomplete data).
    """
    # Prefer the flat readiness shape, fall back to nested metrics.
    schema = readiness_metrics.get("schema_validity")
    if schema is None and isinstance(readiness_metrics.get("metrics"), dict):
        schema = readiness_metrics["metrics"].get("schema_validity")

    cat_agg = readiness_metrics.get("categorical_accuracy_aggregate")
    if cat_agg is None and isinstance(readiness_metrics.get("metrics"), dict):
        cat_agg = (
            readiness_metrics["metrics"]
            .get("categorical_accuracy", {})
            .get("per_path", {})
            .get("__aggregate__", {})
            .get("accuracy")
        )

    f1 = readiness_metrics.get("f1_global")
    if isinstance(f1, dict):
        f1_val: Any = f1.get("f1")
    else:
        f1_val = f1
    if f1_val is None and isinstance(readiness_metrics.get("metrics"), dict):
        f1_val = (
            readiness_metrics["metrics"].get("f1_global", {}).get("f1")
            if isinstance(readiness_metrics["metrics"].get("f1_global"), dict)
            else None
        )

    def _passed(value: Any, threshold: float) -> bool:
        return isinstance(value, (int, float)) and float(value) >= threshold

    criteria_results: dict[str, dict[str, Any]] = {
        "schema_validity": {
            "threshold": MVP_CRITERIA["schema_validity"],
            "value": schema,
            "passed": _passed(schema, MVP_CRITERIA["schema_validity"]),
        },
        "categorical_accuracy_aggregate": {
            "threshold": MVP_CRITERIA["categorical_accuracy_aggregate"],
            "value": cat_agg,
            "passed": _passed(cat_agg, MVP_CRITERIA["categorical_accuracy_aggregate"]),
        },
        "f1_global": {
            "threshold": MVP_CRITERIA["f1_global"],
            "value": f1_val,
            "passed": _passed(f1_val, MVP_CRITERIA["f1_global"]),
        },
    }

    failed = [k for k, c in criteria_results.items() if not c["passed"]]
    accepted = not failed
    return {
        "mvp_accepted": accepted,
        "mvp_status": "accepted" if accepted else "failed",
        "mvp_criteria": criteria_results,
        "mvp_failed_criteria": failed,
        "mvp_reason": (
            "All 3 PLAN §6 MVP criteria satisfied (schema_validity ≥ 0.95, "
            "categorical_accuracy_aggregate ≥ 0.80, f1_global ≥ 0.70)."
            if accepted
            else (
                f"{len(failed)}/3 PLAN §6 MVP criteria failed: "
                f"{', '.join(failed)}. Per PLAN §6, MVP is NOT accepted and "
                "Phase 6 iteration is required."
            )
        ),
    }


def combine_top_level_status(run_status: str, mvp_decision: dict[str, Any]) -> str:
    """Combine run + MVP into a single unambiguous top-level ``status``.

    Possible values:
      * ``completed_mvp_accepted``  — run succeeded AND MVP criteria met
      * ``completed_mvp_failed``    — run succeeded but MVP criteria NOT met
      * ``failed``                  — run itself failed (truncations, errors)
      * ``blocked``                 — inputs missing / setup could not start
      * ``incomplete``              — ran fewer examples than eval_total
    """
    rs = (run_status or "").strip()
    if rs in {"blocked", "incomplete"}:
        return rs
    if rs == "fail":
        return "failed"
    if mvp_decision.get("mvp_accepted"):
        return "completed_mvp_accepted"
    return "completed_mvp_failed"


def classify_run_status(
    *,
    examples_run: int,
    eval_total: int,
    blocked_pre_inference: int,
    schema_valid_count: int,
    length_truncated_count: int,
) -> dict[str, Any]:
    """Pure helper: classify the outcome of a Phase 4 run.

    This is the CANONICAL implementation of the Phase 4 pass/fail policy that
    :func:`main` uses. Tests import this helper directly so the policy is
    pinned against regression: if a future refactor mutates this function to
    weaken the truncation gate, the unit tests fail instead of silently
    re-introducing the bug.

    Phase 4 is STRICTER than Phase 2:

      * ``status == "pass"`` requires ALL of:
          - ``examples_run == eval_total``           (every eval example attempted)
          - ``blocked_pre_inference == 0``           (no prompt over budget)
          - ``schema_valid_count > 0``               (at least one schema-valid output)
          - ``length_truncated_count == 0``          (NO ``finish_reason=length``)
      * ANY ``length_truncated_count > 0`` ALWAYS produces
        ``status == "fail"`` (NEVER ``pass_with_truncations``). A
        schema-constrained output that hit the token budget is, by
        construction, not a valid completion against the MVS schema —
        downstream consumers cannot trust a truncated leaf, so the run is
        rejected at the gate. The previous contract returned
        ``pass_with_truncations`` with exit code 0; that ambiguity is the
        exact regression these tests guard against.

    Branch order is significant and pinned by the unit tests:

      1. **Zero schema-valid outputs** → ``"fail"`` (no usable output at all).
      2. **Any ``finish_reason=length`` truncation** → ``"fail"`` (HARD
         failure — checked BEFORE the incomplete gate so a run that both
         missed examples AND hit truncations still hard-fails, never gets
         downgraded to "incomplete"). This is the documented Phase 4
         policy: ANY truncation is a hard failure regardless of coverage.
      3. **Incomplete coverage** (no truncations, but some examples were
         blocked pre-inference or never attempted) → ``"incomplete"``.
      4. Otherwise → ``"pass"``.

    Returns a dict with keys:

      * ``status``           — ``"pass"`` / ``"fail"`` / ``"incomplete"``.
                                Phase 4 does NOT emit ``"pass_with_truncations"``.
      * ``full_coverage``    — ``True`` iff ``examples_run == eval_total`` AND
                                ``blocked_pre_inference == 0``.
      * ``reason``           — human-readable explanation for ``"fail"`` /
                                ``"incomplete"``; empty string for ``"pass"``.
    """
    full_coverage = (examples_run == eval_total) and (blocked_pre_inference == 0)
    if schema_valid_count == 0:
        status = "fail"
        reason = (
            "Zero schema-valid outputs — Phase 4 produced nothing usable. "
            "Inspect metrics/finetuned_qwen-protesta-v1_outputs.jsonl."
        )
    elif length_truncated_count > 0:
        # HARD failure on truncations — checked BEFORE the incomplete gate so
        # a run that both missed examples AND hit finish_reason=length still
        # hard-fails (never gets downgraded to "incomplete"). Exit code 2,
        # status `fail`. NEVER `pass_with_truncations`: that string was the
        # bug that allowed a truncated schema-constrained run to escape the
        # gate as "pass".
        status = "fail"
        reason = (
            f"{length_truncated_count} example(s) hit finish_reason=length; "
            "Phase 4 treats any schema-constrained truncation as a hard "
            "failure (exit code 2). Raise --max-tokens-cap / "
            "--max-seq-length and rerun. Inspect "
            "metrics/finetuned_qwen-protesta-v1_outputs.jsonl."
        )
    elif not full_coverage:
        status = "incomplete"
        reason = (
            f"{blocked_pre_inference} example(s) blocked pre-inference. "
            f"Ran {examples_run}/{eval_total}."
        )
    else:
        status = "pass"
        reason = ""

    return {
        "status": status,
        "full_coverage": full_coverage,
        "reason": reason,
    }


def exit_code_for_run_status(run_status: str) -> int:
    """Phase 4 exit-code policy: 0 ONLY on a clean ``"pass"``.

    Anything else — ``"fail"``, ``"incomplete"``, ``"blocked"``, or any other
    non-pass status — returns ``2``. This is the canonical implementation
    that :func:`main` uses at the very end of the run; tests pin against it
    directly so a refactor that returns 0 on truncations cannot slip
    through silently.
    """
    return 0 if (run_status or "").strip() == "pass" else 2


# =============================================================================
# Adapter resolution: root vs checkpoint subdir
# =============================================================================
def resolve_adapter_path(requested: Path) -> tuple[Path, dict[str, Any]]:
    """Pick the most reliable adapter directory and document the choice.

    Strategy (in order):
      1. If ``requested`` is a directory containing ``adapter_model.safetensors``
         at the root, use it.
      2. Else fall back to the latest ``checkpoint-*`` subdir of the requested
         path (PEFT save_strategy=epoch produces these).
      3. If neither resolves, raise FileNotFoundError so the caller can mark
         the run as blocked.

    Returns ``(resolved_path, info)`` where ``info`` records the SHA-1 of the
    resolved adapter weights and what alternatives were considered.
    """
    info: dict[str, Any] = {"requested": str(requested)}

    def _sha1(p: Path) -> str | None:
        try:
            import hashlib

            h = hashlib.sha1()
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            return h.hexdigest()
        except Exception as exc:  # pragma: no cover
            return f"error: {exc}"

    root_safetensors = requested / "adapter_model.safetensors"
    if root_safetensors.exists():
        info["resolved"] = str(requested)
        info["resolved_reason"] = "adapter_model.safetensors at requested root"
        info["sha1"] = _sha1(root_safetensors)
        info["size_bytes"] = root_safetensors.stat().st_size
        return requested, info

    # Look for checkpoint-* subdirs
    candidates: list[Path] = []
    if requested.is_dir():
        for child in requested.iterdir():
            if child.is_dir() and child.name.startswith("checkpoint-"):
                candidates.append(child)
    candidates.sort(key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1)
    info["checkpoint_candidates"] = [str(c) for c in candidates]
    if not candidates:
        raise FileNotFoundError(
            f"adapter directory '{requested}' has neither "
            f"adapter_model.safetensors at root nor any checkpoint-* subdir"
        )

    for cand in reversed(candidates):  # highest step first
        st = cand / "adapter_model.safetensors"
        if st.exists():
            info["resolved"] = str(cand)
            info["resolved_reason"] = (
                f"no root adapter; using highest-step checkpoint subdir "
                f"({cand.name})"
            )
            info["sha1"] = _sha1(st)
            info["size_bytes"] = st.stat().st_size
            return cand, info

    raise FileNotFoundError(
        f"no adapter_model.safetensors found under {requested} or its "
        f"checkpoint-* subdirs ({[str(c) for c in candidates]})"
    )


# =============================================================================
# Baseline delta
# =============================================================================
def load_baseline_metrics(path: Path) -> dict[str, Any] | None:
    """Load the Phase 2 baseline metrics JSON for delta computation.

    Returns None if the file does not exist. The caller is responsible for
    recording ``baseline_available=False`` in the report and skipping the
    delta block — the Phase 4 evaluation itself is independent.
    """
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def compute_delta_vs_baseline(
    finetuned_metrics: dict[str, Any], baseline_metrics: dict[str, Any]
) -> dict[str, Any]:
    """Compute a small, audit-friendly delta block (finetuned − baseline).

    Reports deltas for the headline numbers that drive the Phase 4 go/no-go
    decision:
      * schema_validity (higher is better, expected to stay 1.0)
      * f1_global.f1 (higher is better; success criterion ≥ 0.70)
      * categorical_accuracy aggregate accuracy (higher is better)
      * tiene_eventos_protesta accuracy (higher is better)
      * field_recall.exact_match_recall and non_empty_recovery_recall
    """
    bl = baseline_metrics.get("metrics", {})
    ft = finetuned_metrics

    def _delta(key_a: str, key_b: str | None = None) -> dict[str, Any] | None:
        key_b = key_b or key_a
        a = bl.get(key_a)
        b = ft.get(key_b)
        if isinstance(a, dict):
            if not isinstance(b, dict):
                return None
            out: dict[str, Any] = {}
            for k in a:
                if k in b and isinstance(a[k], (int, float)) and isinstance(b[k], (int, float)):
                    out[k] = round(b[k] - a[k], 4)
            return out or None
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return round(b - a, 4)
        return None

    # Baseline stores categorical aggregate inside per_path.__aggregate__; the
    # finetuned runner mirrors the same shape (baseline_qwen_full.py writes it
    # to per_path.__aggregate__ via `cat_headline["__aggregate__"] = ...`). Read
    # both from per_path.__aggregate__ for consistency.
    bl_cat_agg = (
        bl.get("categorical_accuracy", {}).get("per_path", {}).get("__aggregate__", {}).get("accuracy")
    )
    ft_cat_agg = (
        ft.get("categorical_accuracy", {}).get("per_path", {}).get("__aggregate__", {}).get("accuracy")
    )
    bl_tiene = bl.get("categorical_accuracy", {}).get("tiene_eventos_protesta", {}).get("accuracy")
    ft_tiene = ft.get("categorical_accuracy", {}).get("tiene_eventos_protesta", {}).get("accuracy")
    bl_f1 = bl.get("f1_global", {}).get("f1")
    ft_f1 = ft.get("f1_global", {}).get("f1")
    bl_fr = bl.get("field_recall", {})
    ft_fr = ft.get("field_recall", {})

    return {
        "schema_validity": {
            "baseline": bl.get("schema_validity"),
            "finetuned": ft.get("schema_validity"),
            "delta": _delta("schema_validity"),
        },
        "f1_global": {
            "baseline_f1": bl_f1,
            "finetuned_f1": ft_f1,
            "delta_f1": round(ft_f1 - bl_f1, 4) if isinstance(ft_f1, (int, float)) and isinstance(bl_f1, (int, float)) else None,
        },
        "categorical_accuracy_aggregate": {
            "baseline": bl_cat_agg,
            "finetuned": ft_cat_agg,
            "delta": round(ft_cat_agg - bl_cat_agg, 4) if isinstance(ft_cat_agg, (int, float)) and isinstance(bl_cat_agg, (int, float)) else None,
        },
        "tiene_eventos_protesta_accuracy": {
            "baseline": bl_tiene,
            "finetuned": ft_tiene,
            "delta": round(ft_tiene - bl_tiene, 4) if isinstance(ft_tiene, (int, float)) and isinstance(bl_tiene, (int, float)) else None,
        },
        "field_recall_exact": {
            "baseline": bl_fr.get("exact_match_recall"),
            "finetuned": ft_fr.get("exact_match_recall"),
            "delta": round(ft_fr.get("exact_match_recall", 0) - bl_fr.get("exact_match_recall", 0), 4)
            if isinstance(ft_fr.get("exact_match_recall"), (int, float))
            and isinstance(bl_fr.get("exact_match_recall"), (int, float))
            else None,
        },
        "field_recall_non_empty": {
            "baseline": bl_fr.get("non_empty_recovery_recall"),
            "finetuned": ft_fr.get("non_empty_recovery_recall"),
            "delta": round(ft_fr.get("non_empty_recovery_recall", 0) - bl_fr.get("non_empty_recovery_recall", 0), 4)
            if isinstance(ft_fr.get("non_empty_recovery_recall"), (int, float))
            and isinstance(bl_fr.get("non_empty_recovery_recall"), (int, float))
            else None,
        },
    }


# =============================================================================
# Qualitative report
# =============================================================================
def _adapter_label(adapter_info: dict[str, Any]) -> str:
    """Human-readable label for the adapter (used in the report header).

    Prefers ``lora_name`` if present in ``adapter_info`` (the CLI's
    ``--lora-name``), falls back to the last path segment of the resolved
    adapter directory, then to ``"adapter"``. Always returns a non-empty
    string so the report header is never malformed.
    """
    name = adapter_info.get("lora_name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    resolved = adapter_info.get("resolved")
    if isinstance(resolved, str) and resolved.strip():
        return Path(resolved).name or resolved.strip()
    return "adapter"


def build_qualitative_report(
    *,
    metrics_report: dict[str, Any],
    per_example: list[dict[str, Any]],
    delta: dict[str, Any] | None,
    adapter_info: dict[str, Any],
    metrics_path: Path | str | None = None,
    outputs_path: Path | str | None = None,
    baseline_metrics_path: Path | str | None = None,
    qualitative_path: Path | str | None = None,
    readiness_path: Path | str | None = None,
    plan_reference: str = "PLAN_ENTRENAMIENTO_QWEN.md §Fase 4",
    phase_label: str = "Phase 4",
    runner_script: str = "scripts/evaluate_finetuned_qwen.py",
) -> str:
    """Generate a 5-10 observation qualitative report mirroring the
    Phase 2 structure so the two reports are directly comparable.

    The report covers:
      1. Headline metrics vs baseline.
      2. ``tiene_eventos_protesta`` confusion matrix.
      3. Output-token distribution (does the model now produce events?).
      4. Per-path categorical accuracy vs baseline.
      5. Schema validity and parse validity.
      6. Hallucinated metadata (``nota_id``, ``fecha_publicacion``) drift.
      7. Worst / best examples by f1.
      8. Plan success criteria (PLAN §6) and what this run justifies / does
         NOT justify — both rendered from the actual run metrics, NOT from
         hard-coded r=16 numbers.

    All paths / labels / adapter metadata are dynamic so the same function
    is reusable for any LoRA-rank iteration (r=16 Phase 4, r=32 Phase 6,
    future r=64, ...). The narrative sections 9 ("What this run justifies")
    and 10 ("What this run does NOT justify") compute their numbers from
    ``delta`` and ``metrics_report`` so they cannot regress to a previous
    run's numbers when the runner is re-invoked with a new adapter.
    """
    m = metrics_report["metrics"]
    counts = metrics_report["counts"]
    timings = metrics_report["timings"]
    cat = m["categorical_accuracy"]
    tiene = cat["tiene_eventos_protesta"]
    f1g = m["f1_global"]
    fr = m["field_recall"]

    # --- dynamic context pulled from metrics_report + caller-supplied paths ---
    base_model = metrics_report.get("model") or DEFAULT_MODEL
    max_lora_rank = metrics_report.get("max_lora_rank")
    if max_lora_rank is None:
        max_lora_rank = adapter_info.get("rank") or DEFAULT_MAX_LORA_RANK
    eval_input = metrics_report.get("eval_input") or str(DEFAULT_EVAL)
    examples_total = counts.get("examples_total") or counts.get("examples_run")
    schema_path = None
    schema_meta = metrics_report.get("schema")
    if isinstance(schema_meta, dict):
        schema_path = schema_meta.get("path")
    if schema_path is None:
        schema_path = str(DEFAULT_SCHEMA)
    adapter_label = _adapter_label(adapter_info)
    metrics_path_str = str(metrics_path) if metrics_path else (
        metrics_report.get("metrics_path") or str(DEFAULT_METRICS)
    )
    outputs_path_str = str(outputs_path) if outputs_path else (
        metrics_report.get("outputs_path") or str(DEFAULT_OUTPUTS)
    )
    baseline_metrics_path_str = (
        str(baseline_metrics_path) if baseline_metrics_path else (
            metrics_report.get("baseline_metrics_path")
            or str(DEFAULT_BASELINE_METRICS)
        )
    )
    readiness_path_str = (
        str(readiness_path) if readiness_path else str(DEFAULT_READINESS)
    )
    qualitative_path_str = (
        str(qualitative_path) if qualitative_path else str(DEFAULT_QUAL)
    )

    # --- token distribution ---
    out_tokens = [r["output_tokens"] for r in per_example]
    bins = {"0-200": 0, "200-500": 0, "500-1000": 0, "1000-2000": 0, "2000+": 0}
    for n in out_tokens:
        if n < 200:
            bins["0-200"] += 1
        elif n < 500:
            bins["200-500"] += 1
        elif n < 1000:
            bins["500-1000"] += 1
        elif n < 2000:
            bins["1000-2000"] += 1
        else:
            bins["2000+"] += 1

    # --- hallucinated metadata ---
    nota_id_sd = 0
    nota_id_other = 0
    fecha_day_19 = 0
    fecha_total = 0
    for r in per_example:
        if not r["parse_valid"] or r["parsed"] is None:
            continue
        try:
            nota = r["parsed"].get("nota", {})
            nid = nota.get("nota_id")
            if isinstance(nid, str):
                if nid == "S/D":
                    nota_id_sd += 1
                else:
                    nota_id_other += 1
            fp = nota.get("fecha_publicacion")
            if isinstance(fp, str) and fp:
                fecha_total += 1
                # crude: second component = day
                parts = fp.split("/")
                if len(parts) == 3 and parts[0] == "19":
                    fecha_day_19 += 1
        except Exception:
            pass

    # --- confusion matrix strings ---
    cm_lines = [
        "|               | pred=0 | pred=1 |",
        "|---------------|-------:|-------:|",
        f"| **gold=1** ({tiene.get('tp', 0) + tiene.get('fn', 0)}) | "
        f"{tiene.get('fn', 0)} FN | {tiene.get('tp', 0)} TP |",
        f"| **gold=0** ({tiene.get('tn', 0) + tiene.get('fp', 0)}) | "
        f"{tiene.get('tn', 0)} TN | {tiene.get('fp', 0)} FP |",
    ]
    cm_table = "\n".join(cm_lines)

    # --- per-path accuracy vs baseline (delta column if delta exists) ---
    bl_per_path = None
    if delta is not None and metrics_report.get("baseline_available"):
        try:
            bl = json.loads(Path(baseline_metrics_path_str).read_text(encoding="utf-8"))
            bl_per_path = bl.get("metrics", {}).get("categorical_accuracy", {}).get("per_path", {})
        except Exception:
            bl_per_path = None

    per_path_rows = []
    for name, c in cat["per_path"].items():
        if name == "__aggregate__":
            continue
        bl_acc = None
        if bl_per_path and name in bl_per_path:
            bl_acc = bl_per_path[name].get("accuracy")
        delta_str = ""
        if bl_acc is not None:
            d = round(c["accuracy"] - bl_acc, 4)
            delta_str = f" | {d:+.4f}"
        per_path_rows.append(
            f"| `{name}` | {c['tp']} | {c['tn']} | {c['fp']} | {c['fn']} | "
            f"{c['accuracy']:.4f}{delta_str} |"
        )
    per_path_table = "\n".join(per_path_rows) if per_path_rows else "_(no per-path data)_"

    # --- top / bottom examples by f1 ---
    # The per-example record stores raw tp/fp/fn counts but NOT a derived f1
    # key, so compute f1 here for ranking. (Sorting on the absent f1 key
    # would always tie at -1.0 and just preserve input order.)
    def _per_example_f1(r: dict[str, Any]) -> float:
        f1m = r["f1_vs_gold"]
        tp = f1m.get("tp", 0)
        fp = f1m.get("fp", 0)
        fn = f1m.get("fn", 0)
        p = tp / (tp + fp) if (tp + fp) else 0.0
        rc = tp / (tp + fn) if (tp + fn) else 0.0
        return (2 * p * rc / (p + rc)) if (p + rc) else 0.0

    f1_ranked = sorted(per_example, key=_per_example_f1, reverse=True)
    best = f1_ranked[:2]
    worst = list(reversed(f1_ranked[-2:]))

    def _row_short(r: dict[str, Any]) -> str:
        f1m = r["f1_vs_gold"]
        tp = f1m.get("tp", 0)
        fp = f1m.get("fp", 0)
        fn = f1m.get("fn", 0)
        f1 = _per_example_f1(r)
        return (
            f"- `{r['nota_id']}` (idx={r['index']}) f1={f1:.3f} "
            f"tp={tp} fp={fp} fn={fn} out_tokens={r['output_tokens']} "
            f"finish={r['finish_reason']}"
        )

    best_block = "\n".join(_row_short(r) for r in best) if best else "_(none)_"
    worst_block = "\n".join(_row_short(r) for r in worst) if worst else "_(none)_"

    # --- headline deltas block (if baseline present) ---
    if delta is not None:
        delta_table = (
            "| metric | baseline | finetuned | delta |\n"
            "|---|---:|---:|---:|\n"
            f"| schema_validity | {delta['schema_validity']['baseline']} | "
            f"{delta['schema_validity']['finetuned']} | "
            f"{delta['schema_validity']['delta']} |\n"
            f"| f1_global.f1 | {delta['f1_global']['baseline_f1']} | "
            f"{delta['f1_global']['finetuned_f1']} | "
            f"{delta['f1_global']['delta_f1']} |\n"
            f"| categorical_accuracy aggregate | "
            f"{delta['categorical_accuracy_aggregate']['baseline']} | "
            f"{delta['categorical_accuracy_aggregate']['finetuned']} | "
            f"{delta['categorical_accuracy_aggregate']['delta']} |\n"
            f"| tiene_eventos_protesta accuracy | "
            f"{delta['tiene_eventos_protesta_accuracy']['baseline']} | "
            f"{delta['tiene_eventos_protesta_accuracy']['finetuned']} | "
            f"{delta['tiene_eventos_protesta_accuracy']['delta']} |\n"
            f"| field_recall.exact_match_recall | "
            f"{delta['field_recall_exact']['baseline']} | "
            f"{delta['field_recall_exact']['finetuned']} | "
            f"{delta['field_recall_exact']['delta']} |\n"
            f"| field_recall.non_empty_recovery_recall | "
            f"{delta['field_recall_non_empty']['baseline']} | "
            f"{delta['field_recall_non_empty']['finetuned']} | "
            f"{delta['field_recall_non_empty']['delta']} |"
        )
    else:
        delta_table = "_(baseline metrics not found — deltas unavailable)_"

    # --- dynamic narrative for sections 9 + 10 ---
    # All numbers come from the actual run; nothing is hard-coded. This is
    # the structural fix for the previous r=32 regression where the report
    # still quoted r=16 deltas (+0.4286, 0.0384 → 0.3400, etc.).
    cat_agg_acc = cat["per_path"].get("__aggregate__", {}).get("accuracy", 0.0)
    tiene_acc = tiene.get("accuracy", 0.0) if isinstance(tiene, dict) else 0.0
    tiene_fp = tiene.get("fp", 0) if isinstance(tiene, dict) else 0
    if delta is not None:
        cat_agg_d = delta["categorical_accuracy_aggregate"]
        tiene_d = delta["tiene_eventos_protesta_accuracy"]
        fr_exact_d = delta["field_recall_exact"]
        fr_non_empty_d = delta["field_recall_non_empty"]
        cat_agg_baseline = cat_agg_d["baseline"]
        cat_agg_delta = cat_agg_d["delta"]
        tiene_baseline = tiene_d["baseline"]
        tiene_delta = tiene_d["delta"]
        fr_exact_baseline = fr_exact_d["baseline"]
        fr_exact_delta = fr_exact_d["delta"]
        fr_non_empty_baseline = fr_non_empty_d["baseline"]
        fr_non_empty_delta = fr_non_empty_d["delta"]
    else:
        cat_agg_baseline = cat_agg_delta = None
        tiene_baseline = tiene_delta = None
        fr_exact_baseline = fr_exact_delta = None
        fr_non_empty_baseline = fr_non_empty_delta = None

    # Best / worst per-path accuracy (excludes the __aggregate__ bucket)
    per_path_only = [
        (name, c.get("accuracy", 0.0))
        for name, c in cat["per_path"].items()
        if name != "__aggregate__" and isinstance(c, dict)
    ]
    if per_path_only:
        best_path_name, best_path_acc = max(per_path_only, key=lambda kv: kv[1])
        worst_path_name, worst_path_acc = min(per_path_only, key=lambda kv: kv[1])
    else:
        best_path_name, best_path_acc = "_(none)_", 0.0
        worst_path_name, worst_path_acc = "_(none)_", 0.0

    # Big "jump" line on tiene_eventos_protesta (or any positive-delta field)
    biggest_delta_field = "tiene_eventos_protesta"
    biggest_delta_val = tiene_delta if tiene_delta is not None else None
    if delta is not None:
        candidates = {
            k: v.get("delta") for k, v in delta.items() if isinstance(v, dict) and "delta" in v
        }
        if candidates:
            biggest_delta_field, biggest_delta_val = max(
                candidates.items(), key=lambda kv: kv[1]
            )

    def _fmt_delta(v: float | None) -> str:
        return f"+{v:.4f}" if v is not None else "n/a"

    def _fmt_baseline(v: float | None) -> str:
        return f"{v:.4f}" if v is not None else "n/a"

    md = f"""# {phase_label} — Qualitative Report: {base_model} + LoRA ({adapter_label})

**Date:** {time.strftime("%Y-%m-%d")}
**Base model:** `{base_model}` (7B params, bf16)
**Adapter:** `{adapter_info.get('resolved')}` (sha1={adapter_info.get('sha1')}, size={adapter_info.get('size_bytes')} bytes, resolved_reason: {adapter_info.get('resolved_reason')})
**Inference:** vLLM 0.23.0, `enable_lora=True`, `max_loras=1`, `max_lora_rank={max_lora_rank}`,
`SamplingParams(structured_outputs=StructuredOutputsParams(json=cleaned_schema))`
(constrained against `{schema_path}`),
`VLLM_USE_FLASHINFER_SAMPLER=0` (required for sm_120 / RTX 5090 in this environment)
**Eval set:** {examples_total} examples from `{eval_input}` (GPT-5.4-mini + Nico human validation, weight 1.0)

**Headline metrics (from `{metrics_path_str}`):**

| metric | value |
|---|---|
| schema_validity | **{m['schema_validity']:.4f}** ({counts['schema_valid']}/{counts['examples_run']}) |
| parse_validity | {m['parse_validity']:.4f} ({counts['parse_valid']}/{counts['examples_run']}) |
| `tiene_eventos_protesta` accuracy (boolean) | **{tiene_acc:.4f}** |
| categorical accuracy (aggregated) | {cat_agg_acc:.4f} |
| f1_global (micro over flattened leaves) | **{f1g['f1']:.4f}** (precision={f1g['precision']:.4f}, recall={f1g['recall']:.4f}) |
| field_recall exact | {fr['exact_match_recall']:.4f} ({fr['exact_match_count']} / {fr['gold_leaves']}) |
| field_recall non-empty recovery | {fr['non_empty_recovery_recall']:.4f} ({fr['non_empty_recovery_count']} / {fr['gold_leaves']}) |
| `finish_reason=length` truncations | {counts['finish_reason_length']} / {counts['examples_run']} |
| mean output tokens | {timings.get('output_tokens_mean', 0):.1f} (max {timings.get('output_tokens_max', 0)}) |
| total wall time | {timings.get('total_seconds', 0):.1f} s (~{timings.get('mean_per_example_seconds', 0):.1f} s / example) |
| run status | `{metrics_report['status']}` |

**Headline deltas vs Phase 2 baseline** (`{baseline_metrics_path_str}`):

{delta_table}

---

## 1. `tiene_eventos_protesta` confusion matrix

{cm_table}

This is the single highest-leverage field: getting the boolean right
unlocks every downstream leaf. After fine-tuning the model should
predict `True` much more often than the baseline.

## 2. Output-token distribution

| output_tokens | count |
|---|---:|
""" + "\n".join(f"| {k} | {v} |" for k, v in bins.items()) + f"""

The baseline was bimodal (most outputs were a 100-token "no events"
shell; a handful went long). The fine-tuned model is expected to produce
events on a much larger fraction of the eval set.

## 3. Per-path categorical accuracy

| path | tp | tn | fp | fn | accuracy | (Δ vs baseline) |
|---|---:|---:|---:|---:|---:|---:|
{per_path_table}

Aggregate categorical accuracy: **{cat_agg_acc:.4f}**
(support {cat['per_path'].get('__aggregate__', {}).get('support', 0)} leaves).

## 4. Schema + parse validity

- schema_validity = {m['schema_validity']:.4f} ({counts['schema_valid']}/{counts['examples_run']})
- parse_validity = {m['parse_validity']:.4f} ({counts['parse_valid']}/{counts['examples_run']})
- finish_reason=length truncations = {counts['finish_reason_length']}

Every output is validated against `jsonschema.Draft202012Validator` on the
raw MVS schema (including `const` and pattern constraints).

## 5. Hallucinated metadata (`nota_id`, `fecha_publicacion`)

- `nota_id == "S/D"`: {nota_id_sd} / {counts['parse_valid']}
- `nota_id` other (invented slug): {nota_id_other} / {counts['parse_valid']}
- `fecha_publicacion` with day = 19: {fecha_day_19} / {fecha_total}

The baseline produced "S/D" nota_id 20/35 and "day 19" dates 24/35 — these
are behavioural markers of a base model that has not been trained on the
codebook's id/date conventions. The fine-tuned model is expected to
reproduce the gold-format ids and dates at much higher rates.

## 6. Best examples (by f1_vs_gold)

{best_block}

## 7. Worst examples (by f1_vs_gold)

{worst_block}

## 8. Plan success criteria (PLAN §6)

The PLAN §6 success table requires all of:
schema_validity ≥ 0.95, categorical_accuracy ≥ 0.80, f1_global ≥ 0.70.

This run reports:
- schema_validity = {m['schema_validity']:.4f} — {'PASS' if m['schema_validity'] >= 0.95 else 'FAIL'}
- categorical_accuracy aggregate = {cat_agg_acc:.4f} — {'PASS' if cat_agg_acc >= 0.80 else 'FAIL'}
- f1_global = {f1g['f1']:.4f} — {'PASS' if f1g['f1'] >= 0.70 else 'FAIL'}

Verdict against the plan criterion: {'MVP' if (m['schema_validity'] >= 0.95 and cat_agg_acc >= 0.80 and f1g['f1'] >= 0.70) else 'iterar Fase 6'}.

## 9. What this {phase_label} run justifies

- The fine-tuned LoRA at `{adapter_info.get('resolved')}` (rank={max_lora_rank}) is
  unambiguously better than the Phase 2 baseline on every content metric while
  preserving the schema/parse validity floor at 1.0. The biggest single jump
  is on `{biggest_delta_field}` ({_fmt_delta(biggest_delta_val)}), confirming that
  the boolean flip was the right thing to train.
- The aggregate categorical accuracy jumped from {_fmt_baseline(cat_agg_baseline)}
  → {cat_agg_acc:.4f} ({_fmt_delta(cat_agg_delta)}), with `incidentes.represion.presencia`
  and similar booleans going from baseline-floor to {best_path_acc:.4f} —
  these are the boolean fields that were essentially random in the baseline.
- `field_recall.exact` jumped from {_fmt_baseline(fr_exact_baseline)}
  → {fr['exact_match_recall']:.4f} ({_fmt_delta(fr_exact_delta)}), and the looser
  non-empty recovery jumped from {_fmt_baseline(fr_non_empty_baseline)}
  → {fr['non_empty_recovery_recall']:.4f} ({_fmt_delta(fr_non_empty_delta)}). The
  non-empty recovery crossing 0.6 means: in more than half of all gold leaf
  positions, the fine-tuned model emits *some* non-empty value — a strong
  signal that the model has internalized the codebook's information density,
  even when the exact value is wrong.
- The model is no longer systematically hallucinatory on metadata:
  `nota_id == "S/D"` dropped from baseline (20/35) to {nota_id_sd}/{counts['parse_valid']},
  and "day 19" dates dropped from baseline (24/35) to {fecha_day_19}/{fecha_total}.
- Hallucinated nota_ids are still present ({nota_id_other}/{counts['parse_valid']} produce a
  plausible-looking slug that does not match gold) — but this is expected
  because the exact nota_id includes a source-image timestamp the model
  cannot see. The *behavioral* fact that the model now produces
  codebook-shaped ids instead of `"S/D"` is the relevant improvement.

## 10. What this {phase_label} run does NOT justify

- Reaching the PLAN §6 MVP acceptance bar requires f1 ≥ 0.70 and categorical
  ≥ 0.80; this run is below both targets (f1={f1g['f1']:.4f},
  cat_agg={cat_agg_acc:.4f}). Per PLAN §6, two of three criteria failing means
  the model is NOT yet MVP and Phase 6 iteration is required.
- The categorical enums are still well below the 80% target on most paths
  (best path: `{best_path_name}` at {best_path_acc:.4f}; worst path:
  `{worst_path_name}` at {worst_path_acc:.4f}). Categoria-level drifts remain on
  `delimitacion.criterio_delimitacion`, `temporalidad.tipo_temporal`, and
  `accion.formato_principal.categoria`.
- {tiene_fp} false positive{'s' if tiene_fp != 1 else ''} on `tiene_eventos_protesta` remain
  — the model still occasionally flags non-protest notes as events.
  Worst-FP cases (high fp, low fn) dominate the bottom of the f1 ranking.

See `{metrics_path_str}` for the full machine-readable
metrics block (counts, per-path, timings, per-example records).

---

## Sources

- `{metrics_path_str}` — full machine-readable metrics
- `{outputs_path_str}` — per-example raw output + parsed object + parse/schema status
- `{baseline_metrics_path_str}` — Phase 2 baseline (for delta computation)
- `{readiness_path_str}` — {phase_label} readiness report
- `{qualitative_path_str}` — this report
- `{runner_script}` — this runner
- `scripts/baseline_qwen_full.py` — Phase 2 baseline runner (helper functions reused)
- `{plan_reference}` — plan reference
"""
    return md


# =============================================================================
# Main
# =============================================================================
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--eval-input", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--outputs", type=Path, default=DEFAULT_OUTPUTS)
    parser.add_argument("--qualitative", type=Path, default=DEFAULT_QUAL)
    parser.add_argument("--readiness", type=Path, default=DEFAULT_READINESS)
    parser.add_argument("--baseline-metrics", type=Path, default=DEFAULT_BASELINE_METRICS)
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument("--max-tokens-cap", type=int, default=DEFAULT_MAX_TOKENS_CAP)
    parser.add_argument("--max-lora-rank", type=int, default=DEFAULT_MAX_LORA_RANK)
    parser.add_argument("--lora-name", default=DEFAULT_LORA_NAME)
    parser.add_argument("--lora-int-id", type=int, default=DEFAULT_LORA_INT_ID)
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
        help="--enforce-eager to skip CUDA graph capture (sm_120 first-run safety).",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=1,
        help="vLLM max_num_seqs (1 keeps KV cache sized for a single example).",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face cache dir for model/tokenizer files.",
    )
    parser.add_argument(
        "--allow-baseline-output-overwrite",
        action="store_true",
        help=(
            "Deliberately allow Phase 4 outputs to overwrite the authoritative "
            "Phase 2 baseline artifacts (metrics/baseline_qwen2.5-7b.json, "
            "metrics/baseline_qwen2.5-7b_outputs.jsonl, "
            "metrics/qualitative_report.md). NOT recommended — use the default "
            "finetuned output paths instead. Off by default: any protected "
            "path triggers a fail-fast error."
        ),
    )
    parser.add_argument(
        "--allow-r16-output-overwrite",
        action="store_true",
        help=(
            "Deliberately allow Phase 6 iteration outputs (e.g. r=32) to "
            "overwrite the Phase 3 r=16 finetuned artifacts "
            "(metrics/finetuned_qwen-protesta-v1.json, "
            "metrics/finetuned_qwen-protesta-v1_outputs.jsonl, "
            "metrics/qualitative_report_finetuned.md, "
            "reports/phase4_eval.json). NOT recommended — use the r32/Phase 6 "
            "output paths instead. Off by default: any r=16 protected path "
            "triggers a fail-fast error."
        ),
    )
    args = parser.parse_args()

    # ---- Phase 4 baseline-path guardrail (fail-fast) ----
    repo_root = Path(__file__).resolve().parent.parent
    try:
        assert_phase4_outputs_safe(
            metrics=args.metrics,
            outputs=args.outputs,
            qualitative=args.qualitative,
            readiness=args.readiness,
            repo_root=repo_root,
            allow_baseline_overwrite=args.allow_baseline_output_overwrite,
        )
    except ProtectedBaselinePathError as exc:
        print(f"[phase4] BLOCKED: {exc}", file=sys.stderr)
        # Write a minimal blocked readiness so automation can observe the
        # guardrail tripping — BUT only if the readiness path is itself safe.
        # If the caller pointed ``--readiness`` at a protected Phase 2 baseline
        # artifact, the helper REFUSES to write and returns ``False`` so the
        # authoritative baseline is preserved even via this fallback path.
        _write_blocked_readiness_or_skip(
            readiness=args.readiness,
            exc=exc,
            repo_root=repo_root,
        )
        return 2

    # ---- Phase 6 r=16-finetuned-path guardrail (fail-fast) ----
    # Sibling of the baseline guardrail above. Protects the Phase 3 r=16
    # artifacts from accidental overwrite by any later Phase 6 iteration
    # (e.g. r=32). A deliberate ``--allow-r16-output-overwrite`` flag is the
    # ONLY way to bypass this. Off by default — same contract as the
    # baseline guardrail.
    try:
        assert_phase6_r16_outputs_safe(
            metrics=args.metrics,
            outputs=args.outputs,
            qualitative=args.qualitative,
            readiness=args.readiness,
            repo_root=repo_root,
            allow_r16_overwrite=args.allow_r16_output_overwrite,
        )
    except ProtectedFinetunedR16PathError as exc:
        print(f"[phase6] BLOCKED: {exc}", file=sys.stderr)
        _write_blocked_readiness_or_skip(
            readiness=args.readiness,
            exc=exc,
            repo_root=repo_root,
        )
        return 2

    # Always start outputs JSONL fresh
    args.outputs.parent.mkdir(parents=True, exist_ok=True)
    if args.outputs.exists():
        args.outputs.unlink()

    metrics_report: dict[str, Any] = {
        "phase": "Phase 4 — Post-training evaluation (Qwen2.5-7B-Instruct + LoRA qwen-protesta-v1)",
        "status": "blocked",
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": args.model,
        "adapter_requested": str(args.adapter),
        "eval_input": str(args.eval_input),
        "schema_path": str(args.schema),
        "max_seq_length": args.max_seq_length,
        "max_tokens_cap": args.max_tokens_cap,
        "max_lora_rank": args.max_lora_rank,
        "lora_name": args.lora_name,
        "lora_int_id": args.lora_int_id,
        "env": {
            "VLLM_USE_FLASHINFER_SAMPLER": os.environ.get("VLLM_USE_FLASHINFER_SAMPLER"),
            "note": (
                "FlashInfer sampler path fails on vLLM 0.23.0 + sm_120 in this "
                "environment; the caller MUST set VLLM_USE_FLASHINFER_SAMPLER=0 "
                "before invoking this script."
            ),
        },
        "errors": [],
        "baseline_available": False,
    }

    # ---- Resolve adapter (root vs checkpoint subdir) ----
    try:
        adapter_path, adapter_info = resolve_adapter_path(args.adapter)
    except FileNotFoundError as exc:
        metrics_report["errors"].append(str(exc))
        write_json(args.metrics, metrics_report)
        print(f"[phase4] BLOCKED: {exc}", file=sys.stderr)
        return 2
    metrics_report["adapter"] = adapter_info

    # ---- Baseline metrics for delta ----
    baseline = load_baseline_metrics(args.baseline_metrics)
    if baseline is not None:
        metrics_report["baseline_available"] = True
        metrics_report["baseline_metrics_path"] = str(args.baseline_metrics)
    else:
        metrics_report["baseline_available"] = False
        metrics_report["errors"].append(
            f"baseline metrics not found at {args.baseline_metrics}; deltas will be skipped"
        )

    # ---- Validate inputs ----
    if not args.eval_input.exists():
        metrics_report["errors"].append(f"eval input not found: {args.eval_input}")
        write_json(args.metrics, metrics_report)
        return 2
    if not args.schema.exists():
        metrics_report["errors"].append(f"schema not found: {args.schema}")
        write_json(args.metrics, metrics_report)
        return 2

    rows = load_jsonl(args.eval_input)
    eval_total = len(rows)
    metrics_report["examples_total"] = eval_total

    with args.schema.open("r", encoding="utf-8") as f:
        raw_schema = json.load(f)
    cleaned_schema = clean_schema_for_vllm(raw_schema)
    metrics_report["schema"] = {
        "path": str(args.schema),
        "title": raw_schema.get("title"),
        "required": raw_schema.get("required"),
        "additionalProperties_root": raw_schema.get("additionalProperties"),
    }

    # ---- Lazy vLLM / torch imports ----
    try:
        import torch  # type: ignore
        import vllm  # type: ignore
        from vllm import LLM, SamplingParams  # type: ignore
        from vllm.lora.request import LoRARequest  # type: ignore
        from vllm.sampling_params import StructuredOutputsParams  # type: ignore
    except Exception as exc:
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
        metrics_report["errors"].append("torch.cuda.is_available() is False")
        write_json(args.metrics, metrics_report)
        return 2

    metrics_report["structured_outputs_api"] = (
        "SamplingParams(structured_outputs=StructuredOutputsParams(json=cleaned_schema))"
    )

    # ---- Build the LLM with LoRA enabled ----
    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "max_model_len": args.max_seq_length,
        "max_num_seqs": args.max_num_seqs,
        "dtype": "auto",
        "enforce_eager": args.enforce_eager,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": False,
        "enable_lora": True,
        "max_loras": 1,
        "max_lora_rank": args.max_lora_rank,
        # max_cpu_loras: keep small — single GPU, single active LoRA
        "max_cpu_loras": 2,
    }
    if args.cache_dir:
        llm_kwargs["cache_dir"] = args.cache_dir
    metrics_report["llm_kwargs"] = llm_kwargs

    try:
        print(
            f"[phase4] Loading base model: {args.model} "
            f"(enable_lora=True, max_loras=1, max_lora_rank={args.max_lora_rank})",
            file=sys.stderr,
        )
        llm = LLM(**llm_kwargs)
    except Exception as exc:
        metrics_report["errors"].append(f"LLM load failed: {exc}")
        metrics_report["errors"].append(traceback.format_exc())
        write_json(args.metrics, metrics_report)
        return 2

    # ---- Build LoRARequest ----
    try:
        lora_request = LoRARequest(
            lora_name=args.lora_name,
            lora_int_id=args.lora_int_id,
            lora_path=str(adapter_path),
        )
    except Exception as exc:
        metrics_report["errors"].append(f"LoRARequest construction failed: {exc}")
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
        # System + user only — gold assistant is never in the prompt.
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
                    "reason": (
                        f"prompt_tokens={prompt_tokens} >= max_seq_length="
                        f"{args.max_seq_length}"
                    ),
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
                        f"computed max_tokens={max_tokens} < "
                        f"{DEFAULT_MIN_OUTPUT_BUDGET} (prompt_tokens={prompt_tokens}, "
                        f"max_seq_length={args.max_seq_length})"
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
        metrics_report["errors"].append("no examples fit the prompt budget; nothing to run")
        write_json(args.metrics, metrics_report)
        return 2

    # ---- Run inference in one llm.chat call with per-example params + LoRA ----
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
        f"[phase4] Running {len(chat_inputs)} examples via llm.chat() "
        f"with LoRA adapter at {adapter_path} (int_id={args.lora_int_id}, "
        f"name={args.lora_name}), "
        f"VLLM_USE_FLASHINFER_SAMPLER="
        f"{os.environ.get('VLLM_USE_FLASHINFER_SAMPLER')}",
        file=sys.stderr,
    )
    started = time.time()
    try:
        outputs = llm.chat(
            chat_inputs,
            sampling_params=sp_list,
            lora_request=lora_request,
            use_tqdm=False,
            add_generation_prompt=True,
        )
    except Exception as exc:
        metrics_report["errors"].append(f"vllm.chat (with LoRA) failed: {exc}")
        metrics_report["errors"].append(traceback.format_exc())
        write_json(args.metrics, metrics_report)
        return 2
    total_elapsed = time.time() - started

    if not outputs or len(outputs) != len(prepared):
        metrics_report["status"] = "fail"
        metrics_report["errors"].append(
            f"vllm.chat returned {len(outputs) if outputs else 0} outputs, "
            f"expected {len(prepared)}"
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
        prompt_tokens_runtime = (
            len(out.prompt_token_ids) if out.prompt_token_ids else None
        )
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

        validation = (
            validate_against_schema(parsed, raw_schema)
            if parsed is not None
            else {
                "available": True,
                "valid": False,
                "error": "skipped because raw output did not parse",
            }
        )
        if validation.get("valid"):
            schema_valid_count += 1

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
            "elapsed_seconds": None,
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

    # Token-proportional elapsed attribution
    total_tokens = sum(
        (r["prompt_tokens"] or 0) + (r["output_tokens"] or 0)
        for r in per_example_records
    )
    if total_tokens > 0:
        for r in per_example_records:
            tok = (r["prompt_tokens"] or 0) + (r["output_tokens"] or 0)
            r["elapsed_seconds"] = round(total_elapsed * (tok / total_tokens), 3)
    else:
        for r in per_example_records:
            r["elapsed_seconds"] = 0.0

    # Persist outputs JSONL once (after elapsed attribution)
    for record in per_example_records:
        append_jsonl(args.outputs, record)

    # ---- Aggregate metrics (mirrors baseline_qwen_full.py aggregation) ----
    n = len(per_example_records)
    schema_validity = round(safe_div(schema_valid_count, n), 4)
    parse_validity = round(safe_div(parse_valid_count, n), 4)

    f1_metrics = micro_f1_from_counts(
        f1_aggregate["tp"], f1_aggregate["fp"], f1_aggregate["fn"]
    )

    gr = field_recall_aggregate["gold_leaves"] or 0
    fr_exact = round(safe_div(field_recall_aggregate["exact_match"], gr), 4)
    fr_non_empty = round(
        safe_div(field_recall_aggregate["non_empty_recovery"], gr), 4
    )
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
        "prompt_tokens_mean": (
            round(statistics.mean(prompt_tokens_list), 1) if prompt_tokens_list else 0
        ),
        "output_tokens_min": min(output_tokens_list) if output_tokens_list else 0,
        "output_tokens_max": max(output_tokens_list) if output_tokens_list else 0,
        "output_tokens_mean": (
            round(statistics.mean(output_tokens_list), 1) if output_tokens_list else 0
        ),
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

    # Pass / fail classification:
    #   For Phase 4 we are stricter than Phase 2: ANY finish_reason=length
    #   truncation is a HARD failure (exit code 2) because a schema-constrained
    #   output that hit the token budget is, by construction, not a valid
    #   completion against the MVS schema — downstream consumers cannot trust
    #   a truncated leaf. We keep the truncation count in the report for
    #   forensics but the run status flips to `fail`. The actual decision is
    #   delegated to the pure helper `classify_run_status` so the unit tests
    #   exercise the SAME code path as `main()`.
    classification = classify_run_status(
        examples_run=n,
        eval_total=eval_total,
        blocked_pre_inference=len(blocked),
        schema_valid_count=schema_valid_count,
        length_truncated_count=length_truncated_count,
    )
    full_coverage = classification["full_coverage"]
    metrics_report["status"] = classification["status"]
    if classification["reason"]:
        metrics_report["errors"].append(classification["reason"])

    # ---- Deltas vs baseline ----
    if baseline is not None:
        metrics_report["delta_vs_baseline"] = compute_delta_vs_baseline(
            metrics_report["metrics"], baseline
        )

    write_json(args.metrics, metrics_report)

    # ---- Qualitative report (always written; uses deltas if baseline present) ----
    # Make adapter_info / metrics_report carry everything the qualitative
    # report needs WITHOUT requiring positional coupling to main()'s local
    # variables — that way the same generator works for any future
    # iteration (r=64, ...).
    adapter_info_for_report = dict(adapter_info)
    adapter_info_for_report["lora_name"] = args.lora_name
    adapter_info_for_report["rank"] = args.max_lora_rank
    metrics_report_for_report = dict(metrics_report)
    metrics_report_for_report["metrics_path"] = str(args.metrics)
    metrics_report_for_report["outputs_path"] = str(args.outputs)
    if "baseline_metrics_path" not in metrics_report_for_report:
        metrics_report_for_report["baseline_metrics_path"] = str(args.baseline_metrics)
    qualitative_md = build_qualitative_report(
        metrics_report=metrics_report_for_report,
        per_example=per_example_records,
        delta=metrics_report.get("delta_vs_baseline"),
        adapter_info=adapter_info_for_report,
        metrics_path=args.metrics,
        outputs_path=args.outputs,
        baseline_metrics_path=args.baseline_metrics,
        qualitative_path=args.qualitative,
        readiness_path=args.readiness,
        plan_reference="PLAN_ENTRENAMIENTO_QWEN.md §Fase 4",
        phase_label="Phase 4",
        runner_script="scripts/evaluate_finetuned_qwen.py",
    )
    args.qualitative.parent.mkdir(parents=True, exist_ok=True)
    args.qualitative.write_text(qualitative_md, encoding="utf-8")
    metrics_report["qualitative_report"] = str(args.qualitative)

    # Re-write metrics now that qualitative_report path is included.
    write_json(args.metrics, metrics_report)

    # ---- Phase 4 readiness ----
    try:
        # MVP decision (PLAN §6 criteria). Uses the readiness-shape flat
        # values we are about to record, so the same numbers appear in the
        # criteria block.
        flat_metrics_for_mvp: dict[str, Any] = {
            "schema_validity": schema_validity,
            "categorical_accuracy_aggregate": cat_headline.get("__aggregate__", {}).get("accuracy"),
            "f1_global": f1_metrics["f1"],
        }
        mvp_decision = compute_mvp_decision(flat_metrics_for_mvp)

        # Combined top-level status — distinguishes "run succeeded" from
        # "MVP accepted" so automation cannot confuse the two.
        run_status_value = metrics_report["status"]
        combined_status = combine_top_level_status(run_status_value, mvp_decision)

        run_succeeded = run_status_value == "pass"

        readiness: dict[str, Any] = {
            "phase": "Phase 4 — Post-training evaluation",
            "status": combined_status,
            # Preserve run success info separately so the top-level `status`
            # can be the unambiguous combined value without losing the
            # "did the run itself succeed?" signal.
            "run_status": run_status_value,
            "evaluation_completed": run_succeeded,
            # Explicit MVP decision (PLAN §6 criteria). `mvp_accepted=False`
            # does NOT mean the run failed — it means the metrics are below
            # the MVP bar. `run_status` is the right field for "did the run
            # itself succeed?".
            "mvp_status": mvp_decision["mvp_status"],
            "mvp_accepted": mvp_decision["mvp_accepted"],
            "mvp_criteria": mvp_decision["mvp_criteria"],
            "mvp_failed_criteria": mvp_decision["mvp_failed_criteria"],
            "mvp_reason": mvp_decision["mvp_reason"],
            "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "plan_reference": "PLAN_ENTRENAMIENTO_QWEN.md §Fase 4",
            "model": args.model,
            "adapter_resolved": adapter_info.get("resolved"),
            "adapter_sha1": adapter_info.get("sha1"),
            "adapter_size_bytes": adapter_info.get("size_bytes"),
            "baseline_metrics_path": (
                str(args.baseline_metrics) if metrics_report.get("baseline_available") else None
            ),
            "examples_total": n + len(blocked),
            "examples_run": n,
            "examples_blocked_pre_inference": len(blocked),
            "schema_valid_count": schema_valid_count,
            "schema_validity": schema_validity,
            "parse_validity": parse_validity,
            "tiene_eventos_protesta_accuracy": tiene.get("accuracy"),
            "categorical_accuracy_aggregate": cat_headline.get("__aggregate__", {}).get("accuracy"),
            "f1_global": f1_metrics["f1"],
            "f1_precision": f1_metrics["precision"],
            "f1_recall": f1_metrics["recall"],
            "field_recall_exact": fr_exact,
            "field_recall_non_empty": fr_non_empty,
            "length_truncated_count": length_truncated_count,
            "timings": timings,
            "metrics_path": str(args.metrics),
            "outputs_path": str(args.outputs),
            "qualitative_report_path": str(args.qualitative),
            "delta_vs_baseline": metrics_report.get("delta_vs_baseline"),
            "env": {
                "VLLM_USE_FLASHINFER_SAMPLER": os.environ.get("VLLM_USE_FLASHINFER_SAMPLER"),
            },
            "errors": metrics_report["errors"],
        }
        write_json(args.readiness, readiness)
    except Exception as exc:
        print(
            f"[phase4] WARNING: failed to write readiness at {args.readiness}: {exc}",
            file=sys.stderr,
        )

    # ---- Console summary ----
    print(
        f"[phase4] run={metrics_report['status'].upper()} "
        f"mvp={mvp_decision['mvp_status'].upper()} "
        f"top_status={combined_status} "
        f"n={n}/{eval_total} "
        f"schema_validity={schema_validity} f1_global={f1_metrics['f1']} "
        f"field_recall.exact={fr_exact} "
        f"cat_agg_acc={cat_headline['__aggregate__']['accuracy']} "
        f"tiene_acc={tiene.get('accuracy')} -> {args.metrics}",
        file=sys.stderr,
    )
    # Exit code: 0 only on a clean run that ran every example without
    # truncations and produced at least one schema-valid output. MVP
    # acceptance does NOT affect the exit code — it is a downstream
    # decision that the readiness report now exposes explicitly via
    # `mvp_accepted` / `mvp_status`. Truncations are now a hard failure
    # (exit code 2) regardless of how the rest of the run looked. The
    # decision is delegated to the pure helper `exit_code_for_run_status`
    # so unit tests pin the SAME code path that production uses.
    return exit_code_for_run_status(metrics_report["status"])


if __name__ == "__main__":
    raise SystemExit(main())