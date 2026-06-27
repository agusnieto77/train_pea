#!/usr/bin/env python3
"""Unit tests for ``scripts/evaluate_finetuned_qwen.py`` guardrails.

These tests do NOT require vLLM, torch, the model weights, or any GPU.
They exercise only the pure helpers extracted at module level in
``evaluate_finetuned_qwen``:

  * ``_normalize_path``                       — absolute-path comparison helper.
  * ``_is_protected_baseline_path``           — protected-path detection.
  * ``assert_phase4_outputs_safe``            — fail-fast guardrail that keeps
                                                 Phase 4 from overwriting the
                                                 authoritative Phase 2 baseline
                                                 artifacts unless the deliberate
                                                 override flag is set.
  * ``compute_mvp_decision``                  — PLAN §6 MVP acceptance helper.
  * ``combine_top_level_status``              — top-level ``status`` combiner.
  * ``classify_run_status``                   — pure helper that derives the
                                                 run-only status (pass/fail/
                                                 incomplete) from metrics/run
                                                 facts including the truncation
                                                 count. Production ``main()``
                                                 delegates to this helper so
                                                 tests exercise the same code
                                                 path the runner uses.
  * ``exit_code_for_run_status``              — pure helper for the exit-code
                                                 policy: 0 only on a clean
                                                 ``"pass"``, otherwise 2.
                                                 Production ``main()`` delegates
                                                 to this helper.

The tests are deliberately stdlib-only (``unittest`` + ``pathlib``) so they
can run anywhere Python is installed. They are runnable via either:

    .venv/bin/python -m unittest scripts.test_evaluate_finetuned_qwen_paths
    .venv/bin/python scripts/test_evaluate_finetuned_qwen_paths.py

When run via ``python -m unittest scripts.test_evaluate_finetuned_qwen_paths``
from the project root, the ``sys.path`` tweak at the top of this file makes
``evaluate_finetuned_qwen`` importable.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

# Make `evaluate_finetuned_qwen` importable when running from the project root
# via `python -m unittest scripts.test_evaluate_finetuned_qwen_paths`. When
# running directly as a script the current directory (`scripts/`) is already
# on sys.path so this is a no-op.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import evaluate_finetuned_qwen as efq  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parent.parent

PROTECTED_METRICS = Path("metrics/baseline_qwen2.5-7b.json")
PROTECTED_OUTPUTS = Path("metrics/baseline_qwen2.5-7b_outputs.jsonl")
PROTECTED_QUAL = Path("metrics/qualitative_report.md")
# Readiness is also a write path. The Phase 4 guardrail protects it too so
# passing `--readiness metrics/baseline_qwen2.5-7b.json` cannot silently
# overwrite the authoritative Phase 2 baseline metrics artifact.
PROTECTED_READINESS = Path("metrics/baseline_qwen2.5-7b.json")

DEFAULT_METRICS = Path("metrics/finetuned_qwen-protesta-v1.json")
DEFAULT_OUTPUTS = Path("metrics/finetuned_qwen-protesta-v1_outputs.jsonl")
DEFAULT_QUAL = Path("metrics/qualitative_report_finetuned.md")
DEFAULT_READINESS = Path("reports/phase4_eval.json")


# ---------------------------------------------------------------------------
# _normalize_path
# ---------------------------------------------------------------------------
class NormalizePathTests(unittest.TestCase):
    """``_normalize_path`` must collapse relative/absolute/trailing-slash
    variants of the same target into one canonical form so equality checks
    are not bypassed by trivial path rewrites."""

    def test_absolute_input_is_returned_absolute(self):
        out = efq._normalize_path("/tmp/foo.json")
        self.assertTrue(out.is_absolute())
        self.assertEqual(str(out), "/tmp/foo.json")

    def test_relative_input_resolves_against_cwd(self):
        out = efq._normalize_path("metrics/foo.json")
        self.assertTrue(out.is_absolute())
        self.assertTrue(str(out).endswith("metrics/foo.json"))

    def test_dot_segments_collapsed(self):
        self.assertEqual(
            efq._normalize_path("metrics/./foo.json"),
            efq._normalize_path("metrics/foo.json"),
        )
        self.assertEqual(
            efq._normalize_path("metrics/sub/../foo.json"),
            efq._normalize_path("metrics/foo.json"),
        )


# ---------------------------------------------------------------------------
# _is_protected_baseline_path
# ---------------------------------------------------------------------------
class IsProtectedBaselinePathTests(unittest.TestCase):
    """``_is_protected_baseline_path`` must detect every documented Phase 2
    baseline artifact location."""

    def test_protected_metrics_is_detected(self):
        self.assertTrue(
            efq._is_protected_baseline_path(PROTECTED_METRICS, REPO_ROOT)
        )

    def test_protected_outputs_is_detected(self):
        self.assertTrue(
            efq._is_protected_baseline_path(PROTECTED_OUTPUTS, REPO_ROOT)
        )

    def test_protected_qualitative_is_detected(self):
        self.assertTrue(
            efq._is_protected_baseline_path(PROTECTED_QUAL, REPO_ROOT)
        )

    def test_absolute_form_of_protected_is_detected(self):
        absolute = REPO_ROOT / PROTECTED_METRICS
        self.assertTrue(
            efq._is_protected_baseline_path(absolute, REPO_ROOT)
        )

    def test_default_finetuned_metrics_is_not_protected(self):
        self.assertFalse(
            efq._is_protected_baseline_path(DEFAULT_METRICS, REPO_ROOT)
        )

    def test_default_finetuned_outputs_is_not_protected(self):
        self.assertFalse(
            efq._is_protected_baseline_path(DEFAULT_OUTPUTS, REPO_ROOT)
        )

    def test_default_finetuned_qualitative_is_not_protected(self):
        self.assertFalse(
            efq._is_protected_baseline_path(DEFAULT_QUAL, REPO_ROOT)
        )


# ---------------------------------------------------------------------------
# assert_phase4_outputs_safe
# ---------------------------------------------------------------------------
class AssertPhase4OutputsSafeTests(unittest.TestCase):
    """The fail-fast guardrail that protects Phase 2 baseline artifacts.

    The guardrail covers ALL FOUR write paths (metrics, outputs, qualitative,
    readiness). The ``--readiness`` path is critical because a CLI invocation
    such as ``--readiness metrics/baseline_qwen2.5-7b.json`` would otherwise
    silently overwrite the authoritative Phase 2 baseline metrics. The
    guardrail must trip BEFORE any write happens (including the
    blocked-readiness fallback in ``main()``).
    """

    def test_default_finetuned_paths_are_safe(self):
        """The default Phase 4 paths (including readiness) must never trip
        the guardrail."""
        efq.assert_phase4_outputs_safe(
            metrics=DEFAULT_METRICS,
            outputs=DEFAULT_OUTPUTS,
            qualitative=DEFAULT_QUAL,
            readiness=DEFAULT_READINESS,
            repo_root=REPO_ROOT,
            allow_baseline_overwrite=False,
        )  # no exception → success

    def test_default_readiness_path_alone_is_safe(self):
        """The default ``--readiness`` path (reports/phase4_eval.json) must
        not trip the guardrail on its own."""
        efq.assert_phase4_outputs_safe(
            metrics=DEFAULT_METRICS,
            outputs=DEFAULT_OUTPUTS,
            qualitative=DEFAULT_QUAL,
            readiness=DEFAULT_READINESS,
            repo_root=REPO_ROOT,
            allow_baseline_overwrite=False,
        )  # no exception → success

    def test_explicit_path_to_baseline_metrics_raises(self):
        with self.assertRaises(efq.ProtectedBaselinePathError) as ctx:
            efq.assert_phase4_outputs_safe(
                metrics=PROTECTED_METRICS,
                outputs=DEFAULT_OUTPUTS,
                qualitative=DEFAULT_QUAL,
                readiness=DEFAULT_READINESS,
                repo_root=REPO_ROOT,
                allow_baseline_overwrite=False,
            )
        msg = str(ctx.exception)
        self.assertIn("--metrics", msg)
        self.assertIn(str(PROTECTED_METRICS), msg)

    def test_explicit_path_to_baseline_outputs_raises(self):
        with self.assertRaises(efq.ProtectedBaselinePathError) as ctx:
            efq.assert_phase4_outputs_safe(
                metrics=DEFAULT_METRICS,
                outputs=PROTECTED_OUTPUTS,
                qualitative=DEFAULT_QUAL,
                readiness=DEFAULT_READINESS,
                repo_root=REPO_ROOT,
                allow_baseline_overwrite=False,
            )
        msg = str(ctx.exception)
        self.assertIn("--outputs", msg)
        self.assertIn(str(PROTECTED_OUTPUTS), msg)

    def test_explicit_path_to_baseline_qualitative_raises(self):
        with self.assertRaises(efq.ProtectedBaselinePathError) as ctx:
            efq.assert_phase4_outputs_safe(
                metrics=DEFAULT_METRICS,
                outputs=DEFAULT_OUTPUTS,
                qualitative=PROTECTED_QUAL,
                readiness=DEFAULT_READINESS,
                repo_root=REPO_ROOT,
                allow_baseline_overwrite=False,
            )
        msg = str(ctx.exception)
        self.assertIn("--qualitative", msg)
        self.assertIn(str(PROTECTED_QUAL), msg)

    def test_explicit_path_to_baseline_readiness_raises(self):
        """The CRITICAL regression guardrail: passing
        ``--readiness metrics/baseline_qwen2.5-7b.json`` must fail-fast BEFORE
        the blocked-readiness fallback in ``main()`` would otherwise overwrite
        the authoritative Phase 2 baseline metrics. The guardrail must
        protect the readiness path the same way it protects metrics/outputs/
        qualitative."""
        with self.assertRaises(efq.ProtectedBaselinePathError) as ctx:
            efq.assert_phase4_outputs_safe(
                metrics=DEFAULT_METRICS,
                outputs=DEFAULT_OUTPUTS,
                qualitative=DEFAULT_QUAL,
                readiness=PROTECTED_READINESS,
                repo_root=REPO_ROOT,
                allow_baseline_overwrite=False,
            )
        msg = str(ctx.exception)
        self.assertIn("--readiness", msg)
        self.assertIn(str(PROTECTED_READINESS), msg)

    def test_absolute_path_form_of_protected_readiness_still_raises(self):
        """An absolute form of a protected readiness path must still trip the
        guardrail — ``_is_protected_baseline_path`` normalises both sides."""
        with self.assertRaises(efq.ProtectedBaselinePathError) as ctx:
            efq.assert_phase4_outputs_safe(
                metrics=DEFAULT_METRICS,
                outputs=DEFAULT_OUTPUTS,
                qualitative=DEFAULT_QUAL,
                readiness=REPO_ROOT / PROTECTED_READINESS,
                repo_root=REPO_ROOT,
                allow_baseline_overwrite=False,
            )
        msg = str(ctx.exception)
        self.assertIn("--readiness", msg)

    def test_dot_segment_form_of_protected_readiness_still_raises(self):
        """``./metrics/baseline_qwen2.5-7b.json`` style variants must also
        trip the guardrail — path normalisation collapses them."""
        with self.assertRaises(efq.ProtectedBaselinePathError):
            efq.assert_phase4_outputs_safe(
                metrics=DEFAULT_METRICS,
                outputs=DEFAULT_OUTPUTS,
                qualitative=DEFAULT_QUAL,
                readiness=Path("metrics/./baseline_qwen2.5-7b.json"),
                repo_root=REPO_ROOT,
                allow_baseline_overwrite=False,
            )

    def test_all_four_protected_paths_listed_in_single_error(self):
        """All four write paths colliding simultaneously must each appear in
        the error so the caller knows exactly which flags to fix."""
        with self.assertRaises(efq.ProtectedBaselinePathError) as ctx:
            efq.assert_phase4_outputs_safe(
                metrics=PROTECTED_METRICS,
                outputs=PROTECTED_OUTPUTS,
                qualitative=PROTECTED_QUAL,
                readiness=PROTECTED_READINESS,
                repo_root=REPO_ROOT,
                allow_baseline_overwrite=False,
            )
        msg = str(ctx.exception)
        self.assertIn("--metrics", msg)
        self.assertIn("--outputs", msg)
        self.assertIn("--qualitative", msg)
        self.assertIn("--readiness", msg)
        self.assertIn("--allow-baseline-output-overwrite", msg)

    def test_override_flag_bypasses_guardrail_for_all_four(self):
        """With --allow-baseline-output-overwrite the guardrail is silent,
        even if every flag points at a protected path. This is the deliberate
        override contract."""
        efq.assert_phase4_outputs_safe(
            metrics=PROTECTED_METRICS,
            outputs=PROTECTED_OUTPUTS,
            qualitative=PROTECTED_QUAL,
            readiness=PROTECTED_READINESS,
            repo_root=REPO_ROOT,
            allow_baseline_overwrite=True,
        )  # no exception → success

    def test_override_flag_bypasses_guardrail_for_readiness_only(self):
        """Override is a single binary switch — it bypasses the readiness
        check too, not just the metrics/outputs/qualitative checks."""
        efq.assert_phase4_outputs_safe(
            metrics=DEFAULT_METRICS,
            outputs=DEFAULT_OUTPUTS,
            qualitative=DEFAULT_QUAL,
            readiness=PROTECTED_READINESS,
            repo_root=REPO_ROOT,
            allow_baseline_overwrite=True,
        )  # no exception → success

    def test_absolute_path_form_of_protected_metrics_still_raises(self):
        with self.assertRaises(efq.ProtectedBaselinePathError):
            efq.assert_phase4_outputs_safe(
                metrics=REPO_ROOT / PROTECTED_METRICS,
                outputs=DEFAULT_OUTPUTS,
                qualitative=DEFAULT_QUAL,
                readiness=DEFAULT_READINESS,
                repo_root=REPO_ROOT,
                allow_baseline_overwrite=False,
            )

    def test_random_tmp_paths_are_safe(self):
        """The guardrail must not over-trigger — paths outside
        metrics/baseline_* / qualitative_report.md must always pass,
        including the readiness path."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            efq.assert_phase4_outputs_safe(
                metrics=root / "metrics" / "my_run.json",
                outputs=root / "metrics" / "my_run_outputs.jsonl",
                qualitative=root / "metrics" / "my_run_qual.md",
                readiness=root / "reports" / "my_phase4.json",
                repo_root=root,
                allow_baseline_overwrite=False,
            )

    def test_readiness_pointing_at_protected_outputs_path_raises(self):
        """A readiness path that resolves to a different protected artifact
        (outputs or qualitative) must also trip the guardrail — the readiness
        label is just for the CLI flag, the protection is path-based."""
        with self.assertRaises(efq.ProtectedBaselinePathError) as ctx:
            efq.assert_phase4_outputs_safe(
                metrics=DEFAULT_METRICS,
                outputs=DEFAULT_OUTPUTS,
                qualitative=DEFAULT_QUAL,
                readiness=PROTECTED_OUTPUTS,
                repo_root=REPO_ROOT,
                allow_baseline_overwrite=False,
            )
        msg = str(ctx.exception)
        self.assertIn("--readiness", msg)


# ---------------------------------------------------------------------------
# compute_mvp_decision
# ---------------------------------------------------------------------------
class ComputeMvpDecisionTests(unittest.TestCase):
    """PLAN §6 MVP acceptance: all three criteria must be met."""

    def test_all_three_pass_means_accepted(self):
        mvp = efq.compute_mvp_decision({
            "schema_validity": 1.0,
            "categorical_accuracy_aggregate": 0.95,
            "f1_global": 0.85,
        })
        self.assertTrue(mvp["mvp_accepted"])
        self.assertEqual(mvp["mvp_status"], "accepted")
        self.assertEqual(mvp["mvp_failed_criteria"], [])
        for crit in mvp["mvp_criteria"].values():
            self.assertTrue(crit["passed"])

    def test_exact_threshold_is_accepted(self):
        """≥ boundary — values exactly equal to the threshold must pass."""
        mvp = efq.compute_mvp_decision({
            "schema_validity": 0.95,
            "categorical_accuracy_aggregate": 0.80,
            "f1_global": 0.70,
        })
        self.assertTrue(mvp["mvp_accepted"])
        self.assertEqual(mvp["mvp_status"], "accepted")

    def test_one_below_threshold_means_failed(self):
        mvp = efq.compute_mvp_decision({
            "schema_validity": 1.0,
            "categorical_accuracy_aggregate": 0.79,
            "f1_global": 0.85,
        })
        self.assertFalse(mvp["mvp_accepted"])
        self.assertEqual(mvp["mvp_status"], "failed")
        self.assertEqual(mvp["mvp_failed_criteria"], ["categorical_accuracy_aggregate"])

    def test_two_below_threshold_means_failed(self):
        mvp = efq.compute_mvp_decision({
            "schema_validity": 1.0,
            "categorical_accuracy_aggregate": 0.34,
            "f1_global": 0.4637,
        })
        self.assertFalse(mvp["mvp_accepted"])
        self.assertEqual(mvp["mvp_status"], "failed")
        self.assertEqual(
            mvp["mvp_failed_criteria"],
            ["categorical_accuracy_aggregate", "f1_global"],
        )
        # The actual Phase 4 numbers — pin the contract against drift.
        self.assertIn("2/3", mvp["mvp_reason"])
        self.assertIn("categorical_accuracy_aggregate", mvp["mvp_reason"])
        self.assertIn("f1_global", mvp["mvp_reason"])

    def test_three_below_threshold_means_failed(self):
        mvp = efq.compute_mvp_decision({
            "schema_validity": 0.5,
            "categorical_accuracy_aggregate": 0.5,
            "f1_global": 0.5,
        })
        self.assertFalse(mvp["mvp_accepted"])
        self.assertEqual(
            set(mvp["mvp_failed_criteria"]),
            {"schema_validity", "categorical_accuracy_aggregate", "f1_global"},
        )

    def test_missing_metric_means_failed(self):
        """Missing values are treated as conservative failures — never silently
        promotes a run to MVP on incomplete data."""
        mvp = efq.compute_mvp_decision({
            "schema_validity": 1.0,
            "categorical_accuracy_aggregate": 0.95,
            # f1_global missing
        })
        self.assertFalse(mvp["mvp_accepted"])
        self.assertEqual(mvp["mvp_failed_criteria"], ["f1_global"])

    def test_non_numeric_metric_means_failed(self):
        mvp = efq.compute_mvp_decision({
            "schema_validity": 1.0,
            "categorical_accuracy_aggregate": "n/a",
            "f1_global": 0.85,
        })
        self.assertFalse(mvp["mvp_accepted"])
        self.assertEqual(mvp["mvp_failed_criteria"], ["categorical_accuracy_aggregate"])

    def test_nested_metrics_shape_works(self):
        """Accept the nested metrics-report shape (used inside main()) as
        well as the flat readiness shape."""
        nested = {
            "metrics": {
                "schema_validity": 1.0,
                "categorical_accuracy": {
                    "per_path": {"__aggregate__": {"accuracy": 0.95}},
                },
                "f1_global": {"f1": 0.85},
            },
        }
        mvp = efq.compute_mvp_decision(nested)
        self.assertTrue(mvp["mvp_accepted"])

    def test_mvp_criteria_include_threshold_value_passed(self):
        mvp = efq.compute_mvp_decision({
            "schema_validity": 0.94,
            "categorical_accuracy_aggregate": 0.81,
            "f1_global": 0.71,
        })
        c = mvp["mvp_criteria"]["schema_validity"]
        self.assertEqual(c["threshold"], 0.95)
        self.assertEqual(c["value"], 0.94)
        self.assertFalse(c["passed"])
        c = mvp["mvp_criteria"]["categorical_accuracy_aggregate"]
        self.assertEqual(c["threshold"], 0.80)
        self.assertEqual(c["value"], 0.81)
        self.assertTrue(c["passed"])


# ---------------------------------------------------------------------------
# combine_top_level_status
# ---------------------------------------------------------------------------
class CombineTopLevelStatusTests(unittest.TestCase):
    """The combined top-level ``status`` string distinguishes run outcome from
    MVP acceptance so automation can gate on each independently."""

    def test_pass_run_with_mvp_accepted_yields_completed_mvp_accepted(self):
        mvp = {"mvp_accepted": True, "mvp_status": "accepted"}
        self.assertEqual(
            efq.combine_top_level_status("pass", mvp),
            "completed_mvp_accepted",
        )

    def test_pass_run_with_mvp_failed_yields_completed_mvp_failed(self):
        mvp = {"mvp_accepted": False, "mvp_status": "failed"}
        self.assertEqual(
            efq.combine_top_level_status("pass", mvp),
            "completed_mvp_failed",
        )

    def test_fail_run_yields_failed_regardless_of_mvp(self):
        mvp = {"mvp_accepted": True, "mvp_status": "accepted"}
        self.assertEqual(
            efq.combine_top_level_status("fail", mvp),
            "failed",
        )

    def test_blocked_run_yields_blocked_regardless_of_mvp(self):
        mvp = {"mvp_accepted": True, "mvp_status": "accepted"}
        self.assertEqual(
            efq.combine_top_level_status("blocked", mvp),
            "blocked",
        )

    def test_incomplete_run_yields_incomplete_regardless_of_mvp(self):
        mvp = {"mvp_accepted": True, "mvp_status": "accepted"}
        self.assertEqual(
            efq.combine_top_level_status("incomplete", mvp),
            "incomplete",
        )


# ---------------------------------------------------------------------------
# Exit-code policy: truncations must return non-zero
# ---------------------------------------------------------------------------
class TruncationExitCodeTests(unittest.TestCase):
    """The Phase 4 runner must return exit code 2 (non-zero) when ANY example
    hit ``finish_reason=length``. The previous contract returned 0 with the
    misleading ``status=pass_with_truncations`` string — that was the bug
    these tests pin against regression.

    These tests call :func:`evaluate_finetuned_qwen.classify_run_status` and
    :func:`evaluate_finetuned_qwen.exit_code_for_run_status` directly. Both
    are the SAME pure helpers that production ``main()`` delegates to, so
    any refactor that weakens the truncation gate (e.g. reintroducing the
    ``pass_with_truncations`` status or returning exit code 0 on a truncated
    run) will fail THESE tests. The test no longer reimplements the policy
    locally — if the policy changes, the test must change with it.
    """

    def test_truncations_classify_as_hard_failure(self):
        """A run with truncations must NOT reach ``status == 'pass'``. The
        production helper ``classify_run_status`` must classify any
        ``length_truncated_count > 0`` (with full coverage otherwise) as
        ``status='fail'``, and ``exit_code_for_run_status`` must return 2.
        This exercises the SAME code path production uses."""
        cls = efq.classify_run_status(
            examples_run=35,
            eval_total=35,
            blocked_pre_inference=0,
            schema_valid_count=35,
            length_truncated_count=1,
        )
        self.assertEqual(cls["status"], "fail")
        self.assertTrue(cls["full_coverage"])
        # Reason mentions the truncation gate so a regression that drops
        # the message is also pinned here.
        self.assertIn("finish_reason=length", cls["reason"])

        # Exit-code policy: 0 only on a clean run that ran every example
        # without truncations.
        self.assertEqual(efq.exit_code_for_run_status(cls["status"]), 2)

    def test_no_truncations_classify_as_pass(self):
        cls = efq.classify_run_status(
            examples_run=35,
            eval_total=35,
            blocked_pre_inference=0,
            schema_valid_count=35,
            length_truncated_count=0,
        )
        self.assertEqual(cls["status"], "pass")
        self.assertTrue(cls["full_coverage"])
        self.assertEqual(efq.exit_code_for_run_status(cls["status"]), 0)

    def test_pass_with_truncations_status_string_no_longer_emitted(self):
        """The runner must NEVER emit ``pass_with_truncations`` as the
        ``run_status`` — that string was the source of the ambiguity. This
        test pins the policy at the helper layer so a future refactor
        cannot silently re-introduce it. We exercise several truncation
        counts to ensure the policy holds across edge values."""
        for trunc_count in (1, 5, 35):
            cls = efq.classify_run_status(
                examples_run=35,
                eval_total=35,
                blocked_pre_inference=0,
                schema_valid_count=35,
                length_truncated_count=trunc_count,
            )
            self.assertNotEqual(cls["status"], "pass_with_truncations")
            self.assertNotEqual(cls["status"], "pass")
            self.assertEqual(cls["status"], "fail")


# ---------------------------------------------------------------------------
# classify_run_status — full coverage of the Phase 4 policy
# ---------------------------------------------------------------------------
class ClassifyRunStatusTests(unittest.TestCase):
    """Direct, exhaustive coverage of :func:`evaluate_finetuned_qwen.classify_run_status`.

    These tests pin every documented branch of the policy so a regression
    in any one of them breaks a test (not just silently lets a bad run
    through).
    """

    def test_full_clean_run_is_pass(self):
        cls = efq.classify_run_status(
            examples_run=35,
            eval_total=35,
            blocked_pre_inference=0,
            schema_valid_count=35,
            length_truncated_count=0,
        )
        self.assertEqual(cls["status"], "pass")
        self.assertTrue(cls["full_coverage"])
        self.assertEqual(cls["reason"], "")
        self.assertEqual(efq.exit_code_for_run_status(cls["status"]), 0)

    def test_full_clean_run_with_one_truncation_is_fail(self):
        """Single truncation flips status to fail even when every other
        gate is green."""
        cls = efq.classify_run_status(
            examples_run=35,
            eval_total=35,
            blocked_pre_inference=0,
            schema_valid_count=35,
            length_truncated_count=1,
        )
        self.assertEqual(cls["status"], "fail")
        self.assertEqual(efq.exit_code_for_run_status(cls["status"]), 2)

    def test_zero_schema_valid_is_fail_even_without_truncations(self):
        cls = efq.classify_run_status(
            examples_run=35,
            eval_total=35,
            blocked_pre_inference=0,
            schema_valid_count=0,
            length_truncated_count=0,
        )
        self.assertEqual(cls["status"], "fail")
        self.assertEqual(efq.exit_code_for_run_status(cls["status"]), 2)

    def test_blocked_pre_inference_yields_incomplete(self):
        cls = efq.classify_run_status(
            examples_run=34,
            eval_total=35,
            blocked_pre_inference=1,
            schema_valid_count=34,
            length_truncated_count=0,
        )
        self.assertEqual(cls["status"], "incomplete")
        self.assertFalse(cls["full_coverage"])
        self.assertEqual(efq.exit_code_for_run_status(cls["status"]), 2)

    def test_truncations_win_over_incomplete_in_branch_order(self):
        """Truncation has HIGHER precedence than the incomplete check: if
        the run both had pre-inference blocks AND hit ``finish_reason=length``,
        the gate still flips to ``"fail"`` (HARD failure) and NOT
        ``"incomplete"``. Per the documented Phase 4 policy, ANY truncation
        is a hard failure regardless of coverage — a truncated
        schema-constrained output is, by construction, not a valid completion
        and the gate must not let it escape as "incomplete".

        This pins the branch order: zero_schema_valid → fail; truncation →
        fail; not full_coverage → incomplete; else pass. Truncation wins
        over incomplete so the order is FAIL > INCOMPLETE."""
        cls = efq.classify_run_status(
            examples_run=34,
            eval_total=35,
            blocked_pre_inference=1,
            schema_valid_count=34,
            length_truncated_count=2,
        )
        # Truncation wins: status is "fail", NOT "incomplete".
        self.assertEqual(cls["status"], "fail")
        self.assertEqual(efq.exit_code_for_run_status(cls["status"]), 2)
        # Reason mentions finish_reason=length so the failure is traceable.
        self.assertIn("finish_reason=length", cls["reason"])
        # Pin that we NEVER emit "incomplete" for a truncated run.
        self.assertNotEqual(cls["status"], "incomplete")
        self.assertNotEqual(cls["status"], "pass_with_truncations")

    def test_examples_run_less_than_eval_total_yields_incomplete(self):
        """Examples were skipped (not blocked pre-inference but not
        attempted) — still incomplete. Pin the gate against the
        ``examples_run == eval_total`` invariant."""
        cls = efq.classify_run_status(
            examples_run=30,
            eval_total=35,
            blocked_pre_inference=0,
            schema_valid_count=30,
            length_truncated_count=0,
        )
        self.assertEqual(cls["status"], "incomplete")
        self.assertFalse(cls["full_coverage"])
        self.assertEqual(efq.exit_code_for_run_status(cls["status"]), 2)

    def test_full_clean_run_with_truncation_count_one_returns_hard_fail(self):
        """Mutation-resistant: if a future refactor weakened the truncation
        gate (e.g. reintroduced ``pass_with_truncations``), this test
        fails because the helper would emit a non-``"fail"`` status."""
        cls = efq.classify_run_status(
            examples_run=35,
            eval_total=35,
            blocked_pre_inference=0,
            schema_valid_count=35,
            length_truncated_count=1,
        )
        # Hard-fail contract: truncations ⇒ status "fail".
        self.assertEqual(cls["status"], "fail")
        # Hard-fail contract: truncations ⇒ exit code 2.
        self.assertEqual(efq.exit_code_for_run_status(cls["status"]), 2)
        # Reason mentions finish_reason=length so the failure is traceable.
        self.assertIn("finish_reason=length", cls["reason"])
        # Pin that we NEVER emit the old ambiguous "pass_with_truncations".
        self.assertNotEqual(cls["status"], "pass_with_truncations")


# ---------------------------------------------------------------------------
# exit_code_for_run_status — direct table-driven coverage
# ---------------------------------------------------------------------------
class ExitCodeForRunStatusTests(unittest.TestCase):
    """``exit_code_for_run_status`` is the canonical exit-code policy used
    by ``main()``. These tests pin the table directly so a refactor that
    returns 0 on ``"fail"`` / ``"incomplete"`` / ``"blocked"`` cannot slip
    through."""

    def test_pass_returns_zero(self):
        self.assertEqual(efq.exit_code_for_run_status("pass"), 0)

    def test_fail_returns_two(self):
        self.assertEqual(efq.exit_code_for_run_status("fail"), 2)

    def test_incomplete_returns_two(self):
        self.assertEqual(efq.exit_code_for_run_status("incomplete"), 2)

    def test_blocked_returns_two(self):
        self.assertEqual(efq.exit_code_for_run_status("blocked"), 2)

    def test_empty_string_returns_two(self):
        """Defensive: even a malformed/empty status string returns 2 so
        the runner NEVER silently exits 0."""
        self.assertEqual(efq.exit_code_for_run_status(""), 2)

    def test_pass_with_whitespace_returns_zero(self):
        """Helper strips whitespace before comparison."""
        self.assertEqual(efq.exit_code_for_run_status("  pass  "), 0)


# ---------------------------------------------------------------------------
# End-to-end: a simulated write of reports/phase4_eval.json with new semantics
# ---------------------------------------------------------------------------
class ReadinessReportSemanticsTests(unittest.TestCase):
    """Reproduce the exact structure the runner now writes for the existing
    Phase 4 numbers and pin it against the contract."""

    def _simulate_readiness_write(self) -> dict:
        schema = 1.0
        cat_agg = 0.34
        f1 = 0.4637
        run_status = "pass"

        flat = {
            "schema_validity": schema,
            "categorical_accuracy_aggregate": cat_agg,
            "f1_global": f1,
        }
        mvp = efq.compute_mvp_decision(flat)
        combined = efq.combine_top_level_status(run_status, mvp)

        return {
            "status": combined,
            "run_status": run_status,
            "evaluation_completed": run_status == "pass",
            "mvp_accepted": mvp["mvp_accepted"],
            "mvp_status": mvp["mvp_status"],
            "mvp_criteria": mvp["mvp_criteria"],
            "mvp_failed_criteria": mvp["mvp_failed_criteria"],
            "mvp_reason": mvp["mvp_reason"],
            "schema_validity": schema,
            "categorical_accuracy_aggregate": cat_agg,
            "f1_global": f1,
        }

    def test_top_level_status_is_completed_mvp_failed_not_pass(self):
        readiness = self._simulate_readiness_write()
        # The old contract emitted status="pass" — that string is now reserved
        # for run_status. The top-level must distinguish "run finished" from
        # "MVP accepted".
        self.assertEqual(readiness["status"], "completed_mvp_failed")
        self.assertNotEqual(readiness["status"], "pass")

    def test_run_status_preserves_run_success_info(self):
        readiness = self._simulate_readiness_write()
        self.assertEqual(readiness["run_status"], "pass")
        self.assertTrue(readiness["evaluation_completed"])

    def test_mvp_accepted_is_false_but_run_succeeded(self):
        readiness = self._simulate_readiness_write()
        # Run succeeded (run_status=pass) but MVP NOT accepted — the two
        # signals must be independently observable in the JSON.
        self.assertFalse(readiness["mvp_accepted"])
        self.assertEqual(readiness["run_status"], "pass")
        self.assertEqual(readiness["evaluation_completed"], True)

    def test_mvp_failed_criteria_lists_exactly_two(self):
        readiness = self._simulate_readiness_write()
        self.assertEqual(
            set(readiness["mvp_failed_criteria"]),
            {"categorical_accuracy_aggregate", "f1_global"},
        )

    def test_metric_numbers_preserved(self):
        readiness = self._simulate_readiness_write()
        # Per user: "Do not modify baseline metrics except metadata if
        # absolutely necessary; preserve generated finetuned outputs."
        self.assertEqual(readiness["schema_validity"], 1.0)
        self.assertEqual(readiness["categorical_accuracy_aggregate"], 0.34)
        self.assertEqual(readiness["f1_global"], 0.4637)

    def test_report_is_json_serializable_round_trip(self):
        readiness = self._simulate_readiness_write()
        # Round-trip through json.dumps/loads to guarantee the readiness
        # contract is JSON-safe (no Path objects, no tuples, etc.).
        text = json.dumps(readiness, ensure_ascii=False, indent=2)
        parsed = json.loads(text)
        self.assertEqual(parsed["status"], "completed_mvp_failed")
        self.assertEqual(parsed["mvp_accepted"], False)
        self.assertEqual(parsed["run_status"], "pass")


# ---------------------------------------------------------------------------
# Production main() — blocked-readiness fallback must not overwrite a
# protected readiness path.
# ---------------------------------------------------------------------------
class MainBlockedReadinessFallbackTests(unittest.TestCase):
    """Production ``main()`` must not overwrite a protected readiness path
    even via the blocked-readiness fallback.

    Background
    ----------
    The runner used to catch :class:`ProtectedBaselinePathError` inside
    ``main()`` and then write a minimal ``blocked_readiness`` payload to
    ``args.readiness`` so automation could observe the guardrail tripping.
    That was the documented behaviour — but it created a second-order bug:
    if the caller passed ``--readiness metrics/baseline_qwen2.5-7b.json``
    (so ``args.readiness`` itself pointed at a protected Phase 2 baseline
    artifact), the fallback would happily overwrite the authoritative
    baseline metrics with the blocked payload, defeating the very
    guardrail that had just raised.

    The fix is to make the blocked-readiness decision explicit: the
    :func:`_write_blocked_readiness_or_skip` helper checks
    :func:`_is_protected_baseline_path` on the readiness path BEFORE any
    write. If the path is protected, it prints to stderr and returns
    ``False`` so ``main()`` can exit 2 without producing any artifact.

    These tests drive the EXACT production code path (the real ``main()``
    via ``sys.argv`` patching, and the extracted production helper)
    rather than re-implementing the policy locally, so a future refactor
    that reintroduces the overwriting fallback breaks THESE tests.
    """

    SENTINEL_PAYLOAD = (
        '{"phase": "SENTINEL — DO NOT OVERWRITE", "status": "untouched"}\n'
    )

    def setUp(self):
        # Use a temp dir for every test so we never touch real baseline
        # artifacts under /home/.../train_pea/metrics even if the production
        # code regresses.
        self._td_ctx = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._td_ctx.name)
        # Sentinel "protected baseline" file: mimics the real
        # metrics/baseline_qwen2.5-7b.json structure so size + content are
        # both observable in the test.
        self.sentinel = self.tmpdir / "metrics" / "baseline_qwen2.5-7b.json"
        self.sentinel.parent.mkdir(parents=True, exist_ok=True)
        self.sentinel.write_text(self.SENTINEL_PAYLOAD, encoding="utf-8")
        # Capture original module state so tearDown restores it precisely,
        # even if a test fails mid-way.
        self._orig_protected = efq.PROTECTED_BASELINE_OUTPUT_PATHS
        self._orig_r16_protected = efq.PROTECTED_FINETUNED_R16_OUTPUT_PATHS
        self._orig_argv = sys.argv
        # Monkey-patch both protected path tuples:
        #   * baseline → sentinel (so the runner treats it as the protected
        #     Phase 2 baseline artifact for THIS test run).
        #   * r16 → empty (so the OTHER Phase 3 r=16 guardrail does NOT
        #     treat the test's safe readiness path ``phase4_eval.json`` as
        #     protected; that file is treated as "safe" for the purposes
        #     of the baseline-blocked-fallback tests below).
        # The dedicated r16 protection behavior is exercised in
        # ``MainBlockedReadinessFallbackForR16Tests`` further down.
        efq.PROTECTED_BASELINE_OUTPUT_PATHS = (self.sentinel,)
        efq.PROTECTED_FINETUNED_R16_OUTPUT_PATHS = ()
        # Other write paths live in the temp dir so we can confirm what
        # the runner did/did-not write.
        self.safe_metrics = self.tmpdir / "metrics" / "finetuned.json"
        self.safe_outputs = self.tmpdir / "metrics" / "finetuned_outputs.jsonl"
        self.safe_qual = self.tmpdir / "metrics" / "qual.md"
        self.safe_readiness = self.tmpdir / "reports" / "phase4_eval.json"

    def tearDown(self):
        efq.PROTECTED_BASELINE_OUTPUT_PATHS = self._orig_protected
        efq.PROTECTED_FINETUNED_R16_OUTPUT_PATHS = self._orig_r16_protected
        sys.argv = self._orig_argv
        self._td_ctx.cleanup()

    def _invoke_main(self, *cli_args: str) -> int:
        """Invoke ``efq.main()`` with ``sys.argv`` patched to ``cli_args``.

        The repo_root inside ``main()`` is computed as
        ``Path(__file__).resolve().parent.parent`` — i.e. the real repo
        root at ``/home/.../train_pea`` — but our monkey-patched
        ``PROTECTED_BASELINE_OUTPUT_PATHS`` redirects the guardrail
        comparison to our temp sentinel, so the real baseline artifact
        on disk is never at risk.
        """
        sys.argv = ["evaluate_finetuned_qwen.py", *cli_args]
        return efq.main()

    # ------------------------------------------------------------------
    # Direct helper-level tests: pin the extracted production helper
    # in isolation so the contract is regression-proof.
    # ------------------------------------------------------------------
    def test_helper_refuses_to_write_when_readiness_is_protected(self):
        """``_write_blocked_readiness_or_skip`` must return ``False`` and
        leave the protected readiness path UNTOUCHED when the readiness
        path resolves to a protected Phase 2 baseline artifact."""
        sentinel_before = self.sentinel.read_bytes()
        exc = efq.ProtectedBaselinePathError(
            "--readiness=metrics/baseline_qwen2.5-7b.json resolves to a "
            "protected Phase 2 baseline artifact"
        )
        wrote = efq._write_blocked_readiness_or_skip(
            readiness=self.sentinel,
            exc=exc,
            repo_root=self.tmpdir,
        )
        self.assertFalse(wrote, "helper must refuse to write to a protected path")
        # File must NOT have been modified.
        self.assertEqual(self.sentinel.read_bytes(), sentinel_before)
        self.assertEqual(self.sentinel.stat().st_size, len(sentinel_before))

    def test_helper_writes_blocked_payload_when_readiness_is_safe(self):
        """``_write_blocked_readiness_or_skip`` must return ``True`` and
        produce a minimal blocked-readiness JSON when the readiness path
        is safe."""
        exc = efq.ProtectedBaselinePathError(
            "--metrics=metrics/baseline_qwen2.5-7b.json resolves to a "
            "protected Phase 2 baseline artifact"
        )
        wrote = efq._write_blocked_readiness_or_skip(
            readiness=self.safe_readiness,
            exc=exc,
            repo_root=self.tmpdir,
        )
        self.assertTrue(wrote)
        self.assertTrue(self.safe_readiness.exists())
        payload = json.loads(self.safe_readiness.read_text(encoding="utf-8"))
        # Pinned contract for the blocked-readiness payload — automation
        # gates on these keys, so they must always be present.
        self.assertEqual(payload["status"], "blocked")
        self.assertEqual(payload["run_status"], "blocked")
        self.assertFalse(payload["evaluation_completed"])
        self.assertFalse(payload["mvp_accepted"])
        self.assertEqual(payload["mvp_status"], "blocked")
        self.assertIn(str(exc), payload["errors"])

    def test_helper_swallows_write_errors_on_safe_path(self):
        """If the safe readiness path cannot be written (e.g. parent dir
        unwritable), the helper must still return ``True`` so ``main()``
        can exit 2 — a transient filesystem error must not silently mask
        the underlying ``ProtectedBaselinePathError`` or change the exit
        code. We simulate the failure by passing a path whose parent is
        an existing file (so ``mkdir`` fails)."""
        blocker_file = self.tmpdir / "not_a_dir"
        blocker_file.write_text("not a directory", encoding="utf-8")
        bad_readiness = blocker_file / "phase4_eval.json"
        exc = efq.ProtectedBaselinePathError("simulated")
        # Must not raise even though mkdir will fail.
        wrote = efq._write_blocked_readiness_or_skip(
            readiness=bad_readiness,
            exc=exc,
            repo_root=self.tmpdir,
        )
        # Helper still reports "wrote" (the attempt happened) — the
        # failure is logged via stderr and main() exits 2 anyway.
        self.assertTrue(wrote)
        # And of course no file appeared at the bad path.
        self.assertFalse(bad_readiness.exists())

    def test_helper_refuses_for_dot_segment_form_of_protected_readiness(self):
        """``./metrics/baseline_qwen2.5-7b.json`` style variants must also
        be detected as protected by the helper (path normalisation)."""
        # The sentinel lives at self.tmpdir/metrics/baseline_qwen2.5-7b.json
        dot_form = self.tmpdir / "metrics" / "." / "baseline_qwen2.5-7b.json"
        sentinel_before = self.sentinel.read_bytes()
        exc = efq.ProtectedBaselinePathError("simulated")
        wrote = efq._write_blocked_readiness_or_skip(
            readiness=dot_form,
            exc=exc,
            repo_root=self.tmpdir,
        )
        self.assertFalse(wrote)
        self.assertEqual(self.sentinel.read_bytes(), sentinel_before)

    # ------------------------------------------------------------------
    # Production main() tests: drive the EXACT same code path the runner
    # uses, including argparse + the except-branch fallback.
    # ------------------------------------------------------------------
    def test_main_does_not_overwrite_protected_readiness_path(self):
        """CRITICAL regression guardrail at the production level.

        Passing ``--readiness metrics/baseline_qwen2.5-7b.json`` to the
        real ``main()`` must result in:

          * exit code 2,
          * the protected readiness path UNCHANGED on disk,
          * no other write side-effects (no metrics/outputs/qualitative
            files touched — the guardrail trips before any of those code
            paths run).

        The previous bug was that the blocked-readiness fallback in
        ``main()`` would write a minimal payload to ``args.readiness`` —
        silently overwriting the authoritative Phase 2 baseline metrics
        artifact. This test pins the fix.
        """
        sentinel_size_before = self.sentinel.stat().st_size
        sentinel_content_before = self.sentinel.read_text(encoding="utf-8")

        exit_code = self._invoke_main(
            "--readiness", str(self.sentinel),
            "--metrics", str(self.safe_metrics),
            "--outputs", str(self.safe_outputs),
            "--qualitative", str(self.safe_qual),
        )

        # Exit 2 = blocked.
        self.assertEqual(exit_code, 2)

        # The protected readiness path MUST be byte-identical.
        self.assertEqual(self.sentinel.stat().st_size, sentinel_size_before)
        self.assertEqual(
            self.sentinel.read_text(encoding="utf-8"),
            sentinel_content_before,
        )

        # No safe write paths may have been touched either — the
        # guardrail trips before any other write happens.
        self.assertFalse(self.safe_metrics.exists())
        self.assertFalse(self.safe_outputs.exists())
        self.assertFalse(self.safe_qual.exists())

    def test_main_does_not_overwrite_protected_readiness_path_absolute_form(self):
        """Same as above but using an absolute path form for the protected
        readiness target, since the guardrail normalises both sides."""
        # The sentinel is already absolute (self.tmpdir is absolute); the
        # test exercises that the absolute-form path through ``main()``
        # is detected as protected.
        exit_code = self._invoke_main(
            "--readiness", str(self.sentinel.resolve()),
            "--metrics", str(self.safe_metrics),
            "--outputs", str(self.safe_outputs),
            "--qualitative", str(self.safe_qual),
        )
        self.assertEqual(exit_code, 2)
        # Sentinel unchanged.
        self.assertEqual(
            self.sentinel.read_text(encoding="utf-8"),
            self.SENTINEL_PAYLOAD,
        )

    def test_main_writes_blocked_readiness_when_only_metrics_is_protected(self):
        """When ``--metrics`` points at a protected path but ``--readiness``
        is safe, the guardrail still trips (the metrics path is protected)
        but the blocked-readiness fallback CAN write to the safe readiness
        path so automation observes the guardrail.

        This documents the desired current behaviour: refusing to write
        to a protected readiness path is the only restriction on the
        fallback. A safe readiness path always receives a blocked
        payload.
        """
        # Re-set the sentinel to also represent the --metrics path. The
        # protected tuple already covers self.sentinel — we just point
        # --metrics at it.
        exit_code = self._invoke_main(
            "--readiness", str(self.safe_readiness),
            "--metrics", str(self.sentinel),
            "--outputs", str(self.safe_outputs),
            "--qualitative", str(self.safe_qual),
        )
        self.assertEqual(exit_code, 2)

        # The protected sentinel (used here as --metrics) must NOT have
        # been touched by the fallback — the guardrail raised before any
        # metrics write.
        self.assertEqual(
            self.sentinel.read_text(encoding="utf-8"),
            self.SENTINEL_PAYLOAD,
        )

        # The SAFE readiness path WAS written with the blocked payload.
        self.assertTrue(self.safe_readiness.exists())
        payload = json.loads(self.safe_readiness.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "blocked")
        self.assertEqual(payload["run_status"], "blocked")
        self.assertFalse(payload["mvp_accepted"])
        # The error must reference the protected flag that triggered the
        # guardrail so operators can debug from the JSON alone.
        self.assertIn("--metrics", payload["errors"][0])

    def test_main_writes_blocked_readiness_when_outputs_is_protected(self):
        """Same as above but with ``--outputs`` as the protected flag.
        The blocked-readiness fallback still writes to the safe readiness
        path."""
        # Add a second sentinel for the outputs path.
        outputs_sentinel = self.tmpdir / "metrics" / "baseline_qwen2.5-7b_outputs.jsonl"
        outputs_sentinel.write_text(
            '{"phase": "SENTINEL OUTPUTS — DO NOT OVERWRITE"}\n',
            encoding="utf-8",
        )
        efq.PROTECTED_BASELINE_OUTPUT_PATHS = (self.sentinel, outputs_sentinel)

        exit_code = self._invoke_main(
            "--readiness", str(self.safe_readiness),
            "--metrics", str(self.safe_metrics),
            "--outputs", str(outputs_sentinel),
            "--qualitative", str(self.safe_qual),
        )
        self.assertEqual(exit_code, 2)

        # Outputs sentinel unchanged.
        self.assertEqual(
            outputs_sentinel.read_text(encoding="utf-8"),
            '{"phase": "SENTINEL OUTPUTS — DO NOT OVERWRITE"}\n',
        )
        # Safe readiness path written.
        self.assertTrue(self.safe_readiness.exists())
        payload = json.loads(self.safe_readiness.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "blocked")
        self.assertIn("--outputs", payload["errors"][0])

    def test_main_multiple_protected_paths_including_readiness(self):
        """When multiple write paths collide — including the readiness
        path itself — the runner must refuse ALL writes (the blocked
        readiness is the only candidate for a write in the except branch
        and it is refused). The other flags' collisions are reported in
        the error message that goes to stderr (printed by ``main()``)
        but no JSON write can happen."""
        # No read of stderr assertions needed — the important guarantee
        # is that no protected path was touched and the safe paths were
        # not written either.
        sentinel_before = self.sentinel.read_bytes()
        exit_code = self._invoke_main(
            "--readiness", str(self.sentinel),
            "--metrics", str(self.sentinel),
            "--outputs", str(self.safe_outputs),
            "--qualitative", str(self.safe_qual),
        )
        self.assertEqual(exit_code, 2)
        # Sentinel unchanged.
        self.assertEqual(self.sentinel.read_bytes(), sentinel_before)
        # Safe paths unchanged.
        self.assertFalse(self.safe_readiness.exists())
        self.assertFalse(self.safe_outputs.exists())
        self.assertFalse(self.safe_qual.exists())


# ---------------------------------------------------------------------------
# Phase 6 r=16 finetuned-path guardrail
# ---------------------------------------------------------------------------
R16_PROTECTED_METRICS = Path("metrics/finetuned_qwen-protesta-v1.json")
R16_PROTECTED_OUTPUTS = Path("metrics/finetuned_qwen-protesta-v1_outputs.jsonl")
R16_PROTECTED_QUAL = Path("metrics/qualitative_report_finetuned.md")
R16_PROTECTED_READINESS = Path("reports/phase4_eval.json")

R32_METRICS = Path("metrics/finetuned_qwen-protesta-v1-r32.json")
R32_OUTPUTS = Path("metrics/finetuned_qwen-protesta-v1-r32_outputs.jsonl")
R32_QUAL = Path("metrics/qualitative_report_finetuned_r32.md")
R32_READINESS = Path("reports/phase6_r32_eval.json")


class IsProtectedFinetunedR16PathTests(unittest.TestCase):
    """Phase 6 r=16 protection: paths that already hold Phase 3 r=16
    finetuned artifacts must NOT be silently overwritten by a Phase 6
    iteration (e.g. r=32) unless the deliberate override flag is set."""

    def test_r16_metrics_is_protected(self):
        repo_root = Path("/home/agusnieto77/train_pea")
        self.assertTrue(
            efq._is_protected_finetuned_r16_path(R16_PROTECTED_METRICS, repo_root)
        )

    def test_r16_outputs_is_protected(self):
        repo_root = Path("/home/agusnieto77/train_pea")
        self.assertTrue(
            efq._is_protected_finetuned_r16_path(R16_PROTECTED_OUTPUTS, repo_root)
        )

    def test_r16_qualitative_is_protected(self):
        repo_root = Path("/home/agusnieto77/train_pea")
        self.assertTrue(
            efq._is_protected_finetuned_r16_path(R16_PROTECTED_QUAL, repo_root)
        )

    def test_r16_readiness_is_protected(self):
        repo_root = Path("/home/agusnieto77/train_pea")
        self.assertTrue(
            efq._is_protected_finetuned_r16_path(R16_PROTECTED_READINESS, repo_root)
        )

    def test_r32_paths_are_not_protected(self):
        """The Phase 6 r=32 artifact paths MUST be recognised as safe —
        otherwise the new guardrail would refuse to write its own outputs
        and break the iteration."""
        repo_root = Path("/home/agusnieto77/train_pea")
        self.assertFalse(
            efq._is_protected_finetuned_r16_path(R32_METRICS, repo_root)
        )
        self.assertFalse(
            efq._is_protected_finetuned_r16_path(R32_OUTPUTS, repo_root)
        )
        self.assertFalse(
            efq._is_protected_finetuned_r16_path(R32_QUAL, repo_root)
        )
        self.assertFalse(
            efq._is_protected_finetuned_r16_path(R32_READINESS, repo_root)
        )

    def test_baseline_paths_are_not_in_r16_set(self):
        """The baseline guardrail and the r16 guardrail are separate —
        baseline paths must NOT trip the r16 guardrail (they're guarded
        by their own tuple)."""
        repo_root = Path("/home/agusnieto77/train_pea")
        for baseline in (
            Path("metrics/baseline_qwen2.5-7b.json"),
            Path("metrics/baseline_qwen2.5-7b_outputs.jsonl"),
            Path("metrics/qualitative_report.md"),
        ):
            self.assertFalse(
                efq._is_protected_finetuned_r16_path(baseline, repo_root)
            )

    def test_absolute_form_of_r16_path_still_detected(self):
        repo_root = Path("/home/agusnieto77/train_pea")
        abs_form = (repo_root / R16_PROTECTED_METRICS).resolve()
        self.assertTrue(
            efq._is_protected_finetuned_r16_path(abs_form, repo_root)
        )

    def test_dot_segment_form_of_r16_path_still_detected(self):
        """``./metrics/finetuned_qwen-protesta-v1.json`` style variants
        must also trip the r16 guardrail."""
        repo_root = Path("/home/agusnieto77/train_pea")
        dot_form = Path("./metrics/finetuned_qwen-protesta-v1.json")
        self.assertTrue(
            efq._is_protected_finetuned_r16_path(dot_form, repo_root)
        )


class AssertPhase6R16OutputsSafeTests(unittest.TestCase):
    """Phase 6 r=16 guardrail: r=32 eval invocation must fail-fast when
    its output paths collide with the r=16 artifacts unless the
    ``--allow-r16-output-overwrite`` flag is set."""

    def _ok_paths(self) -> dict[str, Path]:
        return {
            "metrics": R32_METRICS,
            "outputs": R32_OUTPUTS,
            "qualitative": R32_QUAL,
            "readiness": R32_READINESS,
        }

    def test_r32_paths_pass_without_override(self):
        """Default r=32 invocation (separate output paths) must NOT trip
        the r16 guardrail."""
        efq.assert_phase6_r16_outputs_safe(
            metrics=self._ok_paths()["metrics"],
            outputs=self._ok_paths()["outputs"],
            qualitative=self._ok_paths()["qualitative"],
            readiness=self._ok_paths()["readiness"],
            repo_root=Path("/home/agusnieto77/train_pea"),
            allow_r16_overwrite=False,
        )

    def test_r16_metrics_path_blocks_with_override_off(self):
        with self.assertRaises(efq.ProtectedFinetunedR16PathError) as ctx:
            efq.assert_phase6_r16_outputs_safe(
                metrics=R16_PROTECTED_METRICS,
                outputs=self._ok_paths()["outputs"],
                qualitative=self._ok_paths()["qualitative"],
                readiness=self._ok_paths()["readiness"],
                repo_root=Path("/home/agusnieto77/train_pea"),
                allow_r16_overwrite=False,
            )
        self.assertIn("--metrics", str(ctx.exception))
        self.assertIn("Phase 3 r=16 finetuned artifact", str(ctx.exception))

    def test_r16_outputs_path_blocks_with_override_off(self):
        with self.assertRaises(efq.ProtectedFinetunedR16PathError):
            efq.assert_phase6_r16_outputs_safe(
                metrics=self._ok_paths()["metrics"],
                outputs=R16_PROTECTED_OUTPUTS,
                qualitative=self._ok_paths()["qualitative"],
                readiness=self._ok_paths()["readiness"],
                repo_root=Path("/home/agusnieto77/train_pea"),
                allow_r16_overwrite=False,
            )

    def test_r16_qualitative_path_blocks_with_override_off(self):
        with self.assertRaises(efq.ProtectedFinetunedR16PathError):
            efq.assert_phase6_r16_outputs_safe(
                metrics=self._ok_paths()["metrics"],
                outputs=self._ok_paths()["outputs"],
                qualitative=R16_PROTECTED_QUAL,
                readiness=self._ok_paths()["readiness"],
                repo_root=Path("/home/agusnieto77/train_pea"),
                allow_r16_overwrite=False,
            )

    def test_r16_readiness_path_blocks_with_override_off(self):
        """``--readiness reports/phase4_eval.json`` (the r=16 readiness
        artifact) must trip the r16 guardrail so a Phase 6 invocation
        cannot overwrite the r=16 readiness report."""
        with self.assertRaises(efq.ProtectedFinetunedR16PathError) as ctx:
            efq.assert_phase6_r16_outputs_safe(
                metrics=self._ok_paths()["metrics"],
                outputs=self._ok_paths()["outputs"],
                qualitative=self._ok_paths()["qualitative"],
                readiness=R16_PROTECTED_READINESS,
                repo_root=Path("/home/agusnieto77/train_pea"),
                allow_r16_overwrite=False,
            )
        self.assertIn("--readiness", str(ctx.exception))

    def test_override_flag_bypasses_r16_guardrail(self):
        """``--allow-r16-output-overwrite`` must be the ONLY way to bypass
        the r16 guardrail — the same contract as the baseline guardrail."""
        # No exception raised when the override is set.
        efq.assert_phase6_r16_outputs_safe(
            metrics=R16_PROTECTED_METRICS,
            outputs=R16_PROTECTED_OUTPUTS,
            qualitative=R16_PROTECTED_QUAL,
            readiness=R16_PROTECTED_READINESS,
            repo_root=Path("/home/agusnieto77/train_pea"),
            allow_r16_overwrite=True,
        )

    def test_all_four_r16_paths_in_single_error(self):
        """When multiple write paths collide with r=16, the error must
        list every collision so operators can debug from stderr alone."""
        with self.assertRaises(efq.ProtectedFinetunedR16PathError) as ctx:
            efq.assert_phase6_r16_outputs_safe(
                metrics=R16_PROTECTED_METRICS,
                outputs=R16_PROTECTED_OUTPUTS,
                qualitative=R16_PROTECTED_QUAL,
                readiness=R16_PROTECTED_READINESS,
                repo_root=Path("/home/agusnieto77/train_pea"),
                allow_r16_overwrite=False,
            )
        msg = str(ctx.exception)
        for label in ("--metrics", "--outputs", "--qualitative", "--readiness"):
            self.assertIn(label, msg)

    def test_absolute_r16_path_form_still_blocks(self):
        """Passing the absolute form of an r=16 path must still trip
        the guardrail (path normalisation covers this)."""
        repo_root = Path("/home/agusnieto77/train_pea")
        abs_metrics = (repo_root / R16_PROTECTED_METRICS).resolve()
        with self.assertRaises(efq.ProtectedFinetunedR16PathError):
            efq.assert_phase6_r16_outputs_safe(
                metrics=abs_metrics,
                outputs=self._ok_paths()["outputs"],
                qualitative=self._ok_paths()["qualitative"],
                readiness=self._ok_paths()["readiness"],
                repo_root=repo_root,
                allow_r16_overwrite=False,
            )

    def test_dot_segment_form_of_r16_readiness_still_blocks(self):
        repo_root = Path("/home/agusnieto77/train_pea")
        with self.assertRaises(efq.ProtectedFinetunedR16PathError):
            efq.assert_phase6_r16_outputs_safe(
                metrics=self._ok_paths()["metrics"],
                outputs=self._ok_paths()["outputs"],
                qualitative=self._ok_paths()["qualitative"],
                readiness=Path("./reports/phase4_eval.json"),
                repo_root=repo_root,
                allow_r16_overwrite=False,
            )


# ---------------------------------------------------------------------------
# Production main() — blocked-readiness fallback must also refuse to write
# when the readiness path is a protected Phase 3 r=16 finetuned artifact.
# ---------------------------------------------------------------------------
class MainBlockedReadinessFallbackForR16Tests(unittest.TestCase):
    """Companion to :class:`MainBlockedReadinessFallbackTests` covering the
    Phase 6 r=16 protection in :func:`_write_blocked_readiness_or_skip`.

    Regression: a Phase 6 invocation (e.g. r=32) with ``--readiness
    reports/phase4_eval.json`` USED TO silently overwrite the r=16
    readiness artifact via the blocked-readiness fallback path inside
    ``main()``. The guardrail would trip (``ProtectedFinetunedR16PathError``),
    but the fallback helper only checked baseline protection, so it would
    write a minimal blocked payload to ``reports/phase4_eval.json`` —
    silently corrupting the authoritative r=16 readiness report.

    Fix: :func:`_write_blocked_readiness_or_skip` now also refuses r=16
    protected paths (via :func:`_is_protected_readiness_path`). These tests
    pin the fix at both the helper level and the production ``main()``
    level so a future refactor cannot reintroduce the bug.
    """

    SENTINEL_PAYLOAD = (
        '{"phase": "SENTINEL R16 READINESS — DO NOT OVERWRITE", "status": "untouched"}\n'
    )

    def setUp(self):
        self._td_ctx = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._td_ctx.name)
        # r=16 readiness sentinel in the temp dir. The protected r=16 tuple
        # already names ``reports/phase4_eval.json``; we point the sentinel
        # at the SAME relative path inside the temp repo-root so the
        # helper, when invoked with repo_root=self.tmpdir, resolves the
        # protected r=16 path against self.tmpdir and matches our
        # sentinel. This is the same trick used by the Phase 4 baseline
        # tests (see ``MainBlockedReadinessFallbackTests.setUp``).
        self.sentinel = self.tmpdir / "reports" / "phase4_eval.json"
        self.sentinel.parent.mkdir(parents=True, exist_ok=True)
        self.sentinel.write_text(self.SENTINEL_PAYLOAD, encoding="utf-8")
        # Module state capture + restore.
        self._orig_baseline = efq.PROTECTED_BASELINE_OUTPUT_PATHS
        self._orig_r16 = efq.PROTECTED_FINETUNED_R16_OUTPUT_PATHS
        self._orig_argv = sys.argv
        # Empty baseline tuple so the baseline guardrail does NOT also trip
        # (this test class focuses on r=16 protection only). Restore in
        # tearDown.
        efq.PROTECTED_BASELINE_OUTPUT_PATHS = ()
        # Make the r=16 guardrail point at our temp sentinel. We add an
        # ABSOLUTE path so the helper's normalisation step resolves it to
        # the same physical target as ``self.sentinel`` regardless of
        # whether ``main()`` resolves the protected tuple against the
        # REAL repo root or against our temp dir.
        efq.PROTECTED_FINETUNED_R16_OUTPUT_PATHS = (self.sentinel,)
        # Other write paths in the temp dir.
        self.safe_metrics = self.tmpdir / "metrics" / "r32_metrics.json"
        self.safe_outputs = self.tmpdir / "metrics" / "r32_outputs.jsonl"
        self.safe_qual = self.tmpdir / "metrics" / "r32_qual.md"
        # Safe readiness: r=32 Phase 6 location — NOT in any protected set.
        self.safe_readiness = self.tmpdir / "reports" / "phase6_r32_eval.json"

    def tearDown(self):
        efq.PROTECTED_BASELINE_OUTPUT_PATHS = self._orig_baseline
        efq.PROTECTED_FINETUNED_R16_OUTPUT_PATHS = self._orig_r16
        sys.argv = self._orig_argv
        self._td_ctx.cleanup()

    def _invoke_main(self, *cli_args: str) -> int:
        sys.argv = ["evaluate_finetuned_qwen.py", *cli_args]
        return efq.main()

    # ------------------------------------------------------------------
    # Helper-level tests (drive the extracted production helper directly).
    # ------------------------------------------------------------------
    def test_helper_refuses_to_write_when_readiness_is_r16_protected(self):
        """``_write_blocked_readiness_or_skip`` must refuse to write when
        the readiness path resolves to a protected Phase 3 r=16 finetuned
        artifact (``reports/phase4_eval.json``). This is the structural
        fix for the CRITICAL Phase 6 review blocker."""
        sentinel_before = self.sentinel.read_bytes()
        exc = efq.ProtectedFinetunedR16PathError(
            "--readiness=reports/phase4_eval.json resolves to a protected "
            "Phase 3 r=16 finetuned artifact"
        )
        wrote = efq._write_blocked_readiness_or_skip(
            readiness=self.sentinel,
            exc=exc,
            repo_root=self.tmpdir,
        )
        self.assertFalse(wrote, "helper must refuse to write to a r=16 protected path")
        self.assertEqual(self.sentinel.read_bytes(), sentinel_before)
        self.assertEqual(self.sentinel.stat().st_size, len(sentinel_before))

    def test_helper_writes_blocked_payload_when_readiness_is_r32_safe(self):
        """Companion: when the readiness path is safe (r=32 default), the
        helper still writes a blocked payload. Confirms the new r16 check
        does NOT regress the safe-path behaviour."""
        exc = efq.ProtectedFinetunedR16PathError(
            "--metrics=... resolves to a protected Phase 3 r=16 finetuned artifact"
        )
        wrote = efq._write_blocked_readiness_or_skip(
            readiness=self.safe_readiness,
            exc=exc,
            repo_root=self.tmpdir,
        )
        self.assertTrue(wrote)
        self.assertTrue(self.safe_readiness.exists())
        payload = json.loads(self.safe_readiness.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "blocked")
        self.assertEqual(payload["run_status"], "blocked")
        self.assertFalse(payload["mvp_accepted"])
        self.assertIn(str(exc), payload["errors"])

    def test_helper_classification_returns_r16_for_r16_path(self):
        """``_is_protected_readiness_path`` must report ``"r16"`` (not
        ``None`` and not ``"baseline"``) for a protected r=16 path."""
        result = efq._is_protected_readiness_path(self.sentinel, self.tmpdir)
        self.assertEqual(result, "r16")

    def test_helper_classification_returns_none_for_safe_path(self):
        """``_is_protected_readiness_path`` must report ``None`` for an
        unprotected readiness path."""
        result = efq._is_protected_readiness_path(self.safe_readiness, self.tmpdir)
        self.assertIsNone(result)

    # ------------------------------------------------------------------
    # Production main() tests — prove the bug is fixed end-to-end.
    # ------------------------------------------------------------------
    def test_main_does_not_overwrite_r16_readiness_via_blocked_fallback(self):
        """CRITICAL Phase 6 review regression guardrail.

        Invoking the real ``main()`` with ``--readiness
        reports/phase4_eval.json`` (the r=16 readiness artifact) MUST
        result in:

          * exit code 2,
          * the r=16 readiness path UNCHANGED on disk,
          * no other write side-effects.

        Previous behaviour: the blocked-readiness fallback in ``main()``
        happily wrote a minimal payload to ``args.readiness`` even when it
        pointed at the r=16 artifact, silently overwriting the
        authoritative r=16 readiness report. This test pins the fix.
        """
        sentinel_size_before = self.sentinel.stat().st_size
        sentinel_content_before = self.sentinel.read_text(encoding="utf-8")

        exit_code = self._invoke_main(
            "--readiness", str(self.sentinel),
            "--metrics", str(self.safe_metrics),
            "--outputs", str(self.safe_outputs),
            "--qualitative", str(self.safe_qual),
        )

        self.assertEqual(exit_code, 2)
        self.assertEqual(self.sentinel.stat().st_size, sentinel_size_before)
        self.assertEqual(
            self.sentinel.read_text(encoding="utf-8"),
            sentinel_content_before,
        )
        # No safe write paths were touched.
        self.assertFalse(self.safe_metrics.exists())
        self.assertFalse(self.safe_outputs.exists())
        self.assertFalse(self.safe_qual.exists())

    def test_main_r32_default_invocation_can_write_blocked_to_safe_readiness(self):
        """Companion: a Phase 6 r=32 invocation with the SAFE default
        readiness path (``reports/phase6_r32_eval.json``) must still write
        the blocked-readiness payload so automation observes the
        guardrail. This proves the new r=16 check does not regress the
        safe-path flow.

        We trigger the guardrail via ``--metrics`` pointing at the r=16
        sentinel (so the r=16 guardrail trips), while ``--readiness`` is
        a safe r=32 location so the blocked-readiness fallback still has
        somewhere to write."""
        exit_code = self._invoke_main(
            "--readiness", str(self.safe_readiness),
            "--metrics", str(self.sentinel),  # r=16 protected — trips the guardrail
            "--outputs", str(self.safe_outputs),
            "--qualitative", str(self.safe_qual),
        )
        self.assertEqual(exit_code, 2)
        # Safe readiness written with blocked payload.
        self.assertTrue(self.safe_readiness.exists())
        payload = json.loads(self.safe_readiness.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "blocked")
        self.assertIn("--metrics", payload["errors"][0])


# ---------------------------------------------------------------------------
# Qualitative report — path / name / rank metadata
# ---------------------------------------------------------------------------
class BuildQualitativeReportMetadataTests(unittest.TestCase):
    """Pin the structural fix for the WARNING Phase 6 review blocker.

    The ``build_qualitative_report`` function used to hard-code:

      * ``max_lora_rank=16`` from ``DEFAULT_MAX_LORA_RANK`` (always 16),
      * ``metrics/finetuned_qwen-protesta-v1.json`` as the headline path,
      * ``metrics/baseline_qwen2.5-7b.json`` and
        ``reports/phase4_eval.json`` as the source paths,
      * ``35 examples from data/chat_formatted/eval.jsonl`` as the eval-set
        size and path,
      * r=16-specific narrative in sections 9 and 10 (e.g. ``+0.4286``
        for ``tiene_eventos_protesta``, ``0.0384 → 0.3400`` for
        categorical_accuracy, etc.).

    These tests feed the function with both r=16 and r=32 metrics
    dictionaries and assert that EVERY path / name / rank reference in
    the rendered markdown matches the caller-supplied inputs and the
    actual metrics, NOT hard-coded defaults. This is the structural
    guard against the regression returning when future ranks are added.
    """

    @staticmethod
    def _minimal_metrics_report(
        *,
        max_lora_rank: int,
        lora_name: str,
        adapter_resolved: str,
        eval_input: str,
        examples_total: int,
        schema_path: str,
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        delta: dict[str, Any] | None = None,
        cat_agg_acc: float = 0.4189,
        tiene_acc: float = 0.7714,
        f1: float = 0.5350,
        schema_validity: float = 1.0,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Build a minimal metrics_report + per_example list good enough
        to exercise ``build_qualitative_report``. The per_example records
        here are simple f=stop, schema-valid, parse-valid examples so the
        generator can render the report without raising."""
        cat_per_path: dict[str, dict[str, Any]] = {
            "extraccion.tiene_eventos_protesta": {
                "tp": 26,
                "tn": 1,
                "fp": 7,
                "fn": 1,
                "accuracy": tiene_acc,
                "support": 35,
            },
            "incidentes.represion.presencia": {
                "tp": 40,
                "tn": 0,
                "fp": 16,
                "fn": 7,
                "accuracy": 0.6349,
                "support": 63,
            },
            "delimitacion.criterio_delimitacion": {
                "tp": 17,
                "tn": 0,
                "fp": 39,
                "fn": 30,
                "accuracy": 0.1977,
                "support": 86,
            },
        }
        # Aggregate is required for the report; values match the r=32 case
        # by default so sections that reference it match.
        cat_per_path["__aggregate__"] = {
            "tp": int(cat_agg_acc * 1000),
            "tn": int(cat_agg_acc * 800),
            "fp": 500,
            "fn": 500,
            "accuracy": cat_agg_acc,
            "support": int((cat_agg_acc * 1800) + 500),
        }
        tiene = cat_per_path["extraccion.tiene_eventos_protesta"]
        m: dict[str, Any] = {
            "schema_validity": schema_validity,
            "parse_validity": schema_validity,
            "categorical_accuracy": {
                "tiene_eventos_protesta": tiene,
                "per_path": cat_per_path,
            },
            "f1_global": {
                "f1": f1,
                "precision": 0.5001,
                "recall": 0.5752,
            },
            "field_recall": {
                "gold_leaves": 3239,
                "exact_match_count": 1673,
                "exact_match_recall": 0.5165,
                "non_empty_recovery_count": 2428,
                "non_empty_recovery_recall": 0.7496,
                "null_or_empty_in_gold": 0,
            },
        }
        per_example: list[dict[str, Any]] = []
        for i in range(examples_total):
            per_example.append({
                "index": i,
                "line_no": i + 1,
                "nota_id": f"IMG_TEST_{i:03d}",
                "prompt_tokens": 3000,
                "max_tokens": 8192,
                "output_tokens": 1000,
                "finish_reason": "stop",
                "elapsed_seconds": 1.0,
                "parse_valid": True,
                "parse_error": None,
                "schema_valid": True,
                "schema_error_count": 0,
                "schema_errors": [],
                "f1_vs_gold": {"tp": 10, "fp": 5, "fn": 3, "gold_leaves": 18, "pred_leaves": 15},
                "field_recall_vs_gold": {"gold_leaves": 18, "exact_match": 10, "non_empty_recovery": 14, "null_or_empty_in_gold": 0},
                "categorical_accuracy_vs_gold": {},
                "raw_text": "{}",
                "parsed": {
                    "nota": {
                        "nota_id": f"IMG_TEST_{i:03d}",
                        "fecha_publicacion": "13/09/1989",
                    },
                },
            })
        metrics_report: dict[str, Any] = {
            "phase": "Phase 4 — Post-training evaluation",
            "status": "pass",
            "checked_at": "2026-06-27T06:30:11-0300",
            "model": model,
            "adapter_requested": adapter_resolved,
            "eval_input": eval_input,
            "schema": {
                "path": schema_path,
                "title": "ExtraccionEventosProtestaDesdeNota-MVS",
                "required": ["schema_version", "nota", "extraccion"],
                "additionalProperties_root": False,
            },
            "max_seq_length": 20480,
            "max_tokens_cap": 8192,
            "max_lora_rank": max_lora_rank,
            "lora_name": lora_name,
            "lora_int_id": 1,
            "baseline_available": delta is not None,
            "baseline_metrics_path": "metrics/baseline_qwen2.5-7b.json",
            "examples_total": examples_total,
            "examples_run": examples_total,
            "examples_blocked_pre_inference": 0,
            "counts": {
                "examples_total": examples_total,
                "examples_run": examples_total,
                "examples_blocked_pre_inference": 0,
                "parse_valid": examples_total,
                "schema_valid": examples_total,
                "finish_reason_length": 0,
            },
            "metrics": m,
            "timings": {
                "total_seconds": 540.8,
                "mean_per_example_seconds": 15.5,
                "prompt_tokens_min": 2440,
                "prompt_tokens_max": 18206,
                "prompt_tokens_mean": 3571.0,
                "output_tokens_min": 102,
                "output_tokens_max": 2920,
                "output_tokens_mean": 1189.5,
                "output_tokens_total": int(1189.5 * examples_total),
            },
        }
        if delta is not None:
            metrics_report["delta_vs_baseline"] = delta
        return metrics_report, per_example

    @staticmethod
    def _adapter_info(resolved: str, lora_name: str, rank: int) -> dict[str, Any]:
        return {
            "requested": resolved,
            "resolved": resolved,
            "resolved_reason": "adapter_model.safetensors at requested root",
            "sha1": "deadbeef" * 5,
            "size_bytes": 161533584,
            "lora_name": lora_name,
            "rank": rank,
        }

    def test_r32_report_uses_max_lora_rank_32_not_default_16(self):
        """Structural fix: r=32 reports must show ``max_lora_rank=32``,
        not the hard-coded ``DEFAULT_MAX_LORA_RANK=16``."""
        delta: dict[str, Any] = {
            "schema_validity": {"baseline": 1.0, "finetuned": 1.0, "delta": 0.0},
            "f1_global": {"baseline_f1": 0.0971, "finetuned_f1": 0.535, "delta_f1": 0.4379},
            "categorical_accuracy_aggregate": {"baseline": 0.0384, "finetuned": 0.4189, "delta": 0.3805},
            "tiene_eventos_protesta_accuracy": {"baseline": 0.2857, "finetuned": 0.7714, "delta": 0.4857},
            "field_recall_exact": {"baseline": 0.054, "finetuned": 0.5165, "delta": 0.4625},
            "field_recall_non_empty": {"baseline": 0.1692, "finetuned": 0.7496, "delta": 0.5804},
        }
        metrics_report, per_example = self._minimal_metrics_report(
            max_lora_rank=32,
            lora_name="qwen_protesta_v1_r32",
            adapter_resolved="checkpoints/qwen-protesta-v1-r32",
            eval_input="data/chat_formatted/eval.jsonl",
            examples_total=35,
            schema_path="esquema_eventos_protesta_entrenamiento_MVS.json",
            delta=delta,
        )
        adapter_info = self._adapter_info(
            "checkpoints/qwen-protesta-v1-r32", "qwen_protesta_v1_r32", 32
        )
        md = efq.build_qualitative_report(
            metrics_report=metrics_report,
            per_example=per_example,
            delta=delta,
            adapter_info=adapter_info,
            metrics_path=Path("metrics/finetuned_qwen-protesta-v1-r32.json"),
            outputs_path=Path("metrics/finetuned_qwen-protesta-v1-r32_outputs.jsonl"),
            baseline_metrics_path=Path("metrics/baseline_qwen2.5-7b.json"),
            qualitative_path=Path("metrics/qualitative_report_finetuned_r32.md"),
            readiness_path=Path("reports/phase6_r32_eval.json"),
            plan_reference="PLAN_ENTRENAMIENTO_QWEN.md §Fase 6",
            phase_label="Phase 6 (r=32)",
            runner_script="scripts/evaluate_finetuned_qwen.py",
        )
        # max_lora_rank must be the value from metrics_report, NOT 16.
        self.assertIn("max_lora_rank=32", md)
        self.assertNotIn("max_lora_rank=16", md)
        # Adapter path/label must reflect r=32.
        self.assertIn("checkpoints/qwen-protesta-v1-r32", md)
        self.assertIn("qwen_protesta_v1_r32", md)
        # Headline metrics reference must point at the r=32 metrics file.
        self.assertIn("metrics/finetuned_qwen-protesta-v1-r32.json", md)
        # Sources must reference the r=32 readiness + outputs + qualitative paths.
        self.assertIn("metrics/finetuned_qwen-protesta-v1-r32_outputs.jsonl", md)
        self.assertIn("reports/phase6_r32_eval.json", md)
        self.assertIn("metrics/qualitative_report_finetuned_r32.md", md)
        # Phase label must be dynamic.
        self.assertIn("# Phase 6 (r=32) — Qualitative Report", md)
        # Plan reference must be dynamic.
        self.assertIn("PLAN_ENTRENAMIENTO_QWEN.md §Fase 6", md)
        # Eval input line must be dynamic.
        self.assertIn("35 examples from `data/chat_formatted/eval.jsonl`", md)

    def test_r16_report_uses_max_lora_rank_16(self):
        """Companion: r=16 reports must show ``max_lora_rank=16``."""
        metrics_report, per_example = self._minimal_metrics_report(
            max_lora_rank=16,
            lora_name="qwen_protesta_v1",
            adapter_resolved="checkpoints/qwen-protesta-v1",
            eval_input="data/chat_formatted/eval.jsonl",
            examples_total=35,
            schema_path="esquema_eventos_protesta_entrenamiento_MVS.json",
        )
        adapter_info = self._adapter_info(
            "checkpoints/qwen-protesta-v1", "qwen_protesta_v1", 16
        )
        md = efq.build_qualitative_report(
            metrics_report=metrics_report,
            per_example=per_example,
            delta=None,
            adapter_info=adapter_info,
            metrics_path=Path("metrics/finetuned_qwen-protesta-v1.json"),
            outputs_path=Path("metrics/finetuned_qwen-protesta-v1_outputs.jsonl"),
            baseline_metrics_path=Path("metrics/baseline_qwen2.5-7b.json"),
            qualitative_path=Path("metrics/qualitative_report_finetuned.md"),
            readiness_path=Path("reports/phase4_eval.json"),
            plan_reference="PLAN_ENTRENAMIENTO_QWEN.md §Fase 4",
            phase_label="Phase 4",
            runner_script="scripts/evaluate_finetuned_qwen.py",
        )
        self.assertIn("max_lora_rank=16", md)
        self.assertIn("checkpoints/qwen-protesta-v1", md)
        self.assertIn("metrics/finetuned_qwen-protesta-v1.json", md)
        self.assertIn("reports/phase4_eval.json", md)
        self.assertIn("metrics/qualitative_report_finetuned.md", md)
        self.assertIn("# Phase 4 — Qualitative Report", md)
        self.assertIn("PLAN_ENTRENAMIENTO_QWEN.md §Fase 4", md)

    def test_r32_report_narrative_uses_actual_deltas_not_r16_values(self):
        """Section 9/10 narrative MUST use the actual run deltas — not
        the previous r=16 hardcoded values. Pin against the exact strings
        that previously regressed."""
        delta: dict[str, Any] = {
            "schema_validity": {"baseline": 1.0, "finetuned": 1.0, "delta": 0.0},
            "f1_global": {"baseline_f1": 0.0971, "finetuned_f1": 0.535, "delta_f1": 0.4379},
            "categorical_accuracy_aggregate": {"baseline": 0.0384, "finetuned": 0.4189, "delta": 0.3805},
            "tiene_eventos_protesta_accuracy": {"baseline": 0.2857, "finetuned": 0.7714, "delta": 0.4857},
            "field_recall_exact": {"baseline": 0.054, "finetuned": 0.5165, "delta": 0.4625},
            "field_recall_non_empty": {"baseline": 0.1692, "finetuned": 0.7496, "delta": 0.5804},
        }
        metrics_report, per_example = self._minimal_metrics_report(
            max_lora_rank=32,
            lora_name="qwen_protesta_v1_r32",
            adapter_resolved="checkpoints/qwen-protesta-v1-r32",
            eval_input="data/chat_formatted/eval.jsonl",
            examples_total=35,
            schema_path="esquema_eventos_protesta_entrenamiento_MVS.json",
            delta=delta,
            cat_agg_acc=0.4189,
            tiene_acc=0.7714,
            f1=0.5350,
        )
        adapter_info = self._adapter_info(
            "checkpoints/qwen-protesta-v1-r32", "qwen_protesta_v1_r32", 32
        )
        md = efq.build_qualitative_report(
            metrics_report=metrics_report,
            per_example=per_example,
            delta=delta,
            adapter_info=adapter_info,
            metrics_path=Path("metrics/finetuned_qwen-protesta-v1-r32.json"),
            outputs_path=Path("metrics/finetuned_qwen-protesta-v1-r32_outputs.jsonl"),
            baseline_metrics_path=Path("metrics/baseline_qwen2.5-7b.json"),
            qualitative_path=Path("metrics/qualitative_report_finetuned_r32.md"),
            readiness_path=Path("reports/phase6_r32_eval.json"),
            plan_reference="PLAN_ENTRENAMIENTO_QWEN.md §Fase 6",
            phase_label="Phase 6 (r=32)",
        )
        # Actual r=32 deltas must appear.
        self.assertIn("0.4189", md)
        self.assertIn("0.535", md)
        self.assertIn("0.7714", md)
        # The previous r=16-only deltas must NOT appear.
        self.assertNotIn("+0.4286", md)
        self.assertNotIn("0.0384 → 0.3400", md)
        self.assertNotIn("0.054 → 0.4134", md)
        self.assertNotIn("0.169 → 0.634", md)
        # Adapter path must reflect r=32, not the previous "qwen-protesta-v1" alone.
        self.assertIn("checkpoints/qwen-protesta-v1-r32", md)

    def test_report_uses_caller_supplied_paths_in_sources_section(self):
        """The Sources section must echo caller-supplied paths verbatim
        so a future iteration cannot regress to the previous hard-coded
        ``finetuned_qwen-protesta-v1.json`` / ``phase4_eval.json``."""
        delta: dict[str, Any] = {
            "schema_validity": {"baseline": 1.0, "finetuned": 1.0, "delta": 0.0},
            "f1_global": {"baseline_f1": 0.0971, "finetuned_f1": 0.535, "delta_f1": 0.4379},
            "categorical_accuracy_aggregate": {"baseline": 0.0384, "finetuned": 0.4189, "delta": 0.3805},
            "tiene_eventos_protesta_accuracy": {"baseline": 0.2857, "finetuned": 0.7714, "delta": 0.4857},
            "field_recall_exact": {"baseline": 0.054, "finetuned": 0.5165, "delta": 0.4625},
            "field_recall_non_empty": {"baseline": 0.1692, "finetuned": 0.7496, "delta": 0.5804},
        }
        metrics_report, per_example = self._minimal_metrics_report(
            max_lora_rank=32,
            lora_name="qwen_protesta_v1_r32",
            adapter_resolved="checkpoints/qwen-protesta-v1-r32",
            eval_input="data/chat_formatted/eval.jsonl",
            examples_total=35,
            schema_path="esquema_eventos_protesta_entrenamiento_MVS.json",
            delta=delta,
        )
        adapter_info = self._adapter_info(
            "checkpoints/qwen-protesta-v1-r32", "qwen_protesta_v1_r32", 32
        )
        md = efq.build_qualitative_report(
            metrics_report=metrics_report,
            per_example=per_example,
            delta=delta,
            adapter_info=adapter_info,
            metrics_path=Path("custom/path/r32_metrics.json"),
            outputs_path=Path("custom/path/r32_outputs.jsonl"),
            baseline_metrics_path=Path("custom/path/baseline.json"),
            qualitative_path=Path("custom/path/r32_qual.md"),
            readiness_path=Path("custom/path/r32_readiness.json"),
            plan_reference="PLAN_ENTRENAMIENTO_QWEN.md §Fase 6",
            phase_label="Phase 6 (r=32)",
            runner_script="scripts/custom_runner.py",
        )
        for expected in (
            "custom/path/r32_metrics.json",
            "custom/path/r32_outputs.jsonl",
            "custom/path/baseline.json",
            "custom/path/r32_qual.md",
            "custom/path/r32_readiness.json",
            "scripts/custom_runner.py",
            "PLAN_ENTRENAMIENTO_QWEN.md §Fase 6",
        ):
            self.assertIn(expected, md, f"sources section missing: {expected}")

    def test_adapter_label_uses_lora_name(self):
        """The report header must use the ``lora_name`` from
        ``adapter_info`` (or the resolved path's last segment) so the
        title is correct for both r=16 and r=32."""
        self.assertEqual(
            efq._adapter_label({"lora_name": "qwen_protesta_v1_r32"}),
            "qwen_protesta_v1_r32",
        )
        self.assertEqual(
            efq._adapter_label({"lora_name": "qwen_protesta_v1"}),
            "qwen_protesta_v1",
        )
        # Fallback to resolved path's last segment.
        self.assertEqual(
            efq._adapter_label({"resolved": "checkpoints/qwen-protesta-v1-r32"}),
            "qwen-protesta-v1-r32",
        )
        # Empty lora_name falls back to resolved.
        self.assertEqual(
            efq._adapter_label({"lora_name": "  ", "resolved": "checkpoints/x"}),
            "x",
        )
        # Ultimate fallback to "adapter".
        self.assertEqual(efq._adapter_label({}), "adapter")


if __name__ == "__main__":
    unittest.main(verbosity=2)