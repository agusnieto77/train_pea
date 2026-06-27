#!/usr/bin/env python3
"""Unit tests for ``scripts/baseline_qwen_full.py`` guardrails.

These tests do NOT require vLLM, torch, the model weights, or any GPU.
They exercise only the pure helpers extracted at module level in
``baseline_qwen_full``:

  * ``_partial_sibling``             — sibling-path naming for rerouted artifacts.
  * ``resolve_limited_run_paths``   — limited-run path auto-route guardrail.
  * ``classify_run_status``         — pass gate that protects Phase 3.
  * ``merge_readiness_for_partial`` — partial-write helper that must never
                                       overwrite top-level readiness / full_baseline.

The tests are deliberately stdlib-only (``unittest`` + ``pathlib``) so they
can run anywhere Python is installed. They are runnable via either:

    .venv/bin/python -m unittest scripts.test_baseline_qwen_full_paths
    .venv/bin/python scripts/test_baseline_qwen_full_paths.py

When run via ``python -m unittest scripts.test_baseline_qwen_full_paths`` from
the project root, the ``sys.path`` tweak at the top of this file makes
``baseline_qwen_full`` importable.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Make `baseline_qwen_full` importable when running from the project root via
# `python -m unittest scripts.test_baseline_qwen_full_paths`. When running
# directly as a script (e.g. `python scripts/test_baseline_qwen_full_paths.py`)
# the current directory (`scripts/`) is already on sys.path so this is a no-op.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import baseline_qwen_full as bfq  # noqa: E402


# ---------------------------------------------------------------------------
# Constants used across tests
# ---------------------------------------------------------------------------
OFFICIAL_METRICS = bfq.OFFICIAL_METRICS_PATH  # metrics/baseline_qwen2.5-7b.json
OFFICIAL_OUTPUTS = bfq.OFFICIAL_OUTPUTS_PATH  # metrics/baseline_qwen2.5-7b_outputs.jsonl
OFFICIAL_QUAL = bfq.OFFICIAL_QUAL_PATH        # metrics/qualitative_report.md


# ---------------------------------------------------------------------------
# _partial_sibling
# ---------------------------------------------------------------------------
class PartialSiblingTests(unittest.TestCase):
    """``_partial_sibling`` produces a sibling path with `_partial` inserted
    before the suffix. It must NEVER collide with the original."""

    def test_metrics_path(self):
        self.assertEqual(
            bfq._partial_sibling(Path("metrics/baseline_qwen2.5-7b.json")),
            Path("metrics/baseline_qwen2.5-7b_partial.json"),
        )

    def test_outputs_jsonl_path(self):
        self.assertEqual(
            bfq._partial_sibling(
                Path("metrics/baseline_qwen2.5-7b_outputs.jsonl")
            ),
            Path("metrics/baseline_qwen2.5-7b_outputs_partial.jsonl"),
        )

    def test_qualitative_md_path(self):
        self.assertEqual(
            bfq._partial_sibling(Path("metrics/qualitative_report.md")),
            Path("metrics/qualitative_report_partial.md"),
        )

    def test_partial_sibling_never_equals_original(self):
        for p in [OFFICIAL_METRICS, OFFICIAL_OUTPUTS, OFFICIAL_QUAL]:
            self.assertNotEqual(bfq._partial_sibling(p), p)
            self.assertNotEqual(
                bfq._partial_sibling(p), p.with_suffix(p.suffix)
            )


# ---------------------------------------------------------------------------
# resolve_limited_run_paths
# ---------------------------------------------------------------------------
class ResolveLimitedRunPathsTests(unittest.TestCase):
    """For limited runs the auto-route guardrail must rewrite any of the three
    official artifact paths to a `_partial` sibling so a debug invocation
    can never overwrite the authoritative Phase 2 baseline artifacts. For
    full runs (limit is None) the input paths are returned verbatim."""

    def test_full_run_returns_paths_unchanged(self):
        m, o, q, warnings = bfq.resolve_limited_run_paths(
            limit=None,
            metrics=OFFICIAL_METRICS,
            outputs=OFFICIAL_OUTPUTS,
            qualitative=OFFICIAL_QUAL,
        )
        self.assertEqual(m, OFFICIAL_METRICS)
        self.assertEqual(o, OFFICIAL_OUTPUTS)
        self.assertEqual(q, OFFICIAL_QUAL)
        self.assertEqual(warnings, [])

    def test_limited_with_all_three_defaults_routes_all_three_to_partial(self):
        m, o, q, warnings = bfq.resolve_limited_run_paths(
            limit=3,
            metrics=OFFICIAL_METRICS,
            outputs=OFFICIAL_OUTPUTS,
            qualitative=OFFICIAL_QUAL,
        )
        # None of the resolved paths may still point at the official locations.
        self.assertNotEqual(m, OFFICIAL_METRICS)
        self.assertNotEqual(o, OFFICIAL_OUTPUTS)
        self.assertNotEqual(q, OFFICIAL_QUAL)
        # Each must be the `_partial` sibling.
        self.assertEqual(m, bfq._partial_sibling(OFFICIAL_METRICS))
        self.assertEqual(o, bfq._partial_sibling(OFFICIAL_OUTPUTS))
        self.assertEqual(q, bfq._partial_sibling(OFFICIAL_QUAL))
        # Exactly three warnings, one per rerouted path.
        self.assertEqual(len(warnings), 3)
        for w in warnings:
            self.assertIn("auto-routed", w)
            self.assertIn(str(OFFICIAL_METRICS.parent), w)

    def test_limited_with_explicit_custom_paths_unchanged(self):
        custom_metrics = Path("/tmp/my_debug_metrics.json")
        custom_outputs = Path("/tmp/my_debug_outputs.jsonl")
        custom_qual = Path("/tmp/my_debug_qual.md")
        m, o, q, warnings = bfq.resolve_limited_run_paths(
            limit=3,
            metrics=custom_metrics,
            outputs=custom_outputs,
            qualitative=custom_qual,
        )
        self.assertEqual(m, custom_metrics)
        self.assertEqual(o, custom_outputs)
        self.assertEqual(q, custom_qual)
        self.assertEqual(warnings, [])

    def test_limited_with_mixed_default_and_custom_routes_only_defaults(self):
        custom_outputs = Path("/tmp/my_debug_outputs.jsonl")
        m, o, q, warnings = bfq.resolve_limited_run_paths(
            limit=3,
            metrics=OFFICIAL_METRICS,        # default → reroute
            outputs=custom_outputs,           # custom → unchanged
            qualitative=OFFICIAL_QUAL,        # default → reroute
        )
        self.assertNotEqual(m, OFFICIAL_METRICS)
        self.assertEqual(m, bfq._partial_sibling(OFFICIAL_METRICS))
        self.assertEqual(o, custom_outputs)
        self.assertNotEqual(q, OFFICIAL_QUAL)
        self.assertEqual(q, bfq._partial_sibling(OFFICIAL_QUAL))
        self.assertEqual(len(warnings), 2)

    def test_partial_paths_do_not_collide_with_official_paths(self):
        """The whole point of the reroute: the resolved partial path must
        never collide with the original official path."""
        m, _, _, _ = bfq.resolve_limited_run_paths(
            limit=3,
            metrics=OFFICIAL_METRICS,
            outputs=OFFICIAL_OUTPUTS,
            qualitative=OFFICIAL_QUAL,
        )
        self.assertNotEqual(m, OFFICIAL_METRICS)
        # And the rerouted file lives in the same directory.
        self.assertEqual(m.parent, OFFICIAL_METRICS.parent)

    def test_rerouted_paths_are_stable(self):
        """Calling twice yields the same result — no random suffixes."""
        a = bfq.resolve_limited_run_paths(
            limit=3,
            metrics=OFFICIAL_METRICS,
            outputs=OFFICIAL_OUTPUTS,
            qualitative=OFFICIAL_QUAL,
        )
        b = bfq.resolve_limited_run_paths(
            limit=3,
            metrics=OFFICIAL_METRICS,
            outputs=OFFICIAL_OUTPUTS,
            qualitative=OFFICIAL_QUAL,
        )
        self.assertEqual(a, b)


# ---------------------------------------------------------------------------
# classify_run_status
# ---------------------------------------------------------------------------
class ClassifyRunStatusTests(unittest.TestCase):
    """Pass gate that protects Phase 3.

    Requirement: ``status == "pass"`` is only reachable for a FULL baseline
    that ran every eval example (examples_run == eval_total) with no
    pre-inference blocks AND no length truncations AND at least one
    schema-valid output. Any limited run (is_full_baseline=False) MUST end
    up with a ``partial_``-prefixed status, never a top-level pass."""

    # ----- FULL RUN -----
    def test_full_pass_only_when_examples_run_equals_eval_total(self):
        cls = bfq.classify_run_status(
            is_full_baseline=True,
            examples_run=35,
            blocked_pre_inference=0,
            schema_valid_count=35,
            length_truncated_count=0,
            eval_total=35,
        )
        self.assertEqual(cls["status"], "pass")
        self.assertTrue(cls["full_coverage"])
        self.assertEqual(cls["status_prefix"], "")

    def test_full_pass_requires_examples_run_equal_to_eval_total(self):
        """examples_run < eval_total means NOT every eval example was
        attempted — the gate must reject this as incomplete, never pass."""
        cls = bfq.classify_run_status(
            is_full_baseline=True,
            examples_run=34,                # one short of the full 35
            blocked_pre_inference=0,
            schema_valid_count=34,
            length_truncated_count=0,
            eval_total=35,
        )
        self.assertNotEqual(cls["status"], "pass")
        self.assertEqual(cls["status"], "incomplete")
        self.assertFalse(cls["full_coverage"])

    def test_full_incomplete_when_any_blocked(self):
        cls = bfq.classify_run_status(
            is_full_baseline=True,
            examples_run=35,                 # ran 35, but 1 was blocked
            blocked_pre_inference=1,
            schema_valid_count=35,
            length_truncated_count=0,
            eval_total=36,                   # eval_total is 36 → 35 ran, 1 blocked
        )
        # examples_run == eval_total is False (35 != 36), so gate fails.
        self.assertEqual(cls["status"], "incomplete")

    def test_full_pass_with_truncations(self):
        cls = bfq.classify_run_status(
            is_full_baseline=True,
            examples_run=35,
            blocked_pre_inference=0,
            schema_valid_count=35,
            length_truncated_count=2,
            eval_total=35,
        )
        self.assertEqual(cls["status"], "pass_with_truncations")
        self.assertNotEqual(cls["status"], "pass")

    def test_full_fail_when_zero_schema_valid(self):
        cls = bfq.classify_run_status(
            is_full_baseline=True,
            examples_run=35,
            blocked_pre_inference=0,
            schema_valid_count=0,
            length_truncated_count=0,
            eval_total=35,
        )
        self.assertEqual(cls["status"], "fail")
        self.assertNotEqual(cls["status"], "pass")

    def test_full_pass_strict_examples_run_must_equal_eval_total_not_just_cover(self):
        """Sanity: even with blocked==0, if examples_run != eval_total
        (which can't happen in normal operation but must be defended against
        if a future refactor changes how rows are sliced), the status must
        NOT reach 'pass'."""
        cls = bfq.classify_run_status(
            is_full_baseline=True,
            examples_run=35,                # covered all 35
            blocked_pre_inference=0,
            schema_valid_count=35,
            length_truncated_count=0,
            eval_total=35,
        )
        # Confirm: examples_run == eval_total → pass
        self.assertEqual(cls["status"], "pass")

        cls = bfq.classify_run_status(
            is_full_baseline=True,
            examples_run=34,
            blocked_pre_inference=0,
            schema_valid_count=34,
            length_truncated_count=0,
            eval_total=35,
        )
        # Confirm: examples_run < eval_total → never pass
        self.assertNotEqual(cls["status"], "pass")

    # ----- LIMITED RUN -----
    def test_limited_clean_run_is_partial_pass_never_top_level_pass(self):
        """A limited run with a clean slice must produce partial_pass,
        NOT pass. This is the key property that keeps a --limit invocation
        from unblocking Phase 3."""
        cls = bfq.classify_run_status(
            is_full_baseline=False,
            examples_run=3,
            blocked_pre_inference=0,
            schema_valid_count=3,
            length_truncated_count=0,
            eval_total=35,
        )
        # The crucial assertions: status is partial_pass, NEVER a top-level pass.
        self.assertNotEqual(cls["status"], "pass")
        self.assertEqual(cls["status"], "partial_pass")
        self.assertEqual(cls["status_prefix"], "partial_")
        # Slice was clean (no pre-inference blocks), so full_coverage is True —
        # but that alone does NOT promote the status to a top-level pass
        # because the status_prefix is "partial_". This is the gate.
        self.assertTrue(cls["full_coverage"])

    def test_limited_with_blocks_is_partial_incomplete(self):
        cls = bfq.classify_run_status(
            is_full_baseline=False,
            examples_run=2,
            blocked_pre_inference=1,
            schema_valid_count=2,
            length_truncated_count=0,
            eval_total=35,
        )
        self.assertEqual(cls["status"], "partial_incomplete")
        self.assertNotEqual(cls["status"], "pass")
        self.assertNotEqual(cls["status"], "incomplete")

    def test_limited_with_truncations_is_partial_pass_with_truncations(self):
        cls = bfq.classify_run_status(
            is_full_baseline=False,
            examples_run=3,
            blocked_pre_inference=0,
            schema_valid_count=3,
            length_truncated_count=1,
            eval_total=35,
        )
        self.assertEqual(cls["status"], "partial_pass_with_truncations")
        self.assertNotEqual(cls["status"], "pass")
        self.assertNotEqual(cls["status"], "pass_with_truncations")

    def test_limited_with_zero_schema_valid_is_partial_fail(self):
        cls = bfq.classify_run_status(
            is_full_baseline=False,
            examples_run=3,
            blocked_pre_inference=0,
            schema_valid_count=0,
            length_truncated_count=0,
            eval_total=35,
        )
        self.assertEqual(cls["status"], "partial_fail")
        self.assertNotEqual(cls["status"], "pass")
        self.assertNotEqual(cls["status"], "fail")

    def test_limited_never_produces_top_level_pass_for_any_clean_slice(self):
        """Defensive sweep: even for arbitrary (examples_run, eval_total)
        combinations that satisfy full_coverage for a full run, a limited
        run still produces a partial_* status."""
        for examples_run in (1, 2, 5, 10, 34, 35):
            cls = bfq.classify_run_status(
                is_full_baseline=False,
                examples_run=examples_run,
                blocked_pre_inference=0,
                schema_valid_count=examples_run,
                length_truncated_count=0,
                eval_total=35,
            )
            self.assertNotEqual(
                cls["status"],
                "pass",
                msg=f"limited run with examples_run={examples_run} produced 'pass'",
            )
            self.assertTrue(
                cls["status"].startswith("partial_"),
                msg=f"limited run produced status={cls['status']!r} (missing partial_ prefix)",
            )

    def test_reason_text_present_for_fail_and_incomplete_and_truncations(self):
        for kwargs in [
            dict(is_full_baseline=True, examples_run=35, blocked_pre_inference=0,
                 schema_valid_count=0, length_truncated_count=0, eval_total=35),
            dict(is_full_baseline=True, examples_run=34, blocked_pre_inference=0,
                 schema_valid_count=34, length_truncated_count=0, eval_total=35),
            dict(is_full_baseline=True, examples_run=35, blocked_pre_inference=0,
                 schema_valid_count=35, length_truncated_count=2, eval_total=35),
        ]:
            cls = bfq.classify_run_status(**kwargs)
            self.assertTrue(cls["reason"], msg=f"empty reason for {cls['status']}")

    def test_reason_text_empty_for_pass(self):
        cls = bfq.classify_run_status(
            is_full_baseline=True,
            examples_run=35,
            blocked_pre_inference=0,
            schema_valid_count=35,
            length_truncated_count=0,
            eval_total=35,
        )
        self.assertEqual(cls["status"], "pass")
        self.assertEqual(cls["reason"], "")


# ---------------------------------------------------------------------------
# merge_readiness_for_partial
# ---------------------------------------------------------------------------
class MergeReadinessForPartialTests(unittest.TestCase):
    """The partial-write helper must NEVER overwrite top-level ``status``,
    ``status_note``, or ``full_baseline`` — those fields are reserved for the
    authoritative full-baseline write. This is the second-half of the gate
    that keeps a ``--limit`` invocation from unblocking Phase 3 even when
    the user explicitly passes ``--update-readiness``."""

    def _make_full_baseline_readiness(self) -> dict:
        """Mirror the shape produced by the authoritative full 35/35 run."""
        return {
            "phase": "Phase 2 — Baseline Qwen2.5-7B-Instruct",
            "status": "pass",
            "status_note": (
                "Full 35-example Phase 2 baseline completed with status=pass."
            ),
            "runner_hardening_note": (
                "scripts/baseline_qwen_full.py was hardened so that a "
                "--limit-limited invocation can no longer write status='pass'."
            ),
            "checked_at": "2026-06-27T00:09:45-0300",
            "full_baseline": {
                "report": "metrics/baseline_qwen2.5-7b.json",
                "raw_outputs": "metrics/baseline_qwen2.5-7b_outputs.jsonl",
                "qualitative_report": "metrics/qualitative_report.md",
                "examples_total": 35,
                "examples_run": 35,
                "examples_blocked_pre_inference": 0,
                "parse_valid": 35,
                "schema_valid": 35,
                "schema_validity": 1.0,
                "categorical_accuracy": 0.0384,
                "tiene_eventos_protesta_accuracy": 0.2857,
                "f1_global": 0.0971,
                "field_recall_exact": 0.054,
                "field_recall_non_empty": 0.1692,
            },
        }

    def _make_partial_run(self, **overrides) -> dict:
        base = {
            "checked_at": "2026-06-27T01:00:00-0300",
            "limit": 3,
            "eval_total": 35,
            "examples_run": 3,
            "examples_blocked_pre_inference": 0,
            "parse_valid": 3,
            "schema_valid": 3,
            "schema_validity": 1.0,
            "categorical_accuracy": 0.05,
            "tiene_eventos_protesta_accuracy": 0.33,
            "f1_global": 0.12,
            "field_recall_exact": 0.06,
            "field_recall_non_empty": 0.18,
            "status": "partial_pass",
            "status_note": "PARTIAL run (--limit=3)...",
            "report": "metrics/baseline_qwen2.5-7b_partial.json",
            "raw_outputs": "metrics/baseline_qwen2.5-7b_outputs_partial.jsonl",
            "env": {"VLLM_USE_FLASHINFER_SAMPLER": "0"},
        }
        base.update(overrides)
        return base

    def test_top_level_status_preserved_after_partial_write(self):
        readiness = self._make_full_baseline_readiness()
        merged = bfq.merge_readiness_for_partial(readiness, self._make_partial_run())
        self.assertEqual(merged["status"], "pass")

    def test_top_level_status_note_preserved_after_partial_write(self):
        readiness = self._make_full_baseline_readiness()
        merged = bfq.merge_readiness_for_partial(readiness, self._make_partial_run())
        self.assertEqual(
            merged["status_note"],
            "Full 35-example Phase 2 baseline completed with status=pass.",
        )

    def test_full_baseline_block_preserved_verbatim_after_partial_write(self):
        readiness = self._make_full_baseline_readiness()
        original_full_baseline = dict(readiness["full_baseline"])
        merged = bfq.merge_readiness_for_partial(
            readiness, self._make_partial_run()
        )
        # Must be byte-equal to the original — no field added, dropped, or
        # changed.
        self.assertEqual(merged["full_baseline"], original_full_baseline)

    def test_runner_hardening_note_preserved_after_partial_write(self):
        readiness = self._make_full_baseline_readiness()
        merged = bfq.merge_readiness_for_partial(readiness, self._make_partial_run())
        self.assertEqual(
            merged["runner_hardening_note"],
            readiness["runner_hardening_note"],
        )

    def test_partial_run_appended_to_partial_baseline_runs(self):
        readiness = self._make_full_baseline_readiness()
        merged = bfq.merge_readiness_for_partial(readiness, self._make_partial_run())
        self.assertIn("partial_baseline_runs", merged)
        self.assertEqual(len(merged["partial_baseline_runs"]), 1)
        self.assertEqual(merged["partial_baseline_runs"][0]["limit"], 3)
        self.assertEqual(merged["partial_baseline_runs"][0]["status"], "partial_pass")
        self.assertEqual(merged["partial_baseline_runs"][0]["examples_run"], 3)

    def test_multiple_partial_writes_accumulate(self):
        readiness = self._make_full_baseline_readiness()
        m1 = bfq.merge_readiness_for_partial(readiness, self._make_partial_run(limit=3))
        m2 = bfq.merge_readiness_for_partial(m1, self._make_partial_run(limit=5))
        m3 = bfq.merge_readiness_for_partial(m2, self._make_partial_run(limit=10))
        self.assertEqual(len(m3["partial_baseline_runs"]), 3)
        self.assertEqual([r["limit"] for r in m3["partial_baseline_runs"]], [3, 5, 10])
        # Top-level status / status_note / full_baseline untouched across all 3.
        self.assertEqual(m3["status"], "pass")
        self.assertEqual(m3["status_note"], readiness["status_note"])
        self.assertEqual(m3["full_baseline"], readiness["full_baseline"])

    def test_merge_does_not_mutate_input(self):
        readiness = self._make_full_baseline_readiness()
        snapshot = {
            "status": readiness["status"],
            "status_note": readiness["status_note"],
            "full_baseline": dict(readiness["full_baseline"]),
            "partial_baseline_runs": list(readiness.get("partial_baseline_runs", [])),
        }
        bfq.merge_readiness_for_partial(readiness, self._make_partial_run())
        # Input dict unchanged.
        self.assertEqual(readiness["status"], snapshot["status"])
        self.assertEqual(readiness["status_note"], snapshot["status_note"])
        self.assertEqual(readiness["full_baseline"], snapshot["full_baseline"])
        self.assertNotIn("partial_baseline_runs", readiness)

    def test_merge_returns_new_dict(self):
        readiness = self._make_full_baseline_readiness()
        merged = bfq.merge_readiness_for_partial(readiness, self._make_partial_run())
        self.assertIsNot(merged, readiness)

    def test_phase_label_refreshed_to_phase_2(self):
        readiness = self._make_full_baseline_readiness()
        readiness["phase"] = "something stale"
        merged = bfq.merge_readiness_for_partial(readiness, self._make_partial_run())
        self.assertEqual(merged["phase"], "Phase 2 — Baseline Qwen2.5-7B-Instruct")

    def test_partial_run_with_no_existing_partial_baseline_runs_creates_array(self):
        readiness = self._make_full_baseline_readiness()
        self.assertNotIn("partial_baseline_runs", readiness)
        merged = bfq.merge_readiness_for_partial(readiness, self._make_partial_run())
        self.assertIsInstance(merged["partial_baseline_runs"], list)
        self.assertEqual(len(merged["partial_baseline_runs"]), 1)


# ---------------------------------------------------------------------------
# Integration: reroute + classify simulate the main() gate for a --limit run
# ---------------------------------------------------------------------------
class EndToEndGuardrailTests(unittest.TestCase):
    """End-to-end proof that a limited run cannot unblock Phase 3:
       1. Default artifact paths are auto-routed away from the official ones
          BEFORE any side-effect.
       2. The resulting run status is `partial_pass` (or partial_*), never
          a top-level `pass`.
       3. A partial write to readiness preserves the previous full_baseline.
    """

    def test_limited_run_with_all_defaults_cannot_touch_official_artifacts(self):
        # Step 1 — path reroute
        m, o, q, warnings = bfq.resolve_limited_run_paths(
            limit=3,
            metrics=OFFICIAL_METRICS,
            outputs=OFFICIAL_OUTPUTS,
            qualitative=OFFICIAL_QUAL,
        )
        self.assertNotEqual(m, OFFICIAL_METRICS)
        self.assertNotEqual(o, OFFICIAL_OUTPUTS)
        self.assertNotEqual(q, OFFICIAL_QUAL)
        self.assertEqual(len(warnings), 3)

        # Step 2 — status gate (slice was clean)
        cls = bfq.classify_run_status(
            is_full_baseline=False,
            examples_run=3,
            blocked_pre_inference=0,
            schema_valid_count=3,
            length_truncated_count=0,
            eval_total=35,
        )
        self.assertEqual(cls["status"], "partial_pass")
        self.assertNotEqual(cls["status"], "pass")

        # Step 3 — even with --update-readiness, top-level readiness stays put
        readiness = {
            "phase": "Phase 2 — Baseline Qwen2.5-7B-Instruct",
            "status": "pass",
            "status_note": "Full 35-example Phase 2 baseline completed with status=pass.",
            "full_baseline": {
                "examples_total": 35,
                "examples_run": 35,
                "f1_global": 0.0971,
            },
        }
        partial = {
            "limit": 3,
            "examples_run": 3,
            "status": cls["status"],
            "status_note": "PARTIAL run (--limit=3)...",
        }
        merged = bfq.merge_readiness_for_partial(readiness, partial)
        self.assertEqual(merged["status"], "pass")
        self.assertEqual(
            merged["full_baseline"],
            {
                "examples_total": 35,
                "examples_run": 35,
                "f1_global": 0.0971,
            },
        )
        self.assertEqual(merged["partial_baseline_runs"][-1]["status"], "partial_pass")

    def test_full_run_still_passes_when_all_conditions_met(self):
        """Sanity: a real full run (no --limit) still reaches 'pass' and
        the merge_readiness helper is NOT the right path for full — main()
        uses the explicit full-write branch in that case. This test pins
        the contract that the pure helpers, when composed for a full run,
        would still produce a top-level pass."""

        # Path reroute: full run does not reroute.
        m, o, q, warnings = bfq.resolve_limited_run_paths(
            limit=None,
            metrics=OFFICIAL_METRICS,
            outputs=OFFICIAL_OUTPUTS,
            qualitative=OFFICIAL_QUAL,
        )
        self.assertEqual(m, OFFICIAL_METRICS)
        self.assertEqual(o, OFFICIAL_OUTPUTS)
        self.assertEqual(q, OFFICIAL_QUAL)
        self.assertEqual(warnings, [])

        # Status gate: full pass conditions met.
        cls = bfq.classify_run_status(
            is_full_baseline=True,
            examples_run=35,
            blocked_pre_inference=0,
            schema_valid_count=35,
            length_truncated_count=0,
            eval_total=35,
        )
        self.assertEqual(cls["status"], "pass")
        self.assertEqual(cls["status_prefix"], "")


if __name__ == "__main__":
    unittest.main(verbosity=2)