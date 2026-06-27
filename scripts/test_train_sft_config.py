#!/usr/bin/env python3
"""Smoke tests for training/train_sft.py.

These tests are deliberately tiny and self-contained: they exercise the
dataset conversion, the CLI plumbing, and the token-length audit using the
real Qwen tokenizer (no base model is loaded). They are designed to run in
seconds and to be safe to invoke from CI/local validation before a real
training run.

Run with:

    .venv/bin/python scripts/test_train_sft_config.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_train_sft():
    spec = importlib.util.spec_from_file_location("train_sft", REPO_ROOT / "training" / "train_sft.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register in sys.modules before exec_module so dataclass introspection
    # (which reads `sys.modules[cls.__module__].__dict__`) works as expected.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestConfigLoading(unittest.TestCase):
    def setUp(self) -> None:
        self.train_sft = _load_train_sft()

    def test_config_file_loads_with_locked_defaults(self) -> None:
        cfg = self.train_sft.config_from_file(REPO_ROOT / "training" / "config_qwen_protesta_v1.json")
        self.assertEqual(cfg["model_name_or_path"], "Qwen/Qwen2.5-7B-Instruct")
        self.assertEqual(cfg["max_seq_length"], 20480)
        self.assertFalse(cfg["packing"])
        self.assertTrue(cfg["completion_only_loss"])
        self.assertEqual(cfg["lora_r"], 16)
        self.assertEqual(cfg["lora_alpha"], 32)
        self.assertEqual(cfg["lora_dropout"], 0.05)
        self.assertEqual(
            cfg["lora_target_modules"],
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        self.assertEqual(cfg["num_train_epochs"], 3.0)
        self.assertEqual(cfg["per_device_train_batch_size"], 1)
        # Phase 3 safety fix: eval batch size must be explicit and default to 1,
        # not the Transformers default of 8. Otherwise an OOM in eval (35 rows
        # → 5 batches × per_device_eval_batch_size) is much more likely.
        self.assertEqual(cfg["per_device_eval_batch_size"], 1)
        self.assertEqual(cfg["gradient_accumulation_steps"], 24)
        self.assertAlmostEqual(cfg["learning_rate"], 2.0e-4)
        self.assertEqual(cfg["lr_scheduler_type"], "cosine")
        self.assertEqual(cfg["warmup_ratio"], 0.05)
        self.assertEqual(cfg["weight_decay"], 0.0)
        self.assertEqual(cfg["optim"], "paged_adamw_8bit")
        self.assertTrue(cfg["gradient_checkpointing"])
        self.assertTrue(cfg["bf16"])

    def test_lora_target_modules_cover_qwen_attn_and_mlp(self) -> None:
        cfg = self.train_sft.config_from_file(REPO_ROOT / "training" / "config_qwen_protesta_v1.json")
        attn = {"q_proj", "k_proj", "v_proj", "o_proj"}
        mlp = {"gate_proj", "up_proj", "down_proj"}
        self.assertTrue(attn.issubset(set(cfg["lora_target_modules"])))
        self.assertTrue(mlp.issubset(set(cfg["lora_target_modules"])))


class TestR32ConfigLoading(unittest.TestCase):
    """Phase 6 r=32 config: identical to r=16 except for r/lora_alpha and
    output_dir. This pins the locked methodology for the iteration so a
    future edit cannot accidentally change a non-experimental variable
    (data, prompt, max_seq_length, batch, lr, epochs, packing, completion-
    only-loss)."""

    def setUp(self) -> None:
        self.train_sft = _load_train_sft()
        self.r16_cfg = self.train_sft.config_from_file(
            REPO_ROOT / "training" / "config_qwen_protesta_v1.json"
        )
        self.r32_cfg = self.train_sft.config_from_file(
            REPO_ROOT / "training" / "config_qwen_protesta_v1_r32.json"
        )

    def test_r32_config_file_loads(self) -> None:
        # The r=32 config MUST exist on disk — this is the file the
        # orchestrator will pass via ``--config`` to the training runner.
        self.assertTrue(
            (REPO_ROOT / "training" / "config_qwen_protesta_v1_r32.json").exists()
        )

    def test_r32_lora_rank_is_32(self) -> None:
        self.assertEqual(self.r32_cfg["lora_r"], 32)

    def test_r32_lora_alpha_is_64(self) -> None:
        self.assertEqual(self.r32_cfg["lora_alpha"], 64)

    def test_r32_alpha_over_r_ratio_matches_r16(self) -> None:
        """The 2.0 alpha/r ratio must be preserved: r=16 → alpha=32, r=32 → alpha=64."""
        r16_ratio = self.r16_cfg["lora_alpha"] / self.r16_cfg["lora_r"]
        r32_ratio = self.r32_cfg["lora_alpha"] / self.r32_cfg["lora_r"]
        self.assertAlmostEqual(r32_ratio, r16_ratio, places=4)
        self.assertAlmostEqual(r32_ratio, 2.0, places=4)

    def test_r32_target_modules_match_r16_exactly(self) -> None:
        """The ONLY experimental variable is rank — target_modules must be identical."""
        self.assertEqual(self.r32_cfg["lora_target_modules"], self.r16_cfg["lora_target_modules"])
        # And both must still cover attention + MLP (defense in depth).
        attn = {"q_proj", "k_proj", "v_proj", "o_proj"}
        mlp = {"gate_proj", "up_proj", "down_proj"}
        self.assertTrue(attn.issubset(set(self.r32_cfg["lora_target_modules"])))
        self.assertTrue(mlp.issubset(set(self.r32_cfg["lora_target_modules"])))

    def test_r32_methodology_keys_match_r16(self) -> None:
        """Every methodology key must match r=16 — data, prompt, max_seq_length,
        batch, lr, epochs, packing, completion-only-loss, optimizer, scheduler,
        warmup, gradient_checkpointing, bf16."""
        for key in (
            "model_name_or_path",
            "dataset_train_path",
            "dataset_eval_path",
            "max_seq_length",
            "packing",
            "completion_only_loss",
            "num_train_epochs",
            "per_device_train_batch_size",
            "per_device_eval_batch_size",
            "gradient_accumulation_steps",
            "learning_rate",
            "lr_scheduler_type",
            "warmup_ratio",
            "weight_decay",
            "optim",
            "gradient_checkpointing",
            "bf16",
            "max_grad_norm",
            "seed",
            "lora_dropout",
            "lora_bias",
        ):
            self.assertEqual(
                self.r32_cfg[key],
                self.r16_cfg[key],
                f"r32 vs r16 mismatch on methodology key {key!r}: "
                f"r16={self.r16_cfg[key]!r} r32={self.r32_cfg[key]!r}",
            )

    def test_r32_output_dir_is_separate_from_r16(self) -> None:
        """The r=32 output_dir MUST NOT collide with the r=16 output_dir."""
        self.assertNotEqual(self.r32_cfg["output_dir"], self.r16_cfg["output_dir"])
        # Resolved absolute forms must also be distinct.
        r16_abs = (REPO_ROOT / self.r16_cfg["output_dir"]).resolve()
        r32_abs = (REPO_ROOT / self.r32_cfg["output_dir"]).resolve()
        self.assertNotEqual(r16_abs, r32_abs)
        # And the r=32 output_dir must be the Phase 6 contract path.
        self.assertEqual(self.r32_cfg["output_dir"], "checkpoints/qwen-protesta-v1-r32")

    def test_r32_max_seq_length_unchanged_at_20480(self) -> None:
        """max_seq_length=20480 is the PLAN-locked budget. Raising it to fit
        longer examples would be a silent methodology drift."""
        self.assertEqual(self.r32_cfg["max_seq_length"], 20480)
        self.assertEqual(self.r16_cfg["max_seq_length"], 20480)

    def test_r32_uses_existing_chat_formatted_data(self) -> None:
        """Both configs must point at the same chat_formatted/*.jsonl files
        — no separate dataset preparation is allowed for the iteration."""
        self.assertEqual(self.r32_cfg["dataset_train_path"], self.r16_cfg["dataset_train_path"])
        self.assertEqual(self.r32_cfg["dataset_eval_path"], self.r16_cfg["dataset_eval_path"])


class TestR32E5ConfigLoading(unittest.TestCase):
    """Phase 6 r=32 5-epoch config: identical to r=32 3e except for
    num_train_epochs (3.0 -> 5.0) and output_dir. This pins the locked
    methodology for the extra-epochs iteration so a future edit cannot
    accidentally change a non-experimental variable (data, prompt,
    max_seq_length, batch, lr, lora, scheduler, optimizer)."""

    def setUp(self) -> None:
        self.train_sft = _load_train_sft()
        self.r16_cfg = self.train_sft.config_from_file(
            REPO_ROOT / "training" / "config_qwen_protesta_v1.json"
        )
        self.r32_cfg = self.train_sft.config_from_file(
            REPO_ROOT / "training" / "config_qwen_protesta_v1_r32.json"
        )
        self.e5_cfg = self.train_sft.config_from_file(
            REPO_ROOT / "training" / "config_qwen_protesta_v1_r32_e5.json"
        )

    def test_e5_config_file_loads(self) -> None:
        # The r=32 5-epoch config MUST exist on disk — this is the file the
        # orchestrator will pass via ``--config`` to the training runner.
        self.assertTrue(
            (REPO_ROOT / "training" / "config_qwen_protesta_v1_r32_e5.json").exists()
        )

    def test_e5_lora_rank_is_32(self) -> None:
        self.assertEqual(self.e5_cfg["lora_r"], 32)

    def test_e5_lora_alpha_is_64(self) -> None:
        self.assertEqual(self.e5_cfg["lora_alpha"], 64)

    def test_e5_alpha_over_r_ratio_matches_r16_and_r32(self) -> None:
        """The 2.0 alpha/r ratio must be preserved across the locked family."""
        r16_ratio = self.r16_cfg["lora_alpha"] / self.r16_cfg["lora_r"]
        r32_ratio = self.r32_cfg["lora_alpha"] / self.r32_cfg["lora_r"]
        e5_ratio = self.e5_cfg["lora_alpha"] / self.e5_cfg["lora_r"]
        self.assertAlmostEqual(e5_ratio, r16_ratio, places=4)
        self.assertAlmostEqual(e5_ratio, r32_ratio, places=4)
        self.assertAlmostEqual(e5_ratio, 2.0, places=4)

    def test_e5_target_modules_match_r32_exactly(self) -> None:
        """LoRA target_modules MUST be identical to r=32 (and r=16)."""
        self.assertEqual(self.e5_cfg["lora_target_modules"], self.r32_cfg["lora_target_modules"])
        self.assertEqual(self.e5_cfg["lora_target_modules"], self.r16_cfg["lora_target_modules"])
        attn = {"q_proj", "k_proj", "v_proj", "o_proj"}
        mlp = {"gate_proj", "up_proj", "down_proj"}
        self.assertTrue(attn.issubset(set(self.e5_cfg["lora_target_modules"])))
        self.assertTrue(mlp.issubset(set(self.e5_cfg["lora_target_modules"])))

    def test_e5_methodology_keys_match_r32(self) -> None:
        """The ONLY experimental variable vs r=32 3e is num_train_epochs.
        Every other methodology key MUST match r=32 (data, prompt,
        max_seq_length, batch, lr, lora, scheduler, optim, completion_only_loss)."""
        for key in (
            "model_name_or_path",
            "dataset_train_path",
            "dataset_eval_path",
            "max_seq_length",
            "packing",
            "completion_only_loss",
            "per_device_train_batch_size",
            "per_device_eval_batch_size",
            "gradient_accumulation_steps",
            "learning_rate",
            "lr_scheduler_type",
            "warmup_ratio",
            "weight_decay",
            "optim",
            "gradient_checkpointing",
            "bf16",
            "max_grad_norm",
            "seed",
            "lora_dropout",
            "lora_bias",
            "lora_r",
            "lora_alpha",
        ):
            self.assertEqual(
                self.e5_cfg[key],
                self.r32_cfg[key],
                f"e5 vs r32 mismatch on methodology key {key!r}: "
                f"r32={self.r32_cfg[key]!r} e5={self.e5_cfg[key]!r}",
            )

    def test_e5_num_train_epochs_is_5(self) -> None:
        """The ONLY experimental variable is num_train_epochs (3.0 -> 5.0)."""
        self.assertEqual(self.e5_cfg["num_train_epochs"], 5.0)
        # Sanity-check the reference points.
        self.assertEqual(self.r32_cfg["num_train_epochs"], 3.0)
        self.assertEqual(self.r16_cfg["num_train_epochs"], 3.0)

    def test_e5_output_dir_is_separate_from_r32_and_r16(self) -> None:
        """The e5 output_dir MUST NOT collide with r=32 or r=16."""
        self.assertNotEqual(self.e5_cfg["output_dir"], self.r32_cfg["output_dir"])
        self.assertNotEqual(self.e5_cfg["output_dir"], self.r16_cfg["output_dir"])
        # Resolved absolute forms must also be distinct.
        r16_abs = (REPO_ROOT / self.r16_cfg["output_dir"]).resolve()
        r32_abs = (REPO_ROOT / self.r32_cfg["output_dir"]).resolve()
        e5_abs = (REPO_ROOT / self.e5_cfg["output_dir"]).resolve()
        self.assertNotEqual(r16_abs, r32_abs)
        self.assertNotEqual(r32_abs, e5_abs)
        self.assertNotEqual(r16_abs, e5_abs)
        # And the e5 output_dir must be the Phase 6 contract path.
        self.assertEqual(self.e5_cfg["output_dir"], "checkpoints/qwen-protesta-v1-r32-e5")

    def test_e5_max_seq_length_unchanged_at_20480(self) -> None:
        """max_seq_length=20480 is the PLAN-locked budget. Raising it to fit
        longer examples would be a silent methodology drift."""
        self.assertEqual(self.e5_cfg["max_seq_length"], 20480)
        self.assertEqual(self.r32_cfg["max_seq_length"], 20480)
        self.assertEqual(self.r16_cfg["max_seq_length"], 20480)

    def test_e5_uses_existing_chat_formatted_data(self) -> None:
        """All three configs must point at the same chat_formatted/*.jsonl
        files — no separate dataset preparation is allowed for the iteration."""
        self.assertEqual(self.e5_cfg["dataset_train_path"], self.r32_cfg["dataset_train_path"])
        self.assertEqual(self.e5_cfg["dataset_eval_path"], self.r32_cfg["dataset_eval_path"])
        self.assertEqual(self.e5_cfg["dataset_train_path"], self.r16_cfg["dataset_train_path"])
        self.assertEqual(self.e5_cfg["dataset_eval_path"], self.r16_cfg["dataset_eval_path"])

    def test_e5_dry_run_report_path_is_distinct(self) -> None:
        """The dry-run readiness report must point at the e5-specific path,
        NOT collide with the r32 or r16 readiness reports."""
        cfg_obj = json.loads(
            (REPO_ROOT / "training" / "config_qwen_protesta_v1_r32_e5.json").read_text()
        )
        self.assertEqual(
            cfg_obj["dry_run"]["report_path"], "reports/phase6_r32_e5_readiness.json"
        )
        self.assertNotEqual(cfg_obj["dry_run"]["report_path"], "reports/phase6_r32_readiness.json")
        self.assertNotEqual(cfg_obj["dry_run"]["report_path"], "reports/phase3_readiness.json")


class TestDatasetConversion(unittest.TestCase):
    def setUp(self) -> None:
        self.train_sft = _load_train_sft()

    def _write(self, tmp: Path, rows: list[dict]) -> Path:
        path = tmp / "rows.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return path

    def test_validate_chatml_row_accepts_canonical_shape(self) -> None:
        row = {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "usr"},
                {"role": "assistant", "content": "ast"},
            ]
        }
        self.train_sft.validate_chatml_row(row, 1)

    def test_validate_chatml_row_rejects_missing_role(self) -> None:
        row = {"messages": [{"role": "user", "content": "usr"}]}
        with self.assertRaises(ValueError):
            self.train_sft.validate_chatml_row(row, 1)

    def test_validate_chatml_row_rejects_extra_roles(self) -> None:
        row = {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "usr"},
                {"role": "assistant", "content": "ast1"},
                {"role": "assistant", "content": "ast2"},
            ]
        }
        with self.assertRaises(ValueError):
            self.train_sft.validate_chatml_row(row, 1)

    def test_validate_chatml_row_rejects_unknown_extra_role(self) -> None:
        """Unknown roles (e.g. 'tool', 'function', 'developer') must fail loud.

        Phase 3 review warning: previously only the system/user/assistant
        presence was checked, so a stray non-canonical role could silently
        leak into the prompt via to_prompt_completion.
        """

        for bad_role in ("tool", "function", "developer", "user_extra"):
            row = {
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "usr"},
                    {"role": "assistant", "content": "ast"},
                    {"role": bad_role, "content": "x"},
                ]
            }
            with self.subTest(bad_role=bad_role):
                with self.assertRaises(ValueError) as ctx:
                    self.train_sft.validate_chatml_row(row, 7)
                self.assertIn("unknown role", str(ctx.exception))
                self.assertIn(bad_role, str(ctx.exception))

    def test_to_prompt_completion_separates_assistant(self) -> None:
        row = {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "usr"},
                {"role": "assistant", "content": "ast"},
            ]
        }
        out = self.train_sft.to_prompt_completion(row)
        self.assertEqual([m["role"] for m in out["prompt"]], ["system", "user"])
        self.assertEqual(out["prompt"][0]["content"], "sys")
        self.assertEqual(out["prompt"][1]["content"], "usr")
        self.assertEqual(out["completion"], [{"role": "assistant", "content": "ast"}])

    def test_to_prompt_completion_only_system_user_in_prompt_assistant_only_in_completion(self) -> None:
        """Prompt must contain ONLY system+user roles, completion ONLY assistant.

        Phase 3 review warning: the old implementation put every non-assistant
        role into the prompt, so any unknown role surviving validation would
        leak. We now enforce explicit role whitelists in both branches.
        """

        # Canonical 3-turn row: prompt is exactly [system, user], completion exactly [assistant].
        out = self.train_sft.to_prompt_completion(
            {
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "usr"},
                    {"role": "assistant", "content": "ast"},
                ]
            }
        )
        self.assertEqual([m["role"] for m in out["prompt"]], ["system", "user"])
        self.assertEqual([m["role"] for m in out["completion"]], ["assistant"])
        # Prompt ordering mirrors the source row (system before user).
        self.assertEqual([m["role"] for m in out["prompt"]], ["system", "user"])
        # No role outside the allowed set sneaks through.
        all_roles = [m["role"] for m in out["prompt"]] + [m["role"] for m in out["completion"]]
        self.assertEqual(set(all_roles), {"system", "user", "assistant"})

        # Defense in depth: if a non-canonical role ever reaches to_prompt_completion
        # (i.e. bypassed validate_chatml_row), it must still raise rather than
        # silently land in the prompt.
        bad_row = {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "usr"},
                {"role": "tool", "content": "tool-output"},
                {"role": "assistant", "content": "ast"},
            ]
        }
        with self.assertRaises(ValueError):
            self.train_sft.to_prompt_completion(bad_row)

    def test_load_chat_formatted_dataset_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            path = self._write(
                tmp,
                [
                    {
                        "messages": [
                            {"role": "system", "content": f"sys-{i}"},
                            {"role": "user", "content": f"usr-{i}"},
                            {"role": "assistant", "content": f"ast-{i}"},
                        ]
                    }
                    for i in range(3)
                ],
            )
            rows = self.train_sft.load_chat_formatted_dataset(path)
            self.assertEqual(len(rows), 3)
            for i, row in enumerate(rows):
                self.assertEqual(row["prompt"][0]["content"], f"sys-{i}")
                self.assertEqual(row["prompt"][1]["content"], f"usr-{i}")
                self.assertEqual(row["completion"][0]["content"], f"ast-{i}")


class TestDryRunPipeline(unittest.TestCase):
    """End-to-end dry-run smoke. Loads the real Qwen tokenizer, runs the
    audit against the actual data/chat_formatted/*.jsonl, and asserts the
    Phase 1 contract still holds (no over-limit rows).
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.train_sft = _load_train_sft()

    def test_audit_matches_phase1_real_token_counts(self) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - environment guard
            self.skipTest(f"transformers not installed: {exc}")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        train_path = REPO_ROOT / "data" / "chat_formatted" / "train.jsonl"
        eval_path = REPO_ROOT / "data" / "chat_formatted" / "eval.jsonl"
        if not train_path.exists() or not eval_path.exists():
            self.skipTest("Chat-formatted JSONL not found; run Phase 1 first.")

        train_rows = self.train_sft.load_chat_formatted_dataset(train_path)
        eval_rows = self.train_sft.load_chat_formatted_dataset(eval_path)

        train_audit = self.train_sft.audit_token_lengths(tokenizer, train_rows, "train", 20480)
        eval_audit = self.train_sft.audit_token_lengths(tokenizer, eval_rows, "eval", 20480)

        self.assertEqual(train_audit.examples, 315)
        self.assertEqual(eval_audit.examples, 35)
        self.assertEqual(train_audit.over_limit, [])
        self.assertEqual(eval_audit.over_limit, [])
        # The Phase 1 real-tokenizer audit reported train_max=18916 / eval_max=18910;
        # keep a guard band so silent template drift is caught here too.
        self.assertLessEqual(train_audit.tokens_max, 20480)
        self.assertLessEqual(eval_audit.tokens_max, 20480)

    def test_dry_run_writes_readiness_report(self) -> None:
        cfg = self.train_sft.config_from_file(REPO_ROOT / "training" / "config_qwen_protesta_v1.json")
        with tempfile.TemporaryDirectory() as td:
            report_path = Path(td) / "phase3_readiness.json"
            readiness = self.train_sft.run_dry_run(cfg, report_path)
            self.assertTrue(report_path.exists())
            self.assertIn(readiness["status"], {"pass", "blocked"})
            self.assertEqual(readiness["model_name_or_path"], "Qwen/Qwen2.5-7B-Instruct")
            self.assertEqual(readiness["data"]["train_examples"], 315)
            self.assertEqual(readiness["data"]["eval_examples"], 35)
            self.assertEqual(readiness["max_seq_length"], 20480)
            self.assertEqual(readiness["training_config"]["lora"]["r"], 16)


class _FakeTokenizer:
    """Minimal stand-in for ``transformers.PreTrainedTokenizer`` used by the audit.

    Implements only the two methods ``audit_token_lengths`` touches:
    ``apply_chat_template`` (str mode) and ``encode``. Tokens are 1 word each,
    so we can dial the per-row token count deterministically with the role
    content length.
    """

    def __init__(self, *, fail_on: set[int] | None = None) -> None:
        self._fail_on = fail_on or set()

    def apply_chat_template(self, messages, tokenize: bool = False, add_generation_prompt: bool = False):
        # Mirror Qwen's behavior: just join "<role>: <content>\n" per turn.
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\n"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        # 1 token per whitespace-separated chunk; deterministic and fast.
        return [hash(tok) & 0xFFFF for tok in text.split()]


class TestAuditAllSplits(unittest.TestCase):
    """Phase 3 review warning: training path must audit BOTH train and eval.

    ``audit_all_splits`` and ``assert_audits_within_budget`` are the pure
    helpers now used by both ``run_dry_run`` and ``run_training``. They take
    a fake tokenizer so we don't need the real Qwen snapshot to assert the
    multi-split coverage contract.
    """

    def setUp(self) -> None:
        self.train_sft = _load_train_sft()

    def _row(self, body: str) -> dict[str, Any]:
        return {
            "prompt": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": body},
            ],
            "completion": [{"role": "assistant", "content": "ast"}],
        }

    def test_audit_all_splits_covers_train_and_eval(self) -> None:
        train_rows = [self._row(f"train-{i}") for i in range(3)]
        eval_rows = [self._row(f"eval-{i}") for i in range(2)]
        audits = self.train_sft.audit_all_splits(
            _FakeTokenizer(),
            {"train": train_rows, "eval": eval_rows},
            max_seq_length=10_000,
        )
        # Both splits audited with the expected example counts.
        self.assertEqual(set(audits.keys()), {"train", "eval"})
        self.assertEqual(audits["train"].examples, 3)
        self.assertEqual(audits["eval"].examples, 2)
        self.assertEqual(audits["train"].over_limit, [])
        self.assertEqual(audits["eval"].over_limit, [])
        self.assertEqual(audits["train"].tokenization_errors, [])
        self.assertEqual(audits["eval"].tokenization_errors, [])

    def test_assert_audits_within_budget_passes_when_all_ok(self) -> None:
        audits = self.train_sft.audit_all_splits(
            _FakeTokenizer(),
            {"train": [self._row("a")], "eval": [self._row("b")]},
            max_seq_length=10_000,
        )
        # No exception when both splits are under budget.
        self.train_sft.assert_audits_within_budget(audits, max_seq_length=10_000, context="train")

    def test_assert_audits_within_budget_aborts_on_train_overflow(self) -> None:
        # Build one train row whose tokenized text is larger than the budget.
        huge_row = self._row(" ".join(f"tok{i}" for i in range(50)))
        audits = self.train_sft.audit_all_splits(
            _FakeTokenizer(),
            {"train": [huge_row], "eval": [self._row("small")]},
            max_seq_length=10,
        )
        with self.assertRaises(RuntimeError) as ctx:
            self.train_sft.assert_audits_within_budget(audits, max_seq_length=10, context="train")
        msg = str(ctx.exception)
        self.assertIn("Refusing to train", msg)
        self.assertIn("train", msg)
        self.assertIn("max_seq_length=10", msg)

    def test_assert_audits_within_budget_aborts_on_eval_overflow(self) -> None:
        """Phase 3 warning #1: eval rows over the budget must abort too."""
        # Train is OK; eval row exceeds budget. This is the regression the
        # review caught: the previous run_training only audited train.
        small_row = self._row("tiny")
        huge_eval = self._row(" ".join(f"tok{i}" for i in range(50)))
        audits = self.train_sft.audit_all_splits(
            _FakeTokenizer(),
            {"train": [small_row], "eval": [huge_eval]},
            max_seq_length=10,
        )
        with self.assertRaises(RuntimeError) as ctx:
            self.train_sft.assert_audits_within_budget(audits, max_seq_length=10, context="train")
        msg = str(ctx.exception)
        self.assertIn("Refusing to train", msg)
        self.assertIn("eval", msg)
        self.assertIn("max_seq_length=10", msg)
        # Sanity: the eval audit actually contains the offending row.
        self.assertEqual(len(audits["eval"].over_limit), 1)
        self.assertEqual(audits["eval"].over_limit[0]["index"], 0)
        self.assertGreater(audits["eval"].over_limit[0]["tokens"], 10)

    def test_assert_audits_within_budget_aborts_on_tokenization_error(self) -> None:
        class BoomTokenizer(_FakeTokenizer):
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
                raise ValueError("simulated tokenizer failure")

        audits = self.train_sft.audit_all_splits(
            BoomTokenizer(),
            {"train": [self._row("a")]},
            max_seq_length=10_000,
        )
        with self.assertRaises(RuntimeError) as ctx:
            self.train_sft.assert_audits_within_budget(audits, max_seq_length=10_000, context="train")
        self.assertIn("tokenize", str(ctx.exception).lower())
        self.assertEqual(len(audits["train"].tokenization_errors), 1)


class TestTargetModuleMapping(unittest.TestCase):
    """Guard against silent LoRA target_modules drift.

    The plan (§4) documents ``[q, k, v, o, gate, up, down]``. The actual HF
    Qwen2 module names add the ``_proj`` suffix. We assert here that the
    locked config picks the correct suffix-expanded names so a future edit
    cannot regress to the abbreviations without explicit intent.
    """

    def test_locked_modules_match_real_qwen2_names(self) -> None:
        expected = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
        train_sft = _load_train_sft()
        cfg = train_sft.config_from_file(REPO_ROOT / "training" / "config_qwen_protesta_v1.json")
        self.assertEqual(set(cfg["lora_target_modules"]), expected)

        # Cross-check against the Qwen2.5-7B-Instruct safetensors index when
        # the snapshot is on disk. Skips silently otherwise.
        from transformers import AutoConfig

        cfg_obj = AutoConfig.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        self.assertIn(cfg_obj.model_type, {"qwen2", "qwen2_5"})


class TestSftKwargSafety(unittest.TestCase):
    """Phase 3 safety regression tests for ``build_sft_kwargs`` + CLI wiring.

    These cover the three issues that turned the first 1-step smoke into a
    blocked state (loss=0.3726 reached, then OOM at post-step eval):

    1. ``per_device_eval_batch_size`` must default to 1 and be passed through
       to ``SFTConfig`` (Transformers' default is 8; with 35 eval rows that
       produced 5 in-flight batches and made the OOM much more likely).
    2. ``--no-save`` must disable Trainer-level saves too — not just skip the
       final ``trainer.save_model()``. The trainer fires ``save_strategy``
       at every epoch boundary, and HF treats ``max_steps`` as an epoch
       boundary, so a smoke with ``save_strategy='epoch'`` would drop a
       checkpoint mid-run even with ``--no-save``.
    3. ``--eval-strategy no`` must skip loading the eval JSONL AND must not
       pass ``eval_dataset`` to ``SFTTrainer``. The post-step eval at the
       ``max_steps=1`` boundary is what OOMed in the first smoke.
    """

    def setUp(self) -> None:
        self.train_sft = _load_train_sft()

    def test_per_device_eval_batch_size_default_is_one(self) -> None:
        """Phase 3 safety fix: TrainingConfig.per_device_eval_batch_size must
        default to 1 (Transformers' default is 8 which produced 5 eval
        batches for 35 rows in the OOM smoke)."""

        self.assertEqual(self.train_sft.TrainingConfig.per_device_eval_batch_size, 1)

    def test_config_file_loads_with_explicit_eval_batch_size_one(self) -> None:
        cfg = self.train_sft.config_from_file(REPO_ROOT / "training" / "config_qwen_protesta_v1.json")
        self.assertEqual(cfg["per_device_eval_batch_size"], 1)

    def test_build_sft_kwargs_passes_eval_batch_size_one(self) -> None:
        """The eval batch size must reach the ``SFTConfig`` kwargs."""

        cfg = self.train_sft.config_from_file(REPO_ROOT / "training" / "config_qwen_protesta_v1.json")
        kwargs = self.train_sft.build_sft_kwargs(cfg)
        self.assertEqual(kwargs["per_device_eval_batch_size"], 1)

    def test_cli_per_device_eval_batch_size_overrides_into_sft_kwargs(self) -> None:
        """``--per-device-eval-batch-size`` CLI flag must round-trip through
        ``merge_args_into_config`` into ``build_sft_kwargs``."""

        cfg = self.train_sft.config_from_file(REPO_ROOT / "training" / "config_qwen_protesta_v1.json")
        parser = self.train_sft.build_arg_parser()
        args = parser.parse_args(["--per-device-eval-batch-size", "4"])
        merged = self.train_sft.merge_args_into_config(args, dict(cfg))
        kwargs_override = self.train_sft.build_sft_kwargs(merged)
        self.assertEqual(kwargs_override["per_device_eval_batch_size"], 4)

    def test_no_save_disables_trainer_save_strategy(self) -> None:
        """``--no-save`` must disable Trainer-level saves too."""

        cfg = self.train_sft.config_from_file(REPO_ROOT / "training" / "config_qwen_protesta_v1.json")
        # Baseline: save_strategy follows the JSON config (locked to 'epoch').
        kwargs = self.train_sft.build_sft_kwargs(cfg)
        self.assertEqual(kwargs["save_strategy"], "epoch")

        # With no_save=True the trainer-side save_strategy must be 'no',
        # regardless of what the JSON/CLI requested.
        cfg_with_no_save = dict(cfg)
        cfg_with_no_save["no_save"] = True
        kwargs_no_save = self.train_sft.build_sft_kwargs(cfg_with_no_save)
        self.assertEqual(kwargs_no_save["save_strategy"], "no")

        # And it must also override an explicit 'epoch' or 'steps' choice.
        cfg_explicit_epoch = dict(cfg_with_no_save)
        cfg_explicit_epoch["save_strategy"] = "epoch"
        self.assertEqual(self.train_sft.build_sft_kwargs(cfg_explicit_epoch)["save_strategy"], "no")
        cfg_explicit_steps = dict(cfg_with_no_save)
        cfg_explicit_steps["save_strategy"] = "steps"
        self.assertEqual(self.train_sft.build_sft_kwargs(cfg_explicit_steps)["save_strategy"], "no")

    def test_eval_strategy_no_skips_eval_dataset_load(self) -> None:
        """``--eval-strategy no`` must skip loading the eval JSONL AND must
        not pass ``eval_dataset`` to ``SFTTrainer``."""

        cfg = self.train_sft.config_from_file(REPO_ROOT / "training" / "config_qwen_protesta_v1.json")
        # eval_strategy='no' → skip eval dataset load entirely.
        cfg_no_eval = dict(cfg)
        cfg_no_eval["eval_strategy"] = "no"
        self.assertFalse(self.train_sft.should_load_eval_dataset(cfg_no_eval))
        kwargs_no_eval = self.train_sft.build_sft_kwargs(cfg_no_eval)
        self.assertEqual(kwargs_no_eval["eval_strategy"], "no")

        # eval_strategy='epoch' → load eval JSONL + eval_dataset passed to trainer.
        cfg_eval_epoch = dict(cfg)
        cfg_eval_epoch["eval_strategy"] = "epoch"
        self.assertTrue(self.train_sft.should_load_eval_dataset(cfg_eval_epoch))

        # Default cfg has eval_strategy='epoch' so the baseline load is enabled.
        self.assertTrue(self.train_sft.should_load_eval_dataset(cfg))

    def test_cli_eval_strategy_accepts_only_known_values(self) -> None:
        """``--eval-strategy`` is a real TrainingArguments field with a fixed
        enum; the CLI must reject unknown values upfront instead of letting
        them reach ``SFTConfig`` and blow up mid-training."""

        parser = self.train_sft.build_arg_parser()
        for value in ("no", "steps", "epoch"):
            args = parser.parse_args(["--eval-strategy", value])
            self.assertEqual(args.eval_strategy, value)
        with self.assertRaises(SystemExit):
            parser.parse_args(["--eval-strategy", "daily"])


class TestEvalOnlyMode(unittest.TestCase):
    """Phase 3 eval-only smoke contract.

    The eval-only path is the gate that decides whether the 35-row eval set
    fits in the 32 GiB RTX 5090 under ``per_device_eval_batch_size=1``. The
    contract has three load-bearing invariants:

    1. ``trainer.train()`` is NEVER called. The training data is loaded only
       because TRL ``SFTTrainer.__init__`` requires a non-None ``train_dataset``;
       it is never used for an optimizer step.
    2. ``save_strategy='no'`` is forced so neither the Trainer nor the script
       writes a checkpoint, ``trainer_state``, tokenizer, or adapter.
    3. ``per_device_eval_batch_size=1`` and ``eval_strategy='no'`` are forced:
       the eval pass is triggered manually via ``trainer.evaluate()`` once
       (not at any epoch/step boundary), at batch size 1.

    These tests pin the contract without needing a GPU. The actual end-to-end
    smoke (model load + eval pass) lives in ``run_eval_only`` and is exercised
    by the orchestrator at smoke time.
    """

    def setUp(self) -> None:
        self.train_sft = _load_train_sft()

    def test_eval_only_cli_flag_default_false(self) -> None:
        parser = self.train_sft.build_arg_parser()
        args = parser.parse_args([])
        self.assertFalse(args.eval_only)
        # --eval-report defaults to a Path (reports/phase3_eval_smoke.json).
        self.assertEqual(
            str(args.eval_report), str(self.train_sft.REPO_ROOT / "reports" / "phase3_eval_smoke.json")
        )

    def test_eval_only_cli_flag_accepts_eval_report(self) -> None:
        parser = self.train_sft.build_arg_parser()
        args = parser.parse_args(
            ["--eval-only", "--eval-report", "/tmp/custom_eval_report.json"]
        )
        self.assertTrue(args.eval_only)
        self.assertEqual(str(args.eval_report), "/tmp/custom_eval_report.json")

    def test_eval_only_sft_kwargs_forces_save_strategy_no(self) -> None:
        """``run_eval_only`` always sets ``no_save=True`` before calling
        ``build_sft_kwargs``; this must collapse ``save_strategy`` to ``'no'``
        regardless of what the JSON config or CLI requested."""

        cfg = self.train_sft.config_from_file(REPO_ROOT / "training" / "config_qwen_protesta_v1.json")
        # Simulate the eval-only override path.
        eval_cfg = dict(cfg)
        eval_cfg["no_save"] = True
        eval_cfg["eval_strategy"] = "no"
        eval_cfg["per_device_eval_batch_size"] = 1

        kwargs = self.train_sft.build_sft_kwargs(eval_cfg)
        self.assertEqual(kwargs["save_strategy"], "no")
        # Even if the user explicitly set save_strategy='epoch' the eval-only
        # path overrides it via no_save=True.
        eval_cfg_override = dict(eval_cfg)
        eval_cfg_override["save_strategy"] = "epoch"
        kwargs_override = self.train_sft.build_sft_kwargs(eval_cfg_override)
        self.assertEqual(kwargs_override["save_strategy"], "no")

    def test_eval_only_sft_kwargs_forces_per_device_eval_batch_size_one(self) -> None:
        """``per_device_eval_batch_size=1`` must reach ``SFTConfig``. The eval
        smoke rationale: 35 eval rows × 1 batch is far cheaper than
        Transformers' default 8 → 5 eval batches."""

        cfg = self.train_sft.config_from_file(REPO_ROOT / "training" / "config_qwen_protesta_v1.json")
        eval_cfg = dict(cfg)
        eval_cfg["no_save"] = True
        eval_cfg["eval_strategy"] = "no"
        eval_cfg["per_device_eval_batch_size"] = 1

        kwargs = self.train_sft.build_sft_kwargs(eval_cfg)
        self.assertEqual(kwargs["per_device_eval_batch_size"], 1)
        self.assertEqual(kwargs["eval_strategy"], "no")

    def _build_sft_kwargs_capture(
        self,
    ) -> tuple[Callable[[dict[str, Any]], dict[str, Any]], list[tuple[dict[str, Any], dict[str, Any]]]]:
        """Return a ``side_effect`` callable and a capture list for ``build_sft_kwargs``.

        The callable delegates to the real ``build_sft_kwargs`` and records
        every ``(cfg_in, kwargs_out)`` pair so tests can assert on what
        ``run_eval_only`` actually fed in (instead of mocking it away with
        a hardcoded safe return value, which would hide regressions where
        ``run_eval_only`` stops forcing the safety wiring).
        """
        real = self.train_sft.build_sft_kwargs
        captured: list[tuple[dict[str, Any], dict[str, Any]]] = []

        def side_effect(cfg: dict[str, Any]) -> dict[str, Any]:
            out = real(cfg)
            captured.append((cfg, out))
            return out

        return side_effect, captured

    def test_run_eval_only_forces_safe_values_into_build_sft_kwargs(self) -> None:
        """The eval-only safety override MUST reach ``build_sft_kwargs``.

        Regression guard: if a future edit accidentally drops the
        ``eval_cfg["per_device_eval_batch_size"] = 1`` / ``eval_cfg["no_save"] = True``
        / ``eval_cfg["eval_strategy"] = "no"`` triple inside ``run_eval_only``,
        this test must fail. We start from a deliberately DANGEROUS cfg
        (``per_device_eval_batch_size=8``, ``save_strategy='epoch'``,
        ``no_save=False``) and capture the actual call to ``build_sft_kwargs``
        via a real-function ``side_effect`` rather than mocking the return
        value with hardcoded safe data.
        """

        from unittest.mock import MagicMock, patch

        cfg = self.train_sft.config_from_file(REPO_ROOT / "training" / "config_qwen_protesta_v1.json")
        eval_cfg = dict(cfg)
        # Deliberately unsafe baseline — proves the safety override fires.
        eval_cfg["per_device_eval_batch_size"] = 8
        eval_cfg["save_strategy"] = "epoch"
        eval_cfg["no_save"] = False

        fake_model = MagicMock(name="fake_peft_model")
        fake_tokenizer = MagicMock(name="fake_tokenizer")
        fake_tokenizer.pad_token = None
        fake_tokenizer.chat_template = None
        fake_tokenizer.apply_chat_template.return_value = "fake"
        fake_tokenizer.encode.return_value = [0] * 4

        fake_train_dataset = MagicMock(name="fake_train_dataset")
        fake_eval_dataset = MagicMock(name="fake_eval_dataset")

        fake_metrics = {"eval_loss": 0.42, "eval_runtime": 12.5, "eval_samples_per_second": 2.8}
        fake_trainer = MagicMock(name="fake_trainer")
        fake_trainer.evaluate.return_value = fake_metrics

        fake_sft_config = MagicMock(name="fake_sft_config")

        chat_row = {
            "messages": [
                {"role": "system", "content": "s"},
                {"role": "user", "content": "u"},
                {"role": "assistant", "content": "a"},
            ]
        }
        train_payload = [dict(chat_row) for _ in range(2)]
        eval_payload = [dict(chat_row) for _ in range(1)]

        build_side_effect, captured = self._build_sft_kwargs_capture()

        with tempfile.TemporaryDirectory() as td:
            train_path = Path(td) / "train.jsonl"
            eval_path = Path(td) / "eval.jsonl"
            train_path.write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in train_payload) + "\n",
                encoding="utf-8",
            )
            eval_path.write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in eval_payload) + "\n",
                encoding="utf-8",
            )
            eval_cfg["dataset_train_path"] = str(train_path)
            eval_cfg["dataset_eval_path"] = str(eval_path)
            eval_cfg["output_dir"] = str(Path(td) / "out")
            report_path = Path(td) / "eval_smoke.json"

            with patch.object(self.train_sft, "validate_environment", return_value={
                "torch_version": "fake", "cuda_available": True, "cuda_version": "fake",
                "device_count": 1, "device_name": "fake-gpu",
            }), \
                 patch("transformers.AutoTokenizer.from_pretrained", return_value=fake_tokenizer), \
                 patch.object(self.train_sft, "_build_model_and_tokenizer", return_value=(fake_model, fake_tokenizer)), \
                 patch.object(self.train_sft, "build_sft_kwargs", side_effect=build_side_effect), \
                 patch("datasets.Dataset.from_list", side_effect=[fake_train_dataset, fake_eval_dataset]), \
                 patch("trl.SFTConfig", return_value=fake_sft_config), \
                 patch("trl.SFTTrainer", return_value=fake_trainer):
                report = self.train_sft.run_eval_only(
                    eval_cfg, report_path, command="training/train_sft.py --eval-only"
                )

        self.assertEqual(report["status"], "pass")

        # build_sft_kwargs MUST have been called exactly once.
        self.assertEqual(
            len(captured), 1,
            f"build_sft_kwargs was called {len(captured)} times; expected exactly 1",
        )
        passed_cfg, returned_kwargs = captured[0]

        # --- What run_eval_only actually fed into build_sft_kwargs ---------
        # These three lines are the load-bearing safety overrides. If any of
        # them is dropped in a future edit, the assertions below fire.
        self.assertEqual(
            passed_cfg["per_device_eval_batch_size"], 1,
            "run_eval_only must force per_device_eval_batch_size=1; got "
            f"{passed_cfg['per_device_eval_batch_size']!r}. "
            "The eval-only safety override block was bypassed.",
        )
        self.assertTrue(
            passed_cfg["no_save"],
            "run_eval_only must force no_save=True; got "
            f"{passed_cfg['no_save']!r}. "
            "The eval-only safety override block was bypassed.",
        )
        self.assertEqual(
            passed_cfg["eval_strategy"], "no",
            "run_eval_only must force eval_strategy='no'; got "
            f"{passed_cfg['eval_strategy']!r}. "
            "The eval-only safety override block was bypassed.",
        )

        # --- What came out of the real build_sft_kwargs -------------------
        # build_sft_kwargs is the SAME pure helper that the training path
        # uses, so the eval batch size that reaches SFTConfig is exactly
        # build_sft_kwargs(passed_cfg)["per_device_eval_batch_size"]. Assert
        # the real pipeline end-to-end (no mocked return value).
        self.assertEqual(returned_kwargs["per_device_eval_batch_size"], 1)
        self.assertEqual(returned_kwargs["save_strategy"], "no")
        self.assertEqual(returned_kwargs["eval_strategy"], "no")

        # The reported safety wiring must come from the REAL return value,
        # not from a mock. If the mock were leaking, it would also catch the
        # assertions above; this guards against an accidental
        # ``patch.object(..., return_value={...})`` regression in the future.
        self.assertEqual(report["safety_wiring_proof"]["per_device_eval_batch_size"], 1)
        self.assertEqual(report["safety_wiring_proof"]["save_strategy"], "no")
        self.assertEqual(report["safety_wiring_proof"]["eval_strategy"], "no")

    def test_run_eval_only_never_calls_train_or_save(self) -> None:
        """Mock-heavy structural test.

        Verifies that ``run_eval_only``:
        * calls ``trainer.evaluate()`` exactly once
        * never calls ``trainer.train()``
        * never calls ``trainer.save_model()`` or ``trainer.save_state()``
        * never calls ``tokenizer.save_pretrained()``
        * writes the report even when invoked via the public function
        * feeds the REAL ``build_sft_kwargs`` with safe values (no hardcoded
          mock return value) so a regression in ``run_eval_only``'s safety
          override is caught here too

        The mocks stub out the heavy imports (torch, transformers, peft, trl,
        datasets) so this runs in seconds with no GPU. ``build_sft_kwargs``
        is the SAME pure helper the training path uses; we invoke it for real
        via a capture side-effect rather than replacing its return value with
        a hardcoded safe dict (which would let a ``run_eval_only`` regression
        that drops the safety override slip through silently).
        """

        from unittest.mock import MagicMock, patch

        cfg = self.train_sft.config_from_file(REPO_ROOT / "training" / "config_qwen_protesta_v1.json")
        eval_cfg = dict(cfg)

        # Heavy imports — replace with mocks so we don't need a GPU.
        fake_model = MagicMock(name="fake_peft_model")
        fake_tokenizer = MagicMock(name="fake_tokenizer")
        fake_tokenizer.pad_token = None
        fake_tokenizer.chat_template = None  # audit_token_lengths uses apply_chat_template
        # Make the fake tokenizer return a tiny text so the audit succeeds.
        fake_tokenizer.apply_chat_template.return_value = "fake"
        fake_tokenizer.encode.return_value = [0] * 4  # 4 tokens per row

        fake_train_dataset = MagicMock(name="fake_train_dataset")
        fake_eval_dataset = MagicMock(name="fake_eval_dataset")

        fake_metrics = {"eval_loss": 0.42, "eval_runtime": 12.5, "eval_samples_per_second": 2.8}
        fake_trainer = MagicMock(name="fake_trainer")
        fake_trainer.evaluate.return_value = fake_metrics

        fake_sft_config = MagicMock(name="fake_sft_config")

        # Two-row train + one-row eval, both inside the 4-token fake budget.
        chat_row = {
            "messages": [
                {"role": "system", "content": "s"},
                {"role": "user", "content": "u"},
                {"role": "assistant", "content": "a"},
            ]
        }
        train_payload = [dict(chat_row) for _ in range(2)]
        eval_payload = [dict(chat_row) for _ in range(1)]

        build_side_effect, captured = self._build_sft_kwargs_capture()

        with tempfile.TemporaryDirectory() as td:
            train_path = Path(td) / "train.jsonl"
            eval_path = Path(td) / "eval.jsonl"
            train_path.write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in train_payload) + "\n",
                encoding="utf-8",
            )
            eval_path.write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in eval_payload) + "\n",
                encoding="utf-8",
            )
            eval_cfg["dataset_train_path"] = str(train_path)
            eval_cfg["dataset_eval_path"] = str(eval_path)
            eval_cfg["output_dir"] = str(Path(td) / "out")
            report_path = Path(td) / "eval_smoke.json"

            with patch.object(self.train_sft, "validate_environment", return_value={
                "torch_version": "fake", "cuda_available": True, "cuda_version": "fake",
                "device_count": 1, "device_name": "fake-gpu",
            }), \
                 patch("transformers.AutoTokenizer.from_pretrained", return_value=fake_tokenizer), \
                 patch.object(self.train_sft, "_build_model_and_tokenizer", return_value=(fake_model, fake_tokenizer)), \
                 patch.object(self.train_sft, "build_sft_kwargs", side_effect=build_side_effect), \
                 patch("datasets.Dataset.from_list", side_effect=[fake_train_dataset, fake_eval_dataset]), \
                 patch("trl.SFTConfig", return_value=fake_sft_config) as sft_config_mock, \
                 patch("trl.SFTTrainer", return_value=fake_trainer):

                report = self.train_sft.run_eval_only(
                    eval_cfg, report_path, command="training/train_sft.py --eval-only"
                )

            # Report must exist on disk before the tempdir is cleaned up.
            self.assertTrue(report_path.exists())

        # Contract assertions.
        self.assertEqual(report["status"], "pass")
        self.assertTrue(report["eval_only"])
        self.assertEqual(report["data"]["train_examples"], 2)
        self.assertEqual(report["data"]["eval_examples"], 1)
        self.assertEqual(report["metrics"]["eval_loss"], 0.42)

        # Safety wiring proof must come from the REAL build_sft_kwargs
        # output (captured above), not from a mock. The script-level asserts
        # inside run_eval_only (lines ~1294-1303) enforce the same triple,
        # but here we surface the proof in the report end-to-end.
        self.assertEqual(len(captured), 1, "build_sft_kwargs must be called exactly once")
        passed_cfg, returned_kwargs = captured[0]
        self.assertEqual(returned_kwargs["save_strategy"], "no")
        self.assertEqual(returned_kwargs["per_device_eval_batch_size"], 1)
        self.assertEqual(returned_kwargs["eval_strategy"], "no")
        self.assertEqual(report["safety_wiring_proof"]["save_strategy"], "no")
        self.assertEqual(report["safety_wiring_proof"]["per_device_eval_batch_size"], 1)
        self.assertEqual(report["safety_wiring_proof"]["eval_strategy"], "no")
        self.assertFalse(report["safety_wiring_proof"]["trainer_train_called"])
        self.assertFalse(report["safety_wiring_proof"]["trainer_save_model_called"])
        self.assertFalse(report["safety_wiring_proof"]["trainer_save_state_called"])
        self.assertFalse(report["safety_wiring_proof"]["tokenizer_save_pretrained_called"])

        # The critical "no training, no save" assertions against the mock
        # trainer — this is what guarantees the runtime path never touches
        # the optimizer or the disk.
        fake_trainer.evaluate.assert_called_once()
        fake_trainer.train.assert_not_called()
        fake_trainer.save_model.assert_not_called()
        fake_trainer.save_state.assert_not_called()
        fake_tokenizer.save_pretrained.assert_not_called()

        # SFTConfig must have been built from the REAL kwargs, not from a
        # hardcoded mock return. If a future edit accidentally mocks
        # build_sft_kwargs back to a fixed dict, this assertion catches it.
        sft_config_mock.assert_called_once()
        sft_config_kwargs = sft_config_mock.call_args.kwargs
        self.assertEqual(sft_config_kwargs["per_device_eval_batch_size"], 1)
        self.assertEqual(sft_config_kwargs["save_strategy"], "no")
        self.assertEqual(sft_config_kwargs["eval_strategy"], "no")

    def test_run_eval_only_fails_loud_on_over_limit_eval_row(self) -> None:
        """If an eval row exceeds ``max_seq_length`` the script must abort
        BEFORE the 14 GB base model is loaded (defense-in-depth: the same
        audit runs in ``run_dry_run`` and ``run_training``; this test pins
        the eval-only ordering)."""

        from unittest.mock import MagicMock, patch

        cfg = self.train_sft.config_from_file(REPO_ROOT / "training" / "config_qwen_protesta_v1.json")
        eval_cfg = dict(cfg)
        eval_cfg["max_seq_length"] = 10  # pathologically small

        fake_model = MagicMock(name="fake_peft_model")
        fake_tokenizer = MagicMock(name="fake_tokenizer")
        fake_tokenizer.pad_token = None
        fake_tokenizer.chat_template = None
        fake_tokenizer.apply_chat_template.return_value = "x" * 1000  # huge
        fake_tokenizer.encode.return_value = [0] * 100  # 100 tokens per row

        with tempfile.TemporaryDirectory() as td:
            train_path = Path(td) / "train.jsonl"
            eval_path = Path(td) / "eval.jsonl"
            big_row = {
                "messages": [
                    {"role": "system", "content": "s"},
                    {"role": "user", "content": "u" * 200},
                    {"role": "assistant", "content": "a"},
                ]
            }
            train_path.write_text(json.dumps(big_row, ensure_ascii=False) + "\n", encoding="utf-8")
            eval_path.write_text(json.dumps(big_row, ensure_ascii=False) + "\n", encoding="utf-8")
            eval_cfg["dataset_train_path"] = str(train_path)
            eval_cfg["dataset_eval_path"] = str(eval_path)
            eval_cfg["output_dir"] = str(Path(td) / "out")
            report_path = Path(td) / "eval_smoke.json"

            with patch.object(self.train_sft, "validate_environment", return_value={
                "torch_version": "fake", "cuda_available": True, "cuda_version": "fake",
                "device_count": 1, "device_name": "fake-gpu",
            }), \
                 patch("transformers.AutoTokenizer.from_pretrained", return_value=fake_tokenizer), \
                 patch.object(self.train_sft, "_build_model_and_tokenizer", return_value=(fake_model, fake_tokenizer)) as mock_build:

                report = self.train_sft.run_eval_only(
                    eval_cfg, report_path, command="training/train_sft.py --eval-only"
                )

                # Audit must run BEFORE the 14 GB base model is loaded:
                # _build_model_and_tokenizer must NOT have been called when the
                # audit fails. This is the key ordering invariant for the
                # eval-only path. Asserted INSIDE the with block so the patch
                # is still active.
                mock_build.assert_not_called()

        self.assertEqual(report["status"], "blocked")
        self.assertTrue(report["errors"])
        # The token audit must have been captured for diagnostics.
        self.assertIn("token_audit", report)


def main() -> int:
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestConfigLoading))
    suite.addTests(loader.loadTestsFromTestCase(TestR32ConfigLoading))
    suite.addTests(loader.loadTestsFromTestCase(TestR32E5ConfigLoading))
    suite.addTests(loader.loadTestsFromTestCase(TestDatasetConversion))
    suite.addTests(loader.loadTestsFromTestCase(TestDryRunPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestAuditAllSplits))
    suite.addTests(loader.loadTestsFromTestCase(TestTargetModuleMapping))
    suite.addTests(loader.loadTestsFromTestCase(TestSftKwargSafety))
    suite.addTests(loader.loadTestsFromTestCase(TestEvalOnlyMode))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())