#!/usr/bin/env python3
"""Phase 3 — QLoRA SFT training for the protest-event extractor on Qwen2.5-7B-Instruct.

This script is the operational entry point for PLAN_ENTRENAMIENTO_QWEN.md §Fase 3.
It targets TRL ``SFTTrainer`` + ``SFTConfig`` with a BitsAndBytes 4-bit (NF4, double
quant) base model and a PEFT LoRA adapter, fine-tuned on the canonical Phase 1
ChatML train split.

Three modes are supported:

* ``--dry-run`` — loads the tokenizer only, converts the JSONL rows from
  ``messages`` to a prompt-completion dataset, applies the Qwen ChatML chat
  template, audits token counts against ``max_seq_length``, validates the
  resolved training configuration, and writes ``reports/phase3_readiness.json``.
  **No base model is loaded and no training step is performed.**
* ``--eval-only`` — loads the base model with QLoRA, instantiates
  ``SFTTrainer`` with the eval split (the train split is loaded only to
  satisfy TRL ``SFTTrainer``'s required ``train_dataset`` kwarg), and calls
  ``trainer.evaluate()`` exactly once. ``trainer.train()`` is **never**
  called; no checkpoint, no ``trainer_state``, no tokenizer, no adapter is
  written. ``per_device_eval_batch_size=1`` and ``save_strategy='no'`` are
  forced. Writes ``reports/phase3_eval_smoke.json`` with metrics.
* default — loads the base model with QLoRA, instantiates ``SFTTrainer``, and
  runs the full training loop. Saving and loading require a real GPU with
  bfloat16 and 4-bit support.

Methodology reminders (must be preserved across regenerations):

* The canonical training data contains 317 gold rows (weight 1.0), split as
  285 train / 32 eval in the current Phase 1 artifacts. It was produced by
  **GPT-5.4-mini + validación humana Nico**. Do **not** regress to ``gpt-5.5``
  as the origin/baseline.
* The user message in ``data/chat_formatted/*.jsonl`` already embeds the full
  ``FECHA DE EDICIÓN DE LA NOTA + título + cuerpo``. Do not re-concatenate
  fecha/título/texto in this script.
* Truncation is opt-in. By default, any example that exceeds
  ``max_seq_length=20480`` aborts the run; raise the limit up to the model's
  32k native cap or fix the offending example instead of silently dropping
  tokens.

Usage:

    # Dry-run / readiness gate (no GPU work for the base model)
    .venv/bin/python training/train_sft.py --dry-run

    # Eval-only smoke (GPU, no training, no save). Tests whether the eval
    # eval set fits in memory under per_device_eval_batch_size=1.
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
        .venv/bin/python training/train_sft.py \\
            --eval-only --output-dir checkpoints/qwen-protesta-eval-smoke \\
            --no-save --per-device-eval-batch-size 1 --report-to none

    # Full training (GPU, will write checkpoints)
    .venv/bin/python training/train_sft.py

    # 1-step smoke (use only if dry-run passed; explicit opt-in)
    .venv/bin/python training/train_sft.py --max-steps 1 --output-dir checkpoints/qwen-protesta-smoke

See ``--help`` for every flag.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "training" / "config_qwen_protesta_v1.json"
DEFAULT_READINESS = REPO_ROOT / "reports" / "phase3_readiness.json"

# Locked LoRA target modules for Qwen2.5-7B-Instruct (HF module names with the
# ``_proj`` suffix). The plan §4 documents the abbreviations [q,k,v,o,gate,up,down];
# the actual names add ``_proj``. See training/config_qwen_protesta_v1.json
# ``_target_modules_note`` for the audit trail.
LOCKED_LORA_TARGET_MODULES: list[str] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


# ---------------------------------------------------------------------------
# Config container
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Mirror of the locked Phase 3 hyperparameters.

    Every field can be overridden from the CLI; see :func:`build_arg_parser`.
    The defaults match ``training/config_qwen_protesta_v1.json`` and
    ``PLAN_ENTRENAMIENTO_QWEN.md`` §4.
    """

    model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct"
    dataset_train_path: str = "data/chat_formatted/train.jsonl"
    dataset_eval_path: str = "data/chat_formatted/eval.jsonl"
    output_dir: str = "checkpoints/qwen-protesta-v1"

    max_seq_length: int = 20480
    packing: bool = False
    completion_only_loss: bool = True

    dtype: str = "bfloat16"
    quantization: str = "nf4_double_quant"
    device_map: str = "auto"
    trust_remote_code: bool = False

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    lora_target_modules: list[str] = field(
        default_factory=lambda: list(LOCKED_LORA_TARGET_MODULES)
    )

    num_train_epochs: float = 3.0
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 24
    learning_rate: float = 2.0e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    weight_decay: float = 0.0
    optim: str = "paged_adamw_8bit"
    gradient_checkpointing: bool = True
    bf16: bool = True
    max_grad_norm: float = 1.0

    save_strategy: str = "epoch"
    save_total_limit: int = 3
    logging_steps: int = 1
    eval_strategy: str = "epoch"
    load_best_model_at_end: bool = False
    report_to: str = "none"
    dataloader_num_workers: int = 0
    remove_unused_columns: bool = True

    seed: int = 42
    max_steps: int = -1
    resume_from_checkpoint: str | None = None
    save_steps: int | None = None
    eval_steps: int | None = None
    gradient_checkpointing_kwargs: dict[str, Any] | None = None

    # Mutually exclusive with batch mode; only honored when not in --dry-run.
    no_save: bool = False


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def config_from_file(path: Path) -> dict[str, Any]:
    """Translate the JSON config schema to :class:`TrainingConfig` kwargs."""

    raw = load_config(path)
    lora = raw.get("lora", {})
    training = raw.get("training", {})
    model_load = raw.get("model_load", {})

    cfg = TrainingConfig(
        model_name_or_path=raw.get("model_name_or_path", TrainingConfig.model_name_or_path),
        dataset_train_path=raw.get("dataset_train_path", TrainingConfig.dataset_train_path),
        dataset_eval_path=raw.get("dataset_eval_path", TrainingConfig.dataset_eval_path),
        output_dir=raw.get("output_dir", TrainingConfig.output_dir),
        max_seq_length=raw.get("max_seq_length", TrainingConfig.max_seq_length),
        packing=raw.get("packing", TrainingConfig.packing),
        completion_only_loss=raw.get("completion_only_loss", TrainingConfig.completion_only_loss),
        dtype=model_load.get("dtype", TrainingConfig.dtype),
        quantization=model_load.get("quantization", TrainingConfig.quantization),
        device_map=model_load.get("device_map", TrainingConfig.device_map),
        trust_remote_code=model_load.get("trust_remote_code", TrainingConfig.trust_remote_code),
        lora_r=lora.get("r", TrainingConfig.lora_r),
        lora_alpha=lora.get("lora_alpha", TrainingConfig.lora_alpha),
        lora_dropout=lora.get("lora_dropout", TrainingConfig.lora_dropout),
        lora_bias=lora.get("bias", TrainingConfig.lora_bias),
        lora_target_modules=list(lora.get("target_modules", LOCKED_LORA_TARGET_MODULES)),
        num_train_epochs=training.get("num_train_epochs", TrainingConfig.num_train_epochs),
        per_device_train_batch_size=training.get(
            "per_device_train_batch_size", TrainingConfig.per_device_train_batch_size
        ),
        per_device_eval_batch_size=training.get(
            "per_device_eval_batch_size", TrainingConfig.per_device_eval_batch_size
        ),
        gradient_accumulation_steps=training.get(
            "gradient_accumulation_steps", TrainingConfig.gradient_accumulation_steps
        ),
        learning_rate=training.get("learning_rate", TrainingConfig.learning_rate),
        lr_scheduler_type=training.get("lr_scheduler_type", TrainingConfig.lr_scheduler_type),
        warmup_ratio=training.get("warmup_ratio", TrainingConfig.warmup_ratio),
        weight_decay=training.get("weight_decay", TrainingConfig.weight_decay),
        optim=training.get("optim", TrainingConfig.optim),
        gradient_checkpointing=training.get(
            "gradient_checkpointing", TrainingConfig.gradient_checkpointing
        ),
        bf16=training.get("bf16", TrainingConfig.bf16),
        max_grad_norm=training.get("max_grad_norm", TrainingConfig.max_grad_norm),
        save_strategy=training.get("save_strategy", TrainingConfig.save_strategy),
        save_total_limit=training.get("save_total_limit", TrainingConfig.save_total_limit),
        logging_steps=training.get("logging_steps", TrainingConfig.logging_steps),
        eval_strategy=training.get("eval_strategy", TrainingConfig.eval_strategy),
        load_best_model_at_end=training.get(
            "load_best_model_at_end", TrainingConfig.load_best_model_at_end
        ),
        report_to=training.get("report_to", TrainingConfig.report_to),
        dataloader_num_workers=training.get(
            "dataloader_num_workers", TrainingConfig.dataloader_num_workers
        ),
        remove_unused_columns=training.get(
            "remove_unused_columns", TrainingConfig.remove_unused_columns
        ),
        seed=raw.get("seed", TrainingConfig.seed),
    )
    return asdict(cfg)


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="JSON config file with locked Phase 3 defaults (default: training/config_qwen_protesta_v1.json).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the readiness audit only. Loads tokenizer, converts the dataset, audits tokens, writes reports/phase3_readiness.json. Does NOT load the base model.",
    )
    parser.add_argument(
        "--readiness-report",
        type=Path,
        default=DEFAULT_READINESS,
        help="Path to write the dry-run readiness report (default: reports/phase3_readiness.json).",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Eval-only smoke: build the model + PEFT + SFTTrainer exactly as for "
        "training, then call trainer.evaluate() once. trainer.train() is NEVER "
        "called. Forces per_device_eval_batch_size=1, eval_strategy='no' (we "
        "trigger eval manually), and save_strategy='no' so nothing is written "
        "to output_dir. Writes reports/phase3_eval_smoke.json with metrics. "
        "Use this to test whether the full eval set fits in memory before "
        "launching a real training run with eval enabled.",
    )
    parser.add_argument(
        "--eval-report",
        type=Path,
        default=REPO_ROOT / "reports" / "phase3_eval_smoke.json",
        help="Path to write the --eval-only report (default: reports/phase3_eval_smoke.json).",
    )

    # Model + data overrides
    parser.add_argument("--model-name-or-path", dest="model_name_or_path")
    parser.add_argument("--dataset-train-path", dest="dataset_train_path")
    parser.add_argument("--dataset-eval-path", dest="dataset_eval_path")
    parser.add_argument("--output-dir", dest="output_dir")
    parser.add_argument("--cache-dir", default=None, help="Optional Hugging Face cache dir.")

    # Sequence / loss
    parser.add_argument("--max-seq-length", dest="max_seq_length", type=int)
    parser.add_argument("--no-completion-only-loss", dest="completion_only_loss", action="store_false")
    parser.add_argument("--enable-packing", dest="packing", action="store_true")

    # LoRA
    parser.add_argument("--lora-r", dest="lora_r", type=int)
    parser.add_argument("--lora-alpha", dest="lora_alpha", type=int)
    parser.add_argument("--lora-dropout", dest="lora_dropout", type=float)
    parser.add_argument(
        "--lora-target-modules",
        dest="lora_target_modules",
        nargs="+",
        help="Override LoRA target_modules (defaults from config).",
    )

    # Training loop
    parser.add_argument("--num-train-epochs", dest="num_train_epochs", type=float)
    parser.add_argument("--per-device-train-batch-size", dest="per_device_train_batch_size", type=int)
    parser.add_argument("--per-device-eval-batch-size", dest="per_device_eval_batch_size", type=int)
    parser.add_argument("--gradient-accumulation-steps", dest="gradient_accumulation_steps", type=int)
    parser.add_argument("--learning-rate", dest="learning_rate", type=float)
    parser.add_argument("--lr-scheduler-type", dest="lr_scheduler_type")
    parser.add_argument("--warmup-ratio", dest="warmup_ratio", type=float)
    parser.add_argument("--weight-decay", dest="weight_decay", type=float)
    parser.add_argument("--optim")
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--no-bf16", dest="bf16", action="store_false")
    parser.add_argument("--max-grad-norm", dest="max_grad_norm", type=float)
    parser.add_argument("--max-steps", dest="max_steps", type=int)
    parser.add_argument("--resume-from-checkpoint", dest="resume_from_checkpoint")
    parser.add_argument("--save-steps", dest="save_steps", type=int)
    parser.add_argument("--eval-steps", dest="eval_steps", type=float)
    parser.add_argument("--logging-steps", dest="logging_steps", type=int)
    parser.add_argument("--save-strategy", dest="save_strategy", choices=["no", "steps", "epoch"])
    parser.add_argument(
        "--eval-strategy",
        dest="eval_strategy",
        choices=["no", "steps", "epoch"],
        help="HF Trainer eval_strategy. Use 'no' for smoke runs so the trainer "
        "does not trigger eval after max_steps=1 (which caused OOM in the "
        "first 1-step smoke). When 'no', --dataset-eval-path is not loaded.",
    )
    parser.add_argument("--report-to", dest="report_to")
    parser.add_argument("--seed", dest="seed", type=int)
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip persisting trainer/model to output_dir (useful for quick smoke runs).",
    )

    return parser


def merge_args_into_config(args: argparse.Namespace, cfg: dict[str, Any]) -> dict[str, Any]:
    """Merge explicit CLI overrides into a loaded config dict.

    We rely on argparse storing the default for booleans, so the user has to
    actively pass --no-X to flip them. Anything they didn't pass stays at the
    config-file value.
    """
    merged = dict(cfg)
    scalar_keys = [
        "model_name_or_path",
        "dataset_train_path",
        "dataset_eval_path",
        "output_dir",
        "max_seq_length",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "num_train_epochs",
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "lr_scheduler_type",
        "warmup_ratio",
        "weight_decay",
        "optim",
        "max_grad_norm",
        "max_steps",
        "resume_from_checkpoint",
        "save_steps",
        "eval_steps",
        "logging_steps",
        "save_strategy",
        "eval_strategy",
        "report_to",
        "seed",
    ]
    for key in scalar_keys:
        cli_val = getattr(args, key, None)
        if cli_val is not None:
            merged[key] = cli_val

    # Booleans (user explicitly flipped them by passing --no-X or --enable-packing)
    merged["completion_only_loss"] = bool(args.completion_only_loss)
    merged["packing"] = bool(args.packing)
    merged["gradient_checkpointing"] = bool(args.gradient_checkpointing)
    merged["bf16"] = bool(args.bf16)

    # List overrides
    if args.lora_target_modules:
        merged["lora_target_modules"] = list(args.lora_target_modules)

    merged["no_save"] = bool(args.no_save)
    return merged


# ---------------------------------------------------------------------------
# Dataset conversion + audit (used by both dry-run and training)
# ---------------------------------------------------------------------------


REQUIRED_ROLES = ("system", "user", "assistant")
ALLOWED_ROLES: frozenset[str] = frozenset({"system", "user", "assistant"})
PROMPT_ROLES: tuple[str, ...] = ("system", "user")


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


def validate_chatml_row(row: dict[str, Any], line_no: int) -> None:
    """Hard-fail on malformed ChatML rows. Used by both audit and training."""

    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"line {line_no}: 'messages' must be a list, got {type(messages).__name__}")
    roles = [m.get("role") for m in messages]
    # Reject unknown roles first so a stray "tool"/"function"/"developer" turn
    # never leaks into the prompt. The training data contract is exactly one
    # of each required role and nothing else.
    bad_roles = [r for r in roles if r not in ALLOWED_ROLES]
    if bad_roles:
        raise ValueError(
            f"line {line_no}: unknown role(s) {sorted(set(bad_roles))!r}; "
            f"only {sorted(ALLOWED_ROLES)} are allowed; got roles={roles}"
        )
    for required in REQUIRED_ROLES:
        if required not in roles:
            raise ValueError(f"line {line_no}: missing role {required!r}; got roles={roles}")
    if roles.count("system") != 1:
        raise ValueError(f"line {line_no}: expected exactly 1 'system' role; got {roles.count('system')}")
    if roles.count("user") != 1:
        raise ValueError(f"line {line_no}: expected exactly 1 'user' role; got {roles.count('user')}")
    if roles.count("assistant") != 1:
        raise ValueError(
            f"line {line_no}: expected exactly 1 'assistant' role; got {roles.count('assistant')}"
        )
    for msg in messages:
        if not isinstance(msg.get("content"), str):
            raise ValueError(
                f"line {line_no}: role {msg.get('role')!r} has non-string content "
                f"({type(msg.get('content')).__name__})"
            )
        if not msg["content"]:
            raise ValueError(f"line {line_no}: role {msg.get('role')!r} has empty content")


def to_prompt_completion(row: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    """Convert a ChatML ``messages`` row to TRL's prompt-completion schema.

    The user message in ``data/chat_formatted/*.jsonl`` already embeds
    ``FECHA DE EDICIÓN DE LA NOTA + título + cuerpo`` (``nota.texto_original``).
    We do **not** reconstruct or duplicate any of that here.

    Only ``system`` and ``user`` turns go into ``prompt`` (in original order);
    ``assistant`` goes into ``completion``. Any other role is rejected here as
    defense in depth — ``validate_chatml_row`` should have caught it already.
    """

    prompt: list[dict[str, str]] = []
    completion: list[dict[str, str]] = []
    for msg in row["messages"]:
        role = msg["role"]
        content = msg["content"]
        if role == "assistant":
            completion.append({"role": "assistant", "content": content})
        elif role in PROMPT_ROLES:
            prompt.append({"role": role, "content": content})
        else:
            raise ValueError(
                f"unexpected role {role!r} in to_prompt_completion; only "
                f"{list(PROMPT_ROLES)} are allowed in prompt and 'assistant' in completion"
            )
    if not prompt or not completion:
        raise ValueError(
            f"row produced empty prompt ({len(prompt)}) or completion ({len(completion)}); "
            f"roles were {[m['role'] for m in row['messages']]}"
        )
    return {"prompt": prompt, "completion": completion}


def load_chat_formatted_dataset(path: Path) -> list[dict[str, Any]]:
    """Load and validate a ChatML JSONL file, returning prompt-completion rows."""

    rows: list[dict[str, Any]] = []
    for line_no, row in iter_jsonl(path):
        validate_chatml_row(row, line_no)
        rows.append(to_prompt_completion(row))
    return rows


# ---------------------------------------------------------------------------
# Token audit (dry-run + a sanity step the training path also runs)
# ---------------------------------------------------------------------------


@dataclass
class SplitAudit:
    split: str
    examples: int
    tokens_min: int
    tokens_max: int
    tokens_mean: float
    tokens_p95: int
    over_limit: list[dict[str, Any]]
    tokenization_errors: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _percentile(sorted_values: Sequence[int], q: float) -> int:
    if not sorted_values:
        return 0
    pos = q * (len(sorted_values) - 1)
    lower = int(pos)
    upper = min(lower + 1, len(sorted_values) - 1)
    return sorted_values[lower] if lower == upper else sorted_values[upper]


def audit_token_lengths(
    tokenizer,
    dataset_rows: list[dict[str, Any]],
    split_name: str,
    max_seq_length: int,
    *,
    eos_token_id: int | None = None,
) -> SplitAudit:
    """Audit token counts the way TRL's SFTTrainer will assemble them.

    Mirrors ``trl.data_utils.apply_chat_template`` for the prompt-completion
    case: render the full prompt+completion in one call and audit the full
    text length. This is what gets encoded and concatenated internally by
    ``SFTTrainer._collate_prompt_completion``, so it is the authoritative
    shape for the ``max_seq_length=20480`` gate.
    """

    counts: list[int] = []
    over_limit: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for idx, row in enumerate(dataset_rows):
        try:
            full_text = tokenizer.apply_chat_template(
                row["prompt"] + row["completion"],
                tokenize=False,
                add_generation_prompt=False,
            )
            ids = tokenizer.encode(full_text, add_special_tokens=False)
            token_count = len(ids)
        except Exception as exc:  # pragma: no cover - defensive
            errors.append({"index": idx, "reason": f"tokenize failed: {exc}"})
            continue

        counts.append(token_count)
        if token_count > max_seq_length:
            over_limit.append(
                {
                    "index": idx,
                    "tokens": token_count,
                    "excess": token_count - max_seq_length,
                    "prompt_first_chars": (row["prompt"][-1]["content"][:120] if row["prompt"] else ""),
                }
            )

    sorted_counts = sorted(counts)
    return SplitAudit(
        split=split_name,
        examples=len(dataset_rows),
        tokens_min=min(counts) if counts else 0,
        tokens_max=max(counts) if counts else 0,
        tokens_mean=round(sum(counts) / len(counts), 2) if counts else 0.0,
        tokens_p95=_percentile(sorted_counts, 0.95),
        over_limit=over_limit,
        tokenization_errors=errors,
    )


def audit_all_splits(
    tokenizer,
    splits: dict[str, list[dict[str, Any]]],
    max_seq_length: int,
) -> dict[str, SplitAudit]:
    """Run :func:`audit_token_lengths` for every (split_name -> rows) pair.

    Pure helper: no module state, no I/O, easy to unit-test with a fake
    tokenizer. Used by both ``run_dry_run`` and ``run_training`` so the same
    audit shape is enforced before the trainer is constructed.
    """

    return {
        name: audit_token_lengths(tokenizer, rows, name, max_seq_length)
        for name, rows in splits.items()
    }


def assert_audits_within_budget(
    audits: dict[str, SplitAudit],
    max_seq_length: int,
    *,
    context: str = "training",
) -> None:
    """Hard-fail if any audited split has rows over budget or tokenization errors.

    Both over-limit rows and tokenization errors are fatal. TRL's
    ``SFTTrainer`` only forwards ``max_length`` and would silently truncate
    eval (or train) examples that overflow, so we refuse to start instead.
    """

    if max_seq_length <= 0:
        raise ValueError(f"max_seq_length must be > 0, got {max_seq_length}")

    for name, audit in audits.items():
        if audit.over_limit:
            preview = ", ".join(
                f"idx={row['index']} tokens={row['tokens']}"
                for row in audit.over_limit[:3]
            )
            raise RuntimeError(
                f"Refusing to {context}: {len(audit.over_limit)} {name} examples exceed "
                f"max_seq_length={max_seq_length} (e.g. {preview}). Inspect "
                "audit_token_lengths output and raise the budget or remove the "
                "offending rows. Silent truncation is disabled."
            )
        if audit.tokenization_errors:
            preview = ", ".join(
                f"idx={row['index']}" for row in audit.tokenization_errors[:3]
            )
            raise RuntimeError(
                f"Refusing to {context}: {len(audit.tokenization_errors)} {name} rows failed "
                f"to tokenize (e.g. {preview}). Inspect audit_token_lengths output "
                "and fix or remove the offending rows."
            )


# ---------------------------------------------------------------------------
# Environment validation
# ---------------------------------------------------------------------------


def validate_environment(cfg: dict[str, Any]) -> dict[str, Any]:
    """Hard-fail loud on environment issues. Always runs (even in dry-run)."""

    import torch

    info: dict[str, Any] = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
    }

    if cfg["bf16"]:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "bf16 training requested but torch.cuda.is_available() is False. "
                "Set --no-bf16 or run on a CUDA-capable host."
            )
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                f"bf16 requested but the active CUDA device ({info['device_name']}) "
                "does not report bfloat16 support."
            )

    if cfg["quantization"] in {"nf4_double_quant", "4bit"}:
        try:
            import bitsandbytes  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "4-bit (NF4) quantization requested but bitsandbytes is not installed. "
                "Install it inside .venv (`uv pip install bitsandbytes==0.49.2`) before training."
            ) from exc
        info["bitsandbytes_version"] = __import__("bitsandbytes").__version__

    return info


def build_bnb_config(cfg: dict[str, Any]):
    """Build a BitsAndBytesConfig matching the locked plan."""

    from transformers import BitsAndBytesConfig
    import torch

    if cfg["quantization"] == "nf4_double_quant":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    if cfg["quantization"] == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
        )
    if cfg["quantization"] == "none":
        return None
    raise ValueError(
        f"Unknown quantization {cfg['quantization']!r}; expected one of "
        "'nf4_double_quant', '4bit', 'none'."
    )


def build_lora_config(cfg: dict[str, Any]):
    from peft import LoraConfig, TaskType

    return LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        bias=cfg["lora_bias"],
        target_modules=list(cfg["lora_target_modules"]),
        task_type=TaskType.CAUSAL_LM,
    )


# ---------------------------------------------------------------------------
# dry-run path
# ---------------------------------------------------------------------------


def run_dry_run(cfg: dict[str, Any], report_path: Path) -> dict[str, Any]:
    from transformers import AutoTokenizer

    print(f"[dry-run] loading tokenizer: {cfg['model_name_or_path']}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_name_or_path"],
        trust_remote_code=cfg["trust_remote_code"],
        cache_dir=os.environ.get("HF_HOME"),
    )
    if tokenizer.chat_template is None:
        raise RuntimeError(
            f"Tokenizer {cfg['model_name_or_path']} has no chat_template; cannot audit."
        )

    train_path = Path(cfg["dataset_train_path"])
    eval_path = Path(cfg["dataset_eval_path"])
    if not train_path.exists():
        raise FileNotFoundError(f"Train JSONL not found: {train_path}")
    if not eval_path.exists():
        raise FileNotFoundError(f"Eval JSONL not found: {eval_path}")

    print(f"[dry-run] loading train split: {train_path}", file=sys.stderr)
    train_rows = load_chat_formatted_dataset(train_path)
    print(f"[dry-run] loading eval split:  {eval_path}", file=sys.stderr)
    eval_rows = load_chat_formatted_dataset(eval_path)

    max_seq_length = int(cfg["max_seq_length"])
    print(f"[dry-run] auditing tokens against max_seq_length={max_seq_length}", file=sys.stderr)
    audits = audit_all_splits(
        tokenizer,
        {"train": train_rows, "eval": eval_rows},
        max_seq_length,
    )
    train_audit = audits["train"]
    eval_audit = audits["eval"]

    overall_max = max(train_audit.tokens_max, eval_audit.tokens_max)
    total_over_limit = len(train_audit.over_limit) + len(eval_audit.over_limit)
    total_tokenization_errors = len(train_audit.tokenization_errors) + len(
        eval_audit.tokenization_errors
    )

    env_info = validate_environment(cfg)
    if cfg["bf16"]:
        # In dry-run we still want to assert bf16 support so silent CPU runs
        # don't slip through.
        assert env_info["cuda_available"], "dry-run bf16 environment check failed"

    readiness: dict[str, Any] = {
        "phase": "Phase 3 — SFT/QLoRA training preparation",
        "status": "pass",
        "checked_at": _utcnow_iso(),
        "plan_reference": "PLAN_ENTRENAMIENTO_QWEN.md §Fase 3",
        "model_name_or_path": cfg["model_name_or_path"],
        "tokenizer": {
            "class": type(tokenizer).__name__,
            "vocab_size": tokenizer.vocab_size,
            "model_max_length": tokenizer.model_max_length,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        },
        "data": {
            "train_path": str(train_path),
            "eval_path": str(eval_path),
            "train_examples": len(train_rows),
            "eval_examples": len(eval_rows),
        },
        "max_seq_length": max_seq_length,
        "token_audit": {
            "train": train_audit.to_dict(),
            "eval": eval_audit.to_dict(),
            "overall_max": overall_max,
            "overall_max_p95": max(train_audit.tokens_p95, eval_audit.tokens_p95),
            "over_limit_total": total_over_limit,
            "tokenization_errors_total": total_tokenization_errors,
            "truncation_policy": "fail-loud: any example > max_seq_length aborts training; no silent truncation.",
        },
        "training_config": {
            "lora": {
                "r": cfg["lora_r"],
                "alpha": cfg["lora_alpha"],
                "dropout": cfg["lora_dropout"],
                "bias": cfg["lora_bias"],
                "target_modules": cfg["lora_target_modules"],
                "task_type": "CAUSAL_LM",
            },
            "loop": {
                "num_train_epochs": cfg["num_train_epochs"],
                "per_device_train_batch_size": cfg["per_device_train_batch_size"],
                "per_device_eval_batch_size": cfg["per_device_eval_batch_size"],
                "gradient_accumulation_steps": cfg["gradient_accumulation_steps"],
                "effective_batch_size": cfg["per_device_train_batch_size"]
                * cfg["gradient_accumulation_steps"],
                "learning_rate": cfg["learning_rate"],
                "lr_scheduler_type": cfg["lr_scheduler_type"],
                "warmup_ratio": cfg["warmup_ratio"],
                "weight_decay": cfg["weight_decay"],
                "optim": cfg["optim"],
                "gradient_checkpointing": cfg["gradient_checkpointing"],
                "bf16": cfg["bf16"],
                "max_grad_norm": cfg["max_grad_norm"],
                "save_strategy": cfg["save_strategy"],
                "eval_strategy": cfg["eval_strategy"],
                "logging_steps": cfg["logging_steps"],
                "report_to": cfg["report_to"],
                "seed": cfg["seed"],
            },
            "completion_only_loss": cfg["completion_only_loss"],
            "packing": cfg["packing"],
        },
        "environment": env_info,
        "ready_for_training": (
            total_over_limit == 0
            and total_tokenization_errors == 0
            and overall_max <= max_seq_length
            and env_info["cuda_available"]
        ),
        "warnings": [],
        "notes": [
            f"Canonical Phase 1 rows are gold (weight 1.0): train={len(train_rows)}, eval={len(eval_rows)}, total={len(train_rows) + len(eval_rows)}. Origin: GPT-5.4-mini + validación humana Nico.",
            "User messages in data/chat_formatted/*.jsonl already embed fecha+título+texto_original; this script does not duplicate them.",
            "Effective batch size = per_device_train_batch_size * gradient_accumulation_steps * world_size (world_size=1 in MVP).",
            "Dry-run does NOT load the base model weights (saves ~14 GB of VRAM). Real training requires loading the base model with QLoRA.",
        ],
    }

    if total_over_limit > 0:
        readiness["status"] = "blocked"
        readiness["warnings"].append(
            f"{total_over_limit} examples exceed max_seq_length={max_seq_length}; raise "
            "the budget (up to model's 32k native cap) or revisit long examples before training."
        )

    if total_tokenization_errors > 0:
        readiness["status"] = "blocked"
        readiness["warnings"].append(f"{total_tokenization_errors} tokenization errors; inspect rows.")

    if not env_info["cuda_available"]:
        readiness["status"] = "blocked"
        readiness["warnings"].append("CUDA not available; actual training impossible until GPU is reachable.")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(readiness, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        f"[dry-run] train={len(train_rows)} eval={len(eval_rows)} "
        f"overall_max={overall_max} over_limit={total_over_limit} "
        f"ready={readiness['ready_for_training']} -> {report_path}",
        file=sys.stderr,
    )
    return readiness


def _utcnow_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# training path
# ---------------------------------------------------------------------------


def should_load_eval_dataset(cfg: dict[str, Any]) -> bool:
    """Return True iff :func:`run_training` should load ``dataset_eval_path``.

    Phase 3 safety fix: when ``eval_strategy='no'`` we skip loading the eval
    JSONL entirely. The previous 1-step smoke OOMed because the trainer
    auto-ran an in-training eval pass at the ``max_steps=1`` boundary; for
    smokes (and any other run that opts out of eval) we want to skip both the
    eval pass AND the eval dataset load so the trainer is constructed without
    ``eval_dataset``.
    """

    return cfg.get("eval_strategy", "no") != "no"


def build_sft_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    """Translate a resolved :class:`TrainingConfig` dict into ``SFTConfig`` kwargs.

    Pure helper extracted from :func:`run_training` so we can unit-test the
    safety-critical wiring (``no_save`` → ``save_strategy='no'``,
    ``eval_strategy='no'`` is honored, ``per_device_eval_batch_size`` defaults
    to 1) without instantiating a Trainer or the base model.

    The returned dict is consumed by ``SFTConfig(**sft_kwargs)``.
    """

    sft_kwargs: dict[str, Any] = dict(
        output_dir=cfg["output_dir"],
        overwrite_output_dir=True,
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        warmup_ratio=cfg["warmup_ratio"],
        weight_decay=cfg["weight_decay"],
        optim=cfg["optim"],
        gradient_checkpointing=cfg["gradient_checkpointing"],
        bf16=cfg["bf16"],
        max_grad_norm=cfg["max_grad_norm"],
        save_strategy=cfg["save_strategy"],
        save_total_limit=cfg["save_total_limit"],
        save_steps=cfg["save_steps"],
        eval_strategy=cfg["eval_strategy"],
        eval_steps=cfg["eval_steps"],
        logging_steps=cfg["logging_steps"],
        report_to=cfg["report_to"],
        seed=cfg["seed"],
        max_steps=cfg["max_steps"],
        max_length=cfg["max_seq_length"],
        packing=cfg["packing"],
        completion_only_loss=cfg["completion_only_loss"],
        dataset_kwargs={"skip_prepare_dataset": False},
        dataloader_num_workers=cfg["dataloader_num_workers"],
        remove_unused_columns=cfg["remove_unused_columns"],
    )

    if cfg.get("gradient_checkpointing_kwargs"):
        sft_kwargs["gradient_checkpointing_kwargs"] = cfg["gradient_checkpointing_kwargs"]

    if cfg.get("load_best_model_at_end"):
        sft_kwargs["load_best_model_at_end"] = True
        # metric_for_best_model / greater_is_better can be added when a custom
        # eval metric is wired in; keep defaults for now.

    # Phase 3 safety fix: --no-save must disable Trainer-level saves too.
    # The previous --no-save only skipped the explicit trainer.save_model()
    # at the end; with save_strategy='epoch' the trainer would still drop a
    # checkpoint at the epoch boundary, which fires after max_steps=1 because
    # HF treats max_steps as an epoch boundary. Force save_strategy='no' so
    # nothing is written to output_dir.
    if cfg.get("no_save"):
        sft_kwargs["save_strategy"] = "no"

    return sft_kwargs


def run_training(cfg: dict[str, Any]) -> None:
    """Instantiate SFTTrainer and start training.

    This path is intentionally simple — most of the heavy lifting happens in
    SFTTrainer's own dataset preparation. ``run_dry_run`` already validated
    the data conversion, so by the time we reach here the rows are known good.
    """

    import torch
    from datasets import Dataset
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    env_info = validate_environment(cfg)
    if not env_info["cuda_available"]:
        raise RuntimeError("CUDA not available; refusing to start training.")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_name_or_path"],
        trust_remote_code=cfg["trust_remote_code"],
        cache_dir=os.environ.get("HF_HOME"),
    )
    if tokenizer.pad_token is None:
        # Qwen2.5 ships with <|endoftext|> as pad; if missing, fall back to eos.
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[train] loading base model in {cfg['dtype']} with quantization={cfg['quantization']}", file=sys.stderr)
    model_kwargs: dict[str, Any] = {
        "device_map": cfg["device_map"],
        "trust_remote_code": cfg["trust_remote_code"],
    }
    if cfg["dtype"] in {"bfloat16", "bf16"}:
        model_kwargs["torch_dtype"] = torch.bfloat16
    elif cfg["dtype"] in {"float16", "fp16"}:
        model_kwargs["torch_dtype"] = torch.float16

    bnb_config = build_bnb_config(cfg)
    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config

    model = AutoModelForCausalLM.from_pretrained(cfg["model_name_or_path"], **model_kwargs)
    lora_config = build_lora_config(cfg)
    model = get_peft_model(model, lora_config)

    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    # Load and convert the dataset. We rely on SFTTrainer's prompt-completion
    # branch by passing {prompt, completion} per row.
    print(f"[train] loading train dataset: {cfg['dataset_train_path']}", file=sys.stderr)
    train_rows = load_chat_formatted_dataset(Path(cfg["dataset_train_path"]))

    # Phase 3 safety fix: only load the eval JSONL if eval_strategy is enabled.
    # Skipping the load keeps the smoke fast and avoids spurious "eval JSONL
    # not found" failures when --dataset-eval-path is omitted on purpose.
    eval_rows: list[dict[str, Any]] | None = None
    if should_load_eval_dataset(cfg):
        eval_path = Path(cfg["dataset_eval_path"])
        if eval_path.exists():
            eval_rows = load_chat_formatted_dataset(eval_path)
        else:
            print(
                f"[train] eval_strategy={cfg['eval_strategy']!r} but eval JSONL "
                f"not found at {eval_path}; skipping eval dataset load.",
                file=sys.stderr,
            )

    train_dataset = Dataset.from_list(train_rows)
    eval_dataset = Dataset.from_list(eval_rows) if eval_rows is not None else None

    # Final token-length gate before training: if any row (train OR eval) exceeds
    # the budget or fails to tokenize, refuse to train rather than silently
    # truncate. TRL's ``SFTTrainer`` only forwards ``max_length`` and would drop
    # overflow tokens without warning, so we enforce the budget up-front.
    splits_to_audit: dict[str, list[dict[str, Any]]] = {"train": train_rows}
    if eval_rows is not None:
        splits_to_audit["eval"] = eval_rows
    audits = audit_all_splits(tokenizer, splits_to_audit, int(cfg["max_seq_length"]))
    assert_audits_within_budget(audits, int(cfg["max_seq_length"]), context="train")

    sft_kwargs = build_sft_kwargs(cfg)
    sft_config = SFTConfig(**sft_kwargs)

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": sft_config,
        "processing_class": tokenizer,
        "train_dataset": train_dataset,
    }
    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset

    trainer = SFTTrainer(**trainer_kwargs)

    print("[train] starting trainer.train()", file=sys.stderr)
    train_result = trainer.train(resume_from_checkpoint=cfg["resume_from_checkpoint"])

    if cfg["no_save"]:
        # Belt-and-braces: build_sft_kwargs already forced save_strategy='no'
        # so the trainer will not drop a checkpoint mid-training, and here we
        # also skip the explicit trainer.save_model/save_state/tokenizer saves.
        print(
            "[train] --no-save set; output_dir left empty (no checkpoint, "
            "no trainer_state, no tokenizer, no metrics).",
            file=sys.stderr,
        )
    else:
        trainer.save_model(cfg["output_dir"])
        trainer.save_state()
        # Persist metrics + tokenizer alongside the adapter.
        metrics_path = Path(cfg["output_dir"]) / "training_metrics.json"
        metrics_path.write_text(
            json.dumps(train_result.metrics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        tokenizer.save_pretrained(cfg["output_dir"])

    print("[train] done", file=sys.stderr)


# ---------------------------------------------------------------------------
# eval-only path
# ---------------------------------------------------------------------------


def _build_model_and_tokenizer(cfg: dict[str, Any]) -> tuple[Any, Any]:
    """Load tokenizer + base model + PEFT wrapper exactly like :func:`run_training`.

    Pure helper extracted so :func:`run_eval_only` and :func:`run_training`
    share one model-construction path (no drift between the two modes). The
    caller is responsible for moving the model to a device and freeing it.
    """

    import torch
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_name_or_path"],
        trust_remote_code=cfg["trust_remote_code"],
        cache_dir=os.environ.get("HF_HOME"),
    )
    if tokenizer.pad_token is None:
        # Qwen2.5 ships with  as pad; if missing, fall back to eos.
        tokenizer.pad_token = tokenizer.eos_token

    print(
        f"[eval-only] loading base model in {cfg['dtype']} with "
        f"quantization={cfg['quantization']}",
        file=sys.stderr,
    )
    model_kwargs: dict[str, Any] = {
        "device_map": cfg["device_map"],
        "trust_remote_code": cfg["trust_remote_code"],
    }
    if cfg["dtype"] in {"bfloat16", "bf16"}:
        model_kwargs["torch_dtype"] = torch.bfloat16
    elif cfg["dtype"] in {"float16", "fp16"}:
        model_kwargs["torch_dtype"] = torch.float16

    bnb_config = build_bnb_config(cfg)
    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config

    model = AutoModelForCausalLM.from_pretrained(cfg["model_name_or_path"], **model_kwargs)
    lora_config = build_lora_config(cfg)
    model = get_peft_model(model, lora_config)
    return model, tokenizer


def run_eval_only(
    cfg: dict[str, Any],
    report_path: Path,
    *,
    command: str = "",
    pytorch_cuda_alloc_conf: str = "",
) -> dict[str, Any]:
    """Eval-only smoke: build the trainer exactly like training, then evaluate.

    This path is the canonical "does the eval set fit in memory at
    per_device_eval_batch_size=1?" gate. It deliberately does **not** call
    :meth:`Trainer.train`. The train dataset is loaded only because TRL's
    ``SFTTrainer.__init__`` raises ``ValueError("`train_dataset` is required")``
    otherwise; the dataset is never used to compute gradients or step the
    optimizer.

    Order of operations (fail-loud on every gate BEFORE committing GPU memory):

    1. validate_environment() — CUDA / bf16 / bitsandbytes must be present.
    2. Load the tokenizer (small; no model weights yet) and audit BOTH train
       and eval splits against ``max_seq_length``. Any over-limit row or
       tokenization error aborts the run with a blocked report.
    3. _build_model_and_tokenizer() — only reached after the audit passes.
    4. Build the SFTTrainer with the eval dataset (and the train dataset,
       required by SFTTrainer but never used for training).
    5. Call ``trainer.evaluate()`` exactly once. Never call ``trainer.train()``.
    6. Write the report and return.

    Safety wiring (all enforced, with belt-and-braces where it matters):

    * ``per_device_eval_batch_size=1`` is forced into ``SFTConfig`` (defaults
      can create larger eval batches and likely OOM on this workload).
    * ``save_strategy='no'`` is forced via ``build_sft_kwargs(no_save=True)``.
    * ``eval_strategy='no'`` is forced so the trainer does not auto-evaluate
      at any epoch/step boundary; we call ``trainer.evaluate()`` exactly
      once from here.
    * ``trainer.train()`` is never called.
    * ``trainer.save_model()``, ``trainer.save_state()``, and
      ``tokenizer.save_pretrained()`` are never called.
    * The output directory is never written to. ``reports/phase3_eval_smoke.json``
      is the only artifact persisted.
    """

    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    eval_cfg = dict(cfg)
    # Belt-and-braces: enforce the safety wiring inside the eval-only path.
    # These override any value that came in via the JSON config or CLI flag,
    # so a misconfigured eval run cannot fall back to the unsafe defaults.
    eval_cfg["eval_strategy"] = "no"
    eval_cfg["no_save"] = True
    eval_cfg["per_device_eval_batch_size"] = 1

    report: dict[str, Any] = {
        "phase": "Phase 3 — eval-only smoke (per_device_eval_batch_size=1)",
        "status": "pass",
        "checked_at": _utcnow_iso(),
        "plan_reference": "PLAN_ENTRENAMIENTO_QWEN.md §Fase 3 (eval smoke)",
        "command": command,
        "command_env_extras": {
            "PYTORCH_CUDA_ALLOC_CONF": pytorch_cuda_alloc_conf,
        },
        "eval_only": True,
        "errors": [],
        "smoke_constraints": {
            "eval_only": True,
            "no_save": True,
            "save_strategy_resolved_by_no_save": "no",
            "eval_strategy": "no",
            "per_device_eval_batch_size": 1,
            "per_device_train_batch_size": eval_cfg.get("per_device_train_batch_size", 1),
            "report_to": eval_cfg.get("report_to", "none"),
            "output_dir": eval_cfg["output_dir"],
            "max_seq_length_kept": int(eval_cfg["max_seq_length"]),
            "data_truncation": False,
            "eval_pass_attempted": True,
            "train_called": False,
        },
    }

    def _finalize_and_return() -> dict[str, Any]:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return report

    # --- gate 1: environment ---------------------------------------------
    env_info = validate_environment(eval_cfg)
    report["environment"] = env_info
    if not env_info["cuda_available"]:
        report["status"] = "blocked"
        report["errors"].append("CUDA not available; refusing to start eval smoke.")
        return _finalize_and_return()

    # --- gate 2: tokenizer + audit (BEFORE 14 GB model load) --------------
    tokenizer = AutoTokenizer.from_pretrained(
        eval_cfg["model_name_or_path"],
        trust_remote_code=eval_cfg["trust_remote_code"],
        cache_dir=os.environ.get("HF_HOME"),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_path = Path(eval_cfg["dataset_train_path"])
    eval_path = Path(eval_cfg["dataset_eval_path"])
    if not train_path.exists():
        report["status"] = "blocked"
        report["errors"].append(f"Train JSONL not found: {train_path}")
        return _finalize_and_return()
    if not eval_path.exists():
        report["status"] = "blocked"
        report["errors"].append(f"Eval JSONL not found: {eval_path}")
        return _finalize_and_return()

    print(f"[eval-only] loading train split: {train_path}", file=sys.stderr)
    train_rows = load_chat_formatted_dataset(train_path)
    print(f"[eval-only] loading eval split:  {eval_path}", file=sys.stderr)
    eval_rows = load_chat_formatted_dataset(eval_path)

    # Audit BOTH splits. Fail loud on over-limit or tokenization errors BEFORE
    # the 14 GB base model is loaded, so a malformed eval set never reaches
    # the trainer (and never allocates a 32 GB CUDA tensor that would OOM
    # before the script can produce a useful report).
    audits = audit_all_splits(
        tokenizer,
        {"train": train_rows, "eval": eval_rows},
        int(eval_cfg["max_seq_length"]),
    )
    report["token_audit"] = {name: audit.to_dict() for name, audit in audits.items()}
    report["data"] = {
        "train_path": str(train_path),
        "eval_path": str(eval_path),
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
    }
    try:
        assert_audits_within_budget(
            audits, int(eval_cfg["max_seq_length"]), context="eval-only"
        )
    except RuntimeError as exc:
        report["status"] = "blocked"
        report["errors"].append(str(exc))
        return _finalize_and_return()

    # --- gate 3: build model + trainer -----------------------------------
    model, tokenizer = _build_model_and_tokenizer(eval_cfg)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    train_dataset = Dataset.from_list(train_rows)
    eval_dataset = Dataset.from_list(eval_rows)

    sft_kwargs = build_sft_kwargs(eval_cfg)
    # Belt-and-braces: reassert that the safety wiring survived build_sft_kwargs.
    assert sft_kwargs["save_strategy"] == "no", (
        f"eval-only must produce save_strategy='no'; got {sft_kwargs['save_strategy']!r}"
    )
    assert sft_kwargs["per_device_eval_batch_size"] == 1, (
        f"eval-only must produce per_device_eval_batch_size=1; got "
        f"{sft_kwargs['per_device_eval_batch_size']!r}"
    )
    assert sft_kwargs["eval_strategy"] == "no", (
        f"eval-only must produce eval_strategy='no'; got {sft_kwargs['eval_strategy']!r}"
    )
    sft_config = SFTConfig(**sft_kwargs)

    print(
        f"[eval-only] constructing SFTTrainer with eval_dataset "
        f"({len(eval_rows)} rows); train_dataset loaded but never trained on",
        file=sys.stderr,
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        processing_class=tokenizer,
        train_dataset=train_dataset,  # required by SFTTrainer; no train() called
        eval_dataset=eval_dataset,
    )

    print("[eval-only] starting trainer.evaluate() — trainer.train() is NOT called", file=sys.stderr)
    metrics = trainer.evaluate()

    # Safety belt: explicitly do NOT call trainer.train(), trainer.save_model(),
    # trainer.save_state(), or tokenizer.save_pretrained(). The output_dir is
    # left empty (Trainer may create the directory but writes nothing inside).

    report.update(
        {
            "training_config": {
                "lora": {
                    "r": eval_cfg["lora_r"],
                    "alpha": eval_cfg["lora_alpha"],
                    "dropout": eval_cfg["lora_dropout"],
                    "bias": eval_cfg["lora_bias"],
                    "target_modules": eval_cfg["lora_target_modules"],
                    "task_type": "CAUSAL_LM",
                },
                "loop": {
                    "per_device_train_batch_size": eval_cfg["per_device_train_batch_size"],
                    "per_device_eval_batch_size": eval_cfg["per_device_eval_batch_size"],
                    "eval_strategy": sft_kwargs["eval_strategy"],
                    "save_strategy": sft_kwargs["save_strategy"],
                    "report_to": eval_cfg["report_to"],
                },
                "completion_only_loss": eval_cfg["completion_only_loss"],
                "packing": eval_cfg["packing"],
            },
            "safety_wiring_proof": {
                "per_device_eval_batch_size": sft_kwargs["per_device_eval_batch_size"],
                "save_strategy": sft_kwargs["save_strategy"],
                "eval_strategy": sft_kwargs["eval_strategy"],
                "trainer_train_called": False,
                "trainer_save_model_called": False,
                "trainer_save_state_called": False,
                "tokenizer_save_pretrained_called": False,
            },
            "metrics": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in metrics.items()},
        }
    )
    return _finalize_and_return()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    cfg: dict[str, Any] = {}
    if args.config is not None and Path(args.config).exists():
        cfg.update(config_from_file(args.config))
    cfg = merge_args_into_config(args, cfg)

    if args.dry_run:
        readiness = run_dry_run(cfg, args.readiness_report)
        # Exit 0 if everything is ready, 2 if readiness is blocked.
        return 0 if readiness["ready_for_training"] else 2

    if args.eval_only:
        # Use the actual CLI argv for the captured command string. When the
        # script is run as `python training/train_sft.py ...`, sys.argv[1:]
        # already contains the trailing args; argv_to_cli_repr adds minimal
        # quoting for tokens containing whitespace.
        effective_argv = list(sys.argv[1:] if argv is None else argv)
        command = "training/train_sft.py " + " ".join(argv_to_cli_repr(effective_argv))
        report = run_eval_only(
            cfg,
            args.eval_report,
            command=command.strip(),
            pytorch_cuda_alloc_conf=os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
        )
        # Exit 0 if smoke passed, 2 if blocked.
        return 0 if report["status"] == "pass" else 2

    # Real training path
    run_training(cfg)
    return 0


def argv_to_cli_repr(argv: list[str]) -> list[str]:
    """Best-effort re-emit of CLI args for the eval-only report.

    Used only to make the captured command string in the report match what
    was actually executed. Paths are stringified verbatim; numeric args are
    kept as-is. We do NOT attempt full shell-quoting — the report is
    consumed by humans, not shells.
    """
    out: list[str] = []
    for tok in argv:
        if any(ch in tok for ch in " \t\"'"):
            out.append(f'"{tok}"')
        else:
            out.append(tok)
    return out


if __name__ == "__main__":
    raise SystemExit(main())


if __name__ == "__main__":
    raise SystemExit(main())
