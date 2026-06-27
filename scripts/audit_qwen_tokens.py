#!/usr/bin/env python3
"""Real Qwen tokenizer audit over the ChatML-formatted SFT data.

This is the Phase 1 gate check that supersedes the char/4 proxy used by
``data/chat_formatter.py``: it loads the actual ``Qwen/Qwen2.5-7B-Instruct``
tokenizer from Hugging Face, applies the model's ChatML template to every
example, and reports real token counts against ``max_seq_length=20480``.

Gate contract:

* ``max_real_tokens <= 20480``
* ``over_limit == []`` (every example fits)

If the gate fails, raise the budget up to the model's native 32768 only after
reviewing the report; never truncate silently.

Usage:

    python scripts/audit_qwen_tokens.py
    python scripts/audit_qwen_tokens.py --max-seq-length 20480 --model Qwen/Qwen2.5-7B-Instruct

Dependencies: ``transformers>=4.37.0`` and ``huggingface-hub<1.0`` must be
installed in the active Python environment (the project's ``.venv`` already
has them after running the setup described in ``README_CONTINUAR_ENTRENAMIENTO.md``).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_TRAIN = Path("data/chat_formatted/train.jsonl")
DEFAULT_EVAL = Path("data/chat_formatted/eval.jsonl")
DEFAULT_REPORT = Path("reports/qwen_tokenizer_audit.json")
DEFAULT_MAX_SEQ_LENGTH = 20480


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


def percentile(sorted_values: list[int], q: float) -> int:
    if not sorted_values:
        return 0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lower = int(pos)
    upper = min(lower + 1, len(sorted_values) - 1)
    frac = pos - lower
    return int(round(sorted_values[lower] * (1 - frac) + sorted_values[upper] * frac))


def audit_split(
    tokenizer,
    input_path: Path,
    split_name: str,
    max_seq_length: int,
) -> dict[str, Any]:
    counts: list[int] = []
    over_limit: list[dict[str, Any]] = []
    malformed: list[dict[str, Any]] = []
    nota_ids: list[str] = []

    for line_no, row in iter_jsonl(input_path):
        messages = row.get("messages")
        if not isinstance(messages, list):
            malformed.append({"line": line_no, "reason": "messages missing or not a list"})
            continue
        for msg in messages:
            if msg.get("role") not in {"system", "user", "assistant"}:
                malformed.append({"line": line_no, "reason": f"unexpected role {msg.get('role')!r}"})
            if not isinstance(msg.get("content"), str):
                malformed.append({"line": line_no, "reason": f"non-string content in role {msg.get('role')!r}"})

        try:
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            ids = tokenizer.encode(rendered, add_special_tokens=False)
        except Exception as exc:  # pragma: no cover - defensive
            malformed.append({"line": line_no, "reason": f"tokenize failed: {exc}"})
            continue

        token_count = len(ids)
        counts.append(token_count)

        try:
            nota_id = messages[1]["content"].splitlines()[0] if len(messages) >= 2 else f"line-{line_no}"
        except Exception:
            nota_id = f"line-{line_no}"
        if nota_id.startswith("FECHA DE EDICIÓN DE LA NOTA:"):
            nota_id = nota_id.split(":", 1)[1].strip() or f"line-{line_no}"
        nota_ids.append(nota_id)

        if token_count > max_seq_length:
            over_limit.append(
                {
                    "line": line_no,
                    "nota_id": nota_id,
                    "tokens": token_count,
                }
            )

    sorted_counts = sorted(counts)
    return {
        "split": split_name,
        "input": str(input_path),
        "examples": len(counts),
        "tokens": {
            "min": min(counts) if counts else 0,
            "max": max(counts) if counts else 0,
            "mean": round(statistics.fmean(counts), 2) if counts else 0,
            "median": int(statistics.median(counts)) if counts else 0,
            "stdev": round(statistics.pstdev(counts), 2) if len(counts) > 1 else 0,
            "p50": percentile(sorted_counts, 0.50),
            "p90": percentile(sorted_counts, 0.90),
            "p95": percentile(sorted_counts, 0.95),
            "p99": percentile(sorted_counts, 0.99),
        },
        "max_seq_length": max_seq_length,
        "over_limit_count": len(over_limit),
        "over_limit": over_limit,
        "malformed": malformed,
        "worst_nota_ids": [
            nota_ids[idx]
            for idx in sorted(range(len(counts)), key=lambda i: counts[i], reverse=True)[:5]
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--train-input", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--eval-input", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional local path to cache the downloaded tokenizer files.",
    )
    args = parser.parse_args()

    if not args.train_input.exists():
        raise SystemExit(f"Train JSONL not found: {args.train_input}")
    if not args.eval_input.exists():
        raise SystemExit(f"Eval JSONL not found: {args.eval_input}")

    print(f"[audit] Loading tokenizer: {args.model}", file=sys.stderr)
    load_kwargs: dict[str, Any] = {"trust_remote_code": False}
    if args.cache_dir:
        load_kwargs["cache_dir"] = args.cache_dir
    tokenizer = AutoTokenizer.from_pretrained(args.model, **load_kwargs)

    if tokenizer.chat_template is None:
        raise SystemExit(
            f"Tokenizer {args.model} has no chat_template; cannot measure real ChatML token counts."
        )

    print(f"[audit] Auditing train split: {args.train_input}", file=sys.stderr)
    train_report = audit_split(tokenizer, args.train_input, "train", args.max_seq_length)
    print(f"[audit] Auditing eval split: {args.eval_input}", file=sys.stderr)
    eval_report = audit_split(tokenizer, args.eval_input, "eval", args.max_seq_length)

    overall_max = max(train_report["tokens"]["max"], eval_report["tokens"]["max"])
    total_over_limit = train_report["over_limit_count"] + eval_report["over_limit_count"]
    total_malformed = len(train_report["malformed"]) + len(eval_report["malformed"])
    passed = overall_max <= args.max_seq_length and total_over_limit == 0 and total_malformed == 0

    gate = {
        "max_real_tokens": overall_max,
        "max_seq_length": args.max_seq_length,
        "over_limit_count": total_over_limit,
        "over_limit_examples": train_report["over_limit"] + eval_report["over_limit"],
        "malformed_count": total_malformed,
        "pass": passed,
    }

    report = {
        "model": args.model,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "tokenizer_model_max_length": tokenizer.model_max_length,
        "chat_template_used": True,
        "chat_template_source": "tokenizer.chat_template (Hugging Face Qwen2.5-7B-Instruct)",
        "max_seq_length": args.max_seq_length,
        "splits": {"train": train_report, "eval": eval_report},
        "gate": gate,
        "notes": [
            "Token counts come from tokenizer.encode(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False), add_special_tokens=False).",
            "ChatML template is the one shipped with Qwen/Qwen2.5-7B-Instruct; matches the assistant turns used in data/chat_formatted/*.jsonl.",
            "If max_real_tokens > max_seq_length, raise max_seq_length up to the model's 32768 native cap (never truncate silently).",
        ],
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        f"[audit] Done. overall_max={overall_max} max_seq_length={args.max_seq_length} "
        f"over_limit={total_over_limit} malformed={total_malformed} pass={passed} -> {args.report}"
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())