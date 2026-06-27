#!/usr/bin/env python3
"""Phase 2 smoke test: one-example baseline inference with Qwen2.5-7B-Instruct via vLLM.

Runs a SINGLE eval example through vLLM offline inference with the MVS schema
as a structured-output constraint, then validates the generated JSON against
``jsonschema.Draft202012Validator``. It is intentionally minimal so the Phase 2
gate can be cleared (or refuted) before running the full 35-example eval set.

What it does NOT do:
  * No metrics aggregation over all 35 eval examples (use the full Phase 2
    runner after this smoke passes).
  * No truncation: the prompt token count is read from vLLM's prompt_token_ids
    and ``max_tokens`` is computed as the remaining budget under the
    Phase 1 ``max_seq_length=20480`` cap, with a small safety margin.

Design choices for this smoke:
  * vLLM 0.23.x removed ``SamplingParams.guided_json``; the new API is
    ``StructuredOutputsParams(json=<schema>)`` passed via
    ``SamplingParams(structured_outputs=StructuredOutputsParams(...))``.
  * ``dtype="auto"`` lets vLLM pick bfloat16 for Qwen2.5.
  * ``enforce_eager=True`` is used to bypass CUDA graph capture on first run
    (sm_120 / Blackwell consumer). Disable for production if memory pressure
    is fine.
  * ``max_num_seqs=1`` keeps the KV cache sized for a single example so the
    smoke is OOM-safe on a 32 GB RTX 5090 with 20480 context.
  * Schema is cleaned before being handed to vLLM: ``$schema``/``$id``/``title``
    are stripped and ``const`` is rewritten to ``enum`` (vLLM's JSON
    structured-output schema parser does not accept those keywords).

Usage:
    .venv/bin/python scripts/baseline_qwen_smoke.py
    .venv/bin/python scripts/baseline_qwen_smoke.py --index 7
    .venv/bin/python scripts/baseline_qwen_smoke.py --model Qwen/Qwen2.5-7B-Instruct

The script writes:
    reports/phase2_smoke.json         (status + key counts + errors)
    metrics/baseline_smoke_output.json (raw assistant output + parsed object)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# We import vLLM lazily so the failure mode (missing install) is captured into
# the report rather than crashing before we can write the blocker file.
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_EVAL = Path("data/chat_formatted/eval.jsonl")
DEFAULT_SCHEMA = Path("esquema_eventos_protesta_entrenamiento_MVS.json")
DEFAULT_REPORT = Path("reports/phase2_smoke.json")
DEFAULT_OUTPUT = Path("metrics/baseline_smoke_output.json")
DEFAULT_MAX_SEQ_LENGTH = 20480
DEFAULT_MAX_TOKENS_CAP = 8192
DEFAULT_PROMPT_SAFETY_MARGIN = 16


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--eval-input", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--index", type=int, default=0, help="Eval example index (0-based).")
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
    args = parser.parse_args()

    report: dict[str, Any] = {
        "phase": "Phase 2 — Baseline Qwen2.5-7B-Instruct (smoke)",
        "status": "blocked",
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": args.model,
        "index_requested": args.index,
        "max_seq_length": args.max_seq_length,
        "errors": [],
    }

    if not args.eval_input.exists():
        report["status"] = "blocked"
        report["errors"].append(f"eval input not found: {args.eval_input}")
        write_json(args.report, report)
        print(f"[smoke] BLOCKED: {report['errors']}", file=sys.stderr)
        return 2
    if not args.schema.exists():
        report["status"] = "blocked"
        report["errors"].append(f"schema not found: {args.schema}")
        write_json(args.report, report)
        print(f"[smoke] BLOCKED: {report['errors']}", file=sys.stderr)
        return 2

    rows = load_jsonl(args.eval_input)
    if args.index < 0 or args.index >= len(rows):
        report["errors"].append(
            f"--index {args.index} out of range (0..{len(rows)-1})"
        )
        write_json(args.report, report)
        return 2
    eval_row = rows[args.index]["row"]
    nota_id = derive_nota_id(eval_row, fallback=f"line_{rows[args.index]['line_no']}")

    # Pull system + user only (assistant is the gold reference, not part of prompt).
    chat_messages = [
        {"role": m["role"], "content": m["content"]}
        for m in eval_row["messages"]
        if m.get("role") in {"system", "user"}
    ]

    with args.schema.open("r", encoding="utf-8") as f:
        raw_schema = json.load(f)
    cleaned_schema = clean_schema_for_vllm(raw_schema)
    report["schema"] = {
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
        report["status"] = "blocked"
        report["errors"].append(f"vLLM/torch import failed: {exc}")
        write_json(args.report, report)
        print(f"[smoke] BLOCKED: {report['errors']}", file=sys.stderr)
        return 2

    report["vllm_version"] = vllm.__version__
    report["torch_version"] = torch.__version__
    report["cuda_runtime"] = torch.version.cuda
    report["cuda_available"] = bool(torch.cuda.is_available())
    if torch.cuda.is_available():
        report["gpu"] = {
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "vram_total_mib": int(torch.cuda.get_device_properties(0).total_memory) // (1024 * 1024),
        }
    report["structured_outputs_api"] = (
        "SamplingParams(structured_outputs=StructuredOutputsParams(json=cleaned_schema))"
    )

    if not torch.cuda.is_available():
        report["status"] = "blocked"
        report["errors"].append("torch.cuda.is_available() is False")
        write_json(args.report, report)
        return 2

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
    report["llm_kwargs"] = llm_kwargs

    try:
        print(f"[smoke] Loading model: {args.model}", file=sys.stderr)
        llm = LLM(**llm_kwargs)
    except Exception as exc:
        report["status"] = "blocked"
        report["errors"].append(f"LLM load failed: {exc}")
        report["errors"].append(traceback.format_exc())
        write_json(args.report, report)
        print(f"[smoke] BLOCKED during LLM load: {exc}", file=sys.stderr)
        return 2

    # Tokenize the prompt the same way vLLM will, to compute the budget.
    tokenizer = llm.get_tokenizer()
    try:
        prompt_text = tokenizer.apply_chat_template(
            chat_messages, tokenize=False, add_generation_prompt=True
        )
    except Exception as exc:
        report["status"] = "blocked"
        report["errors"].append(f"tokenizer.apply_chat_template failed: {exc}")
        write_json(args.report, report)
        return 2

    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    prompt_tokens = len(prompt_ids)
    report["prompt"] = {
        "nota_id": nota_id,
        "messages_count": len(chat_messages),
        "tokens": prompt_tokens,
        "chars": len(prompt_text),
        "user_chars": len(chat_messages[1]["content"]) if len(chat_messages) >= 2 else 0,
        "system_chars": len(chat_messages[0]["content"]) if chat_messages else 0,
        "max_seq_length": args.max_seq_length,
        "remaining_budget_tokens": max(0, args.max_seq_length - prompt_tokens),
    }

    if prompt_tokens >= args.max_seq_length:
        report["status"] = "blocked"
        report["errors"].append(
            f"prompt tokens ({prompt_tokens}) >= max_seq_length ({args.max_seq_length}); "
            "this example would not fit even with max_tokens=0. The Phase 1 gate "
            "(reports/qwen_tokenizer_audit.json) should have flagged this already."
        )
        write_json(args.report, report)
        return 2

    max_tokens = min(
        args.max_tokens_cap,
        max(0, args.max_seq_length - prompt_tokens - DEFAULT_PROMPT_SAFETY_MARGIN),
    )
    if max_tokens < 256:
        report["status"] = "blocked"
        report["errors"].append(
            f"computed max_tokens={max_tokens} too small for a useful output "
            f"(prompt_tokens={prompt_tokens}, max_seq_length={args.max_seq_length})."
        )
        write_json(args.report, report)
        return 2

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_tokens,
        structured_outputs=StructuredOutputsParams(json=cleaned_schema),
    )
    report["sampling_params"] = {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "structured_outputs": {"type": "json", "json_bytes": len(json.dumps(cleaned_schema))},
    }

    # ---- Run inference ----
    started = time.time()
    try:
        outputs = llm.chat(
            [chat_messages],
            sampling_params=sampling_params,
            use_tqdm=False,
            add_generation_prompt=True,
        )
    except Exception as exc:
        report["status"] = "blocked"
        report["errors"].append(f"vllm.chat failed: {exc}")
        report["errors"].append(traceback.format_exc())
        write_json(args.report, report)
        return 2
    elapsed = time.time() - started

    if not outputs:
        report["status"] = "fail"
        report["errors"].append("vllm.chat returned no outputs")
        write_json(args.report, report)
        return 2
    out = outputs[0]
    raw_text = out.outputs[0].text if out.outputs else ""
    finish_reason = out.outputs[0].finish_reason if out.outputs else None
    out_tokens = len(out.outputs[0].token_ids) if out.outputs else 0
    prompt_tokens_runtime = len(out.prompt_token_ids) if out.prompt_token_ids else None

    parsed: Any = None
    parse_error: str | None = None
    try:
        parsed = json.loads(raw_text)
    except Exception as exc:
        parse_error = f"{type(exc).__name__}: {exc}"

    validation = validate_against_schema(parsed, raw_schema) if parsed is not None else {
        "available": True, "valid": False, "error": "skipped because raw output did not parse"
    }

    # Save raw + parsed to metrics/baseline_smoke_output.json.
    write_json(
        args.output,
        {
            "model": args.model,
            "index": args.index,
            "nota_id": nota_id,
            "prompt_tokens_runtime": prompt_tokens_runtime,
            "output_tokens": out_tokens,
            "finish_reason": finish_reason,
            "elapsed_seconds": round(elapsed, 3),
            "raw_text": raw_text,
            "parsed": parsed,
            "parse_error": parse_error,
            "schema_validation": validation,
        },
    )

    report["output"] = {
        "path": str(args.output),
        "finish_reason": finish_reason,
        "output_tokens": out_tokens,
        "elapsed_seconds": round(elapsed, 3),
        "raw_chars": len(raw_text),
    }
    report["parse"] = {
        "valid": parse_error is None,
        "error": parse_error,
    }
    report["schema_validation"] = validation

    if parse_error is not None or not validation.get("valid", False):
        report["status"] = "fail"
    elif finish_reason == "length":
        report["status"] = "fail"
        report["errors"].append(
            "Generation hit max_tokens (finish_reason=length); output may be truncated."
        )
    else:
        report["status"] = "pass"

    write_json(args.report, report)
    print(
        f"[smoke] {report['status'].upper()} nota_id={nota_id} prompt_tokens={prompt_tokens} "
        f"output_tokens={out_tokens} parse_ok={parse_error is None} "
        f"schema_valid={validation.get('valid')} -> {args.report}"
    )
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())