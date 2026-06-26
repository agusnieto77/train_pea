#!/usr/bin/env python3
"""Format projected MVS rows as ChatML-style SFT examples and audit them."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


DEFAULT_SCHEMA = Path("esquema_eventos_protesta_entrenamiento_MVS.json")
DEFAULT_SYSTEM_DOC = Path("SYSTEM_PROMPT_GPT5_USADO.md")
DEFAULT_ORIGINAL = Path("entrenamiento.jsonl")
DEFAULT_TRAIN = Path("data/train_validated.jsonl")
DEFAULT_EVAL = Path("data/eval_set.jsonl")
DEFAULT_TRAIN_OUT = Path("data/chat_formatted/train.jsonl")
DEFAULT_EVAL_OUT = Path("data/chat_formatted/eval.jsonl")
DEFAULT_REPORT = Path("reports/chatml_audit.json")


MVS_OVERRIDE = """IMPORTANTE — TARGET MVS PARA ESTE ENTRENAMIENTO:
El output debe limitarse estrictamente al esquema MVS provisto.
Si el prompt histórico menciona campos ausentes del MVS, como
calidad_extraccion.*, observaciones_extraccion, metadatos_extraccion,
validacion_humana, voces_protagonistas o personas_mencionadas,
NO los generes. Representá la ambigüedad usando únicamente los campos
disponibles en MVS y, si no hay evidencia textual suficiente, usá "S/D"
o null según corresponda."""


FORBIDDEN_ASSISTANT_KEYS = {
    "texto_original",
    "subtitulo",
    "fuente",
    "archivo_fuente",
    "codebook_version",
    "metadatos_extraccion",
    "validacion_humana",
    "calidad_extraccion",
    "observaciones_extraccion",
    "voces_protagonistas",
    "personas_mencionadas",
    "fuente_de_la_cifra",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def nota_id(row: dict[str, Any]) -> str:
    return row["nota"]["nota_id"]


def extract_historical_prompt(system_doc: Path) -> str:
    text = system_doc.read_text(encoding="utf-8")
    marker = "## Prompt"
    start = text.index(marker)
    fenced = re.search(r"```\s*\n(.*?)\n```", text[start:], flags=re.DOTALL)
    if not fenced:
        raise ValueError(f"Could not find fenced historical prompt in {system_doc}")
    return fenced.group(1)


def resolve_ref(schema: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    ref = schema.get("$ref")
    if not ref:
        return schema
    if not ref.startswith("#/"):
        raise ValueError(f"Only local $ref values are supported, got {ref!r}")
    node: Any = root
    for part in ref[2:].split("/"):
        node = node[part]
    return node


def pointer_join(base: str, key: str) -> str:
    escaped = key.replace("~", "~0").replace("/", "~1")
    return f"{base}/{escaped}" if base else f"/{escaped}"


def collect_required_paths(schema: dict[str, Any], root: dict[str, Any], pointer: str = "") -> list[str]:
    schema = resolve_ref(schema, root)
    if schema.get("type") == "object":
        paths: list[str] = []
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            child_pointer = pointer_join(pointer, key)
            paths.append(child_pointer)
            if key in props:
                paths.extend(collect_required_paths(props[key], root, child_pointer))
        return paths
    if schema.get("type") == "array" and "items" in schema:
        return collect_required_paths(schema["items"], root, f"{pointer}/*")
    return []


def build_system_content(historical_prompt: str, schema: dict[str, Any]) -> str:
    paths = collect_required_paths(schema, schema)
    annex = "ANEXO MVS — JSON pointers requeridos:\n" + "\n".join(f"- {path}" for path in paths)
    return "\n\n".join([historical_prompt, MVS_OVERRIDE, annex])


def original_user_messages(original_path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for row in iter_jsonl(original_path):
        mapping[nota_id(row)] = row["nota"]["texto_original"]
    return mapping


def find_forbidden_keys(value: Any, pointer: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_pointer = pointer_join(pointer, key)
            if key in FORBIDDEN_ASSISTANT_KEYS or key.startswith("razonamiento_"):
                found.append(child_pointer)
            found.extend(find_forbidden_keys(child, child_pointer))
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            found.extend(find_forbidden_keys(child, f"{pointer}/{idx}"))
    return found


def token_proxy(text: str) -> int:
    # Conservative tokenizer-free proxy for Spanish prose + JSON. Good enough for
    # spotting max_seq_length risk before using the real Qwen tokenizer.
    return max(1, round(len(text) / 4))


def format_split(
    input_path: Path,
    output_path: Path,
    split_name: str,
    user_by_id: dict[str, str],
    system_content: str,
    validator: Draft202012Validator,
    max_seq_length: int,
) -> dict[str, Any]:
    rows = iter_jsonl(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    missing_user: list[str] = []
    forbidden_assistant: list[dict[str, Any]] = []
    schema_invalid: list[dict[str, Any]] = []
    duplicate_risks: list[str] = []
    token_stats: list[int] = []
    over_limit: list[dict[str, Any]] = []
    role_counts: Counter[str] = Counter()

    with output_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            row_id = nota_id(row)
            user_content = user_by_id.get(row_id)
            if user_content is None:
                missing_user.append(row_id)
                continue
            if user_content.count("FECHA DE EDICIÓN DE LA NOTA:") != 1:
                duplicate_risks.append(row_id)

            schema_errors = sorted(validator.iter_errors(row), key=lambda e: list(e.path))
            if schema_errors:
                schema_invalid.append({"nota_id": row_id, "errors": [e.message for e in schema_errors[:10]]})

            forbidden = find_forbidden_keys(row)
            if forbidden:
                forbidden_assistant.append({"nota_id": row_id, "paths": forbidden[:20]})

            assistant_content = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ]
            for message in messages:
                role_counts[message["role"]] += 1
            approx_tokens = sum(token_proxy(message["content"]) for message in messages)
            token_stats.append(approx_tokens)
            if approx_tokens > max_seq_length:
                over_limit.append({"nota_id": row_id, "approx_tokens": approx_tokens})

            f.write(json.dumps({"messages": messages}, ensure_ascii=False, separators=(",", ":")) + "\n")

    sorted_tokens = sorted(token_stats)
    percentile_95 = sorted_tokens[int(0.95 * (len(sorted_tokens) - 1))] if sorted_tokens else 0
    return {
        "split": split_name,
        "input": str(input_path),
        "output": str(output_path),
        "rows": len(rows),
        "written": len(rows) - len(missing_user),
        "role_counts": dict(sorted(role_counts.items())),
        "missing_user_messages": missing_user,
        "duplicate_user_message_risks": duplicate_risks,
        "assistant_schema_invalid": schema_invalid,
        "assistant_forbidden_paths": forbidden_assistant,
        "token_estimate": {
            "method": "ceil-ish char/4 proxy; replace with Qwen tokenizer audit before final training if desired",
            "max_seq_length": max_seq_length,
            "min": min(token_stats) if token_stats else 0,
            "max": max(token_stats) if token_stats else 0,
            "p95": percentile_95,
            "over_limit": over_limit,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--system-doc", type=Path, default=DEFAULT_SYSTEM_DOC)
    parser.add_argument("--original", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--train-output", type=Path, default=DEFAULT_TRAIN_OUT)
    parser.add_argument("--eval-output", type=Path, default=DEFAULT_EVAL_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-seq-length", type=int, default=20480)
    args = parser.parse_args()

    schema = load_json(args.schema)
    validator = Draft202012Validator(schema)
    historical_prompt = extract_historical_prompt(args.system_doc)
    system_content = build_system_content(historical_prompt, schema)
    system_hash = hashlib.sha256(system_content.encode("utf-8")).hexdigest()
    user_by_id = original_user_messages(args.original)

    train_report = format_split(
        args.train,
        args.train_output,
        "train",
        user_by_id,
        system_content,
        validator,
        args.max_seq_length,
    )
    eval_report = format_split(
        args.eval,
        args.eval_output,
        "eval",
        user_by_id,
        system_content,
        validator,
        args.max_seq_length,
    )

    report = {
        "system": {
            "historical_prompt_source": str(args.system_doc),
            "historical_prompt_sha256": hashlib.sha256(historical_prompt.encode("utf-8")).hexdigest(),
            "full_system_sha256": system_hash,
            "contains_mvs_override": MVS_OVERRIDE in system_content,
            "required_path_count": len(collect_required_paths(schema, schema)),
        },
        "splits": {"train": train_report, "eval": eval_report},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    critical_errors = []
    warnings = []
    for split_report in (train_report, eval_report):
        for key in (
            "missing_user_messages",
            "duplicate_user_message_risks",
            "assistant_schema_invalid",
            "assistant_forbidden_paths",
        ):
            if split_report[key]:
                critical_errors.append(f"{split_report['split']}:{key}")
        if split_report["token_estimate"]["over_limit"]:
            warnings.append(f"{split_report['split']}:token_over_limit")

    if warnings:
        report["warnings"] = warnings
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if critical_errors:
        print(f"ChatML formatting completed with critical audit errors: {critical_errors}. See {args.report}")
        return 1
    if warnings:
        print(f"ChatML OK with warnings {warnings}. Wrote {args.train_output}, {args.eval_output}, {args.report}")
    else:
        print(f"ChatML OK. Wrote {args.train_output}, {args.eval_output}, {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
