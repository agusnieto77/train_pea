#!/usr/bin/env python3
"""Validate the false-event null contract for protest-event JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


FALSE_EVENT_CORE_FIELDS = {"evento_id", "evento_numero", "es_evento_protesta"}


def pointer_join(base: str, key: str) -> str:
    escaped = key.replace("~", "~0").replace("/", "~1")
    return f"{base}/{escaped}" if base else f"/{escaped}"


def iter_leaf_values(value: Any, pointer: str = ""):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from iter_leaf_values(child, pointer_join(pointer, key))
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            yield from iter_leaf_values(child, f"{pointer}/{idx}")
    else:
        yield pointer, value


def false_event_detail_violations(row: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    eventos = row.get("extraccion", {}).get("eventos_protesta", [])
    if not isinstance(eventos, list):
        return violations
    for idx, evento in enumerate(eventos):
        if not isinstance(evento, dict) or evento.get("es_evento_protesta") is not False:
            continue
        for key, value in evento.items():
            if key in FALSE_EVENT_CORE_FIELDS:
                continue
            if value is not None:
                pointer = pointer_join("", key)
                leaf_context = [
                    leaf_pointer
                    for leaf_pointer, leaf_value in iter_leaf_values(value, pointer)
                    if leaf_value is not None
                ]
                context = f"; non_null_leaf_paths={leaf_context[:5]!r}" if leaf_context else ""
                violations.append(f"eventos_protesta[{idx}]{pointer}: top-level value must be null, got {value!r}{context}")
    return violations


def run_self_test() -> int:
    def row_with_event_detail(key: str, value: Any) -> dict[str, Any]:
        return {
            "extraccion": {
                "eventos_protesta": [
                    {
                        "evento_id": "evt-1",
                        "evento_numero": 1,
                        "es_evento_protesta": False,
                        key: value,
                    }
                ]
            }
        }

    cases = [
        ("sujetos: []", row_with_event_detail("sujetos", []), True),
        ("incidentes: {}", row_with_event_detail("incidentes", {}), True),
        (
            'accion: {"descripcion_textual": None}',
            row_with_event_detail("accion", {"descripcion_textual": None}),
            True,
        ),
        ("accion: None", row_with_event_detail("accion", None), False),
    ]

    failures: list[dict[str, Any]] = []
    for name, row, should_violate in cases:
        violations = false_event_detail_violations(row)
        passed = bool(violations) is should_violate
        print(f"{'PASS' if passed else 'FAIL'} self-test {name}: violations={violations}")
        if not passed:
            failures.append({"case": name, "violations": violations, "expected_violation": should_violate})
    return 0 if not failures else 1


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                yield line_no, json.loads(line)


def iter_validation_rows(row: dict[str, Any]):
    """Yield event-extraction payloads from raw projected rows or ChatML rows."""
    if "extraccion" in row:
        yield "row", row
        return

    messages = row.get("messages")
    if not isinstance(messages, list):
        yield "row", row
        return

    for idx, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            yield f"messages[{idx}].content", parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="Run deterministic false-event edge-case checks")
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args()

    if args.self_test:
        return run_self_test()
    if not args.paths:
        parser.error("at least one path is required unless --self-test is used")

    total_rows = 0
    total_violating_rows = 0
    for path in args.paths:
        path_rows = 0
        path_violations: list[dict[str, Any]] = []
        for line_no, row in iter_jsonl(path):
            path_rows += 1
            for source, validation_row in iter_validation_rows(row):
                violations = false_event_detail_violations(validation_row)
                if violations:
                    path_violations.append({"line": line_no, "source": source, "violations": violations[:20]})
        total_rows += path_rows
        total_violating_rows += len(path_violations)
        if path_violations:
            print(f"FAIL {path}: {len(path_violations)}/{path_rows} rows violate false-event null contract")
            print(json.dumps(path_violations[:10], ensure_ascii=False, indent=2))
        else:
            print(f"PASS {path}: {path_rows} rows, 0 false-event detail violations")

    return 0 if total_rows > 0 and total_violating_rows == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
