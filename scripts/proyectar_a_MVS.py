#!/usr/bin/env python3
"""Project validated v1.1.0 protest-event extractions to the MVS training schema.

This is Fase 1's first gate: no training data moves forward unless every row can
be projected to the MVS schema, validated, and checked against basic extraction
invariants.
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


DEFAULT_INPUT = Path("entrenamiento.jsonl")
DEFAULT_SCHEMA = Path("esquema_eventos_protesta_entrenamiento_MVS.json")
DEFAULT_OUTPUT = Path("data/mvs_projected.jsonl")
DEFAULT_REPORT = Path("reports/projection_report.json")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


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


def schema_type(schema: dict[str, Any]) -> str | None:
    t = schema.get("type")
    if isinstance(t, list):
        return next((item for item in t if item != "null"), None)
    return t


def normalize_for_match(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return " ".join(value.casefold().split())


def coerce_scalar(value: Any, schema: dict[str, Any]) -> Any:
    """Coerce common v1.1.0→MVS representation differences."""
    enum = schema.get("enum")
    if enum:
        if isinstance(value, bool) and set(enum) == {"Sí", "No", "S/D"}:
            return "Sí" if value else "No"
        if value is None:
            return "S/D" if "S/D" in enum else value
        if value in enum:
            return value
        if isinstance(value, str):
            normalized = normalize_for_match(value)
            for candidate in enum:
                if normalize_for_match(str(candidate)) == normalized:
                    return candidate
            if "S/D" in enum:
                return "S/D"
    t = schema_type(schema)
    if t == "string" and value is None:
        return "S/D"
    return value


def default_for_schema(schema: dict[str, Any], root_schema: dict[str, Any]) -> Any:
    schema = resolve_ref(schema, root_schema)
    if "const" in schema:
        return schema["const"]
    enum = schema.get("enum")
    if enum and "S/D" in enum:
        return "S/D"
    t = schema_type(schema)
    raw_type = schema.get("type")
    if isinstance(raw_type, list) and "null" in raw_type:
        return None
    if t == "object":
        return {
            key: default_for_schema(prop_schema, root_schema)
            for key, prop_schema in schema.get("properties", {}).items()
        }
    if t == "array":
        return []
    if t == "string":
        return "S/D"
    if t == "integer":
        return 0
    if t == "boolean":
        return False
    return None


def pointer_join(base: str, key: str) -> str:
    escaped = key.replace("~", "~0").replace("/", "~1")
    return f"{base}/{escaped}" if base else f"/{escaped}"


def project_to_schema(
    value: Any,
    schema: dict[str, Any],
    root_schema: dict[str, Any],
    pointer: str = "",
    dropped: Counter[str] | None = None,
) -> Any:
    schema = resolve_ref(schema, root_schema)
    t = schema_type(schema)

    if t == "object":
        if not isinstance(value, dict):
            return default_for_schema(schema, root_schema)
        props = schema.get("properties", {})
        if dropped is not None:
            for key in value.keys() - props.keys():
                dropped[pointer_join(pointer, key)] += 1
        projected: dict[str, Any] = {}
        for key, prop_schema in props.items():
            if "const" in prop_schema:
                projected[key] = prop_schema["const"]
            elif key in value:
                projected[key] = project_to_schema(
                    value[key], prop_schema, root_schema, pointer_join(pointer, key), dropped
                )
            else:
                projected[key] = default_for_schema(prop_schema, root_schema)

        # v1.1.0 incident indicators used boolean `presencia`; MVS keeps a
        # human-readable `presencia` enum and adds `valor_booleano`.
        if "presencia" in props and "valor_booleano" in props and isinstance(value.get("presencia"), bool):
            projected["presencia"] = "Sí" if value["presencia"] else "No"
            projected["valor_booleano"] = value["presencia"]
        return projected

    if t == "array":
        if not isinstance(value, list):
            return value
        item_schema = schema.get("items", {})
        return [project_to_schema(item, item_schema, root_schema, f"{pointer}/*", dropped) for item in value]

    if "const" in schema:
        return schema["const"]
    return coerce_scalar(value, schema)


def find_forbidden_paths(value: Any, forbidden_names: set[str], pointer: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_pointer = pointer_join(pointer, key)
            if key in forbidden_names or key.startswith("razonamiento_"):
                found.append(child_pointer)
            found.extend(find_forbidden_paths(child, forbidden_names, child_pointer))
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            found.extend(find_forbidden_paths(child, forbidden_names, f"{pointer}/{idx}"))
    return found


FALSE_EVENT_CORE_FIELDS = {"evento_id", "evento_numero", "es_evento_protesta"}


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
    """Return non-null top-level event-detail values for false-event records.

    Canonical convention: when ``es_evento_protesta`` is false, event-detail
    fields do not apply and must be null. ``S/D`` is valid only for unknown
    textual/categorical values inside real protest events.

    The check is intentionally top-level for non-core event-detail properties:
    empty containers like ``sujetos: []`` or all-null containers like
    ``accion: {"descripcion_textual": None}`` are still violations because the
    whole detail field must be ``null``.
    """
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
                violations.append(
                    f"eventos_protesta[{idx}]{pointer}: top-level value must be null, got {value!r}{context}"
                )
    return violations


def assert_false_event_detail_validator_catches_containers() -> None:
    """Guard the projection validator against leaf-only regressions."""

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

    failures: list[str] = []
    for name, row, should_violate in cases:
        violations = false_event_detail_violations(row)
        if bool(violations) is not should_violate:
            failures.append(f"{name}: violations={violations!r}, expected_violation={should_violate}")
    if failures:
        raise AssertionError("false-event detail validator regression: " + "; ".join(failures))


def nullify_false_event_details(row: dict[str, Any]) -> dict[str, Any]:
    """Deterministically enforce null details for false-event MVS records."""
    eventos = row.get("extraccion", {}).get("eventos_protesta", [])
    if not isinstance(eventos, list):
        return row
    for evento in eventos:
        if not isinstance(evento, dict) or evento.get("es_evento_protesta") is not False:
            continue
        for key in list(evento.keys()):
            if key not in FALSE_EVENT_CORE_FIELDS:
                evento[key] = None
    return row


def validate_invariants(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    extraccion = row.get("extraccion", {})
    eventos = extraccion.get("eventos_protesta", [])
    total = extraccion.get("total_eventos_protesta")
    tiene = extraccion.get("tiene_eventos_protesta")

    if total != len(eventos):
        errors.append(f"total_eventos_protesta={total} but len(eventos_protesta)={len(eventos)}")
    if tiene != (total > 0):
        errors.append(f"tiene_eventos_protesta={tiene} inconsistent with total_eventos_protesta={total}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    assert_false_event_detail_validator_catches_containers()

    schema = load_json(args.schema)
    validator = Draft202012Validator(schema)
    forbidden_names = {
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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    valid = 0
    invalid_rows: list[dict[str, Any]] = []
    source_false_event_detail_violations: list[dict[str, Any]] = []
    projected_false_event_detail_violations: list[dict[str, Any]] = []
    dropped_paths: Counter[str] = Counter()
    event_count_buckets: Counter[str] = Counter()
    has_events_counts: Counter[str] = Counter()

    with args.output.open("w", encoding="utf-8", newline="\n") as out:
        for line_no, source_row in iter_jsonl(args.input):
            processed += 1
            nota_id = source_row.get("nota", {}).get("nota_id", f"line-{line_no}")
            source_false_violations = false_event_detail_violations(source_row)
            if source_false_violations:
                source_false_event_detail_violations.append(
                    {"line": line_no, "nota_id": nota_id, "violations": source_false_violations[:20]}
                )
            projected = project_to_schema(source_row, schema, schema, dropped=dropped_paths)
            projected = nullify_false_event_details(projected)
            projected_false_violations = false_event_detail_violations(projected)
            if projected_false_violations:
                projected_false_event_detail_violations.append(
                    {"line": line_no, "nota_id": nota_id, "violations": projected_false_violations[:20]}
                )

            row_errors = []
            if source_false_violations:
                row_errors.append(
                    f"source false-event detail fields must be null: {source_false_violations[:10]}"
                )
            if projected_false_violations:
                row_errors.append(
                    f"projected false-event detail fields must be null: {projected_false_violations[:10]}"
                )
            row_errors.extend(validate_invariants(projected))

            forbidden_survivors = find_forbidden_paths(projected, forbidden_names)
            if forbidden_survivors:
                row_errors.append(f"forbidden paths survived: {forbidden_survivors[:10]}")

            schema_errors = sorted(validator.iter_errors(projected), key=lambda e: list(e.path))
            row_errors.extend(error.message for error in schema_errors)

            if row_errors:
                invalid_rows.append({"line": line_no, "nota_id": nota_id, "errors": row_errors[:20]})
                continue

            valid += 1
            total = projected["extraccion"]["total_eventos_protesta"]
            bucket = "3+" if total >= 3 else str(total)
            event_count_buckets[bucket] += 1
            has_events_counts[str(projected["extraccion"]["tiene_eventos_protesta"]).lower()] += 1
            out.write(json.dumps(projected, ensure_ascii=False, separators=(",", ":")) + "\n")

    report = {
        "input": str(args.input),
        "schema": str(args.schema),
        "output": str(args.output),
        "processed": processed,
        "valid": valid,
        "invalid": len(invalid_rows),
        "all_valid": processed == valid and processed > 0,
        "invariants_checked": [
            "total_eventos_protesta == len(eventos_protesta)",
            "tiene_eventos_protesta == (total_eventos_protesta > 0)",
            "no forbidden/podado fields survive in projected output",
            "source false-event detail fields are null",
            "projected false-event detail fields are null after deterministic normalization",
        ],
        "false_event_null_contract": {
            "source_rows_with_non_null_details": len(source_false_event_detail_violations),
            "projected_rows_with_non_null_details": len(
                projected_false_event_detail_violations
            ),
            "pass": not source_false_event_detail_violations
            and not projected_false_event_detail_violations,
        },
        "source_false_event_detail_violations": source_false_event_detail_violations[:20],
        "projected_false_event_detail_violations": projected_false_event_detail_violations[:20],
        "distributions": {
            "tiene_eventos_protesta": dict(sorted(has_events_counts.items())),
            "total_eventos_protesta_bucket": dict(sorted(event_count_buckets.items())),
        },
        "dropped_paths_top": dropped_paths.most_common(100),
        "invalid_rows": invalid_rows,
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if invalid_rows or processed == 0:
        print(f"Projection failed: {valid}/{processed} valid. See {args.report}")
        return 1
    print(f"Projection OK: {valid}/{processed} valid. Wrote {args.output} and {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
