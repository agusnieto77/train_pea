#!/usr/bin/env python3
"""Create a reproducible train/eval split for the projected MVS dataset."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("data/mvs_projected.jsonl")
DEFAULT_TRAIN = Path("data/train_validated.jsonl")
DEFAULT_EVAL = Path("data/eval_set.jsonl")
DEFAULT_REPORT = Path("reports/split_manifest.json")


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def nota_id(row: dict[str, Any]) -> str:
    return row["nota"]["nota_id"]


def total_events(row: dict[str, Any]) -> int:
    return int(row["extraccion"]["total_eventos_protesta"])


def event_bucket(row: dict[str, Any]) -> str:
    total = total_events(row)
    return "3+" if total >= 3 else str(total)


def stratum_key(row: dict[str, Any]) -> str:
    has_events = str(bool(row["extraccion"]["tiene_eventos_protesta"])).lower()
    return f"has_events={has_events}|events={event_bucket(row)}"


def primary_action_categories(row: dict[str, Any]) -> list[str]:
    categories: list[str] = []
    for event in row["extraccion"].get("eventos_protesta", []):
        category = (
            event.get("accion", {})
            .get("formato_principal", {})
            .get("categoria", "S/D")
        )
        categories.append(category or "S/D")
    return categories or ["<sin_eventos>"]


def ensure_eval_category_coverage(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    min_total_count: int = 3,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Swap rows within the same hard stratum to cover evaluable action classes.

    The hard split key remains presence + event-count bucket. Within those cells,
    this makes the eval set more useful for categorical checks by ensuring every
    primary action category with at least `min_total_count` occurrences appears at
    least once when a same-stratum swap is possible.
    """
    all_rows = train_rows + eval_rows
    all_category_counts: Counter[str] = Counter()
    for row in all_rows:
        all_category_counts.update(primary_action_categories(row))

    target_categories = {
        category
        for category, count in all_category_counts.items()
        if category != "<sin_eventos>" and count >= min_total_count
    }
    swaps: list[dict[str, Any]] = []

    def eval_category_counts() -> Counter[str]:
        counts: Counter[str] = Counter()
        for row in eval_rows:
            counts.update(primary_action_categories(row))
        return counts

    for category in sorted(target_categories):
        if eval_category_counts()[category] > 0:
            continue

        donor_idx = next(
            (
                idx
                for idx, row in enumerate(train_rows)
                if category in primary_action_categories(row)
            ),
            None,
        )
        if donor_idx is None:
            continue
        donor = train_rows[donor_idx]
        donor_stratum = stratum_key(donor)
        current_eval_counts = eval_category_counts()

        receiver_idx = None
        for idx, row in enumerate(eval_rows):
            if stratum_key(row) != donor_stratum:
                continue
            row_categories = primary_action_categories(row)
            if all(cat == "<sin_eventos>" or current_eval_counts[cat] > 1 for cat in row_categories):
                receiver_idx = idx
                break
        if receiver_idx is None:
            continue

        receiver = eval_rows[receiver_idx]
        train_rows[donor_idx] = receiver
        eval_rows[receiver_idx] = donor
        swaps.append(
            {
                "category_added_to_eval": category,
                "train_to_eval": nota_id(donor),
                "eval_to_train": nota_id(receiver),
                "stratum": donor_stratum,
            }
        )

    return train_rows, eval_rows, swaps


def distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    has_events = Counter(str(bool(row["extraccion"]["tiene_eventos_protesta"])).lower() for row in rows)
    buckets = Counter(event_bucket(row) for row in rows)
    strata = Counter(stratum_key(row) for row in rows)
    action_categories: Counter[str] = Counter()
    for row in rows:
        action_categories.update(primary_action_categories(row))
    return {
        "rows": len(rows),
        "tiene_eventos_protesta": dict(sorted(has_events.items())),
        "total_eventos_protesta_bucket": dict(sorted(buckets.items())),
        "strata": dict(sorted(strata.items())),
        "accion_formato_principal_categoria": dict(sorted(action_categories.items())),
    }


def allocate_eval_counts(strata: dict[str, list[dict[str, Any]]], eval_size: int) -> dict[str, int]:
    total = sum(len(rows) for rows in strata.values())
    allocations: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []

    for key, rows in strata.items():
        exact = len(rows) * eval_size / total
        base = int(exact)
        if len(rows) > 1 and base == 0:
            base = 1
        base = min(base, len(rows) - 1) if len(rows) > 1 else 0
        allocations[key] = base
        remainders.append((exact - int(exact), key))

    current = sum(allocations.values())
    for _, key in sorted(remainders, reverse=True):
        if current >= eval_size:
            break
        if allocations[key] < len(strata[key]):
            allocations[key] += 1
            current += 1

    for _, key in sorted(remainders):
        if current <= eval_size:
            break
        if allocations[key] > 0 and len(strata[key]) - allocations[key] > 0:
            allocations[key] -= 1
            current -= 1

    if sum(allocations.values()) != eval_size:
        raise RuntimeError(f"Could not allocate eval_size={eval_size}; got {sum(allocations.values())}")
    return allocations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--train-output", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--eval-output", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-ratio", type=float, default=0.10)
    args = parser.parse_args()

    rows = iter_jsonl(args.input)
    if not rows:
        raise SystemExit(f"No rows found in {args.input}")

    seen_ids = [nota_id(row) for row in rows]
    duplicates = sorted(item for item, count in Counter(seen_ids).items() if count > 1)
    if duplicates:
        raise SystemExit(f"Duplicate nota_id values found: {duplicates[:10]}")

    eval_size = round(len(rows) * args.eval_ratio)
    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[stratum_key(row)].append(row)

    rng = random.Random(args.seed)
    allocations = allocate_eval_counts(strata, eval_size)
    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []

    for key in sorted(strata):
        group = list(strata[key])
        rng.shuffle(group)
        n_eval = allocations[key]
        eval_rows.extend(group[:n_eval])
        train_rows.extend(group[n_eval:])

    train_rows, eval_rows, category_coverage_swaps = ensure_eval_category_coverage(train_rows, eval_rows)

    train_rows.sort(key=nota_id)
    eval_rows.sort(key=nota_id)

    if len(train_rows) + len(eval_rows) != len(rows):
        raise SystemExit("Split size mismatch")
    if set(map(nota_id, train_rows)) & set(map(nota_id, eval_rows)):
        raise SystemExit("Train/eval overlap detected")

    write_jsonl(args.train_output, train_rows)
    write_jsonl(args.eval_output, eval_rows)

    report = {
        "input": str(args.input),
        "train_output": str(args.train_output),
        "eval_output": str(args.eval_output),
        "seed": args.seed,
        "eval_ratio": args.eval_ratio,
        "counts": {"all": len(rows), "train": len(train_rows), "eval": len(eval_rows)},
        "stratification": {
            "hard_key": "tiene_eventos_protesta + total_eventos_protesta bucket (0/1/2/3+)",
            "allocations": dict(sorted(allocations.items())),
            "category_coverage": {
                "policy": "within-stratum swaps to include every accion.formato_principal.categoria with >=3 total occurrences when possible",
                "swaps": category_coverage_swaps,
            },
        },
        "nota_ids": {
            "train": [nota_id(row) for row in train_rows],
            "eval": [nota_id(row) for row in eval_rows],
        },
        "distributions": {
            "all": distribution(rows),
            "train": distribution(train_rows),
            "eval": distribution(eval_rows),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        f"Split OK: {len(train_rows)} train / {len(eval_rows)} eval "
        f"(seed={args.seed}). Wrote {args.train_output}, {args.eval_output}, {args.report}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
