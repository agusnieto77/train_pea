"""Generate current 317-r32 error analysis from existing eval artifacts.

This script is intentionally offline: it reads the already generated metrics,
outputs, and gold eval JSONL files. It does not train, run vLLM, call inference,
or mutate checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from baseline_qwen_full import EVENT_CATEGORICAL_PATHS, _resolve_path  # noqa: E402


DEFAULT_METRICS = REPO_ROOT / "metrics" / "finetuned_qwen-protesta-317-r32.json"
DEFAULT_OUTPUTS = REPO_ROOT / "metrics" / "finetuned_qwen-protesta-317-r32_outputs.jsonl"
DEFAULT_GOLD = REPO_ROOT / "data" / "chat_formatted" / "eval.jsonl"
DEFAULT_MD = REPO_ROOT / "metrics" / "error_analysis_317_r32.md"
DEFAULT_JSON = REPO_ROOT / "metrics" / "error_analysis_317_r32.json"
DEFAULT_FIELD_CSV = REPO_ROOT / "metrics" / "error_analysis_317_r32_field_errors.csv"
DEFAULT_WORST_CSV = REPO_ROOT / "metrics" / "error_analysis_317_r32_worst_examples.csv"


PRIORITY_PATHS: list[tuple[str, str]] = [
    ("extraccion.tiene_eventos_protesta", "extraccion.tiene_eventos_protesta"),
    ("extraccion.total_eventos_protesta", "extraccion.total_eventos_protesta"),
    ("delimitacion.criterio_delimitacion", "extraccion.eventos_protesta[].delimitacion_evento.criterio_delimitacion"),
    ("delimitacion.es_accion_principal_con_complementarias", "extraccion.eventos_protesta[].delimitacion_evento.es_accion_principal_con_complementarias"),
    ("accion.formato_principal.categoria", "extraccion.eventos_protesta[].accion.formato_principal.categoria"),
    ("demandas[].categoria", "extraccion.eventos_protesta[].demandas[].categoria"),
    ("sujetos[].categoria", "extraccion.eventos_protesta[].sujetos[].categoria"),
    ("sujetos[].organizaciones[].categoria", "extraccion.eventos_protesta[].sujetos[].organizaciones[].categoria"),
    ("contra_quien[].categoria", "extraccion.eventos_protesta[].contra_quien[].categoria"),
    ("contra_quien[].nivel_institucional", "extraccion.eventos_protesta[].contra_quien[].nivel_institucional"),
    ("lugares[].categoria", "extraccion.eventos_protesta[].lugares[].categoria"),
    ("temporalidad.tipo_temporal", "extraccion.eventos_protesta[].temporalidad.tipo_temporal"),
    ("temporalidad.tempo_verbal", "extraccion.eventos_protesta[].temporalidad.tempo_verbal"),
    ("incidentes.represion.presencia", "extraccion.eventos_protesta[].incidentes.represion.presencia"),
]


ACTION_HINTS = {
    "extraccion.tiene_eventos_protesta": "Agregar hard negatives de notas no-protesta y explicitar criterios de exclusión.",
    "extraccion.total_eventos_protesta": "Revisar reglas de segmentación y conteo de eventos por nota.",
    "delimitacion.criterio_delimitacion": "Anotar casos límite de acción principal + complementarias vs evento único.",
    "delimitacion.es_accion_principal_con_complementarias": "Agregar ejemplos donde la acción principal subsume acciones secundarias.",
    "accion.formato_principal.categoria": "Reforzar pares confusos del formato de acción con ejemplos contrastivos.",
    "demandas[].categoria": "Priorizar ejemplos contrastivos de demanda salarial/laboral/gremial y demandas múltiples.",
    "sujetos[].categoria": "Revisar etiquetas de sujeto para sindicatos, militantes, vecinos y trabajadores no organizados.",
    "sujetos[].organizaciones[].categoria": "Separar con ejemplos la categoría del sujeto de la categoría de su organización.",
    "contra_quien[].categoria": "Anotar destinatarios Estado/patronal/sindicato con evidencia textual explícita.",
    "contra_quien[].nivel_institucional": "Reforzar nivel municipal/provincial/nacional/privado cuando el destinatario es Estado o empresa.",
    "lugares[].categoria": "Agregar casos de vía pública, sede sindical, lugar de trabajo e institución pública.",
    "temporalidad.tipo_temporal": "Revisar fecha del evento vs fecha de publicación y menciones futuras/pasadas.",
    "temporalidad.tempo_verbal": "Agregar ejemplos con tiempos verbales ambiguos o notas retrospectivas.",
    "incidentes.represion.presencia": "Auditar después de corregir conteo/alineación; separar errores directos de errores por evento extra/faltante.",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_gold(path: Path) -> list[dict[str, Any]]:
    gold: list[dict[str, Any]] = []
    for row in load_jsonl(path):
        gold.append(json.loads(row["messages"][2]["content"]))
    return gold


def f1_from_counts(counts: dict[str, int]) -> float:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return (2 * precision * recall / (precision + recall)) if precision + recall else 0.0


def norm(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def compare_lists(gold_values: list[Any], pred_values: list[Any]) -> list[tuple[Any, Any, str]]:
    rows: list[tuple[Any, Any, str]] = []
    max_len = max(len(gold_values), len(pred_values))
    for i in range(max_len):
        g = gold_values[i] if i < len(gold_values) else None
        p = pred_values[i] if i < len(pred_values) else None
        if i >= len(gold_values):
            kind = "fp_extra_pred"
        elif i >= len(pred_values):
            kind = "fn_missing_pred"
        elif g == p:
            kind = "match"
        else:
            kind = "mismatch"
        rows.append((g, p, kind))
    return rows


def tiene_confusion(gold: list[dict[str, Any]], outputs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = {"tp": [], "tn": [], "fp": [], "fn": []}
    for g, o in zip(gold, outputs):
        pred = o["parsed"]
        g_value = bool(g["extraccion"].get("tiene_eventos_protesta"))
        p_value = bool(pred["extraccion"].get("tiene_eventos_protesta"))
        key = "tp" if g_value and p_value else "tn" if not g_value and not p_value else "fp" if p_value else "fn"
        buckets[key].append(example_summary(g, o))
    return buckets


def example_summary(gold: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
    pred = output["parsed"]
    counts = output["f1_vs_gold"]
    return {
        "nota_id": output["nota_id"],
        "gold_tiene": gold["extraccion"].get("tiene_eventos_protesta"),
        "pred_tiene": pred["extraccion"].get("tiene_eventos_protesta"),
        "gold_total": gold["extraccion"].get("total_eventos_protesta"),
        "pred_total": pred["extraccion"].get("total_eventos_protesta"),
        "f1": round(f1_from_counts(counts), 4),
        "tp": counts["tp"],
        "fp": counts["fp"],
        "fn": counts["fn"],
        "gold_leaves": counts["gold_leaves"],
        "pred_leaves": counts["pred_leaves"],
        "finish_reason": output.get("finish_reason"),
        "output_tokens": output.get("output_tokens"),
    }


def field_errors(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    per_path = metrics["metrics"]["categorical_accuracy"]["per_path"]
    rows: list[dict[str, Any]] = []
    for path, counts in per_path.items():
        support = counts.get("support", 0)
        accuracy = counts.get("accuracy", 0.0)
        rows.append(
            {
                "path": path,
                "tp": counts.get("tp", 0),
                "tn": counts.get("tn", 0),
                "fp": counts.get("fp", 0),
                "fn": counts.get("fn", 0),
                "support": support,
                "accuracy": accuracy,
                "error_rate": round(1 - accuracy, 4) if support else 0.0,
            }
        )
    rows.sort(key=lambda r: (r["error_rate"], r["support"]), reverse=True)
    return rows


def event_count_errors(gold: list[dict[str, Any]], outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for g, o in zip(gold, outputs):
        pred = o["parsed"]
        g_total = g["extraccion"].get("total_eventos_protesta") or 0
        p_total = pred["extraccion"].get("total_eventos_protesta") or 0
        rows.append({**example_summary(g, o), "delta": p_total - g_total})
    rows.sort(key=lambda r: (abs(r["delta"]), -r["f1"]), reverse=True)
    return rows


def iter_leaves(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        rows: list[tuple[str, Any]] = []
        for key, child in value.items():
            rows.extend(iter_leaves(child, f"{prefix}.{key}" if prefix else key))
        return rows
    if isinstance(value, list):
        rows = []
        for i, child in enumerate(value):
            rows.extend(iter_leaves(child, f"{prefix}[{i}]"))
        return rows
    return [(prefix, value)]


def false_null_contract(outputs: list[dict[str, Any]]) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    sd_count = 0
    for o in outputs:
        pred = o["parsed"]
        for event_index, event in enumerate(pred.get("extraccion", {}).get("eventos_protesta", []) or []):
            if event.get("es_evento_protesta") is not False:
                continue
            for path, value in iter_leaves(event):
                if path == "es_evento_protesta":
                    continue
                if value is not None:
                    if value == "S/D":
                        sd_count += 1
                    violations.append(
                        {
                            "nota_id": o["nota_id"],
                            "event_index": event_index,
                            "path": path,
                            "value": value,
                        }
                    )
    return {
        "violations": violations,
        "violation_count": len(violations),
        "sd_violation_count": sd_count,
        "status": "pass" if not violations else "fail",
    }


def high_value_diffs(gold: list[dict[str, Any]], outputs: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    priority_note_ids = [r["nota_id"] for r in sorted((example_summary(g, o) for g, o in zip(gold, outputs)), key=lambda x: x["f1"])[:12]]
    for g, o in zip(gold, outputs):
        if o["nota_id"] not in priority_note_ids:
            continue
        pred = o["parsed"]
        for label, path in PRIORITY_PATHS:
            for gv, pv, kind in compare_lists(_resolve_path(g, path), _resolve_path(pred, path)):
                if kind == "match":
                    continue
                rows.append(
                    {
                        "nota_id": o["nota_id"],
                        "path": label,
                        "correcto_gold": gv,
                        "prediccion_modelo": pv,
                        "tipo_error": kind,
                        "por_que_importa": why_it_matters(label),
                        "accion_recomendada": ACTION_HINTS.get(label, "Revisar el codebook y agregar ejemplos contrastivos."),
                    }
                )
    rank = {"extraccion.tiene_eventos_protesta": 0, "extraccion.total_eventos_protesta": 1, "demandas[].categoria": 2}
    rows.sort(key=lambda r: (rank.get(r["path"], 9), priority_note_ids.index(r["nota_id"])))
    return rows[:limit]


def why_it_matters(path: str) -> str:
    if path == "extraccion.tiene_eventos_protesta":
        return "Abre o cierra toda la extracción; un FP crea eventos inexistentes y contamina hojas downstream."
    if path == "extraccion.total_eventos_protesta":
        return "El alineamiento por índice hace que un conteo errado multiplique FP/FN en casi todos los campos."
    if path.startswith("delimitacion"):
        return "Define la unidad de análisis; errores aquí fragmentan o fusionan eventos."
    if path.startswith("demandas"):
        return "Es uno de los objetivos sustantivos centrales del dataset y tiene baja accuracy."
    if path.startswith("contra_quien"):
        return "Afecta inferencias sobre destinatarios institucionales y actores responsabilizados."
    return "Campo categórico de alto impacto en la calidad analítica del evento."


def worst_examples(gold: list[dict[str, Any]], outputs: list[dict[str, Any]], k: int = 5) -> list[dict[str, Any]]:
    rows = [example_summary(g, o) for g, o in zip(gold, outputs)]
    rows.sort(key=lambda r: (r["f1"], -(r["fp"] + r["fn"])))
    return [{"rank": i + 1, **r} for i, r in enumerate(rows[:k])]


def compact_example_diffs(gold: list[dict[str, Any]], outputs: list[dict[str, Any]], worst: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {o["nota_id"]: (g, o) for g, o in zip(gold, outputs)}
    out = []
    for w in worst:
        g, o = by_id[w["nota_id"]]
        pred = o["parsed"]
        diffs = []
        for label, path in PRIORITY_PATHS[:10]:
            for gv, pv, kind in compare_lists(_resolve_path(g, path), _resolve_path(pred, path)):
                if kind != "match":
                    diffs.append({"path": label, "gold": gv, "pred": pv, "tipo_error": kind})
                if len(diffs) >= 8:
                    break
            if len(diffs) >= 8:
                break
        out.append({"nota_id": w["nota_id"], "f1": w["f1"], "diffs": diffs})
    return out


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    metrics = load_json(args.metrics)
    outputs = load_jsonl(args.outputs)
    gold = load_gold(args.gold)
    if len(outputs) != len(gold):
        raise ValueError(f"outputs/gold length mismatch: {len(outputs)} != {len(gold)}")

    fields = field_errors(metrics)
    buckets = tiene_confusion(gold, outputs)
    counts = event_count_errors(gold, outputs)
    worst = worst_examples(gold, outputs)
    null_contract = false_null_contract(outputs)
    examples = high_value_diffs(gold, outputs)
    total_gold_events = sum((g["extraccion"].get("total_eventos_protesta") or 0) for g in gold)
    total_pred_events = sum((o["parsed"]["extraccion"].get("total_eventos_protesta") or 0) for o in outputs)
    event_count_summary = {
        "gold_total_events": total_gold_events,
        "pred_total_events": total_pred_events,
        "delta": total_pred_events - total_gold_events,
        "examples_exact_count": sum(1 for r in counts if r["delta"] == 0),
        "examples_extra_count": sum(1 for r in counts if r["delta"] > 0),
        "examples_missing_count": sum(1 for r in counts if r["delta"] < 0),
    }
    tiene_metrics = metrics["metrics"]["categorical_accuracy"]["tiene_eventos_protesta"]
    return {
        "model_label": args.model_label,
        "current_scope": {
            "dataset_rows": 317,
            "split": {"train": 285, "eval": 32},
            "gold_weight": 1.0,
            "formal_origin": "GPT-5.4-mini + Nico validation",
            "historical_350_comparable": False,
        },
        "inputs": {"metrics": str(args.metrics), "outputs": str(args.outputs), "gold": str(args.gold)},
        "headline_metrics": {
            "schema_validity": metrics["metrics"]["schema_validity"],
            "f1_global": metrics["metrics"]["f1_global"]["f1"],
            "f1_precision": metrics["metrics"]["f1_global"]["precision"],
            "f1_recall": metrics["metrics"]["f1_global"]["recall"],
            "categorical_accuracy_aggregate": metrics["metrics"]["categorical_accuracy"]["per_path"]["__aggregate__"]["accuracy"],
            "tiene_eventos_protesta_accuracy": tiene_metrics["accuracy"],
            "tiene_eventos_protesta": tiene_metrics,
            "examples_total": metrics["examples_total"],
        },
        "tiene_confusion": {k: v for k, v in buckets.items()},
        "event_count_summary": event_count_summary,
        "event_count_errors": counts,
        "field_errors": fields,
        "top_field_errors": fields[:10],
        "false_null_contract": null_contract,
        "gold_vs_pred_examples": examples,
        "worst_examples": worst,
        "worst_example_diffs": compact_example_diffs(gold, outputs, worst),
        "annotation_targets": annotation_targets(fields, buckets, event_count_summary, null_contract),
    }


def annotation_targets(fields: list[dict[str, Any]], buckets: dict[str, list[dict[str, Any]]], event_counts: dict[str, Any], null_contract: dict[str, Any]) -> list[dict[str, str]]:
    top = [f for f in fields if f["path"] != "__aggregate__"][:5]
    targets = [
        {
            "target": "Hard negatives para `tiene_eventos_protesta`",
            "evidence": f"Hay {len(buckets['fp'])} FP y {len(buckets['fn'])} FN en 32 filas; el modelo no produjo TN.",
            "action": "Agregar/curar notas no-protesta parecidas a protesta y exigir detalles null cuando `tiene=false`.",
        },
        {
            "target": "Delimitación y conteo de eventos",
            "evidence": f"Gold total={event_counts['gold_total_events']}, pred total={event_counts['pred_total_events']}, delta={event_counts['delta']}; {event_counts['examples_extra_count']} notas tienen eventos de más y {event_counts['examples_missing_count']} eventos de menos.",
            "action": "Anotar ejemplos contrastivos de evento único vs acciones complementarias y revisar `total_eventos_protesta`.",
        },
    ]
    for f in top:
        targets.append(
            {
                "target": f"Codebook/examples para `{f['path']}`",
                "evidence": f"accuracy={f['accuracy']:.4f}, error_rate={f['error_rate']:.4f}, support={f['support']}, fp={f['fp']}, fn={f['fn']}.",
                "action": ACTION_HINTS.get(f["path"], "Agregar ejemplos contrastivos y revisar definiciones operativas."),
            }
        )
    if null_contract["violation_count"]:
        targets.append(
            {
                "target": "Contrato false-event null",
                "evidence": f"{null_contract['violation_count']} detalles no-null en eventos predichos con `es_evento_protesta=false`; {null_contract['sd_violation_count']} usan `S/D`.",
                "action": "Agregar validación/ejemplos donde false events tengan detalles null, no `S/D`.",
            }
        )
    return targets


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(x).replace("\n", " ") for x in row) + " |")
    return "\n".join(out)


def render_markdown(payload: dict[str, Any]) -> str:
    h = payload["headline_metrics"]
    tc = h["tiene_eventos_protesta"]
    lines = [
        "# Error analysis — clean 317-r32 model",
        "",
        "Este reporte analiza **solo** el modelo clean `317-r32` sobre el eval actual de 32 filas. No compara deltas contra artefactos históricos 350-era, porque son no comparables.",
        "",
        "## Executive summary",
        "",
        f"- Schema validity: **{h['schema_validity']:.4f}**.",
        f"- f1_global: **{h['f1_global']:.4f}** (precision={h['f1_precision']:.4f}, recall={h['f1_recall']:.4f}).",
        f"- Categorical accuracy aggregate: **{h['categorical_accuracy_aggregate']:.4f}**.",
        f"- `extraccion.tiene_eventos_protesta` accuracy: **{h['tiene_eventos_protesta_accuracy']:.4f}** (TP={tc['tp']}, TN={tc['tn']}, FP={tc['fp']}, FN={tc['fn']}).",
        "- Lectura clave: el modelo detecta todos los casos positivos del eval, pero convierte los 5 negativos en protesta; el cuello de botella pasa por hard negatives, conteo/delimitación de eventos y categorías de demanda/acción/sujeto/destinatario.",
        "",
        "## Headline metrics — current 317-r32 only",
        "",
        md_table(
            ["metric", "value"],
            [
                ["schema_validity", f"{h['schema_validity']:.4f}"],
                ["f1_global", f"{h['f1_global']:.4f}"],
                ["categorical_accuracy_aggregate", f"{h['categorical_accuracy_aggregate']:.4f}"],
                ["tiene_eventos_protesta accuracy", f"{h['tiene_eventos_protesta_accuracy']:.4f}"],
                ["eval examples", h["examples_total"]],
            ],
        ),
        "",
        "> Historical 350-era metrics are intentionally omitted from the headline table. They may be useful as provenance, but are not a valid direct delta against this 317-row split.",
        "",
        "## Error taxonomy",
        "",
        "### `extraccion.tiene_eventos_protesta`",
        "",
        md_table(
            ["bucket", "count", "nota_id examples"],
            [[k.upper(), len(v), ", ".join(x["nota_id"] for x in v[:3]) or "—"] for k, v in payload["tiene_confusion"].items()],
        ),
        "",
        "### Event count errors",
        "",
        md_table(
            ["gold_total_events", "pred_total_events", "delta", "exact", "extra", "missing"],
            [[payload["event_count_summary"][k] for k in ["gold_total_events", "pred_total_events", "delta", "examples_exact_count", "examples_extra_count", "examples_missing_count"]]],
        ),
        "",
        "### Top categorical paths by error rate",
        "",
        md_table(
            ["path", "accuracy", "error_rate", "support", "fp", "fn"],
            [[r["path"], f"{r['accuracy']:.4f}", f"{r['error_rate']:.4f}", r["support"], r["fp"], r["fn"]] for r in payload["top_field_errors"]],
        ),
        "",
        "### False-event null contract",
        "",
        f"Status: **{payload['false_null_contract']['status']}**. Violations={payload['false_null_contract']['violation_count']}; `S/D` violations={payload['false_null_contract']['sd_violation_count']}.",
        "",
        "## Gold-vs-pred high-value examples",
        "",
        md_table(
            ["nota_id", "path/campo", "correcto_gold", "prediccion_modelo", "tipo_error", "por_que_importa", "accion_recomendada"],
            [[r["nota_id"], r["path"], norm(r["correcto_gold"]), norm(r["prediccion_modelo"]), r["tipo_error"], r["por_que_importa"], r["accion_recomendada"]] for r in payload["gold_vs_pred_examples"]],
        ),
        "",
        "## Worst examples — top 5 by per-example F1",
        "",
        md_table(
            ["rank", "nota_id", "f1", "tp", "fp", "fn", "gold_total", "pred_total"],
            [[r["rank"], r["nota_id"], f"{r['f1']:.4f}", r["tp"], r["fp"], r["fn"], r["gold_total"], r["pred_total"]] for r in payload["worst_examples"]],
        ),
        "",
    ]
    for item in payload["worst_example_diffs"]:
        lines.extend([
            f"### {item['nota_id']} — f1={item['f1']:.4f}",
            "",
            md_table(["path", "gold", "pred", "tipo_error"], [[d["path"], norm(d["gold"]), norm(d["pred"]), d["tipo_error"]] for d in item["diffs"]]),
            "",
        ])
    lines.extend([
        "## Annotation/action plan",
        "",
        md_table(["target", "evidence", "action"], [[t["target"], t["evidence"], t["action"]] for t in payload["annotation_targets"]]),
        "",
        "## Caveats",
        "",
        "- Eval actual pequeño: 32 filas; usar como diagnóstico, no como conclusión estadística definitiva.",
        "- Artefactos 350-era son históricos y no comparables contra este split 317.",
        "- El alineamiento de eventos es por índice; si `pred_total != gold_total`, muchos FP/FN downstream reflejan desalineación, no necesariamente error semántico independiente del campo.",
        "- Contrato vigente: en falsos eventos los detalles deben ser `null`; `S/D` solo corresponde dentro de eventos reales con valor textual/categorial desconocido.",
        "",
        "## Sources",
        "",
        f"- `{payload['inputs']['metrics']}`",
        f"- `{payload['inputs']['outputs']}`",
        f"- `{payload['inputs']['gold']}`",
    ])
    return "\n".join(lines) + "\n"


def write_csvs(payload: dict[str, Any], field_path: Path, worst_path: Path) -> None:
    with field_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "tp", "tn", "fp", "fn", "support", "accuracy", "error_rate"])
        w.writeheader()
        w.writerows(payload["field_errors"])
    with worst_path.open("w", encoding="utf-8", newline="") as f:
        columns = ["rank", "nota_id", "f1", "tp", "fp", "fn", "gold_leaves", "pred_leaves", "gold_total", "pred_total", "gold_tiene", "pred_tiene", "finish_reason", "output_tokens"]
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for row in payload["worst_examples"]:
            w.writerow({k: row[k] for k in columns})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--outputs", type=Path, default=DEFAULT_OUTPUTS)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--model-label", default="clean 317-r32")
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--field-csv-out", type=Path, default=DEFAULT_FIELD_CSV)
    parser.add_argument("--worst-csv-out", type=Path, default=DEFAULT_WORST_CSV)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_payload(args)
    args.md_out.write_text(render_markdown(payload), encoding="utf-8")
    args.json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csvs(payload, args.field_csv_out, args.worst_csv_out)
    print(f"Wrote {args.md_out}")
    print(f"Wrote {args.json_out}")
    print(f"Wrote {args.field_csv_out}")
    print(f"Wrote {args.worst_csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
