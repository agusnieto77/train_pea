#!/usr/bin/env python3
"""Batch extractor de eventos de protesta desde CSV.

Usa OpenAI Batch API sobre /v1/responses con Structured Outputs.
Cada linea del batch corresponde a una nota; custom_id = txt_file.
El input de la nota se construye como: fecha + "\n\n" + texto.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
from jsonschema import Draft202012Validator, ValidationError

DEFAULT_MODEL = "gpt-5.5"
DEFAULT_ENDPOINT = "/v1/responses"
DEFAULT_SCHEMA_NAME = "extraccion_eventos_protesta_v1_1_0"
DEFAULT_PROMPT_FILE = ""
DEFAULT_PROMPT_CACHE_KEY = "tesis_nico_eventos_protesta_v1_1_0"
DEFAULT_SCHEMA_VERSION = "1.1.0"
DEFAULT_CODEBOOK_VERSION = "2026-05-31_revision_LLM"
MAX_BATCH_BYTES = 200 * 1024 * 1024

DEFAULT_SYSTEM_PROMPT = """\
Eres un cientifico social experto en codificacion de eventos de protesta en notas periodisticas historicas.

Contrato de extraccion:
- Unidad de registro: la nota periodistica completa. Devuelve un unico objeto JSON para la nota.
- Unidad de analisis: el evento de protesta o accion conflictiva.
- Una nota puede contener cero, uno o varios eventos. Cada item de extraccion.eventos_protesta debe representar exactamente un evento/accion atomico.
- Si la nota identifica mas de una accion, separalas solo cuando puedan delimitarse espacial y/o temporalmente con claridad.
- No confundas acciones complementarias con eventos independientes. Ejemplo: un corte con volanteada, junta de firmas, carpa o quema de gomas suele ser un unico evento con accion principal y formatos complementarios, salvo delimitacion temporal/espacial clara.
- Un mismo evento puede tener multiples sujetos, organizaciones, demandas, destinatarios contra_quien y lugares; representalos como arrays sin duplicar eventos.
- Prioridad absoluta: extrae primero la cita textual exacta de la nota y luego clasifica segun las categorias del esquema.
- Si una variable no aparece en el texto, usa estrictamente "S/D" en campos textuales/categoriales y null en campos numericos no observados.
- Las fechas de salida deben estar en formato DD/MM/AAAA. La primera linea del input es la fecha de edicion/publicacion; usala para resolver expresiones relativas como "ayer", "hoy", "manana", "el jueves", etc.
- Si el texto anuncia una accion futura, registra como evento el anuncio, no la accion futura realizada, salvo que la nota tambien informe que esa accion efectivamente ocurrio.
- Para ahorrar salida, en nota.texto_original devuelve "S/D"; no repitas el texto completo de la nota.
- No inventes informacion. Cuando haya ambiguedad, registrala en calidad_extraccion.ambiguedades u observaciones_extraccion.
"""


def load_system_prompt(prompt_path: Optional[str]) -> str:
    if not prompt_path:
        return DEFAULT_SYSTEM_PROMPT
    p = Path(prompt_path)
    if not p.exists():
        raise FileNotFoundError(f"No se encontro el archivo de prompt: {p}")
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"El archivo de prompt esta vacio: {p}")
    return text


@dataclass(frozen=True)
class RowInput:
    row_index: int
    txt_file: str
    fecha: str
    texto: str

    @property
    def input_text(self) -> str:
        return f"FECHA DE EDICIÓN DE LA NOTA: {self.fecha}\n\n{self.texto}"


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def normalize_cell(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def read_csv_with_fallback(path: Path, encoding: Optional[str]) -> pd.DataFrame:
    if encoding:
        return pd.read_csv(path, encoding=encoding)
    last_error: Optional[Exception] = None
    for enc in ("utf-8", "utf-8-sig", "latin1"):
        try:
            logging.info("Leyendo CSV con encoding=%s", enc)
            return pd.read_csv(path, encoding=enc)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise RuntimeError(f"No pude leer el CSV con utf-8, utf-8-sig ni latin1. Ultimo error: {last_error}")


def require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}. Columnas disponibles: {list(df.columns)}")


def iso_to_ddmmyyyy(value: str) -> str:
    value = normalize_cell(value)
    if not value:
        return "S/D"
    for fmt in ("%Y-%m-%d", "%Y_%m_%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(value[:10], fmt).strftime("%d/%m/%Y")
        except ValueError:
            pass
    parsed = pd.to_datetime(value, errors="coerce", dayfirst=False)
    if pd.isna(parsed):
        return "S/D"
    return parsed.strftime("%d/%m/%Y")


def load_schema(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        schema = json.load(f)
    if schema.get("type") != "object":
        raise ValueError("El schema raiz debe ser type=object.")
    return schema


def prepare_schema_for_openai(schema: Dict[str, Any]) -> Dict[str, Any]:
    prepared = copy.deepcopy(schema)

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            node.pop("$schema", None)
            node.pop("$id", None)
            node.pop("title", None)
            if "const" in node:
                node["enum"] = [node.pop("const")]
            for key, value in list(node.items()):
                node[key] = walk(value)
        elif isinstance(node, list):
            return [walk(x) for x in node]
        return node

    return walk(prepared)


def assert_strict_schema_shape(schema: Dict[str, Any]) -> None:
    problems: List[str] = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                props = set(node.get("properties", {}).keys())
                req = set(node.get("required", []))
                missing = sorted(props - req)
                if missing:
                    problems.append(f"{path}: propiedades no requeridas: {missing}")
                if node.get("additionalProperties") is not False:
                    problems.append(f"{path}: additionalProperties debe ser false")
            for key, value in node.items():
                walk(value, f"{path}/{key}")
        elif isinstance(node, list):
            for i, value in enumerate(node):
                walk(value, f"{path}[{i}]")

    walk(schema, "#")
    if problems:
        raise ValueError("El schema no cumple requisitos estrictos:\n" + "\n".join(problems[:80]))


def load_processed_ids(out_path: Optional[Path]) -> Set[str]:
    processed: Set[str] = set()
    if out_path is None or not out_path.exists():
        return processed
    with out_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                nota_id = obj.get("nota", {}).get("nota_id")
                if nota_id:
                    processed.add(str(nota_id))
            except json.JSONDecodeError:
                logging.warning("Linea invalida en salida existente %s:%s", out_path, line_no)
    return processed


def build_rows(
    df: pd.DataFrame,
    id_col: str,
    date_col: str,
    text_col: str,
    start: int,
    limit: Optional[int],
    ids: Optional[List[str]],
    skip_ids: Set[str],
) -> List[RowInput]:
    require_columns(df, [id_col, date_col, text_col])
    if start:
        df = df.iloc[start:]
    if ids:
        wanted = set(ids)
        df = df[df[id_col].astype(str).isin(wanted)]
    if skip_ids:
        df = df[~df[id_col].astype(str).isin(skip_ids)]
    if limit is not None:
        df = df.head(limit)

    rows: List[RowInput] = []
    seen: Set[str] = set()
    for idx, row in df.iterrows():
        txt_file = normalize_cell(row[id_col])
        fecha = normalize_cell(row[date_col])
        texto = normalize_cell(row[text_col])
        if not txt_file:
            raise ValueError(f"Fila {idx}: ID vacio en {id_col}")
        if txt_file in seen:
            raise ValueError(f"ID duplicado en {id_col}: {txt_file}. Batch requiere custom_id unico.")
        if not fecha:
            raise ValueError(f"Fila {idx}: fecha vacia en {date_col}")
        if not texto:
            raise ValueError(f"Fila {idx}: texto vacio en {text_col}")
        seen.add(txt_file)
        rows.append(RowInput(row_index=int(idx), txt_file=txt_file, fecha=fecha, texto=texto))
    return rows


def build_user_message(row: RowInput) -> str:
    """Contenido exacto enviado al modelo como mensaje de usuario: fecha + texto."""
    return row.input_text


def build_batch_request(row: RowInput, schema_for_openai: Dict[str, Any], system_prompt: str, args: argparse.Namespace) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "model": args.model,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_user_message(row)},
        ],
        "reasoning": {"effort": args.reasoning_effort},
        "text": {
            "format": {
                "type": "json_schema",
                "name": args.schema_name,
                "schema": schema_for_openai,
                "strict": True,
            },
            "verbosity": args.verbosity,
        },
        "max_output_tokens": args.max_output_tokens,
    }
    if args.prompt_cache_key:
        body["prompt_cache_key"] = args.prompt_cache_key
    return {"custom_id": row.txt_file, "method": "POST", "url": DEFAULT_ENDPOINT, "body": body}


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            n += 1
    return n


def append_jsonl_line(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def manifest_record(row: RowInput, store_text: bool) -> Dict[str, Any]:
    rec = {"custom_id": row.txt_file, "row_index": row.row_index, "txt_file": row.txt_file, "fecha": row.fecha}
    if store_text:
        rec["texto"] = row.texto
    return rec


def load_manifest(path: Path) -> Dict[str, RowInput]:
    rows: Dict[str, RowInput] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            custom_id = str(rec["custom_id"])
            rows[custom_id] = RowInput(
                row_index=int(rec.get("row_index", line_no - 1)),
                txt_file=str(rec.get("txt_file", custom_id)),
                fecha=str(rec.get("fecha", "S/D")),
                texto=str(rec.get("texto", "")),
            )
    return rows


def postprocess(obj: Dict[str, Any], row: RowInput, store_full_text: bool, args: Optional[argparse.Namespace] = None) -> Dict[str, Any]:
    # Metadatos deterministas: SIEMPRE los inyecta el script desde sus propios defaults.
    # El modelo puede haberlos inventado (alucinación de su propio nombre, modo, etc);
    # los pisamos con los valores reales del pipeline.
    obj["schema_version"] = DEFAULT_SCHEMA_VERSION
    obj["codebook_version"] = DEFAULT_CODEBOOK_VERSION
    nota = obj.setdefault("nota", {})
    nota["nota_id"] = row.txt_file
    nota["archivo_fuente"] = row.txt_file
    nota["fecha_publicacion"] = iso_to_ddmmyyyy(row.fecha)
    if store_full_text:
        if not row.texto:
            raise RuntimeError("--store-full-text requiere haber preparado el manifest con --manifest-store-text")
        nota["texto_original"] = row.texto
    else:
        nota["texto_original"] = "S/D"
    if args is not None:
        fecha_extraccion = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        obj["metadatos_extraccion"] = {
            "modelo": getattr(args, "model", DEFAULT_MODEL),
            "proveedor": "OpenAI",
            "fecha_extraccion": fecha_extraccion,
            "prompt_version": getattr(args, "prompt_version", None) or DEFAULT_PROMPT_CACHE_KEY,
            "extractor_tipo": "LLM",
            "modo_procesamiento": "Batch API",
            "estado_validacion_humana": "No validado",
            "observaciones_metodologicas": (
                "Extracción generada con OpenAI Batch API y Structured Outputs; "
                "pipeline v1.1.0 (schema + codebook + prompt auditados). "
                "Metadatos inyectados por el script post-procesador, no por el modelo. "
                "Requiere validación humana antes de uso en publicación."
            ),
            "codebook_version": DEFAULT_CODEBOOK_VERSION,
        }

    extraccion = obj.setdefault("extraccion", {})
    eventos = extraccion.get("eventos_protesta")
    if not isinstance(eventos, list):
        eventos = []
        extraccion["eventos_protesta"] = eventos
    extraccion["total_eventos_protesta"] = len(eventos)
    extraccion["tiene_eventos_protesta"] = len(eventos) > 0
    if not extraccion.get("observaciones_extraccion"):
        extraccion["observaciones_extraccion"] = "S/D"
    for i, evento in enumerate(eventos, start=1):
        if isinstance(evento, dict):
            evento["evento_numero"] = i
            evento["evento_id"] = f"{row.txt_file}__evento_{i:03d}"
    return obj


def extract_response_text(body: Dict[str, Any]) -> str:
    if body.get("status") and body.get("status") != "completed":
        raise RuntimeError(f"Responses API no completada: status={body.get('status')}; details={body.get('incomplete_details')}")
    if isinstance(body.get("output_text"), str) and body["output_text"].strip():
        return body["output_text"]
    chunks: List[str] = []
    for item in body.get("output", []) or []:
        for part in item.get("content", []) or []:
            part_type = part.get("type")
            if part_type in {"refusal", "output_refusal"}:
                raise RuntimeError(f"El modelo rechazo la solicitud: {part.get('refusal') or part.get('text')}")
            if part_type in {"output_text", "text"} and isinstance(part.get("text"), str):
                chunks.append(part["text"])
    if chunks:
        return "".join(chunks)
    raise RuntimeError("No se encontro output_text en response.body")


def usage_from_body(body: Dict[str, Any]) -> Dict[str, int]:
    usage = body.get("usage") or {}
    details = usage.get("input_tokens_details") or usage.get("prompt_tokens_details") or {}
    input_tokens = int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0)
    output_tokens = int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0)
    total_tokens = int(usage.get("total_tokens", input_tokens + output_tokens) or 0)
    cached_tokens = int(details.get("cached_tokens", 0) or 0)
    return {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total_tokens, "cached_input_tokens": cached_tokens}


def parse_raw_batch_output(
    raw_output: Path,
    manifest: Path,
    schema_path: Path,
    out: Path,
    errors_out: Path,
    usage_out: Optional[Path],
    store_full_text: bool,
    append: bool,
    fail_fast: bool,
    args: Optional[argparse.Namespace] = None,
) -> Tuple[int, int]:
    original_schema = load_schema(schema_path)
    validator = Draft202012Validator(original_schema)
    rows = load_manifest(manifest)
    by_id: Dict[str, Dict[str, Any]] = {}
    usage_by_id: Dict[str, Dict[str, int]] = {}
    errors: List[Dict[str, Any]] = []

    with raw_output.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            custom_id = "S/D"
            try:
                envelope = json.loads(line)
                custom_id = str(envelope.get("custom_id", "S/D"))
                if custom_id not in rows:
                    raise RuntimeError(f"custom_id no encontrado en manifest: {custom_id}")
                if envelope.get("error"):
                    raise RuntimeError(f"Error remoto batch: {envelope['error']}")
                response = envelope.get("response") or {}
                status_code = int(response.get("status_code", 0))
                if status_code < 200 or status_code >= 300:
                    raise RuntimeError(f"HTTP {status_code}: {response.get('body')}")
                body = response.get("body") or {}
                text = extract_response_text(body)
                parsed = json.loads(text)
                if not isinstance(parsed, dict):
                    raise RuntimeError(f"La salida no es objeto JSON raiz: {type(parsed).__name__}")
                parsed = postprocess(parsed, rows[custom_id], store_full_text, args)
                validator.validate(parsed)
                by_id[custom_id] = parsed
                usage_by_id[custom_id] = usage_from_body(body)
            except (json.JSONDecodeError, ValidationError, Exception) as exc:  # noqa: BLE001
                errors.append({"custom_id": custom_id, "line_no": line_no, "error_type": type(exc).__name__, "error": str(exc)})
                if fail_fast:
                    raise

    out.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    ok = 0
    with out.open(mode, encoding="utf-8") as f:
        for row in sorted(rows.values(), key=lambda r: r.row_index):
            obj = by_id.get(row.txt_file)
            if obj is None:
                errors.append({"custom_id": row.txt_file, "error_type": "MissingOutput", "error": "No hay salida exitosa para este custom_id"})
                continue
            f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")
            ok += 1

    write_jsonl(errors_out, errors)
    if usage_out:
        total = {
            "requests_with_usage": len(usage_by_id),
            "input_tokens": sum(x["input_tokens"] for x in usage_by_id.values()),
            "output_tokens": sum(x["output_tokens"] for x in usage_by_id.values()),
            "total_tokens": sum(x["total_tokens"] for x in usage_by_id.values()),
            "cached_input_tokens": sum(x["cached_input_tokens"] for x in usage_by_id.values()),
            "by_custom_id": usage_by_id,
        }
        usage_out.parent.mkdir(parents=True, exist_ok=True)
        usage_out.write_text(json.dumps(total, ensure_ascii=False, indent=2), encoding="utf-8")
    return ok, len(errors)


def to_plain(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if hasattr(obj, "dict"):
        return obj.dict()
    return {"repr": repr(obj)}


def get_openai_client() -> Any:
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("Falta OPENAI_API_KEY en el entorno.")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("No esta instalado openai. Ejecuta: pip install -r requirements_extraccion.txt") from exc
    return OpenAI()


def write_file_response(response: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(response, "write_to_file"):
        response.write_to_file(str(path))
        return
    if hasattr(response, "read"):
        data = response.read()
    elif hasattr(response, "content"):
        data = response.content
    elif hasattr(response, "text"):
        data = response.text() if callable(response.text) else response.text
    else:
        data = str(response)
    if isinstance(data, bytes):
        path.write_bytes(data)
    else:
        path.write_text(str(data), encoding="utf-8")


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_batch_id(batch_id: Optional[str], batch_meta: Optional[str]) -> str:
    if batch_id:
        return batch_id
    if not batch_meta:
        raise ValueError("Debes pasar --batch-id o --batch-meta")
    meta = load_json(Path(batch_meta))
    if meta.get("id"):
        return str(meta["id"])
    if isinstance(meta.get("batch"), dict) and meta["batch"].get("id"):
        return str(meta["batch"]["id"])
    raise ValueError(f"No pude encontrar id de batch en {batch_meta}")


def command_prepare(args: argparse.Namespace) -> int:
    schema = load_schema(Path(args.schema))
    schema_for_openai = prepare_schema_for_openai(schema)
    # assert_strict_schema_shape(schema_for_openai)  # Desactivado: la regla "toda property en required" es excesiva para OpenAI Structured Outputs.
    df = read_csv_with_fallback(Path(args.csv), args.encoding)
    skip = load_processed_ids(Path(args.resume_from_out)) if args.resume_from_out else set()
    rows = build_rows(df, args.id_col, args.date_col, args.text_col, args.start, args.limit, args.ids, skip)
    system_prompt = load_system_prompt(args.prompt_file)
    requests = [build_batch_request(row, schema_for_openai, system_prompt, args) for row in rows]
    manifest = [manifest_record(row, args.manifest_store_text) for row in rows]

    if args.dry_run:
        print("DRY RUN OK")
        print(f"CSV: {df.shape[0]} filas, {df.shape[1]} columnas")
        print(f"Requests batch: {len(requests)}")
        if requests:
            print("\n--- PRIMER REQUEST BATCH ---")
            try:
                print(json.dumps(requests[0], ensure_ascii=False, indent=2)[:6000])
            except UnicodeEncodeError:
                # Windows cp1252 no soporta algunos chars; usamos utf-8
                sys.stdout.buffer.write(json.dumps(requests[0], ensure_ascii=False, indent=2)[:6000].encode("utf-8"))
                sys.stdout.buffer.write(b"\n")
        return 0

    n_req = write_jsonl(Path(args.batch_input), requests)
    n_manifest = write_jsonl(Path(args.manifest), manifest)
    size = Path(args.batch_input).stat().st_size
    if size > MAX_BATCH_BYTES:
        raise RuntimeError(f"El batch input supera 200 MB ({size / 1024 / 1024:.1f} MB). Dividilo con --start/--limit.")
    meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "csv": args.csv,
        "schema": args.schema,
        "batch_input": args.batch_input,
        "manifest": args.manifest,
        "model": args.model,
        "endpoint": DEFAULT_ENDPOINT,
        "requests": n_req,
        "manifest_rows": n_manifest,
        "batch_input_mb": round(size / 1024 / 1024, 3),
        "id_rule": "custom_id = txt_file",
        "input_rule": "'FECHA DE EDICIÓN DE LA NOTA: ' + fecha + '\\n\\n' + texto",
    }
    if args.prepare_meta:
        save_json(Path(args.prepare_meta), meta)
    logging.info("Preparado: %s requests | %.3f MB | %s", n_req, size / 1024 / 1024, args.batch_input)
    logging.info("Manifest: %s filas | %s", n_manifest, args.manifest)
    return 0


def command_submit(args: argparse.Namespace) -> int:
    client = get_openai_client()
    path = Path(args.batch_input)
    if not path.exists():
        raise FileNotFoundError(f"No existe batch input: {path}")
    with path.open("rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint=DEFAULT_ENDPOINT,
        completion_window="24h",
        metadata={"description": args.description, "source_file": path.name},
    )
    meta = {"uploaded_file": to_plain(uploaded), "batch": to_plain(batch), "id": batch.id}
    save_json(Path(args.batch_meta), meta)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


def command_status(args: argparse.Namespace) -> int:
    client = get_openai_client()
    batch_id = resolve_batch_id(args.batch_id, args.batch_meta)
    batch = client.batches.retrieve(batch_id)
    obj = to_plain(batch)
    print(json.dumps(obj, ensure_ascii=False, indent=2))
    if args.save:
        save_json(Path(args.save), obj)
    elif args.batch_meta:
        # Actualiza el archivo de metadatos preservando los datos locales si existen.
        meta_path = Path(args.batch_meta)
        try:
            meta = load_json(meta_path)
        except Exception:  # noqa: BLE001
            meta = {}
        meta["batch"] = obj
        meta["id"] = obj.get("id", batch_id)
        save_json(meta_path, meta)
    return 0


def command_download(args: argparse.Namespace) -> int:
    client = get_openai_client()
    batch_id = resolve_batch_id(args.batch_id, args.batch_meta)
    batch = client.batches.retrieve(batch_id)
    obj = to_plain(batch)
    status = obj.get("status")
    if status != "completed" and not args.allow_incomplete:
        raise RuntimeError(f"El batch no esta completed: status={status}. Usa status o --allow-incomplete si ya hay output_file_id.")
    output_file_id = obj.get("output_file_id")
    if not output_file_id:
        raise RuntimeError(f"El batch no tiene output_file_id. status={status}")
    raw_out = Path(args.raw_out or f"batch_output_{batch_id}.jsonl")
    raw_errors = Path(args.raw_errors_out or f"batch_errors_{batch_id}.jsonl")
    write_file_response(client.files.content(output_file_id), raw_out)
    if obj.get("error_file_id"):
        write_file_response(client.files.content(obj["error_file_id"]), raw_errors)
    else:
        raw_errors.write_text("", encoding="utf-8")
    if args.no_parse:
        print(json.dumps({"raw_out": str(raw_out), "raw_errors": str(raw_errors)}, ensure_ascii=False, indent=2))
        return 0
    errors_out = Path(args.errors_out or (args.out + ".errors.jsonl"))
    usage_out = Path(args.usage_out) if args.usage_out else Path(args.out + ".usage.json")
    ok, errors = parse_raw_batch_output(raw_out, Path(args.manifest), Path(args.schema), Path(args.out), errors_out, usage_out, args.store_full_text, args.append, args.fail_fast, args)
    # Agrega errores remotos crudos, si los hubiera.
    if raw_errors.exists() and raw_errors.stat().st_size > 0:
        with raw_errors.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    append_jsonl_line(errors_out, {"error_type": "BatchErrorFile", "raw": json.loads(line)})
                    errors += 1
    print(json.dumps({"ok": ok, "errors": errors, "out": args.out, "errors_out": str(errors_out), "usage_out": str(usage_out)}, ensure_ascii=False, indent=2))
    return 0 if errors == 0 else 2


def command_parse(args: argparse.Namespace) -> int:
    errors_out = Path(args.errors_out or (args.out + ".errors.jsonl"))
    usage_out = Path(args.usage_out) if args.usage_out else Path(args.out + ".usage.json")
    ok, errors = parse_raw_batch_output(Path(args.raw_out), Path(args.manifest), Path(args.schema), Path(args.out), errors_out, usage_out, args.store_full_text, args.append, args.fail_fast, args)
    if args.raw_errors_out:
        raw_errors = Path(args.raw_errors_out)
        if raw_errors.exists() and raw_errors.stat().st_size > 0:
            with raw_errors.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        append_jsonl_line(errors_out, {"error_type": "BatchErrorFile", "raw": json.loads(line)})
                        errors += 1
    print(json.dumps({"ok": ok, "errors": errors, "out": args.out, "errors_out": str(errors_out), "usage_out": str(usage_out)}, ensure_ascii=False, indent=2))
    return 0 if errors == 0 else 2


def add_selection_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--id-col", default="txt_file")
    p.add_argument("--date-col", default="fecha")
    p.add_argument("--text-col", default="texto")
    p.add_argument("--encoding", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--ids", nargs="*", default=None)
    p.add_argument("--resume-from-out", default=None, help="JSONL final existente; omite nota.nota_id ya procesados.")


def add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--schema-name", default=DEFAULT_SCHEMA_NAME)
    p.add_argument("--prompt-file", default=DEFAULT_PROMPT_FILE, help="Ruta al .md con el system prompt. Cadena vacia usa el prompt por defecto embebido.")
    p.add_argument("--reasoning-effort", default="medium", choices=["none", "low", "medium", "high", "xhigh"])
    p.add_argument("--verbosity", default="low", choices=["low", "medium", "high"])
    p.add_argument("--max-output-tokens", type=int, default=32000)
    p.add_argument("--prompt-cache-key", default=DEFAULT_PROMPT_CACHE_KEY, help="Usar cadena vacia para omitir.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extraccion batch de eventos de protesta con OpenAI Batch API.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="Genera JSONL batch y manifest local desde CSV.")
    p.add_argument("--csv", required=True)
    p.add_argument("--schema", required=True)
    p.add_argument("--batch-input", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--prepare-meta", default=None)
    p.add_argument("--manifest-store-text", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    add_selection_args(p)
    add_model_args(p)

    p = sub.add_parser("submit", help="Sube JSONL batch y crea el job remoto.")
    p.add_argument("--batch-input", required=True)
    p.add_argument("--batch-meta", required=True)
    p.add_argument("--description", default="tesis_nico_eventos_protesta")

    p = sub.add_parser("status", help="Consulta estado del batch.")
    p.add_argument("--batch-id", default=None)
    p.add_argument("--batch-meta", default=None)
    p.add_argument("--save", default=None)

    p = sub.add_parser("download", help="Descarga output_file_id y produce JSONL limpio.")
    p.add_argument("--batch-id", default=None)
    p.add_argument("--batch-meta", default=None)
    p.add_argument("--manifest", required=True)
    p.add_argument("--schema", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--raw-out", default=None)
    p.add_argument("--raw-errors-out", default=None)
    p.add_argument("--errors-out", default=None)
    p.add_argument("--usage-out", default=None)
    p.add_argument("--store-full-text", action="store_true")
    p.add_argument("--append", action="store_true")
    p.add_argument("--no-parse", action="store_true")
    p.add_argument("--allow-incomplete", action="store_true")
    p.add_argument("--fail-fast", action="store_true")

    p = sub.add_parser("parse", help="Convierte un batch raw ya descargado en JSONL limpio.")
    p.add_argument("--raw-out", required=True)
    p.add_argument("--raw-errors-out", default=None)
    p.add_argument("--manifest", required=True)
    p.add_argument("--schema", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--errors-out", default=None)
    p.add_argument("--usage-out", default=None)
    p.add_argument("--store-full-text", action="store_true")
    p.add_argument("--append", action="store_true")
    p.add_argument("--fail-fast", action="store_true")

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    if args.command == "prepare":
        return command_prepare(args)
    if args.command == "submit":
        return command_submit(args)
    if args.command == "status":
        return command_status(args)
    if args.command == "download":
        return command_download(args)
    if args.command == "parse":
        return command_parse(args)
    raise RuntimeError(f"Comando desconocido: {args.command}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrumpido por usuario.", file=sys.stderr)
        raise SystemExit(130)
