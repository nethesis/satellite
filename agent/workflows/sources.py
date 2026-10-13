"""Typed tables and local document retrieval. Configuration contains no expressions."""

import base64
import copy
import io
import json
import math
import re
from collections import Counter
from datetime import date, datetime
from decimal import Decimal, InvalidOperation

from agent.application.contracts import ApplicationError, fields, identifier, integer, string

TYPES = {"string", "integer", "number", "boolean", "date", "month"}
OPERATORS = {"eq", "ne", "in", "lt", "lte", "gt", "gte"}


def columns(value):
    if not isinstance(value, list) or not 1 <= len(value) <= 128:
        raise ApplicationError("invalid_columns")
    seen = set()
    for column in value:
        fields(column, ("name", "source", "type"), ("required", "sensitive", "identity"))
        identifier(column["name"]); string(column["source"], 128)
        if column["name"] in seen or column["name"] == "source_row" or column["type"] not in TYPES:
            raise ApplicationError("invalid_columns")
        seen.add(column["name"])
        for flag in ("required", "sensitive", "identity"):
            if flag in column and type(column[flag]) is not bool:
                raise ApplicationError("invalid_columns")
    return value


def typed(value, column, decimal_separator):
    if value is None or value == "":
        if column.get("required"):
            raise ValueError("missing_value")
        return None
    kind = column["type"]
    if kind == "string":
        if not isinstance(value, str):
            raise ValueError("text_column_requires_text")
        if len(value) > 8192:
            raise ValueError("value_too_long")
        return value.strip()
    if kind in ("number", "integer"):
        if isinstance(value, bool):
            raise ValueError("invalid_number")
        text = str(value)
        if isinstance(value, str) and decimal_separator == ",":
            text = text.replace(".", "").replace(",", ".")
        try:
            number = Decimal(text)
            if not number.is_finite() or abs(number) > Decimal("1e15"):
                raise ValueError("invalid_number")
            if kind == "integer":
                if number != number.to_integral_value():
                    raise ValueError("invalid_integer")
                return int(number)
            return float(number)
        except InvalidOperation:
            raise ValueError("invalid_number") from None
    if kind == "boolean":
        if type(value) is bool:
            return value
        if str(value).lower() not in ("true", "false", "1", "0"):
            raise ValueError("invalid_boolean")
        return str(value).lower() in ("true", "1")
    if isinstance(value, (date, datetime)):
        value = value.date().isoformat() if isinstance(value, datetime) else value.isoformat()
    value = str(value)
    if kind == "month":
        if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", value):
            raise ValueError("invalid_month")
    else:
        try:
            if date.fromisoformat(value).isoformat() != value:
                raise ValueError()
        except ValueError:
            raise ValueError("invalid_date") from None
    return value


def normalize(values, cfg):
    from .data import MAX_CELLS, MAX_ROWS
    header_index = cfg["header_row"] - 1
    if len(values) <= header_index:
        raise ApplicationError("missing_header")
    header = [str(x or "").strip() for x in values[header_index]]
    if len(header) != len(set(header)) or any(c["source"] not in header for c in cfg["columns"]):
        raise ApplicationError("missing_or_duplicate_columns")
    positions = [(c, header.index(c["source"])) for c in cfg["columns"]]
    rows, errors, count = [], [], 0
    for index, cells in enumerate(values[header_index + 1:], header_index + 2):
        count += len(cells)
        if len(rows) >= MAX_ROWS or count > MAX_CELLS:
            raise ApplicationError("table_too_large")
        if not any(v is not None and v != "" for v in cells):
            continue
        try:
            row = {c["name"]: typed(cells[p] if p < len(cells) else None, c, cfg["decimal_separator"]) for c, p in positions}
            rows.append(row | {"source_row": index})
        except ValueError as exc:
            errors.append({"row": index, "error": str(exc)})
            if len(errors) >= 100:
                break
    if not rows and not errors:
        raise ApplicationError("empty_table")
    return rows, errors


def preview_row(row, cfg):
    hidden = {c["name"] for c in cfg.get("columns", []) if c.get("sensitive")}
    return {k: v for k, v in row.items() if k not in hidden and k not in ("verification_code", "name_key", "phone", "text")}


def query_spec(value):
    fields(value, ("resource", "fields", "filters", "limit"), ("aggregate", "sort", "max_age_seconds"))
    fields(value["resource"], ("resource_id", "version"))
    identifier(value["resource"]["resource_id"])
    if value["resource"]["version"] != "latest":
        integer(value["resource"]["version"], 1, 1000000)
    if not isinstance(value["fields"], list) or not 1 <= len(value["fields"]) <= 32:
        raise ApplicationError("invalid_query")
    for name in value["fields"]:
        identifier(name)
    integer(value["limit"], 1, 100)
    if not isinstance(value["filters"], list) or len(value["filters"]) > 16:
        raise ApplicationError("invalid_query")
    for predicate in value["filters"]:
        fields(predicate, ("field", "operator"), ("value", "input", "identity"))
        identifier(predicate["field"])
        if predicate["operator"] not in OPERATORS or sum(k in predicate for k in ("value", "input", "identity")) != 1:
            raise ApplicationError("invalid_query")
        for key in ("input", "identity"):
            if key in predicate:
                identifier(predicate[key])
    if "aggregate" in value:
        fields(value["aggregate"], ("operation", "field"))
        if value["aggregate"]["operation"] not in ("count", "sum", "min", "max", "avg"):
            raise ApplicationError("invalid_aggregate")
        identifier(value["aggregate"]["field"])
    if "sort" in value:
        fields(value["sort"], ("field", "direction")); identifier(value["sort"]["field"])
        if value["sort"]["direction"] not in ("asc", "desc"):
            raise ApplicationError("invalid_query")
    integer(value.get("max_age_seconds", 86400), 60, 31536000)
    return value


def matches(actual, operator, expected):
    if operator == "in":
        return isinstance(expected, list) and any(matches(actual, "eq", value) for value in expected)
    numbers = type(actual) in (int, float) and type(expected) in (int, float)
    compatible = type(actual) is type(expected) or numbers
    if operator == "eq": return compatible and actual == expected
    if operator == "ne": return compatible and actual != expected
    if actual is None or not compatible: return False
    if operator == "lt": return actual < expected
    if operator == "lte": return actual <= expected
    if operator == "gt": return actual > expected
    return actual >= expected


def query(rows, cfg, args, identity, scope, source_columns):
    names = {c["name"] for c in source_columns}
    hidden = {c["name"] for c in source_columns if c.get("sensitive")} | {"verification_code", "name_key", "phone"}
    requested = set(cfg["fields"]) | {p["field"] for p in cfg["filters"]}
    requested |= {cfg[k]["field"] for k in ("aggregate", "sort") if k in cfg}
    if not requested <= names or set(cfg["fields"]) & hidden or cfg.get("aggregate", {}).get("field") in hidden:
        raise ApplicationError("unknown_or_sensitive_field", 403)
    identity_field = scope.get("identity_field")
    if identity_field:
        expected = identity.get(identity_field) if identity.get("verified") else None
        if expected is None or identity_field not in names:
            raise ApplicationError("identity_denied", 403)
        rows = [r for r in rows if matches(r.get(identity_field), "eq", expected)]
    for name, value in scope.get("filters", {}).items():
        if name not in names:
            raise ApplicationError("invalid_scope", 403)
        rows = [r for r in rows if matches(r.get(name), "eq", value)]
    for predicate in cfg["filters"]:
        if "input" in predicate:
            if predicate["input"] not in args:
                raise ApplicationError("missing_query_input")
            expected = args[predicate["input"]]
        elif "identity" in predicate:
            if not identity.get("verified") or predicate["identity"] not in identity:
                raise ApplicationError("identity_denied", 403)
            expected = identity[predicate["identity"]]
        else:
            expected = predicate["value"]
        rows = [r for r in rows if matches(r.get(predicate["field"]), predicate["operator"], expected)]
    count = len(rows)
    if "aggregate" in cfg:
        spec = cfg["aggregate"]; values = [r[spec["field"]] for r in rows if r.get(spec["field"]) is not None]
        if spec["operation"] == "count": result = len(values)
        else:
            if any(type(v) not in (int, float) for v in values):
                raise ApplicationError("aggregate_requires_number")
            decimals = [Decimal(str(v)) for v in values]
            result = None if not decimals else float(sum(decimals) / len(decimals) if spec["operation"] == "avg" else sum(decimals) if spec["operation"] == "sum" else min(decimals) if spec["operation"] == "min" else max(decimals))
        return {"rows": [], "count": count, "value": result, "source_rows": []}
    if "sort" in cfg:
        field = cfg["sort"]["field"]
        rows = sorted(rows, key=lambda r: (r.get(field) is None, r.get(field)), reverse=cfg["sort"]["direction"] == "desc")
    selected = rows[:cfg["limit"]]
    return {"rows": [{k: r.get(k) for k in cfg["fields"]} for r in selected], "count": count,
            "source_rows": [r["source_row"] for r in selected]}


def documents(raw, cfg):
    """Executed inside the existing killable ingestion process."""
    from .data import MAX_UPLOAD
    try:
        files = json.loads(raw)["files"]
    except (ValueError, KeyError, TypeError):
        raise ApplicationError("invalid_documents") from None
    if not isinstance(files, list) or not 0 <= len(files) <= 50:
        raise ApplicationError("document_capacity")
    rows, errors, ids, total = [], [], set(), 0
    total_bytes = 0
    for item in files:
        fields(item, ("document_id", "name", "format", "content"))
        identifier(item["document_id"]); string(item["name"], 128)
        if item["document_id"] in ids or item["format"] not in ("pdf", "txt"):
            raise ApplicationError("invalid_documents")
        ids.add(item["document_id"])
        try:
            content = base64.b64decode(item["content"], validate=True)
        except (ValueError, TypeError):
            raise ApplicationError("invalid_upload") from None
        total_bytes += len(content)
        if total_bytes > MAX_UPLOAD:
            raise ApplicationError("upload_too_large", 413)
        if item["format"] == "pdf":
            from pypdf import PdfReader
            try:
                reader = PdfReader(io.BytesIO(content), strict=True)
                if reader.is_encrypted or len(reader.pages) > 500:
                    raise ApplicationError("unsupported_or_large_pdf")
                pages = [page.extract_text() or "" for page in reader.pages]
            except ApplicationError:
                raise
            except Exception:
                raise ApplicationError("invalid_pdf") from None
        else:
            try:
                pages = [content.decode("utf-8-sig")]
            except UnicodeDecodeError:
                raise ApplicationError("invalid_text_encoding") from None
        if not any(text.strip() for text in pages):
            # An unreadable document stays visible, but never supplies a passage.
            rows.append({"document_id": item["document_id"], "document": item["name"], "page": None,
                         "text": "", "state": "needs_verification", "source_row": len(rows) + 1})
            continue
        for page_number, text in enumerate(pages, 1):
            total += len(text.encode())
            if total > 10 * 1024**2:
                raise ApplicationError("extracted_text_capacity")
            for start in range(0, len(text), 1200):
                passage = text[start:start + 1600].strip()
                if passage:
                    rows.append({"document_id": item["document_id"], "document": item["name"],
                                 "page": page_number if item["format"] == "pdf" else None, "section": start // 1200 + 1,
                                 "text": passage, "state": "indexed", "source_row": len(rows) + 1})
                if len(rows) > 10000:
                    raise ApplicationError("document_capacity")
    return rows, errors


def tokens(text):
    import unicodedata
    return re.findall(r"\w+", "".join(c for c in unicodedata.normalize("NFKD", text.casefold()) if not unicodedata.combining(c)))


def search(rows, question, limit=5):
    string(question, 1024); integer(limit, 1, 10)
    passages = [r for r in rows if r.get("state") == "indexed" and r.get("text")]
    terms = set(tokens(question))
    frequencies = [Counter(tokens(r["text"])) for r in passages]
    if not frequencies or not terms:
        return {"passages": [], "status": "not_found"}
    average = sum(sum(f.values()) for f in frequencies) / len(frequencies)
    document_frequency = {t: sum(t in f for f in frequencies) for t in terms}
    ranked = []
    for row, counts in zip(passages, frequencies):
        score = 0.0
        for term in terms:
            tf = counts[term]
            if tf:
                inverse = math.log(1 + (len(frequencies) - document_frequency[term] + .5) / (document_frequency[term] + .5))
                score += inverse * tf * 2.5 / (tf + 1.5 * (.25 + .75 * sum(counts.values()) / average))
        if score:
            ranked.append((score, row))
    ranked.sort(key=lambda pair: (-pair[0], pair[1]["document_id"], pair[1].get("page") or 0, pair[1].get("section", 0)))
    return {"status": "found" if ranked else "not_found", "passages": [copy.deepcopy(row) for _, row in ranked[:limit]]}
