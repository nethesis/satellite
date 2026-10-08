"""Strict publication contracts. Configuration describes data, never code."""

import copy
import hashlib
import ipaddress
import json
import re
from urllib.parse import urlsplit

from jsonschema import Draft202012Validator, FormatChecker, ValidationError

ID = re.compile(r"^[a-z][a-z0-9_-]{0,47}$")
WIRE = re.compile(r"^nv_connector_[a-z0-9_]{1,48}_v[1-9][0-9]{0,5}$")
HEADER = re.compile(r"^[A-Za-z][A-Za-z0-9-]{0,63}$")
SCOPE = {"runs:create", "runs:read", "runs:cancel", "operations:write"}


class ApplicationError(Exception):
    def __init__(self, code="invalid_request", status=422):
        self.code, self.status = code, status
        super().__init__(code)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def identifier(value):
    if not isinstance(value, str) or not ID.fullmatch(value):
        raise ApplicationError("invalid_id")
    return value


def fields(value, required, optional=()):
    if not isinstance(value, dict) or not set(required) <= set(value) or set(value) - set(required) - set(optional):
        raise ApplicationError()


def integer(value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ApplicationError()
    return value


def string(value, maximum=128, empty=False, multiline=False):
    if not isinstance(value, str) or not (0 if empty else 1) <= len(value) <= maximum or any(ord(c) < 32 and not (multiline and c in "\t\r\n") for c in value):
        raise ApplicationError()
    return value


def schema(value):
    """A bounded local schema subset also compatible with provider tool schemas."""
    if not isinstance(value, dict) or value.get("type") != "object" or value.get("additionalProperties") is not False:
        raise ApplicationError("invalid_schema")
    count = 0

    def walk(item, depth=0):
        nonlocal count
        count += 1
        if depth > 16 or count > 512:
            raise ApplicationError("invalid_schema")
        if isinstance(item, dict):
            if any(k in item for k in ("$ref", "$dynamicRef", "$id", "$schema", "pattern", "patternProperties")):
                raise ApplicationError("invalid_schema")
            for child in item.values():
                walk(child, depth + 1)
        elif isinstance(item, list):
            for child in item:
                walk(child, depth + 1)
    walk(value)
    if len(canonical(value).encode()) > 16384:
        raise ApplicationError("invalid_schema")
    try:
        Draft202012Validator.check_schema(value)
    except Exception:
        raise ApplicationError("invalid_schema") from None
    return value


def validate(value, contract):
    try:
        Draft202012Validator(contract, format_checker=FormatChecker()).validate(value)
        if len(canonical(value).encode()) > 65536:
            raise ValueError()
    except (ValidationError, ValueError, TypeError, RecursionError):
        raise ApplicationError("invalid_input") from None


def connector(value):
    fields(value, ("name", "origin", "secret_ref", "auth", "private_networks", "operations"))
    value = copy.deepcopy(value)
    string(value["name"], 128)
    url = urlsplit(string(value["origin"], 256))
    try:
        port = url.port or 443
    except ValueError:
        raise ApplicationError("invalid_origin") from None
    if url.scheme != "https" or not url.hostname or url.username or url.password or url.path not in ("", "/") or url.query or url.fragment or port not in (443, 8443):
        raise ApplicationError("invalid_origin")
    try:
        hostname = url.hostname.encode("idna").decode("ascii")
    except UnicodeError:
        raise ApplicationError("invalid_origin") from None
    if not re.fullmatch(r"[A-Za-z0-9.:-]{1,253}", hostname):
        raise ApplicationError("invalid_origin")
    value["origin"] = f"https://{'[' + hostname + ']' if ':' in hostname else hostname}" + (f":{port}" if port != 443 else "")
    identifier(value["secret_ref"])
    fields(value["auth"], ("type",), ("header",))
    if value["auth"]["type"] not in ("bearer", "api_key", "basic_api_key"):
        raise ApplicationError("invalid_auth")
    if value["auth"]["type"] == "api_key":
        header = value["auth"].get("header", "")
        if not HEADER.fullmatch(header) or header.lower() in ("host", "content-type", "content-length", "connection", "transfer-encoding", "cookie", "proxy-authorization", "accept-encoding", "idempotency-key"):
            raise ApplicationError("invalid_auth")
    networks = value["private_networks"]
    if not isinstance(networks, list) or len(networks) > 8:
        raise ApplicationError("invalid_networks")
    for net in networks:
        try:
            parsed = ipaddress.ip_network(net, strict=True)
        except (ValueError, TypeError):
            raise ApplicationError("invalid_networks") from None
        approved = tuple(ipaddress.ip_network(v) for v in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "fc00::/7"))
        if not any(parsed.version == net.version and parsed.subnet_of(net) for net in approved) or parsed.prefixlen < (16 if parsed.version == 4 else 48):
            raise ApplicationError("invalid_networks")
    operations = value["operations"]
    if not isinstance(operations, list) or not 1 <= len(operations) <= 20:
        raise ApplicationError("invalid_operations")
    seen = set()
    for op in operations:
        fields(op, ("id", "description", "method", "path", "input_schema", "output_schema", "query", "body", "projection", "read_only", "public_voice", "timeout_seconds", "identity_field"), ("idempotency_header", "reconcile", "array_projection", "coercions", "open_statuses"))
        identifier(op["id"])
        if op["id"] in seen:
            raise ApplicationError("duplicate_operation")
        seen.add(op["id"])
        string(op["description"], 512)
        if type(op["read_only"]) is not bool or type(op["public_voice"]) is not bool:
            raise ApplicationError()
        if op["method"] not in ("GET", "POST", "PUT", "PATCH") or (op["method"] == "GET" and not op["read_only"]) or (op["public_voice"] and not op["read_only"]):
            raise ApplicationError("invalid_effect_policy")
        path = string(op["path"], 512)
        if not path.startswith("/") or path.startswith("//") or any(c in path for c in ("?", "#", "\\", "%")) or any(p in (".", "..") for p in path.split("/")):
            raise ApplicationError("invalid_path")
        schema(op["input_schema"]); schema(op["output_schema"])
        properties = op["input_schema"].get("properties", {})
        if not isinstance(properties, dict) or not set(op["input_schema"].get("required", [])) <= properties.keys():
            raise ApplicationError("invalid_schema")
        for placeholder in re.findall(r"\{([^{}]+)\}", path):
            if placeholder not in properties or properties[placeholder].get("type") not in ("string", "integer"):
                raise ApplicationError("invalid_mapping")
        if re.sub(r"\{[^{}]+\}", "", path).count("{") or re.sub(r"\{[^{}]+\}", "", path).count("}"):
            raise ApplicationError("invalid_path")
        for key in ("query", "body", "projection"):
            mapping = op[key]
            if not isinstance(mapping, dict) or len(mapping) > 32:
                raise ApplicationError("invalid_mapping")
            for target, source in mapping.items():
                if not isinstance(target, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", target) or not isinstance(source, str):
                    raise ApplicationError("invalid_mapping")
                if key != "projection" and source not in properties:
                    raise ApplicationError("invalid_mapping")
                if key == "projection" and not re.fullmatch(r"(?:\$|\$count|[A-Za-z0-9_.]{1,128})", source):
                    raise ApplicationError("invalid_mapping")
        if "array_projection" in op:
            mapping = op["array_projection"]
            if not isinstance(mapping, dict) or len(mapping) > 8:
                raise ApplicationError("invalid_mapping")
            for target, columns in mapping.items():
                if target not in op["projection"] or not isinstance(columns, dict) or len(columns) > 32 or any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", k) or not isinstance(v, str) or not re.fullmatch(r"[A-Za-z0-9_.]{1,128}", v) for k, v in columns.items()):
                    raise ApplicationError("invalid_mapping")
        if "coercions" in op and (not isinstance(op["coercions"], dict) or any(k not in op["projection"] or v != "string" for k, v in op["coercions"].items())):
            raise ApplicationError("invalid_mapping")
        if "open_statuses" in op and (not isinstance(op["open_statuses"], list) or len(op["open_statuses"]) > 20 or any(type(v) is not int or v < 1 for v in op["open_statuses"])):
            raise ApplicationError("invalid_ticket_statuses")
        if op["method"] == "GET" and op["body"]:
            raise ApplicationError("invalid_mapping")
        integer(op["timeout_seconds"], 1, 30)
        identity = op["identity_field"]
        if identity is not None and (identity not in properties or identity not in op["input_schema"].get("required", [])):
            raise ApplicationError("invalid_identity")
        if not op["read_only"] and identity is None:
            raise ApplicationError("invalid_identity")
        if op.get("idempotency_header") is not None and op["idempotency_header"] != "Idempotency-Key":
            raise ApplicationError("invalid_idempotency_header")
        if op.get("reconcile") is not None:
            fields(op["reconcile"], ("operation", "argument", "result_field"))
            identifier(op["reconcile"]["operation"])
            string(op["reconcile"]["argument"], 64); string(op["reconcile"]["result_field"], 64)
    for op in operations:
        if op.get("reconcile"):
            spec = op["reconcile"]
            target = next((v for v in operations if v["id"] == spec["operation"]), None)
            if not target or not target["read_only"] or spec["argument"] not in target["input_schema"].get("properties", {}) or spec["result_field"] not in target["output_schema"].get("properties", {}):
                raise ApplicationError("invalid_reconciliation")
    return value


def references(value):
    if not isinstance(value, list) or len(value) > 20:
        raise ApplicationError("invalid_grants")
    seen = set()
    for item in value:
        fields(item, ("connector_id", "version", "operation_id"))
        identifier(item["connector_id"]); identifier(item["operation_id"])
        integer(item["version"], 1, 1000000)
        key = canonical(item)
        if key in seen:
            raise ApplicationError("invalid_grants")
        seen.add(key)
    return value


def preset(value):
    fields(value, ("name", "provider", "model", "secret_ref", "prompt", "operations", "deadline_seconds", "max_output_tokens", "result_retention_hours"))
    string(value["name"]); identifier(value["secret_ref"])
    if value["provider"] != "openai_responses":
        raise ApplicationError("unsupported_provider")
    if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,100}", value["model"]):
        raise ApplicationError("invalid_model")
    string(value["prompt"], 8192, empty=True, multiline=True); references(value["operations"])
    integer(value["deadline_seconds"], 1, 120)
    integer(value["max_output_tokens"], 256, 8192)
    integer(value["result_retention_hours"], 1, 24)
    return copy.deepcopy(value)


REQUEST_SCHEMA = {"type": "object", "additionalProperties": False,
    "properties": {"action": {"enum": ["lookup", "create_ticket"]},
        "customer_id": {"type": "string", "minLength": 1, "maxLength": 128},
        "summary": {"type": "string", "maxLength": 200},
        "description": {"type": "string", "maxLength": 4000}},
    "required": ["action", "customer_id"]}


def run_request(value):
    fields(value, ("preset_id", "version", "input"))
    if value["preset_id"] != "support-request":
        raise ApplicationError("unknown_preset", 404)
    integer(value["version"], 1, 1000000)
    validate(value["input"], REQUEST_SCHEMA)
    if value["input"]["action"] == "create_ticket" and (not value["input"].get("summary") or not value["input"].get("description")):
        raise ApplicationError("invalid_input")
    return value
