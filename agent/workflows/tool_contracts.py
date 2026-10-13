"""Immutable reusable capabilities and agent-scoped bindings for definition schema 2."""

import copy

from agent.application.contracts import ApplicationError, digest, canonical, fields, identifier, integer, schema, string
from .sources import query_spec


def tool_key(reference):
    return f"tool.{reference['tool_id']}.v{reference['version']}"


def tool_wire(reference):
    return "nv_tool_" + digest(reference)[:32]


def reference(value):
    fields(value, ("tool_id", "version"))
    identifier(value["tool_id"]); integer(value["version"], 1, 1000000)
    return value


def binding(value):
    fields(value, ("tool_id", "version", "conversation", "workflow", "scope", "instructions"), ("origins",))
    reference({k: value[k] for k in ("tool_id", "version")})
    if type(value["conversation"]) is not bool or type(value["workflow"]) is not bool:
        raise ApplicationError("invalid_tool_binding")
    string(value["instructions"], 2048, empty=True, multiline=True)
    fields(value["scope"], (), ("identity_field", "filters"))
    if value["scope"].get("identity_field"):
        identifier(value["scope"]["identity_field"])
    filters = value["scope"].get("filters", {})
    if not isinstance(filters, dict) or len(filters) > 16 or len(canonical(filters)) > 4096:
        raise ApplicationError("invalid_tool_scope")
    for name, expected in filters.items():
        identifier(name)
        if expected is not None and type(expected) not in (str, bool, int, float):
            raise ApplicationError("invalid_tool_scope")
    origins = value.get("origins", ["internal", "external"])
    if not isinstance(origins, list) or not origins or not set(origins) <= {"internal", "external"}:
        raise ApplicationError("invalid_tool_scope")
    return value


def tool(value):
    fields(value, ("name", "description", "kind", "input_schema", "output_schema", "config", "execution_policy", "timeout_seconds"))
    string(value["name"], 128); string(value["description"], 1024, empty=True, multiline=True)
    schema(value["input_schema"])
    # Native result maps intentionally permit dynamic, permission-filtered fields.
    # Their exact exported schemas are checked below; user-defined schemas stay strict.
    schema(value["output_schema"] | {"additionalProperties": False} if value["kind"] == "telephony" else value["output_schema"])
    integer(value["timeout_seconds"], 1, 30)
    if value["execution_policy"] not in ("read_only", "direct"):
        raise ApplicationError("invalid_tool_policy")
    cfg = value["config"]
    if value["kind"] == "api":
        fields(cfg, ("operation",))
        fields(cfg["operation"], ("connector_id", "version", "operation_id"))
        identifier(cfg["operation"]["connector_id"]); identifier(cfg["operation"]["operation_id"])
        integer(cfg["operation"]["version"], 1, 1000000)
    elif value["kind"] == "query":
        query_spec(cfg)
    elif value["kind"] == "documents":
        fields(cfg, ("resource", "limit"), ("max_age_seconds",))
        fields(cfg["resource"], ("resource_id", "version"))
        identifier(cfg["resource"]["resource_id"])
        if cfg["resource"]["version"] != "latest": integer(cfg["resource"]["version"], 1, 1000000)
        integer(cfg["limit"], 1, 10)
        integer(cfg.get("max_age_seconds", 31536000), 60, 31536000)
    elif value["kind"] == "telephony":
        fields(cfg, ("tool_id",))
        from agent.tools.registry import MANIFESTS
        manifest = next((m for m in MANIFESTS if m.id == cfg["tool_id"]), None)
        if manifest is None or value["input_schema"] != manifest.input_schema or value["output_schema"] != manifest.output_schema:
            raise ApplicationError("invalid_builtin_tool")
        if (value["execution_policy"] == "read_only") != manifest.read_only:
            raise ApplicationError("invalid_tool_policy")
    elif value["kind"] == "subflow":
        fields(cfg, ("resource",)); fields(cfg["resource"], ("resource_id", "version"))
        identifier(cfg["resource"]["resource_id"]); integer(cfg["resource"]["version"], 1, 1000000)
    else:
        raise ApplicationError("invalid_tool_kind")
    if value["kind"] in ("query", "documents") and value["execution_policy"] != "read_only":
        raise ApplicationError("invalid_tool_policy")
    if len(canonical(value).encode()) > 65536:
        raise ApplicationError("tool_capacity")
    return copy.deepcopy(value)
