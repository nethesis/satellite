"""Trusted policy and resource access helpers for reusable tools."""

from collections.abc import Mapping

from .models import PermissionDenied


def get(context, key, default=None):
    return context.get(key, default) if isinstance(context, Mapping) else getattr(context, key, default)


def _profile(context):
    return get(context, "profile") or {}


def origin(context):
    value = get(context, "origin")
    return "internal" if isinstance(value, str) and value.lower() == "internal" else "external"


def _ceiling(context, field):
    if origin(context) == "internal" or get(context, "agent_id") == "external":
        return None
    specific = get(context, f"external_{field}")
    if specific is not None:
        return specific
    external = get(context, "external_profile")
    return external.get(field) if isinstance(external, Mapping) else {}


def allowed(context, scope: str) -> bool:
    permissions = get(context, "permissions") or _profile(context).get("permissions", {})
    if permissions.get(scope) != "allow":
        return False
    ceiling = _ceiling(context, "permissions")
    return ceiling is None or ceiling.get(scope) == "allow"


def require(context, scope: str) -> None:
    if not allowed(context, scope):
        raise PermissionDenied("scope denied")


def tool_enabled(context, tool_id: str) -> bool:
    tools = _profile(context).get("tools", {})
    if tools.get(tool_id) != "enabled":
        return False
    ceiling = _ceiling(context, "tools")
    return ceiling is None or ceiling.get(tool_id) == "enabled"


def visible(context, resource: dict) -> bool:
    key = ("internal_allowed" if origin(context) == "internal"
           and get(context, "agent_id") == "internal" else "external_allowed")
    if not resource.get(key, False):
        return False
    scope = {"extension": "directory.extensions", "queue": "directory.queues",
             "ivr": "directory.ivrs"}.get(resource.get("type"))
    return bool(scope and allowed(context, scope))


def directory(context) -> list[dict]:
    value = get(context, "directory") or []
    return value if isinstance(value, list) else []


def find_resource(context, destination_id: str) -> dict | None:
    return next((r for r in directory(context) if r.get("id") == destination_id), None)
