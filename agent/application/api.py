"""Separate machine and private administrator routers with bounded inputs."""

import hmac
import os
import re
import time

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from .contracts import (ApplicationError, SCOPE, connector, fields, identifier,
                        integer, preset, references, string)
from .http import bounded_json


async def body(request, maximum=32768):
    if request.headers.get("content-type", "").split(";")[0].lower() != "application/json":
        raise ApplicationError("invalid_media_type", 400)
    raw = bytearray()
    async for chunk in request.stream():
        raw.extend(chunk)
        if len(raw) > maximum:
            raise ApplicationError("request_too_large", 413)
    try:
        value = bounded_json(raw, maximum)
    except ApplicationError:
        raise ApplicationError("invalid_json", 400) from None
    if not isinstance(value, dict):
        raise ApplicationError("invalid_request", 400)
    return value


def reply(value=None, error=None, status=200):
    headers = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"}
    if error:
        status = error.status
        value = {"error": error.code}
        if status in (429, 503):
            headers["Retry-After"] = "2"
    return JSONResponse(value, status_code=status, headers=headers)


def create_application_routers(application):
    async def private(request: Request):
        expected = os.getenv("API_TOKEN", "")
        if not expected or not hmac.compare_digest(request.headers.get("authorization", "").encode(), ("Bearer " + expected).encode()):
            return None
        actor = request.headers.get("x-agents-actor", "")
        if not re.fullmatch(r"[A-Za-z0-9_.@-]{1,128}", actor):
            return None
        return actor

    admin = APIRouter(prefix="/api/agent/v1/application")
    public = APIRouter(prefix="/agents-api/v1")

    async def guarded(operation, actor):
        if actor is None:
            return reply(error=ApplicationError("unauthorized", 401))
        try:
            application.require_available()
            return reply(await operation())
        except ApplicationError as exc:
            return reply(error=exc)
        except Exception:
            return reply(error=ApplicationError("application_unavailable", 503))

    @admin.get("/inventory")
    async def inventory(actor=Depends(private)):
        async def read():
            return await application.db("inventory") | {"health": application.health()}
        return await guarded(read, actor)

    @admin.put("/settings")
    async def settings(request: Request, actor=Depends(private)):
        async def save():
            value = await body(request)
            fields(value, ("enabled",))
            if type(value["enabled"]) is not bool:
                raise ApplicationError()
            result = await application.db("set_enabled", value["enabled"], actor)
            application.enabled = value["enabled"]
            return result
        return await guarded(save, actor)

    @admin.put("/resources/{kind}/{resource_id}")
    async def draft(kind: str, resource_id: str, request: Request, actor=Depends(private)):
        async def save():
            identifier(resource_id)
            if kind not in ("connector", "preset") or (kind == "preset" and resource_id != "support-request"):
                raise ApplicationError("unknown_resource", 404)
            value = await body(request, 65536)
            fields(value, ("definition", "expected_revision"))
            integer(value["expected_revision"], 0, 1000000)
            definition = (connector if kind == "connector" else preset)(value["definition"])
            return await application.db("save", kind, resource_id, definition, value["expected_revision"], actor)
        return await guarded(save, actor)

    @admin.post("/resources/{kind}/{resource_id}/publish")
    async def publish(kind: str, resource_id: str, request: Request, actor=Depends(private)):
        async def save():
            identifier(resource_id)
            if kind not in ("connector", "preset"):
                raise ApplicationError("unknown_resource", 404)
            value = await body(request)
            fields(value, ("expected_revision",))
            integer(value["expected_revision"], 1, 1000000)
            return await application.db("publish", kind, resource_id, value["expected_revision"], actor, connector if kind == "connector" else preset)
        return await guarded(save, actor)

    @admin.delete("/versions/{kind}/{resource_id}/{version}")
    async def revoke(kind: str, resource_id: str, version: int, actor=Depends(private)):
        async def save():
            identifier(resource_id); integer(version, 1, 1000000)
            if kind not in ("connector", "preset"):
                raise ApplicationError("unknown_resource", 404)
            return await application.db("revoke_version", kind, resource_id, version, actor)
        return await guarded(save, actor)

    @admin.post("/secrets")
    async def secret(request: Request, actor=Depends(private)):
        async def save():
            value = await body(request)
            fields(value, ("secret_id", "value"))
            identifier(value["secret_id"]); string(value["value"], 4096)
            return await application.db("add_secret", value["secret_id"], value["value"], application.key, actor)
        return await guarded(save, actor)

    @admin.delete("/secrets/{secret_id}")
    async def revoke_secret(secret_id: str, actor=Depends(private)):
        async def save():
            identifier(secret_id)
            return await application.db("revoke_secret", secret_id, actor)
        return await guarded(save, actor)

    @admin.put("/grants/{agent_id}")
    async def grants(agent_id: str, request: Request, actor=Depends(private)):
        async def save():
            if agent_id not in ("internal", "external", "support-request"):
                raise ApplicationError("unknown_agent", 404)
            value = await body(request)
            fields(value, ("operations", "expected_revision"))
            references(value["operations"]); integer(value["expected_revision"], 0, 1000000)
            return await application.db("save_grants", agent_id, value["operations"], value["expected_revision"], actor)
        return await guarded(save, actor)

    @admin.post("/clients")
    async def client(request: Request, actor=Depends(private)):
        async def save():
            value = await body(request)
            fields(value, ("client_id", "scopes", "presets", "operations", "customer_ids", "expires"))
            identifier(value["client_id"])
            if not isinstance(value["scopes"], list) or not value["scopes"] or any(not isinstance(v, str) or v not in SCOPE for v in value["scopes"]):
                raise ApplicationError("invalid_scopes")
            if not isinstance(value["presets"], list) or not 1 <= len(value["presets"]) <= 100:
                raise ApplicationError("invalid_presets")
            for target in value["presets"]:
                identifier(target)
                if target != "support-request":
                    graph = (await application.runtime.workflows.db("workflow_active", target))["definition"]
                    if "api" not in graph["entrypoints"]:
                        raise ApplicationError("entrypoint_capability_mismatch")
            references(value["operations"])
            if not isinstance(value["customer_ids"], list) or not 1 <= len(value["customer_ids"]) <= 100:
                raise ApplicationError("invalid_customer_ids")
            for customer in value["customer_ids"]:
                string(customer)
            if type(value["expires"]) not in (float, int) or not time.time() < value["expires"] <= time.time() + 365 * 86400:
                raise ApplicationError("invalid_expiry")
            await application.db("resolve", value["operations"])
            definition = {k: value[k] for k in ("scopes", "presets", "operations", "customer_ids")}
            return await application.db("create_client", value["client_id"], definition, value["expires"], actor)
        return await guarded(save, actor)

    @admin.delete("/clients/{client_id}")
    async def revoke_client(client_id: str, actor=Depends(private)):
        async def save():
            identifier(client_id)
            return await application.db("revoke_client", client_id, actor)
        return await guarded(save, actor)

    @admin.post("/test-runs")
    async def test(request: Request, actor=Depends(private)):
        async def execute():
            value = await body(request)
            fields(value, ("client_id", "request", "idempotency_key", "confirm_write"))
            identifier(value["client_id"])
            if type(value["confirm_write"]) is not bool or (value["request"].get("input", {}).get("action") == "create_ticket" and not value["confirm_write"]):
                raise ApplicationError("write_confirmation_required", 403)
            client = await application.db("client", value["client_id"])
            result = await application.submit(client, value["request"], value["idempotency_key"])
            await application.db("record_audit", actor, "test_run", result["run_id"])
            application.runtime.events.emit("run.test", run_id=result["run_id"], actor=actor)
            return result
        return await guarded(execute, actor)

    @admin.get("/runs/{run_id}/result")
    async def admin_result(run_id: str, actor=Depends(private)):
        async def read():
            try:
                row = await application.db("run", run_id)
                client_id = row["client_id"]
            except ApplicationError as exc:
                if exc.code not in ("not_found", "run_not_found") or getattr(application.runtime, "workflows", None) is None:
                    raise
                client_id = (await application.runtime.workflows.db("execution", run_id))["principal"]
            client = {"client_id": client_id, "definition": {"scopes": ["runs:read"]}}
            return await application.result(client, run_id)
        return await guarded(read, actor)

    @admin.post("/runs/{run_id}/cancel")
    async def admin_cancel(run_id: str, actor=Depends(private)):
        async def cancel():
            try:
                row = await application.db("run", run_id)
                client_id = row["client_id"]
            except ApplicationError as exc:
                if exc.code not in ("not_found", "run_not_found") or getattr(application.runtime, "workflows", None) is None:
                    raise
                client_id = (await application.runtime.workflows.db("execution", run_id))["principal"]
            client = {"client_id": client_id, "definition": {"scopes": ["runs:cancel"]}}
            result = await application.cancel(client, run_id)
            await application.db("record_audit", actor, "run_cancel_requested", run_id)
            return result
        return await guarded(cancel, actor)

    @admin.post("/effects/{operation_id}/reconcile")
    async def reconcile(operation_id: str, actor=Depends(private)):
        return await guarded(lambda: application.reconcile(operation_id, actor), actor)

    async def machine(request, operation, status=200):
        try:
            if request.query_params:
                raise ApplicationError("invalid_query", 400)
            client = await application.authenticate(request.headers.get("authorization", ""))
            return reply(await operation(client), status=status)
        except ApplicationError as exc:
            return reply(error=exc)
        except Exception:
            return reply(error=ApplicationError("application_unavailable", 503))

    @public.post("/runs")
    async def submit(request: Request):
        async def execute(client):
            return await application.submit(client, await body(request), request.headers.get("idempotency-key"))
        return await machine(request, execute, 202)

    @public.get("/runs/{run_id}")
    async def run(run_id: str, request: Request):
        return await machine(request, lambda client: application.run(client, run_id))

    @public.get("/runs/{run_id}/result")
    async def result(run_id: str, request: Request):
        return await machine(request, lambda client: application.result(client, run_id))

    @public.post("/runs/{run_id}/cancel")
    async def cancel(run_id: str, request: Request):
        return await machine(request, lambda client: application.cancel(client, run_id))

    return admin, public
