"""Private administrator workflow routes; credentials never enter definitions."""

import base64
import hmac
import os
import re

from fastapi import APIRouter, Depends, Request

from agent.application.api import body, reply
from agent.application.contracts import ApplicationError, canonical, fields, identifier, integer
from .contracts import BLOCKS, DefinitionError, definition
from .data import MAX_UPLOAD, settings
from .templates import templates
from .connectors import presets


def create_workflow_router(service):
    router = APIRouter(prefix="/api/agent/v1/application/workflows")

    async def private(request: Request):
        expected = os.getenv("API_TOKEN", "")
        actor = request.headers.get("x-agents-actor", "")
        if not expected or not hmac.compare_digest(request.headers.get("authorization", "").encode(), ("Bearer " + expected).encode()) or not re.fullmatch(r"[A-Za-z0-9_.@-]{1,128}", actor):
            return None
        return actor

    async def guarded(actor, operation):
        if actor is None:
            return reply(error=ApplicationError("unauthorized", 401))
        try:
            service.require_available()
            return reply(await operation())
        except DefinitionError as exc:
            return reply({"error": exc.code, "node_id": exc.node}, status=422)
        except ApplicationError as exc:
            return reply(error=exc)
        except Exception:
            return reply(error=ApplicationError("workflow_request_failed", 503))

    @router.get("/inventory")
    async def inventory(actor=Depends(private)):
        return await guarded(actor, service.inventory)

    @router.get("/catalog")
    async def catalog(actor=Depends(private)):
        async def get():
            return {"blocks": BLOCKS, "templates": templates(), "connector_presets": presets()}
        return await guarded(actor, get)

    @router.get("/definitions/{kind}/{agent_id}/versions/{version}")
    async def version(kind: str, agent_id: str, version: int, actor=Depends(private)):
        async def get():
            identifier(agent_id); integer(version, 1, 1000000)
            if kind not in ("agent", "subflow"):
                raise ApplicationError("invalid_resource")
            return {"definition": await service.db("workflow_version", kind, agent_id, version)}
        return await guarded(actor, get)

    @router.put("/definitions/{kind}/{agent_id}")
    async def save(kind: str, agent_id: str, request: Request, actor=Depends(private)):
        async def update():
            identifier(agent_id)
            value = await body(request, 131072)
            fields(value, ("definition", "expected_revision"))
            integer(value["expected_revision"], 0, 1000000)
            draft = value["definition"]
            # An incomplete draft can be saved. Publication is strictly validated.
            if not isinstance(draft, dict) or draft.get("agent_id") != agent_id or not isinstance(draft.get("name"), str) or len(draft["name"]) > 128 or len(canonical(draft).encode()) > 131072:
                raise ApplicationError("invalid_draft")
            if not isinstance(draft.get("nodes"), list) or len(draft["nodes"]) > 100 or not isinstance(draft.get("edges"), list) or len(draft["edges"]) > 200 or not isinstance(draft.get("entrypoints"), list) or not draft['entrypoints'] or any(item not in ('voice', 'api') for item in draft['entrypoints']):
                raise ApplicationError("invalid_draft")
            for node in draft['nodes']:
                if not isinstance(node, dict) or not all(isinstance(node.get(key), str) for key in ('id','name','type')) or not isinstance(node.get('config'), dict) or not isinstance(node.get('inputs'), dict):
                    raise ApplicationError('invalid_draft')
            return await service.db("workflow_save", kind, agent_id, draft, value["expected_revision"], actor)
        return await guarded(actor, update)

    @router.post("/validate")
    async def validate_graph(request: Request, actor=Depends(private)):
        async def execute():
            value = await body(request, 131072); fields(value, ("definition",))
            graph = definition(value["definition"])
            await service.db("workflow_validate", graph)
            return {"valid": True, "tools": service.tool_preview(graph)}
        return await guarded(actor, execute)

    @router.post("/definitions/{kind}/{agent_id}/publish")
    async def publish(kind: str, agent_id: str, request: Request, actor=Depends(private)):
        async def update():
            identifier(agent_id)
            value = await body(request); fields(value, ("expected_revision",))
            integer(value["expected_revision"], 1, 1000000)
            return await service.db("workflow_publish", kind, agent_id, value["expected_revision"], actor)
        return await guarded(actor, update)

    @router.post("/definitions/{kind}/{agent_id}/activate")
    async def activate(kind: str, agent_id: str, request: Request, actor=Depends(private)):
        async def update():
            identifier(agent_id)
            value = await body(request); fields(value, ("version", "enabled", "expected_revision"))
            integer(value["version"], 1, 1000000); integer(value["expected_revision"], 1, 1000000)
            if type(value["enabled"]) is not bool:
                raise ApplicationError()
            return await service.db("workflow_activate", kind, agent_id, value["version"], value["enabled"], value["expected_revision"], actor)
        return await guarded(actor, update)

    @router.post("/test")
    async def test(request: Request, actor=Depends(private)):
        async def execute():
            value = await body(request, 262144)
            fields(value, ("definition", "fixtures", "input", "caller"), ("tables", "destinations"))
            graph = definition(value["definition"])
            if not all(isinstance(value[k], dict) for k in ("fixtures", "input", "caller")):
                raise ApplicationError("invalid_test_input")
            for key, fixture in value["fixtures"].items():
                if not isinstance(fixture, dict) or not isinstance(fixture.get("outcome"), str):
                    raise ApplicationError("invalid_fixture")
            result = await service.test(graph, value["fixtures"], value["input"], value["caller"], value.get("tables"), value.get("destinations"))
            return result | {"test_mode": "mock"}
        return await guarded(actor, execute)

    @router.get("/runs/{run_id}")
    async def run(run_id: str, actor=Depends(private)):
        async def get():
            row = await service.db("execution", run_id)
            result = {k: v for k, v in row.items() if k not in ("result", "idempotency_hash", "input_digest")}
            result["result_available"] = row["result"] is not None
            return result
        return await guarded(actor, get)

    @router.post("/runs/{run_id}/cancel")
    async def cancel(run_id: str, actor=Depends(private)):
        async def update():
            await service.db("execution_cancel", run_id)
            for call in service.runtime.calls.values():
                if call.run_id == run_id:
                    call.cancellation.set()
                    await service.runtime._finish(call, "workflow_cancelled", fallback=True)
                    break
            if run_id in service.active:
                service.active[run_id]["cancellation"].set()
            return {"cancel_requested": True}
        return await guarded(actor, update)

    @router.put("/data/{resource_id}")
    async def save_data(resource_id: str, request: Request, actor=Depends(private)):
        async def update():
            identifier(resource_id)
            value = await body(request); fields(value, ("settings", "expected_revision"))
            integer(value["expected_revision"], 0, 1000000)
            return await service.db("data_save", resource_id, settings(value["settings"]), value["expected_revision"], actor)
        return await guarded(actor, update)

    @router.post("/data/preview")
    async def preview(request: Request, actor=Depends(private)):
        async def execute():
            value = await body(request, 14 * 1024**2); fields(value, ("settings",), ("content",))
            cfg = settings(value["settings"])
            if cfg['format'] in ('google_sheets', 'google_csv'):
                return await service.preview_data(cfg, None)
            try:
                raw = base64.b64decode(value["content"], validate=True)
            except Exception:
                raise ApplicationError("invalid_upload") from None
            if len(raw) > MAX_UPLOAD:
                raise ApplicationError("upload_too_large", 413)
            return await service.preview_data(cfg, raw)
        return await guarded(actor, execute)

    @router.post("/data/{resource_id}/publish")
    async def publish_data(resource_id: str, request: Request, actor=Depends(private)):
        async def execute():
            identifier(resource_id)
            value = await body(request, 14 * 1024**2); fields(value, ("expected_revision", "content"))
            integer(value["expected_revision"], 1, 1000000)
            try:
                raw = base64.b64decode(value["content"], validate=True)
            except Exception:
                raise ApplicationError("invalid_upload") from None
            if len(raw) > MAX_UPLOAD:
                raise ApplicationError("upload_too_large", 413)
            return await service.enqueue_data(resource_id, value["expected_revision"], raw, actor)
        return await guarded(actor, execute)

    @router.get("/jobs/{job_id}")
    async def job(job_id: str, actor=Depends(private)):
        async def get():
            if not re.fullmatch(r"[a-f0-9]{32}", job_id):
                raise ApplicationError("invalid_id")
            return await service.db("ingestion_job", job_id)
        return await guarded(actor, get)

    @router.post("/data/{resource_id}/refresh")
    async def refresh(resource_id: str, request: Request, actor=Depends(private)):
        async def execute():
            identifier(resource_id)
            value = await body(request); fields(value, ("expected_revision",))
            integer(value["expected_revision"], 1, 1000000)
            return await service.enqueue_data(resource_id, value["expected_revision"], None, actor)
        return await guarded(actor, execute)

    @router.delete("/data/{resource_id}/versions/{version}")
    async def revoke(resource_id: str, version: int, actor=Depends(private)):
        async def execute():
            identifier(resource_id); integer(version, 1, 1000000)
            return await service.db("data_revoke", resource_id, version, actor)
        return await guarded(actor, execute)

    return router
