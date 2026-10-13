"""Bounded private administrator endpoints; mock endpoints never dispatch effects."""

import base64
import json
import re
import time

from fastapi import Depends, Request

from agent.application.api import body
from agent.application.contracts import ApplicationError, fields, identifier, integer, string, validate
from .contracts import definition
from .sources import preview_row, query, search
from .tool_contracts import tool, tool_key, reference

RUN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def register_designer_routes(router, service, private, guarded):
    @router.get("/tools")
    async def tools(actor=Depends(private)):
        return await guarded(actor, lambda: service.db("tool_inventory"))

    @router.put("/tools/{tool_id}")
    async def save_tool(tool_id: str, request: Request, actor=Depends(private)):
        async def update():
            identifier(tool_id)
            value = await body(request, 131072); fields(value, ("definition", "expected_revision"))
            integer(value["expected_revision"], 0, 1000000)
            return await service.db("tool_save", tool_id, value["definition"], value["expected_revision"], actor)
        return await guarded(actor, update)

    @router.get("/tools/{tool_id}/versions/{version}")
    async def tool_version(tool_id: str, version: int, actor=Depends(private)):
        async def get():
            identifier(tool_id); integer(version, 1, 1000000)
            return {"definition": await service.db("tool_version", tool_id, version)}
        return await guarded(actor, get)

    @router.post("/tools/{tool_id}/publish")
    async def publish_tool(tool_id: str, request: Request, actor=Depends(private)):
        async def update():
            identifier(tool_id)
            value = await body(request); fields(value, ("expected_revision",))
            integer(value["expected_revision"], 1, 1000000)
            return await service.db("tool_publish", tool_id, value["expected_revision"], actor)
        return await guarded(actor, update)

    @router.delete("/tools/{tool_id}/versions/{version}")
    async def revoke_tool(tool_id: str, version: int, actor=Depends(private)):
        async def update():
            identifier(tool_id); integer(version, 1, 1000000)
            return await service.db("tool_revoke", tool_id, version, actor)
        return await guarded(actor, update)

    @router.post("/tools/test")
    async def test_tool(request: Request, actor=Depends(private)):
        async def execute():
            value = await body(request, 262144)
            fields(value, ("definition", "arguments"), ("fixture", "rows", "columns", "identity", "scope"))
            draft = tool(value["definition"]); validate(value["arguments"], draft["input_schema"])
            rows = value.get("rows", [])
            if not isinstance(rows, list) or len(rows) > 1000 or any(not isinstance(r, dict) for r in rows):
                raise ApplicationError("invalid_fixture")
            if draft["kind"] == "query":
                result = query([{"source_row": index + 1} | row for index, row in enumerate(rows)], draft["config"], value["arguments"], value.get("identity", {}), value.get("scope", {}), value.get("columns", []))
                result["source"] = draft["config"]["resource"]
            elif draft["kind"] == "documents":
                if any(not isinstance(r.get("text", ""), str) for r in rows):
                    raise ApplicationError("invalid_fixture")
                result = search(rows, value["arguments"].get("query", ""), draft["config"]["limit"])
                result["source"] = draft["config"]["resource"]
                for passage in result["passages"]:
                    passage["version"] = draft["config"]["resource"]["version"]
            else:
                if "fixture" not in value:
                    raise ApplicationError("tool_fixture_required")
                result = value["fixture"]
            validate(result, draft["output_schema"])
            return {"test_mode": "mock", "result": result}
        return await guarded(actor, execute)

    @router.get("/dependencies")
    async def dependencies(actor=Depends(private)):
        async def get(): return {"dependencies": await service.db("designer_dependencies")}
        return await guarded(actor, get)

    @router.post("/data/inspect")
    async def inspect_source(request: Request, actor=Depends(private)):
        async def inspect():
            from .data import settings, REQUIRED, MAX_UPLOAD
            value = await body(request, 14 * 1024**2)
            fields(value, ('settings',), ('content',))
            if not isinstance(value['settings'], dict): raise ApplicationError('invalid_source')
            cfg = dict(value['settings'])
            # Inspection precedes mapping. Validate the rest of the complete source configuration.
            cfg['kind'] = 'payment'; cfg['mapping'] = {name: name for name in REQUIRED}; cfg.pop('columns', None)
            cfg = settings(cfg)
            if cfg['format'] not in ('csv', 'xlsx', 'google_sheets', 'google_csv'): raise ApplicationError('unsupported_data_format')
            raw = None
            if cfg['format'] in ('csv', 'xlsx'):
                try: raw = base64.b64decode(value.get('content', ''), validate=True)
                except (ValueError, TypeError): raise ApplicationError('invalid_upload') from None
                if not raw or len(raw) > MAX_UPLOAD: raise ApplicationError('upload_too_large', 413)
            return await service.inspect_data(cfg, raw)
        return await guarded(actor, inspect)

    @router.get("/data/{resource_id}/versions/{version}/preview")
    async def source_preview(resource_id: str, version: int, actor=Depends(private)):
        async def get():
            identifier(resource_id); integer(version, 1, 1000000)
            table = await service.db("data_rows", resource_id, version, service.application.key)
            resource = await service.db("data_settings", resource_id)
            cfg = resource["settings"] | {"columns": table["metadata"].get("columns", []) + [c for c in resource["settings"].get("columns", []) if c.get("sensitive")]}
            return {"rows": [preview_row(r, cfg) for r in table["rows"][:50]],
                    "metadata": table["metadata"], "created": table["created"], "version": version}
        return await guarded(actor, get)

    @router.post("/documents/{resource_id}/publish")
    async def publish_documents(resource_id: str, request: Request, actor=Depends(private)):
        async def update():
            identifier(resource_id)
            value = await body(request, 14 * 1024**2); fields(value, ("expected_revision", "files"), ("remove",))
            integer(value["expected_revision"], 1, 1000000)
            resource = await service.db("data_settings", resource_id)
            if resource["settings"].get("kind") != "documents":
                raise ApplicationError("source_kind_mismatch")
            if not isinstance(value["files"], list) or not isinstance(value.get("remove", []), list):
                raise ApplicationError("invalid_documents")
            old = await service.db("data_original", resource_id, resource["published_version"], service.application.key) if resource["published_version"] else None
            existing = json.loads(base64.b64decode(old))["files"] if old else []
            incoming = {}
            for f in value["files"]:
                if not isinstance(f, dict): raise ApplicationError("invalid_documents")
                identifier(f.get("document_id"))
                if f["document_id"] in incoming: raise ApplicationError("duplicate_document")
                incoming[f["document_id"]] = f
            for name in value.get("remove", []): identifier(name)
            files = [f for f in existing if f["document_id"] not in incoming and f["document_id"] not in value.get("remove", [])] + list(incoming.values())
            raw = json.dumps({"files": files}, ensure_ascii=False).encode()
            if len(raw) > 14 * 1024**2:
                raise ApplicationError("upload_too_large", 413)
            return await service.enqueue_data(resource_id, value["expected_revision"], raw, actor)
        return await guarded(actor, update)

    @router.post("/simulations")
    async def simulation(request: Request, actor=Depends(private)):
        async def execute():
            value = await body(request, 262144)
            fields(value, ("definition", "fixtures", "input", "caller"), ("tables", "destinations", "subflows", "messages"))
            graph = definition(value["definition"])
            for key in ("tables", "subflows"):
                if not isinstance(value.get(key, {}), dict) or len(value.get(key, {})) > 20: raise ApplicationError("invalid_fixture")
            if not isinstance(value.get("destinations", []), list) or len(value.get("destinations", [])) > 100: raise ApplicationError("invalid_fixture")
            if not all(isinstance(value[k], dict) for k in ("fixtures", "input", "caller")):
                raise ApplicationError("invalid_fixture")
            messages = value.get("messages", [])
            if not isinstance(messages, list) or len(messages) > 50:
                raise ApplicationError("invalid_fixture")
            for message in messages:
                fields(message, ("role", "text")); string(message["text"], 8192, multiline=True)
                if message["role"] not in ("caller", "agent"):
                    raise ApplicationError("invalid_fixture")
            return await service.designer.simulation(graph, value["fixtures"], value["input"], value["caller"], messages, actor,
                value.get("tables"), value.get("destinations"), value.get("subflows"))
        return await guarded(actor, execute)

    @router.get("/simulations/{run_id}")
    async def simulation_detail(run_id: str, actor=Depends(private)):
        async def get():
            if not RUN.fullmatch(run_id): raise ApplicationError("invalid_id")
            return await service.db("simulation_detail", run_id, service.application.key, actor)
        return await guarded(actor, get)

    @router.get("/runs/{run_id}/tool-calls")
    async def calls(run_id: str, actor=Depends(private)):
        async def get():
            if not RUN.fullmatch(run_id): raise ApplicationError("invalid_id")
            return {"calls": await service.db("invocation_records", run_id)}
        return await guarded(actor, get)

    @router.get("/runs/{run_id}/tool-calls/{invocation_id}/content")
    async def call_content(run_id: str, invocation_id: str, actor=Depends(private)):
        async def get():
            if not RUN.fullmatch(run_id) or not RUN.fullmatch(invocation_id): raise ApplicationError("invalid_id")
            return await service.db("invocation_content", run_id, invocation_id, service.application.key, actor)
        return await guarded(actor, get)

    @router.post("/activity")
    async def activity(request: Request, actor=Depends(private)):
        async def get():
            value = await body(request)
            fields(value, (), ("since", "until", "agent_id", "kind", "status", "before", "limit"))
            now = time.time(); since = value.get("since", now - 86400); until = value.get("until", now)
            if type(since) not in (int, float) or type(until) not in (int, float) or not 0 <= since <= until <= now + 60:
                raise ApplicationError("invalid_query")
            limit = integer(value.get("limit", 50), 1, 100)
            for name in ("agent_id", "status", "before"):
                string(value.get(name, ""), 128, empty=True)
            if value.get("kind", "") not in ("", "voice", "api", "mock"):
                raise ApplicationError("invalid_query")
            history_available = True
            try:
                await service.runtime.monitoring.read("policy")
            except Exception:
                history_available = False
            result = await service.db("designer_activity", since, until, value.get("agent_id", ""), value.get("kind", ""), value.get("status", ""), value.get("before", ""), limit, history_available)
            return result | {"history_available": history_available}
        return await guarded(actor, get)

    @router.post("/connections/test")
    async def test_connection(request: Request, actor=Depends(private)):
        async def execute():
            from agent.application.http import mapped_request, project
            from agent.application.service import credential_headers
            value = await body(request); fields(value, ("operation", "arguments"))
            item = (await service.application.db("resolve", [value["operation"]]))[0]
            op, cfg = item["operation"], item["connector"]
            if not op["read_only"]:
                raise ApplicationError("connection_test_requires_read_operation")
            validate(value["arguments"], op["input_schema"])
            credential = await service.application.db("secret", cfg["secret_ref"], service.application.key)
            path, query_args, payload = mapped_request(op, value["arguments"])
            response = await service.application.transport.json_request(cfg["origin"], path, op["method"],
                {"Content-Type": "application/json"} | credential_headers(cfg, credential), query_args, payload, cfg["private_networks"], op["timeout_seconds"])
            validate(project(response, op), op["output_schema"])
            return {"status": "ready"}
        return await guarded(actor, execute)

    @router.put("/credentials/{secret_id}")
    async def replace_secret(secret_id: str, request: Request, actor=Depends(private)):
        async def update():
            identifier(secret_id)
            value = await body(request, 32768); fields(value, ("value",))
            string(value["value"], 16384, multiline=True)
            return await service.db("designer_replace_secret", secret_id, value["value"], service.application.key, actor)
        return await guarded(actor, update)
