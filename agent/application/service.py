"""Shared connector execution and bounded non-voice admission/lifecycle."""

import asyncio
import base64
import copy
import hashlib
import json
import time
import uuid
from collections import OrderedDict

from agent.context import get
from agent.models import AgentUnavailable, PermissionDenied, ToolManifest
from .contracts import ApplicationError, canonical, connector, digest, preset, run_request, validate
from .crypto import ContentKey
from .http import HttpTransport, RemoteError, mapped_request, project
from .repository import ApplicationRepository, TERMINAL
from .responses import OpenAIResponses


class ConnectorFailure(AgentUnavailable):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def wire_name(ref):
    key = f"{ref['connector_id']}:{ref['operation_id']}".encode()
    return f"nv_connector_{hashlib.sha256(key).hexdigest()[:20]}_v{ref['version']}"


def reference_key(ref):
    return canonical(ref)


def connector_tool_id(ref):
    return f"connector.{ref['connector_id']}.{ref['operation_id']}.v{ref['version']}"


def credential_headers(cfg, credential):
    if cfg["auth"]["type"] == "bearer":
        return {"Authorization": "Bearer " + credential}
    if cfg["auth"]["type"] == "basic_api_key":
        return {"Authorization": "Basic " + base64.b64encode((credential + ":X").encode()).decode()}
    return {cfg["auth"]["header"]: credential}


class Application:
    def __init__(self, runtime, repository=None, key=None, transport=None, adapter=None):
        self.runtime = runtime
        self.repository = repository or ApplicationRepository()
        self.key = key or ContentKey()
        self.transport = transport or HttpTransport()
        self.adapter = adapter or OpenAIResponses(self.transport)
        self.epoch = uuid.uuid4().hex
        self.active = {}
        self._run_slots = asyncio.Semaphore(4)
        self._io_slots = {"voice": asyncio.Semaphore(4), "api": asyncio.Semaphore(4)}
        self._connector_slots = {}
        self._submit_lock = asyncio.Lock()
        self._db_slots = asyncio.Semaphore(8)
        self._rate = OrderedDict()
        self._maintenance_task = None
        self._stopping = False
        self.available = False
        self.error = "starting"
        self.enabled = False

    async def db(self, method, *args):
        async with self._db_slots:
            try:
                return await asyncio.to_thread(getattr(self.repository, method), *args)
            except ApplicationError:
                raise
            except Exception:
                raise ApplicationError("application_store_unavailable", 503) from None

    async def start(self):
        self._stopping = False
        if self._maintenance_task is None:
            self._maintenance_task = asyncio.create_task(self._maintenance())

    async def _maintenance(self):
        next_purge = 0
        while not self._stopping:
            try:
                if not self.available:
                    async with self._submit_lock:
                        await self.db("initialize", self.epoch)
                        await self.db("recover_orphans", self.epoch, list(self.active))
                self.enabled = await self.db("enabled")
                if time.monotonic() >= next_purge:
                    await self.db("purge")
                    next_purge = time.monotonic() + 60
                self.available, self.error = True, None
            except ApplicationError as exc:
                self.available, self.error = False, exc.code
            await asyncio.sleep(2)

    async def stop(self):
        self._stopping = True
        if self._maintenance_task:
            self._maintenance_task.cancel()
            await asyncio.gather(self._maintenance_task, return_exceptions=True)
            self._maintenance_task = None
        tasks = []
        for item in list(self.active.values()):
            item["cancellation"].set()
            item["task"].cancel()
            tasks.append(item["task"])
        await asyncio.gather(*tasks, return_exceptions=True)

    def health(self):
        return {"available": self.available, "enabled": self.enabled,
                "error_code": self.error, "content_key_available": self.key.key is not None,
                "active_runs": len(self.active)}

    def require_available(self):
        if not self.available or self._stopping:
            raise ApplicationError(self.error or "application_unavailable", 503)
        if self.key.key is None:
            raise ApplicationError("content_key_unavailable", 503)

    async def authenticate(self, bearer):
        self.require_available()
        if not isinstance(bearer, str) or not bearer.startswith("Bearer nv_agent_") or len(bearer) > 256:
            raise ApplicationError("unauthorized", 401)
        return await self.db("authenticate", bearer[7:])

    @staticmethod
    def scope(client, scope):
        if scope not in client["definition"]["scopes"]:
            raise ApplicationError("forbidden", 403)

    def limit_rate(self, client_id):
        now = time.monotonic()
        stamp, tokens = self._rate.get(client_id, (now, 5.0))
        tokens = min(5.0, tokens + (now - stamp) * .5)
        if tokens < 1:
            raise ApplicationError("rate_limited", 429)
        self._rate[client_id] = (now, tokens - 1)
        self._rate.move_to_end(client_id)
        while len(self._rate) > 128:
            self._rate.popitem(last=False)

    async def submit(self, client, request, idempotency):
        if isinstance(request, dict) and "agent_id" in request:
            return await self.runtime.workflows.submit_api(client, request, idempotency)
        self.require_available()
        self.scope(client, "runs:create")
        request = run_request(request)
        if not isinstance(idempotency, str) or not 1 <= len(idempotency) <= 128 or any(ord(c) < 33 or ord(c) > 126 for c in idempotency):
            raise ApplicationError("invalid_idempotency_key", 400)
        client_id = client["client_id"]
        if request["preset_id"] not in client["definition"]["presets"] or request["input"]["customer_id"] not in client["definition"]["customer_ids"]:
            raise ApplicationError("forbidden", 403)
        if request["input"]["action"] == "create_ticket":
            self.scope(client, "operations:write")
        async with self._submit_lock:
            self.limit_rate(client_id)
            if not await self.db("enabled"):
                raise ApplicationError("access_disabled", 503)
            definition = await self.db("version", "preset", request["preset_id"], request["version"])
            grants = await self.db("grants", request["preset_id"])
            permitted = {reference_key(r) for r in grants["operations"]} & {reference_key(r) for r in client["definition"]["operations"]}
            refs = [ref for ref in definition["operations"] if reference_key(ref) in permitted]
            resolved = await self.db("resolve", refs)
            if request["input"]["action"] == "lookup":
                resolved = [item for item in resolved if item["operation"]["read_only"]]
            if not resolved or (request["input"]["action"] == "create_ticket" and not any(not item["operation"]["read_only"] for item in resolved)):
                raise ApplicationError("operation_denied", 403)
            # The seeded preset's one write has exactly these server-approved fields.
            for item in resolved:
                op = item["operation"]
                if not op["read_only"]:
                    expected = {"customer_id": request["input"]["customer_id"], "summary": request["input"]["summary"], "description": request["input"]["description"]}
                    if op["identity_field"] != "customer_id":
                        raise ApplicationError("preset_operation_incompatible", 409)
                    validate(expected, op["input_schema"])
            snapshot = {"preset": definition, "connectors": resolved, "grant_revision": grants["revision"]}
            metadata = {"execution_kind": "api", "agent_id": "support-request", "provider": "openai",
                "definition_revision": request["version"], "grant_revision": grants["revision"],
                "connector_versions": refs, "native_revision": self.runtime.store.revision,
                "native_payload_hash": self.runtime.store.payload_hash}
            row, fresh = await self.db("admit", client_id, idempotency, request, snapshot, self.epoch,
                                       self.key, self.runtime.store.revision, metadata)
            if fresh:
                cancellation = asyncio.Event()
                self.register_monitoring(row)
                self.runtime.events.emit("run.admitted", run_id=row["run_id"], agent_id="support-request", execution_kind="api",
                    client_id=client_id, definition_revision=request["version"], grant_revision=grants["revision"])
                task = asyncio.create_task(self._execute(row, cancellation))
                self.active[row["run_id"]] = {"task": task, "cancellation": cancellation, "row": row}
        return self.public_run(row)

    def public_run(self, row):
        return {key: row[key] for key in ("run_id", "status", "started", "ended", "error_code", "cancel_requested", "reconciliation_required", "preset_version")} | {
            "status_url": "/agents-api/v1/runs/" + row["run_id"], "versions": row["metadata"],
            "result_available": row["result"] is not None and (row["result_expires"] or 0) > time.time()}

    def register_monitoring(self, row):
        self.runtime.monitoring.enqueue("run", {"run_id": row["run_id"], "session_id": None,
            "execution_kind": "api", "agent_id": "support-request", "provider": "openai",
            "cdr_id": None, "destination_id": None, "epoch": self.runtime.monitoring.epoch,
            "revision": row["revision"], "payload_hash": row["metadata"].get("native_payload_hash"),
            "started": row["started"], "capture_version": 0, "transcript_state": "not_applicable",
            "definition_revision": row["preset_version"], "client_id": row["client_id"],
            "connector_versions": row["metadata"]["connector_versions"]})

    async def _execute(self, admitted, cancellation):
        run_id = admitted["run_id"]; watcher = None
        status, error, result = "failed", "execution_failed", None
        try:
            async with self._run_slots:
                row = await self.db("claim", run_id, self.epoch)
                if row is None:
                    return
                request = self.key.decrypt(f"input:{run_id}", row["input"])
                snapshot = self.key.decrypt(f"snapshot:{run_id}", row["snapshot"])
                definition = snapshot["preset"]
                context = {"run_id": run_id, "agent_id": "support-request", "execution_kind": "api",
                    "principal": row["client_id"], "definition_revision": row["preset_version"],
                    "profile": {"tools": {}}, "permissions": {}, "capabilities": ["http"],
                    "deadline_monotonic": time.monotonic() + max(0, definition["deadline_seconds"] - (time.time() - row["started"])),
                    "cancellation": cancellation, "connector_tools": snapshot["connectors"],
                    "authorized_input": request, "grant_revision": snapshot["grant_revision"], "voice": None}
                async def check():
                    if cancellation.is_set():
                        raise ApplicationError("cancelled", 503)
                    current = await self.db("run", run_id)
                    if current["cancel_requested"]:
                        cancellation.set()
                        raise ApplicationError("cancelled", 503)
                    await self.db("client", row["client_id"])
                    if not await self.db("enabled"):
                        raise ApplicationError("access_disabled", 503)
                    await self.db("version", "preset", "support-request", row["preset_version"])
                    await self.db("secret", definition["secret_ref"], self.key)
                    if time.monotonic() >= context["deadline_monotonic"]:
                        raise ApplicationError("execution_timeout", 503)
                async def watch():
                    while not cancellation.is_set():
                        await asyncio.sleep(1)
                        try:
                            await check()
                        except ApplicationError:
                            cancellation.set()
                watcher = asyncio.create_task(watch())
                await check()
                credential = await self.db("secret", definition["secret_ref"], self.key)
                self.runtime.events.emit("run.started", run_id=run_id, execution_kind="api")
                execution = asyncio.create_task(self.adapter.execute(definition, request, context,
                    self.runtime.tools.provider_tools(context), self.runtime.tools.dispatch, credential, check))
                cancel_wait = asyncio.create_task(cancellation.wait())
                try:
                    done, _ = await asyncio.wait({execution, cancel_wait}, timeout=max(0, context["deadline_monotonic"] - time.monotonic()), return_when=asyncio.FIRST_COMPLETED)
                    if execution not in done:
                        execution.cancel()
                        await asyncio.gather(execution, return_exceptions=True)
                        raise ApplicationError("cancelled" if cancellation.is_set() else "execution_timeout", 503)
                    result = await execution
                finally:
                    cancel_wait.cancel()
                    if not execution.done():
                        execution.cancel()
                    await asyncio.gather(cancel_wait, execution, return_exceptions=True)
                status, error = "completed", None
                effects = await self.db("effects", run_id)
                if request["action"] == "create_ticket" and not any(item["state"] == "committed" for item in effects):
                    raise ApplicationError("business_action_incomplete", 503)
                if request["action"] == "lookup" and not any(item.get("ok") for item in result.get("operations", [])):
                    raise ApplicationError("business_action_incomplete", 503)
                if len(canonical(result).encode()) > 65536:
                    result = None
                    raise ApplicationError("result_too_large", 503)
        except asyncio.CancelledError:
            status, error = "interrupted", "runtime_shutdown"
        except ApplicationError as exc:
            status, error = ("cancelled" if exc.code == "cancelled" else "failed"), exc.code
        except Exception:
            status, error = "failed", "execution_failed"
        finally:
            if watcher:
                watcher.cancel()
                await asyncio.gather(watcher, return_exceptions=True)
            try:
                await self.db("finish", run_id, status, error, result, self.key)
                settled = await self.db("run", run_id)
                status, error = settled["status"], settled["error_code"]
            except ApplicationError:
                status, error = "interrupted", "settlement_unavailable"
                self.available, self.error = False, "application_store_unavailable"
            self.runtime.events.emit("run.ended", run_id=run_id, execution_kind="api", outcome=status, reason_code=error or "completed")
            self.active.pop(run_id, None)

    async def run(self, client, run_id):
        self.scope(client, "runs:read")
        try:
            row = await self.db("run", run_id, client["client_id"])
        except ApplicationError as exc:
            if exc.code not in ("not_found", "run_not_found") or getattr(self.runtime, "workflows", None) is None:
                raise
            return await self.runtime.workflows.api_run(client, run_id)
        return self.public_run(row) | {"effects": await self.db("effects", run_id)}

    async def result(self, client, run_id):
        self.scope(client, "runs:read")
        try:
            row = await self.db("run", run_id, client["client_id"])
        except ApplicationError as exc:
            if exc.code not in ("not_found", "run_not_found") or getattr(self.runtime, "workflows", None) is None:
                raise
            return await self.runtime.workflows.api_run(client, run_id, result=True)
        if row["status"] not in TERMINAL:
            return {"state": "pending"}
        if row["result"] is None or (row["result_expires"] or 0) <= time.time():
            return {"state": "expired" if row["result_expires"] and row["result_expires"] <= time.time() else "unavailable"}
        return {"state": "available", "status": row["status"], "reconciliation_required": row["reconciliation_required"],
                "result": self.key.decrypt(f"result:{run_id}", row["result"])}

    async def cancel(self, client, run_id):
        self.scope(client, "runs:cancel")
        try:
            await self.db("run", run_id, client["client_id"])
        except ApplicationError as exc:
            if exc.code not in ("not_found", "run_not_found") or getattr(self.runtime, "workflows", None) is None:
                raise
            return await self.runtime.workflows.api_run(client, run_id, cancel=True)
        result = await self.db("cancel", run_id, client["client_id"])
        if result["cancel_requested"] and run_id in self.active:
            self.active[run_id]["cancellation"].set()
        return result

    async def voice_bindings(self, agent_id, origin):
        # Optional connectors must never prevent a native call from starting.
        if not self.available or self.key.key is None:
            return []
        try:
            if not await self.db("enabled"):
                return []
            grants = await self.db("grants", agent_id)
            resolved = await self.db("resolve", grants["operations"])
            if origin != "internal" and agent_id != "external":
                external = await self.db("grants", "external")
                ceiling = {reference_key(r) for r in external["operations"]}
                resolved = [item for item in resolved if reference_key(item["reference"]) in ceiling]
            # Voice has no verified business identity in the initial slice.
            return [item for item in resolved if item["operation"]["read_only"] and item["operation"]["public_voice"] and item["operation"]["identity_field"] is None]
        except ApplicationError:
            return []

    def manifests(self, context):
        for item in get(context, "connector_tools", []):
            op, ref = item["operation"], item["reference"]
            name = wire_name(ref)
            yield ToolManifest(connector_tool_id(ref), name, str(ref["version"]),
                op["description"], op["input_schema"], op["output_schema"], op["timeout_seconds"], op["read_only"])

    async def authorize(self, item, args, context):
        self.require_available()
        if get(context, "_workflow"):
            return await self.runtime.workflows.authorize_connector(item, args, context)
        ref = item["reference"]; op = item["operation"]
        if not await self.db("enabled"):
            raise ApplicationError("access_disabled", 403)
        await self.db("version", "connector", ref["connector_id"], ref["version"])
        grants = await self.db("grants", get(context, "agent_id"))
        if reference_key(ref) not in {reference_key(r) for r in grants["operations"]}:
            raise ApplicationError("operation_denied", 403)
        if get(context, "execution_kind") != "api":
            if not op["read_only"] or not op["public_voice"] or op["identity_field"] is not None:
                raise ApplicationError("operation_denied", 403)
            if get(context, "origin") != "internal" and get(context, "agent_id") != "external":
                external = await self.db("grants", "external")
                if reference_key(ref) not in {reference_key(r) for r in external["operations"]}:
                    raise ApplicationError("operation_denied", 403)
        else:
            client = await self.db("client", get(context, "principal"))
            if reference_key(ref) not in {reference_key(r) for r in client["definition"]["operations"]}:
                raise ApplicationError("operation_denied", 403)
            request = get(context, "authorized_input")
            if request["customer_id"] not in client["definition"]["customer_ids"]:
                raise ApplicationError("identity_denied", 403)
            identity = op["identity_field"]
            if identity is None and not op["public_voice"]:
                raise ApplicationError("identity_policy_missing", 403)
            if identity and args.get(identity) != request["customer_id"]:
                raise ApplicationError("identity_denied", 403)
            if not op["read_only"]:
                if request["action"] != "create_ticket" or "operations:write" not in client["definition"]["scopes"]:
                    raise ApplicationError("operation_denied", 403)
                expected = {k: request[k] for k in ("customer_id", "summary", "description")}
                if args != expected:
                    raise ApplicationError("write_confirmation_mismatch", 403)

    async def invoke(self, tool_id, args, context):
        item = next((i for i in get(context, "connector_tools", []) if connector_tool_id(i["reference"]) == tool_id), None)
        if item is None:
            raise PermissionDenied("operation_denied")
        try:
            await self.authorize(item, args, context)
            return await self._invoke(item, args, context)
        except ApplicationError as exc:
            raise ConnectorFailure(exc.code) from None

    async def _invoke(self, item, args, context):
        op, cfg, ref = item["operation"], item["connector"], item["reference"]
        validate(args, op["input_schema"])
        path, query, body = mapped_request(op, args)
        timeout = min(op["timeout_seconds"], get(context, "deadline_monotonic", time.monotonic() + 30) - time.monotonic())
        if timeout <= 0:
            raise ApplicationError("execution_timeout", 503)
        kind = "api" if get(context, "execution_kind") == "api" else "voice"
        # Reserve one connector slot for each entrypoint so API work cannot
        # consume the voice connector budget.
        slots = self._connector_slots.setdefault((ref["connector_id"], kind), asyncio.Semaphore(1))
        async with asyncio.timeout(timeout):
            async with self._io_slots[kind], slots:
                await self.authorize(item, args, context)
                credential = await self.db("secret", cfg["secret_ref"], self.key)
                headers = {"Content-Type": "application/json"} | credential_headers(cfg, credential)
                effect = None
                if not op["read_only"]:
                    effect, fresh = await self.db("prepare_effect", get(context, "run_id"), get(context, "effect_key", "create_ticket"), args, ref)
                    if not fresh:
                        if effect["state"] == "committed" and effect["response"] is not None and effect["result_expires"] > time.time():
                            return self.key.decrypt(f"effect:{effect['operation_id']}", effect["response"])
                        raise ApplicationError("effect_" + effect["state"], 409)
                    if op.get("idempotency_header"):
                        headers[op["idempotency_header"]] = effect["operation_id"]
                    await self.db("dispatch_effect", effect["operation_id"])
                    self.runtime.events.emit("effect.dispatched", run_id=get(context, "run_id"), operation_id=effect["operation_id"], tool_id=op["id"], effect_state="dispatched")
                try:
                    attempts = 2 if op["read_only"] else 1
                    for attempt in range(attempts):
                        remaining = min(timeout, get(context, "deadline_monotonic") - time.monotonic())
                        try:
                            response = await self.transport.json_request(cfg["origin"], path, op["method"], headers, query, body,
                                cfg["private_networks"], remaining)
                            break
                        except RemoteError as exc:
                            if attempt + 1 == attempts or exc.http_status not in (None, 429, 500, 502, 503, 504):
                                raise
                            self.runtime.events.emit("tool.retry", run_id=get(context, "run_id"), tool_id=op["id"], retry_count=attempt + 1)
                            await self.authorize(item, args, context)
                    result = project(response, op)
                    if effect:
                        await self.db("settle_effect", effect["operation_id"], "committed", result, self.key)
                        self.runtime.events.emit("effect.committed", run_id=get(context, "run_id"), operation_id=effect["operation_id"], effect_state="committed")
                    return result
                except BaseException as exc:
                    if effect:
                        # Only explicit validation/authentication failures from the remote
                        # service are treated as rejection; uncertain outcomes stay unknown.
                        rejected = isinstance(exc, RemoteError) and exc.http_status in (400, 401, 403, 404, 405, 422)
                        code = exc.code if isinstance(exc, ApplicationError) else "remote_uncertain"
                        try:
                            await asyncio.shield(self.db("settle_effect", effect["operation_id"], "rejected" if rejected else "unknown", None, self.key, code))
                        except ApplicationError:
                            pass
                        self.runtime.events.emit("effect.unknown" if not rejected else "effect.rejected", run_id=get(context, "run_id"), operation_id=effect["operation_id"], effect_state="rejected" if rejected else "unknown")
                    raise

    async def reconcile(self, operation_id, actor):
        self.require_available()
        effect = await self.db("effect", operation_id)
        await self.db("record_audit", actor, "effect_reconciliation", operation_id)
        if effect["state"] != "unknown":
            return {"operation_id": operation_id, "state": effect["state"]}
        ref = {"connector_id": effect["connector_id"], "version": effect["version"], "operation_id": effect["operation"]}
        item = (await self.db("resolve", [ref]))[0]
        spec = item["operation"].get("reconcile")
        if not spec:
            return {"operation_id": operation_id, "state": "unknown", "operator_action_required": True}
        target = next(o for o in item["connector"]["operations"] if o["id"] == spec["operation"])
        args = {spec["argument"]: operation_id}
        validate(args, target["input_schema"])
        path, query, body = mapped_request(target, args)
        cfg = item["connector"]
        credential = await self.db("secret", cfg["secret_ref"], self.key)
        headers = credential_headers(cfg, credential)
        slots = self._connector_slots.setdefault((ref["connector_id"], "api"), asyncio.Semaphore(1))
        async with asyncio.timeout(target["timeout_seconds"]):
            async with self._io_slots["api"], slots:
                response = await self.transport.json_request(cfg["origin"], path, "GET", headers, query, None, cfg["private_networks"], target["timeout_seconds"])
        result = project(response, target)
        if result.get(spec["result_field"]):
            await self.db("settle_effect", operation_id, "committed", result, self.key)
            self.runtime.events.emit("effect.reconciled", run_id=effect["run_id"], operation_id=operation_id, effect_state="committed")
            return {"operation_id": operation_id, "state": "committed"}
        return {"operation_id": operation_id, "state": "unknown", "operator_action_required": True}
