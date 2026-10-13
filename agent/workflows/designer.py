"""Designer capabilities use published server-owned references for every invocation."""

import asyncio
import copy
import time
import uuid

from agent.application.contracts import ApplicationError, digest, validate
from agent.application.service import connector_tool_id, reference_key
from agent.context import origin
from agent.models import ToolManifest
from .sources import query, search, matches
from .tool_contracts import tool_key, tool_wire


class Designer:
    def __init__(self, workflows):
        self.workflows = workflows

    @staticmethod
    def enabled(graph, node, context, mode="conversation"):
        if graph.get("schema_version") != 2:
            return []
        selected = set(node.get("config", {}).get("tools", []))
        inherit = node.get("config", {}).get("tool_mode", "selected") == "inherit"
        result = []
        for b in graph.get("tool_bindings", []):
            if not b[mode] or (context.get("execution_kind") != "api" and origin(context) not in b.get("origins", ["internal", "external"])):
                continue
            if mode == "conversation" and not inherit and tool_key(b) not in selected:
                continue
            result.append(b)
        return result

    async def resolve(self, graph, node, context, mode="conversation"):
        result = []
        for binding in self.enabled(graph, node, context, mode):
            value = await self.workflows.db("tool_version", binding["tool_id"], binding["version"])
            row = {"binding": binding, "definition": value}
            if value["kind"] == "subflow":
                ref = value["config"]["resource"]
                row["subflow"] = await self.workflows.db("subflow_version", ref["resource_id"], ref["version"])
                if not set(graph["entrypoints"]) <= set(row["subflow"]["entrypoints"]):
                    raise ApplicationError("subflow_capability_mismatch")
            result.append(row)
        return result

    @staticmethod
    def manifests(context):
        if not isinstance(context, dict):
            return
        for row in context.get("registry_tools", []):
            value, binding = row["definition"], row["binding"]
            timeout = row["subflow"]["limits"]["max_duration_seconds"] if "subflow" in row else value["timeout_seconds"]
            yield ToolManifest(tool_key(binding), tool_wire(binding), str(binding["version"]),
                               (value["name"] + "\n" + value["description"] + "\n" + binding["instructions"]).strip(),
                               value["input_schema"], value["output_schema"], timeout,
                               value["execution_policy"] == "read_only", "voice" if value["kind"] == "telephony" and value["config"]["tool_id"] == "telephony.handoff" else None)

    async def invoke(self, tool_id, args, context):
        row = next((r for r in context.get("registry_tools", []) if tool_key(r["binding"]) == tool_id), None)
        if row is None:
            raise ApplicationError("operation_denied", 403)
        graph = context["workflow_graph"]
        node = context["workflow_node"]
        mode = context.get("registry_mode", "conversation")
        if tool_id not in {tool_key(b) for b in self.enabled(graph, node, context, mode)}:
            raise ApplicationError("operation_denied", 403)
        b, value = row["binding"], row["definition"]
        await self.workflows.check(graph, context)
        await self.workflows.db("tool_version", b["tool_id"], b["version"])
        validate(args, value["input_schema"])
        scope = b["scope"]
        identity = context.get("identity", {})
        if value["kind"] != "query":
            field = scope.get("identity_field")
            if field and (not identity.get("verified") or identity.get(field) is None or args.get(field) != identity[field]):
                raise ApplicationError("identity_denied", 403)
            if any(not matches(args.get(field), "eq", expected) for field, expected in scope.get("filters", {}).items()):
                raise ApplicationError("caller_scope_mismatch", 403)
        invocation = context.get("_tool_invocation_id")
        if not invocation:
            raise ApplicationError("invocation_required")
        fresh = await self.workflows.db("invocation_start", context, invocation, {"tool_id": b["tool_id"], "version": b["version"]}, args, mode)
        if not fresh:
            raise ApplicationError("invocation_already_recorded", 409)
        started = time.monotonic()
        result, error = None, None
        try:
            timeout = row["subflow"]["limits"]["max_duration_seconds"] if "subflow" in row else value["timeout_seconds"]
            async with asyncio.timeout(min(timeout, context.get("deadline_monotonic", float("inf")) - time.monotonic())):
                result = await self.execute(row, args, context)
            validate(result, value["output_schema"])
            return result
        except BaseException as exc:
            error = exc.code if isinstance(exc, ApplicationError) else "tool_execution_failed"
            raise
        finally:
            # Metadata has no input/result content. Captured content is opt-in and encrypted.
            content = {"arguments": args, "result": result} if graph.get("capture_tool_content", False) and not error else None
            await asyncio.shield(self.workflows.db("invocation_finish", context["run_id"], invocation,
                int((time.monotonic() - started) * 1000), error, content, self.workflows.application.key))

    async def execute(self, row, args, context):
        value, b = row["definition"], row["binding"]
        cfg = value["config"]
        if value["kind"] == "api":
            item = (await self.workflows.application.db("resolve", [cfg["operation"]]))[0]
            connector_context = context | {"_registry_binding": row, "connector_tools": [item],
                                          "effect_key": "tool-" + digest([context["run_id"], context["_tool_invocation_id"]])}
            result = await self.workflows.application.invoke(connector_tool_id(cfg["operation"]), args, connector_context)
            self.workflows.accept_connector_result(context, item, result)
            return result
        if value["kind"] in ("query", "documents"):
            table = await self.workflows.table(context, cfg["resource"])
            max_age = cfg.get("max_age_seconds", 31536000 if value["kind"] == "documents" else 86400)
            if time.time() - table.get("freshness_at", table["created"]) > max_age:
                raise ApplicationError("data_stale", 409)
            pinned = ({"resource_id": cfg["resource"]["resource_id"], "version": context["latest_data"][cfg["resource"]["resource_id"]]["version"]} if cfg["resource"]["version"] == "latest" else cfg["resource"])
            if value["kind"] == "query":
                columns = table["metadata"].get("columns")
                if not columns:
                    from .data import CANONICAL_FIELDS
                    columns = [{"name": n, "sensitive": n == "verification_code"} for n in CANONICAL_FIELDS]
                if table["metadata"].get("kind", "payment") == "payment" and (
                        not context.get("identity", {}).get("verified") or context["identity"].get("resource_id") != cfg["resource"]["resource_id"]):
                    raise ApplicationError("identity_denied", 403)
                scope = b["scope"]
                if table["metadata"].get("kind", "payment") == "payment":
                    scope = scope | {"identity_field": "resident_id"}
                result = await asyncio.to_thread(query, table["rows"], cfg, args, context.get("identity", {}), scope, columns)
            else:
                result = await asyncio.to_thread(search, table["rows"], args.get("query", ""), cfg["limit"])
                for passage in result["passages"]:
                    passage["version"] = pinned["version"]
            return result | {"source": pinned}
        if value["kind"] == "telephony":
            from agent.tools.registry import MANIFESTS
            manifest = next(m for m in MANIFESTS if m.id == cfg["tool_id"])
            graph = copy.deepcopy(context["workflow_graph"])
            graph["tool_grants"] = list(set(graph["tool_grants"]) | {manifest.id})
            return await self.workflows.runtime.tools._invoke(manifest.id, args, context | {"workflow_graph": graph})
        if value["kind"] == "subflow":
            child = row["subflow"]
            if not set(child["tool_grants"]) <= set(context["workflow_graph"]["tool_grants"]):
                raise ApplicationError("subflow_scope_mismatch", 403)
            parent_path = context.get("subflow_path", [])
            child_context = context | {"subflow_path": parent_path + ["tool_" + b["tool_id"]], "agent_id": child["agent_id"], "child_traces": [], "_registry_subflow_binding": row,
                "permissions": {k: "allow" if v == "allow" and child["permissions"].get(k) == "allow" else "deny" for k, v in context.get("permissions", {}).items()}}
            try:
                result = await self.workflows.engine.execute(child, child_context, args, depth=len(parent_path) + 1)
            finally:
                context["step_count"] = child_context.get("step_count", context.get("step_count", 0))
            context.setdefault("child_traces", []).extend(result["trace"])
            if result["status"] != "completed":
                raise ApplicationError("subflow_fallback")
            return result["result"]
        raise ApplicationError("invalid_tool_kind")

    async def node(self, graph, node, args, context):
        scoped = context | {"_workflow": True, "workflow_graph": graph, "workflow_node": node, "registry_mode": "workflow"}
        scoped["registry_tools"] = await self.resolve(graph, node, scoped, "workflow")
        scoped["_tool_invocation_id"] = "step-" + digest([context["run_id"], context.get("subflow_path", []), context["step_count"], node["id"]])
        return await self.invoke(tool_key(node["config"]["tool"]), args, scoped)

    async def simulation(self, graph, fixtures, inputs, caller, messages, actor, tables=None, destinations=None, subflows=None):
        # Validate scripted invocations against the same published schemas and phase selection.
        # Only database metadata and synthetic tables are read here; no dispatcher is called.
        mock_calls = []
        nodes = {n['id']: n for n in graph['nodes']}
        for node_id, fixture in fixtures.items():
            if node_id not in nodes or not isinstance(fixture, dict) or not isinstance(fixture.get('outcome'), str):
                raise ApplicationError('invalid_fixture')
            calls = fixture.get('tool_calls', [])
            if not isinstance(calls, list) or len(calls) > 10: raise ApplicationError('invalid_fixture')
            node = nodes[node_id]
            context = {'execution_kind': 'voice' if graph['entrypoints'] == ['voice'] else 'api', 'caller': caller}
            mode = 'workflow' if node['type'] == 'tool.invoke' else 'conversation'
            enabled = await self.resolve(graph, node, context, mode) if calls or node['type'] == 'tool.invoke' else []
            if node['type'] == 'tool.invoke':
                bound = next((r for r in enabled if tool_key(r['binding']) == tool_key(node['config']['tool'])), None)
                if not bound: raise ApplicationError('operation_denied', 403)
                validate(fixture.get('output', {}), bound['definition']['output_schema'])
            for call in calls:
                from agent.application.contracts import fields
                fields(call, ('tool', 'arguments', 'result'), ('identity',))
                if node['type'] not in ('conversation.collect', 'conversation.decision'): raise ApplicationError('invalid_fixture')
                from .tool_contracts import reference
                reference(call['tool'])
                row = next((r for r in enabled if tool_key(r['binding']) == tool_key(call['tool'])), None)
                if not row: raise ApplicationError('operation_denied', 403)
                value, scope = row['definition'], row['binding']['scope']
                validate(call['arguments'], value['input_schema']); validate(call['result'], value['output_schema'])
                identity = call.get('identity', {})
                if not isinstance(identity, dict): raise ApplicationError('invalid_fixture')
                if value['kind'] != 'query':
                    identity_field = scope.get('identity_field')
                    if identity_field and (not identity.get('verified') or call['arguments'].get(identity_field) != identity.get(identity_field) or identity.get(identity_field) is None):
                        raise ApplicationError('identity_denied', 403)
                    if any(not matches(call['arguments'].get(k), 'eq', v) for k, v in scope.get('filters', {}).items()):
                        raise ApplicationError('caller_scope_mismatch', 403)
                mock_calls.append({'node_id': node_id, 'tool_id': row['binding']['tool_id'], 'version': row['binding']['version'],
                                   'mode': mode, 'state': 'completed', 'duration_ms': 0, 'arguments': call['arguments'], 'result': call['result']})
        result = await self.workflows.test(graph, fixtures, inputs, caller, tables, destinations, subflows)
        visited = {step['node_id'] for step in result['trace']}
        result['tool_calls'] = [c for c in mock_calls if c['node_id'] in visited]
        return await self.workflows.db('simulation_save', uuid.uuid4().hex, graph, result, messages, self.workflows.application.key, actor)
