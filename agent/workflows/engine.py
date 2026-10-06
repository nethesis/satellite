"""Sequential, bounded graph execution; all capabilities supplied by services."""

import asyncio
import copy
import hmac
import time

from agent.application.contracts import ApplicationError, canonical, digest, validate
from .contracts import CATALOG_BY_TYPE, definition, outcomes, output_contract
from .data import normalized_name, normalized_phone, matching_name, safe_row


def select(value, path):
    for key in path.split(".") if path else []:
        if isinstance(value, dict) and key in value:
            value = value[key]
        elif isinstance(value, list) and key.isdigit() and int(key) < len(value):
            value = value[int(key)]
        else:
            raise ApplicationError("output_field_unavailable")
    return value


def bind(bindings, outputs):
    result = {}
    for key, spec in bindings.items():
        if "value" in spec:
            result[key] = copy.deepcopy(spec["value"])
        else:
            try:
                result[key] = copy.deepcopy(select(outputs[spec["node"]], spec["path"]))
            except (KeyError, ApplicationError):
                if spec.get("optional"):
                    result[key] = copy.deepcopy(spec.get("default"))
                else:
                    raise ApplicationError("input_unavailable") from None
    return result


def render(text, inputs):
    import re
    def replace(match):
        value = select(inputs, match.group(1))
        return value if isinstance(value, str) else canonical(value)
    return re.sub(r"\{\{([A-Za-z0-9_.-]+)\}\}", replace, text)


class WorkflowEngine:
    def __init__(self, service):
        self.service = service

    async def execute(self, graph, context, inputs=None, *, depth=0):
        graph = definition(graph)
        if depth > 4:
            raise ApplicationError("subflow_depth_exceeded")
        validate(inputs or {}, graph["input_schema"])
        if depth == 0 and not context.get("test_mode") and hasattr(self.service, "prepare_data"):
            await self.service.prepare_data(graph, context)
        nodes = {node["id"]: node for node in graph["nodes"]}
        edges = {(edge["source"], edge["outcome"]): edge["target"] for edge in graph["edges"]}
        current = next(node["id"] for node in graph["nodes"] if node["type"].startswith("start."))
        outputs = {}; trace = []; result = {}; status = "completed"
        context.setdefault("child_traces", [])
        context.setdefault("step_count", 0)
        context.setdefault("global_max_steps", graph["limits"]["max_steps"])
        initial_steps = context["step_count"]
        deadline = min(context.get("deadline_monotonic", float("inf")), time.monotonic() + graph["limits"]["max_duration_seconds"])
        context["deadline_monotonic"] = deadline
        while current:
            node = nodes[current]
            context["step_count"] += 1
            if context["step_count"] - initial_steps > graph["limits"]["max_steps"] or context["step_count"] > context["global_max_steps"]:
                raise ApplicationError("step_budget_exceeded")
            cancellation = context.get("cancellation")
            if cancellation and cancellation.is_set():
                raise ApplicationError("cancelled", 409)
            if time.monotonic() >= deadline:
                raise ApplicationError("execution_timeout", 503)
            await self.service.check(graph, context)
            sequence = context["step_count"]; started = time.monotonic(); error = None
            await self.service.step(context, sequence, node, "running")
            try:
                node_inputs = bind(node["inputs"], outputs)
                remaining = deadline - time.monotonic()
                if node["type"] == "subflow":
                    # The nested engine applies the child graph deadline. Do not
                    # wrap a whole reusable graph in a single-operation timeout.
                    step_timeout = remaining
                elif node["type"] == "voice.consult":
                    step_timeout = min(remaining, node["config"].get("ring_seconds", 30) + node["config"].get("consult_seconds", 30) + 60)
                else:
                    step_timeout = min(remaining, 120 if node["type"].startswith(("conversation.", "voice.")) or node["type"] == "action.confirm" else node["config"].get("timeout_seconds", 10))
                outcome, value = await asyncio.wait_for(self.run_node(node, node_inputs, graph, context, inputs or {}, depth),
                    timeout=step_timeout)
                if len(canonical(value).encode()) > graph["limits"]["max_context_bytes"]:
                    raise ApplicationError("step_output_too_large")
                outputs[current] = value
                if len(canonical(outputs).encode()) > graph["limits"]["max_context_bytes"]:
                    raise ApplicationError("context_budget_exceeded")
            except asyncio.TimeoutError:
                outcome, value, error = "timeout", {}, "step_timeout"
            except ApplicationError as exc:
                outcome, value, error = "error", {}, exc.code
            except Exception:
                outcome, value, error = "error", {}, "block_failed"
            elapsed = int((time.monotonic() - started) * 1000)
            await self.service.step(context, sequence, node, "failed" if error else "completed", outcome, elapsed, error)
            trace.append({"node_id": current, "subflow_path": '/'.join(context.get('subflow_path', [])), "block_type": node["type"], "outcome": outcome, "duration_ms": elapsed, "error_code": error})
            if context.get("test_mode"):
                trace[-1]["output"] = copy.deepcopy(value)
            trace.extend(context["child_traces"])
            context["child_traces"].clear()
            if node["type"] == "end":
                if error:
                    raise ApplicationError(error, 503)
                result, status = value, node["config"].get("status", "completed")
                break
            if outcome in ("handed_off", "accepted") and node["type"] in ("pbx.route", "agent.route", "voice.consult"):
                result, status = value, "handed_off"
                break
            target = edges.get((current, outcome))
            if not target:
                if error:
                    raise ApplicationError(error, 503)
                raise ApplicationError("unhandled_outcome", 503)
            current = target
        if status == "completed":
            validate(result, graph["output_schema"])
        return {"status": status, "result": result, "trace": trace}

    async def run_node(self, node, args, graph, context, inputs, depth):
        kind, cfg = node["type"], node["config"]
        validate(args, CATALOG_BY_TYPE[kind]['input_schema'])
        fixture_id = "/".join(context.get("subflow_path", []) + [node["id"]])
        if context.get("test_mode") and fixture_id in context.get("fixtures", {}):
            fixture = context["fixtures"][fixture_id]
            if fixture['outcome'] not in outcomes(node):
                raise ApplicationError('invalid_fixture_outcome')
            validate(fixture.get('output', {}), output_contract(node))
            return fixture["outcome"], fixture.get("output", {})
        if kind == "start.call":
            return "success", {key: str(context.get("caller", {}).get(key) or "") for key in ("phone", "name", "did", "origin")}
        if kind == "start.api":
            return "success", inputs
        if kind in ("end", "logic.map"):
            return "success", args
        if kind == 'logic.merge':
            alternatives = [value for value in args.values() if value is not None]
            if len(alternatives) != 1 or not isinstance(alternatives[0], dict):
                raise ApplicationError('ambiguous_merge_input')
            return 'success', alternatives[0]
        if kind == "logic.condition":
            try:
                value = select(args, cfg["field"])
            except ApplicationError:
                value = None
            expected = cfg.get("value"); operator = cfg["operator"]
            if operator == "contains" and (isinstance(value, str) and not isinstance(expected, str) or
                    isinstance(value, dict) and isinstance(expected, (list, dict))):
                raise ApplicationError("invalid_condition_operand")
            matched = (value == expected if operator == "eq" else value != expected if operator == "ne" else
                       value is not None if operator == "exists" else
                       expected in value if operator == "contains" and isinstance(value, (str, list, dict)) else
                       value > expected if operator == "gt" and type(value) in (int, float) and type(expected) in (int, float) else False)
            return "yes" if matched else "no", args
        if kind == "conversation.speak":
            await self.service.speak(context, render(cfg["text"], args))
            return "success", args
        if kind in ("conversation.collect", "conversation.decision"):
            response = await self.service.conversation(context, node, args, graph)
            validate(response["data"], cfg["output_schema"])
            if response["outcome"] not in cfg["outcomes"]:
                raise ApplicationError("invalid_conversation_outcome")
            return response["outcome"], response["data"]
        if kind == "action.confirm":
            if not isinstance(args.get("arguments"), dict) or not isinstance(args.get("operation"), dict):
                raise ApplicationError("confirmation_target_required")
            confirmation_key = digest({"operation": args["operation"], "arguments": args["arguments"]})
            confirmed = context.get("confirmations", {}).get(confirmation_key, 0) >= time.monotonic() if context.get("execution_kind") == "api" else await self.service.confirm(context, render(cfg["prompt"], args))
            if confirmed:
                context.setdefault("confirmations", {})[digest({"operation": args["operation"], "arguments": args["arguments"]})] = time.monotonic() + 120
            return "confirmed" if confirmed else "declined", {"confirmed": confirmed}
        if kind == "pbx.catalog":
            context["router_policy"] = cfg
            return "success", {"destinations": await self.service.destinations(context, cfg)}
        if kind in ("pbx.route", "agent.route"):
            result = await self.service.route(context, args, kind == "agent.route")
            return result["status"], result
        if kind == "voice.consult":
            result = await self.service.consult(context, args, cfg)
            return result["status"], result
        if kind == "pbx.assignee":
            ticket = context.get("tickets", {}).get(str(args.get("ticket_id")))
            if ticket is None:
                raise ApplicationError("ticket_scope_mismatch", 403)
            target = cfg["mapping"].get(str(ticket.get("responder_id")), cfg["fallback_destination"])
            return "success", {"destination_id": target}
        if kind in ("pbx.contacts", "pbx.history"):
            result = await self.service.pbx_data(context, kind, args, cfg)
            return result.get("status", "success"), result
        if kind == "pbx.hours":
            result = await self.service.builtin(context, "nv_opening_hours_v1", args, node["id"])
            return "success", result
        if kind.startswith("identity.") or kind == "data.lookup":
            table = await self.service.table(context, cfg["resource"])
            rows = table["rows"]
            if kind == "identity.resolve":
                context.pop("identity", None)
                country = table["metadata"]["country_code"]
                phone = normalized_phone(context.get("caller", {}).get("phone"), country)
                matched = [r for r in rows if phone and r.get("phone") == phone]
                ids = {row["resident_id"] for row in matched}
                if len(ids) == 1:
                    row = matched[0]
                    context["identity"] = {"resident_id": row["resident_id"], "name": row["name"],
                                           "customer_id": row.get("customer_id"), "resource_id": cfg["resource"]["resource_id"], "verified": False, "assurance": "cli"}
                    return "known", {"resident_id": row["resident_id"], "name": row["name"], "verified": False}
                return "unknown", {"verified": False}
            if kind == "identity.verify":
                context.pop("identity", None)
                name = normalized_name(args.get("name", "")); code = args.get("resident_code", "")
                residents = sorted({r["resident_id"] for r in rows if name and matching_name(name, r["name_key"], cfg.get("name_match", "exact"))})
                await self.service.verification_attempt(context, cfg["resource"], canonical(residents) if residents else name)
                matched = [r for r in rows if name and r.get("verification_code") and isinstance(code, str) and
                           hmac.compare_digest(r["verification_code"].encode(), code.encode()) and
                           matching_name(name, r['name_key'], cfg.get('name_match','exact'))]
                ids = {row["resident_id"] for row in matched}
                if len(ids) != 1:
                    return "denied", {"verified": False}
                row = matched[0]
                context["identity"] = {"resident_id": row["resident_id"], "name": row["name"],
                                       "customer_id": row.get("customer_id"), "resource_id": cfg["resource"]["resource_id"], "verified": True}
                return "verified", {"resident_id": row["resident_id"], "name": row["name"], "verified": True}
            if time.time() - table["created"] > cfg.get("max_age_seconds", 31 * 86400):
                return "stale", {}
            identity = context.get("identity", {})
            if not identity.get("verified") or identity.get("resource_id") != cfg["resource"]["resource_id"]:
                raise ApplicationError("identity_required", 403)
            period = args.get("period")
            if not isinstance(period, str):
                raise ApplicationError("payment_period_required")
            matched = [r for r in rows if r["resident_id"] == identity["resident_id"] and r["period"] == period]
            if len(matched) != 1:
                return "ambiguous" if matched else "not_found", {}
            return "found", {"row": safe_row(matched[0]), "source": table.get("reference", cfg["resource"]), "source_row": matched[0]["source_row"]}
        if kind == "data.name_match":
            import regex
            name = normalized_name(args.get("name", ""))
            candidates = args.get("candidates", [])
            if not isinstance(candidates, list) or len(candidates) > 100:
                raise ApplicationError("candidate_capacity")
            try:
                pattern = regex.compile(cfg["pattern"]) if cfg.get("pattern") else None
                matched = [candidate for candidate in candidates if isinstance(candidate, str) and len(candidate) <= 512 and
                           (pattern.fullmatch(normalized_name(candidate), timeout=0.02) if pattern else normalized_name(candidate) == name)]
            except Exception:
                raise ApplicationError("invalid_or_slow_name_pattern") from None
            return "success", {"candidates": matched, "ambiguous": len(matched) != 1}
        if kind == "connector.invoke":
            return "success", await self.service.connector(context, node, args, graph)
        if kind == "subflow":
            ref = cfg["resource"]
            child = await self.service.subflow(context, ref)
            if not set(child["tool_grants"]) <= set(graph["tool_grants"]):
                raise ApplicationError("subflow_scope_mismatch", 403)
            outer_deadline = context["deadline_monotonic"]
            outer_permissions = context.get('permissions', {})
            outer_path = context.get('subflow_path', [])
            context['permissions'] = {key: 'allow' if value == 'allow' and child['permissions'].get(key) == 'allow' else 'deny' for key, value in outer_permissions.items()}
            context['subflow_path'] = outer_path + [node['id']]
            try:
                result = await self.execute(child, context, args, depth=depth + 1)
            finally:
                context["deadline_monotonic"] = outer_deadline
                context['permissions'] = outer_permissions
                context['subflow_path'] = outer_path
            context["child_traces"].extend(result["trace"])
            if result["status"] != "completed":
                raise ApplicationError("subflow_" + result["status"], 409)
            return "success", result["result"]
        raise ApplicationError("unknown_block")
