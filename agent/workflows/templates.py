"""Editable sample workflows. Connection/resource references need real setup."""

import copy

from .contracts import OBJECT, obj, TEXT


def node(key, kind, config=None, inputs=None):
    return {"id": key, "type": kind, "version": 1, "name": key.replace("_", " ").capitalize(), "config": config or {}, "inputs": inputs or {}}


def source(node_id, path="", optional=False, default=None):
    return {"node": node_id, "path": path} | ({"optional": True, "default": default} if optional else {})


def literal(value):
    return {"value": value}


def conversation(prompt, properties, outcomes=("success",)):
    return {"prompt": prompt, "output_schema": obj(properties, properties), "outcomes": list(outcomes), "max_turns": 5, "tools": []}


def ref(operation):
    return {"connector_id": "freshdesk", "version": 1, "operation_id": operation}


def graph(key, name, nodes, connections):
    result = {"schema_version": 1, "agent_id": key, "name": name, "description": "", "entrypoints": ["voice"],
              "input_schema": OBJECT, "output_schema": OBJECT, "provider_binding_ref": None, "fallback": None,
              "permissions": {"directory.extensions": "allow", "directory.queues": "allow", "directory.ivrs": "allow", "telephony.transfer.extension": "allow", "telephony.transfer.queue": "allow", "telephony.transfer.ivr": "allow", "telephony.consultative_transfer": "allow"},
              "tool_grants": [], "limits": {"max_steps": 200, "max_duration_seconds": 600, "max_context_bytes": 65536},
              "nodes": nodes, "edges": [{"source": a, "outcome": b, "target": c} for a, b, c in connections],
              "layout": {n["id"]: {"x": 70 + (i % 5) * 260, "y": 70 + (i // 5) * 300} for i, n in enumerate(nodes)}}
    result["tool_grants"] = [f"connector.{n['config']['operation']['connector_id']}.{n['config']['operation']['operation_id']}.v{n['config']['operation']['version']}" for n in nodes if n["type"] == "connector.invoke"]
    return result


def templates():
    router = graph("call-router", "Call router", [
        node("call", "start.call"),
        node("destinations", "pbx.catalog", {"defaults": {"extension": False, "queue": False, "ivr": False, "agent": False}, "overrides": {}, "delegations": {}}),
        node("choose", "conversation.decision", conversation("Ask where the caller wants to go. Select an exact permitted destination ID from the supplied catalog. Clarify ambiguity; choose fallback if no destination fits.",
            {"destination_id": TEXT, "reason": TEXT}, ("pbx", "agent", "fallback")), {"destinations": source("destinations", "destinations")}),
        node("route_pbx", "pbx.route", inputs={"destination_id": source("choose", "destination_id"), "reason": source("choose", "reason")}),
        node("route_agent", "agent.route", inputs={"destination_id": source("choose", "destination_id")}),
        node("fallback", "end", {"status": "fallback"}),
    ], [("call", "success", "destinations"), ("destinations", "success", "choose"), ("choose", "pbx", "route_pbx"),
        ("choose", "agent", "route_agent"), ("choose", "fallback", "fallback"), ("route_pbx", "unavailable", "fallback"), ("route_agent", "unavailable", "fallback")])
    resource = {"resource_id": "payments", "version": "latest"}
    secretary = graph("payment-secretary", "Payment secretary", [
        node("call", "start.call"), node("identify", "identity.resolve", {"resource": resource}),
        node("ask_identity", "conversation.collect", conversation("Collect the resident's full name including surname and resident verification code. Explain that a verification code is required before disclosing payment details. Once both values are stated, immediately finish this step; the next step verifies them. Return spoken numeric codes as ASCII digit strings without spaces (spoken zero one becomes 01), preserving leading zeroes. Do not attempt to verify the code yourself or ask again for values already supplied.",
            {"name": TEXT, "resident_code": TEXT})),
        node("verify", "identity.verify", {"resource": resource, "name_match":"similar"}, {"name": source("ask_identity", "name"), "resident_code": source("ask_identity", "resident_code")}),
        node("period", "conversation.collect", conversation("Ask which payment month the caller needs. Return the month as YYYY-MM; clarify the year if needed.", {"period": {"type": "string", "maxLength": 7}})),
        node("payment", "data.lookup", {"resource": resource, "max_age_seconds": 2678400}, {"period": source("period", "period")}),
        node("answer", "conversation.speak", {"text": "Tell the caller this exact payment, including month and currency: {{row}}. Do not calculate another amount."}, {"row": source("payment", "row")}),
        node("done", "end"), node("fallback", "end", {"status": "fallback"}),
    ], [("call", "success", "identify"), ("identify", "known", "ask_identity"), ("identify", "unknown", "ask_identity"),
        ("ask_identity", "success", "verify"), ("verify", "verified", "period"), ("verify", "denied", "fallback"),
        ("period", "success", "payment"), ("payment", "found", "answer"), ("payment", "not_found", "fallback"),
        ("payment", "ambiguous", "fallback"), ("payment", "stale", "fallback"), ("answer", "success", "done")])
    support = graph("customer-support", "Customer support", [
        node("call", "start.call"),
        node('welcome', 'conversation.speak', {'text':'Welcome to customer support. Tell the caller you are checking their contact and open tickets.'}),
        node("company", "pbx.contacts", inputs={"phone": source("call", "phone")}),
        node("history", "pbx.history", {"support_extensions": [], "lookback_days": 90}, {"numbers": source("company", "numbers")}),
        node("contact", "connector.invoke", {"operation": ref("resolve_contact")}, {"phone": source("call", "phone")}),
        node("tickets", "connector.invoke", {"operation": ref("open_tickets")}, {"customer_id": source("contact", "customer_id")}),
        node("triage", "conversation.decision", conversation("Discuss the issue using the open tickets and support history. Select an existing ticket only from the supplied records; otherwise collect a new issue. Assess urgency from the administrator's rule, without changing any ticket yourself.",
            {"ticket_id": {'type':'string','maxLength':64}, "urgent": {"type": "boolean"}, "summary": {'type':'string','maxLength':200}, "description": {'type':'string','maxLength':4000}}, ("existing", "new")),
            {"tickets": source("tickets", "tickets"), "history": source("history", optional=True, default={"operators": []}), 'company':source('company',optional=True,default={'numbers':[]}), 'caller':source('call')}),
        node("urgent", "logic.condition", {"field": "urgent", "operator": "eq", "value": True}, {"urgent": source("triage", "urgent")}),
        node("priority_fields", "logic.map", inputs={"customer_id": source("contact", "customer_id"), "ticket_id": source("triage", "ticket_id"), "priority": literal(4)}),
        node("confirm_priority", "action.confirm", {"prompt": "Explain that ticket {{arguments.ticket_id}} will be escalated to urgent priority. Ask the caller to press 1 to confirm or 2 to keep the current priority."},
             {"operation": literal(ref("update_priority")), "arguments": source("priority_fields")}),
        node("priority", "connector.invoke", {"operation": ref("update_priority"), "write_policy": "confirmed", "priority_field": "priority", "allowed_priorities": [4]},
             {"customer_id": source("contact", "customer_id"), "ticket_id": source("triage", "ticket_id"), "priority": literal(4)}),
        node("assignee", "pbx.assignee", {"mapping": {}, "fallback_destination": "queue:unconfigured"}, {"ticket_id": source("triage", "ticket_id")}),
        node("consult", "voice.consult", {"ring_seconds": 25, "consult_seconds": 30}, {"destination_id": source("assignee", "destination_id"), "summary": source("triage", "summary")}),
        node("resume", "conversation.speak", {"text": "The operator is unavailable. Explain that the caller can speak to the support secretary or leave a request."}),
        node("kapa", "connector.invoke", {"operation": {"connector_id": "kapa", "version": 1, "operation_id": "answer"}, "timeout_seconds": 30},
             {"question": source("triage", "description"), "project_id": literal("configure-kapa-project"), "mode": literal("deep")}),
        node("explain", "conversation.speak", {"text": "Answer the caller's issue using only relevant documentation passages and their source URLs. If the documentation is empty or irrelevant, say that no answer was found. Then offer to open the requested ticket. {{answer}}"}, {"answer": source("kapa")}),
        node("ticket_fields", "logic.map", inputs={"customer_id": source("contact", "customer_id"), "summary": source("triage", "summary"), "description": source("triage", "description"), "priority": literal(2), "status": literal(2)}),
        node("confirm", "action.confirm", {"prompt": "Read back the ticket details {{arguments}}. Ask the caller to press 1 to create this ticket or 2 to decline."},
             {"operation": literal(ref("create_ticket")), "arguments": source("ticket_fields")}),
        node("create", "connector.invoke", {"operation": ref("create_ticket"), "write_policy": "confirmed", "effect_key": "create_ticket"},
             {"customer_id": source("ticket_fields", "customer_id"), "summary": source("ticket_fields", "summary"), "description": source("ticket_fields", "description"), "priority": source("ticket_fields", "priority"), "status": source("ticket_fields", "status")}),
        node("reference", "conversation.speak", {"text": "The ticket service confirmed this new ticket reference: {{ticket_id}}."}, {"ticket_id": source("create", "ticket_id")}),
        node("done", "end"), node("fallback", "end", {"status": "fallback"}),
    ], [("call", "success", "welcome"), ('welcome','success','company'), ("company", "success", "history"), ("company", "not_found", "history"),
        ("company", "ambiguous", "history"), ("company", "error", "contact"), ("history", "success", "contact"), ("history", "error", "contact"),
        ("contact", "success", "tickets"), ("tickets", "success", "triage"), ("triage", "existing", "urgent"),
        ("urgent", "yes", "priority_fields"), ("priority_fields", "success", "confirm_priority"),
        ("confirm_priority", "confirmed", "priority"), ("confirm_priority", "declined", "assignee"),
        ("urgent", "no", "assignee"), ("priority", "success", "assignee"), ("priority", "error", "assignee"),
        ("assignee", "success", "consult"), ("consult", "declined", "resume"), ("consult", "busy", "resume"),
        ("consult", "no_answer", "resume"), ("consult", "unavailable", "resume"), ("consult", "failed", "resume"), ("resume", "success", "fallback"),
        ("triage", "new", "kapa"), ("kapa", "success", "explain"), ("kapa", "error", "ticket_fields"),
        ("explain", "success", "ticket_fields"), ("ticket_fields", "success", "confirm"), ("confirm", "confirmed", "create"),
        ("confirm", "declined", "done"), ("create", "success", "reference"), ("reference", "success", "done")])
    blank_voice = graph("blank-voice", "Blank voice agent", [node("call", "start.call"), node("done", "end")], [("call", "success", "done")])
    blank_api = graph("blank-api", "Blank API agent", [node("request", "start.api"), node("done", "end")], [("request", "success", "done")])
    blank_api["entrypoints"] = ["api"]
    return copy.deepcopy([router, secretary, support, blank_voice, blank_api])
