"""Editable business connector presets; real accounts/keys remain administrator input."""

from .contracts import obj, TEXT


def operation(key, method, path, inputs, output, projection, *, body=None, query=None, identity=None, read_only=True, public=False):
    return {"id": key, "description": key.replace('_', ' '), "method": method, "path": path,
            "input_schema": obj(inputs, inputs), "output_schema": obj(output, output), "projection": projection,
            "body": body or {}, "query": query or {}, "read_only": read_only, "public_voice": public,
            "timeout_seconds": 30 if key == "answer" else 10, "identity_field": identity}


def presets():
    customer = {"type": "integer", "minimum": 1}
    contact = operation("resolve_contact", "GET", "/api/v2/contacts", {"phone": TEXT},
        {"customer_id": customer, "match_count": {"type": "integer"}}, {"customer_id": "0.id", "match_count": "$count"}, query={"phone": "phone"}, identity="phone")
    ticket = obj({"id": {"type": "integer"}, "priority": {"type": "integer"},
                  "status": {"type": "integer"}, "responder_id": {"type": ["integer", "null"]}, "subject": TEXT},
                 ["id", "priority", "status", "responder_id", "subject"])
    tickets = operation("open_tickets", "GET", "/api/v2/tickets", {"customer_id": customer},
        {"tickets": {"type": "array", "items": ticket, "maxItems": 100}}, {"tickets": "$"}, query={"requester_id": "customer_id"}, identity="customer_id")
    tickets["array_projection"] = {"tickets": {key: key for key in ticket["properties"]}}
    tickets["open_statuses"] = [2, 3]
    priority = operation("update_priority", "PUT", "/api/v2/tickets/{ticket_id}",
        {"customer_id": customer, "ticket_id": {"type": "string", "maxLength": 64}, "priority": {"type": "integer", "enum": [3, 4]}},
        {"ticket_id": {"type": "integer"}, "priority": {"type": "integer"}}, {"ticket_id": "id", "priority": "priority"},
        body={"priority": "priority"}, identity="customer_id", read_only=False)
    create = operation("create_ticket", "POST", "/api/v2/tickets", {"customer_id": customer,
        "summary": {"type": "string", "maxLength": 200}, "description": {"type": "string", "maxLength": 4000},
        "priority": {"type": "integer", "enum": [1, 2, 3, 4]}, "status": {"type": "integer", "enum": [2]}},
        {"ticket_id": {"type": "integer"}}, {"ticket_id": "id"}, identity="customer_id", read_only=False,
        body={"requester_id": "customer_id", "subject": "summary", "description": "description", "priority": "priority", "status": "status"})
    freshdesk = {"name": "Freshdesk", "origin": "https://configure.freshdesk.com", "secret_ref": "freshdesk-key",
                 "auth": {"type": "basic_api_key"}, "private_networks": [], "operations": [contact, tickets, priority, create]}
    kapa = {"name": "Kapa documentation", "origin": "https://api.kapa.ai", "secret_ref": "kapa-key",
        "auth": {"type": "api_key", "header": "X-API-KEY"}, "private_networks": [], "operations": [
        operation("answer", "POST", "/query/v1/projects/{project_id}/retrieval/", {"project_id": TEXT, "question": TEXT, "mode": {"type": "string", "enum": ["deep"]}},
                  {"documentation": {}}, {"documentation": "$"}, body={"query": "question", "mode": "mode"}, public=True)]}
    return {"freshdesk": freshdesk, "kapa": kapa}
