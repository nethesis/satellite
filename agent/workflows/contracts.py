"""Canvas-independent workflow contracts and control/data-flow validation."""

import copy
import re

from agent.application.contracts import (ApplicationError, canonical, fields,
                                        identifier, integer, schema, string)

OBJECT = {"type": "object", "properties": {}, "additionalProperties": False}
ANY_OBJECT = {"type": "object"}
REF = {"type": "object", "required": ["resource_id", "version"],
       "properties": {"resource_id": {"type": "string"}, "version": {"type": "integer", "minimum": 1}},
       "additionalProperties": False}


def manifest(kind, category, outcomes=("success", "error", "timeout"), config=None,
             inputs=None, outputs=None, voice=False, effect=False):
    return {"type": kind, "version": 1, "category": category,
            "outcomes": list(outcomes), "config_schema": config or OBJECT,
            "input_schema": inputs or ANY_OBJECT, "output_schema": outputs or ANY_OBJECT,
            "entrypoints": ["voice"] if voice else ["voice", "api"],
            "timeout_seconds": 10, "read_only": not effect}


def obj(properties, required=()):
    return {"type": "object", "properties": properties,
            "required": list(required), "additionalProperties": False}


TEXT = {"type": "string", "maxLength": 8192}
IDS = {"type": "array", "items": {"type": "string"}, "maxItems": 100}
CONVERSATION = obj({"prompt": TEXT, "output_schema": {"type": "object"},
                    "outcomes": IDS, "max_turns": {"type": "integer", "minimum": 1, "maximum": 5},
                    "tools": IDS}, ["prompt", "output_schema", "outcomes", "max_turns"])
CATALOG = obj({"defaults": obj({k: {"type": "boolean"} for k in ("extension", "queue", "ivr", "agent")}),
               "overrides": {"type": "object", "additionalProperties": {"type": "boolean"}},
               "delegations": {"type": "object", "additionalProperties": IDS}})
TABLE = obj({"resource": REF, "max_age_seconds": {"type": "integer", "minimum": 60, "maximum": 31536000}}, ["resource"])
VERIFY_TABLE = copy.deepcopy(TABLE)
VERIFY_TABLE['properties']['name_match'] = {'type':'string','enum':['exact','similar'],'description':'Similar accepts spacing differences and one character edit; the resident code must still match exactly.'}
BLOCKS = [
    manifest("start.call", "start", ("success",), outputs=obj({k: TEXT for k in ("phone", "name", "did", "origin")}), voice=True),
    manifest("start.api", "start", ("success",)),
    manifest("end", "start", (), config=obj({"status": {"enum": ["completed", "fallback"]}})),
    manifest("conversation.speak", "conversation", config=obj({"text": TEXT}, ["text"])),
    manifest("conversation.collect", "conversation", config=CONVERSATION),
    manifest("conversation.decision", "conversation", config=CONVERSATION),
    manifest("logic.condition", "logic", ("yes", "no", "error"),
             config=obj({"field": TEXT, "operator": {"enum": ["eq", "ne", "exists", "contains", "gt"]},
                         "value": {}}, ["field", "operator"])),
    manifest("logic.map", "logic"),
    manifest("logic.merge", "logic"),
    manifest("pbx.catalog", "pbx", config=CATALOG, outputs=obj({"destinations": {"type": "array", "items": ANY_OBJECT}}), voice=True),
    manifest("pbx.route", "pbx", ("handed_off", "unavailable", "error", "timeout"), voice=True, effect=True),
    manifest("agent.route", "pbx", ("handed_off", "unavailable", "error", "timeout"), voice=True, effect=True),
    manifest("pbx.hours", "pbx", voice=True),
    manifest("pbx.assignee", "pbx", config=obj({"mapping": {"type": "object", "additionalProperties": {"type": "string"}},
        "fallback_destination": TEXT}, ["mapping", "fallback_destination"]), voice=True),
    manifest("pbx.contacts", "pbx", ("success", "ambiguous", "not_found", "error", "timeout"), voice=True),
    manifest("pbx.history", "pbx", config=obj({"support_extensions": IDS,
        "lookback_days": {"type": "integer", "minimum": 1, "maximum": 365}}, ["support_extensions", "lookback_days"]), voice=True),
    manifest("identity.resolve", "identity", ("known", "unknown", "error", "timeout"), config=TABLE, voice=True),
    manifest("identity.verify", "identity", ("verified", "denied", "error", "timeout"), config=VERIFY_TABLE, voice=True),
    manifest("data.lookup", "data", ("found", "not_found", "ambiguous", "stale", "error", "timeout"), config=TABLE, voice=True),
    manifest("data.name_match", "data", config=obj({"pattern": TEXT})),
    manifest("connector.invoke", "tools", config=obj({"operation": obj({
        "connector_id": TEXT, "version": {"type": "integer", "minimum": 1}, "operation_id": TEXT},
        ["connector_id", "version", "operation_id"]), "effect_key": {'type':'string','minLength':1,'maxLength':128},
        "write_policy": {"enum": ["confirmed"]},
        "priority_field": TEXT, "allowed_priorities": {"type": "array", "items": {"type": "integer", "enum": [4]}, "maxItems": 1},
        "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 30}}, ["operation"]), effect=True),
    manifest("action.confirm", "conversation", ("confirmed", "declined", "error", "timeout"),
             config=obj({"prompt": TEXT}, ["prompt"])),
    manifest("voice.consult", "pbx", ("accepted", "declined", "busy", "no_answer", "unavailable", "failed", "unknown", "error", "timeout"),
             config=obj({"ring_seconds": {"type": "integer", "minimum": 5, "maximum": 60},
                         "consult_seconds": {"type": "integer", "minimum": 5, "maximum": 60}}, ["ring_seconds", "consult_seconds"]), voice=True, effect=True),
    manifest("subflow", "reuse", config=obj({"resource": REF}, ["resource"])),
]
# Concrete field contracts drive the inspector and validate deterministic inputs.
_BLOCK_FIELDS = {
    'pbx.contacts': ({'phone': TEXT}, ['phone'], {'status': TEXT, 'company': TEXT, 'numbers': IDS}),
    'pbx.history': ({'numbers': IDS}, ['numbers'], {'status': TEXT, 'advisory': {'type':'boolean'}, 'operators': {'type':'array','items':obj({'destination_id':TEXT,'last_call':TEXT,'call_count':{'type':'integer'}})}}),
    'pbx.route': ({'destination_id': TEXT, 'reason': TEXT}, ['destination_id'], {'status':TEXT,'destination_id':TEXT}),
    'agent.route': ({'destination_id': TEXT, 'context': ANY_OBJECT}, ['destination_id'], {'status':TEXT,'destination_id':TEXT,'child_run_id':TEXT}),
    'voice.consult': ({'destination_id':TEXT,'summary':TEXT}, ['destination_id','summary'], {'status':TEXT,'destination_id':TEXT}),
    'pbx.assignee': ({'ticket_id':TEXT}, ['ticket_id'], {'destination_id':TEXT}),
    'identity.resolve': ({}, [], {'resident_id':TEXT,'name':TEXT,'verified':{'type':'boolean'}}),
    'identity.verify': ({'name':TEXT,'resident_code':TEXT}, ['name','resident_code'], {'resident_id':TEXT,'name':TEXT,'verified':{'type':'boolean'}}),
    'data.lookup': ({'period':TEXT}, ['period'], {'row': {'type':'object','properties':{key:TEXT for key in ('resident_id','name','period','amount','currency','building','unit','customer_id')},'additionalProperties':True},'source':REF,'source_row':{'type':'integer'}}),
    'data.name_match': ({'name':TEXT,'candidates':IDS}, ['name','candidates'], {'candidates':IDS,'ambiguous':{'type':'boolean'}}),
    'action.confirm': ({'operation':ANY_OBJECT,'arguments':ANY_OBJECT}, ['operation','arguments'], {'confirmed':{'type':'boolean'}}),
}
for _block in BLOCKS:
    if _block['type'] in _BLOCK_FIELDS:
        _inputs, _required, _outputs = _BLOCK_FIELDS[_block['type']]
        _block['input_schema'] = obj(_inputs, _required)
        _block['output_schema'] = obj(_outputs)

CATALOG_BY_TYPE = {b["type"]: b for b in BLOCKS}


class DefinitionError(ApplicationError):
    def __init__(self, code, node=None):
        super().__init__(code, 422)
        self.node = node


def validate_config(value, contract, node):
    from jsonschema import Draft202012Validator
    try:
        Draft202012Validator(contract).validate(value)
    except Exception:
        raise DefinitionError("invalid_block_configuration", node) from None


def output_contract(node):
    if node["type"] in ("conversation.collect", "conversation.decision"):
        return node["config"]["output_schema"]
    return CATALOG_BY_TYPE[node["type"]]["output_schema"]


def outcomes(node):
    if node["type"] in ("conversation.collect", "conversation.decision"):
        return node["config"]["outcomes"] + ["error", "timeout"]
    return CATALOG_BY_TYPE[node["type"]]["outcomes"]


def path_schema(contract, path):
    for key in path.split(".") if path else []:
        if contract.get("type") == "array" and key.isdigit():
            contract = contract.get("items", {})
        else:
            properties = contract.get("properties", {})
            if key not in properties:
                if contract.get("additionalProperties", True) is False:
                    raise DefinitionError("unknown_output_field")
                return {}
            contract = properties[key]
    return contract


def definition(value, *, subflow=False, input_schemas=None, output_schemas=None):
    fields(value, ("schema_version", "agent_id", "name", "description", "entrypoints",
                   "nodes", "edges", "input_schema", "output_schema", "limits", "layout",
                   "provider_binding_ref", "fallback", "permissions", "tool_grants"), ("text_provider", "voice_settings"))
    value = copy.deepcopy(value)
    if value.get('voice_settings') is None:
        value.pop('voice_settings', None)
    elif isinstance(value['voice_settings'], dict):
        value['voice_settings'] = {key: item for key, item in value['voice_settings'].items() if item not in ('', None)}
    if value.get('fallback') == '':
        value['fallback'] = None
    input_schemas = input_schemas or {}
    output_schemas = output_schemas or {}
    if value["schema_version"] != 1 or type(value["schema_version"]) is not int:
        raise DefinitionError("unsupported_definition_schema")
    identifier(value["agent_id"])
    if value["agent_id"] in ("internal", "external", "support-request"):
        raise DefinitionError("reserved_agent_id")
    string(value["name"], 128); string(value["description"], 1024, empty=True, multiline=True)
    if not isinstance(value["entrypoints"], list) or not value["entrypoints"] or not set(value["entrypoints"]) <= {"voice", "api"}:
        raise DefinitionError("invalid_entrypoints")
    schema(value["input_schema"]); schema(value["output_schema"])
    if value["provider_binding_ref"] is not None:
        string(value["provider_binding_ref"], 48)
    if "text_provider" in value:
        fields(value["text_provider"], ("model", "secret_ref"))
        string(value["text_provider"]["model"], 128); identifier(value["text_provider"]["secret_ref"])
    if "voice_settings" in value:
        fields(value["voice_settings"], (), ("model", "voice", "language"))
        for item in value["voice_settings"].values():
            string(item, 128)
    if value["fallback"] is not None:
        string(value["fallback"], 128)
    if not isinstance(value["permissions"], dict) or any(v not in ("allow", "deny") for v in value["permissions"].values()):
        raise DefinitionError("invalid_permissions")
    if not isinstance(value["tool_grants"], list) or len(value["tool_grants"]) > 100 or any(not isinstance(x, str) or len(x) > 128 for x in value["tool_grants"]):
        raise DefinitionError("invalid_tool_grants")
    fields(value["limits"], ("max_steps", "max_duration_seconds", "max_context_bytes"))
    integer(value["limits"]["max_steps"], 1, 200)
    integer(value["limits"]["max_duration_seconds"], 10, 3600)
    integer(value["limits"]["max_context_bytes"], 1024, 65536)
    if not isinstance(value["layout"], dict) or len(canonical(value["layout"])) > 16384:
        raise DefinitionError("invalid_layout")
    if not isinstance(value["nodes"], list) or not 1 <= len(value["nodes"]) <= 100 or not isinstance(value["edges"], list) or len(value["edges"]) > 200:
        raise DefinitionError("graph_capacity")
    nodes = {}
    for node in value["nodes"]:
        fields(node, ("id", "type", "version", "name", "config", "inputs"))
        identifier(node["id"]); string(node["name"], 128)
        if node["id"] in nodes:
            raise DefinitionError("duplicate_node", node["id"])
        block = CATALOG_BY_TYPE.get(node["type"])
        if subflow and node["type"] in ("pbx.route", "agent.route", "voice.consult"):
            raise DefinitionError("subflow_call_control_forbidden", node["id"])
        if not block or type(node["version"]) is not int or node["version"] != block["version"]:
            raise DefinitionError("unknown_block_version", node["id"])
        if not set(value["entrypoints"]) <= set(block["entrypoints"]):
            raise DefinitionError("entrypoint_capability_mismatch", node["id"])
        validate_config(node["config"], block["config_schema"], node["id"])
        if node["type"].startswith("conversation.") and node["type"] != "conversation.speak":
            schema(node["config"]["output_schema"])
            if not node["config"]["outcomes"] or len(set(node["config"]["outcomes"])) != len(node["config"]["outcomes"]):
                raise DefinitionError("invalid_outcomes", node["id"])
            for port in node["config"]["outcomes"]:
                identifier(port)
                if port in ("error", "timeout"):
                    raise DefinitionError("reserved_outcome", node["id"])
            if not set(node["config"].get("tools", [])) <= set(value["tool_grants"]):
                raise DefinitionError("ungranted_conversation_tool", node["id"])
        if not isinstance(node["inputs"], dict) or len(node["inputs"]) > 64:
            raise DefinitionError("invalid_input_bindings", node["id"])
        nodes[node["id"]] = node
    starts = [n for n in nodes.values() if n["type"].startswith("start.")]
    if len(starts) != 1:
        raise DefinitionError("one_start_required")
    if starts[0]["type"] == "start.call" and value["entrypoints"] != ["voice"]:
        raise DefinitionError("entrypoint_capability_mismatch", starts[0]["id"])
    parents = {key: set() for key in nodes}; children = {key: [] for key in nodes}; seen = set()
    for edge in value["edges"]:
        fields(edge, ("source", "outcome", "target"))
        if edge["source"] not in nodes or edge["target"] not in nodes:
            raise DefinitionError("dangling_edge")
        if edge["outcome"] not in outcomes(nodes[edge["source"]]):
            raise DefinitionError("unknown_outcome", edge["source"])
        if (edge["source"], edge["outcome"]) in seen:
            raise DefinitionError("duplicate_outcome_edge", edge["source"])
        seen.add((edge["source"], edge["outcome"]))
        parents[edge["target"]].add(edge["source"]); children[edge["source"]].append(edge["target"])
    order = []; available = [key for key in nodes if not parents[key]]; pending = copy.deepcopy(parents)
    while available:
        current = available.pop(); order.append(current)
        for child in set(children[current]):
            pending[child].discard(current)
            if not pending[child]:
                available.append(child)
    if len(order) != len(nodes):
        raise DefinitionError("graph_cycle")
    reached = set(); stack = [starts[0]["id"]]
    while stack:
        node_id = stack.pop()
        if node_id not in reached:
            reached.add(node_id); stack.extend(children[node_id])
    if reached != set(nodes):
        raise DefinitionError("unreachable_node")
    dominators = {}; possible = {}
    for key in order:
        ancestors = set.intersection(*(dominators[p] | {p} for p in parents[key])) if parents[key] else set()
        dominators[key] = ancestors
        possible[key] = set.union(*(possible[p] | {p} for p in parents[key])) if parents[key] else set()
        for name, binding in nodes[key]["inputs"].items():
            identifier(name)
            input_contract = input_schemas.get(key, CATALOG_BY_TYPE[nodes[key]['type']]['input_schema'])
            target_schema = input_contract.get('properties', {}).get(name, {})
            if input_contract.get('additionalProperties') is False and name not in input_contract.get('properties', {}):
                raise DefinitionError('unknown_input_field', key)
            if not isinstance(binding, dict) or ("value" in binding) == ("node" in binding):
                raise DefinitionError("invalid_input_binding", key)
            if "value" in binding:
                fields(binding, ("value",))
                validate_config(binding['value'], target_schema, key)
            else:
                fields(binding, ("node", "path"), ("optional", "default"))
                if not isinstance(binding["path"], str) or not re.fullmatch(r"[A-Za-z0-9_.-]{0,256}", binding["path"]):
                    raise DefinitionError("invalid_output_path", key)
                source = binding["node"]
                if source not in nodes or source not in possible[key] or (source not in ancestors and not binding.get("optional")):
                    raise DefinitionError("unavailable_branch_output", key)
                source_contract = output_schemas.get(source, value['input_schema'] if nodes[source]['type'] == 'start.api' else output_contract(nodes[source]))
                source_schema = path_schema(source_contract, binding["path"])
                if source_schema.get("type") and target_schema.get("type") and source_schema["type"] != target_schema["type"]:
                    raise DefinitionError("input_type_mismatch", key)
        required_inputs = input_schemas.get(key, CATALOG_BY_TYPE[nodes[key]['type']]['input_schema']).get('required', [])
        if not set(required_inputs) <= nodes[key]['inputs'].keys():
            raise DefinitionError('missing_input_field', key)
        ports = outcomes(nodes[key])
        if ports and not children[key] and nodes[key]["type"] not in ("pbx.route", "agent.route", "voice.consult"):
            raise DefinitionError("missing_terminal_path", key)
    if len(canonical(value).encode()) > 131072:
        raise DefinitionError("definition_too_large")
    return value


def execution_hash(value):
    from agent.application.contracts import digest
    return digest({key: item for key, item in value.items() if key != "layout"})
