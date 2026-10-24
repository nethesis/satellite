"""Validate and detach a pinned published configuration."""
import copy
import json
from pathlib import Path
from jsonschema import Draft202012Validator, validators

# PHP treats configuration integers strictly; JSON's 20.0 must not bypass it.
StrictValidator = validators.extend(Draft202012Validator, type_checker=
    Draft202012Validator.TYPE_CHECKER.redefine('integer', lambda checker, value: type(value) is int))

SCHEMA = json.loads(Path(__file__).with_name("published-v8.schema.json").read_text())
VALIDATOR = StrictValidator(SCHEMA)
MAX_SNAPSHOT_BYTES = 64 * 1024


def validate_snapshot(snapshot: dict) -> dict:
    try:
        if not isinstance(snapshot, dict):
            raise ValueError("invalid_snapshot")
        for field in ("contract_version", "workflow_version", "max_call_duration_seconds"):
            if type(snapshot.get(field)) is not int:
                raise ValueError("invalid_snapshot")
        if len(json.dumps(snapshot, ensure_ascii=False, separators=(",", ":")).encode()) > MAX_SNAPSHOT_BYTES:
            raise ValueError("invalid_snapshot")
        if snapshot.get("contract_version") != 8 or not VALIDATOR.is_valid(snapshot):
            raise ValueError("invalid_snapshot")
        if snapshot["destination_id"] != "agent:" + snapshot["agent_id"]:
            raise ValueError("invalid_snapshot")
        bindings = snapshot['tool_bindings']
        if len({b['tool_id'] for b in bindings}) != len(bindings) or any(type(b['tool_version']) is not int for b in bindings):
            raise ValueError('invalid_snapshot')
        if snapshot['debug']['transcription_enabled'] and snapshot['provider'] != 'openai':
            raise ValueError('invalid_snapshot')
        from .models import conversation_capability
        conversation_capability(snapshot['provider'],snapshot['conversation'])
        for binding in snapshot['tool_bindings']:
            descriptions = binding['config'].get('descriptions', [])
            if len({d['destination_id'] for d in descriptions}) != len(descriptions):
                raise ValueError('invalid_snapshot')
        return copy.deepcopy(snapshot)
    except (TypeError, OverflowError, UnicodeError) as exc:
        raise ValueError("invalid_snapshot") from exc
