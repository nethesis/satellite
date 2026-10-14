"""Validate and detach a pinned published configuration."""
import copy
import json
from pathlib import Path
from jsonschema import Draft202012Validator

SCHEMA = json.loads(Path(__file__).with_name("published-v1.schema.json").read_text())
VALIDATOR = Draft202012Validator(SCHEMA)
VALIDATORS = {1: VALIDATOR, 2: Draft202012Validator(json.loads(Path(__file__).with_name("published-v2.schema.json").read_text()))}
VALIDATORS[3] = Draft202012Validator(json.loads(Path(__file__).with_name("published-v3.schema.json").read_text()))
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
        if snapshot.get("contract_version") not in VALIDATORS or not VALIDATORS[snapshot["contract_version"]].is_valid(snapshot):
            raise ValueError("invalid_snapshot")
        if snapshot["destination_id"] != "agent:" + snapshot["agent_id"]:
            raise ValueError("invalid_snapshot")
        if snapshot['contract_version'] == 3:
            bindings = snapshot['tool_bindings']
            if len({b['tool_id'] for b in bindings}) != len(bindings) or any(type(b['tool_version']) is not int for b in bindings):
                raise ValueError('invalid_snapshot')
        return copy.deepcopy(snapshot)
    except (TypeError, OverflowError, UnicodeError) as exc:
        raise ValueError("invalid_snapshot") from exc
