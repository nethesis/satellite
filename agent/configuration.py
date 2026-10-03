"""Validate and atomically accept complete FreePBX Agent snapshots."""

import copy
import hashlib
import json
import os
import re
import tempfile
import threading
import time
from datetime import date
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .models import AgentUnavailable, InvalidConfiguration, RevisionConflict


TOOL_IDS = frozenset({
    "directory.find_destinations", "company.get_information",
    "calendar.get_opening_hours", "telephony.handoff",
})
PERMISSION_IDS = frozenset({
    "directory.extensions", "directory.queues", "directory.ivrs",
    "company.public_information", "company.address", "company.email",
    "company.vat_number", "calendar.opening_hours",
    "telephony.transfer.extension", "telephony.transfer.queue",
    "telephony.transfer.ivr", "telephony.consultative_transfer",
    "telephony.message_relay", "telephony.external_destination",
})
_SAFE_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_SAFE_DIALPLAN = re.compile(r"^[A-Za-z0-9_.+@:-]{1,128}$")
_FALLBACK = re.compile(r"^[A-Za-z0-9_-]+,(?:[A-Za-z0-9_.*+#-]+|\$\{EXTEN\}),[1-9][0-9]*$")
_HASH = re.compile(r"^[0-9a-f]{64}$")
_DIRECTORY_ID = re.compile(r"^(extension|queue|ivr):([A-Za-z0-9_*#-]{1,64})$")
_TIME_RULE = re.compile(r"^[^|]{1,32}\|[^|]{1,32}\|[^|]{1,32}\|[^|]{1,32}$")


def _invalid(message: str) -> None:
    raise InvalidConfiguration(message)


def _object(value: Any, name: str) -> dict:
    if not isinstance(value, dict):
        _invalid(f"{name} must be an object")
    return value


def _string(value: Any, name: str, *, nullable: bool = False, limit: int = 4096) -> None:
    if nullable and value is None:
        return
    if not isinstance(value, str) or len(value) > limit:
        _invalid(f"{name} must be a string")


def _id(value: Any, name: str, *, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        _invalid(f"{name} is invalid")


def canonical_hash(payload: dict) -> str:
    try:
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise InvalidConfiguration("payload is not canonical JSON") from exc
    return hashlib.sha256(raw).hexdigest()


def _validate_profile(key: str, profile: Any) -> None:
    p = _object(profile, f"profiles.{key}")
    required = {"trunk_id", "flow", "model", "voice", "language", "greeting",
                "prompt", "permissions", "tools", "max_call_duration_seconds",
                "fallback_destination", "company", "calendar_services"}
    if not required <= p.keys():
        _invalid(f"profiles.{key} lacks required fields")
    _id(p["trunk_id"], "trunk_id", nullable=True)
    for field in ("flow", "model", "voice", "language", "greeting", "prompt"):
        _string(p[field], field, limit=65535 if field == "prompt" else 4096)
    if p["flow"] != key.capitalize():
        _invalid("profile flow does not match agent")
    if type(p["max_call_duration_seconds"]) is not int or not 1 <= p["max_call_duration_seconds"] <= 86400:
        _invalid("invalid maximum call duration")
    _string(p["fallback_destination"], "fallback_destination", nullable=True, limit=255)
    if p["fallback_destination"] is not None and not _FALLBACK.fullmatch(p["fallback_destination"]):
        _invalid("invalid fallback destination")
    permissions = _object(p["permissions"], "permissions")
    if set(permissions) - PERMISSION_IDS or any(v not in ("allow", "deny") for v in permissions.values()):
        _invalid("invalid permission policy")
    tools = _object(p["tools"], "tools")
    if set(tools) - TOOL_IDS or any(v not in ("enabled", "disabled") for v in tools.values()):
        _invalid("invalid tool policy")
    company = _object(p["company"], "company")
    if any(not isinstance(k, str) for k in company):
        _invalid("invalid company field")
    calendars = _object(p["calendar_services"], "calendar_services")
    for service, cal_id in calendars.items():
        _id(service, "calendar service")
        _id(cal_id, "calendar ID")


def validate_payload(payload: Any) -> dict:
    p = _object(payload, "payload")
    if not {"profiles", "bindings", "destinations", "directory", "calendars"} <= p.keys():
        _invalid("incomplete snapshot")
    profiles = _object(p["profiles"], "profiles")
    if set(profiles) != {"internal", "external"}:
        _invalid("both built-in profiles are required")
    for key, profile in profiles.items():
        _validate_profile(key, profile)

    bindings = p["bindings"]
    if not isinstance(bindings, list):
        _invalid("bindings must be an array")
    binding_ids: set[str] = set()
    for binding in bindings:
        b = _object(binding, "binding")
        if not {"id", "provider", "runtime_owner", "trunk_name", "provider_user",
                "provider_host", "api_key", "webhook_secret"} <= b.keys():
            _invalid("incomplete binding")
        _id(b["id"], "binding ID")
        if b["id"] in binding_ids:
            _invalid("duplicate binding ID")
        binding_ids.add(b["id"])
        if b["provider"] not in ("openai", "grok") or b["runtime_owner"] != "builtin":
            _invalid("unsupported provider binding")
        if b["trunk_name"] != f"AgentTrunk_{b['id']}":
            _invalid("binding trunk identity mismatch")
        for field in ("provider_user", "provider_host", "api_key", "webhook_secret"):
            _string(b[field], field, limit=2048)
        if not b["api_key"] or not b["webhook_secret"]:
            _invalid("binding credentials required")
    for profile in profiles.values():
        if profile["trunk_id"] is not None and profile["trunk_id"] not in binding_ids:
            _invalid("profile references missing built-in binding")

    destinations = p["destinations"]
    if not isinstance(destinations, list):
        _invalid("destinations must be an array")
    destination_ids: set[int] = set()
    builtin_types: set[str] = set()
    for destination in destinations:
        d = _object(destination, "destination")
        if not {"id", "agent_type", "profile_key", "fallback_destination"} <= d.keys():
            _invalid("incomplete destination")
        if type(d["id"]) is not int or d["id"] <= 0 or d["id"] in destination_ids:
            _invalid("invalid or duplicate destination ID")
        destination_ids.add(d["id"])
        _string(d["agent_type"], "agent_type", limit=64)
        _string(d["fallback_destination"], "fallback_destination", nullable=True, limit=255)
        if d["fallback_destination"] is not None and not _FALLBACK.fullmatch(d["fallback_destination"]):
            _invalid("invalid destination fallback")
        if d["agent_type"] in ("builtin_internal", "builtin_external"):
            key = d["agent_type"].removeprefix("builtin_")
            if d["profile_key"] != key:
                _invalid("invalid built-in destination ownership")
            builtin_types.add(d["agent_type"])
        elif d["profile_key"] is not None:
            _invalid("non-built-in destination has built-in profile")
    if builtin_types != {"builtin_internal", "builtin_external"}:
        _invalid("both built-in destinations are required")

    directory = p["directory"]
    if not isinstance(directory, list):
        _invalid("directory must be an array")
    directory_ids: set[str] = set()
    for item in directory:
        r = _object(item, "directory resource")
        if not {"id", "type", "name", "description", "synonyms", "internal_allowed",
                "external_allowed", "target"} <= r.keys():
            _invalid("incomplete directory resource")
        match = _DIRECTORY_ID.fullmatch(str(r["id"]))
        if not match or r["type"] != match.group(1) or r["id"] in directory_ids:
            _invalid("invalid or duplicate directory ID")
        directory_ids.add(r["id"])
        _string(r["name"], "name", limit=1024)
        _string(r["description"], "description", limit=4096)
        if not isinstance(r["synonyms"], list) or any(not isinstance(s, str) or len(s) > 128 for s in r["synonyms"]):
            _invalid("invalid directory synonyms")
        if any(type(r[f]) is not bool for f in ("internal_allowed", "external_allowed")):
            _invalid("invalid directory visibility")
        target = _object(r["target"], "target")
        if set(target) != {"context", "exten", "priority"} or target["priority"] != 1:
            _invalid("invalid trusted target")
        if any(not isinstance(target[f], str) or not _SAFE_DIALPLAN.fullmatch(target[f]) for f in ("context", "exten")):
            _invalid("invalid trusted target")

    calendars = _object(p["calendars"], "calendars")
    for cal_id, cal in calendars.items():
        _id(cal_id, "calendar ID")
        c = _object(cal, "calendar")
        if not {"timezone", "rules", "override", "observed_at", "supported"} <= c.keys():
            _invalid("incomplete calendar")
        try:
            ZoneInfo(c["timezone"])
        except (TypeError, KeyError, ZoneInfoNotFoundError):
            _invalid("invalid calendar timezone")
        if not isinstance(c["rules"], list) or any(not isinstance(rule, str) or not _TIME_RULE.fullmatch(rule) for rule in c["rules"]):
            _invalid("invalid calendar rules")
        # Unrecognized PBX expressions make that calendar unknown at execution;
        # they must not prevent activation of the other profiles and tools.
        if c["override"] not in ("auto", "open", "closed", "unknown"):
            _invalid("invalid calendar override")
        if type(c["observed_at"]) is not int or c["observed_at"] < 0 or type(c["supported"]) is not bool:
            _invalid("invalid calendar observation")
        if "max_age_seconds" in c and (type(c["max_age_seconds"]) is not int or not 0 <= c["max_age_seconds"] <= 86400):
            _invalid("invalid calendar freshness")
        if "exceptions" in c:
            if not isinstance(c["exceptions"], dict):
                _invalid("invalid calendar exceptions")
            for exception_date, state in c["exceptions"].items():
                try:
                    if not isinstance(exception_date, str) or len(exception_date) != 10:
                        raise ValueError("bad date")
                    date.fromisoformat(exception_date)
                except ValueError:
                    _invalid("invalid calendar exception date")
                if state not in ("open", "closed"):
                    _invalid("invalid calendar exception state")
    for profile in profiles.values():
        if any(cal_id not in calendars for cal_id in profile["calendar_services"].values()):
            _invalid("profile references missing calendar")
    return p


class ConfigurationStore:
    """A versioned in-memory snapshot with an optional nonsecret disk watermark."""

    def __init__(self, state_path: str | None = None):
        self._lock = threading.RLock()
        self._snapshot: dict | None = None
        self._live_context: dict | None = None
        self._revision = 0
        self._payload_hash: str | None = None
        self._state_path = Path(state_path) if state_path else (Path(os.environ["SATELLITE_AGENT_STATE_PATH"]) if os.getenv("SATELLITE_AGENT_STATE_PATH") else None)
        self._receipts: dict[str, int] = {}
        self.state_error = False
        try:
            if self._state_path and self._state_path.exists():
                state = json.loads(self._state_path.read_text(encoding="utf-8"))
                if (not isinstance(state, dict) or type(state.get("revision")) is not int
                        or state["revision"] < 0 or "payload_hash" not in state
                        or (state["revision"] and not _HASH.fullmatch(state.get("payload_hash") or ""))):
                    raise InvalidConfiguration("invalid persisted Agent watermark")
                self._revision = state["revision"]
                self._payload_hash = state["payload_hash"]
                receipts = state.get("receipts", {})
                if isinstance(receipts, dict):
                    self._receipts = {k: v for k, v in receipts.items()
                                      if isinstance(k, str) and _HASH.fullmatch(k)
                                      and type(v) is int and v > int(time.time()) - 86400}
        except (OSError, ValueError, TypeError, InvalidConfiguration):
            # Do not erase an unreadable watermark and accept an older snapshot.
            # Keep the legacy API alive; only trusted local repair/restore resets it.
            self.state_error = True

    @property
    def snapshot(self) -> dict | None:
        with self._lock:
            return copy.deepcopy(self._snapshot)

    @property
    def live_context(self) -> dict | None:
        with self._lock:
            return copy.deepcopy(self._live_context)

    @property
    def revision(self) -> int:
        with self._lock:
            return self._revision

    @property
    def payload_hash(self) -> str | None:
        with self._lock:
            return self._payload_hash

    def _persist(self, revision: int, payload_hash: str | None) -> None:
        if not self._state_path:
            return
        self._state_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd, temp_name = tempfile.mkstemp(prefix=".agent-state-", dir=self._state_path.parent)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump({"revision": revision, "payload_hash": payload_hash,
                           "receipts": self._receipts}, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, self._state_path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)

    def apply(self, envelope: dict) -> dict:
        if self.state_error:
            raise AgentUnavailable("persisted Agent state unavailable")
        e = _object(envelope, "envelope")
        if e.get("schema_version") != 1 or type(e.get("revision")) is not int or e["revision"] <= 0:
            _invalid("unsupported schema version or revision")
        if not isinstance(e.get("payload_hash"), str) or not _HASH.fullmatch(e["payload_hash"]):
            _invalid("invalid payload hash")
        if canonical_hash(e.get("payload")) != e["payload_hash"]:
            _invalid("payload hash mismatch")
        candidate = copy.deepcopy(validate_payload(e["payload"]))
        with self._lock:
            revision = e["revision"]
            if revision < self._revision or (revision == self._revision and e["payload_hash"] != self._payload_hash):
                raise RevisionConflict("revision is older or conflicts with accepted hash")
            if revision > self._revision:
                self._persist(revision, e["payload_hash"])
            rehydrate_context = self._snapshot is None or revision != self._revision
            self._snapshot = candidate
            if rehydrate_context:
                self._live_context = {"directory": copy.deepcopy(candidate["directory"]),
                                      "calendars": copy.deepcopy(candidate["calendars"])}
            self._revision = revision
            self._payload_hash = e["payload_hash"]
            return {"revision": revision, "payload_hash": self._payload_hash}

    def refresh_context(self, body: dict) -> dict:
        b = _object(body, "context")
        if set(b) - {"directory", "calendars", "payload_hash"} or not {"directory", "calendars"} <= b.keys():
            _invalid("incomplete context refresh")
        with self._lock:
            if self._snapshot is None:
                _invalid("configuration is not active")
            if b.get("payload_hash", self._payload_hash) != self._payload_hash:
                raise RevisionConflict("context refresh is for another revision")
            candidate = copy.deepcopy(self._snapshot)
            candidate["directory"] = b["directory"]
            candidate["calendars"] = b["calendars"]
            validate_payload(candidate)
            self._live_context = {"directory": copy.deepcopy(b["directory"]),
                                  "calendars": copy.deepcopy(b["calendars"])}
            return {"revision": self._revision, "payload_hash": self._payload_hash}

    def reset_for_restore(self) -> None:
        """Trusted local restore lifecycle only; deliberately absent from HTTP API."""
        with self._lock:
            self._receipts.clear()
            self._persist(0, None)
            self.state_error = False
            self._snapshot = None
            self._live_context = None
            self._revision = 0
            self._payload_hash = None

    def has_receipt(self, receipt_id: str) -> bool:
        digest = hashlib.sha256(receipt_id.encode("utf-8")).hexdigest()
        with self._lock:
            return digest in self._receipts

    def remember_receipt(self, receipt_id: str) -> None:
        """Store only a bounded hash of an accepted event identity."""
        digest = hashlib.sha256(receipt_id.encode("utf-8")).hexdigest()
        with self._lock:
            now = int(time.time())
            self._receipts = {k: v for k, v in self._receipts.items() if v > now - 86400}
            self._receipts[digest] = now
            if len(self._receipts) > 1024:
                self._receipts = dict(sorted(self._receipts.items(), key=lambda item: item[1])[-1024:])
            self._persist(self._revision, self._payload_hash)
