"""Bounded, nonblocking, redacted observations of Agent execution."""

import logging
import threading
import time
import uuid
from collections import deque


_SAFE_FIELDS = {
    "run_id", "session_id", "agent_id", "invocation_id", "tool_id",
    "provider", "state", "status", "outcome", "error_code", "duration_ms",
    "revision", "attempt_id", "binding_id", "destination_id", "reason_code",
    "payload_hash", "runtime_epoch", "cdr_id",
    "execution_kind", "client_id", "definition_revision", "grant_revision",
    "operation_id", "effect_state", "retry_count", "actor",
}


class EventSink:
    def __init__(self, capacity: int = 512, logger: logging.Logger | None = None):
        if not 1 <= capacity <= 10000:
            raise ValueError("invalid event capacity")
        self._events: deque[dict] = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._sequences: dict[str, int] = {}
        self.dropped_count = 0
        self.observer = None
        self.logger = logger or logging.getLogger("agent.events")

    def emit(self, event_type: str, **fields) -> dict:
        """Never raise or expose dynamic payload text to logs/history."""
        try:
            clean = {k: v[:128].encode("utf-8",errors="replace").decode("utf-8") if isinstance(v, str) else v
                     for k, v in fields.items() if k in _SAFE_FIELDS
                     and (isinstance(v, str) or type(v) in (int, float, bool))}
            run_id = clean.get("run_id")
            with self._lock:
                sequence = self._sequences.get(run_id, 0) + 1 if run_id else 0
                if run_id:
                    self._sequences[run_id] = sequence
                    if len(self._sequences) > 2048:
                        self._sequences.pop(next(iter(self._sequences)))
                event = {"schema_version": 1, "event_type": str(event_type)[:64],
                         "timestamp": time.time(), "sequence": sequence, "event_id": uuid.uuid4().hex, **clean}
                if len(self._events) == self._events.maxlen:
                    self.dropped_count += 1
                self._events.append(event)
            if self.observer is not None:
                try: self.observer(event)
                except Exception: pass
            try:
                self.logger.info("agent event %s", event)
            except Exception:
                pass
            return event
        except Exception:
            return {"schema_version": 1, "event_type": "event_dropped"}

    def recent(self) -> list[dict]:
        with self._lock:
            return list(self._events)
