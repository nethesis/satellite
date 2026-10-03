"""Small shared data and error contracts for the Agent runtime."""

from dataclasses import dataclass
from typing import Any


class AgentError(Exception):
    code = "agent_error"
    status_code = 400

    def __init__(self, message: str | None = None):
        super().__init__(message or self.code)


class InvalidConfiguration(AgentError):
    code = "invalid_configuration"


class RevisionConflict(AgentError):
    code = "revision_conflict"
    status_code = 409


class PermissionDenied(AgentError):
    code = "permission_denied"
    status_code = 403


class AgentUnavailable(AgentError):
    code = "agent_unavailable"
    status_code = 503


class InvalidInvocation(AgentError):
    code = "invalid_invocation"


@dataclass(frozen=True)
class ToolManifest:
    id: str
    wire_name: str
    version: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    timeout_seconds: float
    read_only: bool
    required_capability: str | None = None
