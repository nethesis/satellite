"""Private authenticated HTTP surface for the Satellite Agent runtime."""

import hmac
import inspect
import json
import os

from fastapi import APIRouter, Depends, HTTPException, Request

from .models import AgentError


MAX_BODY_BYTES = 2 * 1024 * 1024
MAX_EVENT_BYTES = 384 * 1024  # Original 256 KiB body, base64 expansion and headers.


async def _json_body(request: Request, limit: int) -> dict:
    length = request.headers.get("content-length")
    if length is not None:
        try:
            if int(length) > limit:
                raise HTTPException(status_code=413, detail="body_too_large")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid_content_length") from exc
    size = 0
    parts = []
    async for part in request.stream():
        size += len(part)
        if size > limit:
            raise HTTPException(status_code=413, detail="body_too_large")
        parts.append(part)
    try:
        body = json.loads(b"".join(parts))
    except (ValueError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=400, detail="invalid_json") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body_must_be_object")
    return body


async def _call(method, *args):
    value = method(*args)
    return await value if inspect.isawaitable(value) else value


def _runtime_registry(runtime):
    return getattr(runtime, "tools", None) or getattr(runtime, "tool_registry", None)


def create_router(runtime, api_token: str | None = None) -> APIRouter:
    """Mount separately from the legacy API's authentication/listener rules."""
    router = APIRouter(prefix="/api/agent/v1")

    async def authenticate(request: Request) -> None:
        token = api_token if api_token is not None else os.getenv("API_TOKEN")
        if not token:
            raise HTTPException(status_code=503, detail="agent_api_unconfigured")
        header = request.headers.get("authorization", "")
        scheme, _, supplied = header.partition(" ")
        if scheme.lower() != "bearer" or not supplied or not hmac.compare_digest(supplied.encode(), token.encode()):
            raise HTTPException(status_code=401, detail="unauthorized",
                                headers={"WWW-Authenticate": "Bearer"})

    def mapped(exc: Exception) -> HTTPException:
        if isinstance(exc, AgentError):
            return HTTPException(status_code=exc.status_code, detail=exc.code)
        if isinstance(exc, (ValueError, TypeError, KeyError)):
            return HTTPException(status_code=400, detail="invalid_request")
        return HTTPException(status_code=503, detail="agent_unavailable")

    @router.put("/configuration", dependencies=[Depends(authenticate)])
    async def configuration(request: Request):
        body = await _json_body(request, MAX_BODY_BYTES)
        try:
            return await _call(runtime.configure, body)
        except Exception as exc:
            raise mapped(exc) from exc

    @router.put("/context", dependencies=[Depends(authenticate)])
    async def context_refresh(request: Request):
        body = await _json_body(request, MAX_BODY_BYTES)
        try:
            method = getattr(runtime, "refresh_context", None)
            if method is None:
                method = runtime.store.refresh_context
            return await _call(method, body)
        except Exception as exc:
            raise mapped(exc) from exc

    @router.post("/provider-events/{provider}", dependencies=[Depends(authenticate)])
    async def provider_event(provider: str, request: Request):
        if provider not in ("openai", "grok"):
            raise HTTPException(status_code=404, detail="unknown_provider")
        body = await _json_body(request, MAX_EVENT_BYTES)
        if not isinstance(body.get("binding_id"), str) or not isinstance(body.get("raw_body"), str) or not isinstance(body.get("headers"), dict):
            raise HTTPException(status_code=400, detail="invalid_event_envelope")
        try:
            return await _call(runtime.provider_event, provider, body)
        except Exception as exc:
            raise mapped(exc) from exc

    @router.get("/readiness", dependencies=[Depends(authenticate)])
    async def readiness():
        try:
            return await _call(runtime.readiness)
        except Exception as exc:
            raise mapped(exc) from exc

    @router.get("/catalog/tools", dependencies=[Depends(authenticate)])
    async def tools_catalog():
        registry = _runtime_registry(runtime)
        if registry is None:
            raise HTTPException(status_code=503, detail="agent_unavailable")
        return {"tools": registry.catalog()}

    @router.get("/catalog/permissions", dependencies=[Depends(authenticate)])
    async def permissions_catalog():
        registry = _runtime_registry(runtime)
        if registry is None:
            raise HTTPException(status_code=503, detail="agent_unavailable")
        return {"permissions": registry.permissions_catalog()}

    @router.get("/calls/{session_id}", dependencies=[Depends(authenticate)])
    async def call_status(session_id: str):
        if not 1 <= len(session_id) <= 128:
            raise HTTPException(status_code=400, detail="invalid_session_id")
        try:
            result = await _call(runtime.call_status, session_id)
            if result is None:
                raise HTTPException(status_code=404, detail="call_not_found")
            return result
        except HTTPException:
            raise
        except Exception as exc:
            raise mapped(exc) from exc

    @router.post("/calls/{session_id}/handoff", dependencies=[Depends(authenticate)])
    async def handoff(session_id: str, request: Request):
        if not 1 <= len(session_id) <= 128:
            raise HTTPException(status_code=400, detail="invalid_session_id")
        body = await _json_body(request, 16 * 1024)
        try:
            return await _call(runtime.handoff, session_id, body)
        except Exception as exc:
            raise mapped(exc) from exc

    return router
