"""Request dependencies: settings, store, audit log and the authenticated principal."""

from __future__ import annotations

from collections.abc import Callable

from fastapi import Depends, Request

from onconova.audit import AuditLog
from onconova.config import Settings
from onconova.errors import OncoNovaError, PermissionDenied
from onconova.security import LOCAL_HOSTS, Principal, authenticate
from onconova.store import CaseStore


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_store(request: Request) -> CaseStore:
    return request.app.state.store


def get_audit(request: Request) -> AuditLog:
    return request.app.state.audit


def get_principal(request: Request) -> Principal:
    settings: Settings = request.app.state.settings
    if settings.auth_mode == "dev":
        host = request.client.host if request.client else ""
        if host not in LOCAL_HOSTS:
            raise OncoNovaError("dev_mode_remote", "Dev mode only accepts requests from localhost.", status_code=401)
        return Principal("local-dev", "admin")
    principal = authenticate(request.headers.get("X-API-Key"), request.app.state.api_keys)
    if principal is None:
        raise OncoNovaError("unauthenticated", "A valid X-API-Key header is required.", status_code=401)
    return principal


def require(role: str) -> Callable[..., Principal]:
    def dependency(request: Request, principal: Principal = Depends(get_principal)) -> Principal:
        if not principal.has(role):
            request.app.state.audit.record(user=principal.user, role=principal.role, action="access_denied",
                                           resource=request.url.path, outcome="denied",
                                           request_id=getattr(request.state, "request_id", None))
            raise PermissionDenied("forbidden", f"This action needs the {role!r} role.")
        return principal

    return dependency


def audit(request: Request, principal: Principal, action: str, resource: str, **details) -> None:
    request.app.state.audit.record(user=principal.user, role=principal.role, action=action, resource=resource,
                                   request_id=getattr(request.state, "request_id", None), details=details)
