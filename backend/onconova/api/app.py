"""FastAPI application factory.

    uvicorn onconova.api.app:app --reload
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from onconova import __version__
from onconova.api import routes_analysis, routes_cases, routes_system
from onconova.api.binary import EXPOSED_HEADERS
from onconova.audit import AuditLog
from onconova.config import Settings, get_settings
from onconova.errors import OncoNovaError
from onconova.logging_setup import configure
from onconova.safety import RESEARCH_DISCLAIMER, assert_no_device_interfaces
from onconova.security import load_api_keys
from onconova.store import CaseStore
from onconova.synthetic.cases import materialize_demo_cases

log = logging.getLogger("onconova.api")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        assert_no_device_interfaces()
        settings.ensure_dirs()
        if settings.generate_demo_cases:
            created = materialize_demo_cases(app.state.store)
            if created:
                log.info("generated demo cases", extra={"user": "system"})
        yield

    app = FastAPI(
        title="OncoNova AI research API",
        version=__version__,
        description=f"{RESEARCH_DISCLAIMER} No endpoint controls, configures or communicates with medical equipment.",
        lifespan=lifespan,
    )
    configure(settings.log_level)
    settings.ensure_dirs()
    app.state.settings = settings
    app.state.store = CaseStore(settings.cases_dir)
    app.state.audit = AuditLog(settings.audit_log)
    app.state.api_keys = load_api_keys(settings.api_keys_file) if settings.auth_mode == "api_key" else {}

    app.add_middleware(GZipMiddleware, minimum_size=2048)
    app.add_middleware(CORSMiddleware, allow_origins=settings.allowed_origins, allow_credentials=False,
                       allow_methods=["GET", "POST", "PUT", "DELETE"], allow_headers=["Content-Type", "X-API-Key"],
                       expose_headers=EXPOSED_HEADERS)

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        log.info("request", extra={"request_id": request_id, "method": request.method, "path": request.url.path,
                                   "status": response.status_code,
                                   "duration_ms": round((time.perf_counter() - start) * 1000, 1)})
        return response

    @app.exception_handler(OncoNovaError)
    async def onconova_error(request: Request, exc: OncoNovaError):
        return JSONResponse(status_code=exc.status_code,
                            content={"error": exc.to_dict(), "request_id": getattr(request.state, "request_id", None)})

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        details = [{"loc": list(e.get("loc", [])), "msg": e.get("msg")} for e in exc.errors()]
        return JSONResponse(status_code=422, content={
            "error": {"code": "invalid_request", "message": "The request is invalid.", "details": {"errors": details}},
            "request_id": getattr(request.state, "request_id", None)})

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", None)
        log.exception("unhandled error", extra={"request_id": request_id, "error_code": type(exc).__name__})
        return JSONResponse(status_code=500, content={
            "error": {"code": "internal_error", "message": "Internal error. Quote the request ID when reporting it.",
                      "details": {}}, "request_id": request_id})

    app.include_router(routes_system.router)
    app.include_router(routes_cases.router)
    app.include_router(routes_analysis.router)
    return app


app = create_app()
