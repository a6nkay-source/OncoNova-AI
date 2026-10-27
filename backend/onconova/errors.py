"""Typed errors that map to safe, informative API responses."""

from __future__ import annotations

from typing import Any


class OncoNovaError(Exception):
    """Base error. ``code`` is stable and machine-readable; ``message`` is safe to show users."""

    status_code = 400

    def __init__(self, code: str, message: str, *, details: dict[str, Any] | None = None,
                 status_code: int | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}
        if status_code is not None:
            self.status_code = status_code

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": self.details}


class ImagingError(OncoNovaError):
    status_code = 422


class ModelInputError(OncoNovaError):
    status_code = 422


class NotFoundError(OncoNovaError):
    status_code = 404


class PermissionDenied(OncoNovaError):
    status_code = 403


class UploadRejected(OncoNovaError):
    status_code = 400


class ConflictError(OncoNovaError):
    status_code = 409
