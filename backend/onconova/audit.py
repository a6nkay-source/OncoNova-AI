"""Append-only JSONL audit log. Records who did what to which resource, never image content or PHI."""

from __future__ import annotations

import json
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_lock = threading.Lock()


class AuditLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, *, user: str, role: str, action: str, resource: str, outcome: str = "ok",
               request_id: str | None = None, details: dict[str, Any] | None = None) -> None:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "user": user,
            "role": role,
            "action": action,
            "resource": resource,
            "outcome": outcome,
            "request_id": request_id,
            "details": details or {},
        }
        with _lock, self.path.open("a") as fh:
            fh.write(json.dumps(entry) + "\n")

    def tail(self, n: int = 200) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open() as fh:
            return [json.loads(line) for line in deque(fh, maxlen=n)][::-1]
