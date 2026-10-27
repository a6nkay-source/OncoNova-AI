"""Access control: role-based principals from hashed API keys, or localhost-only dev mode."""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from pathlib import Path

import yaml

ROLES = ("viewer", "researcher", "admin")
LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost", "testclient"}


@dataclass(frozen=True)
class Principal:
    user: str
    role: str

    def has(self, role: str) -> bool:
        return ROLES.index(self.role) >= ROLES.index(role)


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def load_api_keys(path: Path | None) -> dict[str, Principal]:
    """Keys file format (store only SHA-256 hashes, never raw keys)::

        keys:
          - sha256: <hex>
            user: alice
            role: researcher
    """
    if path is None or not path.exists():
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    out = {}
    for entry in data.get("keys", []):
        role = entry.get("role", "viewer")
        if role not in ROLES:
            raise ValueError(f"unknown role {role!r} in {path}")
        out[str(entry["sha256"]).lower()] = Principal(str(entry["user"]), role)
    return out


def authenticate(key: str | None, keys: dict[str, Principal]) -> Principal | None:
    if not key:
        return None
    digest = hash_key(key)
    for stored, principal in keys.items():
        if hmac.compare_digest(stored, digest):
            return principal
    return None
