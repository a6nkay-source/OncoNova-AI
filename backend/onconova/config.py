"""Runtime configuration, read from ONCONOVA_* environment variables or .env."""

from __future__ import annotations

import os
import secrets
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ONCONOVA_", env_file=".env", extra="ignore")

    data_dir: Path = REPO_ROOT / "data"
    artifacts_dir: Path = REPO_ROOT / "artifacts"
    configs_dir: Path = REPO_ROOT / "configs"

    # "dev" accepts requests from localhost only and treats the caller as a local
    # admin. "api_key" requires an X-API-Key header that matches a hashed key in
    # api_keys_file. Use "api_key" anywhere other than a single-user laptop.
    auth_mode: Literal["dev", "api_key"] = "dev"
    api_keys_file: Path | None = None

    max_upload_mb: int = 512
    max_zip_members: int = 4000
    max_uncompressed_mb: int = 2048

    allowed_origins: list[str] = ["http://localhost:5174", "http://127.0.0.1:5174"]
    pseudonym_salt: str | None = None
    log_level: str = "INFO"
    generate_demo_cases: bool = True

    @property
    def cases_dir(self) -> Path:
        return self.data_dir / "cases"

    @property
    def tmp_dir(self) -> Path:
        return self.data_dir / "tmp"

    @property
    def audit_log(self) -> Path:
        return self.data_dir / "audit" / "audit.jsonl"

    @property
    def manifests_dir(self) -> Path:
        return self.data_dir / "manifests"

    @property
    def models_dir(self) -> Path:
        return self.artifacts_dir / "models"

    @property
    def evaluations_dir(self) -> Path:
        return self.artifacts_dir / "evaluations"

    def ensure_dirs(self) -> None:
        for path in (self.cases_dir, self.tmp_dir, self.audit_log.parent, self.manifests_dir,
                     self.models_dir, self.evaluations_dir):
            path.mkdir(parents=True, exist_ok=True)
        os.chmod(self.tmp_dir, 0o700)

    def salt(self) -> str:
        """Salt for pseudonymising identifiers. Generated once per install if not configured."""
        if self.pseudonym_salt:
            return self.pseudonym_salt
        salt_file = self.data_dir / ".pseudonym_salt"
        if not salt_file.exists():
            salt_file.parent.mkdir(parents=True, exist_ok=True)
            salt_file.write_text(secrets.token_hex(32))
            os.chmod(salt_file, 0o600)
        return salt_file.read_text().strip()


@lru_cache
def get_settings() -> Settings:
    return Settings()
