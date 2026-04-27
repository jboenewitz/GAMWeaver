"""Application configuration settings."""

import os
from pathlib import Path
from typing import List


def _bool_from_env(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_origins(raw: str) -> List[str]:
    if not raw:
        return []
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


class Settings:
    """Centralized settings with environment-based defaults."""

    def __init__(self) -> None:
        self.environment = os.getenv("ENVIRONMENT", "development").strip().lower()

        default_sqlite = Path(__file__).parent.parent.resolve() / "igann_app.db"
        raw_db_url = os.getenv("DATABASE_URL", f"sqlite:///{default_sqlite}").strip()
        if raw_db_url.startswith("postgres://"):
            raw_db_url = raw_db_url.replace("postgres://", "postgresql://", 1)
        self.database_url = raw_db_url

        raw_origins = os.getenv("CORS_ALLOW_ORIGINS", "").strip()
        self.cors_allow_origins = _parse_origins(raw_origins)
        if not self.cors_allow_origins and self.environment != "production":
            self.cors_allow_origins = [
                "http://localhost:3000",
                "http://localhost:5173",
                "http://127.0.0.1:3000",
                "http://127.0.0.1:5173",
            ]

        self.demo_admin_secret = os.getenv("DEMO_ADMIN_SECRET", "").strip()
        default_allow_destructive = self.environment != "production"
        self.allow_destructive_actions = _bool_from_env(
            os.getenv("ALLOW_DESTRUCTIVE_ACTIONS"),
            default=default_allow_destructive,
        )

        self.superadmin_username = os.getenv("SUPERADMIN_USERNAME", "superadmin").strip()
        self.superadmin_password = os.getenv("SUPERADMIN_PASSWORD", "").strip()
        self.auth_token_secret = os.getenv("AUTH_TOKEN_SECRET", "dev-secret").strip()

        raw_admin_ttl = os.getenv("ADMIN_TOKEN_TTL_HOURS", "8").strip()
        try:
            self.admin_token_ttl_hours = max(1, int(raw_admin_ttl))
        except ValueError:
            self.admin_token_ttl_hours = 8

        raw_ttl = os.getenv("INVITE_TOKEN_TTL_HOURS", "168").strip()
        try:
            self.invite_token_ttl_hours = max(1, int(raw_ttl))
        except ValueError:
            self.invite_token_ttl_hours = 168


settings = Settings()
