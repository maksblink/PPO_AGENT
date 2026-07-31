from __future__ import annotations

import os
from pathlib import Path

import pytest

from train_and_eval.database.session import (
    DatabaseConfigurationError,
    create_database_engine,
    get_database_url,
)


VALID_DATABASE_URL = (
    "postgresql+psycopg://"
    "ppo_agent:password@localhost:5433/ppo_agent"
)


def test_database_url_prefers_environment_variable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "DATABASE_URL="
        "postgresql+psycopg://other:other@localhost/other\n",
        encoding="utf-8",
    )

    monkeypatch.setenv(
        "DATABASE_URL",
        VALID_DATABASE_URL,
    )

    assert get_database_url(
        env_path=env_path
    ) == VALID_DATABASE_URL


def test_database_url_is_read_from_dotenv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(
        "DATABASE_URL",
        raising=False,
    )

    env_path = tmp_path / ".env"
    env_path.write_text(
        f"DATABASE_URL={VALID_DATABASE_URL}\n",
        encoding="utf-8",
    )

    assert get_database_url(
        env_path=env_path
    ) == VALID_DATABASE_URL

    # Reading the file does not modify the process environment.
    assert os.environ.get("DATABASE_URL") is None


def test_database_url_is_required(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(
        "DATABASE_URL",
        raising=False,
    )

    with pytest.raises(
        DatabaseConfigurationError,
        match="DATABASE_URL is not configured",
    ):
        get_database_url(
            env_path=tmp_path / "missing.env"
        )


@pytest.mark.parametrize(
    "database_url",
    [
        "sqlite:///local.db",
        "postgresql+psycopg2://user:password@localhost/db",
    ],
)
def test_database_url_rejects_wrong_backend_or_driver(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    database_url: str,
) -> None:
    monkeypatch.delenv(
        "DATABASE_URL",
        raising=False,
    )

    env_path = tmp_path / ".env"
    env_path.write_text(
        f"DATABASE_URL={database_url}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        DatabaseConfigurationError,
    ):
        get_database_url(
            env_path=env_path
        )


def test_create_database_engine_uses_postgresql_psycopg() -> None:
    engine = create_database_engine(
        VALID_DATABASE_URL
    )

    try:
        assert engine.dialect.name == "postgresql"
        assert engine.dialect.driver == "psycopg"
        assert engine.pool._pre_ping is True
    finally:
        engine.dispose()
