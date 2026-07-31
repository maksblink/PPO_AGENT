from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"


class DatabaseConfigurationError(RuntimeError):
    """Raised when database configuration is missing or invalid."""


class DatabaseConnectionError(RuntimeError):
    """Raised when PostgreSQL cannot be reached."""


@dataclass(frozen=True)
class DatabaseConnectionInfo:
    """Basic information returned by a successful connection check."""

    database: str
    user: str
    server_version: str


def get_database_url(
    *,
    env_path: str | Path = DEFAULT_ENV_PATH,
) -> str:
    """
    Read DATABASE_URL.

    A value exported in the process environment takes precedence.
    Otherwise, DATABASE_URL is read from the project's local .env file.
    """
    database_url = os.environ.get("DATABASE_URL")

    if database_url is None:
        path = Path(env_path).expanduser().resolve()

        if path.is_file():
            values = dotenv_values(path)
            configured_value = values.get("DATABASE_URL")

            if isinstance(configured_value, str):
                database_url = configured_value

    if not database_url:
        raise DatabaseConfigurationError(
            "DATABASE_URL is not configured. "
            f"Expected it in the environment or in: {env_path}"
        )

    try:
        parsed_url = make_url(database_url)
    except Exception as error:
        raise DatabaseConfigurationError(
            "DATABASE_URL is not a valid SQLAlchemy URL."
        ) from error

    if parsed_url.get_backend_name() != "postgresql":
        raise DatabaseConfigurationError(
            "DATABASE_URL must use PostgreSQL."
        )

    if parsed_url.get_driver_name() != "psycopg":
        raise DatabaseConfigurationError(
            "DATABASE_URL must use the psycopg driver: "
            "postgresql+psycopg://..."
        )

    return database_url


def create_database_engine(
    database_url: str | None = None,
) -> Engine:
    """
    Create a synchronous SQLAlchemy engine.

    pool_pre_ping verifies pooled connections before reusing them.
    No database connection is opened until the engine is first used.
    """
    resolved_url = (
        database_url
        if database_url is not None
        else get_database_url()
    )

    return create_engine(
        resolved_url,
        pool_pre_ping=True,
        hide_parameters=True,
    )


def create_session_factory(
    engine: Engine,
) -> sessionmaker[Session]:
    return sessionmaker(
        bind=engine,
        class_=Session,
        autoflush=False,
        expire_on_commit=False,
    )


@contextmanager
def database_session(
    engine: Engine,
) -> Iterator[Session]:
    """
    Provide a transactional database session.

    Successful work is committed. Exceptions cause a rollback.
    """
    factory = create_session_factory(engine)

    with factory() as session:
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise


def check_database_connection(
    engine: Engine | None = None,
) -> DatabaseConnectionInfo:
    """
    Open a real connection and query basic PostgreSQL information.
    """
    owns_engine = engine is None
    active_engine = (
        create_database_engine()
        if engine is None
        else engine
    )

    try:
        with active_engine.connect() as connection:
            row = (
                connection.execute(
                    text(
                        """
                        SELECT
                            current_database() AS database_name,
                            current_user AS user_name,
                            current_setting(
                                'server_version'
                            ) AS server_version
                        """
                    )
                )
                .mappings()
                .one()
            )

        return DatabaseConnectionInfo(
            database=str(row["database_name"]),
            user=str(row["user_name"]),
            server_version=str(row["server_version"]),
        )

    except SQLAlchemyError as error:
        raise DatabaseConnectionError(
            "Could not connect to the PPO_AGENT PostgreSQL database."
        ) from error

    finally:
        if owns_engine:
            active_engine.dispose()


def main() -> int:
    info = check_database_connection()

    print("Database connection: OK")
    print(f"Database: {info.database}")
    print(f"User: {info.user}")
    print(f"PostgreSQL: {info.server_version}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
