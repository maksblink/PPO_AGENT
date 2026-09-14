"""Provision a dedicated PostgreSQL test database and isolate each test schema."""
from contextlib import contextmanager
from pathlib import Path
import os
import uuid

from dotenv import dotenv_values
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url


def resolve_test_database_url(root: Path):
    values = dotenv_values(root / ".env")
    application = os.environ.get("DATABASE_URL") or values.get("DATABASE_URL")
    explicit = os.environ.get("PPO_WALK_FORWARD_TEST_DATABASE_URL") or values.get("PPO_WALK_FORWARD_TEST_DATABASE_URL")
    if not explicit and not application:
        raise ValueError("Configure DATABASE_URL in .env or PPO_WALK_FORWARD_TEST_DATABASE_URL")
    source = make_url(explicit or application)
    if source.drivername != "postgresql+psycopg":
        raise ValueError("Test database requires postgresql+psycopg")
    target = source if explicit else source.set(database="ppo_agent_test")
    if not target.database or not target.database.endswith("_test"):
        raise ValueError("Test database name must end with _test")
    if application and make_url(application).database == target.database:
        raise ValueError("Test and application database names must differ")
    if explicit:
        return target  # Explicit targets must already exist.

    admin = create_engine(source.set(database="postgres"), isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as connection:
            connection.execute(text("SELECT pg_advisory_lock(724013, 1)"))
            try:
                if not connection.scalar(text("SELECT 1 FROM pg_database WHERE datname = 'ppo_agent_test'")):
                    connection.execute(text('CREATE DATABASE "ppo_agent_test"'))
            finally:
                connection.execute(text("SELECT pg_advisory_unlock(724013, 1)"))
    finally:
        admin.dispose()
    return target


@contextmanager
def isolated_test_schema(url):
    schema = "wf_test_" + uuid.uuid4().hex
    admin = create_engine(url, isolation_level="AUTOCOMMIT")
    engine = None
    created = False
    try:
        with admin.connect() as connection:
            connection.execute(text(f'CREATE SCHEMA "{schema}"'))
            created = True
        scoped = url.update_query_dict({"options": f"-csearch_path={schema}"})
        engine = create_engine(scoped, pool_pre_ping=True)
        yield engine, scoped
    finally:
        if engine is not None:
            engine.dispose()
        try:
            if created:
                with admin.connect() as connection:
                    connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
                    if connection.scalar(text("SELECT count(*) FROM pg_namespace WHERE nspname = :schema"), {"schema": schema}):
                        raise RuntimeError("Temporary test schema was not removed")
        finally:
            admin.dispose()
