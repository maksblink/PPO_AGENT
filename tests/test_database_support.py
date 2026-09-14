from unittest.mock import MagicMock

import pytest
from sqlalchemy.engine import make_url

from tests import database_support as support


@pytest.fixture
def environment(monkeypatch, tmp_path):
    monkeypatch.delenv("PPO_WALK_FORWARD_TEST_DATABASE_URL", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://user:password@localhost/ppo_agent")
    return tmp_path


def test_default_provisions_only_test_database(monkeypatch, environment):
    engine = MagicMock()
    connection = engine.connect.return_value.__enter__.return_value
    connection.scalar.return_value = None
    factory = MagicMock(return_value=engine)
    monkeypatch.setattr(support, "create_engine", factory)
    target = support.resolve_test_database_url(environment)
    assert target.database == "ppo_agent_test"
    assert factory.call_args.args[0].database == "postgres"
    statements = [str(c.args[0]) for c in connection.execute.call_args_list]
    assert 'CREATE DATABASE "ppo_agent_test"' in statements
    assert "SELECT pg_advisory_unlock(724013, 1)" in statements
    engine.dispose.assert_called_once()


def test_existing_database_is_reused(monkeypatch, environment):
    engine = MagicMock()
    engine.connect.return_value.__enter__.return_value.scalar.return_value = 1
    monkeypatch.setattr(support, "create_engine", MagicMock(return_value=engine))
    support.resolve_test_database_url(environment)
    statements = [str(c.args[0]) for c in engine.connect.return_value.__enter__.return_value.execute.call_args_list]
    assert not any("CREATE DATABASE" in s for s in statements)


@pytest.mark.parametrize("database", ["ppo_agent", "postgres"])
def test_rejects_application_or_non_test_target(monkeypatch, environment, database):
    monkeypatch.setenv("PPO_WALK_FORWARD_TEST_DATABASE_URL", f"postgresql+psycopg://user@localhost/{database}")
    factory = MagicMock()
    monkeypatch.setattr(support, "create_engine", factory)
    with pytest.raises(ValueError):
        support.resolve_test_database_url(environment)
    factory.assert_not_called()


def test_explicit_target_from_env_file_is_not_provisioned(monkeypatch, environment):
    (environment / ".env").write_text("PPO_WALK_FORWARD_TEST_DATABASE_URL=postgresql+psycopg://user@localhost/custom_test\n")
    factory = MagicMock()
    monkeypatch.setattr(support, "create_engine", factory)
    assert support.resolve_test_database_url(environment).database == "custom_test"
    factory.assert_not_called()


def test_rejects_same_application_test_database(monkeypatch, environment):
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://user@localhost/ppo_agent_test")
    with pytest.raises(ValueError, match="must differ"):
        support.resolve_test_database_url(environment)


@pytest.mark.parametrize("fails", [False, True])
def test_schema_cleanup_after_success_or_failure(monkeypatch, fails):
    admin, engine = MagicMock(), MagicMock()
    connection = admin.connect.return_value.__enter__.return_value
    connection.scalar.return_value = 0
    monkeypatch.setattr(support, "create_engine", MagicMock(side_effect=[admin, engine]))
    def run():
        with support.isolated_test_schema(make_url("postgresql+psycopg://user@localhost/ppo_agent_test")) as (_, scoped):
            assert scoped.query["options"].startswith("-csearch_path=wf_test_")
            if fails:
                raise RuntimeError("simulated migration failure")
    if fails:
        with pytest.raises(RuntimeError, match="simulated"):
            run()
    else:
        run()
    statements = [str(c.args[0]) for c in connection.execute.call_args_list]
    schema = statements[0].split('"')[1]
    assert statements == [f'CREATE SCHEMA "{schema}"', f'DROP SCHEMA "{schema}" CASCADE']
    engine.dispose.assert_called_once()
    admin.dispose.assert_called_once()
