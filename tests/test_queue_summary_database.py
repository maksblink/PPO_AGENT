"""Exercise the summary SQL against the project's isolated PostgreSQL fixture."""
from pathlib import Path
from types import SimpleNamespace
import pytest
from sqlalchemy import text, update
from tests.test_clean_training import cleanup_database, seed
from train_and_eval.database.models import Run
import run_search_queue as queue


@pytest.mark.parametrize('scope', ['run_training', 'run_validation'])
def test_summary_query_loads_identity_and_external_ancestor(cleanup_database, tmp_path, monkeypatch, scope):
    engine = cleanup_database
    ids = seed(engine, tmp_path)
    with engine.begin() as connection:
        connection.execute(update(Run).values(normalized_config_json={
            'ppo': {'hidden_sizes': [384,384,384,384]}, 'data': {'path': 'data/source.parquet'},
        }))
    def query(command, **kwargs):
        with engine.connect() as connection:
            rows = connection.execute(text(command[-1])).all()
        stdout = '\n'.join('\t'.join(queue.DB_NULL if x is None else str(x) for x in row) for row in rows)
        return SimpleNamespace(returncode=0, stdout=stdout, stderr='')
    monkeypatch.setattr(queue.subprocess, 'run', query)
    config = queue.ConfigMeta(Path('unused.yml'), 'second', 9, 3, .0003, .99, .85, .0002, 0)
    result = queue.load_results_from_database([config], data_scope=scope)[0]
    assert result.run_id == ids['second']
    assert result.seed == 1
    assert result.hidden_sizes == (384,384,384,384)
    assert result.chain_data_epochs == 0
    assert result.balanced_score is None
