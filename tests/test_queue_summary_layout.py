from pathlib import Path
import pytest
import run_search_queue as queue


def metadata(**updates):
    data = dict(seed=1, hidden_sizes=[384, 384, 384, 384], data={'train_range': {'start': 'a', 'end': 'b'}}, steps_per_epoch=100, epochs=1,
                source_run_id=None, source_steps=None)
    data.update(updates)
    return data


def test_chain_counts_selected_checkpoint_not_all_parent_training():
    snapshots = {1: metadata(epochs=5), 2: metadata(epochs=2, source_run_id=1, source_steps=50),
                 3: metadata(source_run_id=2, source_steps=100)}
    assert queue.resolve_chain_data_epochs(1, snapshots) == 5
    assert queue.resolve_chain_data_epochs(2, snapshots) == 2.5
    assert queue.resolve_chain_data_epochs(3, snapshots) == 2.5


@pytest.mark.parametrize('parent', [None, metadata(data={}), metadata(steps_per_epoch=200), metadata(source_run_id=2, source_steps=100)])
def test_incomplete_or_incomparable_lineage_is_unknown(parent):
    snapshots = {2: metadata(source_run_id=1, source_steps=100)}
    if parent is not None:
        snapshots[1] = parent
    assert queue.resolve_chain_data_epochs(2, snapshots) is None


def config(name):
    return queue.ConfigMeta(Path('unused.yml'), name, 7, 9, .0003, .99, .85, .0002, 0,
                            hidden_sizes=(384, 384, 384, 384))


def test_ranking_has_only_four_fields_and_uses_score_not_return(capsys):
    results = [queue.RunResult(1, config('high_return'), run_id=10, status='COMPLETED',
                              balanced_score=-.3, agent_return=.9, max_drawdown=-1.2),
               queue.RunResult(2, config('high_score'), run_id=11, status='COMPLETED',
                              balanced_score=.1, agent_return=.2, max_drawdown=-.1,
                              chain_data_epochs=3, always_long=.3, entropy_loss=-.2,
                              explained_variance=.6),
               queue.RunResult(3, config('failed'), run_id=12, status='FAILED', balanced_score=3),
               queue.RunResult(4, config('missing_final'), run_id=13, status='COMPLETED')]
    queue.print_summary(results, scope_label='TRAIN')
    output = capsys.readouterr().out
    table, ranking = output.split('RANKING BY FINAL balanced_score | TRAIN')
    assert 'WINNER' not in output
    for r in results:
        assert r.config.name not in table
    for column in ['seed', 'architecture', 'data_ep', 'ppo_ep', 'long', 'entropy', 'expl_var', 'status']:
        assert column in table
    assert '384x384x384x384' in table
    lines = [line for line in ranking.splitlines() if line.strip()]
    assert lines[0].split() == ["rank", "run", "name", "return", "maxDD"]
    assert set(lines[1]) == {"-"}
    assert [line.split() for line in lines[2:]] == [
        ["1", "#11", "high_score", "+20.00%", "-10.00%"],
        ["2", "#10", "high_return", "+90.00%", "-120.00%"],
    ]
    # Names start at the same column; numeric values end at the same columns.
    assert lines[2].index("high_score") == lines[3].index("high_return")
    assert lines[2].index("%") == lines[3].index("%")
    assert len(lines[2]) == len(lines[3]) == len(lines[0])


def test_database_lineage_includes_parent_outside_selected_queue(monkeypatch):
    import json
    from types import SimpleNamespace
    def row(id, name, snapshot):
        return '\t'.join([str(id), name, 'completed', '10'] + ['0'] * 13 + [json.dumps(snapshot)])
    stdout = '\n'.join([row(1, 'external_parent', metadata(epochs=4)),
                        row(2, 'child', metadata(source_run_id=1, source_steps=50))])
    monkeypatch.setattr(queue.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=0, stdout=stdout))
    result = queue.load_results_from_database([config('child')])[0]
    assert result.chain_data_epochs == 1.5
    assert result.seed == 1
    assert result.hidden_sizes == (384,384,384,384)
