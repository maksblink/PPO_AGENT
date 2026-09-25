from pathlib import Path
from types import SimpleNamespace
import pytest
import run_search_queue as queue


@pytest.mark.parametrize('choice', ['YES', 'SKIP', 'STOP'])
def test_prompt_repeats_until_exact_choice(monkeypatch, capsys, choice):
    answers = iter(['yes', '', ' YES ', 'skip', 'STOP ', choice])
    monkeypatch.setattr('builtins.input', lambda prompt: next(answers))
    assert queue.prompt_restart_choice() == choice
    assert capsys.readouterr().out.count('Invalid answer') == 5


@pytest.mark.parametrize('error', [EOFError, KeyboardInterrupt])
def test_input_interruption_preserves_run(monkeypatch, error):
    def read(prompt): raise error()
    monkeypatch.setattr('builtins.input', read)
    assert queue.prompt_restart_choice() == 'STOP'


@pytest.mark.parametrize('choice', ['YES', 'SKIP', 'STOP'])
def test_only_yes_calls_cleanup(monkeypatch, choice):
    from train_and_eval.database import session
    calls = []
    engine = SimpleNamespace(dispose=lambda: calls.append('dispose'))
    monkeypatch.setattr(session, 'create_database_engine', lambda: engine)
    snapshot = {'id': 9}
    monkeypatch.setattr(queue, 'inspect_incomplete_run', lambda *a: snapshot)
    monkeypatch.setattr(queue, 'prompt_restart_choice', lambda: choice)
    monkeypatch.setattr(queue, 'delete_incomplete_run', lambda *a: calls.append('delete'))
    config = SimpleNamespace(name='example', path=Path('example.yml'))
    assert queue.handle_incomplete_run(config, 9) == choice
    assert calls == (['delete', 'dispose'] if choice == 'YES' else ['dispose'])


@pytest.mark.parametrize('choice,expected', [('YES', ['failed', 'new']), ('SKIP', ['new']), ('STOP', [])])
def test_queue_processes_existing_runs_in_order(monkeypatch, choice, expected):
    configs = [SimpleNamespace(name=name) for name in ('done', 'failed', 'new')]
    monkeypatch.setattr(queue, 'parse_args', lambda: SimpleNamespace(queue='example', start_at=None, select=None, summary_only=False, dry_run=False))
    monkeypatch.setattr(queue, 'load_queue_config_paths', lambda name: configs)
    monkeypatch.setattr(queue, 'read_config_meta', lambda c: c)
    monkeypatch.setattr(queue, 'assert_git_clean', lambda: None)
    monkeypatch.setattr(queue, 'print_queue', lambda *a: None)
    monkeypatch.setattr(queue, 'print_train_then_val_summaries', lambda *a: None)
    statuses = {'done': 'COMPLETED', 'failed': 'FAILED', 'new': 'NOT_FOUND'}
    def load(items):
        return [queue.RunResult(i, c, status=statuses[c.name], run_id=i) for i,c in enumerate(items, 1)]
    monkeypatch.setattr(queue, 'load_results_from_database', load)
    handled = []
    def decide(c, run_id):
        handled.append(c.name)
        return choice
    monkeypatch.setattr(queue, 'handle_incomplete_run', decide)
    started = []
    def run(**kwargs):
        c = kwargs['config']; started.append(c.name)
        return queue.RunResult(1, c, status='COMPLETED')
    monkeypatch.setattr(queue, 'run_config', run)
    assert queue.main() == 0
    assert handled == ['failed']
    assert started == expected


def test_no_legacy_skip_existing_bypass():
    with pytest.raises(SystemExit):
        queue.parse_args(['--queue', 'example', '--skip-existing'])


def test_cleanup_failure_never_starts_training(monkeypatch):
    config = SimpleNamespace(name='failed')
    monkeypatch.setattr(queue, 'parse_args', lambda: SimpleNamespace(queue='example', start_at=None, select=None, summary_only=False, dry_run=False))
    monkeypatch.setattr(queue, 'load_queue_config_paths', lambda name: [config])
    monkeypatch.setattr(queue, 'read_config_meta', lambda c: c)
    monkeypatch.setattr(queue, 'assert_git_clean', lambda: None)
    monkeypatch.setattr(queue, 'print_queue', lambda *a: None)
    monkeypatch.setattr(queue, 'print_train_then_val_summaries', lambda *a: None)
    monkeypatch.setattr(queue, 'load_results_from_database', lambda configs: [queue.RunResult(1, config, status='FAILED', run_id=9)])
    def blocked(*a): raise ValueError('Retained descendant')
    monkeypatch.setattr(queue, 'handle_incomplete_run', blocked)
    monkeypatch.setattr(queue, 'run_config', lambda **k: pytest.fail('must not train'))
    assert queue.main() == 1


def test_queue_does_not_authorize_unrelated_pending_cleanup(monkeypatch):
    from extra_tools import clean_training
    def cleanup(*args, **kwargs):
        kwargs['confirm']('Recover the interrupted cleanup first? Type RECOVER: ')
    monkeypatch.setattr(clean_training, 'cleanup', cleanup)
    with pytest.raises(ValueError, match='Pending cleanup'):
        queue.delete_incomplete_run(None, SimpleNamespace(name='example'), {'id': 9})
