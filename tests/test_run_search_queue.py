from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import run_search_queue as queue


def test_parse_queue_selection_expands_ranges_and_deduplicates() -> None:
    assert queue.parse_queue_selection("5,2-4,3,8") == (
        2,
        3,
        4,
        5,
        8,
    )


@pytest.mark.parametrize(
    "value",
    ["", "0", "3-1", "1-2-3", "2-", "1,,3", "abc"],
)
def test_parse_queue_selection_rejects_invalid_values(
    value: str,
) -> None:
    with pytest.raises(Exception):
        queue.parse_queue_selection(value)


def test_resolve_queue_selection_preserves_manifest_order() -> None:
    configs = ["a", "b", "c", "d", "e"]

    selected = queue.resolve_queue_selection(
        configs,
        start_at=None,
        selected_positions=(2, 4, 5),
    )

    assert selected == [
        (2, "b"),
        (4, "d"),
        (5, "e"),
    ]

    resumed = queue.resolve_queue_selection(
        configs,
        start_at=3,
        selected_positions=None,
    )

    assert resumed == [
        (3, "c"),
        (4, "d"),
        (5, "e"),
    ]


def test_queue_manifest_loading(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    queue_directory = tmp_path / "configs" / "search_queues"
    queue_directory.mkdir(parents=True)

    manifest = queue_directory / "test_queue.yml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "queue_schema_version": 1,
                "name": "test_queue",
                "configs": [
                    "configs/experiments/test/01.yml",
                    "configs/experiments/test/02.yml",
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(queue, "ROOT", tmp_path)
    monkeypatch.setattr(
        queue,
        "QUEUE_DIRECTORY",
        queue_directory,
    )
    monkeypatch.setattr(queue, "DEFAULT_QUEUE", "test_queue")

    assert queue.discover_queue_manifests() == {
        "test_queue": manifest,
    }
    assert queue.load_queue_config_paths("test_queue") == [
        "configs/experiments/test/01.yml",
        "configs/experiments/test/02.yml",
    ]


def test_run_config_inherits_terminal_streams(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FakeProcess:
        def wait(self, timeout=None):
            return 0

        def poll(self):
            return 0

    calls = []

    def fake_popen(command, **kwargs):
        calls.append((command, kwargs))
        return FakeProcess()

    monkeypatch.setattr(queue.subprocess, "Popen", fake_popen)

    config = queue.ConfigMeta(
        path=queue.ROOT / "configs" / "test.yml",
        name="test_run",
        seed=7,
        n_epochs=3,
        learning_rate=0.0003,
        gamma=0.9,
        gae_lambda=0.85,
        ent_coef=0.0002,
        exposure_penalty=0.0,
    )

    result = queue.run_config(
        queue_name="test_queue",
        queue_position=4,
        queue_total=10,
        selected_index=2,
        selected_total=3,
        config=config,
    )

    output = capsys.readouterr().out
    command, kwargs = calls[0]

    assert "PPO QUEUE PROGRESS [2/3]" in output
    assert "Queue position: 4/10" in output
    assert "Remaining after this run: 1" in output
    assert command == [
        queue.sys.executable,
        "-m",
        "train_and_eval.training",
        "--config",
        "configs/test.yml",
    ]
    assert kwargs == {"cwd": queue.ROOT}
    assert result.status == "COMPLETED"


def test_find_existing_selected_runs_uses_database_status() -> None:
    first = queue.ConfigMeta(
        path=queue.ROOT / "configs" / "first.yml",
        name="first",
        seed=1,
        n_epochs=3,
        learning_rate=0.0003,
        gamma=0.9,
        gae_lambda=0.85,
        ent_coef=0.0002,
        exposure_penalty=0.0,
    )
    second = queue.ConfigMeta(
        path=queue.ROOT / "configs" / "second.yml",
        name="second",
        seed=2,
        n_epochs=3,
        learning_rate=0.0003,
        gamma=0.9,
        gae_lambda=0.85,
        ent_coef=0.0002,
        exposure_penalty=0.0,
    )

    completed = queue.RunResult(
        queue_index=1,
        config=first,
        status="COMPLETED",
        run_id=91,
    )
    missing = queue.RunResult(
        queue_index=2,
        config=second,
        status="NOT_FOUND",
    )

    assert queue.find_existing_selected_runs(
        [(1, first), (2, second)],
        [completed, missing],
    ) == [(1, first, completed)]


def test_main_preflight_blocks_existing_run_before_training(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = queue.ConfigMeta(
        path=queue.ROOT / "configs" / "existing.yml",
        name="existing",
        seed=1,
        n_epochs=3,
        learning_rate=0.0003,
        gamma=0.9,
        gae_lambda=0.85,
        ent_coef=0.0002,
        exposure_penalty=0.0,
    )
    existing = queue.RunResult(
        queue_index=1,
        config=config,
        status="COMPLETED",
        run_id=84,
    )

    monkeypatch.setattr(
        queue,
        "parse_args",
        lambda: type(
            "Args",
            (),
            {
                "queue": "test",
                "start_at": None,
                "select": (1,),
                "summary_only": False,
                "dry_run": False,
                "skip_existing": False,
            },
        )(),
    )
    monkeypatch.setattr(
        queue,
        "load_queue_config_paths",
        lambda name: ["configs/existing.yml"],
    )
    monkeypatch.setattr(
        queue,
        "read_config_meta",
        lambda path: config,
    )
    monkeypatch.setattr(queue, "assert_git_clean", lambda: None)
    monkeypatch.setattr(
        queue,
        "load_results_from_database",
        lambda configs: [existing],
    )

    def fail_if_started(**kwargs):
        raise AssertionError("training must not start")

    monkeypatch.setattr(queue, "run_config", fail_if_started)

    assert queue.main() == 2
    output = capsys.readouterr().out
    assert "QUEUE PREFLIGHT FAILED" in output
    assert "run=#84" in output
