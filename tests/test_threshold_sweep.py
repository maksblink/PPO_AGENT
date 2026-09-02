from __future__ import annotations

from types import SimpleNamespace

import pytest

import threshold_sweep as sweep


class FakeScalarResult:
    def __init__(self, value):
        self.value = value

    def one_or_none(self):
        return self.value


class FakeSession:
    def __init__(self, values=()):
        self.values = list(values)
        self.statements = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def scalars(self, statement):
        self.statements.append(statement)
        value = (
            self.values.pop(0)
            if self.values
            else None
        )
        return FakeScalarResult(value)


def _checkpoint(
    *,
    checkpoint_id: int = 338,
):
    run = SimpleNamespace(
        id=80,
        name="nq5m-test",
        seed=3,
    )
    return SimpleNamespace(
        id=checkpoint_id,
        run_id=80,
        run_step=718_848,
        model_step=718_848,
        save_reason="periodic",
        run=run,
    )


def test_parser_accepts_checkpoint_ids() -> None:
    args = sweep.build_parser().parse_args(
        ["--checkpoint-ids", "338,342"]
    )

    assert args.runs is None
    assert args.checkpoint_ids == "338,342"


def test_parser_rejects_runs_with_checkpoint_ids() -> None:
    parser = sweep.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--runs",
                "80",
                "--checkpoint-ids",
                "338",
            ]
        )


def test_load_source_checkpoints_preserves_metadata() -> None:
    checkpoint = _checkpoint()
    session = FakeSession([checkpoint])

    sources = sweep.load_source_checkpoints(
        lambda: session,
        [338],
    )

    assert len(session.statements) == 1
    assert sources == [
        sweep.SourceRun(
            run_id=80,
            name="nq5m-test",
            seed=3,
            checkpoint_id=338,
            checkpoint_run_step=718_848,
            checkpoint_model_step=718_848,
            checkpoint_save_reason="periodic",
        )
    ]


def test_load_source_checkpoints_raises_when_missing() -> None:
    session = FakeSession()

    with pytest.raises(
        RuntimeError,
        match="Checkpoint #999 does not exist",
    ):
        sweep.load_source_checkpoints(
            lambda: session,
            [999],
        )


def test_load_source_runs_includes_checkpoint_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = _checkpoint(checkpoint_id=340)
    checkpoint.save_reason = "final"
    run = checkpoint.run
    run.checkpoints = [checkpoint]

    monkeypatch.setattr(
        sweep,
        "load_run",
        lambda session, *, run_id: run,
    )

    sources = sweep.load_source_runs(
        lambda: FakeSession(),
        [80],
    )

    assert sources[0].checkpoint_id == 340
    assert sources[0].checkpoint_run_step == 718_848
    assert sources[0].checkpoint_save_reason == "final"
