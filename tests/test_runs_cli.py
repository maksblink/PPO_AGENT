from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from io import StringIO
from types import SimpleNamespace

import pytest

import train_and_eval.runs.cli as cli
from train_and_eval.database.models import (
    Checkpoint,
    CheckpointSaveReason,
    ContinuationMode,
    Evaluation,
    EvaluationDataScope,
    EvaluationPolicyMode,
    EvaluationStatus,
    EvaluationTrigger,
    Run,
    RunStatus,
    TrainingDurationUnit,
)
from train_and_eval.runs.formatting import (
    best_evaluation,
)
from train_and_eval.runs.queries import (
    RunRegistryNotFoundError,
    load_run,
    load_runs,
)


NOW = datetime(2026, 8, 2, 16, 0, tzinfo=timezone.utc)


class FakeEngine:
    def __init__(self) -> None:
        self.dispose_calls = 0

    def dispose(self) -> None:
        self.dispose_calls += 1


class FakeSession:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


class FakeScalarResult:
    def __init__(self, values):
        self.values = list(values)

    def all(self):
        return list(self.values)

    def one_or_none(self):
        if not self.values:
            return None
        if len(self.values) != 1:
            raise RuntimeError("multiple rows")
        return self.values[0]


class QuerySession:
    def __init__(self, values):
        self.values = values
        self.statements = []

    def scalars(self, statement):
        self.statements.append(statement)
        return FakeScalarResult(self.values)


def _config() -> dict:
    return {
        "config_schema_version": 1,
        "run": {"name": "registry-test", "seed": 7},
        "continuation": {"mode": "fresh"},
        "data": {"path": "data/test.parquet", "train_ratio": 0.8},
        "training": {
            "duration_unit": "timesteps",
            "duration_amount": 100,
        },
        "environment": {
            "window": 40,
            "context": "baseline_multiscale_v1",
            "position_side": "long_only",
            "stake_pln": 1000.0,
            "fee_bps": 1.0,
            "swap_bps": 3.0,
            "reward_scale": 1.0,
            "exposure_penalty": 0.0,
            "turnover_penalty": 0.0,
            "drawdown_penalty": 0.0,
            "profit_reward_mult": 1.0,
            "loss_reward_mult": 1.0,
        },
        "ppo": {
            "policy": "mlp",
            "device": "cuda",
            "hidden_sizes": [64, 64],
            "activation": "tanh",
            "n_steps": 8,
            "batch_size": 4,
            "n_epochs": 1,
            "learning_rate": 0.0003,
            "gamma": 0.9,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.0002,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "target_kl": None,
        },
        "evaluation": {
            "eval_every_steps": 50,
            "checkpoint_every_steps": 50,
            "policy_mode": "deterministic_argmax",
            "threshold_action": None,
            "probability_threshold": None,
            "best_metric": "balanced_score",
            "early_stop_patience_evals": 5,
        },
    }


def _evaluation(
    *,
    evaluation_id: int,
    checkpoint_id: int,
    trigger: EvaluationTrigger,
    score: float,
    status: EvaluationStatus = EvaluationStatus.COMPLETED,
    data_scope: EvaluationDataScope = EvaluationDataScope.RUN_VALIDATION,
) -> Evaluation:
    evaluation = Evaluation(
        checkpoint_id=checkpoint_id,
        status=status,
        trigger=trigger,
        policy_mode=EvaluationPolicyMode.DETERMINISTIC_ARGMAX,
        threshold_action=None,
        probability_threshold=None,
        seed=7,
        data_scope=data_scope,
        data_path="data/test.parquet",
        data_sha256="d" * 64,
        data_rows=2000,
        evaluation_start_index=1600,
        evaluation_end_index=2000,
        evaluation_start_at=NOW,
        evaluation_end_at=NOW,
        lookback_rows=6145,
        steps_expected=400,
        steps_completed=(400 if status == EvaluationStatus.COMPLETED else 20),
        agent_return=0.25,
        always_long_return=0.20,
        always_short_return=-0.30,
        agent_vs_always_long_return=0.05,
        balanced_score=score,
        agent_max_drawdown=-0.10,
        drawdown_improvement=0.03,
        market_exposure=0.55,
        net_exposure=0.25,
        profit_factor=1.4,
        win_rate=0.6,
        round_trips=42,
        created_at=NOW,
        started_at=NOW,
        finished_at=(NOW if status == EvaluationStatus.COMPLETED else None),
        git_commit="a" * 40,
        git_branch="master",
    )
    evaluation.id = evaluation_id
    return evaluation


def _run() -> Run:
    run = Run(
        name="registry-test",
        status=RunStatus.COMPLETED,
        continuation_mode=ContinuationMode.FRESH,
        source_checkpoint_id=None,
        description="test notes",
        created_at=NOW,
        modified_at=NOW,
        started_at=NOW,
        finished_at=NOW,
        git_commit="a" * 40,
        git_branch="master",
        config_schema_version=1,
        seed=7,
        config_sha256="b" * 64,
        normalized_config_sha256="c" * 64,
        raw_config_yaml="config_schema_version: 1\n",
        normalized_config_json=_config(),
        data_path="data/test.parquet",
        data_sha256="d" * 64,
        duration_unit=TrainingDurationUnit.TIMESTEPS,
        duration_amount=100,
        split_index=1600,
        train_rows=1600,
        validation_rows=400,
        steps_per_data_epoch=1000,
        training_steps_requested=100,
        training_steps_completed=100,
        data_epochs_completed=Decimal("0.10000000"),
        stopped_early=False,
        early_stop_reason=None,
        error_type=None,
        error_message=None,
    )
    run.id = 51

    best_cp = Checkpoint(
        run_id=51,
        run_step=50,
        model_step=50,
        save_reason=CheckpointSaveReason.PERIODIC,
        relative_path=(
            "artifacts/runs/00000051/checkpoints/"
            "step_000000000050_periodic.zip"
        ),
        sha256="e" * 64,
        size_bytes=100_000,
        created_at=NOW,
    )
    best_cp.id = 91
    best_cp.evaluations = [
        _evaluation(
            evaluation_id=301,
            checkpoint_id=91,
            trigger=EvaluationTrigger.SCHEDULED,
            score=0.30,
        )
    ]

    final_cp = Checkpoint(
        run_id=51,
        run_step=100,
        model_step=100,
        save_reason=CheckpointSaveReason.FINAL,
        relative_path=(
            "artifacts/runs/00000051/checkpoints/"
            "step_000000000100_final.zip"
        ),
        sha256="f" * 64,
        size_bytes=120_000,
        created_at=NOW,
    )
    final_cp.id = 92
    final_cp.evaluations = [
        _evaluation(
            evaluation_id=302,
            checkpoint_id=92,
            trigger=EvaluationTrigger.FINAL,
            score=0.20,
        )
    ]

    run.checkpoints = [best_cp, final_cp]
    return run


def _execute(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
):
    engine = FakeEngine()
    session = FakeSession()
    monkeypatch.setattr(cli, "create_database_engine", lambda: engine)
    monkeypatch.setattr(
        cli,
        "create_session_factory",
        lambda received: (
            (lambda: session)
            if received is engine
            else pytest.fail("wrong engine")
        ),
    )
    args = cli.build_parser().parse_args(argv)
    stdout = StringIO()
    stderr = StringIO()
    code = cli.execute_runs_cli(
        args,
        stdout=stdout,
        stderr=stderr,
    )
    return code, stdout.getvalue(), stderr.getvalue(), engine


def test_list_command_passes_filters_and_prints_best_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fake_load_runs(session, **kwargs):
        calls.append((session, kwargs))
        return [_run()]

    monkeypatch.setattr(cli, "load_runs", fake_load_runs)
    code, output, errors, engine = _execute(
        monkeypatch,
        [
            "list",
            "--limit",
            "5",
            "--status",
            "completed",
            "--continuation",
            "fresh",
            "--name",
            "registry",
        ],
    )

    assert code == 0
    assert errors == ""
    assert engine.dispose_calls == 1
    assert calls[0][1] == {
        "status": RunStatus.COMPLETED,
        "continuation_mode": ContinuationMode.FRESH,
        "name_contains": "registry",
        "stopped_early": False,
        "limit": 5,
    }
    assert "registry-test" in output
    assert "100/100" in output
    assert "0.30000000" in output
    assert "50" in output


def test_show_command_separates_best_and_final_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "load_run", lambda session, run_id: _run())
    code, output, errors, _ = _execute(
        monkeypatch,
        ["show", "--run-id", "51"],
    )

    assert code == 0
    assert errors == ""
    assert "===== REPRODUCIBILITY =====" in output
    assert "===== DATA =====" in output
    assert "Context history:" in output
    assert "6,145 rows" in output
    assert "===== BEST RESULT =====" in output
    assert "Checkpoint ID:" in output
    assert "0.30000000" in output
    assert "===== FINAL RESULT =====" in output
    assert "0.20000000" in output
    assert "step_000000000100_final.zip" in output


def test_checkpoints_command_marks_best_and_final(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "load_run", lambda session, run_id: _run())
    code, output, errors, _ = _execute(
        monkeypatch,
        ["checkpoints", "--run-id", "51"],
    )

    assert code == 0
    assert errors == ""
    assert "RUN STEP" in output
    assert "periodic" in output
    assert "final" in output
    assert "best" in output
    assert "0.30000000" in output


def test_evaluations_command_prints_compact_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "load_run", lambda session, run_id: _run())
    code, output, errors, _ = _execute(
        monkeypatch,
        ["evaluations", "--run-id", "51"],
    )

    assert code == 0
    assert errors == ""
    assert "TRIGGER" in output
    assert "scheduled" in output
    assert "final" in output
    assert "0.25000000" in output
    assert "42" in output
    assert "400/400" in output


def test_evaluations_full_prints_every_persisted_column(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "load_run", lambda session, run_id: _run())
    code, output, errors, _ = _execute(
        monkeypatch,
        ["evaluations", "--run-id", "51", "--full"],
    )

    assert code == 0
    assert errors == ""
    assert "===== EVALUATION 301 =====" in output
    assert "checkpoint_run_step:" in output
    assert "profit_factor:" in output
    assert "current_streak_type:" in output
    assert "error_message:" in output


def test_query_failure_is_readable_and_disposes_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(session, run_id):
        raise RunRegistryNotFoundError(
            f"Run {run_id} does not exist."
        )

    monkeypatch.setattr(cli, "load_run", fail)
    code, output, errors, engine = _execute(
        monkeypatch,
        ["show", "--run-id", "999"],
    )

    assert code == 1
    assert output == ""
    assert engine.dispose_calls == 1
    assert "Run registry query: FAILED" in errors
    assert "RunRegistryNotFoundError" in errors
    assert "Run 999 does not exist." in errors


def test_best_evaluation_ignores_failed_and_non_validation_rows() -> None:
    run = _run()
    run.checkpoints[0].evaluations.extend(
        [
            _evaluation(
                evaluation_id=303,
                checkpoint_id=91,
                trigger=EvaluationTrigger.MANUAL,
                score=100.0,
                status=EvaluationStatus.FAILED,
            ),
            _evaluation(
                evaluation_id=304,
                checkpoint_id=91,
                trigger=EvaluationTrigger.MANUAL,
                score=200.0,
                data_scope=EvaluationDataScope.CUSTOM_RANGE,
            ),
        ]
    )

    selected = best_evaluation(run)

    assert selected is not None
    assert selected.id == 301
    assert selected.balanced_score == 0.30


def test_load_runs_uses_scalar_result() -> None:
    run = _run()
    session = QuerySession([run])

    result = load_runs(
        session,
        status=RunStatus.COMPLETED,
        continuation_mode=ContinuationMode.FRESH,
        name_contains="registry",
        stopped_early=True,
        limit=3,
    )

    assert result == [run]
    assert len(session.statements) == 1


def test_load_run_raises_when_missing() -> None:
    session = QuerySession([])

    with pytest.raises(
        RunRegistryNotFoundError,
        match="Run 123 does not exist",
    ):
        load_run(session, run_id=123)


def test_parser_rejects_nonpositive_identifiers() -> None:
    parser = cli.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["show", "--run-id", "0"])

    with pytest.raises(SystemExit):
        parser.parse_args(["list", "--limit", "0"])
