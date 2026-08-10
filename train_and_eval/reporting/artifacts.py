from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import select

from train_and_eval.database.models import Checkpoint, Evaluation, EvaluationStatus, Run, TrainingMetric
from train_and_eval.evaluation.runner import EvaluationRunResult


class ReportArtifactError(RuntimeError):
    """Raised when rebuildable report artifacts cannot be produced."""


def run_artifact_root(project_root: str | Path, artifacts_directory: str | Path, run_id: int) -> Path:
    root = Path(project_root).expanduser().resolve()
    artifacts = Path(artifacts_directory)
    if artifacts.is_absolute():
        raise ReportArtifactError("artifacts_directory must be relative")
    target = (root / artifacts / "runs" / f"{int(run_id):08d}").resolve()
    target.relative_to(root)
    return target


def evaluation_artifact_directory(project_root: str | Path, artifacts_directory: str | Path, run_id: int, evaluation_id: int) -> Path:
    return run_artifact_root(project_root, artifacts_directory, run_id) / "evaluations" / f"{int(evaluation_id):08d}"


def _trajectory_frame(result: EvaluationRunResult) -> pd.DataFrame:
    trajectory = result.trajectory
    frame = pd.DataFrame({
        "execution_index": trajectory.execution_indices,
        "timestamp": trajectory.execution_timestamps,
        "execution_price": trajectory.execution_prices,
        "close_price": trajectory.close_prices,
        "action": trajectory.actions,
        "position": trajectory.positions,
        "agent_equity": trajectory.agent_equity,
        "always_long_equity": trajectory.always_long_equity,
        "always_short_equity": trajectory.always_short_equity,
        "shaped_reward": trajectory.shaped_rewards,
        "fee_cost": trajectory.fee_costs,
        "swap_cost": trajectory.swap_costs,
        "trade_cost": trajectory.trade_costs,
        "drawdown": trajectory.drawdowns,
        "hold_bars": trajectory.hold_bars,
        "selected_action_probability": trajectory.selected_action_probabilities,
    })
    frame["cumulative_fee_cost"] = frame["fee_cost"].cumsum()
    frame["cumulative_swap_cost"] = frame["swap_cost"].cumsum()
    frame["cumulative_trade_cost"] = frame["trade_cost"].cumsum()
    return frame


def _trade_events_frame(result: EvaluationRunResult) -> pd.DataFrame:
    if not result.trade_events:
        return pd.DataFrame(
            {
                "event": pd.Series(dtype="string"),
                "timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
            }
        )
    return pd.DataFrame(list(result.trade_events))


def persist_evaluation_source_artifacts(
    result: EvaluationRunResult,
    *,
    project_root: str | Path,
    artifacts_directory: str | Path,
    run_id: int,
    evaluation_id: int,
    checkpoint_id: int,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Persist lossless inputs used by all detailed validation plots."""
    directory = evaluation_artifact_directory(
        project_root, artifacts_directory, run_id, evaluation_id
    )
    directory.mkdir(parents=True, exist_ok=True)

    _trajectory_frame(result).to_parquet(directory / "trajectory.parquet", index=False)
    _trade_events_frame(result).to_parquet(directory / "trade_events.parquet", index=False)

    payload: dict[str, Any] = {
        "run_id": int(run_id),
        "evaluation_id": int(evaluation_id),
        "checkpoint_id": int(checkpoint_id),
        "steps_completed": int(result.steps_completed),
        "evaluation_start_index": int(result.evaluation_start_index),
        "evaluation_end_index": int(result.evaluation_end_index),
        "lookback_rows": int(result.lookback_rows),
        "metrics": asdict(result.metrics),
    }
    if metadata:
        payload["reproducibility"] = metadata
    (directory / "metrics.json").write_text(
        json.dumps(payload, indent=2, default=str, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return directory


def _pyplot():
    # Matplotlib is intentionally imported only when plot rendering is
    # requested. Normal training and metric/trajectory collection avoid
    # its import and startup cost entirely.
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _save_figure(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    _pyplot().close(fig)


def _render_evaluation_frames(
    frame: pd.DataFrame,
    events: pd.DataFrame,
    directory: Path,
) -> tuple[Path, ...]:
    directory.mkdir(parents=True, exist_ok=True)
    plt = _pyplot()
    x = pd.to_datetime(frame["timestamp"])
    outputs: list[Path] = []

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, frame["agent_equity"], label="agent")
    ax.plot(x, frame["always_long_equity"], label="always_long")
    ax.plot(x, frame["always_short_equity"], label="always_short")
    ax.set_title("Validation equity curves")
    ax.set_ylabel("cumulative return")
    ax.legend()
    path = directory / "equity_curve.png"
    _save_figure(fig, path)
    outputs.append(path)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, frame["drawdown"])
    ax.set_title("Agent drawdown")
    ax.set_ylabel("drawdown")
    path = directory / "drawdown_curve.png"
    _save_figure(fig, path)
    outputs.append(path)

    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.step(x, frame["position"], where="post")
    ax.set_title("Agent position")
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(["SHORT", "FLAT", "LONG"])
    path = directory / "position_timeline.png"
    _save_figure(fig, path)
    outputs.append(path)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, frame["cumulative_fee_cost"], label="fees")
    ax.plot(x, frame["cumulative_swap_cost"], label="swap")
    ax.plot(x, frame["cumulative_trade_cost"], label="total")
    ax.set_title("Cumulative trading costs")
    ax.legend()
    path = directory / "cumulative_costs.png"
    _save_figure(fig, path)
    outputs.append(path)

    close_events = pd.DataFrame()
    if not events.empty and "event" in events.columns:
        close_events = events[
            events["event"].astype(str).str.startswith("CLOSE_")
        ]

    if not close_events.empty and "net_return" in close_events.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(close_events["net_return"].dropna(), bins=50)
        ax.set_title("Closed-trade net returns")
        ax.set_xlabel("net return")
        path = directory / "trade_returns.png"
        _save_figure(fig, path)
        outputs.append(path)

    if not close_events.empty and "bars_held" in close_events.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(close_events["bars_held"].dropna(), bins=50)
        ax.set_title("Holding time distribution")
        ax.set_xlabel("bars held")
        path = directory / "holding_times.png"
        _save_figure(fig, path)
        outputs.append(path)

    return tuple(outputs)


def render_evaluation_result_plots(
    result: EvaluationRunResult,
    *,
    directory: str | Path,
) -> tuple[Path, ...]:
    """Render plots directly from an in-memory evaluation without retaining trajectory."""
    return _render_evaluation_frames(
        _trajectory_frame(result),
        _trade_events_frame(result),
        Path(directory),
    )


def render_evaluation_plots(directory: str | Path) -> tuple[Path, ...]:
    directory = Path(directory)
    trajectory_path = directory / "trajectory.parquet"
    if not trajectory_path.exists():
        raise ReportArtifactError(f"Missing trajectory: {trajectory_path}")
    frame = pd.read_parquet(trajectory_path)
    events_path = directory / "trade_events.parquet"
    events = pd.read_parquet(events_path) if events_path.exists() else pd.DataFrame()
    return _render_evaluation_frames(frame, events, directory)


def render_run_level_artifacts(session_factory, *, run_id: int, project_root: str | Path, artifacts_directory: str | Path) -> tuple[Path, ...]:
    """Render run-level curves exclusively from persisted database rows."""
    report_dir = run_artifact_root(project_root, artifacts_directory, run_id) / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    plt = _pyplot()
    outputs: list[Path] = []

    with session_factory() as session:
        run = session.get(Run, int(run_id))
        if run is None:
            raise ReportArtifactError(f"Run {run_id} does not exist.")
        training = list(session.scalars(select(TrainingMetric).where(TrainingMetric.run_id == run_id).order_by(TrainingMetric.model_step)).all())
        evaluations = list(session.scalars(select(Evaluation).join(Checkpoint, Checkpoint.id == Evaluation.checkpoint_id).where(Checkpoint.run_id == run_id, Evaluation.status == EvaluationStatus.COMPLETED).order_by(Checkpoint.model_step, Evaluation.id)).all())
        checkpoint_steps = {int(cp.id): int(cp.model_step) for cp in session.scalars(select(Checkpoint).where(Checkpoint.run_id == run_id)).all()}

        if training:
            tf = pd.DataFrame([{
                "model_step": int(m.model_step), "run_step": int(m.run_step),
                "ep_reward": m.ep_reward,
                "rollout_reward_mean": m.rollout_reward_mean,
                "rollout_reward_sum": m.rollout_reward_sum,
                "entropy_loss": m.entropy_loss,
                "explained_variance": m.explained_variance, "approx_kl": m.approx_kl,
                "clip_fraction": m.clip_fraction, "policy_gradient_loss": m.policy_gradient_loss,
                "value_loss": m.value_loss, "learning_rate": m.learning_rate,
            } for m in training])
            tf.to_csv(report_dir / "training_metrics.csv", index=False)
            fig, axes = plt.subplots(3, 2, figsize=(13, 12), sharex=True)
            series = [
                ("rollout_reward_mean", "Rollout reward mean"), ("entropy_loss", "Entropy loss"),
                ("explained_variance", "Explained variance"), ("approx_kl", "Approx KL"),
                ("clip_fraction", "Clip fraction"), ("value_loss", "Value loss"),
            ]
            for ax, (column, title) in zip(axes.flat, series):
                ax.plot(tf["model_step"], tf[column])
                ax.set_title(title)
                ax.set_xlabel("model step")
            path = report_dir / "training_curves.png"; _save_figure(fig, path); outputs.append(path)

        if evaluations:
            vf = pd.DataFrame([{
                "evaluation_id": int(e.id),
                "model_step": checkpoint_steps[int(e.checkpoint_id)],
                "balanced_score": e.balanced_score, "agent_return": e.agent_return,
                "always_long_return": e.always_long_return, "always_short_return": e.always_short_return,
                "agent_max_drawdown": e.agent_max_drawdown, "profit_factor": e.profit_factor,
                "win_rate": e.win_rate, "market_exposure": e.market_exposure,
                "round_trips": e.round_trips,
            } for e in evaluations])
            vf.to_csv(report_dir / "validation_metrics.csv", index=False)
            fig, axes = plt.subplots(3, 2, figsize=(13, 12), sharex=True)
            plots = [
                ("balanced_score", "Balanced score"), ("agent_return", "Agent return"),
                ("agent_max_drawdown", "Max drawdown"), ("profit_factor", "Profit factor"),
                ("market_exposure", "Market exposure"), ("round_trips", "Round trips"),
            ]
            for ax, (column, title) in zip(axes.flat, plots):
                ax.plot(vf["model_step"], vf[column], marker="o")
                if column == "agent_return":
                    ax.plot(vf["model_step"], vf["always_long_return"], linestyle="--", label="always_long")
                    ax.legend()
                ax.set_title(title)
                ax.set_xlabel("model step")
            path = report_dir / "validation_curves.png"; _save_figure(fig, path); outputs.append(path)

        summary = {
            "run_id": int(run.id), "run_name": str(run.name), "status": str(run.status.value),
            "git_commit": str(run.git_commit), "data_sha256": str(run.data_sha256),
            "training_metric_snapshots": len(training), "completed_evaluations": len(evaluations),
        }
        path = report_dir / "summary.json"
        path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        outputs.append(path)

    return tuple(outputs)
