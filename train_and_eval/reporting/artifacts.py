from __future__ import annotations

from typing import TYPE_CHECKING

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sqlalchemy import select


if TYPE_CHECKING:
    from train_and_eval.evaluation.runner import (
        EvaluationRunResult,
    )

from train_and_eval.database.models import Checkpoint, Evaluation, EvaluationStatus, Run, TrainingMetric


class ReportArtifactError(RuntimeError):
    """Raised when rebuildable report artifacts cannot be produced."""


ROLLING_EXPOSURE_WINDOW_BARS = 250


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

    # Persist the complete categorical policy distribution rather than
    # only the probability of the selected action.  This allows policy
    # confidence diagnostics to be regenerated later without loading the
    # PPO checkpoint again.
    probabilities = result.policy_trace.probabilities

    if len(probabilities) != len(frame):
        raise ReportArtifactError(
            "Policy probability trace length does not match "
            "the evaluation trajectory."
        )

    action_counts = {
        len(values)
        for values in probabilities
    }

    if len(action_counts) > 1:
        raise ReportArtifactError(
            "Policy probability vectors have inconsistent lengths."
        )

    action_count = (
        next(iter(action_counts))
        if action_counts
        else 0
    )

    for action_index in range(action_count):
        frame[
            f"policy_probability_action_{action_index}"
        ] = [
            float(values[action_index])
            for values in probabilities
        ]

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


def evaluation_trajectory_has_policy_probabilities(
    directory: str | Path,
) -> bool:
    """Return whether a persisted trajectory contains full policy probabilities."""
    trajectory_path = (
        Path(directory)
        / "trajectory.parquet"
    )

    if not trajectory_path.exists():
        return False

    try:
        columns = set(
            pq.read_schema(
                trajectory_path
            ).names
        )
    except Exception as error:
        raise ReportArtifactError(
            f"Could not inspect trajectory schema: "
            f"{trajectory_path}"
        ) from error

    return (
        "policy_probability_action_1"
        in columns
    )


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


def _format_axis_as_percent(ax: Any) -> None:
    from matplotlib.ticker import PercentFormatter

    ax.yaxis.set_major_formatter(PercentFormatter(1.0))


def _format_x_axis_as_percent(ax: Any) -> None:
    from matplotlib.ticker import PercentFormatter

    ax.xaxis.set_major_formatter(PercentFormatter(1.0))


def _additive_drawdown(equity: pd.Series | np.ndarray) -> np.ndarray:
    """Return drawdown with the same additive convention as EvaluationMetrics."""
    values = np.asarray(equity, dtype=float)
    curve = np.concatenate((np.array([0.0]), values))
    running_peak = np.maximum.accumulate(curve)
    return (curve - running_peak)[1:]


def _infer_bar_duration_label(timestamps: pd.Series) -> str | None:
    converted = pd.Series(pd.to_datetime(timestamps)).dropna().sort_values()
    if len(converted) < 2:
        return None

    deltas = converted.diff().dropna()
    deltas = deltas[deltas > pd.Timedelta(0)]
    if deltas.empty:
        return None

    seconds = float(deltas.median().total_seconds())
    if seconds <= 0:
        return None

    units = (
        (86_400.0, "day"),
        (3_600.0, "hour"),
        (60.0, "minute"),
        (1.0, "second"),
    )
    for divisor, unit in units:
        value = seconds / divisor
        if value >= 1.0 and abs(value - round(value)) < 1e-9:
            count = int(round(value))
            suffix = unit if count == 1 else f"{unit}s"
            return f"1 bar ≈ {count} {suffix}"

    return f"1 bar ≈ {seconds:.1f} seconds"


def _last_float(frame: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.iloc[-1]) if not values.empty else float("nan")


def _mark_best_and_final(
    ax: Any,
    frame: pd.DataFrame,
    column: str,
    *,
    best_index: int,
    final_index: int,
    label_markers: bool,
) -> None:
    best = frame.loc[best_index]
    final = frame.loc[final_index]

    best_y = pd.to_numeric(
        pd.Series([best[column]]), errors="coerce"
    ).iloc[0]
    final_y = pd.to_numeric(
        pd.Series([final[column]]), errors="coerce"
    ).iloc[0]

    if pd.notna(best_y):
        ax.scatter(
            [best["model_step"]],
            [best_y],
            marker="*",
            s=150,
            zorder=5,
            label="BEST" if label_markers else None,
        )
    if pd.notna(final_y):
        ax.scatter(
            [final["model_step"]],
            [final_y],
            marker="o",
            s=80,
            zorder=5,
            label="FINAL" if label_markers else None,
        )



def _render_policy_probability_frames(
    frame: pd.DataFrame,
    directory: Path,
) -> tuple[Path, ...]:
    """
    Render diagnostics for action-1 confidence.

    In the current LONG/FLAT policy encoding action 1 is LONG.
    Full action probabilities remain stored generically in
    trajectory.parquet so future policy layouts can add their own
    diagnostics without changing the persisted representation.
    """
    column = "policy_probability_action_1"

    if column not in frame.columns:
        return ()

    values = (
        pd.to_numeric(
            frame[column],
            errors="coerce",
        )
        .dropna()
        .to_numpy(dtype=float)
    )

    values = values[
        np.isfinite(values)
    ]

    if values.size == 0:
        return ()

    if np.any(values < 0.0) or np.any(
        values > 1.0
    ):
        raise ReportArtifactError(
            "Persisted policy probabilities must "
            "be between 0 and 1."
        )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt = _pyplot()
    outputs: list[Path] = []

    # --------------------------------------------------------
    # P(LONG) distribution
    # --------------------------------------------------------

    mean_probability = float(
        np.mean(values)
    )

    median_probability = float(
        np.median(values)
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.hist(
        values,
        bins=np.linspace(
            0.0,
            1.0,
            51,
        ),
        density=True,
    )

    ax.axvline(
        0.50,
        linestyle="--",
        linewidth=1.2,
        label="Argmax boundary (0.50)",
    )

    ax.axvline(
        mean_probability,
        linestyle=":",
        linewidth=1.2,
        label=(
            "Mean "
            f"{mean_probability:.3f}"
        ),
    )

    ax.axvline(
        median_probability,
        linestyle="-.",
        linewidth=1.2,
        label=(
            "Median "
            f"{median_probability:.3f}"
        ),
    )

    ax.set_xlim(
        0.0,
        1.0,
    )

    ax.set_title(
        "Policy P(LONG) distribution"
    )

    ax.set_xlabel(
        "P(LONG)"
    )

    ax.set_ylabel(
        "probability density"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    path = (
        directory
        / "policy_p_long_distribution.png"
    )

    _save_figure(
        fig,
        path,
    )

    outputs.append(path)

    # --------------------------------------------------------
    # P(LONG) confidence survival curve
    # --------------------------------------------------------

    thresholds = np.linspace(
        0.0,
        1.0,
        201,
    )

    fractions = np.asarray(
        [
            np.mean(
                values >= threshold
            )
            for threshold in thresholds
        ],
        dtype=float,
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.plot(
        thresholds,
        fractions,
        linewidth=2.0,
    )

    ax.axvline(
        0.50,
        linestyle="--",
        linewidth=1.2,
        label="Argmax boundary (0.50)",
    )

    ax.set_xlim(
        0.0,
        1.0,
    )

    ax.set_ylim(
        0.0,
        1.0,
    )

    ax.set_title(
        "Policy P(LONG) confidence curve"
    )

    ax.set_xlabel(
        "minimum P(LONG) required"
    )

    ax.set_ylabel(
        "observations with P(LONG) >= threshold"
    )

    _format_x_axis_as_percent(ax)
    _format_axis_as_percent(ax)

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    path = (
        directory
        / "policy_p_long_confidence_curve.png"
    )

    _save_figure(
        fig,
        path,
    )

    outputs.append(path)

    return tuple(outputs)


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
    agent_final = _last_float(frame, "agent_equity")
    long_final = _last_float(frame, "always_long_equity")
    short_final = _last_float(frame, "always_short_equity")
    ax.plot(x, frame["agent_equity"], label=f"agent ({agent_final:+.1%})")
    ax.plot(x, frame["always_long_equity"], label=f"always_long ({long_final:+.1%})")
    ax.plot(x, frame["always_short_equity"], label=f"always_short ({short_final:+.1%})")
    ax.axhline(0.0, linewidth=0.8, linestyle="--")
    ax.set_title("Validation equity curves")
    ax.set_ylabel("cumulative return")
    _format_axis_as_percent(ax)
    ax.legend()
    path = directory / "equity_curve.png"
    _save_figure(fig, path)
    outputs.append(path)

    fig, ax = plt.subplots(figsize=(12, 4))
    agent_drawdown = _additive_drawdown(frame["agent_equity"])
    long_drawdown = _additive_drawdown(frame["always_long_equity"])
    ax.plot(x, agent_drawdown, label=f"agent (Max DD {agent_drawdown.min(initial=0.0):.1%})")
    ax.plot(x, long_drawdown, label=f"always_long (Max DD {long_drawdown.min(initial=0.0):.1%})")
    ax.axhline(0.0, linewidth=0.8)
    ax.set_title("Validation drawdown")
    ax.set_ylabel("drawdown")
    _format_axis_as_percent(ax)
    ax.legend()
    path = directory / "drawdown_curve.png"
    _save_figure(fig, path)
    outputs.append(path)

    # Dense LONG/FLAT/SHORT step plots become unreadable after a few hundred
    # trades. Preserve the raw position in trajectory.parquet and render a
    # market-regime view instead.
    legacy_position_path = directory / "position_timeline.png"
    legacy_position_path.unlink(missing_ok=True)

    rolling_window = ROLLING_EXPOSURE_WINDOW_BARS
    rolling_market_exposure = (
        pd.to_numeric(frame["position"], errors="coerce")
        .abs()
        .rolling(rolling_window, min_periods=rolling_window)
        .mean()
    )
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(x, frame["close_price"])
    axes[0].set_title("Market price and rolling agent exposure")
    axes[0].set_ylabel("close price")
    axes[1].plot(
        x,
        rolling_market_exposure,
        label=f"market exposure ({rolling_window}-bar rolling mean)",
    )
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_ylabel("market exposure")
    _format_axis_as_percent(axes[1])
    axes[1].legend()
    path = directory / "market_and_exposure.png"
    _save_figure(fig, path)
    outputs.append(path)

    fig, ax = plt.subplots(figsize=(12, 4))
    fee_final = _last_float(frame, "cumulative_fee_cost")
    swap_final = _last_float(frame, "cumulative_swap_cost")
    total_final = _last_float(frame, "cumulative_trade_cost")
    ax.plot(
        x,
        frame["cumulative_fee_cost"],
        label=f"fees ({fee_final:.1%})",
    )
    ax.plot(
        x,
        frame["cumulative_swap_cost"],
        label=f"swap ({swap_final:.1%})",
    )
    ax.plot(
        x,
        frame["cumulative_trade_cost"],
        label=f"total ({total_final:.1%})",
    )
    ax.set_title("Cumulative trading costs")
    ax.set_ylabel("cumulative cost [% of stake]")
    _format_axis_as_percent(ax)
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
        returns = pd.to_numeric(close_events["net_return"], errors="coerce").dropna()
        if not returns.empty:
            low = float(returns.quantile(0.01))
            high = float(returns.quantile(0.99))
            central = returns[(returns >= low) & (returns <= high)]
            if central.empty or not low < high:
                central = returns

            mean_return = float(returns.mean())
            median_return = float(returns.median())
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(central, bins=50)
            ax.axvline(0.0, linewidth=0.8)
            ax.axvline(mean_return, linestyle="--", label=f"mean {mean_return:+.3%}")
            ax.axvline(median_return, linestyle=":", label=f"median {median_return:+.3%}")
            ax.set_title("Closed-trade net returns (central 98%)")
            ax.set_xlabel("net return")
            _format_x_axis_as_percent(ax)
            ax.text(
                0.98,
                0.95,
                "\n".join(
                    (
                        f"trades: {len(returns):,}",
                        f"min: {float(returns.min()):+.2%}",
                        f"max: {float(returns.max()):+.2%}",
                    )
                ),
                transform=ax.transAxes,
                ha="right",
                va="top",
            )
            ax.legend()
            path = directory / "trade_returns.png"
            _save_figure(fig, path)
            outputs.append(path)

    if not close_events.empty and "bars_held" in close_events.columns:
        bars = pd.to_numeric(close_events["bars_held"], errors="coerce").dropna()
        bars = bars[bars >= 0]
        if not bars.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            integer_bars = bars.round().astype(int)
            counts = integer_bars.value_counts().sort_index()
            if len(counts) <= 50:
                ax.bar(counts.index, counts.values)
            else:
                ax.hist(integer_bars, bins=50)
            ax.set_title("Holding time distribution")
            ax.set_xlabel("bars held")

            duration_label = _infer_bar_duration_label(pd.Series(x))
            stats = [
                f"mean: {float(bars.mean()):.2f} bars",
                f"median: {float(bars.median()):.0f} bars",
                f"max: {int(bars.max())} bars",
            ]
            if duration_label is not None:
                stats.append(duration_label)
            ax.text(
                0.98,
                0.95,
                "\n".join(stats),
                transform=ax.transAxes,
                ha="right",
                va="top",
            )
            path = directory / "holding_times.png"
            _save_figure(fig, path)
            outputs.append(path)

    outputs.extend(
        _render_policy_probability_frames(
            frame,
            directory,
        )
    )

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
        checkpoints = list(session.scalars(select(Checkpoint).where(Checkpoint.run_id == run_id)).all())
        checkpoint_steps = {int(cp.id): int(cp.model_step) for cp in checkpoints}
        source_model_step: int | None = None
        if run.source_checkpoint_id is not None:
            source_checkpoint = session.get(Checkpoint, int(run.source_checkpoint_id))
            if source_checkpoint is not None:
                source_model_step = int(source_checkpoint.model_step)

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
            fig, axes = plt.subplots(4, 2, figsize=(13, 15), sharex=True)
            series = [
                ("rollout_reward_mean", "Rollout reward mean"), ("entropy_loss", "Entropy loss"),
                ("explained_variance", "Explained variance"), ("approx_kl", "Approx KL"),
                ("clip_fraction", "Clip fraction"), ("value_loss", "Value loss"),
                ("policy_gradient_loss", "Policy gradient loss"), ("learning_rate", "Learning rate"),
            ]
            evaluation_steps = sorted({
                checkpoint_steps[int(e.checkpoint_id)] for e in evaluations
            })
            for axis_index, (ax, (column, title)) in enumerate(zip(axes.flat, series)):
                ax.plot(tf["model_step"], tf[column])
                for step_index, step in enumerate(evaluation_steps):
                    ax.axvline(
                        step,
                        linestyle=":",
                        linewidth=0.8,
                        alpha=0.35,
                        label="validation" if axis_index == 0 and step_index == 0 else None,
                    )
                if source_model_step is not None:
                    ax.axvline(
                        source_model_step,
                        linestyle="--",
                        linewidth=1.0,
                        alpha=0.7,
                        label="resume start" if axis_index == 0 else None,
                    )
                ax.set_title(title)
                ax.set_xlabel("model step")
            if evaluation_steps or source_model_step is not None:
                axes.flat[0].legend()
            path = report_dir / "training_curves.png"; _save_figure(fig, path); outputs.append(path)

        if evaluations:
            vf = pd.DataFrame([{
                "evaluation_id": int(e.id),
                "model_step": checkpoint_steps[int(e.checkpoint_id)],
                "trigger": str(getattr(e.trigger, "value", e.trigger)),
                "balanced_score": e.balanced_score, "agent_return": e.agent_return,
                "always_long_return": e.always_long_return, "always_short_return": e.always_short_return,
                "agent_vs_always_long_return": e.agent_vs_always_long_return,
                "agent_max_drawdown": e.agent_max_drawdown, "profit_factor": e.profit_factor,
                "win_rate": e.win_rate, "market_exposure": e.market_exposure,
                "round_trips": e.round_trips,
            } for e in evaluations])
            vf.to_csv(report_dir / "validation_metrics.csv", index=False)
            fig, axes = plt.subplots(4, 2, figsize=(13, 15), sharex=True)
            best_index = int(vf["balanced_score"].astype(float).idxmax())
            final_rows = vf.index[vf["trigger"] == "final"].tolist()
            final_index = int(final_rows[-1] if final_rows else vf.index[-1])
            plots = [
                ("balanced_score", "Balanced score"),
                ("agent_return", "Agent return"),
                (
                    "agent_vs_always_long_return",
                    "Agent vs always-long return",
                ),
                ("agent_max_drawdown", "Max drawdown"),
                ("profit_factor", "Profit factor"),
                ("market_exposure", "Market exposure"),
                ("round_trips", "Round trips"),
                ("win_rate", "Win rate"),
            ]
            for axis_index, (ax, (column, title)) in enumerate(
                zip(axes.flat, plots)
            ):
                ax.plot(vf["model_step"], vf[column], marker="o")
                if column in {
                    "agent_return",
                    "agent_vs_always_long_return",
                }:
                    ax.axhline(0.0, linewidth=0.8, linestyle="--")
                    _format_axis_as_percent(ax)
                elif column in {
                    "agent_max_drawdown",
                    "market_exposure",
                    "win_rate",
                }:
                    _format_axis_as_percent(ax)
                _mark_best_and_final(
                    ax,
                    vf,
                    column,
                    best_index=best_index,
                    final_index=final_index,
                    label_markers=axis_index == 0,
                )
                ax.set_title(title)
                ax.set_xlabel("model step")
                if axis_index == 0:
                    ax.legend()
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
