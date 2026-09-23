from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from train_and_eval.reporting import artifacts


def test_additive_drawdown_matches_evaluation_metric_convention() -> None:
    equity = pd.Series([0.10, 0.05, 0.20, 0.12])

    result = artifacts._additive_drawdown(equity)

    np.testing.assert_allclose(
        result,
        np.array([0.0, -0.05, 0.0, -0.08]),
    )


def test_infer_bar_duration_label_from_timestamps() -> None:
    timestamps = pd.Series(
        pd.date_range(
            "2026-01-01",
            periods=5,
            freq="1h",
            tz="UTC",
        )
    )

    assert artifacts._infer_bar_duration_label(timestamps) == (
        "1 bar ≈ 1 hour"
    )


def test_render_evaluation_plots_replaces_dense_position_timeline(
    tmp_path: Path,
) -> None:
    timestamps = pd.date_range(
        "2026-01-01",
        periods=8,
        freq="1h",
        tz="UTC",
    )
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "close_price": [100, 101, 100, 102, 103, 102, 104, 105],
            "position": [0, 1, 1, 0, 1, 0, 1, 0],
            "agent_equity": [0.0, 0.01, 0.005, 0.02, 0.018, 0.03, 0.025, 0.04],
            "always_long_equity": [0.0, 0.01, 0.0, 0.02, 0.03, 0.02, 0.04, 0.05],
            "always_short_equity": [0.0, -0.01, 0.0, -0.02, -0.03, -0.02, -0.04, -0.05],
            "cumulative_fee_cost": [0.0, 0.001, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006],
            "cumulative_swap_cost": [0.0, 0.0, 0.0, 0.0, 0.001, 0.001, 0.001, 0.001],
            "cumulative_trade_cost": [0.0, 0.001, 0.001, 0.002, 0.004, 0.005, 0.006, 0.007],
        }
    )
    events = pd.DataFrame(
        {
            "event": ["CLOSE_LONG", "CLOSE_LONG", "CLOSE_LONG"],
            "net_return": [0.01, -0.005, 0.003],
            "bars_held": [1, 2, 1],
        }
    )
    frame.to_parquet(tmp_path / "trajectory_val.parquet", index=False)
    events.to_parquet(tmp_path / "trade_events_val.parquet", index=False)
    legacy = tmp_path / "position_timeline_val.png"
    legacy.write_bytes(b"legacy")

    outputs = artifacts.render_evaluation_plots(tmp_path)
    names = {path.name for path in outputs}

    assert names == {
        "equity_curve_val.png",
        "drawdown_curve_val.png",
        "market_and_exposure_val.png",
        "cumulative_costs_val.png",
        "trade_returns_val.png",
        "holding_times_val.png",
    }
    assert not legacy.exists()
    assert all(path.exists() for path in outputs)



def test_trajectory_frame_persists_full_policy_probabilities() -> None:
    timestamps = tuple(
        pd.date_range(
            "2026-01-01",
            periods=2,
            freq="1h",
            tz="UTC",
        )
    )

    trajectory = SimpleNamespace(
        execution_indices=(10, 11),
        execution_timestamps=timestamps,
        execution_prices=(100.0, 101.0),
        close_prices=(100.0, 101.0),
        actions=(0, 1),
        positions=(0, 1),
        agent_equity=(0.0, 0.01),
        always_long_equity=(0.0, 0.01),
        always_short_equity=(0.0, -0.01),
        shaped_rewards=(0.0, 0.01),
        fee_costs=(0.0, 0.001),
        swap_costs=(0.0, 0.0),
        trade_costs=(0.0, 0.001),
        drawdowns=(0.0, 0.0),
        hold_bars=(0, 1),
        selected_action_probabilities=(
            0.8,
            0.7,
        ),
    )

    policy_trace = SimpleNamespace(
        probabilities=(
            (0.8, 0.2),
            (0.3, 0.7),
        )
    )

    result = SimpleNamespace(
        trajectory=trajectory,
        policy_trace=policy_trace,
    )

    frame = artifacts._trajectory_frame(
        result
    )

    np.testing.assert_allclose(
        frame[
            "policy_probability_action_0"
        ],
        np.array([0.8, 0.3]),
    )

    np.testing.assert_allclose(
        frame[
            "policy_probability_action_1"
        ],
        np.array([0.2, 0.7]),
    )


def test_render_evaluation_plots_adds_policy_probability_diagnostics(
    tmp_path: Path,
) -> None:
    timestamps = pd.date_range(
        "2026-01-01",
        periods=8,
        freq="1h",
        tz="UTC",
    )

    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "close_price": [
                100,
                101,
                100,
                102,
                103,
                102,
                104,
                105,
            ],
            "position": [
                0,
                1,
                1,
                0,
                1,
                0,
                1,
                0,
            ],
            "agent_equity": [
                0.0,
                0.01,
                0.005,
                0.02,
                0.018,
                0.03,
                0.025,
                0.04,
            ],
            "always_long_equity": [
                0.0,
                0.01,
                0.0,
                0.02,
                0.03,
                0.02,
                0.04,
                0.05,
            ],
            "always_short_equity": [
                0.0,
                -0.01,
                0.0,
                -0.02,
                -0.03,
                -0.02,
                -0.04,
                -0.05,
            ],
            "cumulative_fee_cost": [
                0.0,
                0.001,
                0.001,
                0.002,
                0.003,
                0.004,
                0.005,
                0.006,
            ],
            "cumulative_swap_cost": [
                0.0,
                0.0,
                0.0,
                0.0,
                0.001,
                0.001,
                0.001,
                0.001,
            ],
            "cumulative_trade_cost": [
                0.0,
                0.001,
                0.001,
                0.002,
                0.004,
                0.005,
                0.006,
                0.007,
            ],
            "policy_probability_action_1": [
                0.10,
                0.25,
                0.40,
                0.55,
                0.70,
                0.80,
                0.90,
                0.97,
            ],
        }
    )

    frame.to_parquet(
        tmp_path / "trajectory_val.parquet",
        index=False,
    )

    outputs = (
        artifacts.render_evaluation_plots(
            tmp_path
        )
    )

    names = {
        path.name
        for path in outputs
    }

    assert (
        "policy_p_long_distribution_val.png"
        in names
    )

    assert (
        "policy_p_long_confidence_curve_val.png"
        in names
    )

    assert (
        artifacts
        .evaluation_trajectory_has_policy_probabilities(
            tmp_path
        )
    )

    assert all(
        path.exists()
        for path in outputs
    )
