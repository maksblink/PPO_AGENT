from __future__ import annotations

from pathlib import Path

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
    frame.to_parquet(tmp_path / "trajectory.parquet", index=False)
    events.to_parquet(tmp_path / "trade_events.parquet", index=False)
    legacy = tmp_path / "position_timeline.png"
    legacy.write_bytes(b"legacy")

    outputs = artifacts.render_evaluation_plots(tmp_path)
    names = {path.name for path in outputs}

    assert names == {
        "equity_curve.png",
        "drawdown_curve.png",
        "market_and_exposure.png",
        "cumulative_costs.png",
        "trade_returns.png",
        "holding_times.png",
    }
    assert not legacy.exists()
    assert all(path.exists() for path in outputs)
