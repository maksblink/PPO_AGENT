from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from train_and_eval.environment.contexts import (
    MarketContext,
    register_context,
)
from train_and_eval.environment.trading_environment import (
    TradingEnvironment,
)
from train_and_eval.evaluation.metrics import (
    EvaluationMetricsError,
    calculate_evaluation_metrics,
)
from train_and_eval.run_config import (
    EnvironmentSection,
)




@register_context
class EvaluationMetricsTestContext(MarketContext):
    name = "evaluation_metrics_test_v1"
    horizons = ()
    window_features = (
        "direction",
        "range",
        "body",
        "upper_wick",
        "lower_wick",
        "vol_chg",
    )
    rolling_features = ()
    include_time_features = True


def _open_long(
    *,
    fee: float = 0.001,
) -> dict[str, object]:
    return {
        "event": "OPEN_LONG",
        "position_before": 0,
        "position_after": 1,
        "fee": fee,
    }


def _close_long(
    *,
    gross_return: float,
    net_return: float,
    entry_fee: float = 0.001,
    close_fee: float = 0.001,
    swap_cost: float = 0.0,
    swap_events: int = 0,
    bars_held: int = 1,
) -> dict[str, object]:
    return {
        "event": "CLOSE_LONG",
        "position_before": 1,
        "position_after": 0,
        "gross_return": gross_return,
        "net_return": net_return,
        "entry_fee": entry_fee,
        "close_fee": close_fee,
        "swap_cost": swap_cost,
        "swap_events": swap_events,
        "bars_held": bars_held,
    }


def _open_short(
    *,
    fee: float = 0.001,
) -> dict[str, object]:
    return {
        "event": "OPEN_SHORT",
        "position_before": 0,
        "position_after": -1,
        "fee": fee,
    }


def _close_short(
    *,
    gross_return: float,
    net_return: float,
    entry_fee: float = 0.001,
    close_fee: float = 0.001,
    swap_cost: float = 0.0,
    swap_events: int = 0,
    bars_held: int = 1,
) -> dict[str, object]:
    return {
        "event": "CLOSE_SHORT",
        "position_before": -1,
        "position_after": 0,
        "gross_return": gross_return,
        "net_return": net_return,
        "entry_fee": entry_fee,
        "close_fee": close_fee,
        "swap_cost": swap_cost,
        "swap_events": swap_events,
        "bars_held": bars_held,
    }


def test_calculates_mixed_trade_metrics() -> None:
    metrics = calculate_evaluation_metrics(
        agent_equity=[
            0.02,
            -0.01,
            0.04,
            0.03,
        ],
        always_long_equity=[
            0.01,
            0.02,
            -0.02,
            0.01,
        ],
        always_short_equity=[
            -0.01,
            -0.02,
            0.02,
            -0.01,
        ],
        positions=[
            1,
            0,
            -1,
            0,
        ],
        shaped_rewards=[
            0.1,
            -0.2,
            0.3,
            0.4,
        ],
        step_swap_costs=[
            0.0,
            0.0,
            0.001,
            0.0,
        ],
        step_swap_events=[
            0,
            0,
            1,
            0,
        ],
        trade_events=[
            _open_long(),
            _close_long(
                gross_return=0.05,
                net_return=0.048,
                bars_held=2,
            ),
            _open_short(),
            _close_short(
                gross_return=-0.02,
                net_return=-0.023,
                swap_cost=0.001,
                swap_events=1,
                bars_held=1,
            ),
        ],
        open_position_return_at_end=0.0,
    )

    assert metrics.agent_return == pytest.approx(
        0.03
    )
    assert (
        metrics.agent_vs_always_long_return
        == pytest.approx(0.02)
    )

    assert metrics.agent_max_drawdown == pytest.approx(
        -0.03
    )
    assert (
        metrics.always_long_max_drawdown
        == pytest.approx(-0.04)
    )
    assert metrics.drawdown_improvement == pytest.approx(
        0.01
    )
    assert metrics.balanced_score == pytest.approx(
        0.0
    )

    assert metrics.net_exposure == pytest.approx(
        0.0
    )
    assert metrics.long_exposure == pytest.approx(
        0.25
    )
    assert metrics.short_exposure == pytest.approx(
        0.25
    )
    assert metrics.flat_exposure == pytest.approx(
        0.50
    )
    assert metrics.market_exposure == pytest.approx(
        0.50
    )

    assert metrics.trade_events_total == 4
    assert metrics.trade_event_rate == pytest.approx(
        1.0
    )
    assert metrics.round_trips == 2
    assert metrics.round_trip_rate == pytest.approx(
        0.5
    )

    assert metrics.winning_trades == 1
    assert metrics.losing_trades == 1
    assert metrics.win_rate == pytest.approx(
        0.5
    )
    assert metrics.avg_trade_return == pytest.approx(
        0.0125
    )
    assert metrics.profit_factor == pytest.approx(
        0.048 / 0.023
    )
    assert metrics.payoff_ratio == pytest.approx(
        0.048 / 0.023
    )

    assert metrics.min_bars_held == 1
    assert metrics.avg_bars_held == pytest.approx(
        1.5
    )
    assert metrics.median_bars_held == pytest.approx(
        1.5
    )
    assert metrics.max_bars_held == 2

    assert metrics.max_consecutive_wins == 1
    assert metrics.max_consecutive_losses == 1
    assert metrics.current_streak_type == "loss"
    assert metrics.current_streak == 1

    assert metrics.total_fee_return == pytest.approx(
        0.004
    )
    assert metrics.total_swap_return == pytest.approx(
        0.001
    )
    assert metrics.total_cost_return == pytest.approx(
        0.005
    )
    assert (
        metrics.avg_fee_per_trade_return
        == pytest.approx(0.002)
    )
    assert (
        metrics.avg_swap_per_trade_return
        == pytest.approx(0.0005)
    )
    assert (
        metrics.avg_cost_per_trade_return
        == pytest.approx(0.0025)
    )

    assert (
        metrics.cumulative_shaped_reward
        == pytest.approx(0.6)
    )


def test_no_trades_uses_null_trade_statistics() -> None:
    metrics = calculate_evaluation_metrics(
        agent_equity=[0.0, 0.0],
        always_long_equity=[0.01, 0.02],
        always_short_equity=[-0.01, -0.02],
        positions=[0, 0],
        shaped_rewards=[0.0, 0.0],
        step_swap_costs=[0.0, 0.0],
        step_swap_events=[0, 0],
        trade_events=[],
        open_position_return_at_end=0.0,
    )

    assert metrics.trade_events_total == 0
    assert metrics.round_trips == 0
    assert metrics.trade_event_rate == 0.0
    assert metrics.round_trip_rate == 0.0

    assert metrics.win_rate is None
    assert metrics.avg_trade_return is None
    assert metrics.profit_factor is None
    assert metrics.payoff_ratio is None

    assert metrics.min_bars_held is None
    assert metrics.avg_bars_held is None
    assert metrics.max_bars_held is None

    assert metrics.max_consecutive_wins == 0
    assert metrics.max_consecutive_losses == 0
    assert metrics.current_streak_type == "none"
    assert metrics.current_streak == 0

    assert metrics.gross_profit_return == 0.0
    assert metrics.gross_loss_return == 0.0
    assert metrics.net_profit_return == 0.0


def test_drawdown_includes_initial_zero_equity() -> None:
    metrics = calculate_evaluation_metrics(
        agent_equity=[
            -0.10,
            -0.05,
            0.02,
        ],
        always_long_equity=[
            0.0,
            0.0,
            0.0,
        ],
        always_short_equity=[
            0.0,
            0.0,
            0.0,
        ],
        positions=[0, 0, 0],
        shaped_rewards=[0.0, 0.0, 0.0],
        step_swap_costs=[0.0, 0.0, 0.0],
        step_swap_events=[0, 0, 0],
        trade_events=[],
        open_position_return_at_end=0.0,
    )

    assert metrics.agent_max_drawdown == pytest.approx(
        -0.10
    )


def test_open_position_costs_are_in_total_but_not_average_trade_cost() -> None:
    metrics = calculate_evaluation_metrics(
        agent_equity=[-0.003],
        always_long_equity=[0.0],
        always_short_equity=[0.0],
        positions=[1],
        shaped_rewards=[-0.003],
        step_swap_costs=[0.002],
        step_swap_events=[1],
        trade_events=[
            _open_long(fee=0.001),
        ],
        open_position_return_at_end=-0.003,
    )

    assert metrics.round_trips == 0
    assert metrics.total_fee_return == pytest.approx(
        0.001
    )
    assert metrics.total_swap_return == pytest.approx(
        0.002
    )
    assert metrics.total_cost_return == pytest.approx(
        0.003
    )

    assert (
        metrics.avg_fee_per_trade_return
        is None
    )
    assert (
        metrics.avg_swap_per_trade_return
        is None
    )
    assert (
        metrics.avg_cost_per_trade_return
        is None
    )


def test_rejects_series_with_different_lengths() -> None:
    with pytest.raises(
        EvaluationMetricsError,
        match="does not match",
    ):
        calculate_evaluation_metrics(
            agent_equity=[0.0, 0.1],
            always_long_equity=[0.0],
            always_short_equity=[0.0, 0.0],
            positions=[0, 0],
            shaped_rewards=[0.0, 0.0],
            step_swap_costs=[0.0, 0.0],
            step_swap_events=[0, 0],
            trade_events=[],
            open_position_return_at_end=0.0,
        )


def test_rejects_unknown_trade_event() -> None:
    with pytest.raises(
        EvaluationMetricsError,
        match="Unknown trade event",
    ):
        calculate_evaluation_metrics(
            agent_equity=[0.0],
            always_long_equity=[0.0],
            always_short_equity=[0.0],
            positions=[0],
            shaped_rewards=[0.0],
            step_swap_costs=[0.0],
            step_swap_events=[0],
            trade_events=[
                {
                    "event": "UNKNOWN",
                },
            ],
            open_position_return_at_end=0.0,
        )


def test_rejects_incorrect_closed_trade_net_return() -> None:
    with pytest.raises(
        EvaluationMetricsError,
        match="Invalid net_return",
    ):
        calculate_evaluation_metrics(
            agent_equity=[0.0, 0.10],
            always_long_equity=[0.0, 0.0],
            always_short_equity=[0.0, 0.0],
            positions=[1, 0],
            shaped_rewards=[0.0, 0.10],
            step_swap_costs=[0.0, 0.0],
            step_swap_events=[0, 0],
            trade_events=[
                _open_long(),
                _close_long(
                    gross_return=0.10,
                    net_return=0.10,
                ),
            ],
            open_position_return_at_end=0.0,
        )


def _integration_environment_config() -> EnvironmentSection:
    return EnvironmentSection.model_construct(
        window=2,
        context="evaluation_metrics_test_v1",
        position_side="long_only",
        market_timezone="America/New_York",
        rth_open="09:30",
        rth_close="16:00",
        stake_pln=1000.0,
        fee_bps=10.0,
        swap_long_bps=0.0,
        swap_short_bps=0.0,
        swap_time="17:00",
        swap_timezone="America/New_York",
        force_close_on_done=True,
        reward_scale=1.0,
        exposure_penalty=0.0,
        turnover_penalty=0.0,
        drawdown_penalty=0.0,
        profit_reward_mult=1.0,
        loss_reward_mult=1.0,
    )


def _integration_market_frame() -> pd.DataFrame:
    opens = np.asarray(
        [100.0, 100.0, 100.0, 110.0],
        dtype=np.float64,
    )
    closes = np.asarray(
        [100.0, 100.0, 100.0, 110.0],
        dtype=np.float64,
    )

    return pd.DataFrame(
        {
            "DT": pd.date_range(
                "2026-01-05 14:00:00+00:00",
                periods=4,
                freq="5min",
                tz="UTC",
            ),
            "Open": opens,
            "High": opens + 1.0,
            "Low": opens - 1.0,
            "Close": closes,
            "Volume": np.arange(
                4,
                dtype=np.float64,
            ) + 100.0,
        }
    )


def test_real_environment_info_feeds_metrics_calculator() -> None:
    environment = TradingEnvironment(
        _integration_market_frame(),
        _integration_environment_config(),
    )
    environment.reset()

    agent_equity: list[float] = []
    positions: list[int] = []
    shaped_rewards: list[float] = []
    step_swap_costs: list[float] = []
    step_swap_events: list[int] = []

    terminated = False
    final_info: dict[str, object] | None = None

    while not terminated:
        (
            _,
            reward,
            terminated,
            truncated,
            info,
        ) = environment.step(1)

        assert truncated is False

        agent_equity.append(
            float(info["equity"])
        )
        positions.append(
            int(info["position_during_bar"])
        )
        shaped_rewards.append(
            float(reward)
        )
        step_swap_costs.append(
            float(info["swap_cost"])
        )
        step_swap_events.append(
            int(info["swap_events_on_step"])
        )

        final_info = info

    assert final_info is not None
    assert final_info["force_closed"] is True

    # The final position is closed, but the agent was exposed
    # throughout the final evaluated bar.
    assert final_info["position"] == 0
    assert final_info["position_during_bar"] == 1

    metrics = calculate_evaluation_metrics(
        agent_equity=agent_equity,
        always_long_equity=[
            0.0
            for _ in agent_equity
        ],
        always_short_equity=[
            0.0
            for _ in agent_equity
        ],
        positions=positions,
        shaped_rewards=shaped_rewards,
        step_swap_costs=step_swap_costs,
        step_swap_events=step_swap_events,
        trade_events=environment.trade_events,
        open_position_return_at_end=float(
            final_info["unrealized_return"]
        ),
    )

    assert len(agent_equity) == 2
    assert agent_equity[0] == pytest.approx(
        -0.001
    )
    assert agent_equity[-1] == pytest.approx(
        0.098
    )

    assert metrics.agent_return == pytest.approx(
        0.098
    )
    assert metrics.round_trips == 1
    assert metrics.long_round_trips == 1
    assert metrics.short_round_trips == 0

    assert metrics.open_long_count == 1
    assert metrics.close_long_count == 1
    assert metrics.trade_events_total == 2

    assert metrics.long_exposure == pytest.approx(
        1.0
    )
    assert metrics.market_exposure == pytest.approx(
        1.0
    )
    assert metrics.flat_exposure == pytest.approx(
        0.0
    )

    assert metrics.avg_trade_return == pytest.approx(
        0.098
    )
    assert metrics.total_fee_return == pytest.approx(
        0.002
    )
    assert (
        metrics.avg_fee_per_trade_return
        == pytest.approx(0.002)
    )

    assert metrics.min_bars_held == 2
    assert metrics.max_bars_held == 2

    assert (
        metrics.cumulative_shaped_reward
        == pytest.approx(0.098)
    )
    assert (
        metrics.open_position_return_at_end
        == pytest.approx(0.0)
    )
