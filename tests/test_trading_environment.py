from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from train_and_eval.environment.trading_environment import (
    TradingEnvironment,
    TradingEnvironmentError,
    action_names,
    action_targets,
    count_swap_boundaries,
)
from train_and_eval.run_config import (
    EnvironmentSection,
)


def _config(
    *,
    position_side: str = "long_only",
    fee_bps: float = 0.0,
    swap_bps: float = 0.0,
    force_close_on_done: bool = True,
    exposure_penalty: float = 0.0,
    turnover_penalty: float = 0.0,
    drawdown_penalty: float = 0.0,
) -> EnvironmentSection:
    return EnvironmentSection.model_validate(
        {
            "window": 2,
            "context": "baseline_multiscale_v1",
            "position_side": position_side,
            "market_timezone": "America/New_York",
            "rth_open": "09:30",
            "rth_close": "16:00",
            "stake_pln": 1000.0,
            "fee_bps": fee_bps,
            "swap_bps": swap_bps,
            "swap_time": "17:00",
            "swap_timezone": "America/New_York",
            "force_close_on_done": (
                force_close_on_done
            ),
            "reward_scale": 1.0,
            "exposure_penalty": (
                exposure_penalty
            ),
            "turnover_penalty": (
                turnover_penalty
            ),
            "drawdown_penalty": (
                drawdown_penalty
            ),
            "profit_reward_mult": 1.0,
            "loss_reward_mult": 1.0,
        }
    )


def _market_frame(
    *,
    opens: list[float],
    closes: list[float],
    start: str = "2026-01-05 14:00:00+00:00",
    frequency: str = "5min",
) -> pd.DataFrame:
    open_array = np.asarray(
        opens,
        dtype=np.float64,
    )
    close_array = np.asarray(
        closes,
        dtype=np.float64,
    )

    return pd.DataFrame(
        {
            "DT": pd.date_range(
                start,
                periods=len(opens),
                freq=frequency,
                tz="UTC",
            ),
            "Open": open_array,
            "High": (
                np.maximum(
                    open_array,
                    close_array,
                )
                + 1.0
            ),
            "Low": (
                np.minimum(
                    open_array,
                    close_array,
                )
                - 1.0
            ),
            "Close": close_array,
            "Volume": np.arange(
                len(opens),
                dtype=np.float64,
            ) + 100.0,
        }
    )


@pytest.mark.parametrize(
    (
        "position_side",
        "expected_targets",
        "expected_names",
    ),
    [
        (
            "long_only",
            (0, 1),
            ("FLAT", "LONG"),
        ),
        (
            "short_only",
            (0, -1),
            ("FLAT", "SHORT"),
        ),
        (
            "long_short",
            (0, 1, -1),
            ("FLAT", "LONG", "SHORT"),
        ),
    ],
)
def test_position_modes_define_actions(
    position_side: str,
    expected_targets: tuple[int, ...],
    expected_names: tuple[str, ...],
) -> None:
    assert (
        action_targets(position_side)
        == expected_targets
    )
    assert (
        action_names(position_side)
        == expected_names
    )


def test_long_action_executes_at_next_open() -> None:
    frame = _market_frame(
        opens=[
            90.0,
            95.0,
            100.0,
            120.0,
        ],
        closes=[
            91.0,
            96.0,
            110.0,
            121.0,
        ],
    )

    environment = TradingEnvironment(
        frame,
        _config(
            force_close_on_done=False
        ),
    )

    _, reset_info = environment.reset()

    assert reset_info[
        "observation_index"
    ] == 1

    _, reward, terminated, _, info = (
        environment.step(1)
    )

    assert terminated is False
    assert info["execution_index"] == 2
    assert info["execution_price"] == 100.0
    assert environment.entry_price == 100.0
    assert environment.position == 1
    assert info["unrealized_return"] == pytest.approx(
        0.10
    )
    assert reward == pytest.approx(0.10)


def test_short_only_profits_when_price_falls() -> None:
    frame = _market_frame(
        opens=[
            90.0,
            95.0,
            100.0,
            110.0,
        ],
        closes=[
            91.0,
            96.0,
            90.0,
            109.0,
        ],
    )

    environment = TradingEnvironment(
        frame,
        _config(
            position_side="short_only",
            force_close_on_done=False,
        ),
    )

    environment.reset()

    _, reward, _, _, info = (
        environment.step(1)
    )

    assert environment.position == -1
    assert environment.entry_price == 100.0
    assert info["unrealized_return"] == pytest.approx(
        0.10
    )
    assert reward == pytest.approx(0.10)


def test_long_short_can_reverse_position() -> None:
    frame = _market_frame(
        opens=[
            90.0,
            95.0,
            100.0,
            120.0,
            130.0,
        ],
        closes=[
            91.0,
            96.0,
            110.0,
            108.0,
            131.0,
        ],
    )

    environment = TradingEnvironment(
        frame,
        _config(
            position_side="long_short",
            force_close_on_done=False,
        ),
    )

    environment.reset()
    environment.step(1)

    _, reward, _, _, info = (
        environment.step(2)
    )

    assert info["execution_price"] == 120.0
    assert info["turnover"] == 2
    assert environment.position == -1
    assert environment.entry_price == 120.0
    assert info["equity"] == pytest.approx(
        0.30
    )
    assert reward == pytest.approx(0.20)

    assert [
        event["event"]
        for event in environment.trade_events
    ] == [
        "OPEN_LONG",
        "CLOSE_LONG",
        "OPEN_SHORT",
    ]


def test_invalid_action_is_rejected() -> None:
    frame = _market_frame(
        opens=[
            90.0,
            95.0,
            100.0,
        ],
        closes=[
            91.0,
            96.0,
            101.0,
        ],
    )

    environment = TradingEnvironment(
        frame,
        _config(),
    )

    environment.reset()

    with pytest.raises(
        ValueError,
        match="Invalid action",
    ):
        environment.step(2)


def test_force_close_on_final_candle() -> None:
    frame = _market_frame(
        opens=[
            90.0,
            95.0,
            100.0,
        ],
        closes=[
            91.0,
            96.0,
            110.0,
        ],
    )

    environment = TradingEnvironment(
        frame,
        _config(
            force_close_on_done=True
        ),
    )

    environment.reset()

    _, reward, terminated, _, info = (
        environment.step(1)
    )

    assert terminated is True
    assert info["force_closed"] is True
    assert environment.position == 0
    assert info["equity"] == pytest.approx(
        0.10
    )
    assert reward == pytest.approx(0.10)

    assert [
        event["event"]
        for event in environment.trade_events
    ] == [
        "OPEN_LONG",
        "CLOSE_LONG",
    ]


def test_negative_exposure_penalty_is_bonus() -> None:
    frame = _market_frame(
        opens=[
            90.0,
            95.0,
            100.0,
            110.0,
        ],
        closes=[
            91.0,
            96.0,
            110.0,
            111.0,
        ],
    )

    environment = TradingEnvironment(
        frame,
        _config(
            force_close_on_done=False,
            exposure_penalty=-0.05,
        ),
    )

    environment.reset()

    _, reward, _, _, _ = (
        environment.step(1)
    )

    assert reward == pytest.approx(
        0.15
    )


def test_every_weekend_swap_boundary_is_counted() -> None:
    before = pd.Timestamp(
        "2026-01-09 16:00:00",
        tz="America/New_York",
    )
    after = pd.Timestamp(
        "2026-01-12 18:00:00",
        tz="America/New_York",
    )

    assert count_swap_boundaries(
        before,
        after,
        swap_time="17:00",
        swap_timezone="America/New_York",
    ) == 4


def test_step_after_termination_is_rejected() -> None:
    frame = _market_frame(
        opens=[
            90.0,
            95.0,
            100.0,
        ],
        closes=[
            91.0,
            96.0,
            101.0,
        ],
    )

    environment = TradingEnvironment(
        frame,
        _config(),
    )

    environment.reset()
    environment.step(0)

    with pytest.raises(
        TradingEnvironmentError,
        match="after the episode",
    ):
        environment.step(0)
