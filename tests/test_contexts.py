from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from train_and_eval.environment.contexts import (
    AVAILABLE_CONTEXTS,
    BaselineMultiscaleV1Context,
    ContextError,
    MarketContext,
    build_context,
    build_observation,
    get_context_definition,
    get_context_type,
)


def _market_frame(
    rows: int = 32,
    *,
    start: str = "2026-01-05 14:00:00+00:00",
    frequency: str = "5min",
) -> pd.DataFrame:
    index = np.arange(
        rows,
        dtype=np.float64,
    )

    open_price = 100.0 + index

    close = open_price + np.where(
        index % 2 == 0,
        0.5,
        -0.25,
    )

    return pd.DataFrame(
        {
            "DT": pd.date_range(
                start,
                periods=rows,
                freq=frequency,
                tz="UTC",
            ),
            "Open": open_price,
            "High": (
                np.maximum(
                    open_price,
                    close,
                )
                + 1.0
            ),
            "Low": (
                np.minimum(
                    open_price,
                    close,
                )
                - 1.0
            ),
            "Close": close,
            "Volume": 100.0 + index,
        }
    )


def _build(
    frame: pd.DataFrame,
    *,
    market_timezone: str = "America/New_York",
    swap_timezone: str = "America/New_York",
):
    return build_context(
        "baseline_multiscale_v1",
        frame,
        market_timezone=market_timezone,
        rth_open="09:30",
        rth_close="16:00",
        swap_time="17:00",
        swap_timezone=swap_timezone,
    )


def test_baseline_context_class_is_registered() -> None:
    assert AVAILABLE_CONTEXTS == (
        "baseline_multiscale_v1",
    )

    assert (
        get_context_type(
            "baseline_multiscale_v1"
        )
        is BaselineMultiscaleV1Context
    )


def test_baseline_observation_size_for_window_40() -> None:
    definition = (
        BaselineMultiscaleV1Context.definition()
    )

    assert definition.observation_size(40) == 327
    assert definition.required_history_rows(40) == 6145

    assert (
        get_context_definition(
            "baseline_multiscale_v1"
        )
        == definition
    )


def test_context_preserves_feature_order() -> None:
    feature_names = (
        BaselineMultiscaleV1Context
        .context_feature_names()
    )

    assert feature_names[:4] == (
        "log_ret_3",
        "log_ret_6",
        "log_ret_12",
        "log_ret_24",
    )

    assert feature_names[12:16] == (
        "vol_3",
        "vol_6",
        "vol_12",
        "vol_24",
    )

    assert feature_names[-3:] == (
        "swap_pre_active",
        "swap_pre_sin_half",
        "swap_pre_cos_half",
    )


def test_build_context_creates_finite_features() -> None:
    prepared = _build(
        _market_frame()
    )

    definition = prepared.definition

    feature_columns = (
        list(definition.window_features)
        + list(definition.context_features)
    )

    values = prepared.frame[
        feature_columns
    ].to_numpy()

    assert np.isfinite(values).all()

    assert (
        prepared.definition.name
        == "baseline_multiscale_v1"
    )


def test_window_features_use_expected_formulas() -> None:
    prepared = _build(
        _market_frame()
    )

    first = prepared.frame.iloc[0]

    expected_range = (
        first["High"]
        - first["Low"]
    ) / first["Open"]

    expected_body = abs(
        first["Close"]
        - first["Open"]
    ) / first["Open"]

    assert first["direction"] == 1.0

    assert first["range"] == pytest.approx(
        expected_range
    )

    assert first["body"] == pytest.approx(
        expected_body
    )


def test_observation_includes_current_candle() -> None:
    prepared = _build(
        _market_frame(rows=12)
    )

    observation = build_observation(
        prepared,
        current_index=7,
        window=3,
        position=-1,
        unrealized_pnl=0.125,
    )

    definition = prepared.definition

    expected_market_window = (
        prepared.frame.loc[
            5:7,
            list(
                definition.window_features
            ),
        ]
        .to_numpy(dtype=np.float32)
        .reshape(-1)
    )

    np.testing.assert_allclose(
        observation[
            :len(expected_market_window)
        ],
        expected_market_window,
    )

    assert observation.shape == (
        definition.observation_size(3),
    )

    assert observation[-2] == -1.0

    assert observation[-1] == pytest.approx(
        0.125
    )


def test_swap_context_uses_swap_timezone() -> None:
    frame = _market_frame(
        rows=4,
        start="2026-01-05 21:30:00+00:00",
    )

    prepared = _build(
        frame,
        market_timezone="UTC",
        swap_timezone="America/New_York",
    )

    # 21:30 UTC in January equals 16:30 in New York.
    assert (
        prepared.frame.loc[
            0,
            "swap_pre_active",
        ]
        == 1.0
    )


def test_new_context_can_be_defined_by_one_class() -> None:
    class ShortMomentumTestContext(
        MarketContext
    ):
        name = "short_momentum_test_v1"

        horizons = (
            3,
            6,
        )

        window_features = (
            "direction",
            "body",
        )

        rolling_features = (
            "log_return",
            "volatility",
        )

        include_time_features = False

    context = ShortMomentumTestContext()

    prepared = context.build(
        _market_frame(),
        market_timezone="America/New_York",
        rth_open="09:30",
        rth_close="16:00",
        swap_time="17:00",
        swap_timezone="America/New_York",
    )

    assert (
        prepared.definition.context_features
        == (
            "log_ret_3",
            "log_ret_6",
            "vol_3",
            "vol_6",
        )
    )

    assert (
        prepared.definition.window_features
        == (
            "direction",
            "body",
        )
    )

    assert (
        prepared.definition.observation_size(10)
        == 26
    )
    assert (
        prepared.definition.required_history_rows(10)
        == 10
    )
    assert (
        prepared.definition.required_history_rows(2)
        == 7
    )


def test_unknown_context_is_rejected() -> None:
    with pytest.raises(
        ContextError,
        match="Unknown context",
    ):
        build_context(
            "missing_context",
            _market_frame(),
            market_timezone="America/New_York",
            rth_open="09:30",
            rth_close="16:00",
            swap_time="17:00",
            swap_timezone="America/New_York",
        )


def test_longest_log_return_is_complete_only_after_6145_rows() -> None:
    source = _market_frame(rows=6145)
    prepared = _build(source)

    assert prepared.frame.loc[
        6143,
        "log_ret_6144",
    ] == 0.0

    expected = np.log(
        source.loc[6144, "Close"]
        / source.loc[0, "Close"]
    )

    assert prepared.frame.loc[
        6144,
        "log_ret_6144",
    ] == pytest.approx(expected)
