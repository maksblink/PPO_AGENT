from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import pytest
import torch

from train_and_eval.environment.contexts import (
    MarketContext,
    register_context,
)
from train_and_eval.environment.trading_environment import (
    TradingEnvironment,
)
from train_and_eval.evaluation.runner import (
    EvaluationRunnerError,
    run_ppo_evaluation,
)
from train_and_eval.ppo.adapter import (
    create_ppo_model,
)
from train_and_eval.run_config import (
    EnvironmentSection,
    PPOSection,
)




@register_context
class EvaluationRunnerTestContext(MarketContext):
    name = "evaluation_runner_test_v1"
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


PositionSide = Literal[
    "long_only",
    "short_only",
    "long_short",
]


def _market_frame() -> pd.DataFrame:
    prices = np.asarray(
        [
            100.0,
            100.0,
            100.0,
            100.0,
            105.0,
            95.0,
            110.0,
            100.0,
        ],
        dtype=np.float64,
    )

    return pd.DataFrame(
        {
            "DT": pd.date_range(
                "2026-01-05 14:00:00+00:00",
                periods=len(prices),
                freq="5min",
                tz="UTC",
            ),
            "Open": prices,
            "High": prices + 1.0,
            "Low": prices - 1.0,
            "Close": prices,
            "Volume": np.arange(
                len(prices),
                dtype=np.float64,
            ) + 100.0,
        }
    )


def _environment_config(
    position_side: PositionSide,
) -> EnvironmentSection:
    return EnvironmentSection.model_construct(
        window=2,
        context="evaluation_runner_test_v1",
        position_side=position_side,
        market_timezone="America/New_York",
        rth_open="09:30",
        rth_close="16:00",
        stake_pln=1000.0,
        fee_bps=1.0,
        swap_bps=0.0,
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


def _ppo_config() -> PPOSection:
    return PPOSection.model_validate(
        {
            "policy": "mlp",
            "device": "cpu",
            "hidden_sizes": [16, 16],
            "activation": "tanh",
            "n_steps": 8,
            "batch_size": 4,
            "n_epochs": 1,
            "learning_rate": 0.0003,
            "gamma": 0.90,
            "gae_lambda": 0.95,
            "clip_range": 0.20,
            "clip_range_vf": None,
            "normalize_advantage": True,
            "ent_coef": 0.0,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "target_kl": None,
        }
    )


def _model(
    position_side: PositionSide,
    *,
    logits: list[float],
):
    config = _environment_config(
        position_side
    )

    model = create_ppo_model(
        TradingEnvironment(
            _market_frame(),
            config,
        ),
        _ppo_config(),
        seed=123,
    )

    with torch.no_grad():
        model.policy.action_net.weight.zero_()
        model.policy.action_net.bias.copy_(
            torch.tensor(logits)
        )

    return model


def test_runner_uses_exact_execution_range() -> None:
    result = run_ppo_evaluation(
        _model(
            "long_only",
            logits=[0.0, 2.0],
        ),
        _market_frame(),
        _environment_config(
            "long_only"
        ),
        evaluation_start_index=4,
        evaluation_end_index=8,
        lookback_rows=2,
        policy_mode="deterministic_argmax",
        seed=123,
    )

    assert result.steps_completed == 4
    assert (
        result.policy_trace.execution_indices
        == (4, 5, 6, 7)
    )
    assert result.policy_trace.actions == (
        1,
        1,
        1,
        1,
    )

    assert (
        result.metrics.agent_return
        == pytest.approx(
            result.metrics.always_long_return
        )
    )

    assert (
        result.metrics.long_exposure
        == pytest.approx(1.0)
    )
    assert (
        result.metrics.short_exposure
        == pytest.approx(0.0)
    )


def test_runner_supports_dynamic_long_short_threshold() -> None:
    result = run_ppo_evaluation(
        _model(
            "long_short",
            logits=[0.0, 1.0, 2.0],
        ),
        _market_frame(),
        _environment_config(
            "long_short"
        ),
        evaluation_start_index=4,
        evaluation_end_index=8,
        lookback_rows=2,
        policy_mode="probability_threshold",
        seed=123,
        threshold_action=None,
        probability_threshold=0.60,
    )

    assert result.policy_trace.actions == (
        2,
        2,
        2,
        2,
    )

    assert all(
        value is True
        for value in (
            result
            .policy_trace
            .threshold_met
        )
    )

    assert (
        result.metrics.agent_return
        == pytest.approx(
            result.metrics
            .always_short_return
        )
    )

    assert (
        result.metrics.short_exposure
        == pytest.approx(1.0)
    )


def test_runner_threshold_can_remain_flat() -> None:
    result = run_ppo_evaluation(
        _model(
            "long_short",
            logits=[0.0, 1.0, 2.0],
        ),
        _market_frame(),
        _environment_config(
            "long_short"
        ),
        evaluation_start_index=4,
        evaluation_end_index=8,
        lookback_rows=2,
        policy_mode="probability_threshold",
        seed=123,
        threshold_action=None,
        probability_threshold=0.90,
    )

    assert result.policy_trace.actions == (
        0,
        0,
        0,
        0,
    )

    assert all(
        value is False
        for value in (
            result
            .policy_trace
            .threshold_met
        )
    )

    assert result.metrics.agent_return == pytest.approx(
        0.0
    )
    assert result.metrics.flat_exposure == pytest.approx(
        1.0
    )
    assert result.metrics.round_trips == 0


def test_runner_rejects_insufficient_lookback() -> None:
    with pytest.raises(
        EvaluationRunnerError,
        match="lookback_rows must be at least",
    ):
        run_ppo_evaluation(
            _model(
                "long_only",
                logits=[0.0, 2.0],
            ),
            _market_frame(),
            _environment_config(
                "long_only"
            ),
            evaluation_start_index=4,
            evaluation_end_index=8,
            lookback_rows=1,
            policy_mode="deterministic_argmax",
            seed=123,
        )


def test_benchmarks_include_open_and_close_fees() -> None:
    result = run_ppo_evaluation(
        _model(
            "long_only",
            logits=[2.0, 0.0],
        ),
        _market_frame(),
        _environment_config(
            "long_only"
        ),
        evaluation_start_index=4,
        evaluation_end_index=8,
        lookback_rows=2,
        policy_mode="deterministic_argmax",
        seed=123,
    )

    # Benchmark opens at Open[4] = 105 and is force-closed
    # at Close[7] = 100. Both entry and exit cost 1 bp.
    gross_long_return = (
        100.0 / 105.0 - 1.0
    )
    gross_short_return = (
        -gross_long_return
    )
    round_trip_fee = 2.0 / 10_000.0

    assert (
        result.metrics.always_long_return
        == pytest.approx(
            gross_long_return
            - round_trip_fee
        )
    )

    assert (
        result.metrics.always_short_return
        == pytest.approx(
            gross_short_return
            - round_trip_fee
        )
    )

    # Agent remains FLAT, but both benchmarks are still run.
    assert result.metrics.agent_return == pytest.approx(
        0.0
    )
    assert result.metrics.flat_exposure == pytest.approx(
        1.0
    )
