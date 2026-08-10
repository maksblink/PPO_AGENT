from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd
from stable_baselines3 import PPO

from train_and_eval.database.models import (
    EvaluationPolicyMode,
)
from train_and_eval.environment.contexts import (
    get_context_definition,
)
from train_and_eval.environment.trading_environment import (
    TradingEnvironment,
)
from train_and_eval.evaluation.metrics import (
    EvaluationMetrics,
    calculate_evaluation_metrics,
)
from train_and_eval.ppo.policy import (
    PPOPolicySelector,
)
from train_and_eval.run_config import (
    EnvironmentSection,
)


class EvaluationRunnerError(RuntimeError):
    """Raised when a PPO evaluation cannot run safely."""


@dataclass(frozen=True, slots=True)
class EvaluationPolicyTrace:
    """Policy decisions made during one evaluation."""

    execution_indices: tuple[int, ...]
    execution_timestamps: tuple[pd.Timestamp, ...]
    actions: tuple[int, ...]
    probabilities: tuple[
        tuple[float, ...],
        ...,
    ]
    selected_action_probabilities: tuple[
        float,
        ...,
    ]
    threshold_met: tuple[
        bool | None,
        ...,
    ]


@dataclass(frozen=True, slots=True)
class EvaluationTrajectory:
    """Per-step validation data sufficient to regenerate diagnostic plots."""

    execution_indices: tuple[int, ...]
    execution_timestamps: tuple[pd.Timestamp, ...]
    execution_prices: tuple[float, ...]
    close_prices: tuple[float, ...]
    actions: tuple[int, ...]
    positions: tuple[int, ...]
    agent_equity: tuple[float, ...]
    always_long_equity: tuple[float, ...]
    always_short_equity: tuple[float, ...]
    shaped_rewards: tuple[float, ...]
    fee_costs: tuple[float, ...]
    swap_costs: tuple[float, ...]
    trade_costs: tuple[float, ...]
    drawdowns: tuple[float, ...]
    hold_bars: tuple[int, ...]
    selected_action_probabilities: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class EvaluationRunResult:
    """Complete in-memory result of one PPO evaluation."""

    metrics: EvaluationMetrics
    steps_completed: int

    evaluation_start_index: int
    evaluation_end_index: int
    lookback_rows: int

    policy_trace: EvaluationPolicyTrace
    trajectory: EvaluationTrajectory
    trade_events: tuple[dict[str, Any], ...]


def _integer(
    value: Any,
    *,
    name: str,
    minimum: int,
) -> int:
    if isinstance(value, bool):
        raise EvaluationRunnerError(
            f"{name} must be an integer."
        )

    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise EvaluationRunnerError(
            f"{name} must be an integer."
        ) from error

    if result != value:
        raise EvaluationRunnerError(
            f"{name} must be an integer."
        )

    if result < minimum:
        raise EvaluationRunnerError(
            f"{name} must be at least {minimum}."
        )

    return result


def _evaluation_slice(
    market_data: pd.DataFrame,
    *,
    evaluation_start_index: int,
    evaluation_end_index: int,
    lookback_rows: int,
    required_history_rows: int,
) -> tuple[
    pd.DataFrame,
    int,
]:
    if market_data.empty:
        raise EvaluationRunnerError(
            "market_data must not be empty."
        )

    start = _integer(
        evaluation_start_index,
        name="evaluation_start_index",
        minimum=1,
    )
    end = _integer(
        evaluation_end_index,
        name="evaluation_end_index",
        minimum=1,
    )
    lookback = _integer(
        lookback_rows,
        name="lookback_rows",
        minimum=1,
    )

    if end <= start:
        raise EvaluationRunnerError(
            "evaluation_end_index must be greater "
            "than evaluation_start_index."
        )

    if end > len(market_data):
        raise EvaluationRunnerError(
            "evaluation_end_index exceeds the number "
            "of market-data rows."
        )

    if lookback < required_history_rows:
        raise EvaluationRunnerError(
            "lookback_rows must be at least "
            f"{required_history_rows} to provide a "
            "fully initialized observation context."
        )

    if start < lookback:
        raise EvaluationRunnerError(
            "evaluation_start_index does not leave "
            "enough earlier rows for lookback."
        )

    lookback_start = start - lookback

    frame = (
        market_data
        .iloc[lookback_start:end]
        .reset_index(drop=True)
    )
    frame.attrs = dict(
        market_data.attrs
    )

    local_observation_index = (
        lookback - 1
    )

    expected_rows = (
        lookback + end - start
    )

    if len(frame) != expected_rows:
        raise EvaluationRunnerError(
            "Evaluation data slice has an unexpected "
            "number of rows."
        )

    return (
        frame,
        local_observation_index,
    )


def _check_model_environment_compatibility(
    model: PPO,
    environment: TradingEnvironment,
) -> None:
    model_action_count = getattr(
        model.action_space,
        "n",
        None,
    )
    environment_action_count = getattr(
        environment.action_space,
        "n",
        None,
    )

    if model_action_count != environment_action_count:
        raise EvaluationRunnerError(
            "PPO model action space does not match "
            "the evaluation environment."
        )

    model_shape = getattr(
        model.observation_space,
        "shape",
        None,
    )
    environment_shape = getattr(
        environment.observation_space,
        "shape",
        None,
    )

    if model_shape != environment_shape:
        raise EvaluationRunnerError(
            "PPO model observation space does not "
            "match the evaluation environment."
        )


def run_ppo_evaluation(
    model: PPO,
    market_data: pd.DataFrame,
    environment_config: EnvironmentSection,
    *,
    evaluation_start_index: int,
    evaluation_end_index: int,
    lookback_rows: int,
    policy_mode: EvaluationPolicyMode | str,
    seed: int,
    threshold_action: int | None = None,
    probability_threshold: float | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    progress_interval_steps: int = 128,
) -> EvaluationRunResult:
    """
    Evaluate one PPO model over one exact execution range.

    The scored range is:

        [evaluation_start_index, evaluation_end_index)

    These are execution-candle indices in the original market data.
    The first policy observation is made at:

        evaluation_start_index - 1

    Earlier rows are used only as context lookback.
    """
    start = _integer(
        evaluation_start_index,
        name="evaluation_start_index",
        minimum=1,
    )
    end = _integer(
        evaluation_end_index,
        name="evaluation_end_index",
        minimum=1,
    )
    lookback = _integer(
        lookback_rows,
        name="lookback_rows",
        minimum=1,
    )
    resolved_seed = _integer(
        seed,
        name="seed",
        minimum=0,
    )

    definition = get_context_definition(
        environment_config.context
    )
    required_history_rows = (
        definition.required_history_rows(
            int(environment_config.window)
        )
    )

    (
        evaluation_data,
        local_observation_index,
    ) = _evaluation_slice(
        market_data,
        evaluation_start_index=start,
        evaluation_end_index=end,
        lookback_rows=lookback,
        required_history_rows=(
            required_history_rows
        ),
    )

    agent_environment = TradingEnvironment(
        evaluation_data,
        environment_config,
        start_index=local_observation_index,
    )

    always_long_environment = TradingEnvironment(
        evaluation_data,
        environment_config.model_copy(
            update={
                "position_side": "long_only",
            }
        ),
        start_index=local_observation_index,
    )

    always_short_environment = TradingEnvironment(
        evaluation_data,
        environment_config.model_copy(
            update={
                "position_side": "short_only",
            }
        ),
        start_index=local_observation_index,
    )

    _check_model_environment_compatibility(
        model,
        agent_environment,
    )

    selector = PPOPolicySelector(
        model=model,
        policy_mode=policy_mode,
        seed=resolved_seed,
        threshold_action=threshold_action,
        probability_threshold=(
            probability_threshold
        ),
    )

    observation, reset_info = (
        agent_environment.reset(
            seed=resolved_seed
        )
    )
    always_long_environment.reset(
        seed=resolved_seed
    )
    always_short_environment.reset(
        seed=resolved_seed
    )

    expected_initial_local_index = (
        local_observation_index
    )

    if int(
        reset_info["observation_index"]
    ) != expected_initial_local_index:
        raise EvaluationRunnerError(
            "Environment reset at an unexpected "
            "observation index."
        )

    expected_steps = end - start
    lookback_start = start - lookback
    resolved_progress_interval = _integer(
        progress_interval_steps,
        name="progress_interval_steps",
        minimum=1,
    )

    agent_equity: list[float] = []
    always_long_equity: list[float] = []
    always_short_equity: list[float] = []

    positions: list[int] = []
    shaped_rewards: list[float] = []
    step_swap_costs: list[float] = []
    step_swap_events: list[int] = []
    execution_prices: list[float] = []
    close_prices: list[float] = []
    fee_costs: list[float] = []
    trade_costs: list[float] = []
    drawdowns: list[float] = []
    hold_bars: list[int] = []

    execution_indices: list[int] = []
    execution_timestamps: list[
        pd.Timestamp
    ] = []
    actions: list[int] = []
    probabilities: list[
        tuple[float, ...]
    ] = []
    selected_probabilities: list[
        float
    ] = []
    threshold_results: list[
        bool | None
    ] = []

    final_agent_info: dict[
        str,
        Any,
    ] | None = None

    for step_offset in range(
        expected_steps
    ):
        decision = selector.select(
            observation
        )

        (
            observation,
            reward,
            agent_terminated,
            agent_truncated,
            agent_info,
        ) = agent_environment.step(
            decision.action
        )

        (
            _,
            _,
            long_terminated,
            long_truncated,
            long_info,
        ) = always_long_environment.step(1)

        (
            _,
            _,
            short_terminated,
            short_truncated,
            short_info,
        ) = always_short_environment.step(1)

        if (
            agent_truncated
            or long_truncated
            or short_truncated
        ):
            raise EvaluationRunnerError(
                "Evaluation environments must not "
                "truncate the episode."
            )

        expected_terminated = (
            step_offset
            == expected_steps - 1
        )

        if (
            agent_terminated
            != expected_terminated
            or long_terminated
            != expected_terminated
            or short_terminated
            != expected_terminated
        ):
            raise EvaluationRunnerError(
                "Evaluation environments terminated "
                "at an unexpected step."
            )

        agent_local_execution_index = int(
            agent_info["execution_index"]
        )
        long_local_execution_index = int(
            long_info["execution_index"]
        )
        short_local_execution_index = int(
            short_info["execution_index"]
        )

        if not (
            agent_local_execution_index
            == long_local_execution_index
            == short_local_execution_index
        ):
            raise EvaluationRunnerError(
                "Agent and benchmark environments "
                "executed on different indices."
            )

        global_execution_index = (
            lookback_start
            + agent_local_execution_index
        )
        expected_execution_index = (
            start + step_offset
        )

        if (
            global_execution_index
            != expected_execution_index
        ):
            raise EvaluationRunnerError(
                "Environment execution index does not "
                "match the requested evaluation range."
            )

        execution_timestamp = pd.Timestamp(
            agent_info[
                "execution_timestamp"
            ]
        )

        long_timestamp = pd.Timestamp(
            long_info[
                "execution_timestamp"
            ]
        )
        short_timestamp = pd.Timestamp(
            short_info[
                "execution_timestamp"
            ]
        )

        if not (
            execution_timestamp
            == long_timestamp
            == short_timestamp
        ):
            raise EvaluationRunnerError(
                "Agent and benchmark environments "
                "executed on different timestamps."
            )

        if float(
            long_info["execution_price"]
        ) != float(
            short_info["execution_price"]
        ):
            raise EvaluationRunnerError(
                "LONG and SHORT benchmarks used "
                "different execution prices."
            )

        if float(
            long_info["close_price"]
        ) != float(
            short_info["close_price"]
        ):
            raise EvaluationRunnerError(
                "LONG and SHORT benchmarks used "
                "different close prices."
            )

        if float(
            long_info["fee_cost"]
        ) != float(
            short_info["fee_cost"]
        ):
            raise EvaluationRunnerError(
                "LONG and SHORT benchmark fees "
                "are not symmetric."
            )

        if float(
            long_info["swap_cost"]
        ) != float(
            short_info["swap_cost"]
        ):
            raise EvaluationRunnerError(
                "LONG and SHORT benchmark swap costs "
                "are not symmetric."
            )

        if int(
            long_info["swap_events_on_step"]
        ) != int(
            short_info["swap_events_on_step"]
        ):
            raise EvaluationRunnerError(
                "LONG and SHORT benchmarks crossed "
                "different swap boundaries."
            )

        agent_equity.append(
            float(
                agent_info["equity"]
            )
        )
        always_long_equity.append(
            float(
                long_info["equity"]
            )
        )
        always_short_equity.append(
            float(
                short_info["equity"]
            )
        )

        positions.append(
            int(
                agent_info[
                    "position_during_bar"
                ]
            )
        )
        shaped_rewards.append(
            float(reward)
        )
        step_swap_costs.append(
            float(
                agent_info["swap_cost"]
            )
        )
        step_swap_events.append(
            int(
                agent_info[
                    "swap_events_on_step"
                ]
            )
        )
        execution_prices.append(float(agent_info["execution_price"]))
        close_prices.append(float(agent_info["close_price"]))
        fee_costs.append(float(agent_info["fee_cost"]))
        trade_costs.append(float(agent_info["trade_cost"]))
        drawdowns.append(float(agent_info["drawdown"]))
        hold_bars.append(int(agent_info["hold_bars"]))

        execution_indices.append(
            global_execution_index
        )
        execution_timestamps.append(
            execution_timestamp
        )
        actions.append(
            decision.action
        )
        probabilities.append(
            decision.probabilities
        )
        selected_probabilities.append(
            decision
            .selected_action_probability
        )
        threshold_results.append(
            decision.threshold_met
        )

        final_agent_info = agent_info

        completed_steps = step_offset + 1
        if (
            progress_callback is not None
            and (
                completed_steps == expected_steps
                or completed_steps % resolved_progress_interval == 0
            )
        ):
            progress_callback(completed_steps, expected_steps)

    if final_agent_info is None:
        raise EvaluationRunnerError(
            "Evaluation completed without any steps."
        )

    if len(agent_equity) != expected_steps:
        raise EvaluationRunnerError(
            "Evaluation produced an unexpected "
            "number of results."
        )

    metrics = calculate_evaluation_metrics(
        agent_equity=agent_equity,
        always_long_equity=(
            always_long_equity
        ),
        always_short_equity=(
            always_short_equity
        ),
        positions=positions,
        shaped_rewards=shaped_rewards,
        step_swap_costs=(
            step_swap_costs
        ),
        step_swap_events=(
            step_swap_events
        ),
        trade_events=(
            agent_environment.trade_events
        ),
        open_position_return_at_end=float(
            final_agent_info[
                "unrealized_return"
            ]
        ),
    )

    return EvaluationRunResult(
        metrics=metrics,
        steps_completed=expected_steps,
        evaluation_start_index=start,
        evaluation_end_index=end,
        lookback_rows=lookback,
        policy_trace=EvaluationPolicyTrace(
            execution_indices=tuple(execution_indices),
            execution_timestamps=tuple(execution_timestamps),
            actions=tuple(actions),
            probabilities=tuple(probabilities),
            selected_action_probabilities=tuple(selected_probabilities),
            threshold_met=tuple(threshold_results),
        ),
        trajectory=EvaluationTrajectory(
            execution_indices=tuple(execution_indices),
            execution_timestamps=tuple(execution_timestamps),
            execution_prices=tuple(execution_prices),
            close_prices=tuple(close_prices),
            actions=tuple(actions),
            positions=tuple(positions),
            agent_equity=tuple(agent_equity),
            always_long_equity=tuple(always_long_equity),
            always_short_equity=tuple(always_short_equity),
            shaped_rewards=tuple(shaped_rewards),
            fee_costs=tuple(fee_costs),
            swap_costs=tuple(step_swap_costs),
            trade_costs=tuple(trade_costs),
            drawdowns=tuple(drawdowns),
            hold_bars=tuple(hold_bars),
            selected_action_probabilities=tuple(selected_probabilities),
        ),
        trade_events=tuple(dict(event) for event in agent_environment.trade_events),
    )
