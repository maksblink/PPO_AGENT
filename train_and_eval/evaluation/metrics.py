from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isclose
from typing import Any, Literal, Mapping, Sequence

import numpy as np


TradeEvent = Mapping[str, Any]
CurrentStreakType = Literal[
    "none",
    "win",
    "loss",
    "breakeven",
]

_ALLOWED_EVENTS = {
    "OPEN_LONG",
    "OPEN_SHORT",
    "CLOSE_LONG",
    "CLOSE_SHORT",
}


class EvaluationMetricsError(ValueError):
    """Raised when evaluation data cannot produce trustworthy metrics."""


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    agent_return: float
    always_long_return: float
    always_short_return: float
    agent_vs_always_long_return: float
    balanced_score: float

    agent_max_drawdown: float
    always_long_max_drawdown: float
    always_short_max_drawdown: float
    drawdown_improvement: float

    net_exposure: float
    long_exposure: float
    short_exposure: float
    flat_exposure: float
    market_exposure: float

    trade_events_total: int
    trade_event_rate: float
    round_trips: int
    round_trip_rate: float

    winning_trades: int
    losing_trades: int
    breakeven_trades: int
    long_round_trips: int
    short_round_trips: int

    open_long_count: int
    open_short_count: int
    close_long_count: int
    close_short_count: int
    swap_events: int

    win_rate: float | None
    loss_rate: float | None
    breakeven_rate: float | None
    long_win_rate: float | None
    short_win_rate: float | None

    avg_trade_return: float | None
    avg_win_return: float | None
    avg_loss_return: float | None
    median_trade_return: float | None

    long_avg_trade_return: float | None
    short_avg_trade_return: float | None
    long_median_trade_return: float | None
    short_median_trade_return: float | None

    largest_win_return: float | None
    largest_loss_return: float | None

    gross_profit_return: float
    gross_loss_return: float
    net_profit_return: float

    profit_factor: float | None
    payoff_ratio: float | None

    min_bars_held: int | None
    avg_bars_held: float | None
    median_bars_held: float | None
    max_bars_held: int | None

    max_consecutive_wins: int
    max_consecutive_losses: int
    avg_win_streak: float | None
    avg_loss_streak: float | None
    current_streak_type: CurrentStreakType
    current_streak: int

    total_fee_return: float
    total_swap_return: float
    total_cost_return: float

    avg_fee_per_trade_return: float | None
    avg_swap_per_trade_return: float | None
    avg_cost_per_trade_return: float | None

    cumulative_shaped_reward: float
    open_position_return_at_end: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite_array(
    values: Sequence[float],
    *,
    name: str,
    allow_empty: bool = False,
) -> np.ndarray:
    try:
        array = np.asarray(
            values,
            dtype=np.float64,
        )
    except (TypeError, ValueError) as error:
        raise EvaluationMetricsError(
            f"{name} must contain numeric values."
        ) from error

    if array.ndim != 1:
        raise EvaluationMetricsError(
            f"{name} must be one-dimensional."
        )

    if not allow_empty and array.size == 0:
        raise EvaluationMetricsError(
            f"{name} must not be empty."
        )

    if not np.isfinite(array).all():
        raise EvaluationMetricsError(
            f"{name} contains NaN or infinity."
        )

    return array


def _finite_float(
    value: Any,
    *,
    name: str,
) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise EvaluationMetricsError(
            f"{name} must be numeric."
        ) from error

    if not np.isfinite(result):
        raise EvaluationMetricsError(
            f"{name} must be finite."
        )

    return result


def _nonnegative_float(
    value: Any,
    *,
    name: str,
) -> float:
    result = _finite_float(
        value,
        name=name,
    )

    if result < 0.0:
        raise EvaluationMetricsError(
            f"{name} must be nonnegative."
        )

    return result


def _nonnegative_int(
    value: Any,
    *,
    name: str,
) -> int:
    numeric = _finite_float(
        value,
        name=name,
    )

    if not numeric.is_integer():
        raise EvaluationMetricsError(
            f"{name} must be an integer."
        )

    result = int(numeric)

    if result < 0:
        raise EvaluationMetricsError(
            f"{name} must be nonnegative."
        )

    return result


def _required_event_value(
    event: TradeEvent,
    key: str,
    *,
    event_index: int,
) -> Any:
    if key not in event:
        raise EvaluationMetricsError(
            f"trade_events[{event_index}] "
            f"is missing {key!r}."
        )

    return event[key]


def _event_float(
    event: TradeEvent,
    key: str,
    *,
    event_index: int,
) -> float:
    value = _required_event_value(
        event,
        key,
        event_index=event_index,
    )

    return _finite_float(
        value,
        name=(
            f"trade_events[{event_index}]"
            f"[{key!r}]"
        ),
    )


def _event_nonnegative_float(
    event: TradeEvent,
    key: str,
    *,
    event_index: int,
) -> float:
    value = _required_event_value(
        event,
        key,
        event_index=event_index,
    )

    return _nonnegative_float(
        value,
        name=(
            f"trade_events[{event_index}]"
            f"[{key!r}]"
        ),
    )


def _event_int(
    event: TradeEvent,
    key: str,
    *,
    event_index: int,
) -> int:
    value = _required_event_value(
        event,
        key,
        event_index=event_index,
    )

    numeric = _finite_float(
        value,
        name=(
            f"trade_events[{event_index}]"
            f"[{key!r}]"
        ),
    )

    if not numeric.is_integer():
        raise EvaluationMetricsError(
            f"trade_events[{event_index}]"
            f"[{key!r}] must be an integer."
        )

    return int(numeric)


def _mean_or_none(
    values: Sequence[float],
) -> float | None:
    if not values:
        return None

    return float(np.mean(values))


def _median_or_none(
    values: Sequence[float],
) -> float | None:
    if not values:
        return None

    return float(np.median(values))


def _max_drawdown(
    equity: np.ndarray,
) -> float:
    curve = np.concatenate(
        (
            np.array([0.0]),
            equity,
        )
    )

    running_peak = np.maximum.accumulate(
        curve
    )

    drawdown = curve - running_peak

    return float(np.min(drawdown))


def _streak_lengths(
    signs: Sequence[int],
    *,
    target: int,
) -> list[int]:
    lengths: list[int] = []
    current = 0

    for sign in signs:
        if sign == target:
            current += 1
            continue

        if current > 0:
            lengths.append(current)
            current = 0

    if current > 0:
        lengths.append(current)

    return lengths


def _current_streak(
    signs: Sequence[int],
) -> tuple[CurrentStreakType, int]:
    if not signs:
        return "none", 0

    last_sign = signs[-1]
    length = 0

    for sign in reversed(signs):
        if sign != last_sign:
            break

        length += 1

    if last_sign > 0:
        return "win", length

    if last_sign < 0:
        return "loss", length

    return "breakeven", length


def calculate_evaluation_metrics(
    *,
    agent_equity: Sequence[float],
    always_long_equity: Sequence[float],
    always_short_equity: Sequence[float],
    positions: Sequence[int],
    shaped_rewards: Sequence[float],
    step_swap_costs: Sequence[float],
    step_swap_events: Sequence[int],
    trade_events: Sequence[TradeEvent],
    open_position_return_at_end: float,
) -> EvaluationMetrics:
    """
    Calculate canonical evaluation metrics.

    Equity curves contain cumulative decimal returns after every
    evaluation step. For example, 0.10 means +10%.

    Closed-trade statistics use CLOSE_LONG and CLOSE_SHORT events.
    """
    agent = _finite_array(
        agent_equity,
        name="agent_equity",
    )
    always_long = _finite_array(
        always_long_equity,
        name="always_long_equity",
    )
    always_short = _finite_array(
        always_short_equity,
        name="always_short_equity",
    )
    position_values = _finite_array(
        positions,
        name="positions",
    )
    rewards = _finite_array(
        shaped_rewards,
        name="shaped_rewards",
    )
    swap_costs = _finite_array(
        step_swap_costs,
        name="step_swap_costs",
    )
    swap_event_values = _finite_array(
        step_swap_events,
        name="step_swap_events",
    )

    steps_completed = int(agent.size)

    named_lengths = {
        "always_long_equity": always_long.size,
        "always_short_equity": always_short.size,
        "positions": position_values.size,
        "shaped_rewards": rewards.size,
        "step_swap_costs": swap_costs.size,
        "step_swap_events": swap_event_values.size,
    }

    for name, length in named_lengths.items():
        if length != steps_completed:
            raise EvaluationMetricsError(
                f"{name} length {length} does not match "
                f"agent_equity length {steps_completed}."
            )

    if not np.equal(
        position_values,
        np.floor(position_values),
    ).all():
        raise EvaluationMetricsError(
            "positions must contain integer values."
        )

    position_array = position_values.astype(
        np.int64
    )

    invalid_positions = set(
        np.unique(position_array).tolist()
    ) - {-1, 0, 1}

    if invalid_positions:
        raise EvaluationMetricsError(
            "positions contain unsupported values: "
            f"{sorted(invalid_positions)}."
        )

    if np.any(swap_costs < 0.0):
        raise EvaluationMetricsError(
            "step_swap_costs must be nonnegative."
        )

    if not np.equal(
        swap_event_values,
        np.floor(swap_event_values),
    ).all():
        raise EvaluationMetricsError(
            "step_swap_events must contain integers."
        )

    if np.any(swap_event_values < 0.0):
        raise EvaluationMetricsError(
            "step_swap_events must be nonnegative."
        )

    open_position_return = _finite_float(
        open_position_return_at_end,
        name="open_position_return_at_end",
    )

    events = list(trade_events)

    open_long_count = 0
    open_short_count = 0
    close_long_count = 0
    close_short_count = 0

    total_fee_return = 0.0

    closed_returns: list[float] = []
    long_returns: list[float] = []
    short_returns: list[float] = []
    bars_held_values: list[int] = []

    closed_fee_values: list[float] = []
    closed_swap_values: list[float] = []
    closed_cost_values: list[float] = []
    closed_swap_events = 0

    current_position = 0
    current_entry_fee: float | None = None

    expected_transitions = {
        "OPEN_LONG": (0, 1),
        "OPEN_SHORT": (0, -1),
        "CLOSE_LONG": (1, 0),
        "CLOSE_SHORT": (-1, 0),
    }

    for event_index, event in enumerate(events):
        if not isinstance(event, Mapping):
            raise EvaluationMetricsError(
                f"trade_events[{event_index}] "
                "must be a mapping."
            )

        event_name = str(
            event.get("event", "")
        )

        if event_name not in _ALLOWED_EVENTS:
            raise EvaluationMetricsError(
                f"Unknown trade event at index "
                f"{event_index}: {event_name!r}."
            )

        position_before = _event_int(
            event,
            "position_before",
            event_index=event_index,
        )
        position_after = _event_int(
            event,
            "position_after",
            event_index=event_index,
        )

        expected_before, expected_after = (
            expected_transitions[event_name]
        )

        if (
            position_before != expected_before
            or position_after != expected_after
        ):
            raise EvaluationMetricsError(
                f"Invalid transition for {event_name}: "
                f"{position_before} -> {position_after}."
            )

        if position_before != current_position:
            raise EvaluationMetricsError(
                f"Trade event sequence is inconsistent at "
                f"index {event_index}. Expected position "
                f"{current_position}, got {position_before}."
            )

        if event_name.startswith("OPEN_"):
            fee = _event_nonnegative_float(
                event,
                "fee",
                event_index=event_index,
            )

            total_fee_return += fee
            current_entry_fee = fee

            if event_name == "OPEN_LONG":
                open_long_count += 1
            else:
                open_short_count += 1

            current_position = position_after
            continue

        gross_return = _event_float(
            event,
            "gross_return",
            event_index=event_index,
        )
        net_return = _event_float(
            event,
            "net_return",
            event_index=event_index,
        )
        entry_fee = _event_nonnegative_float(
            event,
            "entry_fee",
            event_index=event_index,
        )
        close_fee = _event_nonnegative_float(
            event,
            "close_fee",
            event_index=event_index,
        )
        trade_swap_cost = (
            _event_nonnegative_float(
                event,
                "swap_cost",
                event_index=event_index,
            )
        )
        trade_swap_events = _nonnegative_int(
            _required_event_value(
                event,
                "swap_events",
                event_index=event_index,
            ),
            name=(
                f"trade_events[{event_index}]"
                "['swap_events']"
            ),
        )
        bars_held = _nonnegative_int(
            _required_event_value(
                event,
                "bars_held",
                event_index=event_index,
            ),
            name=(
                f"trade_events[{event_index}]"
                "['bars_held']"
            ),
        )

        if bars_held < 1:
            raise EvaluationMetricsError(
                f"trade_events[{event_index}]"
                "['bars_held'] must be at least 1."
            )

        if current_entry_fee is None:
            raise EvaluationMetricsError(
                f"Close event at index {event_index} "
                "has no matching open event."
            )

        if not isclose(
            entry_fee,
            current_entry_fee,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise EvaluationMetricsError(
                f"Close event at index {event_index} "
                "does not preserve its entry fee."
            )

        expected_net_return = (
            gross_return
            - entry_fee
            - close_fee
            - trade_swap_cost
        )

        if not isclose(
            net_return,
            expected_net_return,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise EvaluationMetricsError(
                f"Invalid net_return for trade event "
                f"at index {event_index}."
            )

        total_fee_return += close_fee

        closed_returns.append(net_return)
        bars_held_values.append(bars_held)

        trade_fee = entry_fee + close_fee
        trade_cost = (
            trade_fee + trade_swap_cost
        )

        closed_fee_values.append(trade_fee)
        closed_swap_values.append(
            trade_swap_cost
        )
        closed_cost_values.append(trade_cost)
        closed_swap_events += (
            trade_swap_events
        )

        if event_name == "CLOSE_LONG":
            close_long_count += 1
            long_returns.append(net_return)
        else:
            close_short_count += 1
            short_returns.append(net_return)

        current_position = position_after
        current_entry_fee = None

    total_swap_return = float(
        np.sum(swap_costs)
    )
    total_swap_events = int(
        np.sum(swap_event_values)
    )

    closed_swap_total = float(
        np.sum(closed_swap_values)
    )

    if closed_swap_total > (
        total_swap_return + 1e-12
    ):
        raise EvaluationMetricsError(
            "Closed-trade swap cost exceeds total "
            "step swap cost."
        )

    if closed_swap_events > total_swap_events:
        raise EvaluationMetricsError(
            "Closed-trade swap events exceed total "
            "step swap events."
        )

    total_cost_return = (
        total_fee_return
        + total_swap_return
    )

    round_trips = len(closed_returns)
    trade_events_total = len(events)

    wins = [
        value
        for value in closed_returns
        if value > 0.0
    ]
    losses = [
        value
        for value in closed_returns
        if value < 0.0
    ]
    breakeven = [
        value
        for value in closed_returns
        if value == 0.0
    ]

    winning_trades = len(wins)
    losing_trades = len(losses)
    breakeven_trades = len(breakeven)

    long_round_trips = len(long_returns)
    short_round_trips = len(short_returns)

    if round_trips > 0:
        win_rate = (
            winning_trades / round_trips
        )
        loss_rate = (
            losing_trades / round_trips
        )
        breakeven_rate = (
            breakeven_trades / round_trips
        )

        avg_trade_return = float(
            np.mean(closed_returns)
        )
        median_trade_return = float(
            np.median(closed_returns)
        )

        min_bars_held = min(
            bars_held_values
        )
        avg_bars_held = float(
            np.mean(bars_held_values)
        )
        median_bars_held = float(
            np.median(bars_held_values)
        )
        max_bars_held = max(
            bars_held_values
        )

        avg_fee_per_trade_return = float(
            np.mean(closed_fee_values)
        )
        avg_swap_per_trade_return = float(
            np.mean(closed_swap_values)
        )
        avg_cost_per_trade_return = float(
            np.mean(closed_cost_values)
        )
    else:
        win_rate = None
        loss_rate = None
        breakeven_rate = None

        avg_trade_return = None
        median_trade_return = None

        min_bars_held = None
        avg_bars_held = None
        median_bars_held = None
        max_bars_held = None

        avg_fee_per_trade_return = None
        avg_swap_per_trade_return = None
        avg_cost_per_trade_return = None

    long_win_rate = (
        sum(value > 0.0 for value in long_returns)
        / long_round_trips
        if long_round_trips > 0
        else None
    )
    short_win_rate = (
        sum(value > 0.0 for value in short_returns)
        / short_round_trips
        if short_round_trips > 0
        else None
    )

    avg_win_return = _mean_or_none(wins)
    avg_loss_return = _mean_or_none(losses)

    largest_win_return = (
        max(wins)
        if wins
        else None
    )
    largest_loss_return = (
        min(losses)
        if losses
        else None
    )

    gross_profit_return = float(
        np.sum(wins)
    )
    gross_loss_return = float(
        np.sum(losses)
    )
    net_profit_return = float(
        np.sum(closed_returns)
    )

    profit_factor = (
        gross_profit_return
        / abs(gross_loss_return)
        if losing_trades > 0
        else None
    )

    payoff_ratio = (
        float(avg_win_return)
        / abs(float(avg_loss_return))
        if (
            avg_win_return is not None
            and avg_loss_return is not None
        )
        else None
    )

    signs = [
        1 if value > 0.0
        else -1 if value < 0.0
        else 0
        for value in closed_returns
    ]

    win_streaks = _streak_lengths(
        signs,
        target=1,
    )
    loss_streaks = _streak_lengths(
        signs,
        target=-1,
    )

    max_consecutive_wins = (
        max(win_streaks)
        if win_streaks
        else 0
    )
    max_consecutive_losses = (
        max(loss_streaks)
        if loss_streaks
        else 0
    )

    avg_win_streak = _mean_or_none(
        win_streaks
    )
    avg_loss_streak = _mean_or_none(
        loss_streaks
    )

    (
        current_streak_type,
        current_streak,
    ) = _current_streak(signs)

    agent_return = float(agent[-1])
    always_long_return = float(
        always_long[-1]
    )
    always_short_return = float(
        always_short[-1]
    )

    agent_vs_always_long_return = (
        agent_return
        - always_long_return
    )

    agent_max_drawdown = _max_drawdown(
        agent
    )
    always_long_max_drawdown = (
        _max_drawdown(always_long)
    )
    always_short_max_drawdown = (
        _max_drawdown(always_short)
    )

    drawdown_improvement = (
        agent_max_drawdown
        - always_long_max_drawdown
    )

    balanced_score = (
        agent_vs_always_long_return
        + drawdown_improvement
    )

    long_exposure = float(
        np.mean(position_array == 1)
    )
    short_exposure = float(
        np.mean(position_array == -1)
    )
    flat_exposure = float(
        np.mean(position_array == 0)
    )
    market_exposure = float(
        np.mean(position_array != 0)
    )
    net_exposure = float(
        np.mean(position_array)
    )

    return EvaluationMetrics(
        agent_return=agent_return,
        always_long_return=always_long_return,
        always_short_return=always_short_return,
        agent_vs_always_long_return=(
            agent_vs_always_long_return
        ),
        balanced_score=balanced_score,
        agent_max_drawdown=agent_max_drawdown,
        always_long_max_drawdown=(
            always_long_max_drawdown
        ),
        always_short_max_drawdown=(
            always_short_max_drawdown
        ),
        drawdown_improvement=(
            drawdown_improvement
        ),
        net_exposure=net_exposure,
        long_exposure=long_exposure,
        short_exposure=short_exposure,
        flat_exposure=flat_exposure,
        market_exposure=market_exposure,
        trade_events_total=trade_events_total,
        trade_event_rate=(
            trade_events_total
            / steps_completed
        ),
        round_trips=round_trips,
        round_trip_rate=(
            round_trips
            / steps_completed
        ),
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        breakeven_trades=breakeven_trades,
        long_round_trips=long_round_trips,
        short_round_trips=short_round_trips,
        open_long_count=open_long_count,
        open_short_count=open_short_count,
        close_long_count=close_long_count,
        close_short_count=close_short_count,
        swap_events=total_swap_events,
        win_rate=win_rate,
        loss_rate=loss_rate,
        breakeven_rate=breakeven_rate,
        long_win_rate=long_win_rate,
        short_win_rate=short_win_rate,
        avg_trade_return=avg_trade_return,
        avg_win_return=avg_win_return,
        avg_loss_return=avg_loss_return,
        median_trade_return=(
            median_trade_return
        ),
        long_avg_trade_return=(
            _mean_or_none(long_returns)
        ),
        short_avg_trade_return=(
            _mean_or_none(short_returns)
        ),
        long_median_trade_return=(
            _median_or_none(long_returns)
        ),
        short_median_trade_return=(
            _median_or_none(short_returns)
        ),
        largest_win_return=(
            largest_win_return
        ),
        largest_loss_return=(
            largest_loss_return
        ),
        gross_profit_return=(
            gross_profit_return
        ),
        gross_loss_return=gross_loss_return,
        net_profit_return=net_profit_return,
        profit_factor=profit_factor,
        payoff_ratio=payoff_ratio,
        min_bars_held=min_bars_held,
        avg_bars_held=avg_bars_held,
        median_bars_held=(
            median_bars_held
        ),
        max_bars_held=max_bars_held,
        max_consecutive_wins=(
            max_consecutive_wins
        ),
        max_consecutive_losses=(
            max_consecutive_losses
        ),
        avg_win_streak=avg_win_streak,
        avg_loss_streak=avg_loss_streak,
        current_streak_type=(
            current_streak_type
        ),
        current_streak=current_streak,
        total_fee_return=total_fee_return,
        total_swap_return=total_swap_return,
        total_cost_return=total_cost_return,
        avg_fee_per_trade_return=(
            avg_fee_per_trade_return
        ),
        avg_swap_per_trade_return=(
            avg_swap_per_trade_return
        ),
        avg_cost_per_trade_return=(
            avg_cost_per_trade_return
        ),
        cumulative_shaped_reward=float(
            np.sum(rewards)
        ),
        open_position_return_at_end=(
            open_position_return
        ),
    )
