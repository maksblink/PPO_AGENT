from __future__ import annotations

from datetime import timedelta
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from train_and_eval.environment.contexts import (
    PreparedContext,
    build_context,
    build_observation,
    get_context_definition,
)
from train_and_eval.run_config import EnvironmentSection


POSITION_TARGETS = {
    "long_only": (0, 1),
    "short_only": (0, -1),
    "long_short": (0, 1, -1),
}

POSITION_ACTION_NAMES = {
    "long_only": ("FLAT", "LONG"),
    "short_only": ("FLAT", "SHORT"),
    "long_short": ("FLAT", "LONG", "SHORT"),
}


class TradingEnvironmentError(RuntimeError):
    """Raised when the trading environment cannot operate safely."""


def position_name(position: int) -> str:
    names = {
        -1: "SHORT",
        0: "FLAT",
        1: "LONG",
    }

    try:
        return names[int(position)]
    except KeyError as error:
        raise TradingEnvironmentError(
            f"Unknown position value: {position}"
        ) from error


def action_targets(
    position_side: str,
) -> tuple[int, ...]:
    try:
        return POSITION_TARGETS[position_side]
    except KeyError as error:
        raise TradingEnvironmentError(
            f"Unsupported position_side: {position_side!r}"
        ) from error


def action_names(
    position_side: str,
) -> tuple[str, ...]:
    try:
        return POSITION_ACTION_NAMES[position_side]
    except KeyError as error:
        raise TradingEnvironmentError(
            f"Unsupported position_side: {position_side!r}"
        ) from error


def _local_timestamp(
    timestamp: object,
    timezone: str,
) -> pd.Timestamp:
    value = pd.Timestamp(timestamp)

    if value.tzinfo is None:
        value = value.tz_localize("UTC")

    return value.tz_convert(timezone)


def _parse_hh_mm(
    value: str,
) -> tuple[int, int]:
    hours, minutes = value.split(":", 1)
    return int(hours), int(minutes)


def count_swap_boundaries(
    before: object,
    after: object,
    *,
    swap_time: str,
    swap_timezone: str,
) -> int:
    """
    Count every daily rollover boundary in the interval (before, after].

    Saturday and Sunday boundaries are intentionally counted. There is no
    weekend shortcut or cap.
    """
    before_local = _local_timestamp(
        before,
        swap_timezone,
    )
    after_local = _local_timestamp(
        after,
        swap_timezone,
    )

    if after_local <= before_local:
        return 0

    hours, minutes = _parse_hh_mm(
        swap_time
    )

    current_date = before_local.date()
    final_date = after_local.date()
    count = 0

    while current_date <= final_date:
        boundary = pd.Timestamp(
            year=current_date.year,
            month=current_date.month,
            day=current_date.day,
            hour=hours,
            minute=minutes,
            tz=swap_timezone,
        )

        if before_local < boundary <= after_local:
            count += 1

        current_date += timedelta(days=1)

    return count


class TradingEnvironment(
    gym.Env[np.ndarray, int]
):
    """
    PPO trading environment using next-open execution.

    Observation:
        produced after candle i has closed.

    Action:
        selected from the observation at candle i.

    Execution:
        performed at Open of candle i+1.

    Reward:
        measured from equity at Close[i] to equity at Close[i+1].
    """

    metadata = {
        "render_modes": [],
    }

    def __init__(
        self,
        market_data: pd.DataFrame,
        config: EnvironmentSection,
        *,
        start_index: int | None = None,
    ) -> None:
        super().__init__()

        self.config = config
        self.window = int(config.window)
        self.start_index = start_index

        definition = get_context_definition(
            config.context
        )
        self.required_history_rows = (
            definition.required_history_rows(
                self.window
            )
        )
        self.minimum_start_index = (
            self.required_history_rows - 1
        )

        minimum_rows = (
            self.required_history_rows + 1
        )

        if len(market_data) < minimum_rows:
            raise TradingEnvironmentError(
                "Market data is too short for one complete "
                "observation context and one next-open step. "
                "Required rows: "
                f"{minimum_rows}, rows: {len(market_data)}."
            )

        self.prepared_context: PreparedContext = (
            build_context(
                config.context,
                market_data,
                market_timezone=config.market_timezone,
                rth_open=config.rth_open,
                rth_close=config.rth_close,
                swap_time=config.swap_time,
                swap_timezone=config.swap_timezone,
            )
        )

        self.frame = (
            self.prepared_context.frame
            .reset_index(drop=True)
        )

        self.open_prices = self.frame[
            "Open"
        ].to_numpy(dtype=np.float64)

        self.close_prices = self.frame[
            "Close"
        ].to_numpy(dtype=np.float64)

        self.timestamps = self.frame[
            "DT"
        ].to_numpy()

        self._action_targets = action_targets(
            config.position_side
        )
        self._action_names = action_names(
            config.position_side
        )

        self.action_space = spaces.Discrete(
            len(self._action_targets)
        )

        observation_size = (
            self.prepared_context
            .definition
            .observation_size(self.window)
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_size,),
            dtype=np.float32,
        )

        self.fee_rate = (
            float(config.fee_bps)
            / 10_000.0
        )
        self.swap_long_rate = (
            float(config.swap_long_bps)
            / 10_000.0
        )

        self.swap_short_rate = float(config.swap_short_bps) / 10_000.0

        self._reset_state(
            self._resolve_start_index(
                start_index
            )
        )

    def _resolve_start_index(
        self,
        requested_index: int | None,
    ) -> int:
        minimum_index = (
            self.minimum_start_index
        )

        if requested_index is None:
            index = minimum_index
        else:
            index = int(requested_index)

        if index < minimum_index:
            raise TradingEnvironmentError(
                "start_index does not leave enough rows "
                "for a complete observation context. "
                "Required history rows: "
                f"{self.required_history_rows}, "
                f"received start_index: {index}."
            )

        if index >= len(self.frame) - 1:
            raise TradingEnvironmentError(
                "start_index leaves no candle available "
                "for next-open execution."
            )

        return index

    def _reset_state(
        self,
        start_index: int,
    ) -> None:
        self.current_index = int(start_index)
        self.position = 0

        self.entry_price = 0.0
        self.entry_index: int | None = None
        self.entry_timestamp: object | None = None
        self.entry_fee_paid = 0.0

        self.position_swap_cost = 0.0
        self.position_swap_events = 0

        self.realized_return = 0.0
        self.equity_peak = 0.0

        self.hold_bars = 0
        self.steps = 0
        self.total_swap_events = 0

        self.trade_events: list[
            dict[str, Any]
        ] = []

        self.last_action: int | None = None
        self.last_reward = 0.0
        self.last_raw_delta = 0.0
        self.last_info: dict[str, Any] = {}

        self._terminated = False

    def _current_close(
        self,
    ) -> float:
        return float(
            self.close_prices[
                self.current_index
            ]
        )

    def _unrealized_return(
        self,
        *,
        price: float | None = None,
    ) -> float:
        if self.position == 0:
            return 0.0

        current_price = (
            self._current_close()
            if price is None
            else float(price)
        )

        return float(
            self.position
            * (
                current_price
                / max(
                    abs(self.entry_price),
                    1e-12,
                )
                - 1.0
            )
        )

    def _equity(
        self,
        *,
        price: float | None = None,
    ) -> float:
        return float(
            self.realized_return
            + self._unrealized_return(
                price=price
            )
        )

    def _drawdown(
        self,
        equity: float,
    ) -> float:
        return float(
            max(
                0.0,
                self.equity_peak - equity,
            )
        )

    def _observation(
        self,
    ) -> np.ndarray:
        return build_observation(
            self.prepared_context,
            current_index=self.current_index,
            window=self.window,
            position=self.position,
            unrealized_pnl=(
                self._unrealized_return()
            ),
        )

    def _log_event(
        self,
        *,
        event: str,
        index: int,
        price: float,
        action: str,
        position_before: int,
        position_after: int,
        **details: Any,
    ) -> None:
        record: dict[str, Any] = {
            "event": event,
            "index": int(index),
            "timestamp": self.timestamps[
                index
            ],
            "price": float(price),
            "action": action,
            "position_before": int(
                position_before
            ),
            "position_after": int(
                position_after
            ),
        }

        record.update(details)
        self.trade_events.append(record)

    def _open_position(
        self,
        target_position: int,
        *,
        price: float,
        index: int,
        action: str,
    ) -> float:
        if target_position == 0:
            return 0.0

        position_before = self.position
        fee = self.fee_rate

        self.realized_return -= fee

        self.position = int(
            target_position
        )
        self.entry_price = float(price)
        self.entry_index = int(index)
        self.entry_timestamp = (
            self.timestamps[index]
        )
        self.entry_fee_paid = fee

        self.position_swap_cost = 0.0
        self.position_swap_events = 0
        self.hold_bars = 0

        self._log_event(
            event=(
                "OPEN_LONG"
                if target_position == 1
                else "OPEN_SHORT"
            ),
            index=index,
            price=price,
            action=action,
            position_before=position_before,
            position_after=target_position,
            fee=fee,
        )

        return fee

    def _close_position(
        self,
        *,
        price: float,
        index: int,
        action: str,
    ) -> tuple[float, float]:
        if self.position == 0:
            return 0.0, 0.0

        old_position = int(self.position)
        close_fee = self.fee_rate

        gross_return = float(
            old_position
            * (
                float(price)
                / max(
                    abs(self.entry_price),
                    1e-12,
                )
                - 1.0
            )
        )

        net_trade_return = float(
            gross_return
            - self.entry_fee_paid
            - close_fee
            - self.position_swap_cost
        )

        self.realized_return += (
            gross_return - close_fee
        )

        entry_index = self.entry_index
        entry_timestamp = (
            self.entry_timestamp
        )
        entry_price = self.entry_price
        entry_fee = self.entry_fee_paid
        bars_held = self.hold_bars
        swap_cost = (
            self.position_swap_cost
        )
        swap_events = (
            self.position_swap_events
        )

        self.position = 0
        self.entry_price = 0.0
        self.entry_index = None
        self.entry_timestamp = None
        self.entry_fee_paid = 0.0

        self.position_swap_cost = 0.0
        self.position_swap_events = 0
        self.hold_bars = 0

        self._log_event(
            event=(
                "CLOSE_LONG"
                if old_position == 1
                else "CLOSE_SHORT"
            ),
            index=index,
            price=price,
            action=action,
            position_before=old_position,
            position_after=0,
            entry_index=entry_index,
            entry_timestamp=entry_timestamp,
            entry_price=entry_price,
            exit_index=index,
            exit_timestamp=(
                self.timestamps[index]
            ),
            gross_return=gross_return,
            net_return=net_trade_return,
            net_return_pln=(
                net_trade_return
                * float(
                    self.config.stake_pln
                )
            ),
            entry_fee=entry_fee,
            close_fee=close_fee,
            swap_cost=swap_cost,
            swap_events=swap_events,
            bars_held=bars_held,
        )

        return (
            net_trade_return,
            close_fee,
        )

    def _apply_swap(
        self,
        *,
        before_index: int,
        after_index: int,
    ) -> tuple[float, int]:
        if self.position == 0:
            return 0.0, 0
        swap_rate = self.swap_long_rate if self.position > 0 else self.swap_short_rate
        if swap_rate <= 0.0:
            return 0.0, 0

        count = count_swap_boundaries(
            self.timestamps[before_index],
            self.timestamps[after_index],
            swap_time=self.config.swap_time,
            swap_timezone=(
                self.config.swap_timezone
            ),
        )

        if count == 0:
            return 0.0, 0

        cost = (
            swap_rate
            * float(count)
        )

        self.realized_return -= cost
        self.position_swap_cost += cost
        self.position_swap_events += count
        self.total_swap_events += count

        return float(cost), int(count)

    def _target_for_action(
        self,
        action: int,
    ) -> int:
        if not self.action_space.contains(
            action
        ):
            raise ValueError(
                f"Invalid action {action} for "
                f"position_side="
                f"{self.config.position_side!r}. "
                f"Allowed actions: "
                f"{tuple(range(self.action_space.n))}"
            )

        return int(
            self._action_targets[
                int(action)
            ]
        )

    def _apply_target(
        self,
        *,
        action: int,
        price: float,
        index: int,
    ) -> tuple[
        float,
        float,
        int,
        int,
    ]:
        target = self._target_for_action(
            action
        )
        old_position = int(self.position)

        if target == old_position:
            return (
                0.0,
                0.0,
                0,
                target,
            )

        action_label = self._action_names[
            int(action)
        ]

        fee_cost = 0.0
        realized_on_step = 0.0

        if old_position != 0:
            (
                closed_trade_return,
                close_fee,
            ) = self._close_position(
                price=price,
                index=index,
                action=action_label,
            )

            realized_on_step += (
                closed_trade_return
            )
            fee_cost += close_fee

        if target != 0:
            fee_cost += self._open_position(
                target,
                price=price,
                index=index,
                action=action_label,
            )

        turnover = abs(
            target - old_position
        )

        return (
            float(fee_cost),
            float(realized_on_step),
            int(turnover),
            int(target),
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[
        np.ndarray,
        dict[str, Any],
    ]:
        super().reset(seed=seed)

        requested_index = self.start_index

        if (
            options is not None
            and "start_index" in options
        ):
            requested_index = int(
                options["start_index"]
            )

        resolved_index = (
            self._resolve_start_index(
                requested_index
            )
        )

        self._reset_state(
            resolved_index
        )

        observation = self._observation()

        info = {
            "observation_index": (
                self.current_index
            ),
            "observation_timestamp": (
                self.timestamps[
                    self.current_index
                ]
            ),
            "position": self.position,
            "position_name": (
                position_name(
                    self.position
                )
            ),
            "equity": self._equity(),
            "required_history_rows": (
                self.required_history_rows
            ),
        }

        self.last_info = dict(info)

        return observation, info

    def step(
        self,
        action: int,
    ) -> tuple[
        np.ndarray,
        float,
        bool,
        bool,
        dict[str, Any],
    ]:
        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = self._step_core(
            action,
            build_observation=True,
        )

        if observation is None:
            raise TradingEnvironmentError(
                "Regular step() did not produce "
                "an observation."
            )

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )

    def step_without_observation(
        self,
        action: int,
    ) -> tuple[
        float,
        bool,
        bool,
        dict[str, Any],
    ]:
        (
            _,
            reward,
            terminated,
            truncated,
            info,
        ) = self._step_core(
            action,
            build_observation=False,
        )

        return (
            reward,
            terminated,
            truncated,
            info,
        )

    def _step_core(
        self,
        action: int,
        *,
        build_observation: bool,
    ) -> tuple[
        np.ndarray | None,
        float,
        bool,
        bool,
        dict[str, Any],
    ]:
        if self._terminated:
            raise TradingEnvironmentError(
                "step() called after the episode "
                "has terminated. Call reset()."
            )

        action = int(action)
        self.last_action = action

        observation_index = int(
            self.current_index
        )
        execution_index = (
            observation_index + 1
        )

        equity_before = self._equity()
        drawdown_before = self._drawdown(
            equity_before
        )

        swap_cost, swap_events = (
            self._apply_swap(
                before_index=observation_index,
                after_index=execution_index,
            )
        )

        execution_price = float(
            self.open_prices[
                execution_index
            ]
        )

        (
            fee_cost,
            realized_on_step,
            turnover,
            target_position,
        ) = self._apply_target(
            action=action,
            price=execution_price,
            index=execution_index,
        )

        position_during_bar = int(
            self.position
        )

        if position_during_bar != 0:
            self.hold_bars += 1

        self.current_index = (
            execution_index
        )
        self.steps += 1

        terminated = (
            self.current_index
            >= len(self.frame) - 1
        )
        truncated = False
        force_closed = False

        if (
            terminated
            and self.config.force_close_on_done
            and self.position != 0
        ):
            (
                final_trade_return,
                final_close_fee,
            ) = self._close_position(
                price=self._current_close(),
                index=self.current_index,
                action="FORCE_CLOSE_END",
            )

            realized_on_step += (
                final_trade_return
            )
            fee_cost += final_close_fee
            force_closed = True

        equity_after = self._equity()

        self.equity_peak = max(
            self.equity_peak,
            equity_after,
        )

        drawdown_after = self._drawdown(
            equity_after
        )

        drawdown_increase = max(
            0.0,
            drawdown_after
            - drawdown_before,
        )

        raw_delta = float(
            equity_after
            - equity_before
        )

        if raw_delta >= 0.0:
            reward_delta = (
                raw_delta
                * float(
                    self.config
                    .profit_reward_mult
                )
            )
        else:
            reward_delta = (
                raw_delta
                * float(
                    self.config
                    .loss_reward_mult
                )
            )

        reward = (
            reward_delta
            * float(
                self.config.reward_scale
            )
        )

        reward -= (
            float(
                self.config
                .exposure_penalty
            )
            * abs(position_during_bar)
        )

        reward -= (
            float(
                self.config
                .turnover_penalty
            )
            * float(turnover)
        )

        reward -= (
            float(
                self.config
                .drawdown_penalty
            )
            * drawdown_increase
            * float(
                self.config.reward_scale
            )
        )

        trade_cost = (
            fee_cost + swap_cost
        )

        observation = (
            self._observation()
            if build_observation
            else None
        )

        info: dict[str, Any] = {
            "observation_index": (
                self.current_index
            ),
            "observation_timestamp": (
                self.timestamps[
                    self.current_index
                ]
            ),
            "execution_index": (
                execution_index
            ),
            "execution_timestamp": (
                self.timestamps[
                    execution_index
                ]
            ),
            "execution_price": (
                execution_price
            ),
            "close_price": (
                self._current_close()
            ),
            "action": action,
            "action_name": (
                self._action_names[action]
            ),
            "target_position": (
                target_position
            ),
            "position": self.position,
            "position_name": (
                position_name(
                    self.position
                )
            ),
            "position_during_bar": (
                position_during_bar
            ),
            "entry_price": (
                self.entry_price
            ),
            "realized_return": (
                self.realized_return
            ),
            "unrealized_return": (
                self._unrealized_return()
            ),
            "equity": equity_after,
            "equity_pct": (
                equity_after * 100.0
            ),
            "equity_pln": (
                equity_after
                * float(
                    self.config.stake_pln
                )
            ),
            "raw_delta": raw_delta,
            "reward_delta": (
                reward_delta
            ),
            "reward": float(reward),
            "fee_cost": float(
                fee_cost
            ),
            "swap_cost": float(
                swap_cost
            ),
            "swap_events_on_step": int(
                swap_events
            ),
            "total_swap_events": int(
                self.total_swap_events
            ),
            "trade_cost": float(
                trade_cost
            ),
            "realized_on_step": float(
                realized_on_step
            ),
            "turnover": int(turnover),
            "hold_bars": int(
                self.hold_bars
            ),
            "drawdown": float(
                drawdown_after
            ),
            "drawdown_increase": float(
                drawdown_increase
            ),
            "force_closed": (
                force_closed
            ),
            "steps": self.steps,
        }

        self.last_raw_delta = (
            raw_delta
        )
        self.last_reward = float(
            reward
        )
        self.last_info = dict(info)
        self._terminated = terminated

        return (
            observation,
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    def get_current_observation(
        self,
    ) -> np.ndarray:
        return self._observation()

    def get_debug_state(
        self,
    ) -> dict[str, Any]:
        return {
            "current_index": (
                self.current_index
            ),
            "timestamp": (
                self.timestamps[
                    self.current_index
                ]
            ),
            "position": self.position,
            "position_name": (
                position_name(
                    self.position
                )
            ),
            "entry_price": (
                self.entry_price
            ),
            "realized_return": (
                self.realized_return
            ),
            "unrealized_return": (
                self._unrealized_return()
            ),
            "equity": self._equity(),
            "equity_peak": (
                self.equity_peak
            ),
            "hold_bars": (
                self.hold_bars
            ),
            "total_swap_events": (
                self.total_swap_events
            ),
            "last_action": (
                self.last_action
            ),
            "last_reward": (
                self.last_reward
            ),
            "last_raw_delta": (
                self.last_raw_delta
            ),
            "last_info": dict(
                self.last_info
            ),
            "trade_events": list(
                self.trade_events
            ),
            "action_names": (
                self._action_names
            ),
        }
