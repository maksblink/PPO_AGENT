from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd



REQUIRED_MARKET_COLUMNS = (
    "DT",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
)

TIME_FEATURES = (
    "tod_sin",
    "tod_cos",
    "dow_sin",
    "dow_cos",
    "rth_active",
    "rth_sin_half",
    "rth_cos_half",
    "overnight_active",
    "overnight_sin_half",
    "overnight_cos_half",
    "swap_pre_active",
    "swap_pre_sin_half",
    "swap_pre_cos_half",
)

STATE_FEATURES = (
    "position",
    "unrealized_pnl",
)

SUPPORTED_WINDOW_FEATURES = frozenset(
    {
        "direction",
        "range",
        "body",
        "upper_wick",
        "lower_wick",
        "vol_chg",
    }
)

ROLLING_FEATURE_PREFIXES = {
    "log_return": "log_ret",
    "volatility": "vol",
    "range_position": "range_pos",
    "close_vs_sma": "close_vs_sma",
    "drawdown_from_high": "dd_from_high",
    "rebound_from_low": "rebound_from_low",
}

SUPPORTED_ROLLING_FEATURES = frozenset(
    ROLLING_FEATURE_PREFIXES
)

# Number of source rows additionally required beyond the horizon
# itself to produce one fully initialized value.
#
# A horizon-H log return compares the current close with close.shift(H),
# therefore it needs H earlier rows plus the current row. Volatility over
# H one-bar returns likewise needs H + 1 closes. The remaining rolling
# features need H rows including the current row.
ROLLING_FEATURE_EXTRA_SOURCE_ROWS = {
    "log_return": 1,
    "volatility": 1,
    "range_position": 0,
    "close_vs_sma": 0,
    "drawdown_from_high": 0,
    "rebound_from_low": 0,
}


class ContextError(ValueError):
    """Raised when an observation context cannot be created."""


@dataclass(frozen=True)
class ContextDefinition:
    """Stable description of information visible to the agent."""

    name: str
    horizons: tuple[int, ...]
    window_features: tuple[str, ...]
    rolling_features: tuple[str, ...]
    context_features: tuple[str, ...]
    state_features: tuple[str, ...]

    def required_history_rows(self, window: int) -> int:
        """
        Return source rows needed for one fully initialized observation.

        The count includes the currently closed candle. For example, a
        log-return horizon of 6144 needs 6145 rows: 6144 earlier closes
        plus the current close.
        """
        if window <= 0:
            raise ContextError(
                "Window must be greater than zero."
            )

        required_rows = int(window)

        for feature_name in self.rolling_features:
            extra_rows = (
                ROLLING_FEATURE_EXTRA_SOURCE_ROWS[
                    feature_name
                ]
            )

            for horizon in self.horizons:
                required_rows = max(
                    required_rows,
                    int(horizon) + extra_rows,
                )

        return required_rows

    def observation_size(self, window: int) -> int:
        if window <= 0:
            raise ContextError(
                "Window must be greater than zero."
            )

        return (
            window * len(self.window_features)
            + len(self.context_features)
            + len(self.state_features)
        )


@dataclass(frozen=True)
class PreparedContext:
    """Precalculated static features used to build observations."""

    frame: pd.DataFrame
    definition: ContextDefinition
    window_values: np.ndarray
    context_values: np.ndarray


class MarketContext:
    """
    Base class for observation contexts.

    A normal new context only overrides class attributes. The build method
    needs to be overridden only when a genuinely new feature calculation
    cannot be expressed using the shared feature library.
    """

    name: ClassVar[str]

    horizons: ClassVar[tuple[int, ...]] = ()
    window_features: ClassVar[tuple[str, ...]] = ()
    rolling_features: ClassVar[tuple[str, ...]] = ()

    include_time_features: ClassVar[bool] = True

    state_features: ClassVar[tuple[str, ...]] = (
        STATE_FEATURES
    )

    @classmethod
    def validate_specification(cls) -> None:
        if not getattr(cls, "name", ""):
            raise ContextError(
                f"{cls.__name__} must define a context name."
            )

        if len(set(cls.horizons)) != len(cls.horizons):
            raise ContextError(
                f"{cls.name}: horizons cannot contain duplicates."
            )

        if any(
            horizon <= 0
            for horizon in cls.horizons
        ):
            raise ContextError(
                f"{cls.name}: every horizon must be positive."
            )

        unknown_window_features = (
            set(cls.window_features)
            - SUPPORTED_WINDOW_FEATURES
        )

        if unknown_window_features:
            raise ContextError(
                f"{cls.name}: unsupported window features: "
                f"{sorted(unknown_window_features)}"
            )

        unknown_rolling_features = (
            set(cls.rolling_features)
            - SUPPORTED_ROLLING_FEATURES
        )

        if unknown_rolling_features:
            raise ContextError(
                f"{cls.name}: unsupported rolling features: "
                f"{sorted(unknown_rolling_features)}"
            )

        if cls.rolling_features and not cls.horizons:
            raise ContextError(
                f"{cls.name}: rolling features require horizons."
            )

    @classmethod
    def rolling_feature_columns(
        cls,
    ) -> tuple[str, ...]:
        cls.validate_specification()

        return tuple(
            f"{ROLLING_FEATURE_PREFIXES[feature]}_{horizon}"
            for feature in cls.rolling_features
            for horizon in cls.horizons
        )

    @classmethod
    def context_feature_names(
        cls,
    ) -> tuple[str, ...]:
        feature_names = cls.rolling_feature_columns()

        if cls.include_time_features:
            feature_names += TIME_FEATURES

        return feature_names

    @classmethod
    def definition(cls) -> ContextDefinition:
        cls.validate_specification()

        return ContextDefinition(
            name=cls.name,
            horizons=cls.horizons,
            window_features=cls.window_features,
            rolling_features=cls.rolling_features,
            context_features=cls.context_feature_names(),
            state_features=cls.state_features,
        )

    def build(
        self,
        market_data: pd.DataFrame,
        *,
        market_timezone: str,
        rth_open: str,
        rth_close: str,
        swap_time: str,
        swap_timezone: str,
    ) -> PreparedContext:
        """
        Build the context using the shared feature calculators.

        A specialized context may override this method while preserving
        the same registry and environment interface.
        """
        self.validate_specification()

        frame = _prepare_source_frame(market_data)

        _add_window_features(
            frame,
            feature_names=self.window_features,
        )

        _add_rolling_features(
            frame,
            feature_names=self.rolling_features,
            horizons=self.horizons,
        )

        if self.include_time_features:
            _add_time_features(
                frame,
                market_timezone=market_timezone,
                rth_open=rth_open,
                rth_close=rth_close,
                swap_time=swap_time,
                swap_timezone=swap_timezone,
            )

        all_feature_columns = (
            self.window_features
            + self.context_feature_names()
        )

        for column in all_feature_columns:
            frame[column] = _clean_feature(
                frame[column]
            )

        definition = self.definition()

        return PreparedContext(
            frame=frame,
            definition=definition,
            window_values=frame.loc[
                :,
                list(definition.window_features),
            ].to_numpy(
                dtype=np.float32,
                copy=True,
            ),
            context_values=frame.loc[
                :,
                list(definition.context_features),
            ].to_numpy(
                dtype=np.float32,
                copy=True,
            ),
        )


CONTEXT_TYPES: dict[
    str,
    type[MarketContext],
] = {}


def register_context(
    context_type: type[MarketContext],
) -> type[MarketContext]:
    """Register a concrete context class under its YAML name."""
    context_type.validate_specification()

    if context_type.name in CONTEXT_TYPES:
        raise ContextError(
            f"Duplicate context name: {context_type.name!r}"
        )

    CONTEXT_TYPES[context_type.name] = context_type
    return context_type


@register_context
class BaselineMultiscaleV1Context(MarketContext):
    """Baseline context migrated from the old V4 feature set."""

    name = "baseline_multiscale_v1"

    horizons = (
        3,
        6,
        12,
        24,
        48,
        96,
        192,
        384,
        768,
        1536,
        3072,
        6144,
    )

    window_features = (
        "direction",
        "range",
        "body",
        "upper_wick",
        "lower_wick",
        "vol_chg",
    )

    rolling_features = (
        "log_return",
        "volatility",
        "range_position",
        "close_vs_sma",
        "drawdown_from_high",
        "rebound_from_low",
    )

    include_time_features = True


AVAILABLE_CONTEXTS = tuple(CONTEXT_TYPES)


def get_context_type(
    name: str,
) -> type[MarketContext]:
    try:
        return CONTEXT_TYPES[name]
    except KeyError as error:
        available = ", ".join(AVAILABLE_CONTEXTS)

        raise ContextError(
            f"Unknown context {name!r}. "
            f"Available: {available}"
        ) from error


def get_context_definition(
    name: str,
) -> ContextDefinition:
    return get_context_type(name).definition()


def build_context(
    name: str,
    market_data: pd.DataFrame,
    *,
    market_timezone: str,
    rth_open: str,
    rth_close: str,
    swap_time: str,
    swap_timezone: str,
) -> PreparedContext:
    context = get_context_type(name)()

    return context.build(
        market_data,
        market_timezone=market_timezone,
        rth_open=rth_open,
        rth_close=rth_close,
        swap_time=swap_time,
        swap_timezone=swap_timezone,
    )


def _parse_hh_mm(value: str) -> int:
    hours, minutes = value.split(":", 1)
    return int(hours) * 60 + int(minutes)


def _clean_feature(
    values: pd.Series | np.ndarray,
    *,
    index: pd.Index | None = None,
) -> pd.Series:
    if isinstance(values, pd.Series):
        series = values.copy()
    else:
        series = pd.Series(values, index=index)

    return (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], 0.0)
        .fillna(0.0)
        .astype(np.float32)
    )


def _prepare_source_frame(
    market_data: pd.DataFrame,
) -> pd.DataFrame:
    missing = [
        column
        for column in REQUIRED_MARKET_COLUMNS
        if column not in market_data.columns
    ]

    if missing:
        raise ContextError(
            f"Market data is missing required columns: {missing}"
        )

    if market_data.empty:
        raise ContextError("Market data contains no rows.")

    frame = market_data.loc[:, REQUIRED_MARKET_COLUMNS].copy()
    frame = frame.reset_index(drop=True)

    frame["DT"] = pd.to_datetime(
        frame["DT"],
        utc=True,
        errors="raise",
    )

    for column in ("Open", "High", "Low", "Close", "Volume"):
        frame[column] = pd.to_numeric(
            frame[column],
            errors="raise",
        )

    return frame


def _previous_weekday(day: pd.Timestamp) -> pd.Timestamp:
    result = day - pd.Timedelta(days=1)

    while result.weekday() >= 5:
        result -= pd.Timedelta(days=1)

    return result


def _next_weekday(day: pd.Timestamp) -> pd.Timestamp:
    result = day + pd.Timedelta(days=1)

    while result.weekday() >= 5:
        result += pd.Timedelta(days=1)

    return result


def _overnight_features(
    local_time: pd.Series,
    *,
    rth_open_minutes: int,
    rth_close_minutes: int,
    in_rth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate overnight progress without looping over every candle.

    Only unique local calendar dates are processed in Python. This keeps
    the calculation practical for multi-million-row datasets.
    """
    normalized_days = local_time.dt.normalize()
    codes, unique_days = pd.factorize(
        normalized_days,
        sort=False,
    )
    unique_days = pd.DatetimeIndex(unique_days)

    count = len(unique_days)

    same_open_ns = np.empty(count, dtype=np.int64)
    same_close_ns = np.empty(count, dtype=np.int64)
    previous_close_ns = np.empty(count, dtype=np.int64)
    next_open_ns = np.empty(count, dtype=np.int64)
    weekday_by_day = np.empty(count, dtype=np.int8)

    for index, day in enumerate(unique_days):
        weekday_by_day[index] = day.weekday()

        same_open_ns[index] = (
            day + pd.Timedelta(minutes=rth_open_minutes)
        ).value
        same_close_ns[index] = (
            day + pd.Timedelta(minutes=rth_close_minutes)
        ).value

        previous_day = _previous_weekday(day)
        next_day = _next_weekday(day)

        previous_close_ns[index] = (
            previous_day
            + pd.Timedelta(minutes=rth_close_minutes)
        ).value
        next_open_ns[index] = (
            next_day
            + pd.Timedelta(minutes=rth_open_minutes)
        ).value

    timestamp_ns = local_time.astype("int64").to_numpy()
    same_open = same_open_ns[codes]
    same_close = same_close_ns[codes]
    previous_close = previous_close_ns[codes]
    next_open = next_open_ns[codes]
    weekdays = weekday_by_day[codes]

    before_open = (
        (weekdays < 5)
        & (timestamp_ns < same_open)
    )
    after_close = (
        (weekdays < 5)
        & (timestamp_ns > same_close)
    )

    overnight_start = np.where(
        before_open,
        previous_close,
        np.where(
            after_close,
            same_close,
            previous_close,
        ),
    )

    overnight_end = np.where(
        before_open,
        same_open,
        next_open,
    )

    denominator = np.maximum(
        overnight_end - overnight_start,
        1,
    )
    numerator = np.maximum(
        timestamp_ns - overnight_start,
        0,
    )

    progress = np.clip(
        numerator / denominator,
        0.0,
        1.0,
    )

    overnight_active = 1.0 - in_rth
    progress = np.where(
        overnight_active > 0.0,
        progress,
        1.0,
    )

    return (
        overnight_active.astype(np.float32),
        progress.astype(np.float32),
    )



def _add_window_features(
    frame: pd.DataFrame,
    *,
    feature_names: tuple[str, ...],
) -> None:
    if not feature_names:
        return

    epsilon = 1e-12

    open_price = (
        frame["Open"]
        .astype(np.float64)
        .clip(lower=epsilon)
    )
    high = frame["High"].astype(np.float64)
    low = frame["Low"].astype(np.float64)
    close = frame["Close"].astype(np.float64)
    volume = (
        frame["Volume"]
        .astype(np.float64)
        .clip(lower=0.0)
    )

    candle_high = pd.Series(
        np.maximum(open_price, close),
        index=frame.index,
    )
    candle_low = pd.Series(
        np.minimum(open_price, close),
        index=frame.index,
    )

    calculated_features = {
        "direction": np.sign(
            close - open_price
        ),
        "range": (
            high - low
        ) / open_price,
        "body": (
            close - open_price
        ).abs() / open_price,
        "upper_wick": (
            (
                high - candle_high
            )
            / open_price
        ).clip(lower=0.0),
        "lower_wick": (
            (
                candle_low - low
            )
            / open_price
        ).clip(lower=0.0),
        "vol_chg": np.log(
            (volume + epsilon)
            / (volume.shift(1) + epsilon)
        ),
    }

    for feature_name in feature_names:
        frame[feature_name] = _clean_feature(
            calculated_features[feature_name],
            index=frame.index,
        )


def _add_rolling_features(
    frame: pd.DataFrame,
    *,
    feature_names: tuple[str, ...],
    horizons: tuple[int, ...],
) -> None:
    if not feature_names:
        return

    epsilon = 1e-12

    high = frame["High"].astype(np.float64)
    low = frame["Low"].astype(np.float64)
    close = frame["Close"].astype(np.float64)
    safe_close = close.clip(lower=epsilon)

    one_bar_log_return: pd.Series | None = None

    if "volatility" in feature_names:
        one_bar_log_return = _clean_feature(
            np.log(
                safe_close
                / safe_close.shift(1).clip(
                    lower=epsilon
                )
            )
        )

    features_using_high_or_low = {
        "range_position",
        "drawdown_from_high",
        "rebound_from_low",
    }

    needs_high_or_low = bool(
        features_using_high_or_low
        & set(feature_names)
    )

    for horizon in horizons:
        rolling_high: pd.Series | None = None
        rolling_low: pd.Series | None = None

        if needs_high_or_low:
            rolling_high = high.rolling(
                horizon,
                min_periods=2,
            ).max()

            rolling_low = low.rolling(
                horizon,
                min_periods=2,
            ).min()

        if "log_return" in feature_names:
            frame[f"log_ret_{horizon}"] = (
                _clean_feature(
                    np.log(
                        safe_close
                        / safe_close.shift(
                            horizon
                        ).clip(lower=epsilon)
                    )
                )
            )

        if "volatility" in feature_names:
            assert one_bar_log_return is not None

            frame[f"vol_{horizon}"] = (
                _clean_feature(
                    one_bar_log_return.rolling(
                        horizon,
                        min_periods=2,
                    ).std()
                )
            )

        if "range_position" in feature_names:
            assert rolling_high is not None
            assert rolling_low is not None

            denominator = (
                rolling_high - rolling_low
            ).replace(0.0, np.nan)

            frame[f"range_pos_{horizon}"] = (
                pd.to_numeric(
                    (
                        (
                            close - rolling_low
                        )
                        / denominator
                    ).clip(0.0, 1.0),
                    errors="coerce",
                )
                .replace(
                    [np.inf, -np.inf],
                    np.nan,
                )
                .fillna(0.5)
                .astype(np.float32)
            )

        if "close_vs_sma" in feature_names:
            moving_average = (
                close.rolling(
                    horizon,
                    min_periods=2,
                )
                .mean()
                .clip(lower=epsilon)
            )

            frame[
                f"close_vs_sma_{horizon}"
            ] = _clean_feature(
                np.log(
                    safe_close
                    / moving_average
                )
            )

        if "drawdown_from_high" in feature_names:
            assert rolling_high is not None

            frame[
                f"dd_from_high_{horizon}"
            ] = _clean_feature(
                (
                    safe_close
                    / rolling_high.clip(
                        lower=epsilon
                    )
                )
                - 1.0
            )

        if "rebound_from_low" in feature_names:
            assert rolling_low is not None

            frame[
                f"rebound_from_low_{horizon}"
            ] = _clean_feature(
                (
                    safe_close
                    / rolling_low.clip(
                        lower=epsilon
                    )
                )
                - 1.0
            )


def _add_time_features(
    frame: pd.DataFrame,
    *,
    market_timezone: str,
    rth_open: str,
    rth_close: str,
    swap_time: str,
    swap_timezone: str,
) -> None:
    market_local_time = frame["DT"].dt.tz_convert(
        market_timezone
    )

    market_minutes = (
        market_local_time.dt.hour * 60
        + market_local_time.dt.minute
    ).astype(np.float64)

    day_of_week = (
        market_local_time.dt.dayofweek
        .astype(np.float64)
    )

    rth_open_minutes = _parse_hh_mm(
        rth_open
    )
    rth_close_minutes = _parse_hh_mm(
        rth_close
    )

    rth_length = float(
        rth_close_minutes
        - rth_open_minutes
    )

    in_rth = (
        (day_of_week < 5)
        & (
            market_minutes
            >= rth_open_minutes
        )
        & (
            market_minutes
            <= rth_close_minutes
        )
    ).astype(np.float32).to_numpy()

    rth_progress = np.clip(
        (
            market_minutes
            - rth_open_minutes
        )
        / rth_length,
        0.0,
        1.0,
    ).to_numpy()

    frame["tod_sin"] = np.sin(
        2.0
        * np.pi
        * market_minutes
        / 1440.0
    ).astype(np.float32)

    frame["tod_cos"] = np.cos(
        2.0
        * np.pi
        * market_minutes
        / 1440.0
    ).astype(np.float32)

    frame["dow_sin"] = np.sin(
        2.0
        * np.pi
        * day_of_week
        / 7.0
    ).astype(np.float32)

    frame["dow_cos"] = np.cos(
        2.0
        * np.pi
        * day_of_week
        / 7.0
    ).astype(np.float32)

    frame["rth_active"] = in_rth

    frame["rth_sin_half"] = np.where(
        in_rth > 0,
        np.sin(
            np.pi * rth_progress
        ),
        0.0,
    ).astype(np.float32)

    frame["rth_cos_half"] = np.where(
        in_rth > 0,
        np.cos(
            np.pi * rth_progress
        ),
        -1.0,
    ).astype(np.float32)

    (
        overnight_active,
        overnight_progress,
    ) = _overnight_features(
        market_local_time,
        rth_open_minutes=rth_open_minutes,
        rth_close_minutes=rth_close_minutes,
        in_rth=in_rth,
    )

    frame[
        "overnight_active"
    ] = overnight_active

    frame[
        "overnight_sin_half"
    ] = np.where(
        overnight_active > 0,
        np.sin(
            np.pi * overnight_progress
        ),
        0.0,
    ).astype(np.float32)

    frame[
        "overnight_cos_half"
    ] = np.where(
        overnight_active > 0,
        np.cos(
            np.pi * overnight_progress
        ),
        -1.0,
    ).astype(np.float32)

    swap_local_time = frame[
        "DT"
    ].dt.tz_convert(
        swap_timezone
    )

    swap_local_minutes = (
        swap_local_time.dt.hour * 60
        + swap_local_time.dt.minute
    ).astype(np.float64)

    swap_minute = float(
        _parse_hh_mm(swap_time)
    )

    pre_swap_window_minutes = 60.0

    pre_swap_start = (
        swap_minute
        - pre_swap_window_minutes
    ) % 1440.0

    if pre_swap_start <= swap_minute:
        in_pre_swap = (
            swap_local_minutes
            >= pre_swap_start
        ) & (
            swap_local_minutes
            <= swap_minute
        )

        pre_swap_elapsed = (
            swap_local_minutes
            - pre_swap_start
        )

    else:
        in_pre_swap = (
            swap_local_minutes
            >= pre_swap_start
        ) | (
            swap_local_minutes
            <= swap_minute
        )

        pre_swap_elapsed = np.where(
            swap_local_minutes
            >= pre_swap_start,
            (
                swap_local_minutes
                - pre_swap_start
            ),
            (
                swap_local_minutes
                + 1440.0
                - pre_swap_start
            ),
        )

    in_pre_swap_array = (
        in_pre_swap
        .astype(np.float32)
        .to_numpy()
    )

    pre_swap_progress = np.clip(
        pre_swap_elapsed
        / pre_swap_window_minutes,
        0.0,
        1.0,
    )

    frame[
        "swap_pre_active"
    ] = in_pre_swap_array

    frame[
        "swap_pre_sin_half"
    ] = np.where(
        in_pre_swap_array > 0,
        np.sin(
            np.pi * pre_swap_progress
        ),
        0.0,
    ).astype(np.float32)

    frame[
        "swap_pre_cos_half"
    ] = np.where(
        in_pre_swap_array > 0,
        np.cos(
            np.pi * pre_swap_progress
        ),
        -1.0,
    ).astype(np.float32)


def build_observation(
    prepared: PreparedContext,
    *,
    current_index: int,
    window: int,
    position: int,
    unrealized_pnl: float,
) -> np.ndarray:
    """
    Build one complete observation.

    The market window includes the currently closed candle:
    [current_index - window + 1, ..., current_index].
    """
    if position not in {-1, 0, 1}:
        raise ContextError(
            "Position must be -1, 0, or 1."
        )

    first_index = current_index - window + 1

    if first_index < 0:
        raise ContextError(
            "Not enough candles to build the requested window."
        )

    if current_index >= len(prepared.frame):
        raise ContextError(
            "Current index is outside the prepared context."
        )

    definition = prepared.definition

    market_window = (
        prepared.window_values[
            first_index:current_index + 1
        ]
        .reshape(-1)
    )

    current_context = (
        prepared.context_values[
            current_index
        ]
    )

    state = np.asarray(
        [position, unrealized_pnl],
        dtype=np.float32,
    )

    observation = np.concatenate(
        [
            market_window,
            current_context,
            state,
        ]
    ).astype(np.float32)

    expected_size = definition.observation_size(window)

    if observation.shape != (expected_size,):
        raise ContextError(
            "Unexpected observation size: "
            f"expected {expected_size}, "
            f"received {observation.shape}"
        )

    return observation
