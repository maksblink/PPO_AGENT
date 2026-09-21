from copy import deepcopy

import pandas as pd
import pytest
from pydantic import ValidationError

from tests.test_run_config import _valid_config
from tests.test_trading_environment import _config, _market_frame
from train_and_eval.environment.trading_environment import TradingEnvironment
from train_and_eval.run_config import RunConfig, parse_persisted_run_config


@pytest.mark.parametrize('side', ['long_only', 'short_only', 'long_short'])
@pytest.mark.parametrize('field', ['swap_long_bps', 'swap_short_bps'])
def test_both_swap_rates_are_required_for_every_position_mode(side, field):
    raw = _valid_config()
    raw['environment']['position_side'] = side
    del raw['environment'][field]
    with pytest.raises(ValidationError, match=field):
        RunConfig.model_validate(raw)


@pytest.mark.parametrize('field', ['swap_long_bps', 'swap_short_bps'])
@pytest.mark.parametrize('value', [-1, float('nan'), float('inf'), -float('inf')])
def test_swap_rates_must_be_finite_and_nonnegative(field, value):
    raw = _valid_config()
    raw['environment'][field] = value
    with pytest.raises(ValidationError, match=field):
        RunConfig.model_validate(raw)


def test_historical_swap_is_read_without_mutating_persisted_config():
    raw = _valid_config()
    env = raw['environment']
    del env['swap_long_bps'], env['swap_short_bps']
    env['swap_bps'] = 7.0
    original = deepcopy(raw)
    # New configs must specify both directions; old field is forbidden.
    with pytest.raises(ValidationError, match='swap_bps'):
        RunConfig.model_validate(raw)
    restored = parse_persisted_run_config(raw)
    assert restored.environment.swap_long_bps == 7
    assert restored.environment.swap_short_bps == 7
    assert raw == original


def test_historical_reader_rejects_ambiguous_mixed_fields():
    raw = _valid_config()
    raw['environment']['swap_bps'] = 7.0
    with pytest.raises(ValidationError, match='swap_bps'):
        parse_persisted_run_config(raw)


@pytest.mark.parametrize('position,expected', [(1, .0002), (-1, .0007), (0, 0)])
def test_swap_uses_actual_position_direction(position, expected):
    frame = _market_frame(opens=[100.] * 6, closes=[100.] * 6,
                          start='2026-01-05 21:50:00+00:00')
    env = TradingEnvironment(frame, _config(position_side='long_short',
                             swap_long_bps=2, swap_short_bps=7))
    env.reset()
    env.position = position
    cost, count = env._apply_swap(before_index=1, after_index=2)
    assert cost == pytest.approx(expected)
    assert count == (1 if position else 0)
    assert env.realized_return == pytest.approx(-expected)


@pytest.mark.parametrize('position', [1, -1])
def test_zero_rate_disables_only_its_direction(position):
    frame = _market_frame(opens=[100.] * 6, closes=[100.] * 6,
                          start='2026-01-05 21:50:00+00:00')
    config = _config(position_side='long_short', swap_long_bps=0, swap_short_bps=7)
    env = TradingEnvironment(frame, config)
    env.reset()
    env.position = position
    cost, count = env._apply_swap(before_index=1, after_index=2)
    assert cost == pytest.approx(0 if position == 1 else .0007)
    assert count == (0 if position == 1 else 1)


def test_directional_swap_counts_weekends_across_market_gap():
    frame = _market_frame(opens=[100.] * 6, closes=[100.] * 6,
                          start='2026-01-02 21:50:00+00:00')
    frame.loc[2, 'DT'] = frame.loc[2, 'DT'] + pd.Timedelta(days=3)
    # Use a chronologically ordered frame spanning Friday through Monday.
    for i in range(3, 6):
        frame.loc[i, 'DT'] = frame.loc[i, 'DT'] + pd.Timedelta(days=3)
    env = TradingEnvironment(frame, _config(swap_long_bps=2, swap_short_bps=7))
    env.reset()
    env.position = -1
    cost, count = env._apply_swap(before_index=1, after_index=2)
    assert count == 4
    assert cost == pytest.approx(4 * .0007)
