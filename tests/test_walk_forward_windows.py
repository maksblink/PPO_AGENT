from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from train_and_eval.run_config import RunConfig
from train_and_eval.walk_forward.config import build_plan, load_protocol, stage_config
from train_and_eval.walk_forward.windows import split_time_ranges


@pytest.fixture
def protocol():
    return load_protocol("configs/walk_forward/nq5m_v1_seed1.yml")


def frame_between(start="2018-01-01", end="2018-06-01"):
    times = pd.date_range(start, end, freq="5min", tz="UTC", inclusive="left")
    times = times[times.weekday < 5]
    frame = pd.DataFrame({"DT": times, "Open": 100., "High": 102., "Low": 99., "Close": 101., "Volume": 10.})
    frame.attrs = {"interval": "5m", "sha256": "a"*64,
                   "source_path": "data/example.parquet"}
    return frame


def temporal(protocol, start, end, alignment):
    raw = protocol.run.model_dump(mode="json")
    raw["data"] = {"path": raw["data"]["path"], "train_range": {"start": start, "end": end}, "alignment": alignment}
    raw["evaluation"]["training_mode"] = "none"
    return RunConfig.model_validate(raw)


def test_initial_context_reserved_before_maximum_trim(protocol):
    frame = frame_between()
    end_index = 6145 + 2*256 + 255
    cfg = temporal(protocol, frame.DT.iloc[0].isoformat(), frame.DT.iloc[end_index].isoformat(), "trim_start")
    split = split_time_ranges(frame, cfg)
    meta = split.window_metadata
    assert meta["warmup_rows"] == 6145
    assert meta["trimmed_steps"] == 255
    assert meta["train"]["start_index"] == 6400
    assert split.steps_per_data_epoch == 512
    assert split.train_data.DT.iloc[split.training_start_index+1] == frame.DT.iloc[6400]
    assert len(split.train_data) == 6145+512


@pytest.mark.parametrize("missing", [0, 1, 7, 159, 160, 255])
def test_week_prepends_available_rows_and_never_future(protocol, missing):
    frame = frame_between()
    start, end = "2018-03-05T00:00:00Z", "2018-03-12T00:00:00Z"
    first = int(frame.DT.searchsorted(pd.Timestamp(start)))
    if missing:
        frame = frame.drop(frame.index[first+10:first+10+missing]).reset_index(drop=True)
    cfg = temporal(protocol, start, end, "prepend")
    split = split_time_ranges(frame, cfg)
    meta = split.window_metadata
    nominal = 1440-missing
    extra = (-nominal) % 256
    assert meta["nominal_train"]["rows"] == nominal
    assert meta["prepended_steps"] == extra
    assert split.steps_per_data_epoch == nominal+extra
    assert split.steps_per_data_epoch % 256 == 0
    assert split.train_data.DT.max() < pd.Timestamp(end)
    actual = meta["train"]["start_index"]
    assert len(split.train_data.iloc[:split.training_start_index+1]) == 6145
    pd.testing.assert_frame_equal(split.train_data, frame.iloc[actual-6145:meta["train"]["end_index"]].reset_index(drop=True))
    poisoned = frame.copy()
    poisoned.loc[poisoned.DT >= pd.Timestamp(end), "Close"] = -999999
    pd.testing.assert_frame_equal(split.train_data, split_time_ranges(poisoned, cfg).train_data)


def test_prepend_cannot_consume_required_context(protocol):
    frame = frame_between()
    cfg = temporal(protocol, frame.DT.iloc[6150].isoformat(), frame.DT.iloc[6300].isoformat(), "prepend")
    with pytest.raises(ValueError, match="context"):
        split_time_ranges(frame, cfg)


def test_validation_keeps_every_available_row(protocol):
    frame = frame_between()
    raw = temporal(protocol, "2018-01-01T00:00:00Z", "2018-03-05T00:00:00Z", "trim_start").model_dump(mode="json")
    raw["data"]["validation_range"] = {"start": "2018-03-05T00:00:00Z", "end": "2018-03-12T00:00:00Z"}
    raw["evaluation"]["training_mode"] = "final_only"
    split = split_time_ranges(frame, RunConfig.model_validate(raw))
    assert split.validation_steps == 1440
    assert len(split.validation_data_with_lookback) == 6145+1440
    assert split.validation_start_index == 6144


def test_frozen_calendar_matches_agreed_dates(protocol):
    # Sparse valid timestamps make row/time fractions substantially different.
    frame = frame_between("2010-06-07", "2026-09-09")
    protocol = protocol.model_copy(update={"run": protocol.run.model_copy(update={"data": protocol.run.data.model_copy(update={"path": "data/example.parquet"})})})
    manifest = {"files": {"5m": {"sha256": "a"*64, "first_timestamp": frame.DT.iloc[0].isoformat(),
                                      "last_timestamp": frame.DT.iloc[-1].isoformat()}},
                "release_id": "test-release", "build": {"target_end_exclusive_utc": "2026-09-09T00:00:00Z"}}
    plan = build_plan(protocol, frame, manifest)
    assert plan["initial_train"]["end"] == "2018-07-23T00:00:00+00:00"
    assert len(plan["cycles"]) == 420
    assert plan["cycles"][0]["test"]["start"] == "2018-08-20T00:00:00+00:00"
    assert plan["cycles"][-1]["test"]["end"] == "2026-09-07T00:00:00+00:00"
    for previous, current in zip(plan["cycles"], plan["cycles"][1:]):
        assert previous["test"]["end"] == current["test"]["start"]
        assert current["update"]["start"] == previous["validation"]["start"]
        assert current["candidate_window"]["train"]["end_index"] <= current["candidate_window"]["validation"]["start_index"]
        assert current["refit_window"]["train"]["end_index"] <= current["test_rows"]["start_index"]


def test_protocol_rejects_overlapping_tests_and_bad_lrs(protocol):
    raw = protocol.model_dump(mode="json")
    raw["test_weeks"] = 2
    with pytest.raises(ValidationError, match="non-overlapping"):
        type(protocol).model_validate(raw)
    raw["test_weeks"] = 1
    raw["learning_rates"] = [float("nan")]
    with pytest.raises(ValidationError, match="finite"):
        type(protocol).model_validate(raw)


def test_refit_has_no_validation_and_prescribed_budget(protocol):
    cycle = {"number": 2, "update": {"start": "2018-03-05T00:00:00Z", "end": "2018-03-12T00:00:00Z"},
             "validation": {"start": "2018-03-12T00:00:00Z", "end": "2018-04-09T00:00:00Z"}}
    configs = [stage_config(protocol, cycle, role="candidate", candidate=i, source=("base", 17)) for i in range(3)]
    assert all(c.continuation.checkpoint == "17" for c in configs)
    assert all(c.data == configs[0].data for c in configs)
    refit = stage_config(protocol, cycle, role="refit", candidate=2, source=("selected", 24))
    assert refit.data.validation_range is None
    assert refit.data.train_range == configs[0].data.validation_range
    assert refit.evaluation.training_mode == "none"
    assert refit.ppo.learning_rate == .000075
    assert refit.training.duration_amount == 1
