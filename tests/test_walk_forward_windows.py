from __future__ import annotations

from pathlib import Path
import yaml
import pandas as pd
import pytest
from pydantic import ValidationError

from train_and_eval.run_config import RunConfig
from train_and_eval.walk_forward.config import build_plan, load_protocol, stage_config
from train_and_eval.walk_forward.windows import split_time_ranges


@pytest.fixture
def protocol():
    base = RunConfig.model_validate(yaml.safe_load(Path("configs/stage_one/nq5m_v1_seed1.yml").read_text()))
    return load_protocol("configs/stage_two/nq5m_v1_seed1.yml").model_copy(update={"run": base})


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


def test_protocol_rejects_overlapping_tests_and_bad_lrs(protocol):
    raw = protocol.model_dump(mode="json")
    raw["test_weeks"] = 2
    with pytest.raises(ValidationError, match="non-overlapping"):
        type(protocol).model_validate(raw)
    raw["test_weeks"] = 1
    raw["grid"]["ppo.learning_rate"] = [float("nan")]
    with pytest.raises(ValidationError, match="finite"):
        type(protocol).model_validate(raw)


def test_refit_has_no_validation_and_prescribed_budget(protocol):
    cycle = {"number": 2, "update": {"start": "2018-03-05T00:00:00Z", "end": "2018-03-12T00:00:00Z"},
             "validation": {"start": "2018-03-12T00:00:00Z", "end": "2018-04-09T00:00:00Z"}}
    config = stage_config(protocol, cycle, role="candidate", source=("base", 17))
    assert config.continuation.checkpoint == "17"
    refit = stage_config(protocol, cycle, role="refit", source=("selected", 24))
    assert refit.data.validation_range is None
    assert refit.data.train_range == config.data.validation_range
    assert refit.evaluation.training_mode == "none"
    assert refit.ppo.learning_rate == config.ppo.learning_rate == protocol.grid["ppo.learning_rate"][0]
    assert refit.training.duration_amount == 1
