from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import Field, model_validator
import yaml

from train_and_eval.run_config import RunConfig, StrictConfigModel
from train_and_eval.walk_forward.windows import range_record, split_time_ranges


def digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class CheckpointWalkForwardConfig(StrictConfigModel):
    """Stage two starts from an explicitly chosen stage-one checkpoint."""

    schema_version: Literal[2] = 2
    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,99}$")
    seed: int = Field(ge=0, strict=True)
    source_checkpoint_id: int = Field(ge=1, strict=True)
    validation_weeks: int = Field(default=4, ge=1, strict=True)
    test_weeks: int = Field(default=1, ge=1, strict=True)
    step_weeks: int = Field(default=1, ge=1, strict=True)
    bootstrap_epochs: int = Field(default=1, ge=1, strict=True)
    update_epochs: int = Field(default=1, ge=1, strict=True)
    refit_epochs: int = Field(default=1, ge=1, strict=True)
    learning_rate: float = Field(default=0.000075, gt=0, allow_inf_nan=False)
    selection_rule: Literal["final_checkpoint"] = "final_checkpoint"
    optimizer_policy: Literal["preserve_independent_copy"] = "preserve_independent_copy"
    reject_updates: Literal[False] = False
    partial_test: Literal["skip"] = "skip"
    # Resolved from the persisted stage-one run, never accepted from YAML.
    run: RunConfig | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def validate_protocol(self):
        if "run" in self.model_fields_set:
            raise ValueError("Stage two inherits its run settings from source_checkpoint_id; do not supply run")
        if self.test_weeks != self.step_weeks:
            raise ValueError("test_weeks must equal step_weeks: tests must be consecutive and non-overlapping")
        if self.step_weeks > self.validation_weeks:
            raise ValueError("step_weeks cannot exceed validation_weeks")
        return self


def load_protocol(path: str | Path) -> CheckpointWalkForwardConfig:
    return CheckpointWalkForwardConfig.model_validate(yaml.safe_load(Path(path).read_text()))


def stage_config(protocol: CheckpointWalkForwardConfig, cycle: dict, *, role: Literal["candidate", "refit"],
                 source: tuple[str, int] | None = None) -> RunConfig:
    if protocol.run is None:
        raise ValueError("Resolve the stage-one run before building stage configs")
    raw = protocol.run.model_dump(mode="json")
    is_refit = role == "refit"
    raw["run"] = {"name": f"{protocol.name}_s{protocol.seed}_c{cycle['number']:04d}_{role}_00", "seed": protocol.seed}
    # A source-free config is used only for planning row ranges, never execution.
    raw["continuation"] = ({"mode": "fresh"} if source is None else
                           {"mode": "resume", "source_run": source[0], "checkpoint": str(source[1])})
    raw["data"] = {"path": protocol.run.data.path,
                   "train_range": cycle["validation"] if is_refit else cycle["update"],
                   "validation_range": None if is_refit else cycle["validation"],
                   "alignment": "prepend"}
    first_epochs = protocol.bootstrap_epochs
    epochs = protocol.refit_epochs if is_refit else (first_epochs if cycle["number"] == 1 else protocol.update_epochs)
    raw["training"] = {"duration_unit": "data_epochs", "duration_amount": epochs}
    raw["ppo"]["learning_rate"] = protocol.learning_rate
    raw["evaluation"]["training_mode"] = "none" if is_refit else "final_only"
    raw["artifacts"]["validation_trajectory"]["mode"] = "all"
    # Candidate selection and aggregate reporting use immutable final checkpoint IDs.
    return RunConfig.model_validate(raw)


def build_plan(protocol: CheckpointWalkForwardConfig, frame: pd.DataFrame, manifest: dict, source: dict | None = None) -> dict:
    if frame.empty or not frame.DT.is_monotonic_increasing or frame.DT.duplicated().any():
        raise ValueError("Planning requires nonempty, sorted, unique candles")
    identity = manifest["files"]["5m"]
    if frame.attrs.get("interval") != "5m" or frame.attrs.get("sha256") != identity["sha256"]:
        raise ValueError("Walk-forward requires the verified 5m file from this manifest")
    if frame.attrs.get("source_path") != protocol.run.data.path:
        raise ValueError("Protocol path differs from the loaded dataset")
    start = pd.Timestamp(identity["first_timestamp"])
    end = pd.Timestamp(manifest["build"]["target_end_exclusive_utc"])
    if end <= frame.DT.iloc[-1] or frame.DT.iloc[-1] != pd.Timestamp(identity["last_timestamp"]):
        raise ValueError("Invalid dataset coverage metadata")
    if source is None or source["checkpoint_id"] != protocol.source_checkpoint_id:
        raise ValueError("Stage two requires the resolved stage-one checkpoint")
    initial = source["train_range"]
    bootstrap = source["validation_range"]
    boundary = pd.Timestamp(bootstrap["end"])
    for value in (*initial.values(), *bootstrap.values()):
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None or timestamp.weekday() != 0 or timestamp != timestamp.normalize():
            raise ValueError("Stage-one ranges must use Monday 00:00 UTC boundaries")
    if (pd.Timestamp(initial["start"]) < start or initial["end"] != bootstrap["start"]
            or boundary > end or source["data_sha256"] != identity["sha256"]):
        raise ValueError("Stage-one ranges or data identity do not match this dataset")
    cycles = []
    vs = boundary
    while vs + pd.Timedelta(weeks=protocol.validation_weeks + protocol.test_weeks) <= end:
        ve = vs + pd.Timedelta(weeks=protocol.validation_weeks)
        te = ve + pd.Timedelta(weeks=protocol.test_weeks)
        number = len(cycles) + 1
        cycle = {"number": number, "base_history": {"start": initial["start"], "end": vs.isoformat()},
                 "update": bootstrap if number == 1 else {"start": (vs-pd.Timedelta(weeks=protocol.step_weeks)).isoformat(), "end": vs.isoformat()},
                 "validation": {"start": vs.isoformat(), "end": ve.isoformat()},
                 "test": {"start": ve.isoformat(), "end": te.isoformat()}}
        cycle["candidate_window"] = split_time_ranges(frame, stage_config(protocol, cycle, role="candidate"), validate_order=False).window_metadata
        cycle["refit_window"] = split_time_ranges(frame, stage_config(protocol, cycle, role="refit"), validate_order=False).window_metadata
        cycle["test_rows"] = range_record(frame, **cycle["test"])
        if cycle["test_rows"]["start_index"] < cycle["refit_window"]["train"]["end_index"]:
            raise ValueError("Test overlaps refit")
        cycles.append(cycle)
        vs += pd.Timedelta(weeks=protocol.step_weeks)
    if not cycles:
        raise ValueError("Dataset is too short for one full test")
    result = {"schema_version": protocol.schema_version, "release_id": manifest["release_id"],
            "manifest_canonical_sha256": digest(manifest), "data_sha256": identity["sha256"], "data_rows": len(frame),
            "coverage_start": start.isoformat(), "coverage_end_exclusive": end.isoformat(),
            "initial_train": initial,
            "cycles": cycles, "unused_tail_start": cycles[-1]["test"]["end"],
            "unused_tail_end": end.isoformat()}

    result["stage_one_source"] = source
    return result
