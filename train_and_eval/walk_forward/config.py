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


class WalkForwardConfig(StrictConfigModel):
    schema_version: Literal[1] = 1
    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,99}$")
    seed: int = Field(ge=0, strict=True)
    initial_time_fraction: float = Field(default=0.5, gt=0, lt=1)
    validation_weeks: int = Field(default=4, ge=1, strict=True)
    test_weeks: int = Field(default=1, ge=1, strict=True)
    step_weeks: int = Field(default=1, ge=1, strict=True)
    initial_epochs: int = Field(default=1, ge=1, strict=True)
    update_epochs: int = Field(default=1, ge=1, strict=True)
    refit_epochs: int = Field(default=1, ge=1, strict=True)
    learning_rates: list[float] = Field(default_factory=lambda: [0.000075, 0.00015, 0.0003], min_length=1)
    refit_learning_rate: float = Field(default=0.000075, gt=0)
    selection_rule: Literal["final_balanced_score_then_candidate_order"] = "final_balanced_score_then_candidate_order"
    optimizer_policy: Literal["preserve_independent_copy"] = "preserve_independent_copy"
    reject_updates: Literal[False] = False
    partial_test: Literal["skip"] = "skip"
    run: RunConfig

    @model_validator(mode="after")
    def validate_protocol(self):
        import math
        if self.test_weeks != self.step_weeks:
            raise ValueError("test_weeks must equal step_weeks: tests must be consecutive and non-overlapping")
        if self.step_weeks > self.validation_weeks:
            raise ValueError("step_weeks cannot exceed validation_weeks")
        if len(set(self.learning_rates)) != len(self.learning_rates) or any(not math.isfinite(x) or x <= 0 for x in self.learning_rates):
            raise ValueError("Candidate learning rates must be unique, finite and positive")
        if not math.isfinite(self.refit_learning_rate):
            raise ValueError("Refit learning rate must be finite")
        if not self.run.environment.force_close_on_done:
            raise ValueError("This protocol requires closing positions at every episode/test boundary")
        if self.run.evaluation.policy_mode != "deterministic_argmax":
            raise ValueError("The initial protocol uses deterministic_argmax")
        if self.run.ppo.n_steps % self.run.ppo.batch_size:
            raise ValueError("n_steps must be divisible by batch_size")
        if self.run.continuation.mode != "fresh":
            raise ValueError("Each study must start fresh; cross-study checkpoint reuse is forbidden")
        return self


def load_protocol(path: str | Path) -> WalkForwardConfig:
    return WalkForwardConfig.model_validate(yaml.safe_load(Path(path).read_text()))


def stage_config(protocol: WalkForwardConfig, cycle: dict, *, role: str, candidate: int = 0,
                 source: tuple[str, int] | None = None) -> RunConfig:
    raw = protocol.run.model_dump(mode="json")
    is_refit = role == "refit"
    raw["run"] = {"name": f"{protocol.name}_s{protocol.seed}_c{cycle['number']:04d}_{role}_{candidate:02d}", "seed": protocol.seed}
    raw["continuation"] = ({"mode": "fresh"} if source is None else
                           {"mode": "resume", "source_run": source[0], "checkpoint": str(source[1])})
    raw["data"] = {"path": protocol.run.data.path,
                   "train_range": cycle["validation"] if is_refit else cycle["update"],
                   "validation_range": None if is_refit else cycle["validation"],
                   "alignment": "prepend" if is_refit or cycle["number"] > 1 else "trim_start"}
    raw["training"] = {"duration_unit": "data_epochs", "duration_amount":
                       protocol.refit_epochs if is_refit else (protocol.initial_epochs if cycle["number"] == 1 else protocol.update_epochs)}
    raw["ppo"]["learning_rate"] = protocol.refit_learning_rate if is_refit else protocol.learning_rates[candidate]
    raw["evaluation"]["training_mode"] = "none" if is_refit else "final_only"
    raw["artifacts"]["validation_trajectory"]["mode"] = "all"
    # Candidate selection and aggregate reporting use immutable final checkpoint IDs.
    return RunConfig.model_validate(raw)


def build_plan(protocol: WalkForwardConfig, frame: pd.DataFrame, manifest: dict) -> dict:
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
    middle = start + (end - start) * protocol.initial_time_fraction
    boundary = middle.normalize() - pd.Timedelta(days=middle.weekday())
    if boundary <= start:
        raise ValueError("Initial calendar range is empty")
    initial = {"start": start.isoformat(), "end": boundary.isoformat()}
    cycles = []
    vs = boundary
    while vs + pd.Timedelta(weeks=protocol.validation_weeks + protocol.test_weeks) <= end:
        ve = vs + pd.Timedelta(weeks=protocol.validation_weeks)
        te = ve + pd.Timedelta(weeks=protocol.test_weeks)
        number = len(cycles) + 1
        cycle = {"number": number, "base_history": {"start": start.isoformat(), "end": vs.isoformat()},
                 "update": initial if number == 1 else {"start": (vs-pd.Timedelta(weeks=protocol.step_weeks)).isoformat(), "end": vs.isoformat()},
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
    return {"schema_version": 1, "release_id": manifest["release_id"],
            "manifest_canonical_sha256": digest(manifest), "data_sha256": identity["sha256"], "data_rows": len(frame),
            "coverage_start": start.isoformat(), "coverage_end_exclusive": end.isoformat(),
            "initial_midpoint": middle.isoformat(), "initial_train": initial,
            "cycles": cycles, "unused_tail_start": cycles[-1]["test"]["end"],
            "unused_tail_end": end.isoformat()}
