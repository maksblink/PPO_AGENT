from __future__ import annotations

import hashlib
import json
import itertools
import math
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import Field, TypeAdapter, ValidationError, model_validator
import yaml

from train_and_eval.run_config import EnvironmentSection, PPOSection, RunConfig, StrictConfigModel
from train_and_eval.walk_forward.windows import range_record, split_time_ranges


def digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


GRID_FIELDS = frozenset({
    "ppo.n_steps", "ppo.batch_size", "ppo.n_epochs", "ppo.learning_rate",
    "ppo.gamma", "ppo.gae_lambda", "ppo.clip_range", "ppo.clip_range_vf",
    "ppo.normalize_advantage", "ppo.ent_coef", "ppo.vf_coef", "ppo.max_grad_norm",
    "ppo.target_kl", "environment.reward_scale", "environment.exposure_penalty",
    "environment.turnover_penalty", "environment.drawdown_penalty",
    "environment.profit_reward_mult", "environment.loss_reward_mult",
})


def grid_candidates(protocol):
    """Sorted field names; YAML option order; rightmost field changes fastest."""
    names = sorted(protocol.grid)
    return [dict(zip(names, values)) for values in
            itertools.product(*(protocol.grid[name] for name in names))]


class CheckpointWalkForwardConfig(StrictConfigModel):
    """Stage two starts from an explicitly chosen stage-one checkpoint."""

    schema_version: Literal[3] = 3
    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,99}$")
    seed: int = Field(ge=0, strict=True)
    source_checkpoint_id: int = Field(ge=1, strict=True)
    validation_weeks: int = Field(default=4, ge=1, strict=True)
    test_weeks: int = Field(default=1, ge=1, strict=True)
    step_weeks: int = Field(default=1, ge=1, strict=True)
    bootstrap_epochs: int = Field(default=1, ge=1, strict=True)
    update_epochs: int = Field(default=1, ge=1, strict=True)
    refit_epochs: int = Field(default=1, ge=1, strict=True)
    grid: dict[str, list[float | int | bool | None]]
    selection_rule: Literal["first_strict_improvement"] = "first_strict_improvement"
    optimizer_policy: Literal["preserve_independent_copy"] = "preserve_independent_copy"
    partial_test: Literal["skip"] = "skip"
    # Resolved from the persisted stage-one run, never accepted from YAML.
    run: RunConfig | None = Field(default=None, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def validate_explicit_grid(cls, raw):
        if not isinstance(raw, dict):
            return raw
        grid = raw.get("grid", {})
        if not isinstance(grid, dict):
            raise ValueError("grid must be a mapping with all 19 required fields")
        errors = [f"grid.{name}: required; define a nonempty list of options"
                  for name in sorted(GRID_FIELDS - grid.keys())]
        for name, options in grid.items():
            if name not in GRID_FIELDS:
                errors.append(f"grid.{name}: not resume-safe or is inherited; remove this field")
                continue
            if not isinstance(options, list) or not options:
                errors.append(f"grid.{name}: must be a nonempty list of options")
                continue
            section, field = name.split(".")
            model = PPOSection if section == "ppo" else EnvironmentSection
            adapter = TypeAdapter(model.model_fields[field].rebuild_annotation())
            seen = set()
            for index, value in enumerate(options):
                label = f"grid.{name}[{index}]"
                if isinstance(value, float) and not math.isfinite(value):
                    errors.append(f"{label}: must be finite")
                    continue
                # Reject YAML strings and booleans used as numeric values.
                if isinstance(value, bool) and field != "normalize_advantage":
                    errors.append(f"{label}: expected a number, not a boolean")
                    continue
                try:
                    adapter.validate_python(value, strict=True)
                except ValidationError as error:
                    errors.extend(f"{label}: {item['msg']}" for item in error.errors())
                    continue
                if field in {"clip_range_vf", "target_kl"} and value is not None and value <= 0:
                    errors.append(f"{label}: must be greater than zero or null")
                    continue
                key = json.dumps(value)
                if key in seen:
                    errors.append(f"{label}: duplicate grid option")
                seen.add(key)
        if not errors:
            for steps in grid["ppo.n_steps"]:
                for batch in grid["ppo.batch_size"]:
                    if batch > steps or steps % batch:
                        errors.append(f"grid.ppo.batch_size={batch} must divide grid.ppo.n_steps={steps} exactly "
                                      "and cannot be greater than n_steps")
        if errors:
            raise ValueError("Invalid stage-two grid:\n" + "\n".join(errors))
        return raw

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
                 source: tuple[str, int] | None = None, candidate_order: int = 0) -> RunConfig:
    if protocol.run is None:
        raise ValueError("Resolve the stage-one run before building stage configs")
    raw = protocol.run.model_dump(mode="json")
    for name, value in grid_candidates(protocol)[candidate_order].items():
        section, field = name.split(".")
        raw[section][field] = value
    is_refit = role == "refit"
    raw["run"] = {"name": f"{protocol.name}_s{protocol.seed}_c{cycle['number']:04d}_{role}_{candidate_order:04d}", "seed": protocol.seed}
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
    candidates = grid_candidates(protocol)
    # Validate every complete configuration, including cross-field constraints,
    # before a study or training run can be created.
    for overrides in candidates:
        raw = protocol.run.model_dump(mode="json")
        for name, value in overrides.items():
            section, field = name.split(".")
            raw[section][field] = value
        RunConfig.model_validate(raw)
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
        cycle["grid_windows"] = []
        by_batch = {}
        for order in range(len(candidates)):
            candidate_config = stage_config(protocol, cycle, role="candidate", candidate_order=order)
            batch = candidate_config.ppo.batch_size
            if batch not in by_batch:
                by_batch[batch] = {
                    "candidate_window": split_time_ranges(frame, candidate_config, validate_order=False).window_metadata,
                    "refit_window": split_time_ranges(frame, stage_config(protocol, cycle, role="refit", candidate_order=order), validate_order=False).window_metadata,
                }
            cycle["grid_windows"].append(by_batch[batch])
        cycle.update(cycle["grid_windows"][0])
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
            "grid_candidates": candidates, "cycles": cycles, "unused_tail_start": cycles[-1]["test"]["end"],
            "unused_tail_end": end.isoformat()}

    result["stage_one_source"] = source
    return result
