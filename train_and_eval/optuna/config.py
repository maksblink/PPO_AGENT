"""Strict configuration: reject typos rather than silently changing the search."""

import math
import re
from copy import deepcopy

import yaml

BASE = {
    "depth",
    "learning_rate",
    "ent_coef",
    "gamma",
    "gae_lambda",
    "n_epochs",
    "data_epochs",
}
INTS = {"depth", "n_epochs", "data_epochs"}
DEFAULTS = {
    "schema_version": 1,
    "history": {"seed": 1, "mode": "all", "best_by": "return", "cohort": None},
    "objective": {"name": "return", "scope": "val"},
    "sampler": {
        "name": "tpe",
        "seed": 42,
        "multivariate": True,
        "group": True,
        "n_startup_trials": 10,
        "n_ei_candidates": 64,
    },
    "suggestions": {
        "count": 1,
        "exclude_evaluated": True,
        "max_sampling_attempts": 100,
    },
    "report": {
        "output_dir": "artifacts/optuna_propositions",
        "formats": ["html", "json"],
        "nearest_observations": 10,
        "include_full_paths": True,
    },
}


class UniqueLoader(yaml.SafeLoader):
    pass


def mapping(loader, node, deep=False):
    result = {}
    for k, v in node.value:
        key = loader.construct_object(k, deep=deep)
        if key in result:
            raise ValueError(f"Duplicate YAML key: {key}")
        result[key] = loader.construct_object(v, deep=deep)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)


def number(v):
    return not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v)


def integer(v, low=1, high=1000000):
    return type(v) is int and low <= v <= high


def validate(raw):
    if not isinstance(raw, dict) or any(not isinstance(k, str) for k in raw):
        raise ValueError("Configuration must be a mapping")
    unknown = set(raw) - set(DEFAULTS) - {"search_space"}
    if unknown:
        raise ValueError(f"Unknown config sections: {sorted(unknown)}")
    cfg = deepcopy(DEFAULTS)
    for k, value in raw.items():
        if k == "search_space":
            continue
        if isinstance(cfg[k], dict):
            if not isinstance(value, dict) or set(value) - set(cfg[k]):
                raise ValueError(f"Unknown or invalid options in {k}")
            cfg[k].update(value)
        else:
            cfg[k] = value
    if type(cfg["schema_version"]) is not int or cfg["schema_version"] != 1:
        raise ValueError("Expected schema_version: 1")
    h, s, p, r = (cfg[k] for k in ("history", "sampler", "suggestions", "report"))
    if type(h["seed"]) is not int or h["seed"] != 1:
        raise ValueError("Only training seed 1 is supported; seed is not optimized")
    if h["mode"] not in ("all", "last", "best") or h["best_by"] not in (
        "return",
        "balanced_score",
    ):
        raise ValueError("Invalid history mode/best_by")
    if h["cohort"] is not None and not isinstance(h["cohort"], str):
        raise ValueError("history.cohort must be null or a reported cohort ID")
    if (
        cfg["objective"]["name"] not in ("return", "balanced_score", "pareto")
        or cfg["objective"]["scope"] != "val"
    ):
        raise ValueError("Objectives: return, balanced_score, pareto; scope: val only")
    if s["name"] != "tpe" or not integer(s["seed"], 0, 2**32 - 1):
        raise ValueError("Invalid sampler name/seed")
    for section, names in (
        (s, ("multivariate", "group")),
        (p, ("exclude_evaluated",)),
        (r, ("include_full_paths",)),
    ):
        if any(type(section[n]) is not bool for n in names):
            raise ValueError("Boolean config fields must be true/false")
    if s["group"] and not s["multivariate"]:
        raise ValueError("group requires multivariate: true")
    for v in (
        s["n_startup_trials"],
        s["n_ei_candidates"],
        p["count"],
        p["max_sampling_attempts"],
        r["nearest_observations"],
    ):
        if not integer(v):
            raise ValueError("Counts and limits must be positive integers")
    if p["count"] > 999:
        raise ValueError("At most 999 proposals per analysis")
    if p["count"] > p["max_sampling_attempts"]:
        raise ValueError("count exceeds max_sampling_attempts")
    if (
        r["formats"] != ["html", "json"]
        or not isinstance(r["output_dir"], str)
        or not r["output_dir"]
    ):
        raise ValueError(
            "Report requires formats: [html, json] and a nonempty output_dir"
        )
    space = deepcopy(raw.get("search_space"))
    if (
        not isinstance(space, dict)
        or any(not isinstance(k, str) for k in space)
        or BASE - set(space)
    ):
        raise ValueError(
            "search_space must contain all seven base axes and active widths"
        )
    for name, spec in space.items():
        if name not in BASE and not re.fullmatch(r"width_[1-9][0-9]*", name):
            raise ValueError(f"Unknown search parameter: {name}")
        if not isinstance(spec, dict) or set(spec) - {
            "type",
            "low",
            "high",
            "step",
            "log",
        }:
            raise ValueError(f"Invalid specification: {name}")
        kind = "int" if name in INTS or name.startswith("width_") else "float"
        if spec.get("type") != kind:
            raise ValueError(f"{name} requires type: {kind}")
        low, high = spec.get("low"), spec.get("high")
        if not number(low) or not number(high) or low > high:
            raise ValueError(f"Invalid finite bounds: {name}")
        spec.setdefault("log", False)
        if type(spec["log"]) is not bool or spec["log"] and low <= 0:
            raise ValueError(f"Log scale requires positive bounds: {name}")
        if kind == "int":
            spec.setdefault("step", 1)
            if (
                not all(integer(v) for v in (low, high, spec["step"]))
                or (high - low) % spec["step"]
            ):
                raise ValueError(
                    f"Integer bounds/step must form an exact positive grid: {name}"
                )
            if spec["log"] and spec["step"] != 1:
                raise ValueError(f"Log integer distributions require step=1: {name}")
        elif "step" in spec:
            raise ValueError(
                f"Float step is not supported; use continuous bounds: {name}"
            )
        if name in ("gamma", "gae_lambda") and not 0 < low <= high <= 1:
            raise ValueError(f"{name} must be in (0,1]")
        if name == "learning_rate" and low <= 0 or name == "ent_coef" and low < 0:
            raise ValueError(f"Invalid bounds: {name}")
    if space["depth"]["high"] > 32:
        raise ValueError("depth above 32 is unsupported")
    widths = {f"width_{i}" for i in range(1, space["depth"]["high"] + 1)}
    if set(space) != BASE | widths:
        raise ValueError("Provide exactly width_1 through width_<maximum depth>")
    cfg["search_space"] = space
    return cfg


def load(path):
    return validate(yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueLoader))
