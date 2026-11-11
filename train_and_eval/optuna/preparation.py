"""Plan optional seed-1 configs from rounded proposals and audited ancestry.

No training, model loading, database writes or implicit configuration defaults.
"""

import hashlib
import json
import re
from copy import deepcopy
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from pathlib import Path

from .history import AXES, assemble, canonical, select

FLOAT_AXES = AXES[:-1]


def rounded_params(params):
    out = deepcopy(params)
    for key in FLOAT_AXES:
        value = Decimal(str(params[key]))
        if not value.is_finite() or value < 0:
            raise ValueError(f"Invalid {key}: {value}")
        with localcontext() as ctx:
            ctx.prec = max(40, len(value.as_tuple().digits) + 10)
            quantum = Decimal(1).scaleb(value.adjusted() - 5) if value else Decimal(1)
            out[key] = float(value.quantize(quantum, rounding=ROUND_HALF_EVEN))
    if out["learning_rate"] <= 0 or not all(
        0 < out[k] <= 1 for k in ("gamma", "gae_lambda")
    ):
        raise ValueError("Rounded parameters violate technical PPO bounds")
    for key in ("depth", "n_epochs", "data_epochs"):
        if type(out[key]) is not int or out[key] < 1:
            raise ValueError(f"Invalid integer {key}")
    if out["n_epochs"] > 99 or out["data_epochs"] > 100:
        raise ValueError("Naming supports ne01..ne99 and stages v00..v99")
    expected = (
        set(AXES)
        | {"depth", "data_epochs"}
        | {f"width_{i}" for i in range(1, out["depth"] + 1)}
    )
    if set(out) != expected:
        raise ValueError("Unexpected/missing proposal axes or inactive widths")
    if any(
        type(out[f"width_{i}"]) is not int or out[f"width_{i}"] < 1
        for i in range(1, out["depth"] + 1)
    ):
        raise ValueError("Widths must be positive integers")
    return out


def scientific(value):
    number = Decimal(str(value))
    exponent = number.adjusted() if number else 0
    if not number.is_finite() or number < 0 or abs(exponent) > 9:
        raise ValueError(f"Number cannot be represented by scientific6 naming: {value}")
    mantissa = format(number.scaleb(-exponent), ".5f")
    if Decimal(mantissa).scaleb(exponent) != number:
        raise ValueError(f"Number requires more than six significant digits: {value}")
    return (
        mantissa.replace(".", "p")
        + ("em" if exponent < 0 else "ep")
        + str(abs(exponent))
    )


def run_name(params, stage):
    widths = [params[f"width_{i}"] for i in range(1, params["depth"] + 1)]
    arch = (
        f"w{widths[0]}x{len(widths)}"
        if len(set(widths)) == 1
        else "".join(f"w{w}" for w in widths)
    )
    if not 0 <= stage <= 99:
        raise ValueError("Stage exceeds v00..v99")
    name = (
        f"op_lr{scientific(params['learning_rate'])}_{'f' if stage == 0 else 'r'}_v{stage:02d}"
        f"_gae{scientific(params['gae_lambda'])}_ent{scientific(params['ent_coef'])}"
        f"_g{scientific(params['gamma'])}_{arch}_ne{params['n_epochs']:02d}_s01"
    )
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", name):
        raise ValueError(
            f"Run name has {len(name)} characters; database/config maximum is 128: {name}"
        )
    return name


def saved_config(run):
    value = run["normalized_config_json"]
    return json.loads(value) if isinstance(value, str) else deepcopy(value)


def identity(params):
    return canonical({k: v for k, v in params.items() if k != "data_epochs"})


def config_for(base, params, stage, parent=None):
    cfg = deepcopy(base)
    cfg["run"] = {"name": run_name(params, stage), "seed": 1}
    cfg["continuation"] = (
        {"mode": "fresh"}
        if parent is None
        else {
            "mode": "resume",
            "source_run": parent["name"],
            "checkpoint": str(parent["checkpoint_id"])
            if parent.get("checkpoint_id") is not None
            else "final",
        }
    )
    for axis in AXES:
        cfg["ppo"][axis] = params[axis]
    cfg["ppo"]["hidden_sizes"] = [
        params[f"width_{i}"] for i in range(1, params["depth"] + 1)
    ]
    return cfg


def build_plan(result, runs, checkpoints, evaluations, timestamp, batch):
    """Pure plan; current snapshot, all stages (independent of history selection mode)."""
    from train_and_eval.run_config import RunConfig, validate_resume_compatibility

    if not re.fullmatch(r"\d{8}T\d{12}Z(?:_[0-9a-f]{8})?", batch):
        raise ValueError("Invalid report/batch directory name")
    proposals = result.get("sampling", {}).get("proposals", [])
    if not 1 <= len(proposals) <= 999:
        raise ValueError("Preparation requires 1..999 proposals")
    rows, _audit = assemble(runs, checkpoints, evaluations)
    cohort = result["history"]["selected_cohort"]
    rows = [r for r in rows if r["cohort"] == cohort]
    if not rows:
        raise ValueError("Selected scientific cohort is no longer available")
    select(rows, "all", "return")  # Also reject conflicting scientific replicas.
    db = {r["id"]: r for r in runs}
    by_id = {r["id"]: r for r in rows}
    by_name = {r["name"]: r for r in runs}
    finals = {
        c["run_id"]: c
        for c in checkpoints
        if c["save_reason"] == "final" and c["run_id"] in by_id
    }
    source = min(rows, key=lambda r: r["id"])
    base = saved_config(db[source["id"]])
    if base["training"] != dict(
        base["training"], duration_unit="data_epochs", duration_amount=1
    ):
        raise ValueError(
            "Export requires one data epoch per stage in the selected cohort"
        )
    # Validate current schema without silently replacing historical settings.
    RunConfig.model_validate(base)
    for row in rows:
        if row["params"]["data_epochs"] != len(row["path"]):
            raise ValueError("Historical path does not contain one epoch per stage")
    stage_root = f"configs/stage_one/optuna_propositions/{batch}"
    queue_root = f"configs/search_queues/optuna_propositions/{batch}"
    plan = {
        "status": "planned",
        "batch": batch,
        "snapshot_at": timestamp,
        "cohort": cohort,
        "template_run_id": source["id"],
        "config_root": stage_root,
        "queue_root": queue_root,
        "queue_name": f"optuna_{batch}",
        "proposals": [],
        "nodes": [],
        "existing_parents": {},
    }
    planned = {}
    for index, proposal in enumerate(proposals, 1):
        params = rounded_params(proposal["params"])
        key = identity(params)
        # Never turn an unfinished/excluded related chain into a new duplicate.
        valid_ids = set(by_id)
        for candidate in runs:
            if candidate["id"] in valid_ids or candidate.get("seed") != 1:
                continue
            raw = saved_config(candidate)
            probe = deepcopy(base)
            probe["ppo"].update({k: params[k] for k in AXES})
            probe["ppo"]["hidden_sizes"] = [
                params[f"width_{i}"] for i in range(1, params["depth"] + 1)
            ]
            fields = ("data", "environment", "ppo", "evaluation", "training")
            if all(raw.get(k) == probe.get(k) for k in fields):
                raise ValueError(
                    f"Related run #{candidate['id']} is not reusable ({candidate['status']}); review it before export"
                )
        matches = [
            r
            for r in rows
            if identity(r["params"]) == key
            and r["params"]["data_epochs"] <= params["data_epochs"]
        ]
        leaf = (
            min(matches, key=lambda r: (-r["params"]["data_epochs"], r["id"]))
            if matches
            else None
        )
        prefix = [by_id[i] for i in leaf["path"]] if leaf else []
        item = {
            "proposal": f"prop_{index:03d}",
            "original": proposal["params"],
            "effective": params,
            "rounding_changes": {
                k: {"original": proposal["params"][k], "effective": v}
                for k, v in params.items()
                if v != proposal["params"][k]
            },
            "outside_requested_bounds": [
                k
                for k, v in params.items()
                if not result["config"]["search_space"][k]["low"]
                <= v
                <= result["config"]["search_space"][k]["high"]
            ],
            "existing_runs": [
                {"id": r["id"], "name": r["name"], "stage": n}
                for n, r in enumerate(prefix)
            ],
            "new_runs": [],
            "shared_runs": [],
        }
        # Prefix always comes from a single verified ancestry path, never pieced together.
        parent = None
        template = base
        if leaf:
            cp = finals[leaf["id"]]
            parent = {"name": leaf["name"], "checkpoint_id": cp["id"]}
            template = saved_config(db[leaf["id"]])
        for stage in range(len(prefix), params["data_epochs"]):
            stage_key = (key, stage)
            if stage_key in planned:
                node = planned[stage_key]
                item["shared_runs"].append(
                    {"name": node["name"], "path": node["path"], "owner": node["owner"]}
                )
                parent = {"name": node["name"]}
                template = node["config"]
                continue
            cfg = config_for(template, params, stage, parent)
            name = cfg["run"]["name"]
            if name in by_name:
                raise ValueError(
                    f"Name collision with a non-reusable DB run #{by_name[name]['id']}: {name}"
                )
            model = RunConfig.model_validate(cfg)
            if parent:
                validate_resume_compatibility(model, RunConfig.model_validate(template))
                if parent.get("checkpoint_id") is not None:
                    plan["existing_parents"][str(parent["checkpoint_id"])] = finals[
                        leaf["id"]
                    ]
            stage_dir = ("f" if stage == 0 else "r") + f"_v{stage:02d}"
            node = {
                "name": name,
                "stage": stage,
                "owner": item["proposal"],
                "params": params,
                "path": f"{stage_root}/{item['proposal']}/{stage_dir}/{name}.yml",
                "config": cfg,
            }
            plan["nodes"].append(node)
            planned[stage_key] = node
            item["new_runs"].append(
                {"name": name, "path": node["path"], "stage": stage}
            )
            parent = {"name": name}
            template = cfg
        item["requires_skip"] = not item["new_runs"]
        plan["proposals"].append(item)
    names = [n["name"] for n in plan["nodes"]]
    if len(names) != len(set(names)):
        raise ValueError("Distinct scientific configurations collided after naming")
    return plan


def verify_checkpoint(root, checkpoint):
    relative = Path(checkpoint["relative_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Unsafe checkpoint path")
    candidates = {root / relative, root / "artifacts" / relative}
    paths = [p for p in candidates if p.is_file()]
    if len(paths) != 1:
        raise ValueError(f"Missing/ambiguous checkpoint #{checkpoint['id']}")
    path = paths[0]
    if not path.resolve().is_relative_to(root.resolve()) or any(
        p.is_symlink() for p in (path, *path.parents)
    ):
        raise ValueError("Checkpoint path escapes project or uses symlinks")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    if (
        path.stat().st_size != checkpoint["size_bytes"]
        or digest.hexdigest() != checkpoint["sha256"]
    ):
        raise ValueError(f"Checkpoint integrity mismatch: {path}")
    return str(path.relative_to(root))
