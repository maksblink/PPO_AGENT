"""Interactive, guarded publication of an already validated preparation plan."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from .yaml_order import ordered_run_config
from .git_guard import assert_git_clean
from .history import snapshot
from .preparation import build_plan, config_for, verify_checkpoint


def safe_path(root, relative):
    path = root / relative
    if not path.is_relative_to(root) or ".." in path.parts:
        raise ValueError(f"Unsafe destination: {path}")
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError(f"Symlink destination: {path}")
    return path


def preflight(root, plan):
    """Only read destination/config/checkpoint state; all checks precede writes."""
    for key in ("config_root", "queue_root"):
        path = safe_path(root, plan[key])
        if path.exists():
            raise ValueError(f"Refusing to overwrite an existing batch: {path}")
        check = subprocess.run(
            ["git", "check-ignore", "-q", "--", plan[key] + "/probe.yml"],
            cwd=root,
            check=False,
        )
        if check.returncode != 0:
            raise ValueError(f"Destination must be ignored by Git: {plan[key]}")
    names = {n["name"] for n in plan["nodes"]}
    # Existing local planned files must not become competing runs in another queue.
    for path in sorted((root / "configs").rglob("*")):
        if path.suffix not in (".yml", ".yaml") or not path.is_file():
            continue
        if path.is_symlink():
            raise ValueError(f"Symlink YAML configuration requires review: {path}")
        try:
            value = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid existing YAML config: {path}") from exc
        if isinstance(value, dict):
            if isinstance(value.get("run"), dict) and value["run"].get("name") in names:
                raise ValueError(
                    f"New run already has a local config: {path}; use/review its existing queue"
                )
            if (
                value.get("queue_schema_version") is not None
                and path.stem == plan["queue_name"]
            ):
                raise ValueError(f"Queue name already exists: {path}")
    plan["verified_checkpoints"] = [
        verify_checkpoint(root, cp) for cp in plan["existing_parents"].values()
    ]


def payloads(plan):
    files = {}
    for node in plan["nodes"]:
        files[node["path"]] = yaml.safe_dump(ordered_run_config(node["config"]), sort_keys=False)
    for item in plan["proposals"]:
        if not item["new_runs"]:
            continue  # SKIP creates no empty proposition directories.
        lines = [
            f"# {item['proposal']}",
            "",
            "Seed 1. One data epoch per run. Parameters stay fixed along the path.",
            "",
            "## Effective parameters",
            "",
            "```json",
            json.dumps(item["effective"], indent=2),
            "```",
            "",
            "## Existing stages",
            "",
        ]
        lines += [
            f"- v{r['stage']:02d}: #{r['id']} `{r['name']}`"
            for r in item["existing_runs"]
        ] or ["None."]
        lines += ["", "## Shared stages", ""]
        lines += [
            f"- `{r['name']}` owned by `{r['owner']}`: `{r['path']}`"
            for r in item["shared_runs"]
        ] or ["None."]
        lines += ["", "## New stages", ""]
        lines += [f"- v{r['stage']:02d}: `{r['path']}`" for r in item["new_runs"]]
        lines += [
            "",
            "Original values, rounding changes, provenance and queue membership: batch preparation.json.",
            "This command prepares files only. Start the managed queue separately.",
            "",
        ]
        files[f"{plan['config_root']}/{item['proposal']}/README.md"] = "\n".join(lines)
    queue_path = f"{plan['queue_root']}/{plan['queue_name']}.yml"
    files[queue_path] = yaml.safe_dump(
        {
            "queue_schema_version": 1,
            "name": plan["queue_name"],
            "configs": [n["path"] for n in plan["nodes"]],
        },
        sort_keys=False,
    )
    provenance = {k: v for k, v in plan.items() if k != "nodes"}
    provenance["status"] = "prepared_manifest"
    provenance["outcome_report"] = (
        "See the Preparation tab in source_report for final verification status."
    )
    provenance["configs"] = [
        {
            "name": n["name"],
            "path": n["path"],
            "stage": n["stage"],
            "owner": n["owner"],
            "sha256": hashlib.sha256(files[n["path"]].encode()).hexdigest(),
        }
        for n in plan["nodes"]
    ]
    files[f"{plan['config_root']}/preparation.json"] = (
        json.dumps(provenance, indent=2, default=str, allow_nan=False) + "\n"
    )
    return files


def validate_files(plan, read):
    """Re-read bytes, YAML values, dependency order and exact scientific naming."""
    from train_and_eval.run_config import RunConfig

    names = set()
    ready = {c["run_id"] for c in plan["existing_parents"].values()}
    external = {
        r["name"]
        for p in plan["proposals"]
        for r in p["existing_runs"]
        if r["id"] in ready
    }
    for node in plan["nodes"]:
        cfg = yaml.safe_load(read(node["path"]))
        if cfg != node["config"]:
            raise ValueError(f"Config round-trip mismatch: {node['path']}")
        RunConfig.model_validate(cfg)
        parent = None
        if node["stage"]:
            continuation = cfg["continuation"]
            source = continuation["source_run"]
            if source not in names | external:
                raise ValueError(f"Unresolved/forward dependency: {source}")
            parent = {"name": source}
            if source in external:
                checkpoint_id = continuation["checkpoint"]
                if checkpoint_id not in plan["existing_parents"]:
                    raise ValueError("External parent checkpoint is not pinned")
                parent["checkpoint_id"] = int(checkpoint_id)
        expected = config_for(cfg, node["params"], node["stage"], parent)
        if expected != cfg or node["name"] in names:
            raise ValueError("Scientific values/name mismatch or repeated run")
        names.add(node["name"])
    path = f"{plan['queue_root']}/{plan['queue_name']}.yml"
    queue = yaml.safe_load(read(path))
    expected = {
        "queue_schema_version": 1,
        "name": plan["queue_name"],
        "configs": [n["path"] for n in plan["nodes"]],
    }
    if queue != expected or len(names) != len(queue["configs"]) or not names:
        raise ValueError("Queue contents/count mismatch")


def publish(root, plan):
    """Stage/verify complete files, publish config tree first and queue last.

    Handled failures roll back only trees created by this invocation. An OS/power
    failure can leave an unqueued config tree; existing destinations are never replaced.
    """
    if not plan["nodes"]:
        return
    files = payloads(plan)
    config_target, queue_target = (
        safe_path(root, plan[k]) for k in ("config_root", "queue_root")
    )
    for target in (config_target, queue_target):
        if target.exists():
            raise ValueError(f"Destination appeared during preparation: {target}")
    created = []
    # Temporary files share the configs filesystem. No manifest is discoverable here.
    with tempfile.TemporaryDirectory(
        prefix=".optuna-preparing-", dir=root / "configs"
    ) as temporary:
        stage = Path(temporary)
        try:
            for path, contents in files.items():
                dest = stage / path
                dest.parent.mkdir(parents=True, exist_ok=True)
                with dest.open("x", encoding="utf-8") as stream:
                    stream.write(contents)
                    stream.flush()
                    os.fsync(stream.fileno())
            validate_files(plan, lambda p: (stage / p).read_text(encoding="utf-8"))
            for target, key in (
                (config_target, "config_root"),
                (queue_target, "queue_root"),
            ):
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() or target.is_symlink():
                    raise ValueError(f"Destination already exists: {target}")
                (stage / plan[key]).rename(target)
                created.append(target)
            validate_files(plan, lambda p: (root / p).read_text(encoding="utf-8"))
        except BaseException:
            for path in reversed(created):
                shutil.rmtree(path)
            raise


def prepare_interactive(root, result, directory, *, ask=input, interactive=None):
    from extra_tools.maintenance.maintenance_guard import (
        activity_guard,
        maintenance_guard,
    )

    interactive = sys.stdin.isatty() if interactive is None else interactive
    if not interactive:
        return {
            "status": "not_requested",
            "reason": "Noninteractive input; run in a terminal to prepare files.",
        }
    try:
        if (
            ask(
                "Prepare configs and a queue? Type PREPARE (anything else cancels): "
            ).strip()
            != "PREPARE"
        ):
            return {"status": "cancelled", "reason": "Preparation not requested"}
        with (
            activity_guard(root, exclusive=True),
            maintenance_guard(root, exclude_managed=True),
        ):
            for path in (
                "extra_tools/maintenance/run_rename/pending.json",
                "artifacts/.cleanup_pending.json",
            ):
                if (root / path).exists():
                    raise ValueError(
                        f"Pending maintenance must be recovered first: {path}"
                    )
            if assert_git_clean(root) != result["provenance"]["git_commit"]:
                raise ValueError("Git commit changed since analysis")
            runs, cps, evs, stamp = snapshot()
            plan = build_plan(result, runs, cps, evs, stamp, directory.name)
            plan["git_commit"] = result["provenance"]["git_commit"]
            plan["source_report"] = str(directory)
            for item in plan["proposals"]:
                print(
                    f"{item['proposal']}: existing={len(item['existing_runs'])}; shared={len(item['shared_runs'])}; new={len(item['new_runs'])}"
                )
                print(
                    "Effective parameters: "
                    + json.dumps(item["effective"], sort_keys=True)
                )
                if item["rounding_changes"]:
                    print(
                        "Rounding: "
                        + json.dumps(item["rounding_changes"], sort_keys=True)
                    )
                if item["outside_requested_bounds"]:
                    print(
                        "Allowed after rounding outside search bounds: "
                        + ", ".join(item["outside_requested_bounds"])
                    )
                if item["requires_skip"]:
                    print(
                        "Already covered: "
                        + json.dumps(item["existing_runs"] + item["shared_runs"])
                    )
                    if (
                        ask(
                            f"{item['proposal']} has no new stages. Type SKIP to continue: "
                        ).strip()
                        != "SKIP"
                    ):
                        plan["status"] = "cancelled"
                        return plan
                    item["decision"] = "SKIP"
            if not plan["nodes"]:
                plan["status"] = "nothing_to_create"
                return plan
            preflight(root, plan)
            print(
                f"Prepared plan: {len(plan['nodes'])} unique new runs; queue {plan['queue_name']}"
            )
            print(
                "Queue order: prop_001 stages, then prop_002 stages, etc. No training will start."
            )
            if (
                ask("Type CREATE to write and verify these configs and queue: ").strip()
                != "CREATE"
            ):
                plan["status"] = "cancelled"
                return plan
            # Recheck before publication; locks exclude cooperating queues/maintenance.
            if assert_git_clean(root) != result["provenance"]["git_commit"]:
                raise ValueError("Git changed before publication")
            publish(root, plan)
            plan["status"] = "created_and_verified"
            print(
                f"PASS: configs/queue written and verified. Queue: {plan['queue_name']}"
            )
            return plan
    except (EOFError, KeyboardInterrupt):
        return {
            "status": "cancelled",
            "reason": "Input interrupted; no incomplete queue published",
        }
