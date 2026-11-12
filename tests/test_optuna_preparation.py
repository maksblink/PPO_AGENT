"""Preparation with synthetic DB history, real schemas/queues, no training/GPU."""

import hashlib
import json
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

pytest.importorskip("optuna")
from tests.test_optuna_propositions import configuration, dataset
from tests.test_run_config import _valid_config
from train_and_eval.optuna.export import (
    preflight,
    prepare_interactive,
    publish,
    validate_files,
)
from train_and_eval.optuna.history import assemble
from train_and_eval.optuna.preparation import (
    build_plan,
    rounded_params,
    run_name,
    scientific,
)

BATCH = "20261110T120000123456Z"


def setup_data(tmp_path):
    runs, cps, evs = dataset()
    for r in runs:
        old = r["normalized_config_json"]
        cfg = _valid_config()
        for key in ("run", "continuation", "training"):
            cfg[key] = old[key]
        cfg["ppo"].update(old["ppo"])
        cfg["evaluation"].update(device="cpu", early_stop_patience_evals=5)
        r["normalized_config_json"] = cfg
    for cp in cps:
        content = f"checkpoint-{cp['id']}".encode()
        relative = f"artifacts/runs/{cp['run_id']:08d}/checkpoints/final.zip"
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        cp.update(
            relative_path=relative,
            size_bytes=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
        )
    rows, _audit = assemble(runs, cps, evs)
    assert len(rows) == 3
    result = {
        "history": {"selected_cohort": rows[0]["cohort"]},
        "config": configuration(),
        "provenance": {"git_commit": "commit"},
        "sampling": {
            "proposals": [{"params": dict(rows[-1]["params"], data_epochs=5)}]
        },
    }
    return result, runs, cps, evs


def plan_for(tmp_path, params=None):
    result, runs, cps, evs = setup_data(tmp_path)
    if params:
        result["sampling"]["proposals"] = [{"params": p} for p in params]
    return build_plan(result, runs, cps, evs, "snapshot", BATCH), result, runs, cps, evs


def test_continuation_reuses_actual_parent_and_only_missing_stages(tmp_path):
    plan, _, _, cps, _ = plan_for(tmp_path)
    assert [n["stage"] for n in plan["nodes"]] == [3, 4]
    assert plan["nodes"][0]["config"]["continuation"] == {
        "mode": "resume",
        "source_run": "run3",
        "checkpoint": "3",
    }
    assert (
        plan["nodes"][1]["config"]["continuation"]["source_run"]
        == plan["nodes"][0]["name"]
    )
    assert plan["existing_parents"] == {"3": cps[-1]}


def test_rounding_names_unequal_and_carry():
    p = {
        "depth": 4,
        "width_1": 903,
        "width_2": 743,
        "width_3": 126,
        "width_4": 32,
        "learning_rate": 0.000022500009,
        "ent_coef": 0.00020000001,
        "gamma": 0.90000001,
        "gae_lambda": 0.85000001,
        "n_epochs": 3,
        "data_epochs": 1,
    }
    effective = rounded_params(p)
    assert (
        run_name(effective, 0)
        == "op_lr2p25000em5_f_v00_gae8p50000em1_ent2p00000em4_g9p00000em1_w903w743w126w32_ne03_s01"
    )
    p["learning_rate"] = 0.0000999999999
    assert scientific(rounded_params(p)["learning_rate"]) == "1p00000em4"
    assert scientific(0) == "0p00000ep0"


def test_rounded_existing_proposal_requires_skip(tmp_path):
    result, runs, cps, evs = setup_data(tmp_path)
    result["sampling"]["proposals"][0]["params"].update(
        data_epochs=2, learning_rate=0.0000400000001
    )
    plan = build_plan(result, runs, cps, evs, "snapshot", BATCH)
    assert not plan["nodes"]
    assert plan["proposals"][0]["requires_skip"]
    assert [r["id"] for r in plan["proposals"][0]["existing_runs"]] == [1, 2]


def test_shared_stages_created_once_proposal_major(tmp_path):
    result, runs, cps, evs = setup_data(tmp_path)
    p = result["sampling"]["proposals"][0]["params"]
    result["sampling"]["proposals"] = [
        {"params": dict(p, learning_rate=1e-5, data_epochs=2)},
        {"params": dict(p, learning_rate=2e-5, data_epochs=2)},
        {"params": dict(p, learning_rate=1e-5, data_epochs=4)},
    ]
    plan = build_plan(result, runs, cps, evs, "snapshot", BATCH)
    assert [n["owner"] for n in plan["nodes"]] == ["prop_001"] * 2 + [
        "prop_002"
    ] * 2 + ["prop_003"] * 2
    assert len(plan["proposals"][2]["shared_runs"]) == 2
    assert (
        plan["nodes"][4]["config"]["continuation"]["source_run"]
        == plan["nodes"][1]["name"]
    )


def test_fully_shared_proposal_skip(tmp_path):
    result, runs, cps, evs = setup_data(tmp_path)
    result["sampling"]["proposals"][0]["params"]
    result["sampling"]["proposals"] *= 2
    plan = build_plan(result, runs, cps, evs, "snapshot", BATCH)
    assert len(plan["nodes"]) == 2 and plan["proposals"][1]["requires_skip"]


def test_fixed_fields_preserved(tmp_path):
    plan, _, runs, _, _ = plan_for(tmp_path)
    for node in plan["nodes"]:
        source = runs[-1]["normalized_config_json"]
        for field in ("data", "environment", "training", "evaluation"):
            assert source[field] == node["config"][field]


@pytest.mark.parametrize("kind", ["name", "incomplete", "epoch", "length"])
def test_invalid_export_blocked(tmp_path, kind):
    result, runs, cps, evs = setup_data(tmp_path)
    params = result["sampling"]["proposals"][0]["params"]
    if kind == "name":
        extra = deepcopy(runs[0])
        extra.update(id=55, name=run_name(params, 3), status="failed", seed=2)
        runs.append(extra)
    elif kind == "incomplete":
        runs[-1]["status"] = "failed"
    elif kind == "epoch":
        params["n_epochs"] = 100
    else:
        params.update(depth=20, **{f"width_{i}": i + 1000 for i in range(1, 21)})
        for i in range(1, 21):
            result["config"]["search_space"][f"width_{i}"] = {"low": 1, "high": 9999}
    with pytest.raises(ValueError):
        build_plan(result, runs, cps, evs, "snapshot", BATCH)


def test_no_cohort_fallback(tmp_path):
    result, runs, cps, evs = setup_data(tmp_path)
    result["history"]["selected_cohort"] = "other"
    with pytest.raises(ValueError, match="no longer available"):
        build_plan(result, runs, cps, evs, "snapshot", BATCH)


def init_git(root):
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / ".gitignore").write_text("/configs/\n")
    (root / "configs").mkdir(exist_ok=True)


def test_publish_real_yaml_queue_dependency_graph_and_checkpoint(tmp_path):
    from types import SimpleNamespace

    from train_and_eval.queue_execution.core import plan_jobs

    plan, *_ = plan_for(tmp_path)
    init_git(tmp_path)
    preflight(tmp_path, plan)
    publish(tmp_path, plan)
    validate_files(plan, lambda p: (tmp_path / p).read_text())
    jobs = plan_jobs(
        list(
            enumerate(
                [
                    SimpleNamespace(path=tmp_path / n["path"], name=n["name"])
                    for n in plan["nodes"]
                ],
                1,
            )
        )
    )
    assert len(jobs) == 2
    assert jobs[0].source == "run3"
    assert not list((tmp_path / "configs").glob(".optuna-preparing-*"))
    assert not list((tmp_path / plan["config_root"]).rglob("f_v00"))
    manifest = json.loads(
        (tmp_path / plan["config_root"] / "preparation.json").read_text()
    )
    assert len(manifest["configs"]) == 2
    with pytest.raises(ValueError, match="overwrite"):
        preflight(tmp_path, plan)


def test_tampered_checkpoint_blocked_before_any_config(tmp_path):
    plan, _, _, cps, _ = plan_for(tmp_path)
    init_git(tmp_path)
    (tmp_path / cps[-1]["relative_path"]).write_bytes(b"bad")
    with pytest.raises(ValueError, match="integrity"):
        preflight(tmp_path, plan)
    assert not (tmp_path / plan["config_root"]).exists()


def test_pending_local_config_name_blocked(tmp_path):
    plan, *_ = plan_for(tmp_path)
    init_git(tmp_path)
    (tmp_path / "configs/old.yml").write_text(
        yaml.safe_dump(plan["nodes"][0]["config"])
    )
    with pytest.raises(ValueError, match="already has a local config"):
        preflight(tmp_path, plan)


def test_publish_failure_rolls_back_only_owned_trees(tmp_path, monkeypatch):
    plan, *_ = plan_for(tmp_path)
    init_git(tmp_path)
    sentinel = tmp_path / "configs/keep.txt"
    sentinel.write_text("keep")
    import train_and_eval.optuna.export as exporter

    validate = exporter.validate_files
    count = 0

    def fail_after_publish(*args):
        nonlocal count
        count += 1
        if count == 2:
            raise ValueError("post-write failure")
        return validate(*args)

    monkeypatch.setattr(exporter, "validate_files", fail_after_publish)
    with pytest.raises(ValueError, match="post-write failure"):
        publish(tmp_path, plan)
    assert sentinel.read_text() == "keep"
    assert not (tmp_path / plan["config_root"]).exists()
    assert not (tmp_path / plan["queue_root"]).exists()


def test_skipping_everything_no_files(tmp_path, monkeypatch):
    result, runs, cps, evs = setup_data(tmp_path)
    result["sampling"]["proposals"][0]["params"]["data_epochs"] = 3
    import train_and_eval.optuna.export as exporter

    monkeypatch.setattr(exporter, "snapshot", lambda: (runs, cps, evs, "now"))
    monkeypatch.setattr(exporter, "assert_git_clean", lambda root: "commit")
    answers = iter(["PREPARE", "SKIP"])
    plan = prepare_interactive(
        tmp_path, result, Path(BATCH), ask=lambda _: next(answers), interactive=True
    )
    assert plan["status"] == "nothing_to_create"
    assert not (tmp_path / "configs").exists()


@pytest.mark.parametrize("answers", [["NO"], ["PREPARE", "NO"]])
def test_cancel_writes_no_configs(tmp_path, monkeypatch, answers):
    result, runs, cps, evs = setup_data(tmp_path)
    import train_and_eval.optuna.export as exporter

    monkeypatch.setattr(exporter, "snapshot", lambda: (runs, cps, evs, "now"))
    monkeypatch.setattr(exporter, "assert_git_clean", lambda root: "commit")
    monkeypatch.setattr(exporter, "preflight", lambda root, plan: None)
    iterator = iter(answers)
    plan = prepare_interactive(
        tmp_path, result, Path(BATCH), ask=lambda _: next(iterator), interactive=True
    )
    assert plan["status"] == "cancelled"
    assert not (tmp_path / "configs").exists()


def test_noninteractive_never_prompts(tmp_path):
    assert (
        prepare_interactive(
            tmp_path,
            {},
            Path(BATCH),
            ask=lambda _: pytest.fail("prompt"),
            interactive=False,
        )["status"]
        == "not_requested"
    )


def test_rounding_outside_domain_allowed(tmp_path):
    result, runs, cps, evs = setup_data(tmp_path)
    p = result["sampling"]["proposals"][0]["params"]
    p["learning_rate"] = 0.000019999999
    result["config"]["search_space"]["learning_rate"].update(
        low=p["learning_rate"], high=p["learning_rate"]
    )
    plan = build_plan(result, runs, cps, evs, "snapshot", BATCH)
    assert plan["proposals"][0]["effective"]["learning_rate"] == 2e-5
    assert "learning_rate" in plan["proposals"][0]["outside_requested_bounds"]


def test_count_limit():
    from train_and_eval.optuna.config import validate

    cfg = configuration()
    cfg["suggestions"].update(count=1000, max_sampling_attempts=1000)
    with pytest.raises(ValueError, match="999"):
        validate(cfg)


def test_actual_queue_loader_and_numeric_metadata(tmp_path, monkeypatch):
    import run_search_queue as queue

    plan, *_ = plan_for(tmp_path)
    init_git(tmp_path)
    preflight(tmp_path, plan)
    publish(tmp_path, plan)
    monkeypatch.setattr(queue, "ROOT", tmp_path)
    monkeypatch.setattr(queue, "QUEUE_DIRECTORY", tmp_path / "configs/search_queues")
    paths = queue.load_queue_config_paths(plan["queue_name"])
    assert paths == [n["path"] for n in plan["nodes"]]
    metadata = [queue.read_config_meta(p) for p in paths]
    assert [m.name for m in metadata] == [n["name"] for n in plan["nodes"]]
    assert all(m.learning_rate == 4e-5 and m.seed == 1 for m in metadata)


def test_full_interactive_export_and_skip_then_create(tmp_path, monkeypatch):
    result, runs, cps, evs = setup_data(tmp_path)
    new = deepcopy(result["sampling"]["proposals"][0])
    result["sampling"]["proposals"][0]["params"]["data_epochs"] = 3
    result["sampling"]["proposals"].append(new)
    init_git(tmp_path)
    import train_and_eval.optuna.export as exporter

    monkeypatch.setattr(exporter, "snapshot", lambda: (runs, cps, evs, "now"))
    monkeypatch.setattr(exporter, "assert_git_clean", lambda root: "commit")
    answers = iter(["PREPARE", "SKIP", "CREATE"])
    plan = prepare_interactive(
        tmp_path, result, Path(BATCH), ask=lambda _: next(answers), interactive=True
    )
    assert plan["status"] == "created_and_verified"
    assert not (tmp_path / plan["config_root"] / "prop_001").exists()
    assert (tmp_path / plan["config_root"] / "prop_002" / "r_v03").is_dir()
    assert len(plan["nodes"]) == 2


def test_scientific_formatter_matches_existing_convention():
    from extra_tools.maintenance.audit_run_names import scientific_token

    for value in (0, 1e-9, 0.0002, 0.85, 0.9, 1, 0.0000125, 0.0000225):
        assert scientific(value) == scientific_token(value)


def test_report_contains_preparation_and_original_values(tmp_path):
    from train_and_eval.optuna.report import save, update

    plan, result, *_ = plan_for(tmp_path)
    result["status"] = "complete"
    # Report cards normally also include descriptive sampler context.
    for item in result["sampling"]["proposals"]:
        item.update(outside_observed_ranges=[], nearest=[])
    directory = save(result, tmp_path / "reports")
    result["preparation"] = plan
    update(result, directory)
    data = json.loads((directory / "report.json").read_text())
    assert data["preparation"]["proposals"][0]["original"]["data_epochs"] == 5
    assert "Preparation" in (directory / "report.html").read_text()
    assert len(list((tmp_path / "reports").iterdir())) == 1


def test_ignored_and_symlink_checks(tmp_path):
    plan, *_ = plan_for(tmp_path)
    init_git(tmp_path)
    (tmp_path / ".gitignore").write_text("")
    with pytest.raises(ValueError, match="ignored"):
        preflight(tmp_path, plan)
    (tmp_path / ".gitignore").write_text("/configs/\n")
    (tmp_path / "configs/stage_one").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="Symlink"):
        preflight(tmp_path, plan)


def test_pending_maintenance_blocks_preparation(tmp_path, monkeypatch):
    import train_and_eval.optuna.export as exporter

    path = tmp_path / "artifacts/.cleanup_pending.json"
    path.parent.mkdir()
    path.write_text("{}")
    monkeypatch.setattr(exporter, "snapshot", lambda: pytest.fail("database accessed"))
    with pytest.raises(ValueError, match="Pending maintenance"):
        prepare_interactive(
            tmp_path, {}, Path(BATCH), ask=lambda _: "PREPARE", interactive=True
        )


def test_first_real_proposal_precision_and_unequal_yaml(tmp_path):
    result, runs, cps, evs = setup_data(tmp_path)
    p = {
        "depth": 3,
        "width_1": 909,
        "width_2": 378,
        "width_3": 414,
        "learning_rate": 0.0001116558362741277,
        "ent_coef": 0.000347657065351728,
        "gamma": 0.9538463021639674,
        "gae_lambda": 0.8503475455655232,
        "n_epochs": 11,
        "data_epochs": 1,
    }
    result["config"]["search_space"]["width_3"] = {"low": 1, "high": 1024}
    result["sampling"]["proposals"] = [{"params": p}]
    plan = build_plan(result, runs, cps, evs, "snapshot", BATCH)
    init_git(tmp_path)
    preflight(tmp_path, plan)
    publish(tmp_path, plan)
    cfg = yaml.safe_load((tmp_path / plan["nodes"][0]["path"]).read_text())
    assert cfg["ppo"]["learning_rate"] == 0.000111656
    assert cfg["ppo"]["ent_coef"] == 0.000347657
    assert cfg["ppo"]["gamma"] == 0.953846
    assert cfg["ppo"]["gae_lambda"] == 0.850348
    assert "w909w378w414" in cfg["run"]["name"]
    assert cfg["continuation"] == {"mode": "fresh"}
    assert plan["proposals"][0]["original"] == p


def test_export_roundtrip_with_omitted_null_train_ratio(tmp_path):
    from train_and_eval.optuna.export import payloads

    plan, *_ = plan_for(tmp_path)
    for node in plan["nodes"]:
        data = node["config"]["data"]
        data["train_ratio"] = None
        data["train_range"] = {
            "start": "2010-01-01T00:00:00+00:00",
            "end": "2020-01-01T00:00:00+00:00",
        }
        data["validation_range"] = {
            "start": "2020-01-01T00:00:00+00:00",
            "end": "2021-01-01T00:00:00+00:00",
        }

    files = payloads(plan)
    for node in plan["nodes"]:
        exported = yaml.safe_load(files[node["path"]])
        assert "train_ratio" not in exported["data"]

    validate_files(plan, files.__getitem__)

    # A real setting change must still be rejected.
    path = plan["nodes"][0]["path"]
    changed = yaml.safe_load(files[path])
    changed["ppo"]["gamma"] = 0.123456
    files[path] = yaml.safe_dump(changed, sort_keys=False)
    with pytest.raises(ValueError, match="Config round-trip mismatch"):
        validate_files(plan, files.__getitem__)
