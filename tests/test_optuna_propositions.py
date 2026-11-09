"""Synthetic database rows and real in-memory Optuna; no GPU or PostgreSQL."""

import json
from copy import deepcopy

import pytest

pytest.importorskip("optuna")
from train_and_eval.optuna.__main__ import analyze, main
from train_and_eval.optuna.config import load, validate
from train_and_eval.optuna.history import assemble, select
from train_and_eval.optuna.propose import (
    fit_space,
    in_domain,
    objective_values,
    suggest,
)
from train_and_eval.optuna.report import render, save


def configuration():
    space = {
        n: {
            "type": "float",
            "low": v,
            "high": v,
            "log": n in ("learning_rate", "ent_coef"),
        }
        for n, v in {
            "learning_rate": 4e-5,
            "ent_coef": 2e-4,
            "gamma": 0.9,
            "gae_lambda": 0.95,
        }.items()
    }
    space.update(
        {
            n: {"type": "int", "low": v, "high": v, "step": 1}
            for n, v in {
                "depth": 2,
                "width_1": 128,
                "width_2": 128,
                "n_epochs": 9,
                "data_epochs": 2,
            }.items()
        }
    )
    return validate({"search_space": space})


def dataset():
    runs, cps, evs = [], [], []
    for i in (1, 2, 3):
        cfg = {
            "run": {"name": f"run{i}", "seed": 1},
            "continuation": {"mode": "fresh"}
            if i == 1
            else {"mode": "resume", "source_run": f"run{i - 1}", "checkpoint": "final"},
            "ppo": {
                "hidden_sizes": [128, 128],
                "learning_rate": 4e-5,
                "ent_coef": 2e-4,
                "gamma": 0.9,
                "gae_lambda": 0.95,
                "n_epochs": 9,
                "device": "cuda",
            },
            "data": {"path": "data.parquet"},
            "environment": {"cost": 0.001},
            "training": {"duration_unit": "data_epochs", "duration_amount": 1},
            "evaluation": {"device": "cpu"},
        }
        runs.append(
            {
                "id": i,
                "name": f"run{i}",
                "seed": 1,
                "status": "completed",
                "cycle_id": None,
                "stopped_early": False,
                "training_steps_requested": 100,
                "training_steps_completed": 100,
                "steps_per_data_epoch": 100,
                "source_checkpoint_id": None if i == 1 else i - 1,
                "data_sha256": "dataset",
                "normalized_config_json": cfg,
            }
        )
        cps.append(
            {
                "id": i,
                "run_id": i,
                "run_step": 100,
                "model_step": 100 * i,
                "save_reason": "final",
            }
        )
        for j, scope in enumerate(("run_training", "run_validation")):
            evs.append(
                {
                    "id": 2 * i + j,
                    "checkpoint_id": i,
                    "data_scope": scope,
                    "status": "completed",
                    "trigger": "final",
                    "steps_completed": 10,
                    "steps_expected": 10,
                    "agent_return": [0.1, 0.5, 0.3][i - 1],
                    "balanced_score": [0.2, 0.1, 0.4][i - 1],
                    "agent_max_drawdown": -0.2,
                    "seed": 1,
                }
            )
    return runs, cps, evs


def rows():
    return assemble(*dataset())[0]


@pytest.mark.parametrize(
    "mode,metric,expected",
    [
        ("all", "return", [1, 2, 3]),
        ("last", "return", [3]),
        ("best", "return", [2]),
        ("best", "balanced_score", [3]),
    ],
)
def test_modes_follow_ancestry(mode, metric, expected):
    result, leaves, _ = select(rows(), mode, metric)
    assert [r["id"] for r in result] == expected
    assert leaves[0]["path"] == [1, 2, 3]


@pytest.mark.parametrize(
    "change",
    [
        lambda r, c, e: r[0].update(status="failed"),
        lambda r, c, e: r[0].update(seed=2),
        lambda r, c, e: r[0].update(stopped_early=True),
        lambda r, c, e: c[0].update(model_step=999),
        lambda r, c, e: c[0].update(save_reason="scheduled"),
        lambda r, c, e: e[1].update(agent_return=float("nan")),
        lambda r, c, e: e[1].update(agent_max_drawdown=0.1),
        lambda r, c, e: e[1].update(steps_completed=2),
        lambda r, c, e: r[0].update(cycle_id=4),
    ],
)
def test_bad_ancestor_excludes_descendants(change):
    data = dataset()
    change(*data)
    rr, audit = assemble(*data)
    assert rr == [] and len(audit["exclusions"]) == 3


def test_parameter_change_excluded():
    data = dataset()
    data[0][1]["normalized_config_json"]["ppo"]["gae_lambda"] = 0.85
    rr, audit = assemble(*data)
    assert [r["id"] for r in rr] == [1] and len(audit["exclusions"]) == 2


def test_conflicting_replica_is_not_cherry_picked():
    rr = rows()
    duplicate = deepcopy(rr[1])
    duplicate.update(id=99, path=[99])
    duplicate["metrics"]["agent_return"] = 9
    with pytest.raises(ValueError, match="Conflicting"):
        select(rr + [duplicate], "last", "return")
    duplicate["metrics"] = deepcopy(rr[1]["metrics"])
    chosen, _, aliases = select(rr + [duplicate], "all", "return")
    assert len(chosen) == 3 and aliases[99] == 2


def test_branch_best_and_shared_prefix_dedup():
    rr = rows()
    other = deepcopy(rr[2])
    other.update(id=4, path=[1, 2, 4])
    other["params"]["data_epochs"] = 4
    chosen, leaves, _ = select(rr + [other], "best", "return")
    assert len(leaves) == 2 and [r["id"] for r in chosen] == [2]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c.update(typo=True),
        lambda c: c["history"].update(seed=2),
        lambda c: c["sampler"].update(multivariate=False),
        lambda c: c["search_space"]["learning_rate"].update(low=0),
        lambda c: c["search_space"]["gamma"].update(high=1.1),
        lambda c: c["search_space"]["width_1"].update(step=0),
        lambda c: c["search_space"]["depth"].update(high=3),
        lambda c: c["search_space"]["learning_rate"].update(step=0.1),
        lambda c: c["objective"].update(name="unknown"),
    ],
)
def test_config_rejects_errors(mutate):
    cfg = configuration()
    mutate(cfg)
    with pytest.raises(ValueError):
        validate(cfg)


def test_yaml_duplicate_rejected(tmp_path):
    path = tmp_path / "c.yml"
    path.write_text("schema_version: 1\nschema_version: 1\n")
    with pytest.raises(ValueError, match="Duplicate"):
        load(path)


def test_expanded_bounds_preserve_history_and_conditional_widths():
    cfg = configuration()
    rr = rows()
    rr[0]["params"].update(depth=3, width_3=768, learning_rate=0.001)
    expanded = fit_space(cfg["search_space"], rr)
    assert expanded["depth"]["high"] == 3 and expanded["width_3"]["low"] == 768
    assert (
        expanded["learning_rate"]["high"] == 0.001
        and cfg["search_space"]["depth"]["high"] == 2
    )
    assert not in_domain(rr[0]["params"], cfg["search_space"])


def test_zero_log_history_not_rewritten():
    cfg = configuration()
    rr = rows()
    rr[0]["params"]["ent_coef"] = 0
    with pytest.raises(ValueError, match="cannot use log"):
        fit_space(cfg["search_space"], rr)


def test_known_fixed_point_returns_no_proposal_even_if_not_selected():
    cfg = configuration()
    rr = rows()
    out = suggest(cfg, [rr[-1]], rr)
    assert (
        out["proposals"] == []
        and out["attempts"] == 1
        and out["rejected"] == {"already_evaluated": 1}
    )


def test_fixed_new_point_and_signed_pareto():
    cfg = configuration()
    cfg["search_space"]["learning_rate"].update(low=1e-5, high=1e-5)
    cfg["objective"]["name"] = "pareto"
    out = suggest(cfg, rows(), rows())
    assert (
        len(out["proposals"]) == 1
        and out["proposals"][0]["params"]["learning_rate"] == 1e-5
    )
    assert (
        objective_values(rows()[0], "pareto") == [0.1, -0.2]
        and out["fit_space"]["learning_rate"]["high"] == 4e-5
    )


def test_real_tpe_deterministic_batch():
    cfg = configuration()
    cfg["sampler"]["n_startup_trials"] = 1
    cfg["suggestions"].update(count=3, max_sampling_attempts=30)
    cfg["search_space"]["learning_rate"].update(low=1e-5, high=1e-4)
    a = suggest(cfg, rows(), rows())
    b = suggest(cfg, rows(), rows())
    assert a["proposals"] == b["proposals"] and len(a["proposals"]) == 3
    assert (
        len({json.dumps(p["params"], sort_keys=True) for p in a["proposals"]}) == 3
        and not a["startup_random"]
    )


def test_actual_conditional_depth_no_unused_widths():
    cfg = configuration()
    cfg["search_space"]["depth"].update(low=1, high=2)
    cfg["search_space"]["learning_rate"].update(low=1e-5, high=1e-5)
    cfg["suggestions"]["count"] = 2
    out = suggest(cfg, rows(), rows())
    assert {p["params"]["depth"] for p in out["proposals"]} == {1, 2}
    for p in out["proposals"]:
        assert ("width_2" in p["params"]) == (p["params"]["depth"] == 2)


def test_multiple_cohorts_require_selection():
    data = dataset()
    r = deepcopy(data[0][0])
    r.update(id=8, name="different")
    r["normalized_config_json"]["run"]["name"] = "different"
    r["normalized_config_json"]["environment"]["cost"] = 0.2
    data[0].append(r)
    data[1].append(
        {
            "id": 8,
            "run_id": 8,
            "run_step": 100,
            "model_step": 100,
            "save_reason": "final",
        }
    )
    for e in deepcopy(data[2][:2]):
        e["checkpoint_id"] = 8
        e["id"] += 100
        data[2].append(e)
    out = analyze(configuration(), *data, "now")
    assert (
        out["status"] == "blocked"
        and len(out["history"]["cohorts"]) == 2
        and "sampling" not in out
    )


def test_reports_escape_and_never_overwrite(tmp_path):
    out = analyze(configuration(), *dataset(), "now")
    out["messages"].append("<script>alert(1)</script>")
    assert "&lt;script&gt;alert(1)" in render(out)
    a = save(out, tmp_path)
    b = save(out, tmp_path)
    assert a != b and {p.name for p in a.iterdir()} == {"report.html", "report.json"}
    assert (
        json.loads((a / "report.json").read_text())["status"] == "partial_or_exhausted"
    )


def test_validation_does_not_read_database(tmp_path, monkeypatch):
    import yaml

    from train_and_eval.optuna import __main__ as cli
    monkeypatch.setattr(cli, "assert_git_clean", lambda root: "test-commit")

    path = tmp_path / "c.yml"
    path.write_text(yaml.safe_dump(configuration()))
    monkeypatch.setattr(cli, "snapshot", lambda: pytest.fail("Database accessed"))
    assert main(["--config", str(path), "--validate-config"]) == 0


def test_snapshot_readonly_and_disposal(monkeypatch):
    from train_and_eval.database import session
    from train_and_eval.optuna import history

    statements = []

    class Result:
        def mappings(self):
            return []

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def begin(self):
            return self

        def execute(self, sql):
            statements.append(str(sql))
            return Result()

        def scalar(self, sql):
            statements.append(str(sql))
            return "time"

    class Engine:
        disposed = False

        def connect(self):
            return Connection()

        def dispose(self):
            self.disposed = True

    engine = Engine()
    monkeypatch.setattr(session, "create_database_engine", lambda: engine)
    assert history.snapshot() == ([], [], [], "time")
    assert statements[0] == "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"
    assert (
        all(s.startswith(("SET ", "SELECT ")) for s in statements) and engine.disposed
    )


def test_bad_metric_report_keeps_exclusion_instead_of_hash_failure():
    data = dataset()
    data[2][1]["agent_return"] = float("nan")
    result = analyze(configuration(), *data, "now")
    assert result["status"] == "blocked" and result["history"]["eligible_runs"] == 0
    json.dumps(result, allow_nan=False)


def test_full_cli_writes_only_new_report(tmp_path, monkeypatch):
    import yaml

    from train_and_eval.optuna import __main__ as cli
    monkeypatch.setattr(cli, "assert_git_clean", lambda root: "test-commit")

    cfg = configuration()
    cfg["search_space"]["learning_rate"].update(low=1e-5, high=1e-5)
    cfg["report"]["output_dir"] = str(tmp_path / "reports")
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(cfg))
    original = path.read_bytes()
    monkeypatch.setattr(cli, "snapshot", lambda: (*dataset(), "snapshot"))
    assert main(["--config", str(path)]) == 0
    assert path.read_bytes() == original
    reports = list((tmp_path / "reports").glob("*/report.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text())
    assert report["status"] == "complete" and report["provenance"]["tool_source_sha256"]
