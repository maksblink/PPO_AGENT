from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from train_and_eval.run_config import RunConfig, validate_resume_compatibility
from train_and_eval.walk_forward.config import CheckpointWalkForwardConfig, GRID_FIELDS, grid_candidates, stage_config
from train_and_eval.walk_forward.service import select_candidate


def complete_grid():
    grid = yaml.safe_load(Path("configs/stage_two/nq5m_v1_seed1.yml").read_text())["grid"]
    # Unit cases start with one option per field; tests add their own search dimensions.
    return {name: [options[0]] for name, options in grid.items()}


def protocol(grid):
    base = RunConfig.model_validate(yaml.safe_load(Path("configs/stage_one/nq5m_v1_seed1.yml").read_text()))
    return CheckpointWalkForwardConfig(name="grid", seed=1, source_checkpoint_id=68, grid=complete_grid() | grid).model_copy(update={"run": base})


CYCLE = {"number": 2, "update": {"start": "2019-07-01T00:00:00Z", "end": "2019-07-08T00:00:00Z"},
         "validation": {"start": "2019-07-08T00:00:00Z", "end": "2019-08-05T00:00:00Z"}}


def test_cartesian_order_is_independent_of_yaml_key_order():
    a = {"ppo.learning_rate": [.001, .002], "environment.reward_scale": [1., 10.]}
    b = dict(reversed(list(a.items())))
    expected = [{"environment.reward_scale": r, "ppo.learning_rate": lr} for r in [1., 10.] for lr in [.001, .002]]
    actual = grid_candidates(protocol(a))
    assert actual == grid_candidates(protocol(b))
    assert [{name: candidate[name] for name in a} for candidate in actual] == expected
    assert len(grid_candidates(protocol({}))) == 1
    assert set(actual[0]) == GRID_FIELDS


@pytest.mark.parametrize("name", ["environment.stake_pln", "environment.fee_bps", "environment.swap_bps",
    "environment.window", "environment.context", "environment.position_side", "environment.force_close_on_done",
    "ppo.hidden_sizes", "ppo.activation", "ppo.device", "unknown"])
def test_inherited_or_structural_fields_cannot_be_grid_options(name):
    with pytest.raises(ValidationError, match="inherited"):
        protocol({name: [1]})


@pytest.mark.parametrize("options", [[], [.001, .001], [float("nan")], [float("inf")], .001])
def test_invalid_option_lists(options):
    with pytest.raises(ValidationError):
        protocol({"ppo.learning_rate": options})


@pytest.mark.parametrize("fields", [{"schema_version": 2}, {"learning_rate": .001}, {"learning_rates": [.001]},
    {"refit_learning_rate": .001}, {"reject_updates": False}, {"selection_rule": "final_checkpoint"}, {"run": None}])
def test_old_protocol_is_rejected(fields):
    with pytest.raises(ValidationError):
        CheckpointWalkForwardConfig.model_validate({"name": "grid", "seed": 1, "source_checkpoint_id": 68, "grid": complete_grid(), **fields})


def test_candidate_and_refit_use_selected_options_and_inherit_costs():
    p = protocol({"ppo.learning_rate": [.0001, .0002], "environment.turnover_penalty": [.003]})
    for order, lr in enumerate([.0001, .0002]):
        candidate = stage_config(p, CYCLE, role="candidate", source=(p.run.run.name, 68), candidate_order=order)
        refit = stage_config(p, CYCLE, role="refit", source=(candidate.run.name, 69), candidate_order=order)
        validate_resume_compatibility(candidate, p.run)
        validate_resume_compatibility(refit, candidate)
        assert refit.ppo.learning_rate == candidate.ppo.learning_rate == lr
        assert refit.environment.turnover_penalty == .003
        assert refit.evaluation.training_mode == "none"
        for name in ["stake_pln", "fee_bps", "swap_bps"]:
            assert getattr(candidate.environment, name) == getattr(refit.environment, name) == getattr(p.run.environment, name)
    assert stage_config(p, CYCLE, role="candidate", candidate_order=0).run.name != stage_config(p, CYCLE, role="candidate", candidate_order=1).run.name


def test_first_improvement_not_best_and_ties_rejected():
    records = [{"balanced_score": v} for v in [-.3, -.2, -.1, .5]]
    assert select_candidate(records, -.2) is records[2]
    assert select_candidate(records[:2], -.2) is None
    assert select_candidate([], -.2) is None
    for value in [float("nan"), float("inf"), -float("inf")]:
        with pytest.raises(ValueError, match="finite"):
            select_candidate([], value)
        with pytest.raises(ValueError, match="finite"):
            select_candidate([{"balanced_score": value}], 0.)


@pytest.mark.parametrize("grid", [{"ppo.learning_rate": [0]}, {"ppo.n_steps": [1000]},
    {"ppo.batch_size": [2048]}, {"environment.reward_scale": [0]}])
def test_invalid_combination_cannot_build_training_config(grid):
    with pytest.raises(ValidationError):
        stage_config(protocol(grid), CYCLE, role="candidate")


@pytest.mark.parametrize("missing", sorted(GRID_FIELDS))
def test_every_grid_field_is_required(missing):
    grid = complete_grid()
    del grid[missing]
    with pytest.raises(ValidationError, match=f"grid.{missing}: required"):
        CheckpointWalkForwardConfig(name="missing", seed=1, source_checkpoint_id=68, grid=grid)


def test_all_missing_fields_are_reported_together():
    for extra in ({}, {"grid": {}}):
        with pytest.raises(ValidationError) as captured:
            CheckpointWalkForwardConfig(name="missing", seed=1, source_checkpoint_id=68, **extra)
        for field in GRID_FIELDS:
            assert f"grid.{field}: required" in str(captured.value)


@pytest.mark.parametrize("name,options", [
    ("ppo.n_epochs", [1.5]), ("ppo.n_epochs", [True]),
    ("ppo.learning_rate", ["0.001"]), ("ppo.learning_rate", [False]),
    ("ppo.normalize_advantage", [1]), ("ppo.normalize_advantage", ["false"]),
    ("ppo.clip_range_vf", [0]), ("ppo.target_kl", [-1]),
    ("environment.profit_reward_mult", [None]), ("environment.loss_reward_mult", [-1]),
    ("ppo.gamma", [0.9, 1.1]), ("ppo.batch_size", [256, 300]),
])
def test_invalid_options_are_rejected_without_resolving_a_checkpoint(name, options):
    with pytest.raises(ValidationError, match="grid."):
        protocol({name: options})


def test_nullable_settings_require_explicit_lists():
    p = protocol({"ppo.clip_range_vf": [None], "ppo.target_kl": [None]})
    assert p.grid["ppo.target_kl"] == [None]
    with pytest.raises(ValidationError, match="nonempty list"):
        protocol({"ppo.target_kl": None})


def test_explicit_options_override_different_stage_one_settings():
    p = protocol({})
    raw = p.run.model_dump(mode="json")
    raw["ppo"].update(learning_rate=.02, gamma=.7, n_epochs=7)
    raw["environment"].update(reward_scale=100, loss_reward_mult=8)
    p = p.model_copy(update={"run": RunConfig.model_validate(raw)})
    for role in ("candidate", "refit"):
        cfg = stage_config(p, CYCLE, role=role)
        for name, options in p.grid.items():
            section, field = name.split(".")
            assert getattr(getattr(cfg, section), field) == options[0]


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_supplied_seed_configs_have_all_grid_fields(seed):
    raw = yaml.safe_load(Path(f"configs/stage_two/nq5m_v1_seed{seed}.yml").read_text())
    raw["source_checkpoint_id"] = 68  # Seed 2/3 templates intentionally require a real checkpoint.
    p = CheckpointWalkForwardConfig.model_validate(raw)
    assert set(p.grid) == GRID_FIELDS


@pytest.mark.parametrize("command", ["plan", "run"])
def test_cli_reports_missing_and_invalid_options_before_database(command, tmp_path, monkeypatch, capsys):
    import sys
    from train_and_eval.walk_forward.__main__ import main
    from train_and_eval.database import session
    def forbidden_database(*args, **kwargs):
        pytest.fail("Invalid config must not create a database engine")
    monkeypatch.setattr(session, "create_database_engine", forbidden_database)
    raw = {"name": "bad", "seed": 1, "source_checkpoint_id": 68, "grid": complete_grid()}
    del raw["grid"]["ppo.gamma"]
    del raw["grid"]["environment.reward_scale"]
    raw["grid"]["ppo.learning_rate"] = [0]
    path = tmp_path / "bad.yml"
    path.write_text(yaml.safe_dump(raw))
    monkeypatch.setattr(sys, "argv", ["walk_forward", command, "--config", str(path)])
    with pytest.raises(SystemExit) as captured:
        main()
    assert captured.value.code == 2
    output = capsys.readouterr().err
    assert "grid.ppo.gamma: required" in output
    assert "grid.environment.reward_scale: required" in output
    assert "grid.ppo.learning_rate[0]" in output
    assert "Traceback" not in output
