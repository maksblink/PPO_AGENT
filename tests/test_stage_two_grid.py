from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from train_and_eval.run_config import RunConfig, validate_resume_compatibility
from train_and_eval.walk_forward.config import CheckpointWalkForwardConfig, grid_candidates, stage_config
from train_and_eval.walk_forward.service import select_candidate


def protocol(grid):
    base = RunConfig.model_validate(yaml.safe_load(Path("configs/stage_one/nq5m_v1_seed1.yml").read_text()))
    return CheckpointWalkForwardConfig(name="grid", seed=1, source_checkpoint_id=68, grid=grid).model_copy(update={"run": base})


CYCLE = {"number": 2, "update": {"start": "2019-07-01T00:00:00Z", "end": "2019-07-08T00:00:00Z"},
         "validation": {"start": "2019-07-08T00:00:00Z", "end": "2019-08-05T00:00:00Z"}}


def test_cartesian_order_is_independent_of_yaml_key_order():
    a = {"ppo.learning_rate": [.001, .002], "environment.reward_scale": [1., 10.]}
    b = dict(reversed(list(a.items())))
    expected = [{"environment.reward_scale": r, "ppo.learning_rate": lr} for r in [1., 10.] for lr in [.001, .002]]
    assert grid_candidates(protocol(a)) == grid_candidates(protocol(b)) == expected
    assert grid_candidates(protocol({})) == [{}]


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
        CheckpointWalkForwardConfig.model_validate({"name": "grid", "seed": 1, "source_checkpoint_id": 68, **fields})


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
