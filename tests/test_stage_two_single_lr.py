import pytest
from pydantic import ValidationError

from train_and_eval.walk_forward.config import CheckpointWalkForwardConfig


def test_one_learning_rate_controls_update_and_refit():
    config = CheckpointWalkForwardConfig(
        name="single_lr", seed=1, source_checkpoint_id=68,
        learning_rate=0.00015,
    )
    assert config.learning_rate == 0.00015
    assert not hasattr(config, "learning_rates")
    assert not hasattr(config, "refit_learning_rate")
    exported = config.model_dump(mode="json")
    assert exported["learning_rate"] == 0.00015
    assert "learning_rates" not in exported
    assert "refit_learning_rate" not in exported
    restored = CheckpointWalkForwardConfig.model_validate(exported)
    assert restored.learning_rate == 0.00015


@pytest.mark.parametrize("field", ["learning_rates", "refit_learning_rate"])
def test_old_lr_fields_are_rejected(field):
    with pytest.raises(ValidationError, match="Extra inputs"):
        CheckpointWalkForwardConfig.model_validate({
            "name": "single_lr", "seed": 1, "source_checkpoint_id": 68,
            field: [0.000075] if field == "learning_rates" else 0.000075,
        })


@pytest.mark.parametrize("lr", [0, -0.001, float("nan"), float("inf")])
def test_invalid_learning_rate_is_rejected(lr):
    with pytest.raises(ValidationError):
        CheckpointWalkForwardConfig(
            name="single_lr", seed=1, source_checkpoint_id=68,
            learning_rate=lr,
        )


@pytest.mark.parametrize("fields", [
    {"schema_version": 1},
    {"initial_time_fraction": .5},
    {"initial_epochs": 1},
    {"run": None},
    {"selection_rule": "final_balanced_score_then_candidate_order"},
])
def test_legacy_protocol_fields_are_rejected(tmp_path, fields):
    import yaml
    from train_and_eval.walk_forward.config import load_protocol
    path = tmp_path / "protocol.yml"
    path.write_text(yaml.safe_dump({
        "schema_version": 2, "name": "stage_two", "seed": 1,
        "source_checkpoint_id": 68, **fields,
    }))
    with pytest.raises(ValidationError):
        load_protocol(path)
