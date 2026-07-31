from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import train_and_eval.run_config as run_config_module
from train_and_eval.run_config import (
    ResumeCompatibilityError,
    RunConfigError,
    load_run_config,
    validate_resume_compatibility,
)


def _valid_config() -> dict:
    return {
        "schema_version": 1,
        "run": {
            "name": "test_run",
            "seed": 7,
        },
        "continuation": {
            "mode": "fresh",
        },
        "data": {
            "path": "data/market.parquet",
            "train_ratio": 0.9,
        },
        "training": {
            "duration_unit": "timesteps",
            "duration_amount": 1000,
        },
        "environment": {
            "window": 40,
            "context": "baseline_multiscale_v1",
            "position_side": "long_only",
            "market_timezone": "America/New_York",
            "rth_open": "09:30",
            "rth_close": "16:00",
            "stake_pln": 1000.0,
            "fee_bps": 1.0,
            "swap_bps": 3.0,
            "swap_time": "17:00",
            "swap_timezone": "America/New_York",
            "force_close_on_done": True,
            "reward_scale": 1.0,
            "exposure_penalty": 0.0,
            "turnover_penalty": 0.0,
            "drawdown_penalty": 0.0,
            "profit_reward_mult": 1.0,
            "loss_reward_mult": 1.0,
        },
        "ppo": {
            "device": "cpu",
            "hidden_sizes": [64, 64],
            "n_steps": 256,
            "batch_size": 64,
            "learning_rate": 0.0003,
            "gamma": 0.9,
            "clip_range": 0.2,
            "ent_coef": 0.0,
        },
        "evaluation": {
            "eval_every_steps": 500,
            "checkpoint_every_steps": 500,
            "best_metric": "balanced_score_pct",
            "early_stop_patience_evals": 5,
        },
    }


def _write_config(
    path: Path,
    config: dict,
    *,
    sort_keys: bool = False,
) -> Path:
    path.write_text(
        yaml.safe_dump(
            config,
            sort_keys=sort_keys,
        ),
        encoding="utf-8",
    )
    return path


def test_load_run_config_accepts_valid_config(
    tmp_path: Path,
) -> None:
    path = _write_config(
        tmp_path / "run.yml",
        _valid_config(),
    )

    loaded = load_run_config(
        path,
        verify_data=False,
    )

    assert loaded.config.run.name == "test_run"
    assert loaded.config.ppo.hidden_sizes == (64, 64)
    assert len(loaded.sha256) == 64
    assert '"schema_version":1' in loaded.normalized_json


def test_load_run_config_rejects_unknown_field(
    tmp_path: Path,
) -> None:
    config = _valid_config()
    config["ppo"]["unknown_parameter"] = 123

    path = _write_config(
        tmp_path / "unknown.yml",
        config,
    )

    with pytest.raises(
        RunConfigError,
        match="unknown_parameter",
    ):
        load_run_config(
            path,
            verify_data=False,
        )


def test_load_run_config_rejects_unsafe_run_name(
    tmp_path: Path,
) -> None:
    config = _valid_config()
    config["run"]["name"] = "../other-directory"

    path = _write_config(
        tmp_path / "unsafe-name.yml",
        config,
    )

    with pytest.raises(
        RunConfigError,
        match="letters, digits",
    ):
        load_run_config(
            path,
            verify_data=False,
        )


def test_load_run_config_rejects_batch_larger_than_rollout(
    tmp_path: Path,
) -> None:
    config = _valid_config()
    config["ppo"]["n_steps"] = 64
    config["ppo"]["batch_size"] = 128

    path = _write_config(
        tmp_path / "invalid-batch.yml",
        config,
    )

    with pytest.raises(
        RunConfigError,
        match="batch_size cannot be greater",
    ):
        load_run_config(
            path,
            verify_data=False,
        )


def test_normalized_config_ignores_yaml_key_order(
    tmp_path: Path,
) -> None:
    config = _valid_config()

    first_path = _write_config(
        tmp_path / "first.yml",
        config,
        sort_keys=False,
    )
    second_path = _write_config(
        tmp_path / "second.yml",
        config,
        sort_keys=True,
    )

    first = load_run_config(
        first_path,
        verify_data=False,
    )
    second = load_run_config(
        second_path,
        verify_data=False,
    )

    assert first.normalized_json == second.normalized_json
    assert first.sha256 != second.sha256


def test_load_run_config_verifies_referenced_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_config(
        tmp_path / "run.yml",
        _valid_config(),
    )

    expected_entry = {
        "path": "data/market.parquet",
        "status": "okay",
        "sha256": "a" * 64,
        "rows": 3,
    }

    def fake_verify_market_data(
        path: str,
        *,
        data_directory: str | Path,
        manifest_path: str | Path,
    ) -> tuple[Path, dict]:
        assert path == "data/market.parquet"
        return Path("/tmp/data/market.parquet"), expected_entry

    monkeypatch.setattr(
        run_config_module,
        "verify_market_data",
        fake_verify_market_data,
    )

    loaded = load_run_config(path)

    assert loaded.data_manifest_entry == expected_entry


def _resume_config(
    *,
    source_run: str = "source_run",
) -> dict:
    config = _valid_config()
    config["run"]["name"] = "continued_run"
    config["continuation"] = {
        "mode": "resume",
        "source_run": source_run,
        "checkpoint": "best",
    }

    return config


def _load_without_data_verification(
    tmp_path: Path,
    filename: str,
    config: dict,
):
    path = _write_config(
        tmp_path / filename,
        config,
    )

    return load_run_config(
        path,
        verify_data=False,
    )


def test_load_run_config_accepts_resume_mode(
    tmp_path: Path,
) -> None:
    loaded = _load_without_data_verification(
        tmp_path,
        "resume.yml",
        _resume_config(),
    )

    assert loaded.config.continuation.mode == "resume"
    assert loaded.config.continuation.source_run == "source_run"
    assert loaded.config.continuation.checkpoint == "best"


def test_resume_mode_requires_source_run(
    tmp_path: Path,
) -> None:
    config = _resume_config()
    del config["continuation"]["source_run"]

    path = _write_config(
        tmp_path / "missing-source.yml",
        config,
    )

    with pytest.raises(
        RunConfigError,
        match="source_run",
    ):
        load_run_config(
            path,
            verify_data=False,
        )


def test_fresh_mode_rejects_resume_fields(
    tmp_path: Path,
) -> None:
    config = _valid_config()
    config["continuation"]["source_run"] = "other_run"

    path = _write_config(
        tmp_path / "invalid-fresh.yml",
        config,
    )

    with pytest.raises(
        RunConfigError,
        match="source_run",
    ):
        load_run_config(
            path,
            verify_data=False,
        )


def test_resume_rejects_changed_hidden_sizes(
    tmp_path: Path,
) -> None:
    source_config = _valid_config()
    source_config["run"]["name"] = "source_run"

    current_config = _resume_config()
    current_config["ppo"]["hidden_sizes"] = [128, 64]

    source = _load_without_data_verification(
        tmp_path,
        "source.yml",
        source_config,
    )
    current = _load_without_data_verification(
        tmp_path,
        "current.yml",
        current_config,
    )

    with pytest.raises(
        ResumeCompatibilityError,
        match="ppo.hidden_sizes",
    ):
        validate_resume_compatibility(
            current.config,
            source.config,
        )


@pytest.mark.parametrize(
    ("section", "field", "new_value", "expected_message"),
    [
        (
            "environment",
            "window",
            80,
            "environment.window",
        ),
        (
            "environment",
            "position_side",
            "long_short",
            "environment.position_side",
        ),
    ],
)
def test_resume_rejects_changed_environment_structure(
    tmp_path: Path,
    section: str,
    field: str,
    new_value: object,
    expected_message: str,
) -> None:
    source_config = _valid_config()
    source_config["run"]["name"] = "source_run"

    current_config = _resume_config()
    current_config[section][field] = new_value

    source = _load_without_data_verification(
        tmp_path,
        "source.yml",
        source_config,
    )
    current = _load_without_data_verification(
        tmp_path,
        "current.yml",
        current_config,
    )

    with pytest.raises(
        ResumeCompatibilityError,
        match=expected_message,
    ):
        validate_resume_compatibility(
            current.config,
            source.config,
        )




def test_resume_rejects_changed_registered_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Resume compatibility must also be tested between two context names
    # that are both valid and registered.
    monkeypatch.setattr(
        run_config_module,
        "AVAILABLE_CONTEXTS",
        (
            "baseline_multiscale_v1",
            "different_context",
        ),
    )

    source_config = _valid_config()
    source_config["run"]["name"] = "source_run"

    current_config = _resume_config()
    current_config["environment"]["context"] = "different_context"

    source = _load_without_data_verification(
        tmp_path,
        "source-context.yml",
        source_config,
    )
    current = _load_without_data_verification(
        tmp_path,
        "current-context.yml",
        current_config,
    )

    with pytest.raises(
        ResumeCompatibilityError,
        match="environment.context",
    ):
        validate_resume_compatibility(
            current.config,
            source.config,
        )


def test_resume_allows_training_parameter_changes(
    tmp_path: Path,
) -> None:
    source_config = _valid_config()
    source_config["run"]["name"] = "source_run"

    current_config = _resume_config()
    current_config["training"]["duration_amount"] = 5000
    current_config["ppo"]["learning_rate"] = 0.0001
    current_config["ppo"]["gamma"] = 0.95
    current_config["ppo"]["n_steps"] = 512
    current_config["ppo"]["batch_size"] = 128
    current_config["environment"]["profit_reward_mult"] = 1.2
    current_config["environment"]["force_close_on_done"] = False
    current_config["evaluation"]["eval_every_steps"] = 1000

    source = _load_without_data_verification(
        tmp_path,
        "source.yml",
        source_config,
    )
    current = _load_without_data_verification(
        tmp_path,
        "current.yml",
        current_config,
    )

    validate_resume_compatibility(
        current.config,
        source.config,
    )


def test_resume_rejects_wrong_source_config(
    tmp_path: Path,
) -> None:
    source_config = _valid_config()
    source_config["run"]["name"] = "different_source"

    current_config = _resume_config(
        source_run="expected_source",
    )

    source = _load_without_data_verification(
        tmp_path,
        "source.yml",
        source_config,
    )
    current = _load_without_data_verification(
        tmp_path,
        "current.yml",
        current_config,
    )

    with pytest.raises(
        ResumeCompatibilityError,
        match="does not match",
    ):
        validate_resume_compatibility(
            current.config,
            source.config,
        )


@pytest.mark.parametrize(
    "position_side",
    [
        "long_only",
        "short_only",
        "long_short",
    ],
)
def test_config_accepts_all_position_modes(
    tmp_path: Path,
    position_side: str,
) -> None:
    config = _valid_config()
    config["environment"]["position_side"] = position_side

    loaded = _load_without_data_verification(
        tmp_path,
        f"{position_side}.yml",
        config,
    )

    assert loaded.config.environment.position_side == position_side


def test_config_accepts_negative_reward_coefficients(
    tmp_path: Path,
) -> None:
    config = _valid_config()
    config["environment"]["exposure_penalty"] = -0.25
    config["environment"]["turnover_penalty"] = -0.10
    config["environment"]["drawdown_penalty"] = -0.05

    loaded = _load_without_data_verification(
        tmp_path,
        "negative-coefficients.yml",
        config,
    )

    environment = loaded.config.environment

    assert environment.exposure_penalty == -0.25
    assert environment.turnover_penalty == -0.10
    assert environment.drawdown_penalty == -0.05


def test_config_rejects_weekend_swap_once(
    tmp_path: Path,
) -> None:
    config = _valid_config()
    config["environment"]["weekend_swap_once"] = True

    path = _write_config(
        tmp_path / "weekend-swap.yml",
        config,
    )

    with pytest.raises(
        RunConfigError,
        match="weekend_swap_once",
    ):
        load_run_config(
            path,
            verify_data=False,
        )


@pytest.mark.parametrize(
    "field",
    [
        "rth_open",
        "rth_close",
        "swap_time",
    ],
)
def test_config_rejects_invalid_time(
    tmp_path: Path,
    field: str,
) -> None:
    config = _valid_config()
    config["environment"][field] = "25:90"

    path = _write_config(
        tmp_path / f"invalid-{field}.yml",
        config,
    )

    with pytest.raises(
        RunConfigError,
        match="HH:MM",
    ):
        load_run_config(
            path,
            verify_data=False,
        )


def test_config_rejects_invalid_timezone(
    tmp_path: Path,
) -> None:
    config = _valid_config()
    config["environment"]["swap_timezone"] = "Invalid/Timezone"

    path = _write_config(
        tmp_path / "invalid-timezone.yml",
        config,
    )

    with pytest.raises(
        RunConfigError,
        match="valid IANA time zone",
    ):
        load_run_config(
            path,
            verify_data=False,
        )


def test_config_rejects_rth_close_before_open(
    tmp_path: Path,
) -> None:
    config = _valid_config()
    config["environment"]["rth_open"] = "16:00"
    config["environment"]["rth_close"] = "09:30"

    path = _write_config(
        tmp_path / "invalid-rth.yml",
        config,
    )

    with pytest.raises(
        RunConfigError,
        match="rth_close must be later",
    ):
        load_run_config(
            path,
            verify_data=False,
        )


@pytest.mark.parametrize(
    ("field", "new_value", "expected_message"),
    [
        (
            "market_timezone",
            "Europe/Paris",
            "environment.market_timezone",
        ),
        (
            "rth_open",
            "10:00",
            "environment.rth_open",
        ),
        (
            "rth_close",
            "15:30",
            "environment.rth_close",
        ),
        (
            "swap_time",
            "18:00",
            "environment.swap_time",
        ),
        (
            "swap_timezone",
            "UTC",
            "environment.swap_timezone",
        ),
    ],
)
def test_resume_rejects_changed_time_context(
    tmp_path: Path,
    field: str,
    new_value: object,
    expected_message: str,
) -> None:
    source_config = _valid_config()
    source_config["run"]["name"] = "source_run"

    current_config = _resume_config()
    current_config["environment"][field] = new_value

    source = _load_without_data_verification(
        tmp_path,
        "source-time-context.yml",
        source_config,
    )
    current = _load_without_data_verification(
        tmp_path,
        "current-time-context.yml",
        current_config,
    )

    with pytest.raises(
        ResumeCompatibilityError,
        match=expected_message,
    ):
        validate_resume_compatibility(
            current.config,
            source.config,
        )


def test_config_rejects_unknown_context(
    tmp_path: Path,
) -> None:
    config = _valid_config()
    config["environment"]["context"] = "unknown_context"

    path = _write_config(
        tmp_path / "unknown-context.yml",
        config,
    )

    with pytest.raises(
        RunConfigError,
        match="unknown context",
    ):
        load_run_config(
            path,
            verify_data=False,
        )


@pytest.mark.parametrize(
    ("duration_unit", "duration_amount"),
    [
        ("data_epochs", 2),
        ("timesteps", 500_000),
    ],
)
def test_config_accepts_training_duration_modes(
    tmp_path: Path,
    duration_unit: str,
    duration_amount: int,
) -> None:
    config = _valid_config()
    config["training"] = {
        "duration_unit": duration_unit,
        "duration_amount": duration_amount,
    }

    loaded = _load_without_data_verification(
        tmp_path,
        f"{duration_unit}.yml",
        config,
    )

    assert (
        loaded.config.training.duration_unit
        == duration_unit
    )
    assert (
        loaded.config.training.duration_amount
        == duration_amount
    )


@pytest.mark.parametrize(
    "duration_amount",
    [
        0,
        -1,
        1.5,
        "2",
    ],
)
def test_config_rejects_invalid_duration_amount(
    tmp_path: Path,
    duration_amount: object,
) -> None:
    config = _valid_config()
    config["training"]["duration_amount"] = duration_amount

    path = _write_config(
        tmp_path / "invalid-duration.yml",
        config,
    )

    with pytest.raises(
        RunConfigError,
        match="duration_amount",
    ):
        load_run_config(
            path,
            verify_data=False,
        )


def test_config_rejects_unknown_duration_unit(
    tmp_path: Path,
) -> None:
    config = _valid_config()
    config["training"]["duration_unit"] = "minutes"

    path = _write_config(
        tmp_path / "invalid-unit.yml",
        config,
    )

    with pytest.raises(
        RunConfigError,
        match="duration_unit",
    ):
        load_run_config(
            path,
            verify_data=False,
        )


def test_ppo_section_rejects_timesteps(
    tmp_path: Path,
) -> None:
    config = _valid_config()
    config["ppo"]["timesteps"] = 1000

    path = _write_config(
        tmp_path / "duplicate-duration.yml",
        config,
    )

    with pytest.raises(
        RunConfigError,
        match="timesteps",
    ):
        load_run_config(
            path,
            verify_data=False,
        )


def test_training_duration_resolves_exact_steps() -> None:
    epochs_config = _valid_config()
    epochs_config["training"] = {
        "duration_unit": "data_epochs",
        "duration_amount": 3,
    }

    timesteps_config = _valid_config()
    timesteps_config["training"] = {
        "duration_unit": "timesteps",
        "duration_amount": 12_345,
    }

    epochs = run_config_module.RunConfig.model_validate(
        epochs_config
    )
    timesteps = run_config_module.RunConfig.model_validate(
        timesteps_config
    )

    assert epochs.training.resolve_training_steps(
        steps_per_data_epoch=1000,
    ) == 3000

    assert timesteps.training.resolve_training_steps(
        steps_per_data_epoch=1000,
    ) == 12_345
