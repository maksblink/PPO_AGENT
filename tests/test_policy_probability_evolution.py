from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from extra_tools.policy_probability_evolution import (
    build_checkpoint_diagnostics,
    build_probability_transitions,
    distribution_metrics,
    empirical_wasserstein_distance,
    select_source_checkpoints,
)
from train_and_eval.database.models import (
    CheckpointSaveReason,
    EvaluationDataScope,
    EvaluationPolicyMode,
    EvaluationStatus,
    EvaluationTrigger,
)


def _evaluation(
    evaluation_id: int,
    *,
    status: EvaluationStatus = EvaluationStatus.COMPLETED,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=evaluation_id,
        status=status,
        trigger=EvaluationTrigger.SCHEDULED,
        data_scope=EvaluationDataScope.RUN_VALIDATION,
        policy_mode=(
            EvaluationPolicyMode.DETERMINISTIC_ARGMAX
        ),
    )


def _checkpoint(
    checkpoint_id: int,
    model_step: int,
    *,
    save_reason: CheckpointSaveReason,
    evaluations: list[SimpleNamespace],
) -> SimpleNamespace:
    return SimpleNamespace(
        id=checkpoint_id,
        run_step=model_step,
        model_step=model_step,
        save_reason=save_reason,
        evaluations=evaluations,
    )


def test_select_source_checkpoints_uses_periodic_and_final_history() -> None:
    run = SimpleNamespace(
        id=36,
        name="seed2",
        seed=2,
        checkpoints=[
            _checkpoint(
                140,
                0,
                save_reason=(
                    CheckpointSaveReason.INITIAL
                ),
                evaluations=[],
            ),
            _checkpoint(
                142,
                36_864,
                save_reason=(
                    CheckpointSaveReason.PERIODIC
                ),
                evaluations=[_evaluation(242)],
            ),
            _checkpoint(
                141,
                18_432,
                save_reason=(
                    CheckpointSaveReason.PERIODIC
                ),
                evaluations=[_evaluation(241)],
            ),
            _checkpoint(
                144,
                68_608,
                save_reason=(
                    CheckpointSaveReason.FINAL
                ),
                evaluations=[_evaluation(244)],
            ),
        ],
    )

    sources = select_source_checkpoints(run)

    assert [
        source.checkpoint_id
        for source in sources
    ] == [141, 142, 144]
    assert [
        source.evaluation_id
        for source in sources
    ] == [241, 242, 244]
    assert [
        source.model_step
        for source in sources
    ] == [18_432, 36_864, 68_608]


def test_select_source_checkpoints_rejects_ambiguous_evaluation() -> None:
    run = SimpleNamespace(
        id=36,
        name="seed2",
        seed=2,
        checkpoints=[
            _checkpoint(
                141,
                18_432,
                save_reason=(
                    CheckpointSaveReason.PERIODIC
                ),
                evaluations=[
                    _evaluation(241),
                    _evaluation(341),
                ],
            )
        ],
    )

    with pytest.raises(
        RuntimeError,
        match="expected exactly one",
    ):
        select_source_checkpoints(run)


def test_distribution_metrics_measure_flat_and_long_mass() -> None:
    result = distribution_metrics(
        np.asarray(
            [0.05, 0.20, 0.40, 0.60, 0.80, 0.95]
        )
    )

    assert result == {
        "fraction_le_010": 1 / 6,
        "fraction_le_030": 2 / 6,
        "fraction_lt_050": 3 / 6,
        "fraction_ge_090": 1 / 6,
    }


def test_empirical_wasserstein_distance_for_equal_samples() -> None:
    result = empirical_wasserstein_distance(
        np.asarray([0.10, 0.20]),
        np.asarray([0.80, 0.90]),
    )

    assert result == pytest.approx(0.70)


def test_build_probability_transitions_calculates_deltas() -> None:
    probability_frame = pd.DataFrame(
        {
            "checkpoint_id": [11, 11, 12, 12],
            "p_long": [0.10, 0.20, 0.80, 0.90],
        }
    )

    common = {
        "run_id": 7,
        "run_name": "example",
        "seed": 2,
        "std_p_long": 0.05,
        "p10_p_long": 0.10,
        "p90_p_long": 0.20,
        "fraction_le_010": 0.50,
        "fraction_le_030": 1.00,
        "fraction_lt_050": 1.00,
        "fraction_ge_050": 0.00,
        "fraction_ge_070": 0.00,
        "fraction_ge_090": 0.00,
        "argmax_market_exposure": 0.00,
        "argmax_round_trips": 10,
        "argmax_agent_return": -0.10,
        "argmax_max_drawdown": -0.20,
        "argmax_balanced_score": -0.30,
    }

    summary_frame = pd.DataFrame(
        [
            {
                **common,
                "checkpoint_id": 11,
                "evaluation_id": 21,
                "model_step": 100,
                "mean_p_long": 0.15,
                "median_p_long": 0.15,
            },
            {
                **common,
                "checkpoint_id": 12,
                "evaluation_id": 22,
                "model_step": 200,
                "mean_p_long": 0.85,
                "median_p_long": 0.85,
                "p10_p_long": 0.80,
                "p90_p_long": 0.90,
                "fraction_le_010": 0.00,
                "fraction_le_030": 0.00,
                "fraction_lt_050": 0.00,
                "fraction_ge_050": 1.00,
                "fraction_ge_070": 1.00,
                "fraction_ge_090": 0.50,
                "argmax_market_exposure": 1.00,
                "argmax_round_trips": 2,
                "argmax_agent_return": 0.20,
                "argmax_max_drawdown": -0.10,
                "argmax_balanced_score": 0.10,
            },
        ]
    )

    result = build_probability_transitions(
        probability_frame,
        summary_frame,
    )

    assert len(result) == 1
    row = result.iloc[0]
    assert row["from_checkpoint_id"] == 11
    assert row["to_checkpoint_id"] == 12
    assert row["model_step_delta"] == 100
    assert row[
        "p_long_wasserstein_distance"
    ] == pytest.approx(0.70)
    assert row[
        "delta_mean_p_long"
    ] == pytest.approx(0.70)
    assert row[
        "delta_fraction_ge_050"
    ] == pytest.approx(1.00)
    assert row[
        "delta_argmax_round_trips"
    ] == pytest.approx(-8.0)


def test_build_checkpoint_diagnostics_requires_exact_step_match() -> None:
    summary_frame = pd.DataFrame(
        {
            "run_id": [34],
            "run_name": ["seed1"],
            "seed": [1],
            "checkpoint_id": [133],
            "run_step": [18_432],
            "model_step": [18_432],
        }
    )
    training_frame = pd.DataFrame(
        {
            "run_id": [34],
            "run_name": ["seed1"],
            "seed": [1],
            "run_step": [20_480],
            "model_step": [20_480],
            "approx_kl": [0.001],
        }
    )

    with pytest.raises(
        RuntimeError,
        match="without an exact training metric match",
    ):
        build_checkpoint_diagnostics(
            summary_frame,
            training_frame,
        )
