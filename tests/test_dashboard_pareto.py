from __future__ import annotations

import pandas as pd

from train_and_eval.dashboard.pareto import (
    pareto_mask,
)


def test_pareto_mask_maximizes_both_objectives() -> None:
    frame = pd.DataFrame(
        {
            "x": [1.0, 2.0, 0.0, 0.5],
            "y": [1.0, 0.0, 2.0, 0.5],
        }
    )

    result = pareto_mask(
        frame,
        x_column="x",
        y_column="y",
        maximize_x=True,
        maximize_y=True,
    )

    assert result.tolist() == [
        True,
        True,
        True,
        False,
    ]


def test_pareto_mask_supports_mixed_objective_directions() -> None:
    frame = pd.DataFrame(
        {
            "risk": [1.0, 2.0, 3.0, 2.0],
            "return": [1.0, 3.0, 2.0, 0.5],
        }
    )

    result = pareto_mask(
        frame,
        x_column="risk",
        y_column="return",
        maximize_x=False,
        maximize_y=True,
    )

    assert result.tolist() == [
        True,
        True,
        False,
        False,
    ]


def test_pareto_mask_keeps_identical_nondominated_points() -> None:
    frame = pd.DataFrame(
        {
            "x": [1.0, 1.0, 0.0],
            "y": [1.0, 1.0, 0.0],
        }
    )

    result = pareto_mask(
        frame,
        x_column="x",
        y_column="y",
        maximize_x=True,
        maximize_y=True,
    )

    assert result.tolist() == [
        True,
        True,
        False,
    ]


def test_pareto_mask_excludes_missing_values() -> None:
    frame = pd.DataFrame(
        {
            "x": [1.0, None, 2.0],
            "y": [1.0, 5.0, None],
        }
    )

    result = pareto_mask(
        frame,
        x_column="x",
        y_column="y",
        maximize_x=True,
        maximize_y=True,
    )

    assert result.tolist() == [
        True,
        False,
        False,
    ]
