from __future__ import annotations

import pandas as pd


def pareto_mask(
    frame: pd.DataFrame,
    *,
    x_column: str,
    y_column: str,
    maximize_x: bool,
    maximize_y: bool,
) -> pd.Series:
    """
    Return a boolean mask identifying the 2D Pareto front.

    A point is dominated when another point is at least as good
    in both objectives and strictly better in at least one.

    Rows with missing/non-numeric objective values are never
    considered Pareto-optimal.
    """
    if x_column not in frame.columns:
        raise KeyError(
            f"Missing Pareto X column: {x_column}"
        )

    if y_column not in frame.columns:
        raise KeyError(
            f"Missing Pareto Y column: {y_column}"
        )

    x = pd.to_numeric(
        frame[x_column],
        errors="coerce",
    )

    y = pd.to_numeric(
        frame[y_column],
        errors="coerce",
    )

    valid = x.notna() & y.notna()

    result = pd.Series(
        False,
        index=frame.index,
        dtype=bool,
    )

    valid_indices = list(
        frame.index[valid]
    )

    if not valid_indices:
        return result

    x_values = x.loc[valid_indices].astype(float)
    y_values = y.loc[valid_indices].astype(float)

    if not maximize_x:
        x_values = -x_values

    if not maximize_y:
        y_values = -y_values

    for index in valid_indices:
        current_x = x_values.loc[index]
        current_y = y_values.loc[index]

        at_least_as_good = (
            (x_values >= current_x)
            & (y_values >= current_y)
        )

        strictly_better = (
            (x_values > current_x)
            | (y_values > current_y)
        )

        dominated = bool(
            (
                at_least_as_good
                & strictly_better
            ).any()
        )

        result.loc[index] = not dominated

    return result
