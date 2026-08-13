from __future__ import annotations

from typing import Iterable

import pandas as pd
import plotly.express as px
import streamlit as st

from train_and_eval.dashboard.data import (
    DashboardData,
    load_dashboard_data,
)


st.set_page_config(
    page_title="PPO Experiment Dashboard",
    page_icon="📈",
    layout="wide",
)


DISPLAY_LABEL_SUFFIXES = [
    ("run_id", "Run ID"),
    ("run.name", "Run name"),
    ("run.status", "Status"),
    ("run.seed", "Seed"),

    (".n_epochs", "PPO epochs"),
    (".learning_rate", "Learning rate"),
    (".gamma", "Gamma"),
    (".gae_lambda", "GAE lambda"),
    (".ent_coef", "Entropy coefficient"),

    (
        ".environment.exposure_penalty",
        "Exposure penalty",
    ),
    (
        ".environment.turnover_penalty",
        "Turnover penalty",
    ),
    (
        ".environment.drawdown_penalty",
        "Drawdown penalty",
    ),

    ("eval.balanced_score", "Balanced score"),
    ("eval.agent_return", "Agent return"),
    (
        "eval.always_long_return",
        "Always-long return",
    ),
    (
        "eval.agent_max_drawdown",
        "Max drawdown",
    ),
    (
        "eval.always_long_max_drawdown",
        "Always-long max drawdown",
    ),
    (
        "eval.market_exposure",
        "Market exposure",
    ),
    ("eval.round_trips", "Round trips"),
    ("eval.profit_factor", "Profit factor"),
    ("eval.win_rate", "Win rate"),

    ("train.approx_kl", "Approx KL"),
    ("train.clip_fraction", "Clip fraction"),
    (
        "train.explained_variance",
        "Explained variance",
    ),
    ("train.entropy_loss", "Entropy loss"),
    ("train.policy_loss", "Policy loss"),
    ("train.value_loss", "Value loss"),
]


def display_name(column: str) -> str:
    for suffix, label in DISPLAY_LABEL_SUFFIXES:
        if (
            column == suffix
            or column.endswith(suffix)
        ):
            return label

    # Reasonable fallback for columns we have not explicitly
    # named yet.  This keeps newly added metrics readable.
    short = column.split(".")[-1]

    return (
        short
        .replace("_", " ")
        .strip()
        .title()
    )


def is_seed_column(column: str | None) -> bool:
    return bool(
        column
        and (
            column == "run.seed"
            or column.endswith(".seed")
        )
    )


def is_percent_column(column: str | None) -> bool:
    if not column:
        return False

    return any(
        column.endswith(suffix)
        for suffix in (
            "agent_return",
            "always_long_return",
            "agent_max_drawdown",
            "always_long_max_drawdown",
            "market_exposure",
            "win_rate",
        )
    )


def is_scientific_column(column: str | None) -> bool:
    if not column:
        return False

    return any(
        column.endswith(suffix)
        for suffix in (
            "learning_rate",
            "ent_coef",
            "exposure_penalty",
            "turnover_penalty",
            "drawdown_penalty",
            "approx_kl",
        )
    )


def column_index(
    columns: list[str],
    preferred: str | None,
    fallback: int = 0,
) -> int:
    if preferred in columns:
        return columns.index(preferred)

    return min(
        fallback,
        max(len(columns) - 1, 0),
    )


def plotly_labels(
    columns: Iterable[str],
) -> dict[str, str]:
    return {
        column: display_name(column)
        for column in columns
    }


def hover_format(
    column: str,
) -> str | bool:
    if is_percent_column(column):
        return ":+.2%"

    if is_scientific_column(column):
        return ":.3e"

    if column.endswith("round_trips"):
        return ":,.0f"

    if column.endswith("balanced_score"):
        return ":+.5f"

    if column.endswith("profit_factor"):
        return ":.3f"

    return True


@st.cache_data(ttl=30)
def load_data() -> DashboardData:
    return load_dashboard_data()


def find_column(
    columns: Iterable[str],
    *suffixes: str,
) -> str | None:
    candidates = list(columns)

    for suffix in suffixes:
        suffix_lower = suffix.lower()

        exact = [
            column
            for column in candidates
            if column.lower() == suffix_lower
        ]

        if exact:
            return exact[0]

        matching = [
            column
            for column in candidates
            if column.lower().endswith(suffix_lower)
        ]

        if matching:
            return matching[0]

    return None


def numeric_columns(
    frame: pd.DataFrame,
) -> list[str]:
    return [
        column
        for column in frame.columns
        if pd.api.types.is_numeric_dtype(frame[column])
    ]


def usable_columns(
    frame: pd.DataFrame,
) -> list[str]:
    return [
        column
        for column in frame.columns
        if not frame[column].map(
            lambda value: isinstance(
                value,
                (dict, list, tuple, set),
            )
        ).any()
    ]


def apply_sidebar_filters(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.copy()

    st.sidebar.header("Filters")

    filterable = usable_columns(result)

    selected_columns = st.sidebar.multiselect(
        "Filter by any column",
        options=filterable,
        default=[],
        format_func=display_name,
    )

    for index, column in enumerate(selected_columns):
        series = result[column]

        st.sidebar.markdown(
            f"**{display_name(column)}**"
        )

        if pd.api.types.is_numeric_dtype(series):
            valid = series.dropna()

            if valid.empty:
                continue

            minimum = float(valid.min())
            maximum = float(valid.max())

            col1, col2 = st.sidebar.columns(2)

            lower = col1.number_input(
                "min",
                value=minimum,
                key=f"filter_{index}_{column}_min",
                format="%.8g",
            )

            upper = col2.number_input(
                "max",
                value=maximum,
                key=f"filter_{index}_{column}_max",
                format="%.8g",
            )

            result = result.loc[
                series.isna()
                | (
                    (series >= lower)
                    & (series <= upper)
                )
            ]

        else:
            values = sorted(
                {
                    str(value)
                    for value in series.dropna()
                }
            )

            selected = st.sidebar.multiselect(
                "values",
                options=values,
                default=values,
                key=f"filter_{index}_{column}_values",
            )

            if selected:
                result = result.loc[
                    series.astype(str).isin(selected)
                ]

    return result


def metric_value(
    frame: pd.DataFrame,
    column: str | None,
    *,
    mode: str = "max",
    percent: bool = False,
) -> str:
    if column is None or frame.empty:
        return "—"

    values = pd.to_numeric(
        frame[column],
        errors="coerce",
    ).dropna()

    if values.empty:
        return "—"

    if mode == "mean":
        value = values.mean()
    elif mode == "min":
        value = values.min()
    else:
        value = values.max()

    if percent:
        return f"{100.0 * value:+.2f}%"

    return f"{value:+.5f}"


def default_table_columns(
    frame: pd.DataFrame,
) -> list[str]:
    wanted_suffixes = [
        "run_id",
        "run.name",
        "run.status",
        ".seed",
        ".n_epochs",
        ".learning_rate",
        ".gamma",
        ".gae_lambda",
        ".ent_coef",
        ".exposure_penalty",
        ".turnover_penalty",
        "eval.balanced_score",
        "eval.agent_return",
        "eval.agent_max_drawdown",
        "eval.profit_factor",
        "eval.win_rate",
        "eval.market_exposure",
        "eval.round_trips",
        "train.approx_kl",
        "train.clip_fraction",
        "train.explained_variance",
    ]

    result: list[str] = []

    for suffix in wanted_suffixes:
        column = find_column(
            frame.columns,
            suffix,
        )

        if column and column not in result:
            result.append(column)

    return result


def explorer_tab(
    frame: pd.DataFrame,
) -> None:
    st.subheader("Run Explorer")

    available = usable_columns(frame)

    defaults = default_table_columns(frame)

    selected = st.multiselect(
        "Columns",
        options=available,
        default=defaults,
        key="explorer_columns",
        format_func=display_name,
    )

    if not selected:
        st.info("Select at least one column.")
        return

    table = frame[selected].rename(
        columns={
            column: display_name(column)
            for column in selected
        }
    )

    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
    )


def scatter_tab(
    frame: pd.DataFrame,
) -> None:
    st.subheader("Scatter Explorer")

    numeric = numeric_columns(frame)

    if len(numeric) < 2:
        st.warning("Not enough numeric columns.")
        return

    default_x = find_column(
        frame.columns,
        "eval.market_exposure",
    )

    default_y = find_column(
        frame.columns,
        "eval.agent_return",
    )

    default_color = find_column(
        frame.columns,
        "run.seed",
        ".seed",
    )

    default_size = find_column(
        frame.columns,
        "eval.round_trips",
    )

    col1, col2, col3, col4 = st.columns(4)

    x = col1.selectbox(
        "X axis",
        numeric,
        index=column_index(
            numeric,
            default_x,
            0,
        ),
        format_func=display_name,
    )

    y = col2.selectbox(
        "Y axis",
        numeric,
        index=column_index(
            numeric,
            default_y,
            1,
        ),
        format_func=display_name,
    )

    color_options = ["None"] + usable_columns(frame)

    color = col3.selectbox(
        "Color",
        color_options,
        index=(
            color_options.index(default_color)
            if default_color in color_options
            else 0
        ),
        format_func=lambda value: (
            "None"
            if value == "None"
            else display_name(value)
        ),
    )

    size_options = ["None"] + numeric

    size = col4.selectbox(
        "Size",
        size_options,
        index=(
            size_options.index(default_size)
            if default_size in size_options
            else 0
        ),
        format_func=lambda value: (
            "None"
            if value == "None"
            else display_name(value)
        ),
    )

    required = [x, y]

    if color != "None":
        required.append(color)

    if size != "None":
        required.append(size)

    plot_frame = frame.dropna(
        subset=required,
    ).copy()

    # Seed is categorical.  Without this conversion Plotly
    # displays a continuous 1 -> 3 color scale, which is
    # misleading.
    if color != "None" and is_seed_column(color):
        plot_frame[color] = (
            pd.to_numeric(
                plot_frame[color],
                errors="coerce",
            )
            .astype("Int64")
            .astype("string")
        )

    run_name = find_column(
        frame.columns,
        "run.name",
    )

    seed = find_column(
        frame.columns,
        "run.seed",
        ".seed",
    )

    exposure_penalty = find_column(
        frame.columns,
        ".environment.exposure_penalty",
    )

    score = find_column(
        frame.columns,
        "eval.balanced_score",
    )

    agent_return = find_column(
        frame.columns,
        "eval.agent_return",
    )

    max_drawdown = find_column(
        frame.columns,
        "eval.agent_max_drawdown",
    )

    exposure = find_column(
        frame.columns,
        "eval.market_exposure",
    )

    trips = find_column(
        frame.columns,
        "eval.round_trips",
    )

    profit_factor = find_column(
        frame.columns,
        "eval.profit_factor",
    )

    hover_columns = [
        column
        for column in (
            "run_id",
            run_name,
            seed,
            exposure_penalty,
            score,
            agent_return,
            max_drawdown,
            exposure,
            trips,
            profit_factor,
        )
        if (
            column is not None
            and column in plot_frame.columns
        )
    ]

    hover_data = {
        column: hover_format(column)
        for column in hover_columns
    }

    figure = px.scatter(
        plot_frame,
        x=x,
        y=y,
        color=(
            None
            if color == "None"
            else color
        ),
        size=(
            None
            if size == "None"
            else size
        ),
        hover_data=hover_data,
        labels=plotly_labels(
            plot_frame.columns
        ),
    )

    if is_percent_column(x):
        figure.update_xaxes(
            tickformat=".0%",
        )

    if is_percent_column(y):
        figure.update_yaxes(
            tickformat=".0%",
        )

    figure.update_layout(
        legend_title_text=(
            ""
            if color == "None"
            else display_name(color)
        ),
    )

    st.plotly_chart(
        figure,
        width="stretch",
    )

    st.caption(
        "Each point is one run. "
        "Use X for the parameter/cause you want to inspect "
        "and Y for the resulting metric."
    )


def activity_tab(
    frame: pd.DataFrame,
) -> None:
    st.subheader("Activity Map")

    exposure = find_column(
        frame.columns,
        "eval.market_exposure",
        ".market_exposure",
    )

    trips = find_column(
        frame.columns,
        "eval.round_trips",
        ".round_trips",
    )

    agent_return = find_column(
        frame.columns,
        "eval.agent_return",
        ".agent_return",
    )

    seed = find_column(
        frame.columns,
        "run.seed",
        ".seed",
    )

    run_name = find_column(
        frame.columns,
        "run.name",
    )

    score = find_column(
        frame.columns,
        "eval.balanced_score",
    )

    max_drawdown = find_column(
        frame.columns,
        "eval.agent_max_drawdown",
    )

    exposure_penalty = find_column(
        frame.columns,
        ".environment.exposure_penalty",
    )

    if exposure is None or trips is None:
        st.warning(
            "Could not find market exposure "
            "and round trips columns."
        )
        return

    plot_frame = frame.dropna(
        subset=[exposure, trips]
    ).copy()

    if seed and seed in plot_frame.columns:
        plot_frame[seed] = (
            pd.to_numeric(
                plot_frame[seed],
                errors="coerce",
            )
            .astype("Int64")
            .astype("string")
        )

    hover_columns = [
        column
        for column in (
            "run_id",
            run_name,
            seed,
            exposure_penalty,
            score,
            agent_return,
            max_drawdown,
            exposure,
            trips,
        )
        if (
            column is not None
            and column in plot_frame.columns
        )
    ]

    figure = px.scatter(
        plot_frame,
        x=exposure,
        y=trips,
        color=agent_return,
        symbol=seed,
        hover_data={
            column: hover_format(column)
            for column in hover_columns
        },
        labels=plotly_labels(
            plot_frame.columns
        ),
    )

    figure.update_xaxes(
        tickformat=".0%",
    )

    if agent_return:
        figure.update_coloraxes(
            colorbar_title="Agent return",
            colorbar_tickformat=".0%",
        )

    st.plotly_chart(
        figure,
        width="stretch",
    )

    st.caption(
        "X = market exposure, "
        "Y = round trips, "
        "color = agent return, "
        "symbol = seed."
    )


def group_comparison_tab(
    frame: pd.DataFrame,
) -> None:
    st.subheader("Group / Seed Comparison")

    available = usable_columns(frame)
    numeric = numeric_columns(frame)

    default_groups = [
        column
        for column in (
            find_column(frame.columns, ".n_epochs"),
            find_column(
                frame.columns,
                ".learning_rate",
            ),
            find_column(frame.columns, ".gamma"),
            find_column(frame.columns, ".gae_lambda"),
            find_column(frame.columns, ".ent_coef"),
            find_column(
                frame.columns,
                ".exposure_penalty",
            ),
        )
        if column is not None
    ]

    groups = st.multiselect(
        "Group identical runs by",
        options=available,
        default=default_groups,
        format_func=display_name,
    )

    default_metric = find_column(
        frame.columns,
        "eval.balanced_score",
    )

    metric_index = (
        numeric.index(default_metric)
        if default_metric in numeric
        else 0
    )

    metric = st.selectbox(
        "Metric",
        numeric,
        index=metric_index,
        format_func=display_name,
    )

    if not groups:
        st.info("Select at least one grouping column.")
        return

    source = frame[
        groups + [metric]
    ].dropna(
        subset=[metric]
    )

    if source.empty:
        st.info("No data for this grouping.")
        return

    grouped = (
        source
        .groupby(
            groups,
            dropna=False,
        )[metric]
        .agg(
            count="count",
            mean="mean",
            std="std",
            minimum="min",
            maximum="max",
        )
        .reset_index()
        .sort_values(
            "mean",
            ascending=False,
        )
    )

    grouped_display = grouped.rename(
        columns={
            **{
                column: display_name(column)
                for column in groups
            },
            "count": "Runs",
            "mean": "Mean",
            "std": "Std",
            "minimum": "Min",
            "maximum": "Max",
        }
    )

    st.dataframe(
        grouped_display,
        width="stretch",
        hide_index=True,
    )


def run_detail_tab(
    data: DashboardData,
    explorer: pd.DataFrame,
) -> None:
    st.subheader("Run Detail")

    if explorer.empty or "run_id" not in explorer:
        st.info("No runs available.")
        return

    run_name_column = find_column(
        explorer.columns,
        "run.name",
    )

    choices = explorer[
        [
            column
            for column in (
                "run_id",
                run_name_column,
            )
            if column is not None
        ]
    ].drop_duplicates()

    labels: dict[str, int] = {}

    for _, row in choices.iterrows():
        run_id = int(row["run_id"])

        if run_name_column:
            label = (
                f"#{run_id} — "
                f"{row[run_name_column]}"
            )
        else:
            label = f"#{run_id}"

        labels[label] = run_id

    selected_label = st.selectbox(
        "Run",
        list(labels),
    )

    run_id = labels[selected_label]

    selected_row = explorer.loc[
        explorer["run_id"] == run_id
    ]

    detail_frame = (
        selected_row
        .iloc[0]
        .rename_axis("field")
        .reset_index(name="value")
    )

    # A single run contains heterogeneous values:
    # ints, floats, timestamps and strings.  Streamlit uses
    # Arrow for dataframe transport, so keeping all of them in
    # one object column can cause ArrowInvalid type inference.
    detail_frame["value"] = (
        detail_frame["value"]
        .astype("string")
        .fillna("—")
    )

    st.dataframe(
        detail_frame,
        width="stretch",
        hide_index=True,
    )

    st.markdown("### Evaluations")

    evaluations = data.evaluations

    if (
        not evaluations.empty
        and "run_id" in evaluations.columns
    ):
        run_evaluations = evaluations.loc[
            evaluations["run_id"] == run_id
        ].copy()

        st.dataframe(
            run_evaluations,
            width="stretch",
            hide_index=True,
        )

        numeric = numeric_columns(run_evaluations)

        if numeric:
            metric = st.selectbox(
                "Evaluation metric",
                numeric,
                index=0,
                key="evaluation_metric",
            )

            x_column = find_column(
                run_evaluations.columns,
                "run_step",
                "model_step",
                "id",
            )

            if x_column:
                chart_data = run_evaluations.dropna(
                    subset=[x_column, metric]
                )

                figure = px.line(
                    chart_data,
                    x=x_column,
                    y=metric,
                    markers=True,
                )

                st.plotly_chart(
                    figure,
                    width="stretch",
                )

    st.markdown("### Training diagnostics")

    metrics = data.training_metrics

    if (
        not metrics.empty
        and "run_id" in metrics.columns
    ):
        run_metrics = metrics.loc[
            metrics["run_id"] == run_id
        ].copy()

        st.dataframe(
            run_metrics,
            width="stretch",
            hide_index=True,
        )

        x_column = find_column(
            run_metrics.columns,
            "run_step",
            "model_step",
            "id",
        )

        numeric = [
            column
            for column in numeric_columns(run_metrics)
            if column not in {"run_id", x_column}
        ]

        if x_column and numeric:
            selected_metrics = st.multiselect(
                "Training metrics",
                numeric,
                default=[
                    column
                    for column in (
                        find_column(
                            numeric,
                            "approx_kl",
                        ),
                        find_column(
                            numeric,
                            "clip_fraction",
                        ),
                        find_column(
                            numeric,
                            "explained_variance",
                        ),
                    )
                    if column is not None
                ],
            )

            if selected_metrics:
                melted = run_metrics[
                    [x_column] + selected_metrics
                ].melt(
                    id_vars=[x_column],
                    var_name="metric",
                    value_name="value",
                )

                figure = px.line(
                    melted,
                    x=x_column,
                    y="value",
                    color="metric",
                )

                st.plotly_chart(
                    figure,
                    width="stretch",
                )


st.title("PPO Experiment Dashboard")

try:
    data = load_data()
except Exception as exc:
    st.error("Could not load experiment database.")
    st.exception(exc)
    st.stop()

frame = data.explorer

if frame.empty:
    st.warning("The runs table is empty.")
    st.stop()

filtered = apply_sidebar_filters(frame)

score_column = find_column(
    frame.columns,
    "eval.balanced_score",
)

return_column = find_column(
    frame.columns,
    "eval.agent_return",
)

drawdown_column = find_column(
    frame.columns,
    "eval.agent_max_drawdown",
)

status_column = find_column(
    frame.columns,
    "run.status",
)

completed_count = len(filtered)

if status_column is not None:
    completed_count = int(
        (
            filtered[status_column]
            .astype(str)
            .str.lower()
            == "completed"
        ).sum()
    )

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "Visible runs",
    len(filtered),
)

c2.metric(
    "Completed",
    completed_count,
)

c3.metric(
    "Best score",
    metric_value(
        filtered,
        score_column,
    ),
)

c4.metric(
    "Best return",
    metric_value(
        filtered,
        return_column,
        percent=True,
    ),
)

c5.metric(
    "Lowest max DD",
    metric_value(
        filtered,
        drawdown_column,
        percent=True,
    ),
)

st.caption(
    "Filters in the sidebar apply to all dashboard tabs."
)

tabs = st.tabs(
    [
        "Run Explorer",
        "Scatter Explorer",
        "Activity Map",
        "Group Comparison",
        "Run Detail",
    ]
)

with tabs[0]:
    explorer_tab(filtered)

with tabs[1]:
    scatter_tab(filtered)

with tabs[2]:
    activity_tab(filtered)

with tabs[3]:
    group_comparison_tab(filtered)

with tabs[4]:
    run_detail_tab(data, filtered)
