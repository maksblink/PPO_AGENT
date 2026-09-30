"""Training-path selection and metric charts; no persisted search-queue records."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def queue_choices(frame: pd.DataFrame, search: str = "") -> list[int]:
    """Search any member's ID/name and return its fresh-run root."""
    query = search.strip().casefold()
    selected = frame
    if query:
        matches = frame["run.name"].fillna("").astype(str).str.casefold().str.contains(query, regex=False)
        matches |= frame["run_id"].map(lambda value: str(int(value))).str.contains(query.lstrip("#"), regex=False)
        selected = frame.loc[matches]
    return sorted(int(value) for value in selected["path.root_run_id"].dropna().unique())


def queue_runs(frame: pd.DataFrame, root_id: int) -> pd.DataFrame:
    return frame.loc[frame["path.root_run_id"] == root_id].sort_values(
        ["path.position", "run_id"], kind="stable"
    ).copy()


def queue_figure(frame: pd.DataFrame, metric: str, *, label: str,
                 percent: bool = False, focus_run_id: int | None = None) -> go.Figure:
    """Points are run summaries; edges follow parents, never sibling row order."""
    frame = frame.copy()
    frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    shown = frame.dropna(subset=[metric]).set_index("run_id", drop=False)
    xs, ys = [], []
    for child_id, child in shown.iterrows():
        parent_id = child["path.parent_run_id"]
        if pd.isna(parent_id) or int(parent_id) not in shown.index:
            continue
        parent = shown.loc[int(parent_id)]
        # Evaluation points must represent the checkpoint actually resumed.
        if metric.startswith("eval."):
            source = child.get("run.source_checkpoint_id")
            evaluated = parent.get("eval.checkpoint_id")
            if pd.isna(source) or pd.isna(evaluated) or source != evaluated:
                continue
        xs.extend([int(parent["path.position"]), int(child["path.position"]), None])
        ys.extend([parent[metric], child[metric], None])
    figure = go.Figure()
    if xs:
        figure.add_scatter(x=xs, y=ys, mode="lines", name="Continuation",
                           line={"color": "#6ab7ef"}, hoverinfo="skip", connectgaps=False)
    value_format = "+.2%" if percent else ".6g"
    custom = [[int(run_id), row["run.name"], row["run.status"],
               "—" if pd.isna(row["path.parent_run_id"]) else str(int(row["path.parent_run_id"]))]
              for run_id, row in shown.iterrows()]
    figure.add_scatter(
        x=shown["path.position"].astype(int).tolist(), y=shown[metric].tolist(),
        mode="markers+text", name=label,
        text=[f"#{int(i)}" for i in shown.index], textposition="top center",
        marker={"size": [14 if i == focus_run_id else 9 for i in shown.index],
                "color": ["#ffb347" if i == focus_run_id else "#6ab7ef" for i in shown.index]},
        customdata=custom,
        hovertemplate=("Run #%{customdata[0]}<br>%{customdata[1]}"
                       "<br>Position: %{x}<br>Status: %{customdata[2]}"
                       "<br>Parent: %{customdata[3]}<br>Value: %{y:" + value_format + "}<extra></extra>"),
    )
    positions = sorted(int(v) for v in frame["path.position"].unique())
    figure.update_xaxes(title="Run position in training path", tickmode="array", tickvals=positions)
    figure.update_yaxes(title=label, tickformat=".1%" if percent else None)
    figure.update_layout(hovermode="closest")
    return figure
