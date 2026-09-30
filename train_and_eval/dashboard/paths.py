"""Read-only architecture and checkpoint lineage for dashboard views."""
from __future__ import annotations

from numbers import Integral

import pandas as pd


def architecture_label(config: object) -> str:
    if not isinstance(config, dict):
        return "Unknown"
    ppo = config.get("ppo")
    sizes = ppo.get("hidden_sizes") if isinstance(ppo, dict) else None
    if not isinstance(sizes, (list, tuple)) or not sizes:
        return "Unknown"
    if any(isinstance(n, bool) or not isinstance(n, Integral) or n <= 0 for n in sizes):
        return "Unknown"
    if len(set(sizes)) == 1:
        return f"{sizes[0]}×{len(sizes)}"
    return " → ".join(str(n) for n in sizes)


def run_path_metadata(runs: pd.DataFrame, checkpoints: pd.DataFrame) -> pd.DataFrame:
    """Resolve ancestry before UI filtering. A position counts runs, not epochs."""
    checkpoint_runs = (
        dict(zip(checkpoints["id"], checkpoints["run_id"]))
        if {"id", "run_id"}.issubset(checkpoints.columns) else {}
    )
    records = {int(row["id"]): row for row in runs.to_dict("records")}
    parents: dict[int, int | None] = {}
    missing: set[int] = set()
    for run_id, row in records.items():
        source = row.get("source_checkpoint_id")
        if source is None or pd.isna(source):
            parents[run_id] = None
            if row.get("continuation_mode") == "resume":
                missing.add(run_id)
        else:
            parent = checkpoint_runs.get(source)
            parents[run_id] = int(parent) if parent is not None and pd.notna(parent) else None
            if parents[run_id] not in records:
                missing.add(run_id)

    resolved: dict[int, tuple[int | None, int | None, str]] = {}
    for start in records:
        chain: list[int] = []
        seen: set[int] = set()
        current = start
        while current not in resolved:
            if current in seen:
                for node in chain:
                    resolved[node] = (None, None, "Cyclic lineage")
                break
            seen.add(current)
            chain.append(current)
            if current in missing:
                resolved[current] = (None, None, "Missing ancestor")
                break
            parent = parents[current]
            if parent is None:
                resolved[current] = (current, 1, "Complete")
                break
            current = parent
        for node in reversed(chain):
            if node in resolved:
                continue
            root, position, status = resolved[parents[node]]
            resolved[node] = (root, position + 1 if position is not None else None, status)

    result = []
    for run_id, row in records.items():
        root, position, status = resolved[run_id]
        result.append({
            "run_id": run_id,
            "run.architecture": architecture_label(row.get("normalized_config_json")),
            "path.root_run_id": root,
            "path.parent_run_id": parents[run_id],
            "path.position": position,
            "path.status": status,
        })
    frame = pd.DataFrame(result)
    for column in ("path.root_run_id", "path.parent_run_id", "path.position"):
        frame[column] = frame[column].astype("Int64")
    return frame


def visible_path_edges(frame: pd.DataFrame) -> list[tuple[int, int]]:
    """Never bridge hidden runs, siblings, or a different source checkpoint."""
    required = {"run_id", "path.parent_run_id", "path.status",
                "run.source_checkpoint_id", "eval.checkpoint_id"}
    if not required.issubset(frame.columns):
        return []
    records = {int(row["run_id"]): row for row in frame.to_dict("records")}
    edges = []
    for child_id, child in records.items():
        parent_id = child["path.parent_run_id"]
        if pd.isna(parent_id) or child["path.status"] != "Complete":
            continue
        parent = records.get(int(parent_id))
        if parent is None or parent["path.status"] != "Complete":
            continue
        source, shown = child["run.source_checkpoint_id"], parent["eval.checkpoint_id"]
        if pd.notna(source) and pd.notna(shown) and source == shown:
            edges.append((int(parent_id), child_id))
    return sorted(edges)


def add_path_connections(figure, frame: pd.DataFrame, *, x: str, y: str) -> None:
    """Overlay directed, dashed lineage edges without altering scatter groups."""
    visible = frame.dropna(subset=[x, y])
    records = visible.set_index("run_id")
    edges = visible_path_edges(visible)
    if not edges:
        return
    xs, ys = [], []
    for parent_id, child_id in edges:
        parent, child = records.loc[parent_id], records.loc[child_id]
        xs.extend([parent[x], child[x], None])
        ys.extend([parent[y], child[y], None])
        if parent[x] != child[x] or parent[y] != child[y]:
            figure.add_annotation(
                x=child[x], y=child[y], ax=(parent[x] + child[x]) / 2,
                ay=(parent[y] + child[y]) / 2,
                xref="x", yref="y", axref="x", ayref="y",
                text="", showarrow=True, arrowhead=2, arrowsize=1,
                arrowwidth=1, arrowcolor="#a0a0a0",
            )
    figure.add_scatter(
        x=xs, y=ys, mode="lines", name="Checkpoint lineage →",
        line={"color": "#a0a0a0", "width": 1, "dash": "dot"},
        hoverinfo="skip", connectgaps=False,
    )
