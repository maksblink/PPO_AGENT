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
