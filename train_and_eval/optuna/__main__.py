"""Run from the project root: python -m train_and_eval.optuna --config PATH."""

import argparse
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

from . import VERSION
from .config import load
from .history import assemble, digest, select, snapshot
from .git_guard import assert_git_clean
from .propose import suggest
from .report import save


def analyze(cfg, runs, checkpoints, evaluations, timestamp):
    rows, audit = assemble(runs, checkpoints, evaluations)
    result = {
        "schema_version": 1,
        "tool_version": VERSION,
        "status": "blocked",
        "messages": [],
        "config": cfg,
        "history": audit,
        "provenance": {
            "snapshot_at": timestamp,
            "python_version": platform.python_version(),
            "created_at": datetime.now(UTC).isoformat(),
            "input_sha256": digest(
                {"runs": runs, "checkpoints": checkpoints, "evaluations": evaluations}
            ),
        },
    }
    cohorts = {r["cohort"] for r in rows}
    cohort = cfg["history"]["cohort"]
    if cohort is None and len(cohorts) == 1:
        cohort = next(iter(cohorts))
    if cohort not in cohorts:
        result["messages"].append(
            "Set history.cohort to one of the reported cohort IDs; no compatible history selected."
        )
        return result
    rows = [r for r in rows if r["cohort"] == cohort]
    result["history"]["selected_cohort"] = cohort
    try:
        selected, leaves, aliases = select(
            rows, cfg["history"]["mode"], cfg["history"]["best_by"]
        )
        result["history"].update(
            mode=cfg["history"]["mode"],
            selected_observations=selected,
            selected_count=len(selected),
            maximal_paths=len(leaves),
            duplicate_aliases={str(k): v for k, v in aliases.items() if k != v},
        )
        if cfg["report"]["include_full_paths"]:
            by_id = {r["id"]: r for r in rows}
            result["paths"] = [
                {"leaf_run_id": r["id"], "runs": [by_id[i] for i in r["path"]]}
                for r in leaves
            ]
        result["sampling"] = suggest(cfg, selected, rows)
        sampling = result["sampling"]
        result["status"] = (
            "complete"
            if len(sampling["proposals"]) == sampling["requested_count"]
            else "partial_or_exhausted"
        )
        result["messages"] += [
            "TPE models promising regions using imported outcomes; it does not supply a predicted return or a reasoning trace.",
            "All candidate parameters remain frozen across the proposed path. data_epochs means cumulative training duration.",
            "Best mode selects historical peaks and has selection bias; one seed and repeated validation selection do not establish generalization.",
            "Nearby rows and marginal-range flags are descriptive context, not TPE attribution or calibrated confidence.",
        ]
        if sampling["fit_space_expanded"]:
            result["messages"].append(
                "Model bounds include historical values outside the requested domain. Candidates are filtered back to requested bounds/steps; fixed parameters use PartialFixedSampler. Narrow domains may exhaust attempts. This is not conditional inference."
            )
        if sampling["startup_random"]:
            result["messages"].append(
                "Too few selected history observations for TPE startup threshold: proposals use random startup sampling."
            )
        if result["status"] != "complete":
            result["messages"].append(
                "Attempt budget exhausted or fixed configuration already known. No synthetic outcomes were assigned."
            )
    except ValueError as exc:
        result["messages"].append(str(exc))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Read-only seed-1 history -> in-memory Optuna -> terminal + HTML/JSON proposals. No training/config generation."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Validate YAML only; no database or report writes",
    )
    args = parser.parse_args(argv)
    git_commit = assert_git_clean(Path(__file__).resolve().parents[2])
    cfg = load(args.config)
    if args.validate_config:
        print("PASS: configuration valid. No database access or reports written.")
        return 0
    print("Reading one consistent read-only database snapshot...", flush=True)
    runs, cps, evs, stamp = snapshot()
    result = analyze(cfg, runs, cps, evs, stamp)
    result["provenance"]["git_commit"] = git_commit
    result["provenance"]["config_sha256"] = digest(cfg)
    result["provenance"]["tool_source_sha256"] = digest(
        {
            p.name: p.read_text(encoding="utf-8")
            for p in sorted(Path(__file__).parent.glob("*.py"))
        }
    )
    print(
        f"Eligible: {result['history']['eligible_runs']}/{len(runs)}; status: {result['status']}"
    )
    for c in result["history"]["cohorts"]:
        print(f"Cohort {c['id']}: {c['runs']} runs")
    for i, p in enumerate(result.get("sampling", {}).get("proposals", []), 1):
        print(f"\nPROPOSAL {i}\n" + json.dumps(p["params"], indent=2))
    for message in result["messages"]:
        print(message)
    directory = save(result, cfg["report"]["output_dir"])
    print(f"HTML: {directory / 'report.html'}\nJSON: {directory / 'report.json'}")
    print(
        "Database, configs, checkpoints and existing reports unchanged. Only new reports written."
    )
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        raise SystemExit(2)
