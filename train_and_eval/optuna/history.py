"""Snapshot loading and conservative scientific-history validation. No model imports."""

import hashlib
import json
import math
from collections import Counter, defaultdict
from copy import deepcopy

AXES = ("learning_rate", "ent_coef", "gamma", "gae_lambda", "n_epochs")
METRICS = ("agent_return", "balanced_score", "agent_max_drawdown")


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str, allow_nan=False
    )


def digest(value):
    # The digest also covers rejected source rows, which may contain NaN/Inf.
    # These JSON tokens are used only in hash input, never in report JSON.
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str, allow_nan=True
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def snapshot():
    from sqlalchemy import text

    from train_and_eval.database.session import create_database_engine

    engine = create_database_engine()
    try:
        with engine.connect() as con, con.begin():
            con.execute(
                text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
            )
            con.execute(text("SET LOCAL statement_timeout = '120s'"))
            runs = list(con.execute(text("SELECT * FROM runs ORDER BY id")).mappings())
            cps = list(
                con.execute(text("SELECT * FROM checkpoints ORDER BY id")).mappings()
            )
            evs = list(
                con.execute(
                    text(
                        "SELECT * FROM evaluations WHERE trigger='final' AND data_scope IN ('run_training','run_validation') ORDER BY id"
                    )
                ).mappings()
            )
            stamp = str(con.scalar(text("SELECT transaction_timestamp()")))
            return (
                [dict(x) for x in runs],
                [dict(x) for x in cps],
                [dict(x) for x in evs],
                stamp,
            )
    finally:
        engine.dispose()


def assemble(runs, checkpoints, evaluations):
    by_id = {r["id"]: r for r in runs}
    cps = {c["id"]: c for c in checkpoints}
    finals, evs = defaultdict(list), defaultdict(list)
    for c in checkpoints:
        if c["save_reason"] == "final":
            finals[c["run_id"]].append(c)
    for e in evaluations:
        if e["status"] == "completed":
            evs[(e["checkpoint_id"], e["data_scope"])].append(e)
    valid, rejected, visiting, signatures = {}, {}, set(), {}

    def visit(rid):
        if rid in valid:
            return valid[rid]
        if rid in rejected:
            raise ValueError("Ancestor excluded: " + rejected[rid])
        if rid in visiting:
            raise ValueError("Ancestry cycle")
        visiting.add(rid)
        try:
            r = by_id[rid]
            c = r["normalized_config_json"]
            c = json.loads(c) if isinstance(c, str) else c
            if r["seed"] != 1 or c["run"]["seed"] != 1:
                raise ValueError("Not training seed 1")
            if r["status"] != "completed":
                raise ValueError("Run not completed")
            if r.get("cycle_id") is not None:
                raise ValueError("Walk-forward run")
            if r.get("stopped_early"):
                raise ValueError("Early-stopped run")
            steps = int(r["training_steps_completed"])
            unit = int(r["steps_per_data_epoch"])
            if unit <= 0 or steps <= 0 or steps != r["training_steps_requested"]:
                raise ValueError("Incomplete/invalid training duration")
            if len(finals[rid]) != 1:
                raise ValueError("Missing/ambiguous final checkpoint")
            cp = finals[rid][0]
            if cp["run_step"] != steps:
                raise ValueError("Final checkpoint step mismatch")
            if c["run"]["name"] != r["name"]:
                raise ValueError("Saved run identity mismatch")
            p = {k: c["ppo"][k] for k in AXES}
            if type(p["n_epochs"]) is not int or p["n_epochs"] < 1:
                raise ValueError("Invalid PPO epoch count")
            if (
                p["learning_rate"] <= 0
                or p["ent_coef"] < 0
                or not 0 < p["gamma"] <= 1
                or not 0 < p["gae_lambda"] <= 1
            ):
                raise ValueError("Invalid scientific parameter range")
            p = {k: (v if k == "n_epochs" else float(v)) for k, v in p.items()}
            widths = c["ppo"]["hidden_sizes"]
            if not widths or any(type(w) is not int or w <= 0 for w in widths):
                raise ValueError("Invalid architecture")
            p.update(
                depth=len(widths), **{f"width_{i + 1}": w for i, w in enumerate(widths)}
            )
            if any(not math.isfinite(float(v)) for v in p.values()):
                raise ValueError("Nonfinite parameters")
            fixed = {
                k: deepcopy(c.get(k))
                for k in ("data", "environment", "ppo", "evaluation", "training")
            }
            for k in AXES + ("hidden_sizes",):
                fixed["ppo"].pop(k, None)
            scopes = {}
            for scope in ("run_training", "run_validation"):
                es = evs[(cp["id"], scope)]
                if len(es) != 1:
                    raise ValueError("Missing/ambiguous completed final " + scope)
                e = es[0]
                if e["steps_completed"] != e["steps_expected"]:
                    raise ValueError("Incomplete evaluation")
                if any(
                    e.get(k) is None or not math.isfinite(float(e[k])) for k in METRICS
                ):
                    raise ValueError("Missing/nonfinite final metric")
                if float(e["agent_max_drawdown"]) > 0:
                    raise ValueError("Expected signed nonpositive drawdown")
                scopes[scope] = e
            sig = {
                "fixed": fixed,
                "data": r["data_sha256"],
                "steps_per_epoch": unit,
                "windows": {
                    k: r.get(k)
                    for k in ("train_rows", "validation_rows", "split_index")
                },
                "evaluations": {
                    s: {
                        k: e.get(k)
                        for k in (
                            "data_sha256",
                            "evaluation_start_index",
                            "evaluation_end_index",
                            "evaluation_start_at",
                            "evaluation_end_at",
                            "lookback_rows",
                            "policy_mode",
                            "threshold_action",
                            "probability_threshold",
                            "seed",
                        )
                    }
                    for s, e in scopes.items()
                },
            }
            cohort = digest(sig)[:16]
            signatures[cohort] = sig
            source = r.get("source_checkpoint_id")
            path, cumulative = [], steps
            if source is not None:
                parent_cp = cps.get(source)
                if not parent_cp or parent_cp["save_reason"] != "final":
                    raise ValueError("Resume must use final parent checkpoint")
                parent = visit(parent_cp["run_id"])
                if (
                    c["continuation"]["mode"] != "resume"
                    or c["continuation"]["source_run"] != parent["name"]
                ):
                    raise ValueError("Saved continuation identity mismatch")
                if (
                    parent["cohort"] != cohort
                    or {k: v for k, v in parent["params"].items() if k != "data_epochs"}
                    != p
                ):
                    raise ValueError("Scientific settings change along path")
                path = parent["path"][:]
                cumulative += parent["cumulative_steps"]
            elif c["continuation"]["mode"] != "fresh":
                raise ValueError("Resume without source checkpoint")
            if cumulative % unit:
                raise ValueError("Noninteger cumulative data epochs")
            if cp["model_step"] != cumulative:
                raise ValueError("Cumulative checkpoint steps mismatch")
            p["data_epochs"] = cumulative // unit
            row = {
                "id": rid,
                "name": r["name"],
                "cohort": cohort,
                "params": p,
                "path": path + [rid],
                "cumulative_steps": cumulative,
                "metrics": {k: float(scopes["run_validation"][k]) for k in METRICS},
                "train_metrics": {k: float(scopes["run_training"][k]) for k in METRICS},
            }
            valid[rid] = row
            return row
        except (KeyError, ValueError, TypeError, OverflowError) as exc:
            rejected[rid] = str(exc)
            raise ValueError(str(exc)) from exc
        finally:
            visiting.discard(rid)

    for r in runs:
        try:
            visit(r["id"])
        except ValueError:
            pass
    return sorted(valid.values(), key=lambda r: r["id"]), {
        "inventory_runs": len(runs),
        "eligible_runs": len(valid),
        "exclusions": [{"id": k, "reason": v} for k, v in sorted(rejected.items())],
        "exclusion_counts": dict(Counter(rejected.values())),
        "cohorts": [
            {
                "id": k,
                "runs": sum(r["cohort"] == k for r in valid.values()),
                "signature": v,
            }
            for k, v in sorted(signatures.items())
            if any(r["cohort"] == k for r in valid.values())
        ],
    }


def select(rows, mode, best_by):
    """Maximal ancestry paths; shared prefixes are observations only once."""
    ids = {r["id"]: r for r in rows}
    ancestors = {i for r in rows for i in r["path"][:-1]}
    leaves = [r for r in rows if r["id"] not in ancestors]
    if mode == "all":
        chosen = rows
    elif mode == "last":
        chosen = leaves
    else:
        metric = "agent_return" if best_by == "return" else "balanced_score"
        chosen = [
            min(
                (ids[i] for i in leaf["path"]),
                key=lambda r: (
                    -r["metrics"][metric],
                    r["params"]["data_epochs"],
                    r["id"],
                ),
            )
            for leaf in leaves
        ]
    chosen = list({r["id"]: r for r in chosen}.values())
    # Reject conflicting replicas in ALL eligible history, even if not selected.
    seen, aliases = {}, {}
    for row in rows:
        key = canonical(row["params"])
        if key in seen and seen[key]["metrics"] != row["metrics"]:
            raise ValueError(
                f"Conflicting same-parameter observations: #{seen[key]['id']} and #{row['id']}"
            )
        if key not in seen:
            seen[key] = row
        aliases[row["id"]] = seen[key]["id"]
    selected = {}
    for row in chosen:
        selected[canonical(row["params"])] = seen[canonical(row["params"])]
    return sorted(selected.values(), key=lambda r: r["id"]), leaves, aliases
