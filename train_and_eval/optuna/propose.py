"""Public Optuna APIs only; proposal-domain rejection never fabricates outcomes."""

import math
import warnings
from collections import Counter
from copy import deepcopy

from .history import canonical


def fit_space(requested, history):
    space = deepcopy(requested)
    for row in history:
        for name, value in row["params"].items():
            if name not in space:
                if not name.startswith("width_"):
                    raise ValueError("Unknown historical axis " + name)
                space[name] = {
                    "type": "int",
                    "low": value,
                    "high": value,
                    "step": 1,
                    "log": False,
                }
            spec = space[name]
            if spec["log"] and value <= 0:
                raise ValueError(
                    f"Historical {name}={value} cannot use log scale; choose linear bounds to retain it"
                )
            spec["low"] = min(spec["low"], value)
            spec["high"] = max(spec["high"], value)
            if spec["type"] == "int":
                spec["step"] = 1
    return space


def in_domain(params, space):
    for name, value in params.items():
        if name not in space:
            return False
        spec = space[name]
        if not spec["low"] <= value <= spec["high"]:
            return False
        if spec["type"] == "int" and (value - spec["low"]) % spec["step"]:
            return False
    return True


def objective_values(row, objective):
    m = row["metrics"]
    if objective == "pareto":
        return [m["agent_return"], m["agent_max_drawdown"]]
    return [m["agent_return" if objective == "return" else "balanced_score"]]


def distance(a, b, space):
    terms = []
    for key in set(a) | set(b):
        if key not in a or key not in b:
            terms.append(1.0)
            continue
        spec = space[key]
        f = math.log if spec["log"] else float
        lo, hi = f(spec["low"]), f(spec["high"])
        terms.append(0.0 if hi == lo else ((f(a[key]) - f(b[key])) / (hi - lo)) ** 2)
    return math.sqrt(sum(terms) / len(terms))


def suggest(cfg, selected, all_rows):
    import optuna
    from optuna.distributions import FloatDistribution, IntDistribution
    from optuna.trial import TrialState, create_trial

    if not selected:
        raise ValueError("No usable observations")
    requested = cfg["search_space"]
    fit = fit_space(requested, selected)
    distributions = {}
    for name, s in fit.items():
        cls = IntDistribution if s["type"] == "int" else FloatDistribution
        distributions[name] = cls(
            s["low"],
            s["high"],
            log=s["log"],
            **({"step": s["step"]} if s["type"] == "int" else {}),
        )
    fixed = {n: s["low"] for n, s in requested.items() if s["low"] == s["high"]}
    notices = []
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", optuna.exceptions.ExperimentalWarning)
        settings = {k: v for k, v in cfg["sampler"].items() if k != "name"}
        sampler = optuna.samplers.TPESampler(**settings, constant_liar=True)
        if fixed:

            class FixedDomainSampler(optuna.samplers.PartialFixedSampler):
                # Grouped TPE may return fixed axes from its internal decomposed
                # space. Keep the public relative-sampling contract: only return
                # names actually requested; fixed axes use sample_independent.
                def sample_relative(self, study, trial, search_space):
                    sampled = super().sample_relative(study, trial, search_space)
                    return {k: v for k, v in sampled.items() if k in search_space}

            sampler = FixedDomainSampler(fixed, sampler)
        study = optuna.create_study(
            directions=["maximize"]
            * (2 if cfg["objective"]["name"] == "pareto" else 1),
            sampler=sampler,
        )
        for r in selected:
            study.add_trial(
                create_trial(
                    params=r["params"],
                    distributions={n: distributions[n] for n in r["params"]},
                    values=objective_values(r, cfg["objective"]["name"]),
                    user_attrs={"source_run_id": r["id"]},
                )
            )
        known = {canonical(r["params"]) for r in all_rows}
        seen, proposals, rejected = set(), [], Counter()
        attempts = 0
        for _ in range(cfg["suggestions"]["max_sampling_attempts"]):
            attempts += 1
            trial = study.ask()

            def draw(name, trial=trial):
                s = fit[name]
                if s["type"] == "int":
                    return trial.suggest_int(
                        name, s["low"], s["high"], step=s["step"], log=s["log"]
                    )
                return trial.suggest_float(name, s["low"], s["high"], log=s["log"])

            params = {"depth": draw("depth")}
            for i in range(1, params["depth"] + 1):
                params[f"width_{i}"] = draw(f"width_{i}")
            for name in (
                "learning_rate",
                "ent_coef",
                "gamma",
                "gae_lambda",
                "n_epochs",
                "data_epochs",
            ):
                params[name] = draw(name)
            key = canonical(params)
            reason = (
                "outside_requested_domain"
                if not in_domain(params, requested)
                else "already_evaluated"
                if cfg["suggestions"]["exclude_evaluated"] and key in known
                else "duplicate_proposal"
                if key in seen
                else None
            )
            if reason:
                rejected[reason] += 1
                study.tell(trial, state=TrialState.FAIL)
                if all(n in fixed for n in params):
                    break
                continue
            seen.add(key)
            nearest = sorted(
                all_rows, key=lambda r: (distance(params, r["params"], fit), r["id"])
            )[: cfg["report"]["nearest_observations"]]
            proposals.append(
                {
                    "params": params,
                    "already_evaluated": key in known,
                    "outside_observed_ranges": [
                        n
                        for n, v in params.items()
                        if not any(n in r["params"] for r in selected)
                        or v < min(r["params"][n] for r in selected if n in r["params"])
                        or v > max(r["params"][n] for r in selected if n in r["params"])
                    ],
                    "nearest": [
                        {
                            "run_id": r["id"],
                            "distance": distance(params, r["params"], fit),
                            "params": r["params"],
                            "metrics": r["metrics"],
                        }
                        for r in nearest
                    ],
                }
            )
            # Leave accepted proposals RUNNING for constant-liar batch diversity.
            if len(proposals) == cfg["suggestions"]["count"]:
                break
        notices = sorted({str(w.message) for w in captured})
    return {
        "proposals": proposals,
        "attempts": attempts,
        "rejected": dict(rejected),
        "requested_count": cfg["suggestions"]["count"],
        "fit_space": fit,
        "fit_space_expanded": fit != requested,
        "warnings": notices,
        "startup_random": len(selected) < cfg["sampler"]["n_startup_trials"],
        "optuna_version": optuna.__version__,
    }
