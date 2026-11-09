# Optuna propositions

Use completed stage-one results to propose the next experiment. This tool reads
PostgreSQL, reconstructs an in-memory Optuna study, prints parameter suggestions,
and writes a standalone HTML report plus JSON. Every invocation starts with the
current history. It never starts training, creates training configs or queues,
loads models, changes hardware, or writes to the database.

## Quick start

From the PPO_AGENT project root, with the project virtual environment active:

```bash
python -m pip install 'optuna==4.5.0'
mkdir -p configs/optuna
cp -n train_and_eval/optuna/example.yml configs/optuna/propositions.yml
python -m train_and_eval.optuna --config configs/optuna/propositions.yml --validate-config
python -m train_and_eval.optuna --config configs/optuna/propositions.yml
```

The optional package extra is `.[optuna]`. Installing only the pinned Optuna
package as above avoids asking pip to resolve the entire PPO training stack.
The tracked `example.yml` is a reusable template. Its working copy lives under
the already ignored `configs/` tree; `cp -n` preserves any existing local config. It is not a generated training config. Run the tool as a module;
do not execute files inside this package directly.

Reports are created in a new directory on each invocation:

```text
artifacts/optuna_propositions/<UTC timestamp>_<unique suffix>/
    report.html
    report.json
```

Open `report.html` locally. It has Proposals, History, Sampling, Paths,
Configuration, and Provenance tabs and needs no network connection. The JSON
contains the same analysis with exact numeric parameters for reuse. Existing
reports are never removed or overwritten.

## Clean Git requirement

Both analysis and `--validate-config` require a clean project Git working tree.
The check runs before loading the YAML or reading the database. Staged changes,
uncommitted modifications, deletions and nonignored untracked files block the
command. An unavailable Git executable or failed Git check also blocks it.
`--help` remains available. There is no bypass flag.

Ignored files, including local `configs/` and reports, do not make Git dirty.
You can edit the working search-space YAML between analyses without committing
it. The exact effective configuration and its digest are saved in each report;
the report also records the checked Git commit. This is a startup check, not a
lock against concurrent source edits: avoid changing code during analysis.
Commit the installed changes before running the tool.

## History and comparability

Only **training seed 1** is accepted. It is a history filter, not a search axis.
`sampler.seed` controls Optuna's randomness and is unrelated to training seeds.

The tool reads runs, checkpoints and final evaluations inside one PostgreSQL
`REPEATABLE READ, READ ONLY` transaction. It does not read training metric rows,
checkpoint files or market data. A completion after the snapshot begins is
picked up on the next invocation.

A usable run must be completed, outside walk-forward cycles, not early stopped,
with its requested steps completed, one matching final checkpoint and one
completed final TRAIN and VAL evaluation. Core metrics must be finite and
signed max drawdown must be nonpositive. Ancestry must lead to a fresh run,
use final parent checkpoints, and keep scientific parameters unchanged.
Cumulative checkpoint counters must reconcile with the sum of local steps.
Noninteger cumulative data epochs are excluded. Invalid ancestors exclude their
descendants. The report lists every exclusion and its reason.

A cohort holds fixed the dataset hash, data configuration and windows,
environment, evaluation settings, devices, per-stage training settings, and PPO
settings outside the search axes. Different cohorts are **never pooled**.
With exactly one cohort, selection is automatic. Otherwise the first invocation
writes a report without suggestions. Copy the appropriate reported ID into:

```yaml
history:
  cohort: "<reported ID>"
```

Cohort signatures are included for inspection. Stored device configuration is
used; this tool does not independently verify which hardware executed a run.
Scientific fields absent from old configs are not silently filled with current
defaults. If necessary, historical differences require review.

Configs shared by multiple grids do not duplicate a DB run. Separate DB runs
with exactly the same parameter vector and cumulative duration are deduplicated
only if all three VAL objective metrics are identical. Conflicting results stop
suggestions and identify the run IDs. This checks all eligible observations in
the selected cohort, even if `last` or `best` would omit one.

## Three history modes

| `history.mode` | Imported observations |
| --- | --- |
| `all` | Every eligible run, with its actual cumulative `data_epochs`. |
| `last` | Last completed point of each maximal ancestry path. |
| `best` | Best point of each path according to `history.best_by`. |

`best_by` is `return` or `balanced_score`; it also resolves the historical point
when the optimization objective is Pareto. Ties prefer shorter duration, then
lower run ID. Branches are separate maximal paths, but shared observations are
imported only once. Ordering is based on actual checkpoint ancestry, never run
names, creation times or lexicographic version labels.

`all` gives longer paths more observations. Those points are correlated, not
independent repetitions. `best` deliberately introduces historical peak-selection
bias. `last` compares differing budgets when paths have different lengths.
In every mode the cumulative training duration is retained as a parameter.
Repeated validation use and a single seed do not establish out-of-sample quality.

## Objectives

```yaml
objective:
  name: return
  scope: val
```

| Name | Optuna objective |
| --- | --- |
| `return` | Maximize saved final VAL `agent_return`. |
| `balanced_score` | Maximize saved final VAL `balanced_score`. |
| `pareto` | Maximize VAL return and signed max drawdown simultaneously. |

A drawdown of `-0.10` is better than `-0.30`. No absolute-value inversion or
recalculation of the saved score is performed. Units are the stored units, not
rounded displayed percentages. Pareto proposals are multi-objective TPE samples;
the tool does not promise a particular trade-off or a Pareto improvement.

## Search space

All seven base axes are required: `depth`, `learning_rate`, `ent_coef`, `gamma`,
`gae_lambda`, `n_epochs`, `data_epochs`, plus `width_1` through the maximum depth.
Unknown keys, duplicate YAML keys, incompatible types and invalid bounds fail
validation. The example includes every axis.

| Axis | Type / typical scale | Meaning |
| --- | --- | --- |
| `depth` | int, linear | Number of hidden layers. |
| `width_1`, `width_2`, ... | int, linear | Neurons in each layer, including unequal widths. |
| `learning_rate` | float, log | The shared PPO learning rate; no actor/critic split. |
| `ent_coef` | float, log for strictly positive values | Entropy coefficient. |
| `gamma` | float, linear | Discount factor in `(0, 1]`. |
| `gae_lambda` | float, linear | GAE coefficient in `(0, 1]`. |
| `n_epochs` | int, linear | PPO optimizer epochs per rollout. |
| `data_epochs` | int, linear | Cumulative data passes, not continuation index. |

Widths beyond a proposal's depth are inactive and absent from its parameter
vector. Historical equal-width architectures are imported as their individual
layer widths; unequal widths can be proposed. This representation does not
encode a preference for equal layers or guarantee generalization to new shapes.
All parameters except cumulative duration describe a **constant configuration
from the fresh run through the end of the path**. A suggestion is not an
instruction to switch hyperparameters partway through an existing chain.

Set `low == high` to fix any axis. To force two layers of width 128:

```yaml
search_space:
  depth: {type: int, low: 2, high: 2, step: 1}
  width_1: {type: int, low: 128, high: 128, step: 1}
  width_2: {type: int, low: 128, high: 128, step: 1}
  # Include the other six base axes; omit width_3 and later widths.
```

Integer ranges accept a positive `step`; the upper endpoint must lie on that
grid. Log integers require step 1. Float axes are continuous and do not accept a
step. Zero entropy is allowed with `log: false`; a log scale cannot include zero.
The example bounds allow interpolation and extrapolation and are editable
starting values, not evidence that every allowed setting is worth training.

### Historical values outside requested bounds

History selection and the proposal domain are separate. The fitting distributions
cover both the imported historical values and requested bounds. Historical
values are never clipped, rounded onto a new grid, or discarded merely for being
outside the requested bounds. Integer fitting uses step 1 to retain old points.
Extra historical layers are retained even when the requested maximum depth is
smaller. Candidate suggestions are then checked against the requested domain
and its step sizes.

Fixed parameters use `PartialFixedSampler`. The wrapper filters grouped TPE's
relative outputs to the requested relative-search keys, preserving the public
sampler contract. Fixed coordinates are imposed on generated proposals; this
is **not conditional inference restricted to the fixed architecture**. A narrow
requested domain can reject many candidates. Increase the attempt budget or
review the domain if no candidate is returned. The report shows the actual
fitting bounds and rejection counts.

If a historical value is zero and the corresponding requested scale is log,
the analysis stops with an explanation; use a linear scale to retain that data.
No arbitrary epsilon is substituted.

## Sampler and suggestion settings

- `name: tpe`: Optuna Tree-structured Parzen Estimator (TPE).
- `seed: 42`: reproducible proposal sampling for identical ordered inputs,
  settings, package versions and numerical environment.
- `multivariate: true`: model parameter relationships jointly where supported.
- `group: true`: decompose the conditional search space, including layer widths.
  Requires multivariate mode. These Optuna options are experimental; warnings
  and the installed Optuna version are recorded.
- `n_startup_trials: 10`: imported completed observations count toward this
  threshold. Below it, sampling is random and the report explicitly says so.
- `n_ei_candidates: 64`: internal TPE candidate pool size; not training runs.
- `count: 1`: number of distinct proposals requested.
- `exclude_evaluated: true`: reject parameter vectors already evaluated at the
  same cumulative duration, checking all eligible rows in the chosen cohort.
- `max_sampling_attempts: 100`: total outer proposal attempts for the whole
  invocation, including rejected duplicates and out-of-domain samples. It is
  not a training budget or the internal candidate pool size.

Rejected proposals are marked FAIL without invented objective values. Accepted
proposals remain RUNNING in the ephemeral study so `constant_liar=True` can
encourage batch diversity. No new outcome is fed back until real training is
performed and a future invocation imports the new database results.

Duplicate comparison uses exact numeric values, with no rounding to run-name
precision. Very close continuous values are distinct candidates. A fixed fully
known point is detected after one attempt; the tool does not loop needlessly.
Exit code 0 means the requested proposal count was produced; code 2 means
invalid input, blocked history or a partial/exhausted search. Valid partial
proposals remain in the saved report.

## Reading the report

- **Proposals:** exact parameter values, marginal extrapolation flags, nearby
  completed observations with their actual outcomes.
- **History:** cohort signatures, exclusions, selected observation list and
  duplicate aliases. Inspect this before interpreting suggestions.
- **Sampling:** requested count, attempts, rejection reasons, actual fitting
  bounds, startup status, package version and warnings.
- **Paths:** full per-stage summaries when `include_full_paths: true`.
  Turning this off does not reduce the history used by Optuna.
- **Configuration / Provenance:** effective YAML settings, tool/Python versions,
  snapshot time and SHA-256 digests of the snapshot and configuration.

`nearest_observations` limits only displayed neighbors. Their distance is the
root-mean-square normalized axis distance (log coordinates for log axes), with
penalty 1 for an inactive/missing layer. It is descriptive and is not a TPE
acquisition score, influence estimate or explanation of a specific decision.
Marginal-range flags do not describe how densely a joint neighborhood was tested.

Optuna does not provide a human reasoning trace, calibrated confidence or a
predicted return here. Reports expose the actual inputs, settings and selected
points instead. No predicted economic outcomes are invented.

## Development checks

```bash
python -m pytest -q tests/test_optuna_propositions.py
python -m pytest -q
```

The dedicated tests use synthetic DB rows, a mocked read-only connection and
real in-memory Optuna sampling. No live database, training, Torch or GPU is
needed for those tests. They are skipped when the optional Optuna dependency is
not installed. Existing training code, schemas and migrations are unchanged.
