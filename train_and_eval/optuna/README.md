# Optuna propositions

Use completed stage-one results to propose the next experiment. This tool reads
PostgreSQL, reconstructs an in-memory Optuna study, prints parameter suggestions,
and writes a standalone HTML report plus JSON. Every invocation starts with the
current history. After saving the report, it can optionally prepare training configs
and one queue after explicit terminal confirmation. It never starts training,
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
artifacts/optuna_propositions/<UTC timestamp>/
    report.html
    report.json
```

New directory names contain UTC date/time with microseconds, for example
`20261111T123456123456Z`, without a random suffix. The same identifier is reused
for configs and queues. Existing suffixed directories remain unchanged and
accepted for compatibility. If a timestamp directory already exists, creation
fails instead of overwriting it.

Open `report.html` locally. It has Proposals, History, Sampling, Paths,
Preparation, Configuration, and Provenance tabs and needs no network connection. The JSON
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
checkpoint files or market data during analysis. Optional preparation additionally
reads and hashes reused parent checkpoints. A completion after the snapshot begins is
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

Giving `width_1` and `width_2` identical ranges does **not** make them equal:
they remain separate sampled parameters. Fixed equal values above enforce one
specific uniform architecture. Searching a common variable width across all
layers requires an additional parameterization; it is not a supported YAML mode.


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
- `count: 1`: number of distinct proposals requested (maximum 999).
- `exclude_evaluated: true`: reject parameter vectors already evaluated at the
  same cumulative duration, checking all eligible rows in the chosen cohort.
- `max_sampling_attempts: 100`: total outer proposal attempts for the whole
  invocation, including rejected duplicates and out-of-domain samples. It is
  not a training budget or the internal candidate pool size.

Rejected proposals are marked FAIL without invented objective values. Accepted
proposals remain RUNNING in the ephemeral study so `constant_liar=True` can
encourage batch diversity. No new outcome is fed back until real training is
performed and a future invocation imports the new database results.

Sampling duplicate comparison uses exact numeric values, with no rounding to
run-name precision. Very close continuous values are distinct candidates. Optional
preparation rounds values and performs a second duplicate/ancestry check. A fixed fully
known point is detected after one attempt; the tool does not loop needlessly.
Exit code 0 means the requested proposal count was produced; code 2 means
invalid input, blocked history or a partial/exhausted search. Valid partial
proposals remain in the saved report.

## Batch size and training budget

`count` counts propositions, not runs. Before reuse, the stage budget is the sum
of proposed `data_epochs`; existing and shared prefixes can reduce new work.
A batch of 100 five-epoch propositions can require 500 new runs. Use the printed
preparation plan for the actual number; runtime depends on architecture, PPO
epochs, evaluation work and dependency scheduling.

Batch proposals use the same imported history. They do not learn from one
another's future results. Run the tool again after training to import real new
outcomes. Deleting trial results from the database also removes that feedback;
`exclude_evaluated` cannot remember deleted experiments from old reports.

`best_by` only chooses history points in `best` mode; it does not change the
objective. For example, selecting peaks by return while optimizing balanced score
is allowed, but differs from selecting peaks by balanced score.

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
python -m pytest -q tests/test_optuna_propositions.py tests/test_optuna_git_guard.py tests/test_optuna_preparation.py tests/test_optuna_timestamp.py
python -m pytest -q
```

The dedicated tests use synthetic DB rows, a mocked read-only connection and
real in-memory Optuna sampling. No live database, training, Torch or GPU is
needed for those tests. They are skipped when the optional Optuna dependency is
not installed. Existing training code, schemas and migrations are unchanged.


## Optional config and queue preparation

Run the usual command from the project root:

```bash
python -m train_and_eval.optuna --config configs/optuna/propositions.yml
```

After the analysis and report, a terminal prompt accepts `PREPARE`. The tool
then acquires the existing exclusive activity/maintenance guards, checks for
pending cleanup/rename recovery, checks the Git commit again and reads a **new
read-only database snapshot**. No training or cleanup can run concurrently through
those cooperating guards. Do not manually edit configs, checkpoints or the DB
while preparing files.

It prints effective rounded parameters, reused stages, shared stages and new
stage counts. For a proposal with no new stages, type **`SKIP`** to continue with
other proposals. Any other answer cancels the entire preparation before config
writes. After preflight, **`CREATE`** publishes and verifies the reviewed plan.
EOF/Ctrl-C during input cancels. No implicit yes or automatic training exists.

For analysis without any preparation prompt:

```bash
python -m train_and_eval.optuna --config configs/optuna/propositions.yml --report-only
```

Noninteractive input also remains report-only; piping confirmation text does not
create configs. `--validate-config` continues to validate only the search YAML.
The tracked `example.yml` and your working search YAML are not modified.

### Paths and order

One identifier, taken directly from the newly created report directory, is reused
for both config trees:

```text
configs/stage_one/optuna_propositions/<report-directory-name>/
    preparation.json
    prop_001/
        README.md
        f_v00/<run-name>.yml
        r_v01/<run-name>.yml
    prop_002/
        README.md
        r_v03/<run-name>.yml
        r_v04/<run-name>.yml

configs/search_queues/optuna_propositions/<same-report-directory-name>/
    optuna_<report-directory-name>.yml
```

A proposition has only the stages it actually needs to create. Fully skipped
propositions have no directory; numbering retains the original report positions.
A proposition extending an existing three-epoch path to five epochs therefore
starts with `r_v03`, not `f_v00`.

The single queue lists the new owned stages of `prop_001` in ascending stage
order, followed by those of `prop_002`, and so on. Each run appears exactly once.
Shared new prefixes are owned by the earliest proposal that needs them. Later
proposals refer to them and create only additional stages. The existing
scheduler scans for ready jobs across the queue, so independent proposals can
use both workers even with proposal-major listing. A single dependent chain
cannot train two of its stages at once.

### Reuse and scientific identity

Reuse checks **all eligible completed runs in the selected cohort**, regardless
of `history.mode` and regardless of whether their names begin with `op_`.
The rounded PPO axes, exact architecture, seed, cumulative duration, fixed
scientific cohort and full ancestry must match. A completed prefix comes from a
single ancestry path; unrelated stages are never stitched together. Conflicting
replicas block preparation. A related unfinished or otherwise excluded run also
requires review instead of silently creating another run.

The first missing stage points to the actual existing parent run name and its
verified final checkpoint ID. New stages point to the previous new run with
`checkpoint: final`. Reused checkpoint files must exist, remain inside the
project and pass size/SHA-256 validation. Reused configs are not copied into the
new proposition directory. If every requested stage already exists (or is
already covered by earlier propositions), only `SKIP` permits proceeding.

Fixed settings are copied from the selected cohort's stored configuration. For
an extension, its parent supplies the template. The exporter requires one data
epoch per stage and validates with the existing `RunConfig` and resume rules.
It does not replace scientific settings with newly chosen defaults or alter
existing configs, run records, checkpoints, historical reports or ID sequences.
An already prepared local config with the same new run name blocks another
export; use/review its queue first. This avoids silently making competing plans.

### Effective parameters and names

LR, entropy, gamma and GAE are rounded to **six significant decimal digits** using
round-half-even. The original Optuna proposal remains unchanged in the report.
The Preparation tab and `preparation.json` store the effective values, changes,
source snapshot, reused run/checkpoint IDs, config hashes and created paths.
Only effective values are written to configs and encoded in their names.

Examples:

```text
op_lr2p25000em5_r_v01_gae8p50000em1_ent2p00000em4_g9p00000em1_w384x3_ne03_s01
op_lr2p25000em5_f_v00_gae8p50000em1_ent2p00000em4_g9p00000em1_w903w743w126w32_ne03_s01
```

Equal-width architectures use `w384x3`; unequal widths list every layer.
`op_` does not bypass validation. Names must fit the actual database/config limit
of **128 characters**, use one-digit exponent magnitudes (0..9), `ne01..ne99`,
and stages `v00..v99` (up to 100 cumulative data epochs). Unrepresentable
proposals remain in the analysis report, but preparation stops with an error.
No truncation, name hashing, or database migration is performed.

Rounding beyond requested search bounds is allowed and reported; it is not
clipped back into the domain. Technical PPO/schema bounds still apply. Rounding
that turns a proposal into a known completed path triggers the `SKIP` flow.
Different scientific configurations cannot silently share the same run name.

### Verification and failure behavior

Before writes: validate the whole plan, ignored destinations, names, local
collisions, all configs, ancestry and reused checkpoint hashes. Files are staged
outside queue discovery. Re-read YAML values and validate the dependency graph,
exact names, parameters, paths and counts. Publish the complete config tree
first and queue tree last, then repeat validation from the published paths.
Handled publication failures remove only the newly created batch trees.
Existing files are never overwritten.

An abrupt OS/power failure can leave an unqueued config tree or a complete queue;
publication is not a cross-filesystem database transaction. The batch timestamp
and provenance make this identifiable. Existing destinations block replacement;
review an interrupted batch before retrying. Filesystem/DB checks are not an
atomic system-wide snapshot against manual external changes.

The Preparation tab records `created_and_verified`, `cancelled`,
`nothing_to_create`, `not_requested`, or `blocked`. Preparation failure does not
remove the analysis report. The tool stops after file preparation. Run a queue
manually only when you choose to:

```bash
python run_managed_queue.py --queue optuna_<report-directory-name> --dry-run
python run_managed_queue.py --queue optuna_<report-directory-name>
```

For later cleanup, use the existing training cleanup tool for selected run IDs
and their owned artifacts, inspect descendant scope, then remove only the
corresponding local config/queue batch. Never delete a shared parent merely
because one proposition referenced it.

## Exported YAML field order

New run configs follow the established grid layout: `config_schema_version`,
`run`, `continuation`, `data`, `training`, `environment`, `ppo`, `evaluation`,
`logging`, and `artifacts`. Date ranges use `start` before `end`. Nested fields
follow the same grid convention. Additional fields are preserved after known
fields; explicit null values are retained. This only changes YAML presentation,
not configuration values, run identities, or training behavior. Existing configs,
reports, and database records are not rewritten.
