# PPO_AGENT

Research infrastructure for training, evaluating, comparing, and reproducing
PPO trading agents on historical market data.

The project is built around a strict experiment lifecycle:

- experiments are defined in versioned YAML files;
- market data is imported from a published pipeline release and verified against
  manifest v2 and SHA-256 hashes;
- training requires a clean Git worktree;
- every `run.name` is globally unique;
- PostgreSQL stores run identity, resolved configuration, metrics, checkpoints,
  and evaluation results;
- checkpoints are immutable and can be replayed without retraining;
- reports and dashboard views are derived from persisted experiment data.

The target research interval is **5 minutes**. Longer intervals such as
**1 hour** are primarily used for rapid iteration, infrastructure validation,
diagnostics, and proof-of-concept experiments. A result discovered on 1-hour
data is not considered confirmed until it is tested on the target 5-minute
setup and across multiple seeds.

> [!IMPORTANT]
> This repository is research software. Backtest results are not evidence of
> future profitability and should not be interpreted as financial advice.

## Contents

- [Core capabilities](#core-capabilities)
- [Architecture](#architecture)
- [Quick start](#quick-start)
- [Market data](#market-data)
- [Experiment configuration](#experiment-configuration)
- [Reproducibility contract](#reproducibility-contract)
- [Single-run training](#single-run-training)
- [Search queues](#search-queues)
- [Resume training](#resume-training)
- [Run registry](#run-registry)
- [Evaluation and checkpoint replay](#evaluation-and-checkpoint-replay)
- [Threshold sweeps](#threshold-sweeps)
- [Reports and artifacts](#reports-and-artifacts)
- [Dashboard](#dashboard)
- [Database model](#database-model)
- [Testing and development](#testing-and-development)
- [Recommended research workflow](#recommended-research-workflow)
- [Troubleshooting](#troubleshooting)

## Core capabilities

- YAML-configured fresh and resumed PPO runs
- chronological train/validation split
- exact, batch-aligned training duration
- periodic and final immutable checkpoints
- periodic and final validation
- early stopping based on validation score
- separate devices for training and evaluation
- full PostgreSQL experiment registry
- unique run-name preflight before training
- Git commit and branch tracking
- market-data identity and SHA-256 tracking
- persistent PPO, rollout, and critic diagnostics
- persisted evaluation metrics and trajectories
- deterministic, stochastic, and probability-threshold policies
- replay of arbitrary existing checkpoints
- probability-threshold sweeps without retraining
- rebuildable run and evaluation reports
- Streamlit experiment dashboard
- Pareto-front exploration
- YAML search-queue manifests with selective execution
- Alembic-managed schema migrations
- automated tests with pytest

## Architecture

```text
YAML config + verified market data + clean Git commit
                         |
                         v
                    PPO training
                         |
             +-----------+-----------+
             |                       |
             v                       v
      PostgreSQL registry      checkpoint files
             |                       |
             +-----------+-----------+
                         |
                         v
               evaluation / replay
                         |
             +-----------+-----------+
             |                       |
             v                       v
       persisted metrics      rebuildable artifacts
                                     |
                                     v
                            reports and dashboard
```

PostgreSQL is the source of truth for experiment history. Checkpoint files are
the immutable model source for historical replay. CSV, Parquet, JSON, and PNG
outputs are analysis artifacts and can be rebuilt when their source data still
exists.

## Project structure

```text
PPO_AGENT/
├── alembic/
│   └── versions/                 # PostgreSQL schema migrations
├── artifacts/
│   └── runs/                     # checkpoints and generated reports
├── configs/
│   ├── experiments/              # individual training configurations
│   └── search_queues/            # ordered queue manifests
├── data/                         # local market-data files
├── extra_tools/                  # cross-run diagnostic tools
├── tests/
├── train_and_eval/
│   ├── checkpoints/
│   ├── dashboard/
│   ├── database/
│   ├── environment/
│   ├── evaluation/
│   ├── market_data/
│   ├── ppo/
│   ├── reporting/
│   ├── runs/
│   └── training/
├── pyproject.toml
├── run_search_queue.py
└── threshold_sweep.py
```

Generated runtime artifacts are not intended to be committed to Git.

## Quick start

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install the project

Runtime dependencies:

```bash
python -m pip install -e .
```

Runtime and development dependencies:

```bash
python -m pip install -e ".[dev]"
```

### 3. Start PostgreSQL

```bash
docker compose up -d postgres
docker compose ps
```

The local compose setup may expose PostgreSQL on a non-default host port, for
example:

```text
127.0.0.1:5433 -> postgres:5432
```

### 4. Configure the database connection

Set `DATABASE_URL` in the environment or in the local `.env` file:

```env
DATABASE_URL=postgresql+psycopg://ppo_agent:<password>@127.0.0.1:5433/ppo_agent
```

Do not commit real credentials.

### 5. Apply migrations

```bash
alembic upgrade head
alembic current
```

### 6. Verify the installation

```bash
python -m pytest -q
```

### 7. Import and verify market data

Follow [Manual snapshot import](#manual-snapshot-import) to place the published
Parquet files under `data/` and the unchanged pipeline manifest at
`train_and_eval/market_data/manifest.json`.

```bash
python -m train_and_eval.market_data.validate_market_data --all
```

This command verifies the imported files without modifying data or manifest.
It requires manifest v2; it cannot generate a manifest from loose Parquet files.

### 8. Train one committed experiment

```bash
git status --short
python -m train_and_eval.training \
  --config configs/experiments/<group>/<experiment>.yml
```

### 9. Inspect registered runs

```bash
python -m train_and_eval.runs list
```

### 10. Start the dashboard

```bash
streamlit run train_and_eval/dashboard/app.py
```

## Market data

### Repository responsibilities

`NQ_HISTORICAL_DATA_PIPELINE` owns historical acquisition, source provenance,
normalization, merging, continuous-contract handling, calendar and gap analysis,
resampling, quality reports, and publication of Parquet files and manifest v2.

`PPO_AGENT` consumes that published snapshot, verifies its identity, and performs
the chronological split, training, evaluation, reporting, and experiment analysis.
It does not download or repair market data and does not generate or update the
producer's manifest.

### Manual snapshot import

Market-data Parquet files live under `data/`. The imported manifest lives at:

```text
train_and_eval/market_data/manifest.json
```

Import is a manual operation:

1. Preserve the previous snapshot and its manifest before replacing them.
   Keep the source pipeline release and any repair reports needed to explain
   how its input history was produced.
2. Copy the published Parquet files from the pipeline's `OUT/` into `data/`,
   preserving their filenames and any relative subdirectories.
3. Copy the pipeline's `OUT/manifest.json` unchanged to
   `train_and_eval/market_data/manifest.json`. Do not add `data/` prefixes inside
   the manifest or combine entries from different releases.
4. Keep the four pipeline reports with the archived source release. The
   PPO_AGENT loader does not require them locally and does not verify their
   contents or hashes during training.
5. Run the verification command below, update experiment `data.path` values to
   the new filenames, and use new `run.name` values for the new snapshot.
6. Review and commit tracked manifest, configuration, code, and documentation
   changes before training. Avoid training or replay during a snapshot import.

The producer stores file entries under interval keys (`1m`, `5m`, `15m`, `30m`,
`1h`). Each entry's `path` is relative to the imported data root. For example,
`files["5m"]["path"] = "NQ_CONTINUOUS_5m_<snapshot>.parquet"` maps to
`data/NQ_CONTINUOUS_5m_<snapshot>.parquet` in PPO_AGENT. The loader adapts paths
in memory while preserving the original manifest bytes.

### Manifest and file verification

The consumer requires `manifest_version: 2` and `data_schema_version: 1`.
Manifest v1 is no longer supported; changing its version number manually is
not a migration.

The manifest must describe a published dataset with zero structural errors.
Accepted status combinations are:

| Dataset status | Validation status | Warning acceptance |
|---|---|---|
| `published` | `passed` | No warnings; acceptance is `not_required` |
| `published_with_warnings` | `passed_with_warnings` | Warnings accepted through `interactive` or `cli_flag` mode |

Every file entry must have `validation_status: passed`. The loader checks
supported schema versions and intervals, normalized relative paths, unique
file paths, warning-count consistency, acceptance metadata, and required file
identity fields. Duplicate JSON keys and paths escaping the data directory
are rejected.

A training config is loaded with data verification enabled before a run is
created. Verification requires an exact manifest entry, matching file size,
and matching SHA-256. Loading also checks the Arrow schema, row count, first
and last timestamps, and rechecks SHA-256 after reading.

This prevents an experiment from silently using a changed file under an old
filename. Hash checks establish agreement with the imported manifest; they do
not independently establish the accuracy of the source market data.

### Verification commands

Verify every file referenced by the imported manifest:

```bash
python -m train_and_eval.market_data.validate_market_data --all
```

Verify one file, including local row-level checks:

```bash
python -m train_and_eval.market_data.validate_market_data \
  data/NQ_CONTINUOUS_5m_<snapshot>.parquet
```

Verify identity and load a file through the training loader:

```bash
python -m train_and_eval.market_data.load_market_data \
  data/NQ_CONTINUOUS_5m_<snapshot>.parquet
```

The validator checks schema, nulls, duplicate timestamps, ordering, finite
positive prices, OHLC relationships, interval alignment, UTC weekdays, and
nonempty symbols, in addition to manifest identity and file metadata.
Missing minutes and calendar expectations remain the pipeline's responsibility.

Both successful and failed validation leave the manifest and Parquet files
unchanged. `--all` checks manifest-listed files, reports missing files, and
does not register unrelated local Parquet files. Console results `okay` and
`not_okay` are local check outcomes, not new producer manifest entries.

### Data schema and training interface

Published Parquet column order and physical types remain:

| Parquet column | Arrow type | Loaded column |
|---|---|---|
| `Datetime` | `timestamp[ns, tz=UTC]` | `DT` |
| `Open_NQ` | `float64` | `Open` |
| `High_NQ` | `float64` | `High` |
| `Low_NQ` | `float64` | `Low` |
| `Close_NQ` | `float64` | `Close` |
| `Volume_NQ` | `uint64` | `Volume` |
| `symbol` | `large_string` | `symbol` |
| `instrument_id` | `uint32` | `instrument_id` |

The loader preserves `frame.attrs["source_path"]` in the local `data/...`
convention and `frame.attrs["sha256"]`. It also exposes `dataset_id`,
`release_id`, and `interval` in `frame.attrs`. These additional attributes do
not introduce new database columns. The chronological splitter copies the
attributes into both partitions.

### Chronological split

The split is chronological:

```text
oldest rows                                  newest rows
|---------------- TRAIN ----------------|--- VALIDATION ---|
                                    split_index
```

There is no random shuffle between train and validation data.

The environment uses historical rows before the scored range as observation
lookback only. At the end of an episode, `force_close_on_done: true` closes any
open position so the final equity includes the closing transaction.

### Data identity rules

- never overwrite a registered dataset and continue using its old hash;
- import the matching producer manifest unchanged when replacing a snapshot;
- make corrections in the data project and publish a new release instead of
  editing an imported manifest to approve changed bytes;
- use new run names for experiments on a new data snapshot;
- preserve the previous dataset, manifest, database, and artifacts for any
  historical results that must remain reproducible;
- replay checks the dataset SHA-256 recorded for the source run: a new snapshot
  cannot silently replace the old dataset for an existing checkpoint evaluation.

Historical replay of a manifest-v1 snapshot requires its archived compatible
code and environment, or an explicit, separately verified migration. Retaining
old files alone does not make manifest v1 readable by this loader.

## Experiment configuration

Individual experiments are stored under:

```text
configs/experiments/
```

A configuration contains these main sections:

```text
run
continuation
data
training
logging
environment
ppo
evaluation
artifacts
```

Representative structure:

```yaml
config_schema_version: 1

run:
  name: nq5m_example_v1_seed1
  seed: 1

continuation:
  mode: fresh

data:
  path: data/NQ_CONTINUOUS_5m_<snapshot>.parquet
  train_ratio: 0.9

training:
  duration_unit: data_epochs
  duration_amount: 1

logging:
  training_progress_every_steps: 120000
  validation_progress_every_steps: 24000

environment:
  window: 40
  context: baseline_multiscale_v1
  position_side: long_only
  market_timezone: America/New_York
  rth_open: 09:30
  rth_close: "16:00"
  stake_pln: 1000.0
  fee_bps: 1.0
  swap_bps: 3.0
  swap_time: "17:00"
  swap_timezone: America/New_York
  force_close_on_done: true
  reward_scale: 1.0
  exposure_penalty: 0.0
  turnover_penalty: 0.0
  drawdown_penalty: 0.0
  profit_reward_mult: 1.0
  loss_reward_mult: 1.0

ppo:
  policy: mlp
  device: cuda
  hidden_sizes: [384, 384, 384]
  activation: tanh
  value_head_init_scale: 0.003
  initial_long_probability: 0.55
  n_steps: 2048
  batch_size: 1024
  n_epochs: 3
  learning_rate: 0.0003
  gamma: 0.9
  gae_lambda: 0.85
  clip_range: 0.2
  clip_range_vf: null
  normalize_advantage: true
  ent_coef: 0.0002
  vf_coef: 0.5
  max_grad_norm: 0.5
  target_kl: null

evaluation:
  device: cpu
  eval_every_steps: 240000
  checkpoint_every_steps: 240000
  policy_mode: deterministic_argmax
  threshold_action: null
  probability_threshold: null
  best_metric: balanced_score
  early_stop_patience_evals: 5

artifacts:
  training_metrics:
    enabled: true
    every_steps: 2048
  validation_trajectory:
    mode: all
  plots:
    during_run: false
```

The original YAML, its hash, the normalized configuration, and the normalized
configuration hash are persisted with the run. Database comparisons therefore
use the effective experiment definition, not only a filesystem path.

### Training duration

Supported duration styles include a fixed number of timesteps and complete
data epochs. The training preflight resolves the requested duration into an
exact batch-aligned execution plan.

The terminal shows:

- original scored training steps;
- any oldest training steps trimmed for batch alignment;
- effective steps per data epoch;
- requested and resolved duration;
- resolved checkpoint and evaluation cadence;
- all scheduled event steps.

Stable-Baselines3 normally collects fixed-size rollouts. The project preserves
full rollouts and uses a smaller final rollout buffer when required, so the
resolved training duration is executed exactly.

### Training and evaluation devices

The devices are configured independently:

```yaml
ppo:
  device: cuda

evaluation:
  device: cpu
```

Accepted values are `auto`, `cpu`, and `cuda`.

## Reproducibility contract

### Clean Git is mandatory

Training and historical replay reject a dirty worktree:

```bash
git status --short
```

The output should be empty before a run. Commit intentional code, config,
migration, or test changes first.

### Run names are immutable identities

`run.name` is unique in PostgreSQL. Running an already registered config fails
before expensive training work begins:

```text
Training run: FAILED
Error type: RunAlreadyExistsError
Error: Run name '<name>' is already registered as run #<id> with status '<status>'.
```

To create a new experiment, use a new meaningful `run.name`. Do not delete an
existing run merely to reuse its name.

### Checkpoints are immutable

Each checkpoint row stores its path, SHA-256, size, local run step, and complete
model-history step. Historical replay verifies and uses the persisted
checkpoint rather than rebuilding model state from an informal filename.

## Single-run training

Run one config:

```bash
python -m train_and_eval.training \
  --config configs/experiments/<group>/<experiment>.yml
```

The normal terminal view includes:

1. training-resolution preflight;
2. resolved checkpoint/evaluation schedule;
3. run identity;
4. live training progress;
5. latest PPO diagnostics;
6. final validation metrics;
7. best and final checkpoint information.

A successful training execution may persist:

- one `runs` row;
- multiple `training_metrics` rows;
- periodic checkpoints and one final checkpoint;
- scheduled evaluations and one final evaluation;
- validation trajectories and report artifacts, depending on config.

Use `--traceback` when diagnosing an unexpected failure:

```bash
python -m train_and_eval.training \
  --config configs/experiments/<group>/<experiment>.yml \
  --traceback
```

## Search queues

Queues run committed experiment configs sequentially. Their definitions are
separate YAML manifests under:

```text
configs/search_queues/
```

Example manifest:

```yaml
queue_schema_version: 1
name: nq5m_candidate_confirmation_v1
configs:
  - configs/experiments/nq5m_candidate_confirmation_v1/00_fresh_seed1.yml
  - configs/experiments/nq5m_candidate_confirmation_v1/01_fresh_seed2.yml
  - configs/experiments/nq5m_candidate_confirmation_v1/02_fresh_seed3.yml
```

The manifest order defines the stable 1-based queue positions used by the CLI.
The queue name must match the manifest filename, and duplicate config paths are
rejected.

### Inspect a queue

```bash
python run_search_queue.py \
  --queue nq5m_candidate_confirmation_v1 \
  --dry-run
```

The table marks selected configs with `[x]` and prints the selected/total count.

### Run every config

```bash
python run_search_queue.py \
  --queue nq5m_candidate_confirmation_v1
```

### Run selected positions

```bash
python run_search_queue.py \
  --queue nq5m_candidate_confirmation_v1 \
  --select 1,3,5-8
```

Selections are executed in manifest order, even if the expression is written
in another order.

### Continue from a position

```bash
python run_search_queue.py \
  --queue nq5m_candidate_confirmation_v1 \
  --start-at 5
```

`--start-at` and `--select` are mutually exclusive.

### Handle already registered configs

By default, the queue checks PostgreSQL before starting any selected training.
If at least one selected `run.name` already exists, the queue lists all
conflicts and starts nothing.

To execute only selected configs that are still missing:

```bash
python run_search_queue.py \
  --queue nq5m_candidate_confirmation_v1 \
  --select 1-12 \
  --skip-existing
```

If every selected config already exists, no child process is started.

### Print an existing comparison only

```bash
python run_search_queue.py \
  --queue nq5m_candidate_confirmation_v1 \
  --summary-only
```

The summary is rebuilt from PostgreSQL and includes final metrics, best score,
latest PPO diagnostics, runtime, ranking, and winner information.

### Queue terminal behavior

Before each child run, the queue prints:

- selected-run progress such as `[2/5]`;
- original queue position;
- remaining selected runs;
- config path.

The child training process inherits the terminal directly. Its output is the
same live training interface as a normal single-run command. The queue does not
replace it with raw Stable-Baselines3 tables.

If one run fails, the queue stops instead of silently continuing with dependent
configs.

## Resume training

A resumed run starts from an immutable checkpoint selected from an earlier run.

```yaml
run:
  name: nq5m_example_v1_epoch2_lr3e4_seed1
  seed: 1

continuation:
  mode: resume
  source_run: nq5m_example_v1_epoch1_lr7p5e4_seed1
  checkpoint: best
```

Common checkpoint selectors are:

- `best` — checkpoint with the best persisted validation score;
- `final` — final checkpoint of the source run.

The new run must have its own unique name. Resume compatibility validation
protects against continuing with incompatible data, environment, model, or PPO
identity settings.

Checkpoint steps have two meanings:

| Field | Meaning |
|---|---|
| `run_step` | Local step within the run that created the checkpoint |
| `model_step` | Complete step count including resume history |

Lineage is persisted as:

```text
runs.source_checkpoint_id -> checkpoints.id
```

## Run registry

The registry CLI is read-only except for report regeneration.

### List recent runs

```bash
python -m train_and_eval.runs list
```

### Show one complete run

```bash
python -m train_and_eval.runs show --run-id 83
```

### List checkpoints

```bash
python -m train_and_eval.runs checkpoints --run-id 83
```

### List persisted evaluations

```bash
python -m train_and_eval.runs evaluations --run-id 83
```

This command displays existing evaluation rows. It does not replay a model.

### Rebuild reports

```bash
python -m train_and_eval.runs report --run-id 83
```

For exact available options:

```bash
python -m train_and_eval.runs --help
python -m train_and_eval.runs report --help
```

## Evaluation and checkpoint replay

Evaluations belong to checkpoints, not directly to runs:

```text
runs.id
   |
   +----< checkpoints.run_id
                    |
                    +----< evaluations.checkpoint_id
```

There is currently no general-purpose command such as:

```bash
python -m train_and_eval.evaluation
```

Use the run registry to inspect persisted evaluations, `runs report` to rebuild
derived artifacts, and `threshold_sweep.py` for evaluation-only replay under
different probability thresholds.

Supported policy modes are:

```text
deterministic_argmax
stochastic_sample
probability_threshold
```

### Deterministic argmax

For the binary LONG/FLAT policy, the action with the greater probability is
selected.

### Stochastic sample

The action is sampled from the policy distribution. The evaluation seed makes
the sampled sequence reproducible.

### Probability threshold

For `threshold_action: 1`, action `1` is LONG and the configured LONG
probability threshold determines whether the policy enters or remains in the
market.

The report service may replay immutable checkpoints to rebuild missing or
legacy derived artifacts. Historical replay does not insert a new evaluation
row and must reproduce persisted evaluation metrics within the service's
verification rules.

## Threshold sweeps

`threshold_sweep.py` replays existing checkpoints across multiple LONG
probability thresholds. It does not train a model and does not insert new
evaluation rows.

### Sweep final checkpoints selected by run ID

```bash
python threshold_sweep.py \
  --runs 80,81,82,83 \
  --thresholds 0,0.1,0.2,0.3,0.4,0.45,0.5,0.55,0.6,0.7 \
  --output /tmp/ppo_threshold_sweep.csv
```

`--runs` resolves the final checkpoint of every selected run.

### Sweep arbitrary checkpoint IDs

```bash
python threshold_sweep.py \
  --checkpoint-ids 338,342,348,355 \
  --thresholds 0,0.1,0.2,0.3,0.4,0.45,0.5,0.55,0.6,0.7 \
  --output /tmp/ppo_checkpoint_threshold_sweep.csv
```

`--runs` and `--checkpoint-ids` are mutually exclusive.

The terminal and CSV identify each source by:

- run ID and name;
- seed;
- checkpoint ID;
- local run step;
- complete model step;
- checkpoint save reason.

Reported metrics include exposure, flat fraction, round trips, return, maximum
drawdown, profit factor, win rate, and balanced score.

The CSV is written only after all replay operations finish. This allows replay
to pass the clean-Git verification without the output file making the worktree
dirty during the sweep.

For a binary LONG/FLAT policy, a threshold near `0.5` normally reproduces the
argmax boundary away from an exact probability tie. Use the actual replay
result rather than assuming equivalence at the tie boundary.

### Confidence curve versus threshold sweep

The confidence curve answers:

```text
For each threshold t, on the existing trajectory, how often was P(LONG) >= t?
```

The threshold sweep answers:

```text
What trading trajectory and metrics result when actions are replayed with t?
```

These are different because changing an action can change later environment
state, observations, probabilities, positions, costs, and equity.

## Reports and artifacts

### Rebuild a run report

```bash
python -m train_and_eval.runs report --run-id 83
```

The report pipeline reads persisted metadata and metrics, verifies source
artifacts, replays checkpoints when required, and generates current-format
derived files.

Typical run-level directory:

```text
artifacts/runs/00000083/reports/
```

Typical files:

```text
training_curves.png
validation_curves.png
training_metrics.csv
validation_metrics.csv
summary.json
```

Typical evaluation directory:

```text
artifacts/runs/00000083/evaluations/00000355/
```

Possible files include:

```text
trajectory.parquet
trade_events.parquet
metrics.json
equity_curve.png
drawdown_curve.png
market_and_exposure.png
cumulative_costs.png
trade_returns.png
holding_times.png
policy_p_long_distribution.png
policy_p_long_confidence_curve.png
```

Trade-specific outputs may be absent when the trajectory contains no required
events.

### Policy-probability columns

For a two-action LONG/FLAT policy, trajectories may store:

```text
policy_probability_action_0 = P(FLAT)
policy_probability_action_1 = P(LONG)
selected_action_probability
```

This supports analysis of policy saturation, polarization, and confidence
without inferring the unselected probability from action labels.

### Cross-run diagnostics

Cross-run tools live under `extra_tools/`. For example:

```bash
python -m extra_tools.policy_probability_diagnostic \
  --runs 34,36,37 \
  --output-dir /tmp/ppo_policy_probability_diag
```

Checkpoint-by-checkpoint probability evolution:

```bash
python -m extra_tools.policy_probability_evolution \
  --runs 34,36,37 \
  --output-dir /tmp/ppo_policy_probability_evolution
```

Use each tool's `--help` output as the authoritative option reference.

## Dashboard

Start the Streamlit dashboard with:

```bash
streamlit run train_and_eval/dashboard/app.py
```

Do not use:

```bash
python -m train_and_eval.dashboard
```

The dashboard package does not expose a `__main__` module.

The dashboard reads the experiment registry and joins runs, checkpoints,
evaluations, and training metrics into an analysis dataset. Sidebar filters
apply across tabs, so filter the visible experiment family or dataset before
interpreting a Pareto front.

Current analysis views include:

- Run Explorer
- Scatter Explorer
- Activity Map
- Pareto Explorer
- Group Comparison
- Run Detail

### Pareto Explorer

A common setup is:

```text
X metric    = Max drawdown
X objective = Maximize
Y metric    = Agent return
Y objective = Maximize
Color       = Seed
```

Maximum drawdown is stored as a negative return. Therefore `-15%` is better
than `-30%`, and maximizing it means moving closer to zero.

A visible run is Pareto-optimal when no other visible run is at least as good
in both selected objectives and strictly better in at least one. The front is
recomputed after dashboard filters are applied.

### Multi-seed interpretation

Group Comparison aggregates results across hyperparameter groups and seeds.
One exceptional seed is a candidate, not a robust conclusion. Prefer settings
whose behavior survives independent initializations.

## Database model

Main runtime tables:

```text
runs
checkpoints
training_metrics
evaluations
```

Relationships:

```text
runs.id
  |
  +----< training_metrics.run_id
  |
  +----< checkpoints.run_id
               |
               +----< evaluations.checkpoint_id

runs.source_checkpoint_id
  |
  +--------> checkpoints.id
```

### `runs`

One row represents one training execution. It stores, among other fields:

- unique name, status, and seed;
- fresh/resume mode and source checkpoint;
- Git commit and branch;
- raw and normalized configuration with hashes;
- data path and SHA-256;
- chronological split identity;
- requested/completed steps and data epochs;
- early-stop and failure information;
- lifecycle timestamps.

### `checkpoints`

One row represents one immutable model file. It stores:

- owning run;
- local `run_step` and history-aware `model_step`;
- save reason (`initial`, `periodic`, `final`, `manual`, or `interrupted`);
- relative artifact path;
- SHA-256 and file size;
- creation time.

The database enforces at most one final checkpoint per run.

### `training_metrics`

Training rows capture diagnostics such as:

- rollout and episode reward statistics;
- approximate KL divergence;
- clip fraction;
- entropy loss;
- explained variance;
- policy-gradient and value losses;
- learning rate;
- critic prediction, target, error, and correlation diagnostics where
  available.

### `evaluations`

An evaluation stores its checkpoint identity, trigger, status, policy mode,
data scope, progress, reproducibility metadata, and trading metrics.

Common metrics include:

- agent, always-long, and always-short returns;
- balanced score;
- maximum drawdowns;
- market, net, long, short, and flat exposure;
- round trips and trade-event rates;
- profit factor, payoff ratio, and win/loss rates;
- trade-return, holding-time, streak, fee, swap, and total-cost statistics.

The current balanced-score convention is:

```text
balanced_score = agent_return + agent_max_drawdown
```

Because drawdown is negative, the score penalizes its absolute magnitude.

### Sources of truth

| Location | Purpose | Authority |
|---|---|---|
| Git | Code, configs, migrations, tests | Source/config history |
| PostgreSQL `runs` | Experiment identity and resolved config | Authoritative |
| PostgreSQL `training_metrics` | PPO training history | Authoritative |
| PostgreSQL `checkpoints` | Checkpoint identity and integrity | Authoritative metadata |
| PostgreSQL `evaluations` | Persisted evaluation results | Authoritative |
| Checkpoint ZIP files | Model weights for replay/resume | Required model source |
| Imported market-data Parquet and manifest v2 | Dataset identity and training input | Required data source |
| Parquet/JSON/CSV/PNG artifacts | Detailed analysis and presentation | Derived/rebuildable |
| Streamlit dashboard | Interactive presentation | Not a source of truth |

## Database migrations

Apply all migrations:

```bash
alembic upgrade head
```

Inspect state and history:

```bash
alembic current
alembic history
```

When adding persisted fields:

1. update the SQLAlchemy model;
2. create an Alembic migration;
3. update persistence and query code;
4. add or update tests;
5. apply the migration;
6. verify the real PostgreSQL schema.

## Testing and development

Run the full suite:

```bash
python -m pytest -q
```

Targeted examples:

```bash
python -m pytest -q tests/test_training_cli.py
python -m pytest -q tests/test_run_search_queue.py
python -m pytest -q tests/test_threshold_sweep.py
python -m pytest -q tests/test_dashboard_pareto.py
```

Market-data import and consumer integration checks:

```bash
python -m pytest -q \
  tests/test_market_data_loading.py \
  tests/test_market_data_validation.py \
  tests/test_run_config.py \
  tests/test_evaluation_service.py
```

Syntax checks:

```bash
python -m py_compile run_search_queue.py threshold_sweep.py
```

Before committing:

```bash
git diff --check
git status --short
```

After staging:

```bash
git diff --cached --check
git diff --cached --stat
```

## Recommended research workflow

```text
1. Define one question and one baseline
2. Create explicit YAML configs
3. Add or update a queue manifest when comparing several configs
4. Validate configs and run the queue with --dry-run
5. Run targeted and full tests
6. Commit code, configs, manifests, migrations, and tests
7. Confirm a clean Git worktree
8. Train one config or the selected queue positions
9. Inspect runs, checkpoints, and evaluation histories
10. Replay thresholds or rebuild reports when needed
11. Compare only the relevant dataset/experiment family in the dashboard
12. Confirm promising behavior on multiple seeds
13. Re-test surviving candidates on the target 5-minute data
```

Useful command sequence:

```bash
git status --short
python run_search_queue.py --queue <QUEUE_NAME> --dry-run
python run_search_queue.py --queue <QUEUE_NAME> --select 1-3
python -m train_and_eval.runs evaluations --run-id <RUN_ID>
python -m train_and_eval.runs checkpoints --run-id <RUN_ID>
streamlit run train_and_eval/dashboard/app.py
```

Research principle:

> Do not optimize one attractive run. Look for behavior that survives seed,
> checkpoint, and target-data changes.

## 1-hour research versus 5-minute target

```text
research question
      |
      v
fast 1-hour experiment
      |
      v
diagnostics and infrastructure checks
      |
      v
promising across seeds?
      |
      +-- no --> reject or revise
      |
      +-- yes
            |
            v
      target 5-minute experiment
            |
            v
      multi-seed confirmation
            |
            v
      untouched final-test evaluation
```

Hyperparameters discovered on 1-hour data should not be transferred blindly.
The number of observations, market microstructure, turnover, fee impact,
episode length, training duration, and policy dynamics all change at 5-minute
resolution.

## Troubleshooting

### Missing or unsupported market-data manifest

Import the original manifest v2 from the same pipeline release as the Parquet
files. Place it at `train_and_eval/market_data/manifest.json`.
`validate_market_data --all` no longer creates, resets, or upgrades manifests.

### Market-data hash, size, or timestamp mismatch

Verify that the manifest and Parquet files belong to the same release and that
the files were copied completely. Restore matching source files or publish a
new dataset through the data pipeline. Do not replace the expected hash merely
to make a modified file pass verification.

### Dataset warnings are not approved

Complete warning review and publication in `NQ_HISTORICAL_DATA_PIPELINE`, then
import its published manifest. PPO_AGENT does not provide a warning-acceptance
override and does not rewrite the producer's acceptance record.

### `RunAlreadyExistsError`

Cause: the config's `run.name` is already registered.

Resolution:

```bash
python -m train_and_eval.runs list
```

Inspect the existing run or create a genuinely new config with a unique name.

### Dirty Git rejection

Cause: training or replay detected uncommitted repository changes.

Resolution:

```bash
git status --short
git diff --check
```

Review and commit intentional changes. Do not bypass the check for a real
experiment.

### Queue preflight finds existing runs

Choose new configs, narrow the selection, or skip already registered entries:

```bash
python run_search_queue.py \
  --queue <QUEUE_NAME> \
  --select 1-12 \
  --skip-existing
```

### Dashboard package cannot be executed

Use Streamlit:

```bash
streamlit run train_and_eval/dashboard/app.py
```

### `runs evaluations` does not replay anything

That command only lists persisted evaluations. Use `threshold_sweep.py` for
probability-threshold replay, or `runs report` to rebuild report artifacts.

### PostgreSQL connection fails

```bash
docker compose ps
```

Then verify `DATABASE_URL`, the exposed host port, credentials, and migration
state.

### A resumed run cannot resolve its source

Verify the exact source name and its available checkpoints:

```bash
python -m train_and_eval.runs list
python -m train_and_eval.runs checkpoints --run-id <SOURCE_RUN_ID>
```

The source checkpoint, checkpoint file, and archived configuration must all be
available and compatible.

## Two manually started training stages

The recommended workflow separates model development from weekly walk-forward.
Stage one uses the existing training CLI and ordinary RunConfig YAML files.
Stage two uses a checkpoint-based walk-forward protocol (schema_version: 3).
Nothing automatically launches stage two after stage one finishes.

### Stage one: explicit dates, ordinary training and validation

Configurations: configs/stage_one/nq5m_v1_seed1.yml (also seed2 and seed3).
The data section contains concrete UTC [start, end) ranges:

```yaml
data:
  path: data/NQ_CONTINUOUS_5m_WEEKDAYS_2010-06-07_00-00_2026-09-08_23-55_20260909_163317.parquet
  train_range:
    start: '2010-06-07T00:00:00+00:00'
    end: '2018-07-30T00:00:00+00:00'
  validation_range:
    start: '2018-07-30T00:00:00+00:00'
    end: '2019-07-01T00:00:00+00:00'
  alignment: trim_start
```

Training reads these dates directly. It does not recalculate percentages or
change dates when new market data is published. Initial observation context is
reserved first; the remaining oldest training candles are trimmed by N % batch_size.
Validation retains every available candle. Missing candles are not imputed.

The supplied starting configurations retain the MLP, trading costs and PPO
settings of the earlier baseline. Their initial budget is one data epoch;
learning rate is 0.0003. These are editable starting points, not a claim of
strategy quality. evaluation.training_mode is scheduled: periodic and final
validation, periodic/final checkpoints, existing early stopping and reports all
remain available. Use distinct run names for different configurations; ordinary
resume remains available for further development on the same explicit ranges.
Choose the source checkpoint using stage-one validation, before seeing stage-two
tests. A source can be a validated periodic checkpoint, not only the final one.

### Calculate dates and confirm the config update

The standalone helper uses the verified dataset and manifest from the selected
config. Defaults are 50% of the dataset's calendar span for training and a 90/10
train/validation ratio within stage one. These percentages are helper arguments,
not training-config fields:

```bash
python set_stage_one_ranges.py --config configs/stage_one/nq5m_v1_seed1.yml \
  --train-fraction 0.5 --train-split 0.9
```

It computes the nominal training end, rounds it UP to Monday 00:00 UTC, and
computes validation length from that rounded training duration. Validation weeks
are ceil(train_weeks * (1 - train_split) / train_split). An already aligned
boundary is unchanged. If the dataset starts midweek, the first partial week is
excluded so the training starts on the next Monday. Both windows must fit inside
the published coverage. For the current release: 425 training weeks + 48 validation
weeks; actual calendar split = 425 / 473, approximately 0.89852.

The helper previews old/new ranges, weeks, actual split, context, batch trimming
and candle counts, then asks for confirmation in the terminal. Enter y/yes or
t/tak to save; Enter, no, EOF or Ctrl-C leaves the config unchanged. It replaces
only the two block-style range fields, preserves unrelated YAML text, checks for
concurrent edits, and writes atomically. Use the supplied block-style YAML;
ambiguous/unsupported layouts are rejected. There is no automatic-confirm flag.
The helper changes no database rows and starts no training.

After reviewing and committing config changes, start stage one manually:

```bash
python -m train_and_eval.training --config configs/stage_one/nq5m_v1_seed1.yml
```

### Stage two: choose a checkpoint and start walk-forward manually

Configurations: configs/stage_two/nq5m_v1_seed1.yml (also seed2 and seed3).
Replace source_checkpoint_id: null with the numeric ID you selected in stage one.
The null placeholder intentionally prevents accidental training from an arbitrary
checkpoint. The source must belong to a completed ordinary temporal run and have
a completed validation. The entire resume ancestry must use the same explicit
training/validation ranges and dataset; walk-forward candidate/refit runs are not
accepted as stage-one sources. The seed must match the source.

Stage two inherits architecture, observation context, stake_pln, fee_bps, swap_bps,
trading/session rules, data path, device, checkpoint cadence and logging settings
from the persisted stage-one source. Costs and stake cannot be grid options.
The protocol requires deterministic_argmax, forced position closing and
batch-aligned n_steps. Stage one is unchanged.

### Stage-two grid: first strict validation improvement

Stage-two YAML uses schema_version: 3. The old scalar learning_rate,
reject_updates and final_checkpoint selection rule are not accepted. Existing
studies/results remain stored and reportable, but cannot be continued with this
new protocol. Use a new study name and commit the configuration before running.
The supplied seed configs have new study names and preserve their checkpoint IDs.

```yaml
schema_version: 3
name: nq5m_stage_two_grid_example
seed: 1
source_checkpoint_id: 68  # Replace with your validated stage-one checkpoint.
validation_weeks: 4
test_weeks: 1
step_weeks: 1
bootstrap_epochs: 1
update_epochs: 1
refit_epochs: 1
grid:
  ppo.learning_rate: [0.000075]
  environment.turnover_penalty: [0.0]
selection_rule: first_strict_improvement
optimizer_policy: preserve_independent_copy
partial_test: skip
```

This is a syntax example, not a recommended search space. Supplied configs keep
only the previous learning rate as a one-option grid and otherwise inherit the
source. Add options deliberately before starting a new study. Every grid value
must be a nonempty list; duplicate and nonfinite options are rejected. Fields
are sorted alphabetically, options retain their listed order, and the rightmost
field changes fastest in the Cartesian product. Config mapping order does not
change the search. An empty grid means one candidate with inherited settings.

Allowed fields:

- PPO: ppo.n_steps, ppo.batch_size, ppo.n_epochs, ppo.learning_rate, ppo.gamma,
  ppo.gae_lambda, ppo.clip_range, ppo.clip_range_vf, ppo.normalize_advantage,
  ppo.ent_coef, ppo.vf_coef, ppo.max_grad_norm, ppo.target_kl.
- Reward: environment.reward_scale, environment.exposure_penalty,
  environment.turnover_penalty, environment.drawdown_penalty,
  environment.profit_reward_mult, environment.loss_reward_mult.

All other fields are inherited and forbidden in the grid, including stake and
fees/swaps, observation shape, position_side, architecture, device and session
rules. Omitted settings are inherited from the original stage-one configuration,
not from a previous cycle's winning options. Nullable PPO settings accept null;
normalize_advantage accepts true/false. Normal RunConfig constraints still apply.
All combinations are validated before any study/run writes; invalid combinations
are errors, not silently skipped trials. Calendar plans include candidate/refit
row bounds for each combination, since batch_size can change prepended rows.

Each cycle:

1. Evaluate the unchanged base on the current validation window as reference.
2. For each grid combination, reload an independent copy of that same base
   checkpoint and optimizer state with the same seed. Train on the cycle's update
   window, then validate its final checkpoint on the same window as reference.
   Rejected trials never become the next trial's starting point.
3. Accept the FIRST candidate with balanced_score strictly greater than reference.
   Ties are rejected. Stop searching immediately; later combinations are not run.
   This is first improvement, not the best score over the entire grid.
4. Keep the accepted pre-refit checkpoint as the next cycle's base. Refit a separate
   copy on the whole validation window using the accepted grid options and the
   predetermined refit_epochs budget. Refit has no validation/checkpoint selection.
5. Freeze the refitted copy and evaluate the following test window. Test results
   never influence candidate acceptance.

Cycle 1 candidates train on the ENTIRE stage-one validation period using
bootstrap_epochs. Later cycles train on the oldest step_weeks leaving validation
using update_epochs. Each training window prepends up to batch_size - 1 historical
candles for batch alignment, with a separate full preceding context. Missing
candles are not imputed. Trading/refit copies never replace the base.

If every combination fails to improve, the cycle and study become stopped.
There is NO refit, test or advance to the next cycle. The terminal prints the
cycle, attempt count, reference score and best candidate score. Earlier tests,
all attempted runs/checkpoints and their validation results remain available.
A report is also generated when no test has completed; it contains validation
and stopping details without inventing a test P&L. Re-running an exhausted study
does not create additional trials. Changing the grid requires a new study name.
A nonfinite score or an operational failure is an error, not ordinary exhaustion.

Each completed attempt is persisted immediately, including order, overrides,
seed, source identity, run/checkpoint/evaluation IDs and validation score.
After interruption between completed attempts, continuation reuses those attempts;
interrupted/failed training runs are not silently retrained. A new study name is
required in that case. Independent optimizer copies remain mandatory.

Only checkpoint-based stage-two protocols (schema_version: 3) are accepted.
No automatic stage-one training or backward-compatible scalar/grid mode is added.
No database migration is required; existing experiments/artifacts are not deleted.

For the supplied stage-one dates, cycle 1 updates on 2018-07-30 to 2019-07-01,
validates on 2019-07-01 to 2019-07-29, then tests on 2019-07-29 to 2019-08-05.
There are 371 complete weekly tests through 2026-09-07; the final partial week is
omitted. Dates are always UTC and end-exclusive. Stage-one validation is never
included in the aggregate test curve.

After setting the checkpoint ID and committing the stage-two config:

```bash
python -m train_and_eval.walk_forward plan \
  --config configs/stage_two/nq5m_v1_seed1.yml --output /tmp/ppo_stage_two_plan.json
python -m train_and_eval.walk_forward run \
  --config configs/stage_two/nq5m_v1_seed1.yml --max-cycles 3
```

The plan command reads the database to resolve source identity but creates no
study or training run. To continue, rerun without --max-cycles using the same
config, dataset and code commit. Source ID, hash, originating run/commit, ancestry,
stage-one bounds and every cycle's actual candle ranges are frozen in plan.json
and the study record. The source run may originate from an earlier Git commit;
within an existing stage-two study, commit changes still block continuation.
Changing tests or documentation therefore also requires attention before resuming
an old study. No existing study, experiment, checkpoint or dataset is deleted by
this update. No database migration is needed.

All existing per-stage persistence, test trajectories, always-long comparison,
separate validation/test tables and additive fixed-stake aggregate reports remain
in use. Reports are saved under artifacts/walk_forward/<study-id>/report.html.
Only consecutive non-overlapping tests contribute to the combined capital and
global drawdown curves.

### Stage-two terminal progress

Stage two shows a table-based study panel instead of per-run PPO preflight, loss metrics
and validation progress panels. Stage-one terminal output is unchanged.
The panel refreshes during training rollouts and evaluation callbacks (at most
twice per second), and immediately after operations complete. The timer measures
the current invocation, not previous sessions.

The progress table has Done, Total, Left and progress-bar columns. Separate bars
show completed test cycles and the active operation. Validation and test tables
have Metric, Mean, Median, Best and Worst columns. Values remain raw fractions,
as in the saved reports; DD is agent_max_drawdown.

Interactive terminals of at least 80 columns by 24 rows refresh the panel inline,
in the normal terminal buffer, like stage-one progress. Shell scrollback remains
available: the renderer does not switch screens, hide the cursor or clear the
terminal. It replaces only its own previous panel and reserves a row for the
cursor to prevent the first panel line scrolling out of reach.
Live extrema identify cycles; terminals at least 150 columns wide and the final
printed tables include full UTC date ranges. The final snapshot stays in history.
After a terminal resize a fresh panel is appended because old cursor offsets
may no longer be valid. Smaller terminals use complete plain snapshots without
live redraw. Training still occupies the foreground shell until it ends.

During stage-two training and evaluation, Python warnings and writes through
sys.stdout/sys.stderr clear the current panel before printing. Messages remain
in scrollback; the next refresh draws the panel below them. Warning filters and
warning-as-error behavior are preserved, and stream/warning hooks are restored
even if an operation raises. This does not intercept native writes directly to
file descriptors or log handlers holding their own pre-existing stream.


Week counters show completed / total / remaining work for base training, trading
refit, candidate validation, unchanged-source reference validation and test.
Training counts nominal calendar weeks multiplied by the configured data epochs.
Overlapping validation/refit windows count again each time they are processed.
Warm-up context and batch-alignment prepends do not add nominal weeks. Counters
advance when an operation finishes; a separate percentage describes the active
operation. These are workload counters, not unique weeks of market history.

Validation and test have separate running best/worst values (with cycle and UTC
[start, end) dates), means and medians for balanced_score, agent_return and
agent_max_drawdown. Validation statistics describe entire validation windows;
test statistics describe test windows (one week by default). Each metric has its
own best/worst window. For signed drawdown, the value closest to zero is best.
Only completed results enter statistics, with no reference evaluations mixed in.
These aggregate means/medians/extrema are display-only. Acceptance uses the
individual candidate balanced_score versus reference on the same window.

On resume the panel rebuilds counters and statistics from persisted cycle results.
Totals cover the full study even with --max-cycles; the invocation limit is shown
separately. Without an interactive terminal, or with --plain-output, output is
limited to snapshots at start, cycle completion and exit. Each snapshot includes
the aggregate statistics. Errors still propagate normally.

### Tests for the two-stage workflow

```bash
docker compose up -d --wait postgres
python -m pytest -q
```

The PostgreSQL integration suite trains stage one on generated data,
starts stage two from its explicitly selected checkpoint, executes three cycles,
resumes without duplicate runs, checks lineage and generates a report. It uses the
same isolated test database/schema cleanup described below.

### Persistence, interruption and replay

`walk_forward_studies` stores the frozen protocol and calendar. Each row in
`walk_forward_cycles` stores its source/selected/refit checkpoint IDs, candidate
selection evidence, reference evaluation used for acceptance and final test evaluation.
Existing `runs` rows record cycle, stage role and candidate identity.

`runs.window_metadata` is authoritative for temporal runs: it stores the full
dataset row count, nominal and actual training bounds, observation lookback,
trim/prepend counts and validation bounds. In temporal mode, `train_rows` counts
the environment slice including context; legacy `split_index` is retained for
compatibility but is **not** a global dataset boundary. Consumers must use
`window_metadata` for these runs. Ratio-mode runs retain their original meaning.

`runs.stage_summary` records environment steps, rollout sizes, the difference
in PPO's epoch-update counter, actual optimizer-step calls, initial policy hash
and initial checkpoint where applicable. PPO epochs and optimizer steps are
separate quantities.

Candidate validations use `run_validation`, unchanged-source reference evaluations use
`custom_range`, and test evaluations use `extended_out_of_sample`. Replay of a
persisted evaluation uses its own archived start/end indices, including when
regenerating a missing test trajectory. Dataset and checkpoint identity checks
remain enabled.

Only one process may advance a named study at a time, guarded by a PostgreSQL
session advisory lock. Rerunning a paused study reuses completed stages and
exact completed evaluations; it does not rerank using test scores or overwrite
completed tests. Selection is committed before refit and before test.

Automatic recovery does not silently retrain a failed, running or pending
training run. Such a state requires diagnosis; preserve it and start a new
study name for a full restart. An interrupted evaluation can be repeated from
the unchanged checkpoint; an already completed exact evaluation is reused.
A changed protocol, data release or code commit also requires a new study name.

### Aggregate reports

Rebuild a report from a consecutive completed prefix:

```bash
python -m train_and_eval.walk_forward report --study-id <STUDY_ID>
```

Outputs are under `artifacts/walk_forward/<zero-padded-study-id>/`:

- `report.html`: standalone report with embedded charts;
- `validation.csv`: candidate and unchanged-source validation comparisons;
- `tests.csv`: separate per-test metrics and checkpoint lineage;
- `summary.json`: weekly mean, median, minimum, maximum, positive-window share,
  exposure, trades, fees, swaps, total costs and full-path drawdown;
- `test_trajectory.parquet`: the additive, chronological test-only path;
- `protocol.json`, `plan.json`, `stages.json`: protocol, exact ranges and stage
  provenance; PNG charts are also saved individually.

Individual validation/test trajectories, trade events and checkpoints remain
under the existing `artifacts/runs/` structure. Existing run-level PPO diagnostic
plots can still be generated with `python -m train_and_eval.runs report`.

For fixed `stake_pln`, `agent_return` is additive cumulative P&L expressed as a
fraction of that nominal. Aggregate PLN P&L is `stake_pln * sum(test returns)`;
there is **no compounding**. Each weekly trajectory is offset by previous test
results before computing the maximum drawdown of the entire path, including
its initial zero. Drawdown is expressed against the fixed nominal, not current
account equity. Validation results never enter that path.

The always-long comparator uses the same execution candles, fees, swaps and
weekly forced closing/reopening rules. It is not an uninterrupted multi-year
buy-and-hold position. Aggregate exposure is weighted by available scored bars,
not elapsed time, because missing candles are not imputed.

### Walk-forward tests

The ordinary suite includes calendar, gap, warm-up, alignment, future-data
isolation, optimizer-copy, training-without-validation, selection and additive
reporting checks:

```bash
python -m pytest -q
```

The integration test runs automatically with the ordinary pytest suite. It runs
real PPO on generated data, applies all migrations, executes three cycles,
verifies lineage and completed-stage reuse, and reconstructs a missing trajectory.

By default, tests read DATABASE_URL from the environment or local .env and
create/reuse ppo_agent_test on the same PostgreSQL server with the same credentials.
Provisioning connects to the postgres maintenance database, not the application
database. First creation requires CREATEDB permission, available to the user
initialized by the supplied Compose image. Start PostgreSQL before testing:

```bash
docker compose up -d --wait postgres
python -m pytest -q
```

Each invocation creates a random wf_test_* schema inside the test database.
Migrations, runs and sequences live only there. A finally block removes that
schema after success or failure, including migration failures, and verifies its
removal. Application experiments, artifacts and ID counters are untouched.
Generated data and artifacts use pytest temporary directories. The empty test
database is retained for reuse. A forced process kill or database outage may
leave a temporary schema behind; later tests use new schemas and never remove
unrelated schemas.

Optionally set PPO_WALK_FORWARD_TEST_DATABASE_URL in the environment or .env to
use an already existing dedicated database. Its name must end in _test and differ
from the application database name. Explicit targets are not provisioned.
Missing configuration, connection failures and insufficient permissions fail the
test instead of skipping it. To run only the walk-forward integration test:

```bash
python -m pytest -q -s tests/test_walk_forward_integration.py
```

The downgrade refuses to discard existing walk-forward history or invalidate
training-only temporal runs.

### Grid search reporting and progress

The terminal shows the active candidate number. Base/validation work counts all
completed attempts (including rejected ones); their totals and remaining budgets
are upper bounds assuming every combination runs in every cycle. Early acceptance
skips unused combinations, so these bars need not reach 100% even when all tests
complete. Refit/reference/test retain their usual cycle-based counts. Epochs and
overlapping windows are counted as repeated work, not distinct calendar history.
Validation statistics contain accepted candidates only; rejected attempts remain
in validation.csv and attempts.json. Test statistics contain completed tests only.
The inline panel preserves terminal history and warning messages as before.

Reports include attempts.json with all attempted cycle selections and stop reasons,
plus validation.csv with candidates and reference, including an exhausted cycle.
Only the consecutive completed tests contribute to P&L, aggregate drawdown and
weekly test statistics. No capitalization is introduced.

## Cleaning experiment stages

`extra_tools/clean_training.py` removes experiment records and their registered
artifacts. Stop training, evaluation and report processes before use. The script
refuses to proceed while the database records running work, locks the experiment
tables during planning/deletion, prints the exact scope, and requires typing
`SURE` (exactly, uppercase, without surrounding spaces) before deletion. There is no unattended confirmation flag.

Modes:

- `clean-first-stage`: ordinary runs (`cycle_id` and `stage_role` both null),
  including their checkpoints, evaluations and training metrics. Use repeated
  `--run-id` arguments to select specific runs; without filters it selects all
  ordinary runs, including historical experiments. Dependencies from retained
  runs/studies block deletion. Clean stage two first or use `clean-all`.
- `clean-second-stage`: complete walk-forward studies with all cycles, candidate
  and refit runs, checkpoints, metrics, evaluations and study reports. Select
  studies with repeated `--study-id`; `--run-id` expands to the complete owning
  study and the expanded scope is printed. No filters means all studies. Reference
  evaluations attached to stage-one checkpoints are removed if no retained cycle
  uses them. Stage-one training checkpoints and validations are preserved.
- `clean-all`: all experiment rows in the six experiment tables, plus their
  registered run/study artifacts and orphan numeric directories discovered on
  disk under artifacts/runs and artifacts/walk_forward. It also works when the
  database is empty. All six experiment ID sequences restart at 1. It does not
  accept ID filters.

Examples (each modifying invocation asks for confirmation):

```bash
python extra_tools/clean_training.py clean-first-stage --run-id 13 --dry-run
python extra_tools/clean_training.py clean-second-stage --study-id 8 --dry-run
python extra_tools/clean_training.py clean-second-stage
python extra_tools/clean_training.py clean-first-stage
python extra_tools/clean_training.py clean-all
```

`--dry-run` prints the plan without modifying records or files. The script uses
DATABASE_URL and the project's .env through the existing database module.
`--project-root` can explicitly select the project directory.

Market data, manifest, configurations, migrations and schema are preserved.
Stage-specific cleanup preserves ID sequences. After `clean-all`, the next
insert into each experiment table receives ID 1: runs, checkpoints, evaluations,
training_metrics, walk_forward_cycles and walk_forward_studies. Migration history
is retained, so the experiment database is empty and ready for new runs without
recreating its schema. Configured checkpoint IDs are not rewritten automatically.

The preview lists `sequence_resets`, including the actual sequence names and
restart values. Resetting an already empty database also requires `SURE`.
`--dry-run` never resets counters. For cleanup with artifacts, counters restart
only after artifact deletion completes, including recovery after an interruption.
The reset checks that all six tables are empty and uses transactional
`ALTER SEQUENCE ... RESTART WITH 1` under experiment-table locks.
 Artifact deletion is restricted to selected `artifacts/runs/<id>` and
`artifacts/walk_forward/<id>` directories, plus owned reference evaluation
subdirectories. Symlink paths and nonstandard checkpoint locations are rejected.
For clean-all, the preview includes orphan_paths: canonical positive ID directories
(e.g. 00000001) with no matching run/study in the connected database. These are
included in paths and removed only after SURE. Ownership is relative to the
connected database; verify the displayed database/project before confirming.
Stage-specific modes do not infer the stage of orphan runs and leave these
unowned directories alone. Parent directories, README.md, .gitignore, ordinary
files and noncanonical names remain. Symlinks are rejected rather than followed.
Manual exports (for example diagnostic CSV/plots under /tmp) and unrelated folders
are not swept. The check reads the filesystem, independently of IDE refresh timing.

Before database commit, artifacts are renamed into a temporary directory under
artifacts. A durable artifacts/.cleanup_pending.json journal allows recovery after
an exception/interruption. Rerun the script and type RECOVER when prompted: if the
experiment rows still exist, artifacts are restored; if the transaction committed,
staged artifacts are deleted. Keep the same database/project and do not start
training or manually remove the journal/staged files before recovery. Recovery
refuses inconsistent or ambiguous database state. This is interruption recovery,
not a backup after successful deletion.

For an orphan-only operation there are no database records to roll back. Its
journal records the confirmed deletion scope: RECOVER finishes removing both
staged and not-yet-staged orphan directories. Recovery refuses deletion if an ID
has since acquired an owner in the connected database. In a mixed operation,
orphans are restored along with registered artifacts if the DB transaction rolls
back. The recovery hint is printed on errors only when the journal exists.

The dedicated tests cover both stage selections, shared reference preservation,
blocking dependent deletions, confirmation/dry-run, complete deletion, rollback
recovery, post-commit recovery, orphan discovery/deletion/recovery, path restrictions,
all six sequence resets, empty-database resets and transactional reset rollback. Database tests use the
existing isolated test database/schema helper and never the experiment database.
