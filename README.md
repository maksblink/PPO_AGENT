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
