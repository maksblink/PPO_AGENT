# PPO_AGENT

PPO-based reinforcement learning research project for training and evaluating trading agents on historical market data.

The project is designed around **reproducible experiments**, **PostgreSQL-backed run tracking**, **immutable checkpoints**, **persistent evaluation results**, **resumable training**, and **rebuildable artifacts**.

The target trading setup is **5-minute market data**. Longer intervals such as **1 hour** are used for faster research, infrastructure validation, hyperparameter screening, and diagnostic experiments. Final conclusions about the trading agent should be confirmed on the target 5-minute setup.

---

## Table of contents

- [Main features](#main-features)
- [Quick start](#quick-start)
- [Project structure](#project-structure)
- [Run configuration](#run-configuration)
- [Training and evaluation devices](#training-and-evaluation-devices)
- [Training](#training)
- [Resume training](#resume-training)
- [Run registry CLI](#run-registry-cli)
- [Database](#database)
- [Evaluation policy modes](#evaluation-policy-modes)
- [Reports and artifacts](#reports-and-artifacts)
- [Policy probability diagnostics](#policy-probability-diagnostics)
- [Dashboard](#dashboard)
- [Pareto Explorer](#pareto-explorer)
- [Extra tools](#extra-tools)
- [Artifact layout](#artifact-layout)
- [What is stored where](#what-is-stored-where)
- [Database migrations](#database-migrations)
- [Testing](#testing)
- [Recommended research workflow](#recommended-research-workflow)
- [1-hour research vs 5-minute target](#1-hour-research-vs-5-minute-target)

---

## Main features

- YAML-configured PPO experiments
- chronological train/validation split
- market-data validation and SHA-256 integrity tracking
- PostgreSQL run registry
- Git commit and branch tracking for every run
- clean-Git requirement for reproducible training and historical replay
- fresh and resumed training runs
- immutable PPO checkpoints
- periodic and final validation
- early stopping based on validation performance
- persistent PPO training metrics
- critic and rollout diagnostics
- evaluation metrics persisted in PostgreSQL
- validation trajectories stored as Parquet when required
- full policy action probabilities persisted in evaluation trajectories
- rebuildable reports and plots
- automatic policy-confidence plots
- Streamlit experiment dashboard
- interactive Pareto-front analysis
- cross-run policy-probability comparison tools
- probability-threshold replay diagnostics
- separate devices for training and evaluation
- Alembic-managed database migrations
- automated tests with pytest

---

# Quick start

## 1. Create and activate a virtual environment

Example on Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 2. Install the project

Install runtime dependencies:

```bash
python -m pip install -e .
```

Install development dependencies too:

```bash
python -m pip install -e ".[dev]"
```

## 3. Start PostgreSQL

```bash
docker compose up -d postgres
```

Check it:

```bash
docker compose ps
```

A local development setup may expose PostgreSQL on a non-default host port, for example:

```text
127.0.0.1:5433 -> container:5432
```

## 4. Configure `DATABASE_URL`

The database connection is read from the environment or local `.env`.

Example:

```env
DATABASE_URL=postgresql+psycopg://ppo_agent:<password>@127.0.0.1:5433/ppo_agent
```

Do not commit real secrets.

The database layer validates that the URL points to PostgreSQL and uses psycopg.

## 5. Apply migrations

```bash
alembic upgrade head
```

## 6. Run tests

```bash
pytest -q
```

## 7. Train one experiment

```bash
python -m train_and_eval.training   --config configs/experiments/nq1h_search_v1/34_nepochs3_lr2p25e4_gamma090_gae085_exp1p5em5_seed1.yml
```

## 8. Inspect the registry

```bash
python -m train_and_eval.runs list
```

## 9. Start the dashboard

```bash
streamlit run train_and_eval/dashboard/app.py
```

---

# Project structure

```text
PPO_AGENT/
├── alembic/
│   └── versions/                  # database migrations
├── artifacts/
│   └── runs/                      # generated checkpoints/reports/evaluations
├── configs/
│   └── experiments/               # YAML experiment configurations
├── data/                          # market-data inputs
├── extra_tools/
│   └── policy_probability_diagnostic.py
├── tests/
├── train_and_eval/
│   ├── dashboard/
│   │   ├── app.py
│   │   ├── data.py
│   │   └── pareto.py
│   ├── database/
│   │   ├── models.py
│   │   └── session.py
│   ├── environment/
│   ├── evaluation/
│   ├── ppo/
│   ├── reporting/
│   │   ├── artifacts.py
│   │   └── service.py
│   ├── runs/
│   └── training/
├── pyproject.toml
├── run_search_queue.py
└── threshold_sweep.py
```

Generated runtime artifacts are not intended to be committed to Git.

---

# Run configuration

Experiments are defined in YAML files under:

```text
configs/experiments/
```

A run configuration contains sections for:

```text
run
continuation
data
training
logging
artifacts
environment
ppo
evaluation
```

The full **normalized configuration** is persisted with every run in PostgreSQL.

This means the database stores the effective configuration that was actually used, not only the original YAML path.

A controlled experiment should normally change one variable at a time:

```text
baseline
  |
  +-- learning_rate only
  |
  +-- gamma only
  |
  +-- gae_lambda only
  |
  +-- confirm candidate on additional seeds
```

---

# Training and evaluation devices

Training and evaluation can use different devices.

```yaml
ppo:
  device: cuda

evaluation:
  device: cpu
```

Supported values:

```text
auto
cpu
cuda
```

This is useful because PPO optimization can benefit from GPU acceleration while sequential validation can sometimes be more efficient on CPU.

---

# Training

Run one YAML-configured experiment:

```bash
python -m train_and_eval.training   --config configs/experiments/<experiment>.yml
```

Example:

```bash
python -m train_and_eval.training   --config configs/experiments/nq1h_search_v1/34_nepochs3_lr2p25e4_gamma090_gae085_exp1p5em5_seed1.yml
```

A training run can create:

- one row in `runs`
- many rows in `training_metrics`
- periodic checkpoints
- one final checkpoint
- scheduled evaluations
- one final evaluation
- evaluation trajectories
- evaluation plots
- run-level reports

The project is intentionally strict about reproducibility. Training and historical replay require a clean Git state so the recorded commit corresponds to the code that produced the experiment.

Check before important runs:

```bash
git status --short
```

For a reproducible run this should normally be empty.

---

## Search queue

The repository also contains:

```text
run_search_queue.py
```

It is used for ordered experiment-series execution.

Dry run:

```bash
python run_search_queue.py --dry-run
```

The queue also supports continuation/inspection modes such as:

```text
--start-at
--summary-only
```

Use the queue for controlled experiment series where configs are created and committed before training.

---

# Resume training

Training can continue from a checkpoint produced by an earlier run.

Conceptually:

```text
Run A
  |
  └── Checkpoint 42
          |
          └── Run B (resume)
```

The lineage is stored through:

```text
runs.source_checkpoint_id -> checkpoints.id
```

Resume compatibility checks protect against continuing with incompatible model, environment, or PPO settings.

Checkpoint steps have two meanings:

```text
run_step
model_step
```

- `run_step` is local to the run that created the checkpoint
- `model_step` includes the complete model history across resume boundaries

---

# Run registry CLI

## List runs

```bash
python -m train_and_eval.runs list
```

## Show one run

```bash
python -m train_and_eval.runs show --run-id 34
```

## List checkpoints

```bash
python -m train_and_eval.runs checkpoints --run-id 34
```

## List evaluations

```bash
python -m train_and_eval.runs evaluations --run-id 34
```

This is a **read-only lookup**. It does not regenerate artifacts.

## Show full evaluation information

```bash
python -m train_and_eval.runs evaluations   --run-id 34   --full
```

## Rebuild reports

```bash
python -m train_and_eval.runs report --run-id 34
```

Unlike `evaluations`, `report` actually regenerates report/artifact files.

Rebuild only one evaluation:

```bash
python -m train_and_eval.runs report   --run-id 34   --evaluation-id 136
```

---

# Database

PostgreSQL is the persistent experiment registry and should be treated as the main source of truth for experiment identity, configuration, training history, checkpoints, and evaluation results.

Main runtime tables:

```text
runs
checkpoints
training_metrics
evaluations
```

## Database architecture

```text
                               runs
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
              ▼                 ▼                 │
      training_metrics     checkpoints            │
                                │                 │
                                ▼                 │
                           evaluations            │
                                                  │
                 resume relationship              │
                 runs.source_checkpoint_id ───────┘
                          -> checkpoints.id
```

More explicitly:

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

A key design detail is that `evaluations` are linked to a **checkpoint**, not directly to a run.

To resolve the run for an evaluation:

```text
evaluations.checkpoint_id
        ->
checkpoints.id
        ->
checkpoints.run_id
        ->
runs.id
```

This matters for SQL queries, dashboard joins, and analysis scripts.

---

## `runs`

One row represents one complete training execution.

The run record stores experiment identity and reproducibility information such as:

- run ID
- unique run name
- run status
- seed
- fresh/resume continuation mode
- source checkpoint for resumed runs
- Git commit
- Git branch
- market-data SHA-256
- normalized resolved configuration
- train/validation row counts
- split index
- steps per data epoch
- requested training steps
- completed training steps
- completed data epochs
- early-stop information
- failure/error information when relevant
- start/finish timestamps
- other run-level reproducibility metadata

The run row answers:

> What exactly was trained, with which configuration, data, seed, and source code?

---

## `checkpoints`

One row represents one immutable saved PPO model file.

Important persisted information includes:

- checkpoint ID
- `run_id`
- `run_step`
- `model_step`
- save reason
- relative artifact path
- SHA-256
- file size
- creation timestamp

Save reasons include values such as:

```text
initial
periodic
final
manual
interrupted
```

The database enforces at most one final checkpoint per run.

Example:

```text
Run #34
  |
  +-- checkpoint at step 18,432
  +-- checkpoint at step 36,864
  +-- checkpoint at step 55,296
  └-- final checkpoint at step 68,608
```

---

## `training_metrics`

Training metrics store PPO/rollout diagnostics throughout training.

Typical persisted values include:

- `run_id`
- local `run_step`
- history-aware `model_step`
- episode/rollout reward information
- rollout reward mean
- rollout reward sum
- entropy loss
- explained variance
- approximate KL divergence
- clip fraction
- policy-gradient loss
- value loss
- learning rate
- additional critic/rollout diagnostics when available

These rows rebuild:

```text
training_curves.png
```

and support training-stability analysis.

Typical questions:

```text
Did approx_kl spike?
Did clip_fraction increase?
Did explained_variance collapse?
Did entropy change strongly?
Did value loss become unstable?
```

---

## `evaluations`

An evaluation row stores the validation result for one checkpoint.

It stores execution/reproducibility metadata such as:

- evaluation ID
- checkpoint ID
- trigger
- status
- data scope
- policy mode
- probability-threshold settings when applicable
- evaluation seed
- progress/completion information
- failure information when relevant

It also stores trading metrics.

### Return and benchmark metrics

Examples:

```text
agent_return
always_long_return
always_short_return
agent_vs_always_long_return
balanced_score
```

Current balanced-score convention:

```text
balanced_score = agent_return + agent_max_drawdown
```

Because drawdown is negative, this subtracts its absolute magnitude from return.

### Drawdown metrics

Examples:

```text
agent_max_drawdown
always_long_max_drawdown
always_short_max_drawdown
drawdown_improvement
```

Example:

```text
-20% is better than -40%
```

### Exposure metrics

Examples:

```text
net_exposure
long_exposure
short_exposure
flat_exposure
market_exposure
```

For LONG/FLAT:

```text
market_exposure = fraction of evaluation time in the market
flat_exposure   = fraction of evaluation time flat
```

### Trading activity

Examples:

```text
trade_events_total
trade_event_rate
round_trips
round_trip_rate
open_long_count
open_short_count
close_long_count
close_short_count
```

### Win/loss metrics

Examples:

```text
winning_trades
losing_trades
breakeven_trades
win_rate
loss_rate
breakeven_rate
long_win_rate
short_win_rate
```

### Trade-return metrics

Examples:

```text
avg_trade_return
avg_win_return
avg_loss_return
median_trade_return
largest_win_return
largest_loss_return
gross_profit_return
gross_loss_return
net_profit_return
profit_factor
payoff_ratio
```

### Holding-time and streak metrics

Examples:

```text
min_bars_held
avg_bars_held
median_bars_held
max_bars_held
max_consecutive_wins
max_consecutive_losses
avg_win_streak
avg_loss_streak
current_streak_type
current_streak
```

### Trading costs

Examples:

```text
total_fee_return
total_swap_return
total_cost_return
avg_fee_per_trade_return
avg_swap_per_trade_return
avg_cost_per_trade_return
```

Additional persisted evaluation information includes shaped-reward totals and open-position return at the end of evaluation.

---

## Important database relationships

### Run -> training metrics

```text
runs.id
    |
    +----< training_metrics.run_id
```

### Run -> checkpoints

```text
runs.id
    |
    +----< checkpoints.run_id
```

### Checkpoint -> evaluations

```text
checkpoints.id
    |
    +----< evaluations.checkpoint_id
```

### Resume lineage

```text
runs.source_checkpoint_id
    |
    +----> checkpoints.id
```

---

# Evaluation policy modes

Supported modes:

```text
deterministic_argmax
stochastic_sample
probability_threshold
```

## Deterministic argmax

For LONG/FLAT:

```text
P(FLAT) > P(LONG) -> FLAT
P(LONG) > P(FLAT) -> LONG
```

This gives a decision boundary around `P(LONG) = 0.5`.

## Stochastic sampling

Actions are sampled from the policy distribution.

The evaluation seed keeps the action sequence reproducible.

## Probability threshold

For the current LONG/FLAT setup:

```text
threshold_action = 1
```

where action `1` is LONG.

Decision rule:

```text
P(LONG) >= probability_threshold -> LONG
P(LONG) <  probability_threshold -> FLAT
```

---

# Reports and artifacts

Rebuild all completed evaluations for a run:

```bash
python -m train_and_eval.runs report --run-id 34
```

Rebuild one evaluation:

```bash
python -m train_and_eval.runs report   --run-id 34   --evaluation-id 136
```

The report system:

1. reads run/checkpoint/evaluation metadata from PostgreSQL
2. checks whether required source artifacts exist
3. replays the immutable checkpoint when artifacts are missing or legacy
4. verifies replayed metrics against the persisted evaluation row
5. writes/rebuilds trajectory artifacts
6. renders evaluation plots
7. renders run-level training/validation curves

Historical replay does **not** create a new evaluation row.

This allows plotting/reporting code to evolve without changing historical evaluation results.

---

## Run-level report files

Typical path:

```text
artifacts/runs/00000034/reports/
```

Typical files:

```text
training_curves.png
validation_curves.png
training_metrics.csv
validation_metrics.csv
summary.json
```

---

## Evaluation-level files

Typical path:

```text
artifacts/runs/00000034/evaluations/00000136/
```

Typical files:

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

Trade-specific plots may be absent if the evaluation does not contain the required events.

---

# Policy probability diagnostics

The project persists the complete categorical policy distribution in `trajectory.parquet`.

For a two-action LONG/FLAT policy:

```text
policy_probability_action_0 = P(FLAT)
policy_probability_action_1 = P(LONG)
```

It also stores:

```text
selected_action_probability
```

which is the probability assigned to the action that was actually selected.

Example:

```text
action = 0
P(FLAT) = 0.90
P(LONG) = 0.10
selected_action_probability = 0.90
```

versus:

```text
action = 1
P(FLAT) = 0.10
P(LONG) = 0.90
selected_action_probability = 0.90
```

The full probability columns make `P(LONG)` analysis independent of the selected action.

---

## `policy_p_long_distribution.png`

Histogram of `P(LONG)` over the evaluation trajectory.

It contains:

- argmax boundary at `0.50`
- mean `P(LONG)`
- median `P(LONG)`

Interpretation:

- mass near `1.0`: strong LONG saturation
- broad distribution: policy uses many confidence levels
- mass near both `0.0` and `1.0`: polarized policy

---

## `policy_p_long_confidence_curve.png`

For every threshold `t`, this shows:

```text
fraction of observations where P(LONG) >= t
```

This is a confidence-distribution / survival curve.

Important:

> It is **not** a full threshold backtest.

Changing the actual threshold can alter actions, future environment state, future observations, and therefore future probabilities.

For exact trading results under a different threshold, use `threshold_sweep.py`.

---

# Dashboard

Start:

```bash
streamlit run train_and_eval/dashboard/app.py
```

The dashboard reads the experiment database and builds a flattened explorer dataset from:

```text
runs
checkpoints
evaluations
training_metrics
```

The evaluation-to-run relationship is resolved through checkpoints.

Sidebar filters apply to all tabs.

Current tabs:

```text
Run Explorer
Scatter Explorer
Activity Map
Pareto Explorer
Group Comparison
Run Detail
```

---

## Run Explorer

Interactive experiment table.

Useful for:

- selecting visible columns
- inspecting hyperparameters
- comparing evaluation metrics
- inspecting PPO diagnostics
- filtering experiment families

---

## Scatter Explorer

General-purpose 2D experiment view.

Example:

```text
X     = Market exposure
Y     = Agent return
Color = Seed
Size  = Round trips
```

Interpretation:

```text
right  -> more exposure
left   -> more time FLAT
up     -> higher return
larger -> more round trips
```

---

## Activity Map

Specialized trading-policy view.

Typical mapping:

```text
X      = Market exposure
Y      = Round trips
Color  = Agent return
Symbol = Seed
```

Useful for separating:

- near always-long policies
- active LONG/FLAT policies
- highly selective policies
- high-turnover regimes

---

# Pareto Explorer

The Pareto Explorer compares runs under two objectives.

Default:

```text
X metric    = Max drawdown
X objective = Maximize

Y metric    = Agent return
Y objective = Maximize

Color       = Seed
```

Because max drawdown is stored as a negative return:

```text
-20% > -30% > -40%
```

therefore maximizing max drawdown means moving closer to zero, which is better.

A run is Pareto-optimal if no other currently visible run is:

- at least as good in both objectives
- strictly better in at least one objective

Example of domination:

```text
Run A:
return = +30%
DD     = -25%

Run B:
return = +20%
DD     = -35%
```

Run B is dominated by Run A.

Example where neither dominates:

```text
Run A:
return = +60%
DD     = -40%

Run B:
return = +40%
DD     = -20%
```

A has better return, B has better drawdown.

The Pareto front is recalculated after sidebar filters are applied.

The tab also shows the Pareto-optimal run table.

---

## Group Comparison

Groups experiments by selected hyperparameters and computes statistics such as:

```text
count
mean
std
min
max
```

Useful for multi-seed confirmation.

A strong single seed should not be treated as a robust result without confirmation.

---

## Run Detail

Detailed inspection of one run, including:

- run metadata
- normalized configuration
- evaluation history
- training-metric history
- charts over model steps

---

# Extra tools

Cross-run diagnostics live under:

```text
extra_tools/
```

This keeps the normal single-run evaluation pipeline focused on reproducible per-run artifacts.

---

## Cross-run policy probability comparison

Tool:

```text
extra_tools/policy_probability_diagnostic.py
```

Example:

```bash
python -m extra_tools.policy_probability_diagnostic   --runs 34,36,37   --output-dir /tmp/ppo_policy_probability_diag
```

Typical outputs:

```text
policy_probabilities.csv
probability_summary.csv
p_long_histogram_overlay.png
p_long_confidence_curve.png
```

This tool is intentionally **cross-run only**. It compares an explicitly selected set of runs.

Single-run policy probability plots are generated automatically by the normal evaluation/reporting pipeline:

```text
policy_p_long_distribution.png
policy_p_long_confidence_curve.png
```

Useful for comparing seed behavior, saturation, and policy polarization.

---

## Probability-threshold sweep

Tool:

```text
threshold_sweep.py
```

Example:

```bash
python threshold_sweep.py   --runs 34,36,37   --thresholds 0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70   --output /tmp/nq1h_threshold_sweep_runs34_36_37.csv
```

The tool:

- loads existing final checkpoints
- performs validation replays
- does not retrain
- evaluates `probability_threshold` mode
- reports exposure, trips, return, max drawdown, PF, balanced score, etc.
- writes CSV output after all replays complete

Use this tool when the question is:

> What actually happens to trading results if I change the policy threshold?

This is different from the confidence curve, which only describes the probability distribution on the existing trajectory.

---

# Artifact layout

Example:

```text
artifacts/
└── runs/
    └── 00000034/
        ├── checkpoints/
        │   └── ...
        ├── reports/
        │   ├── training_curves.png
        │   ├── validation_curves.png
        │   ├── training_metrics.csv
        │   ├── validation_metrics.csv
        │   └── summary.json
        └── evaluations/
            ├── 00000133/
            │   └── ...
            ├── 00000134/
            │   └── ...
            ├── 00000135/
            │   └── ...
            └── 00000136/
                ├── trajectory.parquet
                ├── trade_events.parquet
                ├── metrics.json
                ├── equity_curve.png
                ├── drawdown_curve.png
                ├── market_and_exposure.png
                ├── cumulative_costs.png
                ├── trade_returns.png
                ├── holding_times.png
                ├── policy_p_long_distribution.png
                └── policy_p_long_confidence_curve.png
```

---

# What is stored where

| Location | Purpose | Source of truth? |
|---|---|---|
| Git | code, configs, migrations, tests | Yes, for source/config history |
| PostgreSQL `runs` | experiment identity and resolved config | Yes |
| PostgreSQL `training_metrics` | PPO training history | Yes |
| PostgreSQL `checkpoints` | checkpoint metadata and integrity | Yes |
| PostgreSQL `evaluations` | persisted validation results | Yes |
| checkpoint files | immutable model weights | Yes, for replay |
| `trajectory.parquet` | per-step evaluation trajectory | Rebuildable |
| `trade_events.parquet` | detailed trade event stream | Rebuildable |
| PNG/CSV reports | visualization / analysis convenience | Rebuildable |
| Streamlit dashboard | presentation layer over database | No |

The database should be used for durable comparisons and queries.

Plots should not be treated as the only copy of important metrics.

---

# Database migrations

Schema changes are managed with Alembic.

Apply migrations:

```bash
alembic upgrade head
```

Check current state:

```bash
alembic current
```

Inspect history:

```bash
alembic history
```

When adding persisted fields:

```text
1. modify SQLAlchemy model
2. create Alembic migration
3. update persistence logic
4. add tests
5. apply migration
6. verify schema
```

---

# Testing

Run all tests:

```bash
pytest -q
```

Targeted examples:

```bash
pytest -q tests/test_reporting_artifacts.py
```

```bash
pytest -q tests/test_dashboard_pareto.py
```

Syntax checks:

```bash
python -m py_compile   train_and_eval/dashboard/app.py   train_and_eval/reporting/artifacts.py
```

Before committing:

```bash
git diff --check
git status --short
```

---

# Recommended research workflow

```text
1. Define one research question
        |
        v
2. Create/change one YAML config
        |
        v
3. Validate config / dry-run queue if applicable
        |
        v
4. Commit code + config
        |
        v
5. Confirm clean Git
        |
        v
6. Train
        |
        v
7. Inspect DB / run registry
        |
        v
8. Inspect or rebuild reports
        |
        v
9. Compare in dashboard
        |
        v
10. Confirm promising settings on more seeds
        |
        v
11. Move to the next research question
```

Typical commands:

```bash
git status --short
```

```bash
python -m train_and_eval.training   --config configs/experiments/<experiment>.yml
```

```bash
python -m train_and_eval.runs evaluations --run-id <RUN_ID>
```

```bash
python -m train_and_eval.runs report --run-id <RUN_ID>
```

```bash
streamlit run train_and_eval/dashboard/app.py
```

Research principle:

> Do not optimize one attractive run. Look for behavior that survives seed changes.

---

# 1-hour research vs 5-minute target

Longer intervals such as 1 hour are intentionally used for fast iteration.

```text
idea
  |
  v
fast experiment on 1h
  |
  v
diagnostics / dashboard / multi-seed checks
  |
  v
promising?
  |
  +-- no --> reject or revise
  |
  +-- yes
        |
        v
target 5m experiment
        |
        v
multi-seed confirmation
        |
        v
final comparison
```

Hyperparameters and reward settings found on 1-hour data should **not** be blindly transferred to 5-minute data.

The 1-hour interval is a research and infrastructure tool. Final conclusions about the trading agent should be based on the target 5-minute setup.
