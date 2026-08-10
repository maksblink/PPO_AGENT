# PPO_AGENT

PPO-based reinforcement learning research project for training and evaluating trading agents on historical market data.

The project is designed around reproducible experiments, strict run tracking, persistent evaluation results, resumable training, and rebuildable artifacts.

The target agent operates on **5-minute market data**. Longer intervals such as **1 hour** are used for faster development, infrastructure validation, and early-stage experiments.

## Main features

- YAML-configured PPO experiments
- chronological train/validation data split
- market-data validation and SHA-256 integrity tracking
- PostgreSQL run registry
- Git commit and branch tracking for every run
- clean-Git requirement before training and evaluation
- fresh and resumed training runs
- persistent PPO checkpoints
- periodic and final validation
- early stopping based on validation performance
- persistent training metrics
- evaluation metrics stored in PostgreSQL
- optional validation trajectories stored as Parquet
- rebuildable reports and plots
- separate devices for training and evaluation
- automated tests with pytest

## Training and evaluation devices

Training and evaluation can use different devices.

```yaml
ppo:
  device: cuda

evaluation:
  device: cpu
```

Supported device values:

```text
auto
cpu
cuda
```

This separation is useful because PPO training benefits from GPU acceleration, while sequential validation with small batch sizes can be faster on CPU.

## Run configuration

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

The full normalized configuration is persisted with every run.

## Training

Run one YAML-configured training job:

```bash
python -m train_and_eval.training \
  --config configs/experiments/example_5m_timesteps.yml
```

Training runs are registered in PostgreSQL and receive a unique run ID.

A run can produce:

- PPO checkpoints
- periodic validation evaluations
- training metrics
- validation trajectories
- plots and reports

## Resume training

Training can continue from a checkpoint produced by an earlier run.

Resume compatibility checks protect against accidentally continuing training with incompatible model, environment, or PPO settings.

The continuation source and checkpoint selection are configured in YAML.

## Run registry

List runs:

```bash
python -m train_and_eval.runs list
```

Show one run:

```bash
python -m train_and_eval.runs show --run-id 1
```

List checkpoints:

```bash
python -m train_and_eval.runs checkpoints --run-id 1
```

List evaluations:

```bash
python -m train_and_eval.runs evaluations --run-id 1
```

Show full evaluation information:

```bash
python -m train_and_eval.runs evaluations \
  --run-id 1 \
  --full
```

## Reports

Generate or rebuild reports for a run:

```bash
python -m train_and_eval.runs report \
  --run-id 1
```

Generate or rebuild artifacts for one evaluation:

```bash
python -m train_and_eval.runs report \
  --run-id 1 \
  --evaluation-id 1
```

If stored trajectory artifacts are missing, the evaluation can be replayed from its archived checkpoint.

Historical replay does not create a new evaluation row in PostgreSQL.

## Artifact storage

Generated files are stored under:

```text
artifacts/runs/
```

Typical run-level directories:

```text
checkpoints/
reports/
evaluations/
```

Evaluation artifacts can include:

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
```

Generated runtime artifacts are not committed to Git.

See `artifacts/README.md` for details about artifact storage and rebuildability.

## Database

The project uses PostgreSQL for persistent run metadata, checkpoints, evaluations, and training metrics.

Database schema changes are managed with Alembic.

Apply all migrations:

```bash
alembic upgrade head
```

Main runtime tables:

```text
runs
checkpoints
evaluations
training_metrics
```

## Testing

Run the complete test suite:

```bash
pytest -q
```

Tests cover configuration validation, market-data handling, environments, PPO integration, training, resume behavior, evaluation, persistence, artifact generation, and reporting.

## Research workflow

The intended workflow is:

```text
idea
  |
  v
fast experiment on 1h data
  |
  v
promising?
  |
  +-- no --> reject or revise
  |
  +-- yes
        |
        v
full experiment on 5m data
        |
        v
multi-seed confirmation and comparison
```

The 1-hour interval is primarily an experimental and development tool. Final conclusions about the trading agent should be based on the target 5-minute setup.
