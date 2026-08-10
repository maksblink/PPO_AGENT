# Runtime artifacts

This directory stores generated runtime artifacts produced by PPO_AGENT.

Generated artifacts are **not committed to Git**. Only this README and the directory `.gitignore` are tracked.

## Directory layout

Artifacts are organized by run ID:

```text
artifacts/
└── runs/
    └── 00000001/
        ├── checkpoints/
        ├── reports/
        └── evaluations/
```

A typical run can contain:

```text
artifacts/runs/00000001/
├── checkpoints/
│   ├── checkpoint_*.zip
│   └── ...
├── reports/
│   ├── training_metrics.csv
│   ├── training_curves.png
│   ├── validation_metrics.csv
│   ├── validation_curves.png
│   └── summary.json
└── evaluations/
    └── 00000001/
        ├── trajectory.parquet
        ├── trade_events.parquet
        ├── metrics.json
        ├── equity_curve.png
        ├── drawdown_curve.png
        ├── market_and_exposure.png
        ├── cumulative_costs.png
        ├── trade_returns.png
        └── holding_times.png
```

Exact filenames can vary with the run configuration and artifact settings.

## Checkpoints

PPO checkpoints are persistent model artifacts used for:

- periodic snapshots during training
- best-checkpoint selection
- final checkpoint storage
- resume training
- historical evaluation replay

Checkpoint metadata is recorded in PostgreSQL, including run association and integrity information.

## Training metrics

Training metrics are persisted in PostgreSQL and can also be exported into run-level report files such as:

```text
training_metrics.csv
training_curves.png
```

Training metrics are important because they generally cannot be reconstructed without rerunning training.

## Evaluation metrics

Evaluation summary metrics are persisted in PostgreSQL.

Examples include:

- balanced score
- agent return
- maximum drawdown
- profit factor
- win rate
- market exposure
- number of round trips
- always-long and always-short benchmark results

The database remains the authoritative persistent record for completed evaluations.

## Evaluation trajectories

When enabled by configuration, detailed validation trajectories are stored as Parquet files:

```text
trajectory.parquet
trade_events.parquet
```

These files contain step-level information used to build detailed evaluation reports and plots.

Keeping trajectory artifacts makes later report generation much faster because the evaluation does not need to be replayed from the checkpoint.

## Rebuildable artifacts

Plots and report files are derived artifacts.

Regenerate reports for a run:

```bash
python -m train_and_eval.runs report \
  --run-id 1
```

Regenerate one evaluation:

```bash
python -m train_and_eval.runs report \
  --run-id 1 \
  --evaluation-id 1
```

If trajectory artifacts already exist, reports can be rebuilt directly from them.

If trajectory artifacts are missing, PPO_AGENT can replay the archived checkpoint over the historical validation range and rebuild them.

Historical replay:

- requires a clean Git working tree
- does not create a new Evaluation row
- verifies replayed metrics against the persisted historical evaluation before accepting regenerated artifacts

## Evaluation plots

### `equity_curve.png`

Agent equity versus benchmark equity over the validation period.

### `drawdown_curve.png`

Drawdown over time using the standard negative drawdown convention.

### `market_and_exposure.png`

Market price together with rolling agent market exposure.

### `cumulative_costs.png`

Cumulative fees, swap costs, and total trading costs.

### `trade_returns.png`

Distribution of round-trip trade returns.

### `holding_times.png`

Distribution of trade holding durations.

## Run-level plots

### `training_curves.png`

Training diagnostics such as rollout reward, entropy, explained variance, KL divergence, clip fraction, losses, and learning rate.

### `validation_curves.png`

Validation metrics across checkpoints, including best and final evaluation markers.

## Artifact configuration

Artifact behavior is controlled independently from console logging.

Example:

```yaml
artifacts:
  training_metrics:
    enabled: true
    every_steps: 2048

  validation_trajectory:
    mode: all

  plots:
    during_run: false
```

Typical trajectory modes are:

```text
disabled
final_only
all
```

The exact allowed values are defined by the current run configuration schema.

## PostgreSQL versus filesystem artifacts

PPO_AGENT separates persistent structured metadata from larger generated files.

PostgreSQL stores structured records such as:

```text
runs
checkpoints
evaluations
training_metrics
```

The filesystem stores larger or derived artifacts such as:

```text
PPO checkpoint ZIP files
Parquet trajectories
CSV exports
PNG plots
JSON summaries
```

This keeps the database suitable for querying while larger experiment outputs remain simple filesystem artifacts.

## Cleaning artifacts

Runtime artifacts can be removed without deleting the tracked documentation files:

```bash
find artifacts \
  -mindepth 1 \
  -maxdepth 1 \
  ! -name '.gitignore' \
  ! -name 'README.md' \
  -exec rm -rf {} +
```

This preserves:

```text
artifacts/.gitignore
artifacts/README.md
```

Deleting filesystem artifacts does not automatically delete the corresponding PostgreSQL run metadata.
