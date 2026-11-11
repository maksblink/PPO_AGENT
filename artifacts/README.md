# Runtime artifacts

This directory stores generated runtime artifacts produced by PPO_AGENT.

Generated artifacts are **not committed to Git**. Only this README and the directory `.gitignore` are tracked.

## Report and session directories

| Directory | Contents |
| --- | --- |
| `runs/<run ID>/` | Owned checkpoints, run reports and evaluation artifacts. |
| `queue_analysis/<timestamp>/` | Grid/comparison/path HTML and JSON reports. |
| `training_analysis/<timestamp>/` | Training metric HTML and JSON reports. |
| `optuna_propositions/<timestamp>/` | Optuna HTML/JSON, including preparation details when requested. |
| `search_queues/<queue name>/<session ID>/` | Queue plans, events, status and managed hardware telemetry. |
| `walk_forward/` | Study plans, attempts and aggregate reports. |

Optuna batch timestamps are UTC with microseconds, without a random suffix for
new reports. The same identifier locates the generated config and queue trees
under `configs/`. Older suffixed report identifiers remain valid and unchanged.
Other tools keep their own session identifier formats.

Reports are derived outputs, not additional database runs. Removing a report does
not remove training results or prevent Optuna from importing them again. Conversely,
removing a run from the database removes its evidence from future history imports.
See [Optuna propositions](../train_and_eval/optuna/README.md) and
[managed queues](../MANAGED_QUEUE.md) for the different lifecycles.

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
│   ├── evaluation_metrics_<scope>.csv
│   ├── evaluation_curves_<scope>.png
│   └── summary.json
└── evaluations/
    └── 00000001/
        ├── trajectory_<scope>.parquet
        ├── trade_events_<scope>.parquet
        ├── metrics_<scope>.json
        ├── equity_curve.png
        ├── drawdown_curve.png
        ├── market_and_exposure.png
        ├── cumulative_costs.png
        ├── trade_returns.png
        └── holding_times.png
```

Exact filenames can vary with the run configuration and artifact settings.
Trade-return and holding-time plots can legitimately be absent when there are no
applicable trades. Empty trade-event tables can be valid. Assess completeness
against the configured writers and evaluation scope, not one reference run's
file list. Readability/hash checks do not establish economic correctness or
visual chart quality.

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
trajectory_<scope>.parquet
trade_events_<scope>.parquet
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

### `evaluation_curves_<scope>.png`

Validation metrics across checkpoints, including best and final evaluation markers.

## Artifact configuration

Artifact behavior is controlled independently from console logging.

Example:

```yaml
artifacts:
  training_metrics:
    enabled: true
    every_steps: 2048

  evaluation_trajectory:
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

Remove experiments through the maintained cleanup tool so database records and
owned artifact directories stay consistent. Inspect selected IDs and descendants
before applying a plan:

```bash
python extra_tools/maintenance/clean_training.py clean-first-stage --run-id <RUN_ID> --dry-run
```

After reviewing the plan, use the same command without `--dry-run` and follow its
confirmation. Do not delete the whole `artifacts/` tree while retaining database
references. Cleanup preserves configs; inspect and remove only corresponding
unused config/queue batches separately. Never delete a shared parent merely
because a removed experiment referenced it.

Derived analysis reports and finished queue sessions have separate lifecycles;
review their exact directories before deleting them. Preserve this README and
project control/lock files. Do not clean artifacts during training, evaluation,
backup or report rebuilding.

Evaluation scope suffixes are `train`, `val`, `reference` and `test`.
Stage one saves paired evaluations of each assessed checkpoint; all evaluation
plots carry the same suffix. PPO update diagnostics are not frozen TRAIN metrics.


## Backup boundary

The [cold-storage tool](../extra_tools/maintenance/COLD_STORAGE.md) captures this
entire directory, including ignored files, together with the rest of the project
and a separate full PostgreSQL dump. Run/checkpoint files and database records
must remain consistent: stop writers during capture and use the maintenance
cleanup tools when removing runs.

Backup control files and local scan captures live in the project-root
`.cold_storage/`, not under `artifacts/`. The external managed backup retains only
the latest verified generation after successful synchronization. It is not a
historical archive or a directly executable project tree. Restore files to a new
Linux directory and restore the corresponding database before using the restored
project. Do not delete local run artifacts merely because a checksum verification
passed; a complete isolated restore rehearsal remains a separate check.
