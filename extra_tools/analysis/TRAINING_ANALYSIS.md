# Training metrics analysis

`ppo_analyze_training.py` is a separate, read-only stage-one diagnostic tool. It uses the adjacent `ppo_analyze_grid.py` for queue loading, final evaluation/ancestry validation, and scientific matching. It never trains a model, changes hardware, modifies the database, or removes reports.

## Quick start: grid3

Run from the project root with the project virtual environment active:

```bash
python extra_tools/analysis/ppo_analyze_training.py \
  --queue grid3_s1 \
  --compare-by ent_coef gae_lambda \
  --reference ent_coef=0.0002 \
  --reference gae_lambda=0.85
```

This selects all eligible completed runs declared in grid3, not just a shortlist. Shared config copies resolve to the same DB run and are analyzed once. Run names are not parsed to infer parameters.

## Selection and comparisons

Choose one scope:

```bash
# Every eligible stage-one run; walk-forward runs are outside this tool's scope.
python extra_tools/analysis/ppo_analyze_training.py --all-runs

# Multiple queues, deduplicated by run identity.
python extra_tools/analysis/ppo_analyze_training.py --queue grid1_s1 --queue grid3_s1

# Individual run IDs. Ancestors are loaded for validation, not silently added to plots.
python extra_tools/analysis/ppo_analyze_training.py --run-id 3939 --run-id 3608

# Compare seeds while holding other settings and ancestry fixed.
python extra_tools/analysis/ppo_analyze_training.py --queue grid1_s1 --queue grid1_s2 \
  --compare-by seed --reference seed=1

# Restrict the population; assignments are JSON values and combine with AND.
python extra_tools/analysis/ppo_analyze_training.py --queue grid3_s1 \
  --where n_epochs=9 --where 'architecture=[768,768,768,768]' \
  --compare-by gae_lambda --reference gae_lambda=0.85

# Compare progression within equivalent paths; this is not an independent replicate.
python extra_tools/analysis/ppo_analyze_training.py --queue grid3_s1 \
  --compare-by stage --reference stage=2
```

Aliases: `seed`, `architecture`, `learning_rate`, `gamma`, `n_epochs`, `ent_coef`, `gae_lambda`, `stage`. Scientific dotted paths supported by the grid analyzer can also be used, e.g. `ppo.vf_coef`. Architecture values are JSON lists. Stage is zero-based: fresh=0, first continuation=1.

`--compare-by` can contain multiple axes. If using `--reference`, specify every comparison axis. Otherwise all available variant pairs are compared. All other scientific settings, final evaluation conditions, seed (unless varied), and complete ancestry are held fixed. Unmatched or ambiguous cells are reported; no nearest-neighbor substitution is made. Stage comparisons match equivalent roots and measure progression.

`--group-by architecture n_epochs` changes descriptive grouping, not matching rules. Groups also retain fixed-condition cohorts from the grid analyzer; metrics from incompatible datasets/evaluation conditions are not pooled. Group counts can differ. Associations are descriptive Pearson correlations, not statistical evidence of causality or independent-replicate significance.

## Reports

Each invocation writes a new timestamp directory under `artifacts/training_analysis/`:

- `report.html`: standalone offline report with embedded Plotly and all selected run summaries/curves.
- `report.json`: compact report for sharing; per-run key tail diagnostics, final TRAIN/VAL results, group summaries, matched pair differences, coverage, issues and settings. No rollout time series.

HTML tabs:

| Tab | Purpose |
|---|---|
| Overview | VAL return vs a selected tail diagnostic, all selected runs sorted by VAL return, diagnostic flags. Click a scatter point to inspect its run. |
| Run metrics | Search by ID/name, choose any stored numeric metric or derived diagnostic. Post NMSE, EV, bias, prediction scale and correlation also show their pre-update counterparts. Tables include all/head/tail statistics. |
| Paths | Diagnostic tail medians and TRAIN/VAL outcomes across selected ancestors up to the chosen run. Missing selected ancestors are explicitly listed. Stage+1 is an ordinal label, not an inferred number of data epochs. |
| Matched comparisons | B vs A curves on each run's local step axis and differences of tail medians, plus final evaluation deltas. |
| Groups | Descriptive tail distributions, VAL leaders and correlations with VAL return within fixed-condition groups. |
| Quality & definitions | Selection/exclusion details, missing fields, invalid values, ancestry/matching coverage, settings and interpretation notes. |

Every metric statistic uses all stored rollout records. Head/tail windows use `ceil(record_count * tail_fraction)` records, default 10%. Tail is chronological, not selected to maximize a result. Metrics missing within a window are omitted from statistics, with the finite count reported. They are never replaced with zero.

Curves use at most 80 equal-index bins per run by default. Each bin retains its median, minimum and maximum, with raw first/last endpoints also plotted. The x coordinate is the median run_step in the bin. Charts do not preserve exact spike timing within a bin; increase `--chart-points` to the stored record count for an unbinned display. This does not alter statistics. Large selections can produce large HTML files, while JSON omits curves.

Options:

```bash
python extra_tools/analysis/ppo_analyze_training.py --queue grid3_s1 \
  --tail-fraction 0.1 --chart-points 624 --json-detail full
```

`--json-detail full` adds all metric statistics but still omits curves. `--output-dir PATH` changes the parent report directory. No old report is deleted.

## Critic metrics

Targets are stored rollout-buffer returns (bootstrapped GAE targets). Pre-update predictions are those collected during rollout. Post-update predictions evaluate the same observations after PPO optimization against the same fixed targets. This measures in-sample fitting, not performance on unseen targets.

| Diagnostic | Definition / interpretation |
|---|---|
| `pre_mse` | `value_error_std² + value_error_mean²`, using the stored population moments. |
| `pre_nmse`, `post_nmse` | MSE divided by `value_target_std²`. A value of 1 equals the MSE of predicting the rollout target mean. That mean is an in-sample comparator, not a trading baseline. |
| `nmse_improvement` | Pre NMSE minus post NMSE on each rollout. Positive means lower fixed-target error after optimization. |
| `ev_improvement` | Post explained variance minus pre explained variance. |
| `pre_bias_scaled`, `post_bias_scaled` | Signed mean error / target standard deviation. Error is target minus prediction: positive means underestimation. |
| `pre_prediction_scale`, `post_prediction_scale` | Prediction standard deviation / target standard deviation. |
| `explained_variance` | `1 - Var(error)/Var(target)`. It does not detect a constant bias. Negative values indicate worse error variance than predicting a constant. |
| Target/prediction correlation | Association, not calibration. High correlation does not exclude bias or incorrect amplitude. |
| `value_loss` | PPO optimizer log, not necessarily the final fixed-target MSE. |

Normalization is undefined when target standard deviation is at most `--min-target-std` (default `1e-12`). Such values remain null and are counted; no epsilon is added to make them appear finite. Ratios may still be unstable near the threshold. Changing gamma/GAE changes the targets themselves: normalization aids comparison but does not make these identical prediction tasks.

## Policy and reward metrics

All stored numeric fields are available, including KL, clipping, entropy loss, policy gradient loss, value loss, total loss, learning rate, update count and rollout reward. `policy_entropy = -entropy_loss`; lower entropy alone is not proof of a bad or good policy. Reward is the training reward, not automatically trading return. `ep_reward` and `ep_len` summarize the completed-episode buffer; fresh runs may have no values before episode completion, and resumed runs may retain earlier episodes. These two fields are excluded from the unexpected-null flag, but nonfinite values are always reported.

## Diagnostic flags

These thresholds are descriptive prompts, not an automatic model score:

- tail median post NMSE > 1;
- tail median post EV < 0;
- absolute tail median signed scaled bias > 1 (opposite signs can cancel; also inspect distributions);
- tail median prediction scale < 0.1;
- tail median policy entropy < 0.05 nats;
- more than half of comparable updates worsen NMSE by more than `1e-10`.

Quality flags additionally identify missing/nonfinite metrics, missing end-of-run metric, non-increasing rollout sequence, negative standard deviations and inconsistent post-MSE moments. Duplicate/non-increasing run steps stop analysis. The presence of rows and a matching final step do not by themselves prove the expected logging cadence has no gaps.

## Data access and integrity

Metadata, final evaluations, metric counts and all metric rows are read within one PostgreSQL `REPEATABLE READ, READ ONLY` transaction. Metrics are loaded one run at a time; downsampled curves and full summaries are retained. The analyzer loads checkpoint ZIP metadata (`system_info.txt`) for saved device validation through the existing grid analyzer; it does not deserialize weights or import PyTorch. Therefore corresponding checkpoint files must be available. No market parquet or trajectory files are read.

Completed runs must pass the existing grid eligibility checks (including final TRAIN/VAL and ancestry). Excluded requested runs and reasons are reported. Unknown/missing parents or duplicate run identities stop selection. Filters apply after validation; ancestors remain validation context only. When training is active, the snapshot remains consistent, but disk/DB reads can compete with training resources. For a final report, use a completed queue.

A complete metric table is evidence of available observations, not evidence of a good critic. Compare critic fit, policy behavior and final evaluation outcomes together. Do not rank across different target scales using raw MSE/value_loss alone.

## Validation

```bash
python -m pytest -q tests/test_training_analysis.py
python -m pytest -q
```

Tests use synthetic records and a mocked read-only DB connection. Real PostgreSQL integration and the complete project suite must be run locally. The report includes the database snapshot timestamp, queue/config hashes for queue selections, analysis settings and tool version. Reports are not deterministic byte-for-byte because timestamps change.
