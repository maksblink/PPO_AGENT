# Grid analysis and seed comparisons

Run from the repository root using the project virtual environment:

```bash
python extra_tools/ppo_analyze_grid.py --all-runs
```

This inventories saved runs, checks stage-one ancestry, completed final
evaluations and checkpoint device metadata, and generates local HTML plus
compact JSON in `artifacts/queue_analysis/<UTC timestamp>/`.

## Compare two seeds

```bash
python extra_tools/ppo_analyze_grid.py \
  --all-runs --compare-seeds 1 2 \
  --comparison-queue replay_queue_a \
  --comparison-queue replay_queue_b
```

Replace example queue names with existing manifest stems. Discovery is recursive.
Expected queues make completely absent or ineligible replay runs visible.
Without them the status is `OBSERVED RUNS ONLY`, not proof that a planned replay
is complete. `INCOMPLETE` preserves a report with explicit issues; inspect the
readiness rather than treating report creation as validation success.

Pairing uses scientific settings, architecture, LR, gamma, PPO epochs, stage,
data hash, evaluation ranges/policy and device category. Run and evaluation
seeds may differ; other evaluation signature fields must match. Ancestry must
map parent-to-parent. Ambiguous duplicate cells and their descendants are
excluded from paired comparisons.

Candidate selection runs separately for each seed and comparable cohort. It
combines top 1% VAL return and score (ceiling, exact cutoff ties), the VAL
Pareto front and illustrative DD budgets. The replayed subset is selected, so
its top 1% has a different denominator from the full original grid.

Reference candidates are recomputed using only the reference seed. To freeze
names from a previous selection, supply its JSON with
`--selection-report /path/to/original/report.json`. Its SHA256 is recorded.
Current database metrics are used for both seeds; this option fixes candidate
membership, not historical metrics.

## Reading the comparison

- Original candidates: exactly the same stage across seeds.
- Full paths: every available stage, TRAIN/VAL metrics and parent-child deltas.
- Locked stage: replay performance at the best stage chosen on the reference
  seed only; exact ties are retained.
- Replay-best stage: explicitly labelled post-selection on the replay seed.
- Worst-of-two candidate rankings: exploratory prioritization using both seeds.
- Missing or incompatible matches: explicit integrity issues.

Positive deltas mean better return/score or less-negative DD. Raw JSON metrics
use stored fractions; 0.01 equals one percentage point for return/DD/exposure.
Undefined profit factor remains null. Exposure and trading activity are
descriptive, not a combined quality score.

Two seeds on the same periods do not establish out-of-sample performance.
Linked stages are not independent samples. This tool compares final metrics,
not checkpoint tensors or trajectories for bitwise reproducibility.

## Files and access

Database reads use a REPEATABLE READ, READ ONLY transaction. Checkpoint ZIPs
are opened only for system metadata. GPU availability metadata is considered
alongside configured devices; it is not independent execution proof. No model
deserialization, market-data loading or training is performed.

HTML retains full eligible-run tables. Compact JSON contains selected
candidates, paths, fronts, distribution tails and seed comparisons. Matched
path metrics are stored in pairs without repeating them in chain lists.
HTML needs no external scripts or network assets.

Older reports are retained by default. Explicit `--clean-old-reports` only
removes recognized superseded reports in the default analysis directory.
Reports containing seed comparisons are retained. Preserve the historical
selection report if it is needed as an audit record.

## Verification

```bash
python -m pytest -q tests/test_grid_seed_analysis.py
```

Tests use synthetic snapshots and temporary files. They do not connect to a
database or access hardware. Run the full project test suite before committing.

Based on the previously delivered version 2.2.0; current version: 3.0.0.
