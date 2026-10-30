# Grid, flexible comparison, seed and path analysis

Version 5.0.0. Entry point: `extra_tools/analysis/ppo_analyze_grid.py`.
Run from the project root with its virtual environment. Analysis reads PostgreSQL
and existing artifacts; only new HTML/JSON reports are written. No training,
model loading, hardware changes or network requests occur.

## Two analysis interfaces

- **Flexible comparison:** `--compare-by FIELD [FIELD ...]`. Compare any supported
  scientific config field or combination, with exact matching of every other
  setting. This is the recommended interface for new parameter experiments.
- **Legacy grid/seed interface:** omit `--compare-by`. Existing commands and old
  report identities continue to work. In this interface entropy and GAE remain
  fixed cohort conditions, as in v4. Do not interpret its separate fronts as a
  global comparison across entropy/GAE variants.

Both accept `--all-runs` OR repeated `--queue NAME_OR_PATH`. Queue names are
resolved recursively. Identical run names across queues are deduplicated;
conflicting configs are rejected. The selected scope must include ancestors.
`--all-runs` inventories database runs, not hypothetical missing configurations.
Queue scope can identify missing planned runs.

## Flexible comparison options

| Option | Meaning |
|---|---|
| `--compare-by A B` | Treat the joint values of A and B as variants. Match all remaining scientific conditions and stage. |
| `--where FIELD=JSON_VALUE` | Exact filter; repeat for AND. Applied after integrity/ancestry checks. |
| `--reference FIELD=JSON_VALUE` | Compare each variant against a reference instead of every pair of variants. Specify every compared field. |
| `--rank-across A B ...` | Allow these fields to vary within a ranking/Pareto cohort. Independent of pair matching. Defaults to compare-by fields. |
| `--group-by A B ...` | Additional descriptive group summaries, upper tails and leaders. Does not relax pair matching. |
| `--top-percent P` | Candidate selection per ranking cohort, rounded up including cutoff ties. Does not limit analysis scope. |
| `--comparison-export summary` | Default compact JSON: coverage, effects, rankings, group tails, candidate paths; omits all individual pairs. |
| `--comparison-export pairs` | Also include every matched pair and its metrics/deltas. |
| `--comparison-export all` | Also include metrics for every selected run. No raw trajectories. |
| `--trajectory-pair A_ID:B_ID` | Detailed VAL artifacts for an explicitly selected matched pair; repeat as needed. Requires auto/full mode. |

Aliases: `seed`, `stage`, `architecture`, `learning_rate`, `gamma`, `n_epochs`,
`ent_coef`, `gae_lambda`. Architecture values are JSON arrays, e.g.
`--where 'architecture=[768,768,768,768]'`.

Use dotted paths for other scientific config fields, for example
`ppo.clip_range`, `ppo.vf_coef`, `ppo.activation`,
`environment.exposure_penalty`, `training.duration_amount`.
Fields must exist in every eligible config; misspellings stop analysis. Identity,
logging and output fields are not scientific axes. Use `stage` rather than
`continuation.mode` or `continuation.source_run`.

Values use JSON syntax, including quoted strings:
`--where 'ppo.activation="tanh"'`. Array values are exact arrays, not lists of
allowed choices. Filters are equality/AND; arbitrary expressions and ranges are
not evaluated. Restrict other scopes through queues.

## Practical examples

### Compare entropy and GAE, with one baseline

```bash
python extra_tools/analysis/ppo_analyze_grid.py --mode grid --queue grid3_s1 \
  --compare-by ent_coef gae_lambda \
  --reference ent_coef=0.0002 --reference gae_lambda=0.85 \
  --rank-across architecture learning_rate gamma n_epochs ent_coef gae_lambda stage \
  --group-by architecture n_epochs --top-percent 1
```

With the complete 576-run grid, this selects four variants and three baseline
contrasts of 144 matched stages each (432 pairs total), if all checks pass.
Each contrast holds architecture, LR, gamma, PPO epochs, seed and stage fixed.
The requested ranking pools all 576 within compatible evaluation conditions and
seed. A stage is one run in a continuation chain; it is not `ppo.n_epochs`.

Omit `--reference` for all six contrasts between four variants. For a single-axis
effect, use `--compare-by ent_coef`; GAE then stays fixed in each matched pair.

### Restrict analysis to an architecture and final stage

```bash
python extra_tools/analysis/ppo_analyze_grid.py --queue grid3_s1 \
  --compare-by ent_coef gae_lambda \
  --where 'architecture=[768,768,768,768]' --where n_epochs=9 --where stage=2
```

Filters do not remove required ancestors before validation. Candidate paths retain
all available stages from the original eligible scope, even outside the filters.

### Global rankings and comparisons across all available runs

```bash
python extra_tools/analysis/ppo_analyze_grid.py --all-runs \
  --compare-by ent_coef gae_lambda --where seed=1 \
  --rank-across architecture learning_rate gamma n_epochs ent_coef gae_lambda stage \
  --group-by architecture n_epochs --top-percent 1
```

Global rankings include old and new variants. Matched contrasts use only their
common cells; configurations available in only one variant are listed as
unmatched. A shared grid1/grid3 run remains one database observation. Rankings do
not assign a run exclusively to a grid or claim that every leader is new.

### Seeds, PPO epochs, or another parameter

```bash
python extra_tools/analysis/ppo_analyze_grid.py --all-runs \
  --compare-by seed --reference seed=1 --group-by n_epochs architecture

python extra_tools/analysis/ppo_analyze_grid.py --queue grid3_s1 \
  --compare-by n_epochs --reference n_epochs=3 --group-by ent_coef gae_lambda

python extra_tools/analysis/ppo_analyze_grid.py --all-runs \
  --compare-by ppo.clip_range ppo.vf_coef --where seed=1
```

No special case is needed for seed comparisons in this interface. If selected
fields never vary, there are no contrasts; group summaries/rankings still exist.

### Detailed comparison of two matched runs

Read IDs and A/B direction from the report's matched-pair table, then:

```bash
python extra_tools/analysis/ppo_analyze_grid.py --mode full --queue grid3_s1 \
  --compare-by ent_coef gae_lambda --trajectory-pair 101:202
```

IDs are examples. The tool rejects pairs that do not match the selected scientific
contrast. It rechecks identities, ancestry and final TRAIN/VAL metrics against
PostgreSQL before reading VAL trajectories. Reports include return/DD/exposure
plots, weekly/monthly attribution, fees/swaps, DD recovery, trade concentration,
position agreement and correlations. These diagnostics are no longer restricted
to seeds. Without an explicit pair, generic grid analysis reads no trajectories.

## Matching, ranking and interpretation

Matching removes only the selected axes from scientific configurations. Seed and
stage remain fixed unless explicitly selected. Remaining data, environment,
training, PPO, evaluation and continuation settings must agree, as must evaluation
windows/signatures and saved device categories. Full ancestor signatures are
matched too. Stage comparisons use equivalent fresh roots and compare progression;
they are not independent replicas.

Data hashes, evaluation intervals/policies and actual device categories remain
protected comparability conditions, even when a related config field is selected.
The tool may report no pairs for incompatible datasets/evaluations. It will not
silently pool unrelated evaluation tasks to produce a ranking.

Inherited integrity checks currently support stage-one fresh/resume chains with
constant scientific settings along each chain. Walk-forward runs and chains that
change scientific settings mid-path remain excluded. This is a supported-scope
limit, not an assertion that such experiments are invalid.

Repeated scientific cells are reported as ambiguous and excluded from pairing.
They are retained as distinct runs in rankings; repeated copies of the same queue
run name are deduplicated before database reads. Missing matches are counted per
contrast, with unmatched run IDs in JSON. Never treat unmatched data as zero.

Effect direction is B minus A. Higher return/balanced score and less-negative DD
are better. Exposure, trade counts and other behavioral metrics remain descriptive.
Ranking cohorts are independent from matching cohorts. Broader `--rank-across`
settings can produce a global front without confounding paired effects.

Groups show medians, upper quantiles and leaders; they may cover unequal search
spaces. Candidate selection is the union of VAL Pareto, top VAL return and top
VAL balanced score. Generic mode does not apply legacy fixed DD-budget selections.
All stages and all contrasts are dependent observations. Reports make no causal
or statistical-significance claim. Known-VAL selection is still selection on VAL.

## Reports, tabs and retention

New timestamped directory under `artifacts/queue_analysis`, or `--output-dir`:
`report.html` and `report.json`. HTML has tabs for matched comparisons, rankings,
variant/group upper tails, candidate paths, integrity and requested diagnostics.
It retains all selected run rows and all matched pairs. JSON defaults to compact
summaries plus full candidate-path metrics and small parameter/ID mappings.
No raw per-bar trajectories or plot images are exported to JSON.

Old reports are retained. Generic comparisons reject `--clean-old-reports` because
arbitrary filters and pooling choices do not establish that a report supersedes
another. Existing generic reports are not sources for legacy `--mode paths`.
Rerun the generic command with `--trajectory-pair` to add diagnostics.

## Legacy seed/path interface

Existing modes and options remain available without `--compare-by`:

```bash
python extra_tools/analysis/ppo_analyze_grid.py --mode grid --all-runs --top-percent 1

python extra_tools/analysis/ppo_analyze_grid.py --mode full --all-runs \
  --compare-seeds 1 2 --detail-run-id 101

python extra_tools/analysis/ppo_analyze_grid.py --mode paths \
  --report artifacts/queue_analysis/SESSION/report.json --detail-run-id 101
```

`--comparison-queue` specifies expected replay coverage. `--selection-report`
freezes prior candidates. `--details-config` accepts `{"reference_run_ids":[101]}`.
`--detail-all-matched` reads all matched-stage artifacts and can be expensive.
These options cannot be mixed with the generic interface. Existing source reports
must have a PASS seed comparison. Legacy `--clean-old-reports` restrictions remain.

## Integrity and numerical conventions

One REPEATABLE READ, READ ONLY snapshot for grid integrity and metrics. Detailed
artifacts use a second verified read-only snapshot, not a single transaction
spanning all file reads. Checkpoint ZIPs are opened for metadata only.
GPU Enabled records availability, not independent proof of execution device.
Do not clean selected runs concurrently.

Detailed VAL artifacts are reconciled for IDs, final return, DD, exposure, trades,
length, timestamps, execution-index continuity and costs. Files are hashed and
checked for size/mtime changes. Forced-final-close trade PnL must reconcile.

Returns are additive on stake, not compounded account equity. Raw 0.01 = 1 pp.
DD begins at zero. Calendar buckets use UTC and may be partial. Period DD uses
the full VAL high-water mark. Removing best weeks/trades is arithmetic attribution,
not a simulation of a changed strategy. High position agreement can reflect two
nearly always-long policies. Two seeds do not establish statistical robustness.

## Tests

```bash
python -m pytest -q tests/test_flexible_grid_comparison.py \
  tests/test_grid_seed_analysis.py tests/test_seed_path_analysis.py \
  tests/test_grid_path_integration.py
```

Synthetic and mocked tests, no database/hardware access. Run the full project
suite before committing. Local report generation performs real-data validation.
