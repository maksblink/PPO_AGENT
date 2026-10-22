# Grid, seed and path analysis

Version 4.0.0. One entry point: `extra_tools/analysis/ppo_analyze_grid.py`.
Run from the project root using its virtual environment. Requirements are the
project's existing NumPy, pandas, PyArrow, matplotlib, SQLAlchemy and database
configuration. No new service is required.

## Modes

| Mode | Work performed | Required inputs |
|---|---|---|
| `grid` | Inventory, integrity, Pareto fronts, top-percent candidates, group summaries and optional seed comparison | `--all-runs` or repeated `--queue` |
| `full` | Grid analysis plus full selected paths and detailed VAL trajectories for selected stage pairs | Grid scope, `--compare-seeds`, detailed selection |
| `paths` | Selected paths and VAL diagnostics from an existing seed-comparison report; current DB/artifacts are verified | `--report`, detailed selection |
| `auto` (default) | Backward-compatible: grid unless detailed selections are supplied, then full | Same options as the resulting mode |

### Grid only

```bash
python extra_tools/analysis/ppo_analyze_grid.py --mode grid --all-runs --top-percent 1
```

Add `--compare-seeds 1 2` for matched-stage and complete-path seed comparisons.
Use repeated `--comparison-queue NAME` to specify the expected replay coverage.
`--selection-report PATH` freezes candidate names from a previous report instead
of recomputing the reference candidate selection on the current inventory.
Queues can be nested under `configs/search_queues`; names must be unique.

### Full report

```bash
python extra_tools/analysis/ppo_analyze_grid.py --mode full --all-runs \
  --compare-seeds 1 2 --top-percent 1 \
  --detail-run-id 101 --detail-run-id 202
```

IDs above are examples. Specify reference-seed IDs that have a unique matched
replay run. Each selected pair causes its complete available matched path to be
included, while detailed artifacts are read only for the selected stage pair.

### Paths only, without recomputing the grid

```bash
python extra_tools/analysis/ppo_analyze_grid.py --mode paths \
  --report artifacts/queue_analysis/SESSION/report.json \
  --detail-run-id 101 --detail-run-id 202
```

The source must be a grid/full report with a PASS seed comparison. A paths-only
report is an output, not a replacement source inventory. Current identities,
ancestry and final metrics are compared with the source before analysis proceeds.

### Reusable selection

Create a local JSON file containing only:

```json
{"reference_run_ids": [101, 202]}
```

Pass `--details-config /path/to/selection.json` instead of repeating IDs. IDs can
also be added using `--detail-run-id`; duplicates are removed. The package does
not include a private selection or change Git ignore rules.

`--detail-all-matched` explicitly selects every matched stage pair. It can read
many artifacts and generate large HTML files. Selected IDs are preferable for
routine diagnostics. Grid mode rejects trajectory selections rather than silently
ignoring them. Full mode requires an explicit selection.

## Reports and tabs

Each invocation writes `report.html` and `report.json` into a new timestamped
folder under `artifacts/queue_analysis/`. Override the parent with `--output-dir`.

- **Grid overview:** eligible-run inventory, integrity issues, cohort summaries,
  candidates, TRAIN/VAL Pareto fronts, frontier geometry, DD-budget winners,
  descriptive parameter comparisons and seed comparisons when requested.
- **Full paths:** stage-by-stage metrics for both seeds for selected paths.
- **VAL #reference / #replay:** return, drawdown and exposure plots; weekly/monthly
  return and costs; DD peak/trough/recovery; closed-trade statistics and top-trade
  attribution; position agreement and weekly/monthly return correlations.

In paths-only mode, the first tab identifies the source instead of recomputing
its grid. Without a detailed selection, the diagnostics tab explains how to
request one. Tabs work offline, support arrow keys/Home/End, and show all content
when JavaScript is disabled.

Detailed attribution includes excess return versus always-long, weekly correlation
with that benchmark, total fees/swaps and return after arithmetically subtracting
the five best weeks. The tool reports evidence; it does not automatically promote
candidates or generate claims of statistical significance.

JSON retains computed summaries, candidate/path metrics and artifact hashes.
It omits raw per-bar trajectories, individual trade records and plot images.
Integrated reports store details in `path_diagnostics`; this is null when not
requested. Paths-only JSON has `metadata` and `path_diagnostics`.

## Candidate selection and comparison

`--top-percent` defaults to 1. Counts round upward within each eligible cohort,
including all cutoff ties. Shortlists combine top VAL return, top VAL balanced
score, Pareto and configured DD-budget selections, with duplicates removed.
Primary fronts pool architectures and n_epochs only within comparable cohorts.
Seeds retain separate ranking cohorts; the replay subset is not a full-grid
comparison of architecture or hyperparameter quality.

Pareto means no other run improves one objective without worsening the other.
Supported-front and endpoint-chord labels are geometric descriptors, not evidence
of robustness. Common-cell group summaries are supplementary descriptions, not
an instruction to choose the parameter family with the highest average.

Seed comparison matches scientific conditions and ancestry, checks expected queue
coverage, retains stage alignment and distinguishes a fixed reference stage from
a retrospectively selected replay optimum. Stages of one chain are dependent.
Two seeds do not establish statistical robustness. Selection on known VAL remains
selection on known VAL, even when the seed changes.

## Integrity and read-only operation

The grid reads one REPEATABLE READ, READ ONLY database snapshot. Detailed analysis
uses a second read-only snapshot and verifies selected final TRAIN/VAL metrics,
identities, source checkpoints and device configuration against the grid/source
report. These are verified snapshots, not one transaction spanning all file reads.

Checkpoint ZIPs are opened for metadata only. GPU Enabled records availability;
it is not independent proof of the actual execution device. No models are
loaded, no training occurs and no hardware settings change.

For detailed VAL pairs, artifact identities and core metrics are checked against
the DB. Trajectory length, timestamps, execution-index continuity, final return,
additive drawdown, exposure, trade counts and cost accounting are validated.
With forced final closing, summed closed-trade net return must match final equity.
Missing or inconsistent required data causes a stop. Files are hashed and checked
for size/mtime changes during reading. Do not clean selected runs concurrently.

All reported returns are additive on stake, not compounded account-equity returns.
Raw values use fractions: 0.01 is one percentage point. DD starts from zero.
Calendar periods use UTC; boundary periods may be partial. Period drawdown uses
the full VAL high-water mark, not a reset at each calendar boundary.

Top winning-trade shares use total positive trade PnL as denominator. Subtracting
best trades/weeks is arithmetic attribution, not a counterfactual simulation.
High position agreement can result from both policies almost always being long.
Correlations are descriptive, not statistical significance tests.

## Retention

Old reports are retained by default. `--clean-old-reports` is explicit, limited
to recognized superseded grid reports under the default output parent, and is
not available in paths mode. Reports containing seed comparisons are retained.
Keep any historical report used to freeze a selection.

## Migration and tests

This version replaces both previous analysis scripts with this single script.
The standalone command changes from `ppo_analyze_seed_paths.py --report ...
--reference-run-id ID` to `ppo_analyze_grid.py --mode paths --report ...
--detail-run-id ID`. Existing grid/full commands remain valid through auto mode.
The old SEED_PATH_ANALYSIS.md is merged here. The obsolete
extend_ne3_midpoint_lrs.py is removed by the package installer at user request.

```bash
python -m pytest -q tests/test_grid_seed_analysis.py \
  tests/test_seed_path_analysis.py tests/test_grid_path_integration.py
```

Tests use synthetic data and mocked reads, without database or hardware access.
Actual integrity verification occurs when the tool runs on your local data.
