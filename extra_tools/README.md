# Extra tools

Run commands from the repository root with the project virtual environment.

| Directory | Purpose |
|---|---|
| `analysis/` | Grid rankings, seed comparisons and detailed path reports. See [GRID_ANALYSIS.md](analysis/GRID_ANALYSIS.md). |
| `diagnostics/` | Policy probability distributions and evolution across checkpoints. |
| `maintenance/` | Cleanup, name auditing/renaming and [latest-only cold storage](maintenance/COLD_STORAGE.md). |

Examples (use `--help` to inspect available options):

```bash
python extra_tools/analysis/ppo_analyze_grid.py --help
python -m extra_tools.diagnostics.policy_probability_diagnostic --help
python -m extra_tools.diagnostics.policy_probability_evolution --help
python extra_tools/maintenance/clean_training.py --help
python extra_tools/maintenance/audit_run_names.py --help
python extra_tools/maintenance/rename_runs.py --help
python extra_tools/maintenance/cold_storage.py --help
```

Historical rename audits and recovery journals stay in `maintenance/run_rename/`.
They are data, not Python tools; do not move or rewrite them during code cleanup.
No legacy entry-point wrappers remain at the top level. Update local shortcuts
or external scripts that still use the old paths. Analysis behavior and report
output directories are unchanged by this reorganization.

Cold storage includes ignored project files and a full PostgreSQL dump. Successful
`sync` keeps one verified external generation and automatically removes obsolete
managed content. Live capture/restoration excludes guarded queue and maintenance
operations. See the linked guide for prerequisites, local state, recovery and
remaining restore rehearsals; `verify` checks hashes, not database restoration.

## Shared stage-one configurations

Different grids may contain physical copies of the same run configuration.
Copies must have the same run name and identical normalized configuration,
including continuation source, data, devices and artifact settings. YAML comments,
key order and equivalent representations may differ. Different normalized values
under one name are a conflict, not a new experiment.

The name audit counts physical `configs`, `unique_config_names` and
`shared_config_names` separately. Its JSON stores every path and raw hash in
`config_copies`; CSV output has one row per physical copy. Rename planning
revalidates the complete inventory and every copy before changing anything. It
updates one database run per identity, all physical files and their queue paths.
An added, removed or edited copy after the audit requires a new audit. Existing
single-file audits are accepted only if their exact inventory is still current.

Grid Coverage merges matching copies across queues into one planned run while
retaining queue membership. Analysis merges matching names between selected
queues. One queue must still contain each run name only once. Completed runs
remain shared database records; copying a config does not create another run.
The cleanup tool continues to preserve configs, including all shared copies.
