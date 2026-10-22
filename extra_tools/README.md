# Extra tools

Run commands from the repository root with the project virtual environment.

| Directory | Purpose |
|---|---|
| `analysis/` | Grid rankings, seed comparisons and detailed path reports. See [GRID_ANALYSIS.md](analysis/GRID_ANALYSIS.md). |
| `diagnostics/` | Policy probability distributions and evolution across checkpoints. |
| `maintenance/` | Cleanup, name auditing/renaming and cold storage. |

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
