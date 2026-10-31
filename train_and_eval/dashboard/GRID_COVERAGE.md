# Grid coverage

Select one **Grid / queue** manifest. Its listed configurations define the complete expected scope. Other queues and database runs do not contribute expected architectures, learning rates, gamma values or stages.

- Sparse grids are supported: combinations absent from the manifest are not missing runs.
- Each declared path has its own stage count. Continuation depth is not a measured number of data epochs.
- Shared configurations can reuse a matching database run regardless of the grid in which it was first executed.
- An unreadable or conflicting plan stops coverage reporting rather than presenting partial data as complete.
- Families keep non-axis settings separate, including data ranges, reward, costs and evaluation protocol.
- Explicit filters cover PPO epochs, seed, entropy coefficient, GAE lambda and architecture. Cells show gamma versus learning rate for one selected variant.
- Ready means completed status, full recorded requested budget when available, matching config and parent, one final checkpoint record, and completed final TRAIN and VAL evaluations. This view does not check checkpoint files on disk.
- The queue-wide caption counts all declared runs. Metric cards, charts and CSV exports reflect the currently selected filters.
- Unfinished planned paths and stage details show pending work and integrity problems. Missing parent configurations remain integrity issues.

## Other dashboard views

Grid Analysis exposes entropy and GAE in its default filters and grouped statistics. Users may change grouping explicitly; overall statistics still summarize all selected runs. The main explorer already supports these settings through flattened configuration fields, column selection and numeric filters. Scatter plots now include them in hover details.

Performance views retain their own filters; choosing a queue in Grid Coverage does not silently filter other tabs. Pareto dominance calculations and checkpoint-based lineage are unchanged.

All views are read-only with respect to the experiment database and artifacts.
