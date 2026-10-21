# Hardware-managed training queues

`run_managed_queue.py` prepares and restores the laptop's hardware settings.
It invokes `run_search_queue.py`, which resolves dependencies and launches
separate worker processes. Runs use the project's configured database and
artifact directory. The wrapper does not create an experimental database or
change the database schema.

## Running a queue

Run commands from the repository root with the project environment activated:

```bash
cd /path/to/project
source .venv/bin/activate

python run_managed_queue.py \
  --queue example_queue \
  --dry-run

python run_managed_queue.py \
  --queue example_queue
```

The wrapper also accepts `--select 1-10`, `--start-at N`, and `--summary-only`.
`--dry-run` and `--summary-only` do not change hardware settings or start training.
The default frequency cap is `--max-mhz 5000`. The repository must be clean
before training.

Run one managed queue at a time. Each queue normally uses two workers; start the
next queue after the previous one has stopped and restored hardware settings.

## Configuration and queue directories

Queue manifests live under `configs/search_queues/` and may be organized into
subdirectories. Each manifest references the configuration files needed for its
runs. Keep continuation references consistent with the intended parent runs.

Discovery searches recursively for `.yml` and `.yaml` manifests. Pass the unique
filename stem to `--queue`, without a directory or extension. Duplicate manifest
stems are rejected, even when the files are in different subdirectories.

For example, a manifest at `configs/search_queues/example/example_queue.yml`
is selected with `--queue example_queue`. This is an illustrative path; use an
existing manifest from your checkout. Moving a manifest into a subdirectory does
not change its queue name or the location of its existing session artifacts.

## Scheduling, preflight, and interruption

Completed runs are skipped. A continuation can start only after its parent has
completed. Before launching its first worker, the queue reads database state
and resolves incomplete-run decisions using the existing YES/SKIP/STOP prompts.
There is no automatic deletion of historical runs. State is refreshed after
these decisions; skipped parents block unfinished descendants.

The two-worker limit includes training, waiting for evaluation, evaluation, and
finalization. A worker waiting for evaluation still occupies its slot and CPU
group. With only one runnable configuration, the other slot may remain idle.

A worker failure blocks new starts while other already-running workers finish.
A thermal abort or Ctrl+C signals the queue's owned workers to stop and persist
interrupted state. Allow shutdown and hardware restoration to finish before
starting another queue. On the next launch, incomplete runs are handled by the
normal preflight restart/cleanup workflow.

## CPU affinity and threads

Topology is discovered through sysfs. Split mode requires two L3 groups with
eight available physical cores each. It selects one logical CPU per physical
core and divides each L3 group into two disjoint four-core masks.

Logical CPU numbers depend on the host topology. Inspect the recorded groups in
`hardware.json` to identify the masks used by a particular session.

At most one active run process is allocated to each L3 group. When a run exits,
its group becomes available and the next assignment to that group uses the
alternate mask. Rotation occurs between run processes, not periodically during
training or at the transition from training to evaluation. Selection is based
on group availability, not on which individual cores are coolest.

The allocator chooses an available L3 group. Worker slot numbers should not be
used as hardware identifiers; use recorded affinity assignments instead. Mask
rotation within a group does not provide cross-group load balancing.

Affinity restricts worker execution but does not reserve those CPUs exclusively.
The operating system, browser, dashboard, and other processes may use the same
cores or their SMT siblings. Four allowed physical cores do not imply constant
four-core utilization.

Fresh-model construction and continuation loading use 16 PyTorch intra-op
threads. Subsequent work uses one intra-op and one inter-op thread, without
expanding CPU affinity. Sixteen initialization threads do not require exclusive
access to the entire CPU.

## Devices and evaluation

- Training requires explicit `ppo.device: cuda`.
- Evaluation requires explicit `evaluation.device: cpu`.
- The training service rejects unsupported device settings and checks the
  actual model device. There is no automatic training fallback to CPU.
- Evaluation checkpoint copies are loaded onto CPU. Worker evaluations are
  serialized through the evaluation gate.
- If CUDA is unavailable during the worker's startup check, the worker waits
  30 seconds and retries once in a fresh process via `exec`, then fails if CUDA
  remains unavailable. This is not an automatic retry of partially completed
  training after a CUDA error.
- Evaluation phases can include plotting and persistence in addition to model
  inference. Plot generation can therefore contribute to CPU load while a
  worker is reported as `EVAL_TRAIN` or `EVAL_VAL`.

## Hardware settings and thermal control

The managed session requires AC power and the Legion `powermode` and
`fan_fullspeed` interfaces. Missing interfaces stop preparation. The wrapper
requests sudo credentials at startup and maintains authorization during work.

| Setting | Current default |
| --- | --- |
| Maximum CPU frequency | 5000 MHz |
| CPU boost | Enabled |
| Fans during the session | Full speed |
| Initial cooldown | CPU at or below 65°C continuously for 30 seconds |
| Maximum initial cooldown wait | 10 minutes |
| Pause admission of new runs | CPU at or above 91.5°C |
| Resume admission | CPU at or below 84.5°C |
| Abort queue | CPU at or above 97.5°C |
| CPU control interval | Approximately 3 seconds |
| Full telemetry interval | Approximately 6 seconds |
| Telemetry buffer flush interval | At most approximately 30 seconds |
| Terminal status refresh | 90 seconds, plus immediate thermal transitions |

Admission pause blocks only new run starts. Already-active workers continue
training, waiting, or evaluating. It is a global queue state, not a pause
assigned to a particular worker. A thermal abort interrupts the owned workers.
Loss of the required CPU temperature reading during work also stops them.

The control temperature uses the hottest monitored CPU sensor. The abort value
is a software reaction threshold, not a guaranteed hardware temperature ceiling
or evidence that no throttling occurs. Userspace scheduling and sampling delays
can allow a peak above the threshold before shutdown takes effect.

Previous settings are restored after normal completion, errors, and handled
interruptions. SIGKILL, system failure, or power loss cannot execute the cleanup
handler. The recorded previous settings remain in `hardware.json` for inspection.

## Session status and files

Managed-session files are stored in:

```text
artifacts/search_queues/QUEUE_NAME/TIME_ID/
```

This path uses the queue name, without the manifest's parent subdirectories.

| File | Purpose |
| --- | --- |
| `plan.json` | Selected configurations, dependencies, restart decisions, completed/skipped/blocked runs, and final-stage targets. |
| `hardware.json` | Launch arguments, topology, limits, previous/applied settings, and restoration outcome. Direct queue execution without the wrapper has no hardware file. |
| `events.jsonl` | Phase changes, attempt registrations, exits, thermal transitions, and errors. Attempt registration and exit records are flushed and fsynced. |
| `telemetry.csv` | Hardware and worker samples, retained uncompressed after completion or interruption. |
| `telemetry_summary.json` | Aggregate hardware measurements produced during finalization. |
| `summary.json` | New completed runs, session duration/rate, final-stage counters, and outcomes. |
| `worker_1_last.state.json`, `worker_2_last.state.json` | Latest atomic state snapshots, overwritten during work and retained after exit. |

A worker state file appears when the slot is first used. It does not grow into
a history; use `events.jsonl` for historical events. A stale state file does not
prove that its PID still belongs to the original process. Final summary files
may not yet exist during an active session.

CSV samples contain CPU group utilization, CPU sensor temperatures, available
RAM, GPU utilization/VRAM/temperature, and worker names, PIDs, phases, and masks.
The three-second control maximum is recorded in `summary.json`; a short peak
may be absent from the six-second CSV samples. Buffer flushing does not postpone
thermal control. A buffer flush is not itself a guarantee of durability across
power loss.

No automatic cleanup of old sessions is performed. Temporary atomic-write
files may remain after a crash.

## Terminal output and training records

The run list appears once after cooldown. Preliminary validation remains silent
unless it fails. The English status block refreshes every 90 seconds; only
changed rows are rewritten in a TTY. Worker 1 remains above Worker 2 and idle
slots stay visible. Start/completed messages are recorded as events rather than
printed individually.

Thermal transitions update the bottom status row immediately. The row may retain
the latest transition message, such as `Thermal admission resumed: CPU=84.5`,
rather than reverting to the initial `Thermal admission: RUNNING` text. A resumed
message indicates that admission was reopened; it is not an active pause.

Redirected output contains initial/final status tables and thermal transitions
without ANSI updates. Narrow terminals truncate long rows to prevent wrapping;
widen the terminal to see full run names. Display timing does not change the
3/6/30-second control, telemetry, and flush intervals.

TRAIN, EVAL_TRAIN, and EVAL_VAL counters represent final stages per run in the
current session. Remaining includes active work. Periodic evaluations are
counted separately, and only successful evaluations count. `training_mode: none`
has zero evaluation targets. Early stopping still has one final evaluation pair.
The completion rate is newly completed runs divided by queue elapsed time,
including overlap, evaluation waits, and admission pauses. It is not the sum of
individual worker durations. Initial hardware cooldown is outside queue runtime.

No worker `.log` files or log archives are generated. SB3 text tables and periodic
text dumps are disabled for managed workers, including resumed models. Ordinary
stdout/stderr is drained and discarded. A bounded output tail of up to 256 KiB
per worker is kept in memory and written to events on failure/interruption;
long tracebacks may be truncated. Structured phase events use a separate bounded
channel. Abrupt system loss cannot preserve a RAM-only output tail.

Training metrics in the database, checkpoints, evaluation metrics, trajectories,
plots, and their configured persistence intervals are unaffected by these text
logging settings.

The recovery mechanism accepts `--attempt PATH/events.jsonl --attempt-id ID`.
It checks recorded run identity, configuration hash, PID start time, and database
claims before changing abandoned status. Legacy `.attempt.json` receipts remain
readable. These are recovery options, not ordinary managed-queue launch arguments.

## Validation and environment persistence

Run the full project test suite after changing queue or hardware-management code:

```bash
python -m pytest -q
```

Database integration tests must use the project's isolated test-database setup,
never the production database through overridden test settings.

After changes to the operating system, drivers, Python environment, or numerical
libraries, verify the environment before starting a long queue:

- Activate the project's virtual environment and confirm its interpreter path.
- Run `python -m pip check` and the project tests.
- Confirm CUDA availability and execute a small tensor operation on the GPU.
- Check that the required hardware-control interfaces are accessible.
- Verify database connectivity and any required container services.

A virtual environment must reference an interpreter that remains installed.
Activate it in each new shell or invoke its interpreter explicitly. Service
startup, container restart policies, and hardware-module loading must be
configured separately if they are required after reboot.

A custom hardware module may need to be rebuilt and signed after a kernel
update. An autoload configuration alone does not perform that rebuild.

For reproducibility-sensitive workloads, replay representative fresh and
continuation runs and compare saved training metrics, evaluation results,
checkpoint tensor payloads, trajectories, and artifact metrics. Use the same
source checkpoint for a continuation replay. Import checks and passing unit
tests alone do not establish numerical reproducibility.
