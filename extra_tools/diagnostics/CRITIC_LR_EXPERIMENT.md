# Temporary fixed-rollout critic LR experiment

## Scope

This experiment extends the existing diagnostic replay only. Normal run configs,
PPO configuration schema, existing grids and installed SB3 files are unchanged.
The single-capture run name is `diagnostic_capture_3732`. It repeats #3732 from
#3730's final checkpoint using the same training parameters.

Run #4079 only recorded statistics. It cannot supply the full optimizer/model state
and rollout before the anomaly. One additional complete managed replay captures
these at local step 404480, before PPO updates that rollout. The normal replay then
finishes. Existing #3730, #3732 and #4079 are retained.

## Workflow

Apply the extension patch, run tests, and commit only the extension files before
starting the managed queue (the queue requires a clean Git working tree).

```bash
python extra_tools/diagnostics/prepare_critic_replay.py --capture
python extra_tools/diagnostics/prepare_critic_replay.py --capture --apply
python run_managed_queue.py --queue critic_capture_3732 --dry-run
python run_managed_queue.py --queue critic_capture_3732
```

After the capture run finishes, use a plain terminal without debugger/coverage:

```bash
python extra_tools/diagnostics/critic_lr_experiment.py
```

The default locates the one completed capture run by its exact name in a read-only
DB query. Alternatively pass a trusted local capture directory explicitly:

```bash
python extra_tools/diagnostics/critic_lr_experiment.py \
  --capture artifacts/runs/NEW_PADDED_ID/critic_capture_404480 \
  --factors 0.5 0.25 0.1
```

Do not run this GPU comparison alongside a training queue. It does not modify
hardware settings or start a managed thermal session. It runs five short, fixed
rollout updates on the recorded device, not full market-data training runs. Original
runtime thread/determinism/math settings are restored in this process only.

## Controls and variants

Each trial reloads identical pre-update weights, per-parameter Adam state, the
full rollout and RNG states. Captures include CPU and CUDA RNG state where used.
All trials use the same target values and minibatch sampling sequence.

1. Original single-group Adam must reproduce the captured post-update weights,
   named optimizer states, RNG states and diagnostic measurements exactly.
2. Splitting actor/critic parameter groups at equal LR must reproduce the same
   results. Adam is not reset; its existing parameter states are retained.
3. Only after both controls pass, test critic LR factors 0.5, 0.25 and 0.1.
   Actor LR remains 0.00004; critic LR becomes 0.00002, 0.00001 and 0.000004.

PPO resets LR at train entry. The experimental optimizer pre-step hook explicitly
sets only the critic group's LR afterward on every step. PPO.train remains original.
Global actor/critic gradient clipping remains unchanged. If clipping activates,
changing critic updates may indirectly affect later actor updates; the report
includes clipping counts and changed actor tensor counts. This is not a promise
that actor weights stay identical in every possible setting.

No variant creates a database run, overwrites a checkpoint, or changes a grid.
The comparison includes MSE after each minibatch/epoch, final NMSE/EV and peak MSE.
It assesses fitting stability on one visited rollout, not validation return or
out-of-sample performance. Do not select a production LR from this test alone.

## Storage and integrity

Everything is below the new capture run's artifact directory:

- `critic_diagnostics.jsonl`: the normal diagnostic timeline;
- `critic_capture_404480/before.zip`: model and Adam immediately before the update;
- `critic_capture_404480/rollout.pt`: full rollout and RNG states;
- `critic_capture_404480/expected_after.pt`: expected weights, named Adam states and RNG;
- runtime metadata and original detailed measurements;
- `complete.json`: checksums, written last after successful capture;
- `critic_lr_experiment/<timestamp>/report.json`: compact results;
- per-trial JSONL traces with all measured optimizer updates.

Capture refuses to overwrite an existing directory. Incomplete or corrupted
captures are rejected. Pickled payloads are for trusted local captures from this
tool only. The tool checks Python/library/GPU identity and stops on baseline
mismatch before running experimental LR variants. Runtime metadata does not prove
that all driver/kernel conditions are identical; the exact baseline is the final
reproducibility gate. A mismatch produces a stopped report, not a successful result.

Local tests cover CPU capture/replay, equal-LR control, preserved optimizer/RNG,
actor isolation when clipping is inactive, LR application, checksum failures,
invalid factors, capture collisions, observer cleanup and the complete CLI flow.
The actual CUDA equivalence controls run on the user's laptop.

## Reversible cleanup

Keep this extension in its own commit. After analysis:

1. Preview and delete only the new capture run and, when no longer needed, #4079
   with the existing clean-first-stage mechanism. Review the actual IDs first.
   Their run artifact directories contain the captures and experiment results.
2. Remove their disposable configs, queue manifests and queue session directories.
3. `git revert <extension-commit>` removes this extension while retaining the
   earlier training-analysis tool and earlier diagnostics.

The earlier diagnostic was committed together with training analysis. Do not
revert that earlier mixed commit wholesale if training analysis should remain.
Removing that earlier diagnostic requires a separate targeted cleanup change.
No cleanup or Git revert is performed automatically by this tool.

## Multi-rollout extension

Use a separate run `diagnostic_capture_multi_3732` to keep #4080 and the first
experiment unchanged. This run follows the original #3732 configuration from
parent #3730; only its name differs. Capture does not change LR during training.

```bash
python extra_tools/diagnostics/prepare_critic_replay.py --capture-multi
python extra_tools/diagnostics/prepare_critic_replay.py --capture-multi --apply
python run_managed_queue.py --queue critic_capture_multi_3732 --dry-run
python run_managed_queue.py --queue critic_capture_multi_3732
```

The 16 fixed local steps are selected before running new LR variants:

| Group | Steps | Reason |
|---|---|---|
| Known spikes | 404480, 475136, 585728 | Recheck three observed instabilities |
| Neighbors | 403456, 405504, 474112, 476160, 584704, 586752 | Behavior immediately before and after each spike |
| Scheduled controls | 1024, 64512, 128000, 256000, 384000, 512000, 638976 | Cover early, middle and late training |

The first four scheduled controls have nonzero rewards in the saved #4079
timeline. Controls are not claimed to be uniformly stable: step 512000 also
worsened in the original update. The report preserves actual reward counts,
action counts and target variance context. This is a targeted diagnostic sample,
not a random sample or an unbiased estimate of all training updates.

Every selected step stores pre-update weights, Adam, RNG and rollout, plus the
expected original post-update state and detailed measurements. Captures may use
several GiB in total; allow extra disk space and replay time. No existing capture
is overwritten. The existing single-capture commands continue to work.

After the new managed replay completes, with no other training running:

```bash
python extra_tools/diagnostics/critic_lr_sweep.py
```

The sweep validates all 16 capture inventories before running trials. It invokes
the existing guarded experiment in a fresh process per rollout, restoring its
recorded runtime settings. Both exact-reproduction controls must pass separately
for every rollout before its lower-LR trials. The sweep stops at the first failed
control and retains an explicitly stopped partial report.

Each variant starts from that rollout's ORIGINAL pre-update state. Lower-LR
results are not carried into the next rollout. Thus this tests immediate fitting
behavior, not the trajectory of a model trained with a lower LR throughout.

Outputs are stored in the NEW run directory under `critic_lr_sweep/<timestamp>/`:

- `report.json`: compact combined report with every rollout, category, reward
  count, actor-change/clipping counts and the full per-update MSE sequences;
- `<step>/report.json`: single-rollout comparison and runtime information;
- `<step>/*.jsonl`: original, equal-LR control and lower-LR detailed traces.

Final/initial MSE tests improvement from the captured state. Final/original MSE
compares each variant with the original update; below 1 means lower fixed-target
error. Also inspect peak error, NMSE, EV, actor changes and individual controls.
Do not select LR solely from the most extreme spike or aggregate all categories
into one winner. Zero denominators produce null ratios. The combined report does
not create additional database runs or perform TRAIN/VAL market evaluation.

The sweep runs GPU updates without changing hardware controls; do not run it
alongside a queue. Use a plain terminal, without debugger or coverage tracing.
Original full-training equivalence to #3732 is a separate check: the sweep's
controls verify each saved update against the NEW capture run, not the entire
new run against the historical run.

Keep the extension in a separate commit. Cleanup later removes only the new
multi-capture run through clean-first-stage, its diagnostic config/queue and
queue session logs, then reverts the extension if desired. Preserve #4079, #4080
and all original runs until results have been reviewed. No cleanup is automatic.
