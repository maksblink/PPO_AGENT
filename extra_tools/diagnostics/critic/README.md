# Critic diagnostics

Reusable, explicit experiments for standalone PPO runs and checkpoints. These
commands do not change the normal actor/critic learning-rate configuration.
They support this project's reviewed SB3 PPO MLP implementation with separate
actor/value networks, a single environment, and single-group Adam. They reject
unsupported PPO source changes, shared trainable feature extractors, BatchNorm,
Dropout and debugger/coverage tracing. This is not an arbitrary-policy framework.

## Commands

| File | Purpose |
|---|---|
| `prepare.py` | Prepare a replay of any eligible run or a new continuation from a checkpoint |
| `verify.py` | Check a run replay's metrics, source/config and final policy/optimizer against its reference |
| `sweep.py` | Replay captured updates and compare critic LR factors |
| `compare.py` | Compare completed sweep reports over shared steps |
| `_trial.py` | Internal subprocess for one capture; restores runtime before CUDA initialization |

Run commands from the project root with the project virtual environment active.
Help is available with `--help` on every command. Preparation is a preview unless
`--apply` is specified. No command starts training implicitly.

## Replay a completed run

```bash
python extra_tools/diagnostics/critic/prepare.py \
  --run-id ORIGINAL_RUN_ID --name critic_check_a \
  --steps 1024 2048 --top-mse-increases 3 --neighbors 1
```

Replace IDs and steps with actual values for your run. `--steps` uses local
`run_step`, not cumulative `model_step`. Automatic selection ranks saved
post-MSE / pre-MSE (pre-MSE = error_std² + error_mean²), skipping undefined
or nonfinite entries. Neighbors are neighboring available metric records.
Explicit and automatic selections are combined and deduplicated. Select a
small number: each capture includes a full model, optimizer and rollout.

Repeat with `--apply` to create:

```text
configs/diagnostics/critic/critic_check_a.yml
configs/diagnostics/critic/critic_check_a.yml.critic.json
configs/search_queues/diagnostics/critic/critic_check_a.yml
```

The sidecar is the explicit opt-in, bound to the YAML SHA-256 and source
checkpoint ID/hash. The service validates actual rollout boundaries before
creating a DB run. Normal configs without this sidecar do not enable diagnostics,
even if their names resemble old diagnostic names. Keep the YAML and sidecar
together; modifying the YAML invalidates the sidecar. Config bytes are stable
between preview/apply. Source checkpoint selectors are pinned to IDs rather than
mutable `best`/`final` labels. Saved runs and source checkpoints remain unchanged.

Commit code changes before using the managed runner (configs remain subject to
your existing Git ignore rules):

```bash
python run_managed_queue.py --queue critic_check_a --dry-run
python run_managed_queue.py --queue critic_check_a
```

An ordinary run, checkpoints, TRAIN/VAL evaluations and diagnostics are created.
Every rollout gets summary measurements; selected steps also get per-optimizer-
step measurements and a capture. The resolved experiment plan is saved as
`artifacts/runs/<ID>/critic_experiment.json`. Early stopping may end before all
selected steps: the sweep rejects missing captures instead of silently omitting
them. Existing diagnostic names are never silently reused by preparation.

After completion:

```bash
python extra_tools/diagnostics/critic/verify.py --run-id DIAGNOSTIC_RUN_ID
python extra_tools/diagnostics/critic/sweep.py --run-id DIAGNOSTIC_RUN_ID \
  --factors 0.5 0.25 0.1
```

Verification compares configurations (ignoring name and equivalent checkpoint
selector text), actual source checkpoint ID, all training metric records except
IDs/timestamps, and final `policy.pth` / `policy.optimizer.pth` bytes. It also
checks checkpoint hashes against the DB. It does not compare evaluation or
trajectory files. Historical replay can fail if code, environment, data or
runtime changed; a failed check is evidence to investigate, not bypass.

## Start from a checkpoint

```bash
python extra_tools/diagnostics/critic/prepare.py \
  --checkpoint-id CHECKPOINT_ID --config path/to/continuation.yml \
  --name critic_checkpoint_a --steps 1024 2048
```

Supply duration, seed, data and PPO settings explicitly in the config. Preparation
sets the continuation source to the selected checkpoint and checks resume
compatibility. Add `--apply`, then run its queue. This is a new continuation,
not an exact reproduction of a historical run; `verify.py` therefore rejects it.
Automatic step selection from historical metrics is only available in run mode.

## Offline LR sweeps

Use `--run-id` for a completed experiment in the DB, or `--run-dir` for a trusted
local artifact directory containing its experiment plan and complete captures.
Optional `--steps` restricts execution to a subset of the planned captures.
Only use captures created locally by this tool: model/buffer loading uses Python
serialization and is unsafe for untrusted inputs.

The sweep verifies checksums and actual optimizer-update counts against captured
rollout size, batch size and n_epochs. It allows target-KL early stopping but
stops a trial if altered LR changes the number of updates/RNG consumption.
There is no fixed 36-update assumption. Original and equal-LR split-group controls
must reproduce weights, per-parameter Adam state, RNG and diagnostics exactly.
Only then are positive critic LR factors tested, with actor LR unchanged.

Each variant restarts from the original captured state. Global gradient clipping
is retained and can couple the actor and critic; inspect actor-change and clipping
flags. Targets remain fixed. This does not simulate a full lower-LR training path
and does not demonstrate better trading returns. Zero target variance gives
undefined NMSE, not zero. A capture with zero optimizer steps stops the LR comparison with an explicit
message: it offers no evidence about learning-rate effects.

The offline sweep uses the captured CPU/CUDA runtime and does not run the managed
thermal controller. Do not run concurrent training queues. Outputs go to a new
`critic_lr_sweep/<timestamp>/` directory inside the diagnostic run; failed controls
leave a stopped report. Existing outputs/captures are not overwritten.

## Compare experiments

```bash
python extra_tools/diagnostics/critic/compare.py \
  --report path/to/first/report.json \
  --report path/to/second/report.json
```

Repeat `--report` for additional experiments. Factors must match. The tool prints
configuration differences, per-step relative MSE, improvement counts, medians,
actor changes and clipping. Only the intersection of steps is summarized; optional
`--steps` selects a smaller common subset. Excluded steps are counted explicitly.
Same step numbers do not guarantee identical actions, observations or targets.
Selected failures are not a representative population sample or a causal test.

## Cleanup

First preview only the diagnostic IDs with the existing maintenance tool:

```bash
python extra_tools/maintenance/clean_training.py clean-first-stage \
  --run-id DIAGNOSTIC_RUN_ID --dry-run
```

After inspecting the preview, rerun without `--dry-run`. The existing journaled
cleanup protects retained dependencies and deletes run artifacts, including all
captures/sweep outputs. ID sequences are preserved. Separately remove the specific
experiment's three config/sidecar/queue files and its session-log directory after
confirming no remaining work references them. Never use `clean-all` for an
individual diagnostic experiment. General training-metric analysis lives in
`extra_tools/analysis/ppo_analyze_training.py` and remains independent.
