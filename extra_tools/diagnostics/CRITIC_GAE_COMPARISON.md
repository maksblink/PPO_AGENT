# Critic learning-rate comparison across two GAE paths

This temporary diagnostic experiment compares original run 3732 (GAE 0.95,
parent 3730) with run 2143 (GAE 0.85, parent 2089). Both use actor and critic
LR 0.00004, gamma 0.90, entropy coefficient 0.0002, architecture 768x4,
PPO n_epochs 9, seed 1, stage r_v02.

The new replay preserves run 2143's configuration except its name and uses
its own original parent. Preparation checks full saved configurations, excluding
GAE and the run/parent names. It refuses an existing diagnostic run name.
Only the exact diagnostic name enables capture; normal runs are unchanged.

The GAE 0.85 capture records the previous 16 steps plus 199680, 248832 and
273408 (19 total). 475136 is already among the common steps. Each capture
preserves weights, Adam state, RNG, rollout and fixed targets. Replay verifies
original and equal-LR split controls before changing only critic LR.

## Run locally

Commit the code after tests pass; the managed runner requires a clean tree.
No other training queue should be running during the offline GPU sweep.

```bash
python extra_tools/diagnostics/prepare_critic_replay.py --source-run-id 2143 --capture-multi
python extra_tools/diagnostics/prepare_critic_replay.py --source-run-id 2143 --capture-multi --apply
python run_managed_queue.py --queue critic_capture_multi_2143 --dry-run
python run_managed_queue.py --queue critic_capture_multi_2143
python extra_tools/diagnostics/critic_lr_sweep.py --source-run-id 2143
```

The sweep resolves the completed diagnostic run in the database; no run ID
needs to be guessed. It creates a new timestamped report and never overwrites
captures. It performs 19 x 5 independent PPO updates (original, split control,
and factors 0.5, 0.25, 0.1). Each PPO update has 36 optimizer steps.

Use the final report path printed by the sweep:

```bash
python extra_tools/diagnostics/compare_critic_gae.py \
  --gae95-report artifacts/runs/00004083/critic_lr_sweep/20261104T225740373426Z/report.json \
  --gae85-report PATH_PRINTED_BY_NEW_SWEEP/report.json
```

This prints a compact comparison without modifying files or the database.
The summary reports common steps separately from three additional selected
GAE 0.85 failures. It shows ratios relative to each path's original LR,
counts of improvements, actor changes and clipping. A changed actor despite
unchanged actor LR can arise from global gradient clipping; inspect these flags.

## Interpretation and limits

Matching steps are not identical observations/actions/targets: the policies
and optimizer histories developed separately. This checks whether critic-LR
sensitivity also occurs in the GAE 0.85 path. It does not isolate the direct
causal effect of GAE, estimate population-wide frequencies, or measure trading
returns under a lower LR. The 19 captures are selected diagnostic samples.
All interventions use frozen rollout targets and restart from the original
captured state; they are not a continuous lower-LR training trajectory.

The diagnostic replay creates one run and ordinary TRAIN/VAL evaluations.
Offline sweeps only write experiment outputs. Keep original runs 2143, 2089,
3732 and 3730 unchanged. Remove diagnostic runs later using the project's
normal guarded cleanup tool, after retaining needed conclusions.
