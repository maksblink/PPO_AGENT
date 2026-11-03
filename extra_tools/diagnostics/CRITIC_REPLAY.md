# Critic replay diagnostic

This temporary diagnostic reproduces original run #3732 from the final checkpoint
of parent run #3730, using the normal managed queue and database. The only config
change is `run.name = diagnostic_repeat_3732`. No automatic deletion is performed.

## Prepare and execute

```bash
python -m pytest -q tests/test_critic_diagnostics.py
python -m pytest -q
python extra_tools/diagnostics/prepare_critic_replay.py
python extra_tools/diagnostics/prepare_critic_replay.py --apply
python run_managed_queue.py --queue critic_repeat_3732 --dry-run
python run_managed_queue.py --queue critic_repeat_3732
```

Use a plain terminal, without a debugger or coverage tracer. Preparation uses a
read-only DB snapshot, checks original/parent identities and the reviewed PPO
source, validates the config, and refuses a conflicting config or existing replay.
Normal queue preflight and hardware safeguards remain in place.

## Measurements

Only the exact special run name enables measurements. Every other run calls the
original training method directly. Nothing in installed SB3 is modified.

The original `PPO.train()` runs once per rollout. Python line tracing and optimizer
hooks observe it; no alternate optimizer or PPO implementation is substituted.
The observer never requests minibatches, samples actions, or calls backward.

Every rollout records reward/advantage/target statistics, action counts and full
buffer pre/post critic MSE, NMSE and explained variance. Detailed windows are the
first and last rollout and each known anomaly (404480, 475136, 585728) plus its
preceding and following rollout. For these 11 rollouts every optimizer step records:

- raw and normalized minibatch advantage statistics;
- policy/value/entropy/total loss;
- separate actor/critic gradient norms before and after joint clipping;
- norms of critic Adam first and second moments before the update;
- critic parameter change norm;
- critic error on the full unchanged rollout after the step.

Epoch numbers allow reading error after each complete PPO epoch. No position time
series, per-parameter gradients or full Adam tensors are stored. Measurements add
runtime and GPU/CPU work, particularly in detailed windows. No duration guarantee.
Full-buffer predictions use the same arithmetic dtype as the model; error statistics
use float64. Undefined normalized metrics are represented by null.

Output: `artifacts/runs/<new-run-id padded to 8 digits>/critic_diagnostics.jsonl`.
Expected completed run: 624 records, 11 detailed records with 36 optimizer updates
each (for the reviewed original configuration without early KL stopping).
Output is incremental; an interrupted run may have only partial diagnostics.

## Reproducibility and cleanup

Tests compare CPU policy weights, Adam state, ordinary SB3 logs and RNG states
with and without the observer; they also test hook cleanup after an exception.
CPU tests do not prove CUDA bitwise equivalence. Compare the actual replay's
ordinary training metrics, evaluations, checkpoint weights and optimizer against
#3732 before drawing conclusions about the original anomaly. The existing
checkpoint comparison workflow should be used after the new run ID is known.

Do not delete the original #3730 or #3732. After inspection and comparison, preview
cleanup for the NEW run ID with:

```bash
python extra_tools/maintenance/clean_training.py clean-first-stage --run-id NEW_ID --dry-run
```

Follow the existing cleanup tool's confirmation flow. The diagnostics live under
the replay's run artifact directory, so deleting that directory also removes them.
The disposable config/queue and queue session logs are separate and can be removed
afterward. No automatic cleanup or sequence reset is added by this diagnostic.
