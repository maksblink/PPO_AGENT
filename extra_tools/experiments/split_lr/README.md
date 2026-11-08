# Experimental actor / critic learning rates

This opt-in stage-one experiment keeps the actor learning rate unchanged while
scaling the critic learning rate. It is not enabled in existing grids and does
not change the default PPO path. There is no database migration.

## Configuration

Add this field to the `ppo` section of a **new, separately named fresh config**:

```yaml
ppo:
  learning_rate: 0.00004
  experimental_critic_lr_factor: 0.5
```

This fragment must be merged into a complete existing config. It is not a
standalone training config. Actor LR is 0.00004 and critic LR is 0.00002.
The existing `learning_rate` metric and queue LR column show the actor/base LR.
The critic LR is that value multiplied by the recorded factor; no new DB metric
column is added. Both the normalized run configuration and checkpoint record the
factor. An explicit factor of 1.0 enables the split-group equivalence control.
Omitting the field (or setting it to null) uses ordinary PPO, including its
original single Adam group and original normalized-config representation.

Use unique names, for example `exp_clr050_family_a_s01_v00` through `v04`.
Keep configs and queues in their own experiment directory. Every resume config
must keep the same factor and refer to the previous **experimental** run's final
checkpoint. All other fields should match the corresponding baseline config.
The first version deliberately rejects switching between ordinary and split LR,
or changing the factor inside a continuation chain. Start from fresh to compare
complete training paths. Existing immutable source checkpoints are never edited.
Changing the base LR on resume is still supported; it changes both effective LRs
while preserving the ratio and Adam moments. Do not do that in a fixed-LR trial.

Run the usual managed queue after committing the code and validating the queue:

```bash
python run_managed_queue.py --queue YOUR_EXPERIMENT_QUEUE --dry-run
python run_managed_queue.py --queue YOUR_EXPERIMENT_QUEUE
```

This patch creates no experiment configs, queue, DB runs, or training outputs.
Choose experiment families and create their configs separately.

## Implementation and restrictions

The implementation is isolated in
`train_and_eval/ppo/experimental_split_lr.py`. A custom MLP policy builds the
same actor and critic networks and the same Adam optimizer, then partitions its
parameters into two disjoint groups. The value MLP and value head belong to the
critic; the policy MLP and action head belong to the actor. Every parameter must
belong to exactly one group. Trainable shared feature extractors and continuous
action policies are rejected. Only the existing discrete-action MLP/Adam setup
is supported.

SB3 sets both group LRs from its schedule at each `train()` entry. An optimizer
pre-step hook sets critic LR to current actor LR times the configured factor,
immediately before each Adam step. This hook is reconstructed when loading the
policy, before Adam state is restored. The group order, roles and ratio are
validated on load. Adam moments, step counters, clipping, loss coefficients,
rollout collection and PPO epoch count are preserved. Global gradient clipping
still operates jointly across actor and critic, as before; separate LRs do not
make the two optimizations completely independent.

Native PPO ZIP save/load, evaluation-only loading and project checkpoint loading
are covered by tests. A resume also checks its configuration against the factor
inside the loaded model. Experimental checkpoints require this module to remain
available, including for evaluation.

Standalone stage-one only: walk-forward and simultaneous fixed-rollout critic
capture plans are rejected. Existing offline critic LR sweep tools continue to
support ordinary captures; this experiment does not expand their supported scope.
Canonical name migration deliberately stops if it encounters an experimental
split-LR config, so it cannot erase the factor from experiment identity.

## Validation before training

```bash
python -m pytest -q tests/test_experimental_split_lr.py
python -m pytest -q
git diff --check
```

The factor-1 control compares exact weights, per-parameter Adam state and RNG
state after complete PPO updates. It runs on CPU and also CUDA when available.
The suite also checks schedule resets, serialization, checkpoint restoration,
further training, configuration mismatch rejection and project checkpoint
loading. Passing small controls does not establish full market-run equivalence;
for a high-confidence control, replay one complete baseline path with factor 1.0
and compare metrics and policy/optimizer state before interpreting factor 0.5.

## Analyze and remove

The generic analyzers can compare the scientific field
`ppo.experimental_critic_lr_factor`. Use factor-1 and factor-0.5 experiment configs
for a numeric reference comparison. Ordinary configs omit this field and must
not silently be treated as split factor 1.0. Keep other scientific parameters
and full ancestry matched. Record ordinary-versus-factor-1 equivalence separately.
Do not mix experimental results into baseline Pareto rankings without explicitly
choosing to rank across this field. Existing dashboard grids are not a new
factor-aware comparison UI; use the generic analyzers for that purpose.

```bash
python extra_tools/analysis/ppo_analyze_training.py \
  --queue YOUR_EXPERIMENT_QUEUE \
  --compare-by ppo.experimental_critic_lr_factor \
  --reference ppo.experimental_critic_lr_factor=1.0
```

Evaluate critic stability and fit alongside final TRAIN/VAL outcomes and policy
behavior. Lower training MSE is not by itself a successful trading experiment.

Keep this change in a separate Git commit. To remove it, first stop training and
maintenance, then use `extra_tools/maintenance/clean_training.py` to preview and
remove explicitly selected experiment runs and artifacts. Remove their configs,
queues and experiment reports separately. Only then revert this commit. Do not
remove the module while keeping checkpoints that depend on it. Existing grid
runs, checkpoints and ID sequences should remain intact.
