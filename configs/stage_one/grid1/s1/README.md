# grid1 — seed 1

Main seed-1 grid comparing architectures, learning rates, discount factors and PPO optimization epochs.

- Queue: `grid1_s1`
- Config directory: `configs/stage_one/grid1/s1`
- Queue file: `configs/search_queues/grid1/s1/grid1_s1.yml`
- Runs: 2400; fresh roots: 480
  - `n_epochs=3`: 1200 runs
  - `n_epochs=9`: 1200 runs
- Architectures: `384x384x384`, `384x384x384x384`, `768x768x768`, `768x768x768x768`
- Learning rates: `0.000005`, `0.0000075`, `0.00001`, `0.0000125`, `0.000015`, `0.0000175`, `0.00002`, `0.0000225`, `0.000025`, `0.0000275`, `0.00003`, `0.000035`, `0.00004`, `0.000045`, `0.00005`, `0.00015`, `0.00025`, `0.00035`, `0.00045`, `0.00055`
- Gamma: `0.9`, `0.95`, `0.99`
- PPO n_epochs: 3, 9
- Stages: `fresh v0`, `v1`, `v2`, `v3`, `v4`
- Each run adds one data epoch. PPO n_epochs controls optimization passes per rollout, not the number of data epochs.
- Training uses CUDA; evaluation uses CPU.

Queue entries list all lower-n_epochs configs before higher-n_epochs configs. Dependencies determine which runs are ready to start; this ordering does not impose a completion barrier between groups.

Run from the repository root:

```bash
python run_managed_queue.py --queue grid1_s1
```

Add `--dry-run` to inspect the plan without starting training.
