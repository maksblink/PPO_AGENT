# PPO_AGENT

Clean PPO agent training and evaluation project.

The project will provide:

- YAML-configured training runs;
- PostgreSQL run registry;
- Git commit tracking for every run;
- automated tests with pytest;
- selective migration of functionality from the old RL_mini project.

## Command-line entry points

Run one YAML-configured training job:

```bash
python -m train_and_eval.training \
  --config configs/experiments/example_5m_timesteps.yml
```

Inspect the read-only PostgreSQL run registry:

```bash
python -m train_and_eval.runs list
python -m train_and_eval.runs show --run-id 10
python -m train_and_eval.runs checkpoints --run-id 10
python -m train_and_eval.runs evaluations --run-id 10
python -m train_and_eval.runs evaluations --run-id 10 --full
```
