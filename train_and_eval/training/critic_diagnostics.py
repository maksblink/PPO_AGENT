"""Opt-in observer of the unmodified SB3 update. No checkpoint/model loading."""
from __future__ import annotations
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch
from stable_baselines3 import PPO

REPLAY_NAME = "diagnostic_repeat_3732"
# Full per-minibatch measurements only around known spikes and boundaries.
DETAILED_STEPS = {1024, 638976} | {
    step + offset for step in (404480, 475136, 585728)
    for offset in (-1024, 0, 1024)
}
EXPECTED_TRAIN_HASH = "7c8d09940123242f0ac401c5ec2cc858b8bf3339f571c222382f048ffd8306ae"


def stats(values):
    a = np.asarray(values, dtype=np.float64).reshape(-1)
    return {"mean": float(a.mean()), "std": float(a.std()),
            "min": float(a.min()), "max": float(a.max())}


def norm(tensors):
    return math.sqrt(sum(float(t.detach().double().square().sum().item())
                         for t in tensors if t is not None))


def measure(model):
    b = model.rollout_buffer
    # Buffer.get() flattens every relevant array consistently on first use.
    observations = b.observations
    if not b.generator_ready:
        observations = b.swap_and_flatten(observations)
    obs, _ = model.policy.obs_to_tensor(observations)
    with torch.no_grad():
        predictions = model.policy.predict_values(obs).detach().cpu().numpy().reshape(-1).astype(np.float64)
    targets = np.asarray(b.returns, dtype=np.float64).reshape(-1)
    errors = targets - predictions
    variance = float(targets.var())
    mse = float(np.mean(errors**2))
    return {"mse": mse, "nmse": mse / variance if variance > 0 else None,
            "ev": 1 - float(errors.var()) / variance if variance > 0 else None,
            "prediction": stats(predictions), "error": stats(errors)}


def train_observed(model, path, local_step, *, capture=False):
    """Call original train exactly once; remove all observers even on failure."""
    if hashlib.sha256(inspect.getsource(PPO.train).encode()).hexdigest() != EXPECTED_TRAIN_HASH:
        raise RuntimeError("Unsupported PPO.train source; diagnostics not started")
    if type(model) is not PPO:
        raise RuntimeError("Diagnostics require the original PPO class")
    if sys.gettrace() is not None:
        raise RuntimeError("Run diagnostics without debugger/coverage tracing")
    if any(isinstance(m, (torch.nn.modules.dropout._DropoutNd,
                         torch.nn.modules.batchnorm._BatchNorm))
           for m in model.policy.modules()):
        raise RuntimeError("Diagnostics require a policy without Dropout/BatchNorm")
    detailed = local_step in DETAILED_STEPS
    b = model.rollout_buffer
    row = {"run_step": local_step, "model_step": int(model.num_timesteps),
           "detailed": detailed, "target": stats(b.returns),
           "advantages": stats(b.advantages), "reward": stats(b.rewards),
           "reward_nonzero": int(np.count_nonzero(b.rewards)),
           "actions": {str(k): int(v) for k, v in zip(*np.unique(b.actions, return_counts=True))},
           "before": measure(model), "updates": []}
    named = list(model.policy.named_parameters())
    actor = [p for n,p in named if n.startswith(("mlp_extractor.policy_net.", "action_net."))]
    critic = [p for n,p in named if n.startswith(("mlp_extractor.value_net.", "value_net."))]
    if set(map(id, actor + critic)) != set(map(id, model.policy.parameters())):
        raise RuntimeError("Unexpected shared/trainable feature extractor")
    current = {}
    saved = []
    source, first = inspect.getsourcelines(PPO.train)
    markers = {}
    for offset, line in enumerate(source):
        for key, token in (("advantages", "ratio = th.exp("),
                           ("loss", "self.policy.optimizer.zero_grad()"),
                           ("grad", "th.nn.utils.clip_grad_norm_(")):
            if token in line:
                markers[first + offset] = key
    if len(markers) != 3:
        raise RuntimeError("PPO observer markers not found")

    def trace(frame, event, arg):
        if frame.f_code is not PPO.train.__code__:
            return None
        if event == "line":
            loc = frame.f_locals
            marker = markers.get(frame.f_lineno)
            if marker == "advantages":
                current.clear()
                current["epoch"] = int(loc["epoch"]) + 1
                current["raw_advantages"] = stats(loc["rollout_data"].advantages.detach().cpu().numpy())
                current["normalized_advantages"] = stats(loc["advantages"].detach().cpu().numpy())
            elif marker == "loss":
                for key in ("policy_loss", "value_loss", "entropy_loss", "loss"):
                    current[key] = float(loc[key].detach().item())
            elif marker == "grad":
                current["actor_grad_before_clip"] = norm(p.grad for p in actor)
                current["critic_grad_before_clip"] = norm(p.grad for p in critic)
        return trace

    def before_step(optimizer, args, kwargs):
        current["actor_grad_after_clip"] = norm(p.grad for p in actor)
        current["critic_grad_after_clip"] = norm(p.grad for p in critic)
        current["adam_exp_avg_norm"] = norm(optimizer.state[p].get("exp_avg") for p in critic)
        current["adam_exp_avg_sq_norm"] = norm(optimizer.state[p].get("exp_avg_sq") for p in critic)
        saved[:] = [p.detach().clone() for p in critic]

    def after_step(optimizer, args, kwargs):
        current["critic_parameter_delta"] = norm(p.detach() - old for p,old in zip(critic, saved))
        current["post"] = measure(model)
        current["update"] = len(row["updates"]) + 1
        row["updates"].append(dict(current))
        saved.clear()

    capture_directory = None
    if capture:
        from .critic_rollout_experiment import CAPTURE_STEP, capture_before
        if local_step == CAPTURE_STEP:
            capture_directory = capture_before(model, path, local_step)
    handles = []
    try:
        if detailed:
            handles = [model.policy.optimizer.register_step_pre_hook(before_step),
                       model.policy.optimizer.register_step_post_hook(after_step)]
            sys.settrace(trace)
        model.train()
    finally:
        if detailed:
            sys.settrace(None)
        for handle in handles:
            handle.remove()
    row["after"] = measure(model)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")

    if capture_directory is not None:
        from .critic_rollout_experiment import capture_after
        capture_after(model, capture_directory, row)
