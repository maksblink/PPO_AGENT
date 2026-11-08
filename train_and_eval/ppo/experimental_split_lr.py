"""Opt-in stage-one actor/critic LR experiment; ordinary PPO never imports this."""
from __future__ import annotations

import copy
import math

import torch
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import FlattenExtractor


def validate_factor(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("critic LR factor must be a finite positive number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError("critic LR factor must be a finite positive number")
    return float(value)


def validate_groups(optimizer, factor):
    if len(optimizer.param_groups) != 2:
        raise ValueError("Split LR requires exactly two optimizer groups")
    for group, role in zip(optimizer.param_groups, ("actor", "critic")):
        if group.get("split_lr_role") != role or group.get("split_lr_factor") != factor:
            raise ValueError("Checkpoint split LR optimizer metadata mismatch")


class SplitLRPolicy(ActorCriticPolicy):
    """Rebuild the same group layout before SB3 restores saved Adam moments."""

    def __init__(self, *args, critic_lr_factor, **kwargs):
        self.critic_lr_factor = validate_factor(critic_lr_factor)
        if kwargs.get("optimizer_class", torch.optim.Adam) is not torch.optim.Adam:
            raise ValueError("Split LR supports Adam only")
        if kwargs.get("features_extractor_class", FlattenExtractor) is not FlattenExtractor:
            raise ValueError("Split LR supports parameter-free FlattenExtractor only")
        kwargs["optimizer_class"] = torch.optim.Adam
        options = dict(kwargs.get("optimizer_kwargs") or {})
        options.setdefault("eps", 1e-5)
        kwargs["optimizer_kwargs"] = options
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule):
        if not isinstance(self.action_space, spaces.Discrete):
            raise ValueError("Split LR supports discrete-action MLP policies only")
        super()._build(lr_schedule)
        named = list(self.named_parameters())
        actor = [p for n, p in named if n.startswith(("mlp_extractor.policy_net.", "action_net."))]
        critic = [p for n, p in named if n.startswith(("mlp_extractor.value_net.", "value_net."))]
        if not actor or not critic or len(actor) + len(critic) != len(named):
            raise ValueError("Unsupported shared or unclassified trainable parameters")
        if set(map(id, actor)) & set(map(id, critic)):
            raise ValueError("Actor and critic parameters must be disjoint")
        group = self.optimizer.param_groups[0]
        options = {k: copy.deepcopy(v) for k, v in group.items() if k != "params"}
        group.update(params=actor, split_lr_role="actor", split_lr_factor=self.critic_lr_factor)
        self.optimizer.add_param_group(dict(
            options, params=critic, split_lr_role="critic",
            split_lr_factor=self.critic_lr_factor,
        ))
        self.optimizer.param_groups[1]["lr"] *= self.critic_lr_factor
        factor = self.critic_lr_factor

        def apply_ratio(optimizer, args, kwargs):
            validate_groups(optimizer, factor)
            optimizer.param_groups[1]["lr"] = optimizer.param_groups[0]["lr"] * factor

        def check_loaded_state(optimizer):
            validate_groups(optimizer, factor)

        # Hooks are not serialized in optimizer.state_dict(). _build installs them
        # again before PPO.load restores Adam state, including evaluation loads.
        self.optimizer.register_step_pre_hook(apply_ratio)
        self.optimizer.register_load_state_dict_post_hook(check_loaded_state)

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()
        data.update(critic_lr_factor=self.critic_lr_factor, optimizer_class=torch.optim.Adam)
        data["optimizer_kwargs"] = dict(data["optimizer_kwargs"])
        data["optimizer_kwargs"].pop("critic_lr_factor", None)
        return data


def verify_resume_factor(model, expected):
    actual = getattr(model.policy, "critic_lr_factor", None)
    if actual != expected:
        raise ValueError(f"Split LR resume mismatch: checkpoint={actual!r}, config={expected!r}")
    if expected is not None:
        if not isinstance(model.policy, SplitLRPolicy) or type(model.policy.optimizer) is not torch.optim.Adam:
            raise ValueError("Split LR checkpoint has an unexpected policy or optimizer")
        validate_groups(model.policy.optimizer, expected)
