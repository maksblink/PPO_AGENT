from __future__ import annotations

from dataclasses import dataclass, field
from math import isclose
from typing import Any

import numpy as np
import torch
from gymnasium.spaces import Discrete
from stable_baselines3 import PPO
from stable_baselines3.common.distributions import (
    CategoricalDistribution,
)
from stable_baselines3.common.utils import (
    set_random_seed,
)

from train_and_eval.database.models import (
    EvaluationPolicyMode,
)


class PPOPolicySelectionError(RuntimeError):
    """Raised when an evaluation policy cannot select an action."""


@dataclass(frozen=True, slots=True)
class PPOPolicyDecision:
    """One action decision and its categorical probabilities."""

    action: int
    probabilities: tuple[float, ...]
    selected_action_probability: float
    threshold_met: bool | None


def _nonnegative_integer(
    value: Any,
    *,
    name: str,
) -> int:
    if isinstance(value, bool):
        raise PPOPolicySelectionError(
            f"{name} must be an integer."
        )

    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise PPOPolicySelectionError(
            f"{name} must be an integer."
        ) from error

    if result != value:
        raise PPOPolicySelectionError(
            f"{name} must be an integer."
        )

    if result < 0:
        raise PPOPolicySelectionError(
            f"{name} must be nonnegative."
        )

    return result


def _policy_mode(
    value: EvaluationPolicyMode | str,
) -> EvaluationPolicyMode:
    if isinstance(
        value,
        EvaluationPolicyMode,
    ):
        return value

    raw_value = getattr(
        value,
        "value",
        value,
    )

    try:
        return EvaluationPolicyMode(
            str(raw_value)
        )
    except ValueError as error:
        allowed = ", ".join(
            mode.value
            for mode in EvaluationPolicyMode
        )

        raise PPOPolicySelectionError(
            "Unsupported policy_mode "
            f"{raw_value!r}. Allowed values: "
            f"{allowed}."
        ) from error


def _probability_threshold(
    value: Any,
) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise PPOPolicySelectionError(
            "probability_threshold must be numeric."
        ) from error

    if not np.isfinite(result):
        raise PPOPolicySelectionError(
            "probability_threshold must be finite."
        )

    if result < 0.0 or result > 1.0:
        raise PPOPolicySelectionError(
            "probability_threshold must be "
            "between 0 and 1."
        )

    return result


@dataclass(slots=True)
class PPOPolicySelector:
    """
    Select actions from one PPO policy during one evaluation.

    A selector is created once per evaluation. Its seed initializes
    stochastic sampling once, after which successive calls form one
    reproducible sequence.
    """

    model: PPO
    policy_mode: EvaluationPolicyMode | str
    seed: int
    threshold_action: int | None = None
    probability_threshold: float | None = None

    _resolved_mode: EvaluationPolicyMode = field(
        init=False,
        repr=False,
    )
    _action_count: int = field(
        init=False,
        repr=False,
    )
    _resolved_threshold_action: int | None = field(
        init=False,
        repr=False,
    )
    _resolved_probability_threshold: float | None = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        mode = _policy_mode(
            self.policy_mode
        )
        resolved_seed = _nonnegative_integer(
            self.seed,
            name="seed",
        )

        action_space = self.model.action_space

        if not isinstance(
            action_space,
            Discrete,
        ):
            raise PPOPolicySelectionError(
                "PPO evaluation currently requires "
                "a Discrete action space."
            )

        action_count = int(
            action_space.n
        )

        threshold_action: int | None = None
        threshold: float | None = None

        if mode == (
            EvaluationPolicyMode
            .PROBABILITY_THRESHOLD
        ):
            if self.probability_threshold is None:
                raise PPOPolicySelectionError(
                    "probability_threshold is required "
                    "for probability_threshold mode."
                )

            if action_count not in {2, 3}:
                raise PPOPolicySelectionError(
                    "probability_threshold mode requires "
                    "either two actions (FLAT plus one "
                    "direction) or three actions "
                    "(FLAT, LONG, SHORT)."
                )

            threshold = _probability_threshold(
                self.probability_threshold
            )

            if action_count == 2:
                if self.threshold_action is None:
                    raise PPOPolicySelectionError(
                        "threshold_action=1 is required "
                        "for a two-action threshold policy."
                    )

                threshold_action = (
                    _nonnegative_integer(
                        self.threshold_action,
                        name="threshold_action",
                    )
                )

                if threshold_action != 1:
                    raise PPOPolicySelectionError(
                        "A two-action threshold policy "
                        "requires threshold_action=1; "
                        "action 0 is the FLAT fallback."
                    )

            else:
                if self.threshold_action is not None:
                    raise PPOPolicySelectionError(
                        "threshold_action must be null for "
                        "a three-action threshold policy; "
                        "LONG or SHORT is selected "
                        "dynamically."
                    )

        else:
            if self.threshold_action is not None:
                raise PPOPolicySelectionError(
                    "threshold_action must be null unless "
                    "policy_mode is probability_threshold."
                )

            if self.probability_threshold is not None:
                raise PPOPolicySelectionError(
                    "probability_threshold must be null "
                    "unless policy_mode is "
                    "probability_threshold."
                )

        set_random_seed(
            resolved_seed,
            using_cuda=(
                self.model.device.type == "cuda"
            ),
        )

        self.seed = resolved_seed
        self._resolved_mode = mode
        self._action_count = action_count
        self._resolved_threshold_action = (
            threshold_action
        )
        self._resolved_probability_threshold = (
            threshold
        )

    @property
    def resolved_mode(
        self,
    ) -> EvaluationPolicyMode:
        return self._resolved_mode

    def _distribution(
        self,
        observation: np.ndarray,
    ) -> tuple[
        CategoricalDistribution,
        tuple[float, ...],
    ]:
        observation_tensor, _ = (
            self.model.policy.obs_to_tensor(
                observation
            )
        )

        with torch.no_grad():
            distribution = (
                self.model.policy.get_distribution(
                    observation_tensor
                )
            )

        if not isinstance(
            distribution,
            CategoricalDistribution,
        ):
            raise PPOPolicySelectionError(
                "PPO policy did not produce a "
                "categorical action distribution."
            )

        probabilities_tensor = (
            distribution.distribution.probs
        )

        if (
            probabilities_tensor.ndim != 2
            or probabilities_tensor.shape[0] != 1
        ):
            raise PPOPolicySelectionError(
                "Policy selection expects exactly "
                "one observation."
            )

        probabilities_array = (
            probabilities_tensor[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )

        if probabilities_array.shape != (
            self._action_count,
        ):
            raise PPOPolicySelectionError(
                "Policy probability count does not "
                "match the action space."
            )

        if not np.isfinite(
            probabilities_array
        ).all():
            raise PPOPolicySelectionError(
                "Policy probabilities contain "
                "NaN or infinity."
            )

        if np.any(
            probabilities_array < 0.0
        ):
            raise PPOPolicySelectionError(
                "Policy probabilities must be "
                "nonnegative."
            )

        probability_sum = float(
            np.sum(probabilities_array)
        )

        if not isclose(
            probability_sum,
            1.0,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise PPOPolicySelectionError(
                "Policy probabilities do not sum "
                "to one."
            )

        return (
            distribution,
            tuple(
                float(value)
                for value in probabilities_array
            ),
        )

    def select(
        self,
        observation: np.ndarray,
    ) -> PPOPolicyDecision:
        """Select one action for one unbatched observation."""
        distribution, probabilities = (
            self._distribution(
                observation
            )
        )

        threshold_met: bool | None = None

        if self._resolved_mode == (
            EvaluationPolicyMode
            .DETERMINISTIC_ARGMAX
        ):
            action = int(
                np.argmax(probabilities)
            )

        elif self._resolved_mode == (
            EvaluationPolicyMode
            .STOCHASTIC_SAMPLE
        ):
            with torch.no_grad():
                action_tensor = (
                    distribution.get_actions(
                        deterministic=False
                    )
                )

            action_array = (
                action_tensor
                .detach()
                .cpu()
                .numpy()
            )

            if action_array.size != 1:
                raise PPOPolicySelectionError(
                    "Stochastic policy returned more "
                    "than one action."
                )

            action = int(
                action_array.item()
            )

        else:
            threshold = (
                self
                ._resolved_probability_threshold
            )

            if threshold is None:
                raise PPOPolicySelectionError(
                    "Threshold selector is not "
                    "fully configured."
                )

            if self._action_count == 2:
                threshold_action = (
                    self._resolved_threshold_action
                )

                if threshold_action is None:
                    raise PPOPolicySelectionError(
                        "Two-action threshold selector "
                        "has no threshold_action."
                    )

                threshold_met = (
                    probabilities[
                        threshold_action
                    ]
                    >= threshold
                )

                action = (
                    threshold_action
                    if threshold_met
                    else 0
                )

            else:
                long_probability = probabilities[1]
                short_probability = probabilities[2]

                directional_action: int | None

                if (
                    long_probability
                    > short_probability
                ):
                    directional_action = 1
                elif (
                    short_probability
                    > long_probability
                ):
                    directional_action = 2
                else:
                    directional_action = None

                if directional_action is None:
                    threshold_met = False
                    action = 0
                else:
                    threshold_met = (
                        probabilities[
                            directional_action
                        ]
                        >= threshold
                    )

                    action = (
                        directional_action
                        if threshold_met
                        else 0
                    )

        if (
            action < 0
            or action >= self._action_count
        ):
            raise PPOPolicySelectionError(
                "Selected action is outside the "
                "action space."
            )

        return PPOPolicyDecision(
            action=action,
            probabilities=probabilities,
            selected_action_probability=(
                probabilities[action]
            ),
            threshold_met=threshold_met,
        )
