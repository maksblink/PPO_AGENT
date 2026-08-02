from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


class EarlyStoppingError(ValueError):
    """Raised when evaluation results cannot drive early stopping."""


@dataclass(frozen=True, slots=True)
class EarlyStoppingDecision:
    checkpoint_id: int
    evaluation_id: int
    score: float
    improved: bool
    best_checkpoint_id: int
    best_evaluation_id: int
    best_score: float
    evaluations_seen: int
    no_improvement_evals: int
    should_stop: bool


class BalancedScoreEarlyStopping:
    """Track strict balanced-score improvements and patience."""

    def __init__(self, *, patience: int) -> None:
        self.patience = self._positive_integer(
            patience,
            name="patience",
        )
        self.evaluations_seen = 0
        self.no_improvement_evals = 0
        self.best_checkpoint_id: int | None = None
        self.best_evaluation_id: int | None = None
        self.best_score: float | None = None

    @staticmethod
    def _positive_integer(
        value: Any,
        *,
        name: str,
    ) -> int:
        if isinstance(value, bool):
            raise EarlyStoppingError(
                f"{name} must be a positive integer."
            )

        try:
            result = int(value)
        except (TypeError, ValueError) as error:
            raise EarlyStoppingError(
                f"{name} must be a positive integer."
            ) from error

        if result != value or result < 1:
            raise EarlyStoppingError(
                f"{name} must be a positive integer."
            )

        return result

    @staticmethod
    def _finite_score(value: Any) -> float:
        if isinstance(value, bool):
            raise EarlyStoppingError(
                "balanced_score must be a finite number."
            )

        try:
            result = float(value)
        except (TypeError, ValueError) as error:
            raise EarlyStoppingError(
                "balanced_score must be a finite number."
            ) from error

        if not math.isfinite(result):
            raise EarlyStoppingError(
                "balanced_score must be a finite number."
            )

        return result

    def observe(
        self,
        *,
        checkpoint_id: int,
        evaluation_id: int,
        balanced_score: float,
    ) -> EarlyStoppingDecision:
        resolved_checkpoint_id = self._positive_integer(
            checkpoint_id,
            name="checkpoint_id",
        )
        resolved_evaluation_id = self._positive_integer(
            evaluation_id,
            name="evaluation_id",
        )
        score = self._finite_score(
            balanced_score
        )

        improved = (
            self.best_score is None
            or score > self.best_score
        )

        self.evaluations_seen += 1

        if improved:
            self.best_score = score
            self.best_checkpoint_id = (
                resolved_checkpoint_id
            )
            self.best_evaluation_id = (
                resolved_evaluation_id
            )
            self.no_improvement_evals = 0
        else:
            self.no_improvement_evals += 1

        assert self.best_score is not None
        assert self.best_checkpoint_id is not None
        assert self.best_evaluation_id is not None

        return EarlyStoppingDecision(
            checkpoint_id=resolved_checkpoint_id,
            evaluation_id=resolved_evaluation_id,
            score=score,
            improved=improved,
            best_checkpoint_id=(
                self.best_checkpoint_id
            ),
            best_evaluation_id=(
                self.best_evaluation_id
            ),
            best_score=self.best_score,
            evaluations_seen=(
                self.evaluations_seen
            ),
            no_improvement_evals=(
                self.no_improvement_evals
            ),
            should_stop=(
                self.no_improvement_evals
                >= self.patience
            ),
        )
