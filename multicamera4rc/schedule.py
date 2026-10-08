"""Explicit, checkpointed learning-rate phases for supervised continuation."""

from dataclasses import asdict, dataclass
import math


@dataclass
class LearningRatePlan:
    mode: str
    start_step: int
    end_step: int
    base_lrs: list[float]
    warmup_steps: int = 0
    min_ratio: float = .1
    warmup_start_ratio: float = 0.

    def __post_init__(self):
        if self.mode not in ("constant", "cosine"):
            raise ValueError("learning-rate mode must be constant/cosine")
        if self.start_step < 0 or self.end_step <= self.start_step:
            raise ValueError("learning-rate phase end must exceed its start")
        if self.warmup_steps < 0 or self.warmup_steps >= self.end_step-self.start_step:
            raise ValueError("warmup must fit inside the learning-rate phase")
        if not math.isfinite(self.min_ratio) or not 0 < self.min_ratio <= 1:
            raise ValueError("min learning-rate ratio must lie in (0,1]")
        if not math.isfinite(self.warmup_start_ratio) or not 0 <= self.warmup_start_ratio <= 1:
            raise ValueError("warmup start ratio must lie in [0,1]")
        if any(not math.isfinite(lr) or lr <= 0 for lr in self.base_lrs):
            raise ValueError("base learning rates must be finite and positive")

    def factor(self, step):
        position = max(0, step-self.start_step)
        if position < self.warmup_steps:
            return self.warmup_start_ratio + (1-self.warmup_start_ratio)*(position+1)/self.warmup_steps
        if self.mode == "constant":
            return 1.
        span = max(1, self.end_step-self.start_step-self.warmup_steps-1)
        progress = min(1., max(0., (position-self.warmup_steps)/span))
        return self.min_ratio+(1-self.min_ratio)*(1+math.cos(math.pi*progress))/2

    def apply(self, optimizer, step):
        if len(optimizer.param_groups) != len(self.base_lrs):
            raise ValueError("learning-rate plan does not match optimizer groups")
        for group, lr in zip(optimizer.param_groups, self.base_lrs):
            group["lr"] = lr*self.factor(step)

    def to_config(self):
        return asdict(self)
