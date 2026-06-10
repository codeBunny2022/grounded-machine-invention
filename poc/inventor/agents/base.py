"""Common agent interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from ..memory import Memory
from ..primitives import Plank
from ..tasks import Task
from ..world import SimulationOutcome


@dataclass
class EpisodeResult:
    task: str
    agent: str
    planks: List[Plank]
    latent: np.ndarray
    outcome: SimulationOutcome
    fitness: float
    iteration_cost: int  # internal physics simulations OR LLM calls
    llm_calls: int = 0  # *external* LLM API calls (the H4 cost metric)
    predicted_fitness: float | None = None
    description: str = ""
    trace: list[dict] = field(default_factory=list)


class Agent:
    name: str = "base"

    def run_episode(
        self,
        task: Task,
        memory: Memory,
        seed: int = 0,
    ) -> EpisodeResult:  # pragma: no cover - interface
        raise NotImplementedError
