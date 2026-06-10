"""Persistent memory of accepted inventions.

Each invention is stored together with the latent that produced it, the
decoded plank list, and the simulation outcome. Memory exposes simple
nearest-neighbor queries used by the originality metric and by the
cascading-invention measurement.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Sequence

import numpy as np

from .primitives import Plank, planks_to_json
from .world import SimulationOutcome


@dataclass
class Invention:
    task: str
    agent: str
    latent: List[float]
    planks: List[dict]  # serialized, see ``planks_to_json``
    outcome: dict
    fitness: float
    iteration_cost: int  # number of internal sim or LLM calls used
    llm_calls: int = 0  # external LLM API calls (the H4 cost metric)


@dataclass
class Memory:
    inventions: List[Invention] = field(default_factory=list)

    def add(
        self,
        *,
        task: str,
        agent: str,
        latent: np.ndarray,
        planks: Sequence[Plank],
        outcome: SimulationOutcome,
        fitness: float,
        iteration_cost: int,
        llm_calls: int = 0,
    ) -> Invention:
        inv = Invention(
            task=task,
            agent=agent,
            latent=list(map(float, np.asarray(latent).ravel())),
            planks=planks_to_json(list(planks)),
            outcome=asdict(outcome),
            fitness=float(fitness),
            iteration_cost=int(iteration_cost),
            llm_calls=int(llm_calls),
        )
        self.inventions.append(inv)
        return inv

    def for_agent(self, agent: str) -> List[Invention]:
        return [i for i in self.inventions if i.agent == agent]

    def successful(self, agent: str | None = None) -> List[Invention]:
        items = self.inventions if agent is None else self.for_agent(agent)
        return [i for i in items if i.outcome.get("reached")]

    def min_distance(self, latent: np.ndarray, agent: str | None = None) -> float:
        """Smallest L2 distance between ``latent`` and any stored invention."""
        items = self.inventions if agent is None else self.for_agent(agent)
        if not items:
            return float("inf")
        z = np.asarray(latent, dtype=np.float64).ravel()
        dists = [
            float(np.linalg.norm(z - np.asarray(i.latent, dtype=np.float64)))
            for i in items
        ]
        return min(dists)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps([asdict(i) for i in self.inventions], indent=2)
        )

    @classmethod
    def load(cls, path: str | Path) -> "Memory":
        data = json.loads(Path(path).read_text())
        return cls(inventions=[Invention(**d) for d in data])
