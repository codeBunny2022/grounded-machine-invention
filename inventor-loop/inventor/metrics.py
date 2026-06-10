"""Implementation of the U-O-S framework and the cascading-invention metric.

We map paper §5 onto this PoC as follows:

* **Usefulness** — fraction of episodes where the invention solves the task.
* **Originality** — mean L2 distance from each accepted invention's latent
  to the nearest *prior* accepted invention's latent (per agent).
  Higher = more diverse inventions.
* **Surprise** — absolute error between the agent's *predicted* fitness
  for its final hypothesis and the *true* simulated fitness. Larger error
  on accepted inventions means the agent is producing ideas that surprise
  even its own internal model — a hallmark of genuine creativity in
  active-inference frameworks (paper §3.2).
* **Cascading Invention Rate** — the rate at which solving an earlier
  task in a curriculum reduces the iteration cost on a later task. We
  compute it as ``1 - (median_cost_with_memory / median_cost_without)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

from .memory import Invention


@dataclass
class AgentMetrics:
    agent: str
    n_episodes: int
    usefulness: float
    originality: float
    surprise: float
    median_iteration_cost: float  # internal sims (inventor) or attempts (token)
    median_llm_calls: float  # external LLM API calls — the H4 cost metric

    def as_row(self) -> dict:
        return {
            "agent": self.agent,
            "n_episodes": self.n_episodes,
            "usefulness": round(self.usefulness, 4),
            "originality": round(self.originality, 4),
            "surprise": round(self.surprise, 4),
            "median_iteration_cost": round(self.median_iteration_cost, 2),
            "median_llm_calls": round(self.median_llm_calls, 2),
        }


def _pairwise_min_distances(latents: List[np.ndarray]) -> List[float]:
    """For each latent, distance to its nearest neighbor *earlier* in the list."""
    dists: List[float] = []
    for i, z in enumerate(latents):
        if i == 0:
            continue
        prev = np.stack(latents[:i])
        d = float(np.min(np.linalg.norm(prev - z[None, :], axis=1)))
        dists.append(d)
    return dists


def compute_agent_metrics(
    inventions: Iterable[Invention],
    predicted_fitness: dict[int, float] | None = None,
) -> AgentMetrics:
    """Compute U-O-S for a single agent's inventions.

    ``predicted_fitness`` maps invention index (in iteration order) to the
    agent's own pre-simulation prediction. If absent, surprise is reported
    as 0.0 because the agent declined to commit to a prediction.
    """
    invs = list(inventions)
    if not invs:
        return AgentMetrics(
            agent="<empty>",
            n_episodes=0,
            usefulness=0.0,
            originality=0.0,
            surprise=0.0,
            median_iteration_cost=0.0,
            median_llm_calls=0.0,
        )

    agent = invs[0].agent
    successes = [i for i in invs if i.outcome.get("reached")]
    usefulness = len(successes) / len(invs)

    accepted_latents = [np.asarray(i.latent, dtype=np.float64) for i in successes]
    if len(accepted_latents) >= 2:
        originality = float(np.mean(_pairwise_min_distances(accepted_latents)))
    else:
        originality = 0.0

    if predicted_fitness:
        errs = [
            abs(predicted_fitness.get(idx, inv.fitness) - inv.fitness)
            for idx, inv in enumerate(invs)
        ]
        surprise = float(np.mean(errs))
    else:
        surprise = 0.0

    median_cost = float(np.median([i.iteration_cost for i in invs]))
    median_llm = float(np.median([getattr(i, "llm_calls", 0) for i in invs]))

    return AgentMetrics(
        agent=agent,
        n_episodes=len(invs),
        usefulness=usefulness,
        originality=originality,
        surprise=surprise,
        median_iteration_cost=median_cost,
        median_llm_calls=median_llm,
    )


def cascading_invention_rate(
    invs_with_memory: List[Invention],
    invs_without_memory: List[Invention],
) -> float:
    """Quantify how much accumulated memory accelerates later inventions.

    Compares median iteration cost of an agent given access to its prior
    inventions vs. the same agent run from scratch. A positive value means
    early inventions *cascaded* into faster downstream invention.
    """
    if not invs_with_memory or not invs_without_memory:
        return 0.0
    a = float(np.median([i.iteration_cost for i in invs_with_memory]))
    b = float(np.median([i.iteration_cost for i in invs_without_memory]))
    if b <= 0:
        return 0.0
    return float(1.0 - (a / b))
