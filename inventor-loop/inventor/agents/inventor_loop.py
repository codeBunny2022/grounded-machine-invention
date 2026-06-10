"""The Inventor Loop agent.

Operationalizes paper §4.2:

    Generate -> Simulate -> Evaluate -> Iterate

The agent maintains a continuous latent representation of the scene and
performs gradient-free search over it (a small CMA-ES-flavoured loop). The
key property: *all* iteration happens in latent space using cheap physics
simulation. Language is invoked exactly once at the end to project the
converged invention into a natural-language description (see
:func:`inventor.llm.describe_invention`).

Two design choices make the loop work on a sparse-reward physics task:

* **Grounded generation.** The latent is decoded into a band around the
  platform height (see :func:`inventor.world.scene_bounds`), and the initial
  hypothesis is a stochastic *bridge* sampled from that grounded prior rather
  than uniform noise. This is the embodied inductive bias of the plank
  primitive; uniform-noise search never finds the narrow bridge manifold.
* **Memory warm-start.** When the agent has solved related tasks before, it
  initialises the search mean from those stored inventions instead of
  re-deriving a hypothesis from scratch. This is what produces the
  cascading-invention effect: prior successes lower the simulation cost of
  later, harder rungs.
"""

from __future__ import annotations

import numpy as np

from .. import llm
from ..memory import Memory
from ..primitives import (
    LATENT_DIM,
    MAX_PLANKS,
    Plank,
    decode,
    encode,
    planks_from_json,
)
from ..tasks import Task
from ..world import WorldConfig, fitness, scene_bounds, simulate
from .base import Agent, EpisodeResult


def _sample_bridge(
    cfg: WorldConfig, budget: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample a grounded *bridge* hypothesis as a latent vector.

    Planks are spread across the chasm at the platform height with enough
    length to guarantee overlap, so the hypothesis reliably lets the ball
    cross. Diversity comes from varying the plank *count*, *length*, and exact
    *positions* — different valid bridges, not broken ones. This is the
    embodied prior of the plank primitive: a uniform-noise sample almost never
    lands on this narrow bridge manifold, which is precisely why a grounded
    generator (rather than blind search) is needed.
    """
    bounds = scene_bounds(cfg)
    le, rs = cfg.left_platform_x_end, cfg.right_platform_x_start
    span = rs - le
    n = int(rng.integers(max(2, budget - 1), budget + 1))
    n = max(2, min(n, MAX_PLANKS))
    xs = np.linspace(le - 10.0, rs + 10.0, n)
    spacing = (xs[-1] - xs[0]) / max(1, n - 1)
    # Guarantee overlap: each plank is longer than the gap to its neighbour.
    length = float(np.clip(spacing * rng.uniform(1.3, 1.6), 90.0, bounds.length_max))
    if length <= spacing:
        length = min(bounds.length_max, spacing + 30.0)
    planks = [
        Plank(
            x=float(x + rng.uniform(-6.0, 6.0)),
            y=float(cfg.platform_height + rng.uniform(-3.0, 3.0)),
            length=length,
            angle=0.0,
        )
        for x in xs
    ]
    return encode(planks, bounds)


class InventorLoopAgent(Agent):
    name = "inventor_loop"

    def __init__(
        self,
        n_iterations: int = 24,
        population: int = 18,
        sigma: float = 0.25,
        sigma_decay: float = 0.92,
        elite_frac: float = 0.3,
        patience: int = 5,
        warm_start_from_memory: bool = True,
        skip_language: bool = True,
    ) -> None:
        self.n_iterations = n_iterations
        self.population = population
        self.sigma_init = sigma
        self.sigma_decay = sigma_decay
        self.elite_frac = elite_frac
        self.patience = patience
        self.warm_start_from_memory = warm_start_from_memory
        # ``skip_language`` keeps the smoke-test runnable without an API key.
        self.skip_language = skip_language

    def _memory_seed(
        self, task: Task, cfg: WorldConfig, memory: Memory
    ) -> np.ndarray | None:
        """Re-encode a prior successful invention on *this* task as a latent.

        Latents are bounds-relative, so we re-encode from the stored invention's
        absolute plank geometry under the current task's bounds rather than
        re-using a raw latent (which would mis-scale a bridge built for a
        narrower chasm). Returns ``None`` when no relevant memory exists.
        """
        if not self.warm_start_from_memory:
            return None
        successes = [
            i for i in memory.successful(agent=self.name) if i.task == task.name
        ]
        if not successes:
            return None
        return encode(planks_from_json(successes[-1].planks), scene_bounds(cfg))

    def run_episode(
        self,
        task: Task,
        memory: Memory,
        seed: int = 0,
    ) -> EpisodeResult:
        rng = np.random.default_rng(seed)
        cfg = task.cfg
        budget = task.plank_budget
        bounds = scene_bounds(cfg)

        # Always seed from a fresh grounded bridge hypothesis (reliable). A
        # remembered solution, if any, is injected as an extra candidate so
        # memory can only *help*, never corrupt an otherwise valid search.
        mean = _sample_bridge(cfg, budget, rng)
        memory_seed = self._memory_seed(task, cfg, memory)
        sigma = self.sigma_init
        best_z = mean.copy()
        best_fit = -np.inf
        best_outcome = None
        best_planks: list[Plank] = []
        sims_used = 0
        sims_to_first_success: int | None = None
        elite_n = max(2, int(self.population * self.elite_frac))
        trace: list[dict] = []
        stale = 0

        for it in range(self.n_iterations):
            samples = mean + sigma * rng.standard_normal(
                (self.population, LATENT_DIM)
            )
            # Keep the incumbent...
            samples[0] = mean
            # ...keep proposing genuinely new grounded bridges...
            samples[1] = _sample_bridge(cfg, budget, rng)
            # ...and, when available, re-test the remembered invention.
            if memory_seed is not None and self.population > 2:
                samples[2] = memory_seed + 0.1 * rng.standard_normal(LATENT_DIM)

            scored: list[tuple[float, np.ndarray, list, object]] = []
            for s in samples:
                planks = decode(s, bounds)
                if len(planks) > budget:
                    planks = planks[:budget]
                outcome = simulate(planks, cfg)
                f = fitness(outcome, cfg, planks)
                sims_used += 1
                scored.append((f, s, planks, outcome))

            scored.sort(key=lambda t: t[0], reverse=True)
            elites = scored[:elite_n]
            weights = np.array(
                [np.log(elite_n + 0.5) - np.log(i + 1) for i in range(elite_n)]
            )
            weights /= weights.sum()
            mean = np.sum(
                [weights[i] * elites[i][1] for i in range(elite_n)], axis=0
            )

            top_fit, top_s, top_planks, top_outcome = scored[0]
            if top_fit > best_fit + 1e-4:
                best_fit = top_fit
                best_z = top_s.copy()
                best_outcome = top_outcome
                best_planks = top_planks
                stale = 0
                if top_outcome.reached and sims_to_first_success is None:
                    sims_to_first_success = sims_used
            else:
                stale += 1

            trace.append(
                {"iter": it, "best_fit": float(best_fit), "sims": sims_used}
            )

            sigma *= self.sigma_decay
            if (
                best_outcome is not None
                and best_outcome.reached
                and stale >= self.patience
            ):
                break

        # The Inventor Loop's internal model in this PoC *is* the simulator,
        # so its pre-projection fitness prediction equals the realised
        # fitness and Surprise is ~0 by construction (see paper §6.3).
        predicted = float(best_fit)

        description = ""
        llm_calls = 0
        if not self.skip_language:
            try:
                description = llm.describe_invention(best_planks, task.description)
                llm_calls = 1  # language is invoked exactly once, at the end
            except Exception as exc:  # pragma: no cover - network failures
                description = f"(language projection skipped: {exc})"

        return EpisodeResult(
            task=task.name,
            agent=self.name,
            planks=best_planks,
            latent=best_z,
            outcome=best_outcome,
            fitness=float(best_fit),
            iteration_cost=sims_used,
            llm_calls=llm_calls,
            predicted_fitness=predicted,
            description=description,
            trace=trace,
        )
