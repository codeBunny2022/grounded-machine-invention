"""Token-only baseline agent.

This agent represents the dominant paradigm the paper critiques: every
iteration step happens *in language*. The agent prompts an LLM for a JSON
list of plank placements, executes them, and (if unsuccessful) re-prompts
with the failure log appended.

There is no internal physics simulator the agent uses for its own
hypothesizing — the LLM is the hypothesis generator and the physics
engine only provides the final pass/fail signal per attempt.
"""

from __future__ import annotations

import numpy as np

from .. import llm
from ..memory import Memory
from ..primitives import Plank, encode
from ..tasks import Task
from ..world import fitness, scene_bounds, simulate
from .base import Agent, EpisodeResult


class TokenBaselineAgent(Agent):
    name = "token_baseline"

    def __init__(self, max_attempts: int = 5) -> None:
        self.max_attempts = max_attempts

    def run_episode(
        self,
        task: Task,
        memory: Memory,
        seed: int = 0,
    ) -> EpisodeResult:
        cfg = task.cfg
        budget = task.plank_budget
        bounds = scene_bounds(cfg)
        history: list[dict] = []
        best_planks: list[Plank] = []
        best_outcome = None
        best_fit = -np.inf
        attempts = 0
        trace: list[dict] = []

        geometry = {
            "world_width": cfg.width,
            "world_height": cfg.height,
            "platform_height": cfg.platform_height,
            "left_end": cfg.left_platform_x_end,
            "right_start": cfg.right_platform_x_start,
            "goal_x": cfg.goal_x,
            "ball_x": cfg.ball_start[0],
        }
        for attempt in range(self.max_attempts):
            planks = llm.propose_planks(task.description, budget, history, geometry)
            if len(planks) > budget:
                planks = planks[:budget]
            outcome = simulate(planks, cfg)
            f = fitness(outcome, cfg, planks)
            attempts += 1

            history.append(
                {
                    "planks": [
                        {
                            "x": round(p.x, 1),
                            "y": round(p.y, 1),
                            "length": round(p.length, 1),
                            "angle_deg": round(np.degrees(p.angle), 1),
                        }
                        for p in planks
                    ],
                    "outcome": {
                        "reached": outcome.reached,
                        "max_x": outcome.max_x,
                        "final_x": outcome.final_x,
                    },
                }
            )
            trace.append({"attempt": attempt, "fitness": float(f)})

            if f > best_fit:
                best_fit = f
                best_planks = planks
                best_outcome = outcome
            if outcome.reached:
                break

        latent = encode(best_planks, bounds)
        return EpisodeResult(
            task=task.name,
            agent=self.name,
            planks=best_planks,
            latent=latent,
            outcome=best_outcome,
            fitness=float(best_fit),
            iteration_cost=attempts,
            llm_calls=attempts,  # every attempt is an external LLM round-trip
            predicted_fitness=None,
            description="",
            trace=trace,
        )
