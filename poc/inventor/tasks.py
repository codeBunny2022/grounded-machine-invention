"""Curriculum of *Cross the Chasm* task variants.

Tasks are graded by chasm width and primitive budget. Running an agent
across the full curriculum lets us measure the *cascading invention*
metric: did motifs discovered on easier tasks accelerate solutions on
harder ones?
"""

from __future__ import annotations

from dataclasses import dataclass

from .world import WorldConfig


@dataclass(frozen=True)
class Task:
    name: str
    cfg: WorldConfig
    plank_budget: int
    description: str


TASKS: dict[str, Task] = {
    "chasm_easy": Task(
        name="chasm_easy",
        cfg=WorldConfig(chasm_width=200.0),
        plank_budget=4,
        description=(
            "A 200-pixel chasm separates two platforms 100 pixels off the floor. "
            "Place at most 4 planks so that a ball released on the left platform "
            "ends up on the right platform within 10 simulated seconds."
        ),
    ),
    "chasm_medium": Task(
        name="chasm_medium",
        cfg=WorldConfig(chasm_width=300.0),
        plank_budget=5,
        description=(
            "A 300-pixel chasm separates two platforms 100 pixels off the floor. "
            "Place at most 5 planks so that a ball released on the left platform "
            "ends up on the right platform within 10 simulated seconds."
        ),
    ),
    "chasm_hard": Task(
        name="chasm_hard",
        cfg=WorldConfig(chasm_width=400.0),
        plank_budget=6,
        description=(
            "A 400-pixel chasm separates two platforms 100 pixels off the floor. "
            "Place at most 6 planks so that a ball released on the left platform "
            "ends up on the right platform within 10 simulated seconds."
        ),
    ),
}


def get_task(name: str) -> Task:
    if name not in TASKS:
        raise KeyError(f"Unknown task '{name}'. Available: {sorted(TASKS)}")
    return TASKS[name]
