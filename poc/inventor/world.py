"""Headless 2D physics world for the *Cross the Chasm* invention task.

The world is a deterministic, seeded ``pymunk`` simulation. Both the
Inventor Loop and the token baseline call :func:`simulate` to obtain a
ground-truth outcome for any candidate invention. Determinism (fixed
timestep + fixed iteration count + no randomness in physics) is what makes
the U-O-S metrics in :mod:`inventor.metrics` meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pymunk

from .primitives import LENGTH_MAX, LENGTH_MIN, Plank, SceneBounds


@dataclass(frozen=True)
class WorldConfig:
    width: float = 800.0
    height: float = 600.0
    platform_height: float = 100.0
    left_platform_width: float = 150.0
    right_platform_width: float = 150.0
    chasm_width: float = 300.0  # distance between inner edges of platforms
    ball_radius: float = 10.0
    ball_mass: float = 1.0
    ball_initial_vx: float = 220.0
    gravity: float = -900.0
    sim_dt: float = 1.0 / 60.0
    sim_steps: int = 600  # 10 seconds of simulated time
    platform_friction: float = 0.05  # slippery so the ball slides off
    plank_friction: float = 0.9
    ball_friction: float = 0.5

    @property
    def left_platform_x_end(self) -> float:
        return self.left_platform_width

    @property
    def right_platform_x_start(self) -> float:
        return self.left_platform_width + self.chasm_width

    @property
    def right_platform_x_end(self) -> float:
        return (
            self.left_platform_width + self.chasm_width + self.right_platform_width
        )

    @property
    def ball_start(self) -> tuple[float, float]:
        return (
            self.left_platform_width * 0.5,
            self.platform_height + self.ball_radius + 1.0,
        )

    @property
    def goal_x(self) -> float:
        # Goal is the inner edge of the right platform.
        return self.right_platform_x_start


def scene_bounds(cfg: "WorldConfig") -> SceneBounds:
    """The grounded band+span planks are decoded into for this task.

    Planks are confined to a horizontal band straddling the platform height
    and to the playable span from under the ball to just past the goal edge.
    This is the embodied prior over *where bridge-building planks belong*.
    """
    return SceneBounds(
        x_min=cfg.left_platform_width * 0.4,
        x_max=cfg.right_platform_x_start + 50.0,
        y_min=cfg.platform_height - 50.0,
        y_max=cfg.platform_height + 90.0,
        length_min=LENGTH_MIN,
        length_max=LENGTH_MAX,
    )


@dataclass
class SimulationOutcome:
    reached: bool
    final_x: float
    final_y: float
    max_x: float
    steps_to_reach: int  # ``sim_steps`` if never reached
    fell_into_chasm: bool


def _build_static_world(space: pymunk.Space, cfg: WorldConfig) -> None:
    static_body = space.static_body
    floor_y = 0.0

    # Left platform top edge
    left_top = pymunk.Segment(
        static_body,
        (0.0, cfg.platform_height),
        (cfg.left_platform_x_end, cfg.platform_height),
        2.0,
    )
    # Right platform top edge
    right_top = pymunk.Segment(
        static_body,
        (cfg.right_platform_x_start, cfg.platform_height),
        (cfg.right_platform_x_end, cfg.platform_height),
        2.0,
    )
    # World floor (catches anything that falls off, marks chasm failure)
    floor = pymunk.Segment(
        static_body, (0.0, floor_y), (cfg.width, floor_y), 2.0
    )
    # Inner cliff edges (so ball doesn't tunnel)
    left_cliff = pymunk.Segment(
        static_body,
        (cfg.left_platform_x_end, cfg.platform_height),
        (cfg.left_platform_x_end, floor_y),
        2.0,
    )
    right_cliff = pymunk.Segment(
        static_body,
        (cfg.right_platform_x_start, cfg.platform_height),
        (cfg.right_platform_x_start, floor_y),
        2.0,
    )

    for seg in (left_top, right_top, floor, left_cliff, right_cliff):
        seg.friction = cfg.platform_friction
        seg.elasticity = 0.1
        space.add(seg)


def _add_planks(
    space: pymunk.Space, planks: List[Plank], cfg: WorldConfig
) -> None:
    for p in planks:
        if p.length <= 0:
            continue
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        body.position = (p.x, p.y)
        body.angle = p.angle
        seg = pymunk.Segment(body, (-p.length / 2, 0.0), (p.length / 2, 0.0), 4.0)
        seg.friction = cfg.plank_friction
        seg.elasticity = 0.05
        space.add(body, seg)


def simulate(planks: List[Plank], cfg: WorldConfig | None = None) -> SimulationOutcome:
    """Run a single deterministic episode and return the outcome."""
    cfg = cfg or WorldConfig()
    space = pymunk.Space()
    space.gravity = (0.0, cfg.gravity)
    space.iterations = 30

    _build_static_world(space, cfg)
    _add_planks(space, planks, cfg)

    moment = pymunk.moment_for_circle(cfg.ball_mass, 0.0, cfg.ball_radius)
    ball_body = pymunk.Body(cfg.ball_mass, moment)
    ball_body.position = cfg.ball_start
    ball_body.velocity = (cfg.ball_initial_vx, 0.0)
    ball_shape = pymunk.Circle(ball_body, cfg.ball_radius)
    ball_shape.friction = cfg.ball_friction
    ball_shape.elasticity = 0.2
    space.add(ball_body, ball_shape)

    max_x = ball_body.position.x
    steps_to_reach = cfg.sim_steps
    reached = False

    for step in range(cfg.sim_steps):
        space.step(cfg.sim_dt)
        x = ball_body.position.x
        if x > max_x:
            max_x = x
        if not reached and x >= cfg.goal_x and ball_body.position.y >= cfg.platform_height - 5.0:
            reached = True
            steps_to_reach = step
            break

    final_x = float(ball_body.position.x)
    final_y = float(ball_body.position.y)
    fell = (not reached) and final_y < cfg.platform_height - 30.0
    return SimulationOutcome(
        reached=reached,
        final_x=final_x,
        final_y=final_y,
        max_x=float(max_x),
        steps_to_reach=int(steps_to_reach),
        fell_into_chasm=fell,
    )


def path_coverage(
    planks: List[Plank], cfg: WorldConfig | None = None, samples: int = 36
) -> float:
    """Fraction of the chasm span spanned by a near-platform-height surface.

    This is a dense shaping signal: it rewards *structure* (a continuous
    walkable surface) independently of whether the ball happened to make it
    across on this particular run. Without it the reward is almost flat until
    a bridge is complete, and gradient-free search has nothing to climb.
    """
    cfg = cfg or WorldConfig()
    if not planks:
        return 0.0
    xs = np.linspace(cfg.left_platform_x_end, cfg.right_platform_x_start, samples)
    lo = cfg.platform_height - 70.0
    hi = cfg.platform_height + 45.0
    covered = 0
    for x in xs:
        for p in planks:
            half = (p.length / 2.0) * np.cos(p.angle)
            if p.x - half <= x <= p.x + half:
                surface_y = p.y + np.tan(p.angle) * (x - p.x)
                if lo <= surface_y <= hi:
                    covered += 1
                    break
    return covered / samples


def fitness(
    outcome: SimulationOutcome,
    cfg: WorldConfig | None = None,
    planks: List[Plank] | None = None,
) -> float:
    """A scalar score both agents maximize.

    Reaching the goal is worth 1.0 plus a time bonus; otherwise the agent
    earns partial credit for how far across the chasm the ball travelled,
    plus a dense :func:`path_coverage` shaping term that rewards building a
    continuous surface even before the ball completes the crossing. Falling
    into the chasm is penalized.
    """
    cfg = cfg or WorldConfig()
    if outcome.reached:
        # Reaching the goal must strictly dominate *any* non-reaching scene:
        # the base of 2.0 sits above the maximum unsolved score (progress
        # <= 0.99 plus coverage shaping <= 0.4), so a slowly-reached bridge
        # always beats a merely high-coverage one that never crossed.
        time_bonus = 1.0 - (outcome.steps_to_reach / cfg.sim_steps)
        return 2.0 + 0.5 * time_bonus
    progress = max(0.0, outcome.max_x - cfg.left_platform_x_end) / max(
        1e-6, cfg.chasm_width
    )
    score = float(np.clip(progress, 0.0, 0.99))
    if outcome.fell_into_chasm:
        score -= 0.2
    if planks is not None:
        score += 0.4 * path_coverage(planks, cfg)
    return float(score)
