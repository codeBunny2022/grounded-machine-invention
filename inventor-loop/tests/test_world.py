"""Smoke tests. Run with: python -m pytest tests/

These do not call any LLM (LLM_PROVIDER=mock is the default) so they are
safe in CI.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("LLM_PROVIDER", "mock")

from inventor.agents.inventor_loop import InventorLoopAgent  # noqa: E402
from inventor.agents.token_baseline import TokenBaselineAgent  # noqa: E402
from inventor.memory import Memory  # noqa: E402
from inventor.metrics import compute_agent_metrics  # noqa: E402
from inventor.primitives import (  # noqa: E402
    LATENT_DIM,
    LENGTH_MAX,
    LENGTH_MIN,
    Plank,
    decode,
    encode,
)
from inventor.tasks import get_task  # noqa: E402
from inventor.world import WorldConfig, fitness, scene_bounds, simulate  # noqa: E402


def test_decode_returns_planks_within_bounds():
    cfg = WorldConfig()
    bounds = scene_bounds(cfg)
    rng = np.random.default_rng(0)
    z = rng.normal(0, 1, LATENT_DIM)
    planks = decode(z, bounds)
    for p in planks:
        assert bounds.x_min <= p.x <= bounds.x_max
        assert bounds.y_min <= p.y <= bounds.y_max
        assert LENGTH_MIN <= p.length <= LENGTH_MAX


def test_encode_decode_roundtrip_is_stable():
    cfg = WorldConfig()
    bounds = scene_bounds(cfg)
    planks = [
        Plank(x=200.0, y=100.0, length=120.0, angle=0.0),
        Plank(x=320.0, y=95.0, length=140.0, angle=0.1),
    ]
    recovered = decode(encode(planks, bounds), bounds)
    assert len(recovered) == len(planks)
    for a, b in zip(planks, recovered):
        assert abs(a.x - b.x) < 1.0
        assert abs(a.y - b.y) < 1.0
        assert abs(a.length - b.length) < 1.0


def test_simulate_no_planks_means_ball_falls_into_chasm():
    cfg = WorldConfig()
    outcome = simulate([], cfg)
    assert not outcome.reached
    assert outcome.fell_into_chasm or outcome.final_x < cfg.goal_x


def test_simulate_bridge_outperforms_empty_scene():
    """A big plank spanning the chasm should produce strictly more progress
    than the empty scene, regardless of exact physics tuning."""
    cfg = WorldConfig(chasm_width=200.0)
    no_planks = simulate([], cfg)
    bridge = [
        Plank(
            x=(cfg.left_platform_x_end + cfg.right_platform_x_start) / 2,
            y=cfg.platform_height,
            length=cfg.chasm_width + 100.0,
            angle=0.0,
        )
    ]
    with_bridge = simulate(bridge, cfg)
    assert fitness(with_bridge, cfg) > fitness(no_planks, cfg)


def test_inventor_loop_solves_easy():
    """The Inventor Loop should reliably solve the easy chasm via grounded
    latent iteration (this is hypothesis H1 in miniature)."""
    task = get_task("chasm_easy")
    memory = Memory()
    agent = InventorLoopAgent(n_iterations=15, skip_language=True)
    result = agent.run_episode(task, memory, seed=0)
    assert result.iteration_cost > 0
    assert result.outcome.reached
    assert result.fitness > 1.0


def test_token_baseline_runs_with_mock_llm():
    task = get_task("chasm_easy")
    memory = Memory()
    agent = TokenBaselineAgent(max_attempts=2)
    result = agent.run_episode(task, memory, seed=0)
    assert result.iteration_cost >= 1
    assert result.fitness >= 0.0


def test_metrics_handle_single_invention():
    task = get_task("chasm_easy")
    memory = Memory()
    agent = InventorLoopAgent(n_iterations=5, population=8, skip_language=True)
    r = agent.run_episode(task, memory, seed=0)
    memory.add(
        task=task.name,
        agent=agent.name,
        latent=r.latent,
        planks=r.planks,
        outcome=r.outcome,
        fitness=r.fitness,
        iteration_cost=r.iteration_cost,
    )
    m = compute_agent_metrics(memory.inventions)
    assert m.n_episodes == 1
    assert 0.0 <= m.usefulness <= 1.0
    assert m.originality == 0.0  # only one item


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
