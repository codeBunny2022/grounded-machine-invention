"""Head-to-head comparison runner.

Example:

    python scripts/run_comparison.py \\
        --agents inventor,token \\
        --tasks chasm_easy,chasm_medium,chasm_hard \\
        --episodes 5 --seed 0 \\
        --out results/main.json

Writes a JSON record of all episodes plus a U-O-S summary table, and
saves a comparison plot under ``figures/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inventor.agents.inventor_loop import InventorLoopAgent  # noqa: E402
from inventor.agents.token_baseline import TokenBaselineAgent  # noqa: E402
from inventor.memory import Memory  # noqa: E402
from inventor.metrics import (  # noqa: E402
    cascading_invention_rate,
    compute_agent_metrics,
)
from inventor.tasks import TASKS, get_task  # noqa: E402

AGENTS = {
    "inventor": InventorLoopAgent,
    "token": TokenBaselineAgent,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--agents", default="inventor", help="comma-separated agent ids")
    p.add_argument(
        "--tasks", default="chasm_easy,chasm_medium,chasm_hard",
        help="comma-separated task ids",
    )
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(ROOT / "results" / "comparison.json"))
    p.add_argument(
        "--inventor-iterations", type=int, default=20,
        help="iterations for the Inventor Loop",
    )
    p.add_argument(
        "--token-attempts", type=int, default=5,
        help="max attempts for the token baseline per episode",
    )
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def build_agent(agent_id: str, args: argparse.Namespace):
    if agent_id == "inventor":
        return InventorLoopAgent(
            n_iterations=args.inventor_iterations,
            skip_language=True,
        )
    if agent_id == "token":
        return TokenBaselineAgent(max_attempts=args.token_attempts)
    raise KeyError(f"unknown agent {agent_id!r}")


def measure_cascade(tasks: list[str], args: argparse.Namespace) -> dict:
    """Quantify the cascading-invention effect for the Inventor Loop.

    The hardest rung is solved twice: once *cold* (no access to prior
    inventions) and once *warm* (memory pre-populated by solving the easier
    rungs first). A positive rate means memory of earlier successes lowered
    the simulation cost of the harder invention.
    """
    hard = get_task(tasks[-1])
    task_offset = {name: i for i, name in enumerate(sorted(TASKS))}

    cold_invs = []
    for ep in range(args.episodes):
        agent = InventorLoopAgent(
            n_iterations=args.inventor_iterations,
            warm_start_from_memory=False,
            skip_language=True,
        )
        mem = Memory()
        seed = args.seed + ep * 1000 + task_offset[hard.name] * 100
        r = agent.run_episode(hard, mem, seed=seed)
        cold_invs.append(
            mem.add(
                task=hard.name, agent=agent.name, latent=r.latent,
                planks=r.planks, outcome=r.outcome, fitness=r.fitness,
                iteration_cost=r.iteration_cost,
            )
        )

    warm_invs = []
    for ep in range(args.episodes):
        agent = InventorLoopAgent(
            n_iterations=args.inventor_iterations,
            warm_start_from_memory=True,
            skip_language=True,
        )
        mem = Memory()
        # Pre-populate memory by solving the easier rungs first.
        for warm_task_id in tasks[:-1]:
            wt = get_task(warm_task_id)
            seed = args.seed + ep * 1000 + task_offset[warm_task_id] * 100
            wr = agent.run_episode(wt, mem, seed=seed)
            mem.add(
                task=wt.name, agent=agent.name, latent=wr.latent,
                planks=wr.planks, outcome=wr.outcome, fitness=wr.fitness,
                iteration_cost=wr.iteration_cost,
            )
        seed = args.seed + ep * 1000 + task_offset[hard.name] * 100
        r = agent.run_episode(hard, mem, seed=seed)
        warm_invs.append(
            mem.add(
                task=hard.name, agent=agent.name, latent=r.latent,
                planks=r.planks, outcome=r.outcome, fitness=r.fitness,
                iteration_cost=r.iteration_cost,
            )
        )

    rate = cascading_invention_rate(warm_invs, cold_invs)
    cold_cost = float(np.median([i.iteration_cost for i in cold_invs]))
    warm_cost = float(np.median([i.iteration_cost for i in warm_invs]))
    return {
        "task": hard.name,
        "rate": rate,
        "cold_cost": cold_cost,
        "warm_cost": warm_cost,
    }


def main() -> int:
    args = parse_args()
    agents = [a.strip() for a in args.agents.split(",") if a.strip()]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    all_episodes = []
    metrics_rows = []

    # Deterministic per-task offset (Python's ``hash`` is salted per process
    # and would silently break reproducibility across runs).
    task_offset = {name: i for i, name in enumerate(sorted(TASKS))}

    for agent_id in agents:
        memory = Memory()
        predicted: dict[int, float] = {}
        for task_id in tasks:
            task = get_task(task_id)
            agent = build_agent(agent_id, args)
            for ep in range(args.episodes):
                seed = args.seed + ep * 1000 + task_offset[task_id] * 100
                result = agent.run_episode(task, memory, seed=seed)
                if result.predicted_fitness is not None:
                    predicted[len(memory.inventions)] = result.predicted_fitness
                memory.add(
                    task=task.name,
                    agent=agent.name,
                    latent=result.latent,
                    planks=result.planks,
                    outcome=result.outcome,
                    fitness=result.fitness,
                    iteration_cost=result.iteration_cost,
                    llm_calls=result.llm_calls,
                )
                all_episodes.append(
                    {
                        "agent": agent.name,
                        "task": task.name,
                        "episode": ep,
                        "seed": seed,
                        "fitness": result.fitness,
                        "iteration_cost": result.iteration_cost,
                        "llm_calls": result.llm_calls,
                        "reached": bool(result.outcome.reached),
                        "predicted_fitness": result.predicted_fitness,
                        "trace": result.trace,
                    }
                )
                print(
                    f"[{agent.name} | {task.name} | ep {ep}] "
                    f"reached={result.outcome.reached} "
                    f"fit={result.fitness:.3f} "
                    f"cost={result.iteration_cost}"
                )

        m = compute_agent_metrics(memory.inventions, predicted or None)
        metrics_rows.append(m.as_row())

    cascade = None
    if "inventor" in agents and len(tasks) >= 2:
        cascade = measure_cascade(tasks, args)
        print(
            f"\nCascading Invention Rate (inventor, '{tasks[-1]}'): "
            f"{cascade['rate']:.3f}  "
            f"(cold median cost {cascade['cold_cost']:.0f} vs "
            f"warm {cascade['warm_cost']:.0f})"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "episodes": all_episodes,
                "metrics": metrics_rows,
                "cascade": cascade,
            },
            indent=2,
            default=float,
        )
    )
    print("\nU-O-S summary:")
    for row in metrics_rows:
        print(" ", row)
    print(f"\nWrote {out_path}")

    if not args.no_plot:
        try:
            _make_plot(all_episodes, ROOT / "figures" / "comparison.png")
        except Exception as exc:  # pragma: no cover - plotting is optional
            print(f"(plot skipped: {exc})")

    return 0


def _make_plot(episodes: list[dict], path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    agents = sorted({e["agent"] for e in episodes})
    tasks = sorted({e["task"] for e in episodes})

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    width = 0.35
    x = np.arange(len(tasks))
    for i, agent in enumerate(agents):
        success = []
        calls = []
        for t in tasks:
            es = [e for e in episodes if e["agent"] == agent and e["task"] == t]
            success.append(np.mean([e["reached"] for e in es]) if es else 0.0)
            calls.append(
                np.median([e.get("llm_calls", 0) for e in es]) if es else 0.0
            )
        axes[0].bar(x + i * width, success, width, label=agent)
        axes[1].bar(x + i * width, calls, width, label=agent)

    for ax, title, ylabel in [
        (axes[0], "Usefulness (success rate)", "fraction reached"),
        (axes[1], "External cost (median LLM calls)", "LLM API calls / episode"),
    ]:
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(tasks, rotation=15)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Wrote plot {path}")


if __name__ == "__main__":
    raise SystemExit(main())
