# Inventor Loop — Proof of Concept

A minimal, reproducible proof-of-concept for the **Grounded Machine Invention
(GMI)** framework described in [`Research Paper`](../paper.md).

This PoC operationalizes the paper's central claim:

> Invention emerges when agents iterate over grounded latent world models
> **before** language generation.

We pit two agents against each other on the same 2D-physics invention task:

| Agent | Iteration substrate | Feedback loop |
|---|---|---|
| **Token Baseline** | Natural-language tokens (LLM → JSON) | LLM re-prompted with task description and execution log, up to *T* attempts. Every attempt is an external LLM round-trip. |
| **Inventor Loop** *(ours)* | Continuous latent of primitive placements | Cheap internal physics simulation drives a gradient-free (CMA-ES-flavoured) search. Language is invoked **zero** times during iteration (and at most once, at the end, to *describe* the converged invention). |

Both agents share the same decoder, simulator, primitive budget, and fitness
function, so the comparison isolates *where iteration happens*.

## The task: *Cross the Chasm*

A ball is released on the left of two raised platforms separated by a gap.
Each agent places a budget of **planks** (line segments with position, length,
and angle) so that, under deterministic rigid-body physics (`pymunk`, fixed
timestep, no stochastic forces), the ball reaches the right platform within a
fixed horizon. The task is a three-rung curriculum (`chasm_easy`,
`chasm_medium`, `chasm_hard`) of increasing gap width and plank budget.

The task admits qualitatively different solutions (bridges, ramps,
stepping-stones), which is what makes *originality* a meaningful quantity.

## How the Inventor Loop actually works

A uniform-noise search over plank parameters essentially never lands on the
narrow "valid bridge" manifold, so two ingredients make the loop work — both
faithful to the paper's thesis:

1. **Grounded decoding** (`inventor/primitives.py`, `inventor/world.py`). The
   latent is decoded into a *band* around the platform height and the playable
   span across the chasm — the embodied region where bridge-building planks
   belong, not the whole 800×600 world.
2. **Grounded generation + dense shaping.** Each iteration samples stochastic
   *bridge hypotheses* (varied plank count, length, and positions, guaranteed
   to overlap) and scores them with a `path_coverage` shaping term that
   rewards building a continuous surface even before the ball completes the
   crossing. The search refines the best hypothesis in latent space.

## Repository layout

```
poc/
├── inventor/
│   ├── world.py            2D physics world, scene bounds, coverage-shaped fitness
│   ├── primitives.py       Plank primitive + grounded latent <-> scene codec
│   ├── tasks.py            Curriculum (chasm widths, budgets, goals)
│   ├── memory.py           Persistent memory of accepted inventions
│   ├── metrics.py          Usefulness / Originality / Surprise / Cascade
│   ├── llm.py              Thin LLM wrapper (OpenAI / Anthropic / mock)
│   └── agents/
│       ├── base.py
│       ├── token_baseline.py
│       └── inventor_loop.py
├── scripts/
│   └── run_comparison.py   Head-to-head runner; writes JSON + a figure
├── tests/
│   └── test_world.py       Smoke tests (offline, no API key)
├── pyproject.toml
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

The project pins to Python 3.12. With [pyenv](https://github.com/pyenv/pyenv)
the included `.python-version` selects the `lab` virtualenv automatically:

```bash
cd poc
pyenv activate lab          # or: python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# optional, for clean imports without the sys.path shim:
pip install -e ".[llm,dev]"
cp .env.example .env        # only needed if you want a real LLM baseline
```

By default `LLM_PROVIDER=mock`, so everything below runs **offline with no API
key**. Set `LLM_PROVIDER=openai` (or `anthropic`) in `.env` to use a real model
for the token baseline.

## Run

```bash
# Smoke tests (offline, ~1s):
python -m pytest

# Inventor Loop only (no LLM at all):
python scripts/run_comparison.py --agents inventor --episodes 5

# Full head-to-head (mock LLM, fully offline):
python scripts/run_comparison.py \
    --agents token,inventor \
    --tasks chasm_easy,chasm_medium,chasm_hard \
    --episodes 8 --seed 0 --out results/main.json
```

Results are written as JSON to `results/` and a comparison plot is saved to
`figures/comparison.png`.

## What the metrics mean (paper §5 → this PoC)

- **Usefulness** — episode success rate (did the ball reach the goal?).
- **Originality** — mean L₂ distance, in the shared latent space, between each
  accepted invention and its nearest prior accepted invention by the same
  agent. Higher = more diverse inventions.
- **Surprise** — |predicted fitness − realized fitness|. The Inventor Loop's
  internal model in this PoC *is* the simulator, so its surprise is ≈ 0 **by
  construction** (a limitation we are explicit about).
- **External cost (H4)** — median number of LLM API calls per episode. This is
  the commensurate cross-agent cost: the token baseline spends one call per
  attempt; the Inventor Loop spends **zero** during iteration.
- **Cascading Invention Rate** — `1 − (cost_with_memory / cost_without_memory)`
  on the hardest rung.

## Representative results (offline, `--episodes 8 --seed 0`, `mock` LLM)

| Metric | Token Baseline | Inventor Loop |
|---|---|---|
| Usefulness (full attempt budget) | 1.00 | **1.00** |
| Usefulness (equal external budget = 1 call) | 0.33 | **1.00** |
| Originality | 1.53 | **2.97** |
| Surprise | 0.00 | 0.00 |
| Median external LLM calls / episode | 4 | **0** |
| Median internal sims / episode | — | ~180 |
| Cascading Invention Rate (`chasm_hard`) | — | 0.00 |

**Reading the results honestly.** The two paradigms are cleanly separated on
**originality** (the Inventor Loop's accepted inventions are roughly 2× more
diverse) and on **external cost** (zero LLM calls during iteration vs. several
per episode). When the external budget is *equalized* to a single call, the
token baseline collapses to 0.33 success while the Inventor Loop stays at 1.00
— this is hypothesis **H1**. Two hypotheses are **not** supported by this
minimal sandbox, and we say so:

- **Surprise ≈ 0** because the world model is the simulator itself.
- **Cascading Invention Rate ≈ 0** because the grounded generator already
  solves each rung cheaply, leaving no headroom for memory to accelerate
  later rungs. Distinguishing cascades needs a task with *multiple* solution
  modes, which is exactly the richer setting the full GMI architecture targets.

Numbers are deterministic given the seed but depend on the physics/search
constants in `world.py` and `agents/inventor_loop.py`.

## Limitations of this PoC

This is intentionally a *minimal* proof of concept. It demonstrates the
**iterate-before-language** principle on a deliberately constrained 2D
sandbox. The full vision in the paper requires learned (not hand-crafted)
world models, higher-dimensional invention spaces, and long-horizon,
open-ended task curricula. See §6.6 of the paper for the full discussion.
