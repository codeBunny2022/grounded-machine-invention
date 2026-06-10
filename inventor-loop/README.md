# Inventor Loop sandbox

Python implementation of the physics experiment from Section 6 of the GMI paper. See the [repository README](../README.md) for results and context.

## Agents

**Token baseline** (`inventor/agents/token_baseline.py`)  
Calls an LLM each attempt. The model returns a JSON list of plank placements; failures are appended to the prompt and the model tries again, up to `T` times.

**Inventor Loop** (`inventor/agents/inventor_loop.py`)  
Maintains a 30-dimensional latent (6 planks × 5 parameters). Each iteration decodes candidates, runs pymunk, and updates the search distribution. Language is optional and limited to a single description call at the end.

## Key modules

| File | Role |
|---|---|
| `inventor/world.py` | pymunk simulation, fitness, path-coverage shaping |
| `inventor/primitives.py` | `Plank` type, `encode` / `decode` between latent and scene |
| `inventor/tasks.py` | `chasm_easy`, `chasm_medium`, `chasm_hard` definitions |
| `inventor/metrics.py` | Usefulness, originality, surprise, cascade rate |
| `inventor/memory.py` | Stores accepted inventions for warm-start |
| `inventor/llm.py` | OpenAI, Anthropic, or mock proposer |
| `scripts/run_comparison.py` | Runs both agents, writes JSON + figure |

## Setup

```bash
cd inventor-loop
pip install -r requirements.txt
cp .env.example .env   # optional; mock mode needs no keys
```

Python 3.10+. The included `.python-version` targets a pyenv env named `lab` if you use that.

## Run

```bash
python -m pytest
python scripts/run_comparison.py --agents token,inventor --episodes 8 --seed 0
```

Outputs:

- `results/main.json` (or path passed to `--out`)
- `figures/comparison.png` unless `--no-plot` is set

## Metrics (U-O-S + cascade)

- **Usefulness** — fraction of episodes where the ball reaches the goal.
- **Originality** — mean nearest-neighbor distance between successful inventions in latent space.
- **Surprise** — |predicted fitness − realized fitness|. Near zero here because the internal model is the simulator.
- **Cascade rate** — whether memory of earlier successes lowers sim cost on harder rungs (`measure_cascade` in the runner).

## Reproducing the bundled results

```bash
LLM_PROVIDER=mock python scripts/run_comparison.py \
  --agents token,inventor \
  --tasks chasm_easy,chasm_medium,chasm_hard \
  --episodes 8 --seed 0 --out results/main.json
```

This should match the numbers in the root README within rounding.
