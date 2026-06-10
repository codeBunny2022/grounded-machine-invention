# Grounded Machine Invention — Inventor Loop

A reproducible physics sandbox that compares two invention agents on the same task:

- **Inventor Loop** — iterates in a grounded latent space using internal simulation
- **Token baseline** — iterates via LLM prompts and JSON plank placements

All code, tests, figures, and experiment results live in [`inventor-loop/`](inventor-loop/).

```bash
cd inventor-loop
pip install -r requirements.txt
python -m pytest
python scripts/run_comparison.py --agents token,inventor --episodes 8 --seed 0
```
