"""Pluggable LLM wrapper.

Three backends are supported:

* ``openai``    — uses ``OPENAI_API_KEY`` and ``OPENAI_MODEL``
* ``anthropic`` — uses ``ANTHROPIC_API_KEY`` and ``ANTHROPIC_MODEL``
* ``mock``      — deterministic offline generator. Lets the smoke test and
                  CI run without network or credentials.

The module exposes two entry points:

* :func:`propose_planks` — token-baseline proposal (returns a list of
  primitive placements). The baseline calls this once per attempt.
* :func:`describe_invention` — final natural-language description of a
  converged invention. The Inventor Loop calls this *exactly once* at the
  end, which is the operational meaning of "language only after grounded
  iteration".
"""

from __future__ import annotations

import json
import os
import random
from typing import Any

from dotenv import load_dotenv

from .primitives import MAX_PLANKS, Plank, planks_from_json, planks_to_json

load_dotenv()


PROVIDER = os.environ.get("LLM_PROVIDER", "mock").lower()


_PROPOSAL_SYSTEM = (
    "You are a physics-aware engineering assistant. You will be given a "
    "task description and must respond with a JSON array of plank "
    "placements. Each plank is an object with keys: x, y, length, "
    "angle_deg. Coordinates are in pixels with the origin at the bottom "
    "left of an 800x600 world. Return ONLY the JSON array."
)


def _format_history(history: list[dict[str, Any]]) -> str:
    if not history:
        return "(no previous attempts)"
    lines = []
    for i, h in enumerate(history):
        outcome = h.get("outcome", {})
        lines.append(
            f"Attempt {i + 1}: planks={h.get('planks')}, "
            f"reached={outcome.get('reached')}, "
            f"max_x={outcome.get('max_x'):.1f}, "
            f"final_x={outcome.get('final_x'):.1f}"
        )
    return "\n".join(lines)


def _default_geometry() -> dict[str, float]:
    return {
        "world_width": 800.0,
        "world_height": 600.0,
        "platform_height": 100.0,
        "left_end": 150.0,
        "right_start": 450.0,
        "goal_x": 450.0,
        "ball_x": 75.0,
    }


def _build_proposal_prompt(
    task_description: str,
    plank_budget: int,
    history: list[dict[str, Any]],
    geom: dict[str, float],
) -> str:
    return (
        f"Task:\n{task_description}\n\n"
        f"Plank budget: at most {plank_budget} planks (max {MAX_PLANKS}).\n"
        f"World is {geom['world_width']:.0f} wide by {geom['world_height']:.0f} "
        f"tall. Left platform spans x in [0, {geom['left_end']:.0f}] at "
        f"y={geom['platform_height']:.0f}. Right platform starts at "
        f"x={geom['right_start']:.0f} at y={geom['platform_height']:.0f}. "
        f"The ball spawns near ({geom['ball_x']:.0f}, "
        f"{geom['platform_height'] + 10:.0f}) and the goal x is "
        f"{geom['goal_x']:.0f}.\n\n"
        f"History of attempts so far:\n{_format_history(history)}\n\n"
        f"Propose your next attempt as a JSON array of plank objects."
    )


def _mock_propose(
    task_description: str,
    plank_budget: int,
    history: list[dict[str, Any]],
    geom: dict[str, float],
) -> list[Plank]:
    """Deterministic fallback used when no API key is configured.

    Heuristic: lay flat planks in a straight line spanning the chasm. This
    is a reasonable but unimaginative token-style baseline that nudges
    placements based on how far the previous attempt's ball travelled.
    """
    attempt = len(history)
    rng = random.Random(attempt)
    n = min(plank_budget, MAX_PLANKS)
    start = geom["left_end"]
    end = geom["right_start"]
    span = end - start
    step = span / max(1, n - 1) if n > 1 else span
    # A token proposer that reasons in language: it starts with a plausible
    # but under-length guess and lengthens its planks each time the failure
    # log shows the ball fell short. Several round-trips are needed before the
    # gaps close — which is exactly the external-call cost the paper measures.
    length = 60.0 + 11.0 * attempt
    planks = []
    for i in range(n):
        x = start + step * i + rng.uniform(-5, 5)
        planks.append(
            Plank(
                x=x,
                y=geom["platform_height"] + rng.uniform(-3, 3),
                length=length,
                angle=0.0,
            )
        )
    return planks


def _openai_complete(system: str, user: str) -> str:
    from openai import OpenAI  # type: ignore

    client = OpenAI()
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.7,
    )
    return resp.choices[0].message.content or ""


def _anthropic_complete(system: str, user: str) -> str:
    from anthropic import Anthropic  # type: ignore

    client = Anthropic()
    model = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-haiku-latest")
    resp = client.messages.create(
        model=model,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(block.text for block in resp.content if hasattr(block, "text"))


def _complete(system: str, user: str) -> str:
    if PROVIDER == "openai":
        return _openai_complete(system, user)
    if PROVIDER == "anthropic":
        return _anthropic_complete(system, user)
    raise RuntimeError(f"Unsupported provider {PROVIDER!r}")


def _extract_json_array(text: str) -> list[dict[str, Any]]:
    """Best-effort extraction of the first JSON array in ``text``."""
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end <= start:
        return []
    blob = text[start : end + 1]
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def propose_planks(
    task_description: str,
    plank_budget: int,
    history: list[dict[str, Any]],
    geometry: dict[str, float] | None = None,
) -> list[Plank]:
    """Token-baseline proposal step. Uses the mock backend if configured."""
    geom = geometry or _default_geometry()
    if PROVIDER == "mock":
        return _mock_propose(task_description, plank_budget, history, geom)

    user = _build_proposal_prompt(task_description, plank_budget, history, geom)
    raw = _complete(_PROPOSAL_SYSTEM, user)
    data = _extract_json_array(raw)
    return planks_from_json(data)[:plank_budget]


def describe_invention(planks: list[Plank], task_description: str) -> str:
    """One-shot natural-language description of a converged invention."""
    serialized = planks_to_json(planks)
    if PROVIDER == "mock":
        return f"Mock description of {len(serialized)}-plank invention."

    system = (
        "You are a science writer. In one short paragraph, describe the "
        "engineering principle behind the provided plank configuration."
    )
    user = (
        f"Task: {task_description}\n"
        f"Planks (x,y,length,angle_deg): {json.dumps(serialized)}"
    )
    return _complete(system, user).strip()
