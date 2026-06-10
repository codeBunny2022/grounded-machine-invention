"""Invention primitives and the latent <-> primitive encoding.

A *scene* is encoded as a flat parameter vector of dimension
    ``MAX_PLANKS * PARAMS_PER_PLANK``.

Each plank uses 5 floats interpreted as:
    [use_gate, x_norm, y_norm, length_norm, angle_norm]

This continuous representation is the substrate the Inventor Loop iterates
in. :func:`decode` maps a latent deterministically to physical placements and
:func:`encode` is its (approximate) inverse, so a concrete scene can be lifted
back into latent space. Both agents share the same codec and the same
:class:`SceneBounds`, which is what makes the cross-agent originality metric
an apples-to-apples comparison.

The decoder is *grounded*: rather than scattering planks anywhere in the
800x600 world, it confines them to a horizontal band around the platform
height and to the playable span across the chasm. This band is the natural
operating region of the plank primitive and is the embodied inductive bias
the paper argues for (planks are for building paths near the platforms, not
for floating in mid-air).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

MAX_PLANKS = 6
PARAMS_PER_PLANK = 5
LATENT_DIM = MAX_PLANKS * PARAMS_PER_PLANK

LENGTH_MIN = 40.0
LENGTH_MAX = 200.0


@dataclass(frozen=True)
class Plank:
    """A static rectangular obstacle the ball can roll on."""

    x: float
    y: float
    length: float
    angle: float  # radians


@dataclass(frozen=True)
class SceneBounds:
    """The grounded region planks are decoded into.

    Defaults span the whole world; callers that know the task geometry
    (see :func:`inventor.world.scene_bounds`) pass a tight band around the
    platform height and the chasm span.
    """

    x_min: float = 0.0
    x_max: float = 800.0
    y_min: float = 0.0
    y_max: float = 600.0
    length_min: float = LENGTH_MIN
    length_max: float = LENGTH_MAX


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _logit(p: np.ndarray | float) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-4, 1.0 - 1e-4)
    return np.log(p / (1.0 - p))


def decode(z: np.ndarray, bounds: SceneBounds | None = None) -> List[Plank]:
    """Map a latent vector to a list of placed planks.

    The ``use_gate`` slot acts as a soft on/off switch. Planks with a gate
    value below 0.5 are dropped, so the same latent dimensionality can
    represent inventions that use anywhere from 0 to ``MAX_PLANKS`` planks.
    """
    b = bounds or SceneBounds()
    z = np.asarray(z, dtype=np.float64).reshape(MAX_PLANKS, PARAMS_PER_PLANK)
    gates = _sigmoid(z[:, 0])
    xs = b.x_min + _sigmoid(z[:, 1]) * (b.x_max - b.x_min)
    ys = b.y_min + _sigmoid(z[:, 2]) * (b.y_max - b.y_min)
    lengths = b.length_min + _sigmoid(z[:, 3]) * (b.length_max - b.length_min)
    angles = np.tanh(z[:, 4]) * (np.pi / 2.0)

    planks: List[Plank] = []
    for i in range(MAX_PLANKS):
        if gates[i] >= 0.5:
            planks.append(
                Plank(
                    x=float(xs[i]),
                    y=float(ys[i]),
                    length=float(lengths[i]),
                    angle=float(angles[i]),
                )
            )
    return planks


def encode(planks: List[Plank], bounds: SceneBounds | None = None) -> np.ndarray:
    """Approximate inverse of :func:`decode`.

    Lifts a concrete scene back into the latent space so that the Inventor
    Loop can warm-start from a stored invention and so the token baseline's
    placements live in the same space for the originality metric. Slots
    beyond ``len(planks)`` are gated off.
    """
    b = bounds or SceneBounds()
    z = np.full((MAX_PLANKS, PARAMS_PER_PLANK), -6.0, dtype=np.float64)
    span_x = max(1e-6, b.x_max - b.x_min)
    span_y = max(1e-6, b.y_max - b.y_min)
    span_l = max(1e-6, b.length_max - b.length_min)
    for i, p in enumerate(planks[:MAX_PLANKS]):
        z[i, 0] = 6.0  # gate on
        z[i, 1] = _logit((p.x - b.x_min) / span_x)
        z[i, 2] = _logit((p.y - b.y_min) / span_y)
        z[i, 3] = _logit(
            (np.clip(p.length, b.length_min, b.length_max) - b.length_min) / span_l
        )
        z[i, 4] = float(np.arctanh(np.clip(p.angle / (np.pi / 2.0), -0.999, 0.999)))
    return z.ravel()


def random_latent(rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
    """Sample an initial latent vector."""
    return rng.normal(0.0, scale, size=LATENT_DIM)


def planks_to_json(planks: List[Plank]) -> list[dict]:
    """Serialize planks for prompts and result logs."""
    return [
        {
            "x": round(p.x, 2),
            "y": round(p.y, 2),
            "length": round(p.length, 2),
            "angle_deg": round(np.degrees(p.angle), 2),
        }
        for p in planks
    ]


def planks_from_json(data: list[dict]) -> List[Plank]:
    """Inverse of ``planks_to_json``; tolerant to missing keys."""
    out: List[Plank] = []
    for d in data[:MAX_PLANKS]:
        try:
            out.append(
                Plank(
                    x=float(d["x"]),
                    y=float(d["y"]),
                    length=float(d.get("length", 80.0)),
                    angle=float(np.radians(d.get("angle_deg", 0.0))),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return out
