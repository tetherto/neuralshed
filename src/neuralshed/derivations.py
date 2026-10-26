"""Streams derived from other streams, and the registry :class:`DeriveStream` names.

A *derivation* produces a new quantity out of what a source published: the velocity of a
position, the direction from a cursor to its target. It belongs to the dataset and to the
question being asked, not to any one model, and it is never undone. That is what separates it
from normalization (``models/bit/preprocess.py``), which leaves the quantity alone, rescales
it for one network, is fitted on the training split, and is reversed before scoring.

A derivation runs over a whole timeline rather than a trial window, so a difference has a real
predecessor at every bin except the recording's first.
"""

import dataclasses
import typing as tp

import numpy as np

#: Below this displacement the cursor is on the target and there is no direction to point in.
_INTENTION_EPS = 1e-8


def _velocity(position: np.ndarray, *, rate: float) -> np.ndarray:
    """Per-bin velocity of a ``(C, T)`` position, in its own units per second."""
    return np.diff(position, axis=1, prepend=position[:, :1]) * rate


def _intention(cursor: np.ndarray, target: np.ndarray, *, rate: float) -> np.ndarray:
    """Unit vector from ``cursor`` to ``target``, per bin, from two ``(C, T)`` arrays.

    Both are published in the same unit and frame by every member, so their difference is a
    displacement and dividing it by its own norm leaves a dimensionless direction.
    """
    delta = target - cursor
    norm = np.linalg.norm(delta, axis=0, keepdims=True)
    return np.divide(delta, norm, out=np.zeros_like(delta), where=norm > _INTENTION_EPS)


def _unit_and_velocity(
    cursor: np.ndarray, target: np.ndarray, rate: float
) -> tuple[np.ndarray, np.ndarray]:
    """The direction to the target and the cursor's own velocity, per bin."""
    if cursor.shape[0] != 2:
        raise ValueError(
            f"splitting a velocity needs a 2-D cursor, got {cursor.shape[0]} channels"
        )
    return _intention(cursor, target, rate=rate), _velocity(cursor, rate=rate)


def _aligned(cursor: np.ndarray, target: np.ndarray, *, rate: float) -> np.ndarray:
    """How fast the cursor closed on its target, ``(1, T)``: the velocity along the
    direction to it. This is the part of the cursor's motion the intention explains."""
    unit, velocity = _unit_and_velocity(cursor, target, rate)
    return (velocity * unit).sum(axis=0, keepdims=True)


def _residual(cursor: np.ndarray, target: np.ndarray, *, rate: float) -> np.ndarray:
    """What is left of the cursor's velocity across the direction to the target, ``(1, T)``,
    signed by which side it fell on.

    The intention cannot explain this component by construction, so a model predicting it
    from neural activity has learned something only whatever drove the cursor knows. In a
    closed-loop block that is the online decoder.
    """
    unit, velocity = _unit_and_velocity(cursor, target, rate)
    across = velocity - (velocity * unit).sum(axis=0, keepdims=True) * unit
    side = np.sign(unit[0] * across[1] - unit[1] * across[0])
    return (side * np.linalg.norm(across, axis=0))[None, :]


@dataclasses.dataclass(frozen=True)
class Derivation:
    """One named operation: what it consumes, what it emits, and what unit comes out."""

    fn: tp.Callable[..., np.ndarray]
    n_sources: int
    emits: str
    unit: tp.Callable[[str], str]
    summary: str

    #: Names for the emitted channels, when the operation does not keep its sources' own.
    dims: tuple[str, ...] | None = None

    def apply(self, arrays: tp.Sequence[np.ndarray], *, rate: float) -> np.ndarray:
        if len(arrays) != self.n_sources:
            raise ValueError(
                f"{self.summary!r} takes {self.n_sources} source stream(s), "
                f"got {len(arrays)}"
            )
        widths = {a.shape[0] for a in arrays}
        if len(widths) > 1:
            raise ValueError(f"source streams disagree on width: {sorted(widths)}")
        lengths = {a.shape[1] for a in arrays}
        if len(lengths) > 1:
            raise ValueError(f"source streams disagree on length: {sorted(lengths)}")
        return np.ascontiguousarray(
            self.fn(*arrays, rate=float(rate)), dtype=np.float32
        )


#: Operation name -> what it does. The sources are given in the order the operation reads
#: them, which is also the order a ``DeriveStream`` step must list its event types in.
DERIVATIONS: dict[str, Derivation] = {
    "velocity": Derivation(
        fn=_velocity,
        n_sources=1,
        emits="CursorVelocitySeries",
        unit=lambda unit: f"{unit}/s" if unit else "",
        summary="velocity of a position stream",
    ),
    "intention": Derivation(
        fn=_intention,
        n_sources=2,
        emits="CursorIntentionSeries",
        unit=lambda unit: "a.u.",
        summary="unit vector from a cursor to its target",
    ),
    "aligned_velocity": Derivation(
        fn=_aligned,
        n_sources=2,
        emits="AlignedCursorVelocitySeries",
        dims=("along",),
        unit=lambda unit: f"{unit}/s" if unit else "",
        summary="cursor velocity along the direction to its target",
    ),
    "residual_velocity": Derivation(
        fn=_residual,
        n_sources=2,
        emits="ResidualCursorVelocitySeries",
        dims=("across",),
        unit=lambda unit: f"{unit}/s" if unit else "",
        summary="cursor velocity across the direction to its target",
    ),
}

__all__ = ["DERIVATIONS", "Derivation"]
