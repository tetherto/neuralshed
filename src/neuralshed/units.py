"""Registered `unit` and `normalization` values for `DenseArrayEvent`, and how they interact.

`NeuralFeatureExtractor` reads both (with `event.frequency`) to harmonize binned neural
features. An unregistered value raises rather than passing through.

`unit` decides how to convert, in three classes: `count` (per-bin, extensive),
`rate` (per-second, intensive) and `non_convertible`. It holds the pre-normalization
quantity where known, so a z-scored count stream declares `counts/bin`, not `a.u.`.

`normalization` decides whether datasets can be pooled. It is a one-way chain,
`raw -> mean_subtracted -> z_scored`, so a pool's target must be at least as processed as
every member. It names only the statistic that was removed. Anything else the source did
(a non-linear step, the scope of the statistics, smoothing) goes in `event.processing`.

Conversion is valid iff `normalization` is `raw` or `mean_subtracted`; above that the
values are dimensionless and a frequency factor is a no-op, not an error.

See `docs/standardization.md` and `docs/tutorials/02_neural_harmonization.ipynb`.
"""

import typing as tp

Kind = tp.Literal["count", "rate", "non_convertible"]
TargetUnit = tp.Literal["native", "Hz", "counts/bin"]
Normalization = tp.Literal["raw", "mean_subtracted", "z_scored"]

COUNT = "counts/bin"
RATE = "Hz"

ALIASES: dict[str, str] = {
    "count": COUNT,
    "au": "a.u.",
    "arbitrary": "a.u.",
}

_CONVERTIBLE: dict[str, Kind] = {COUNT: "count", RATE: "rate"}

NON_CONVERTIBLE = frozenset(
    {
        # neural, neither a count nor a rate
        "uV^2",
        "a.u.",  # the source quantity is genuinely unknown
        "a.u./s",
        # behavioral, carried by the same DenseArrayEvent base
        "px",
        "px/s",
        "screen_height",
        "screen_height/s",
        "proportion",
        "binary",
        "LPCNet",
    }
)

_TARGET_KIND: dict[str, Kind] = {RATE: "rate", COUNT: "count"}

# least to most processed; a target is reachable iff it sits at or above the source
NORMALIZATION_ORDER: tuple[Normalization, ...] = (
    "raw",
    "mean_subtracted",
    "z_scored",
)
NORMALIZATIONS = frozenset(NORMALIZATION_ORDER)

_RANK: dict[str, int] = {name: i for i, name in enumerate(NORMALIZATION_ORDER)}

_UNIT_PRESERVING = frozenset({"raw", "mean_subtracted"})


def base_unit(unit: str) -> str:
    """Canonical unit, alias-resolved. `"count"` -> `"counts/bin"`, `"au"` -> `"a.u."`."""
    return ALIASES.get(unit.strip(), unit.strip())


def is_registered(unit: str) -> bool:
    """Whether `unit` resolves to a known string (no raise). For the CI audit."""
    base = base_unit(unit)
    return base in _CONVERTIBLE or base in NON_CONVERTIBLE


def classify(unit: str) -> Kind:
    """`"count"`, `"rate"` or `"non_convertible"`. Raises on an absent or unknown unit."""
    base = base_unit(unit)
    if base in _CONVERTIBLE:
        return _CONVERTIBLE[base]
    if base in NON_CONVERTIBLE:
        return "non_convertible"
    raise ValueError(
        f"unregistered unit {unit!r}. Declare the unit on the event, and register the "
        f"string in neuralshed.units (known: {sorted(_CONVERTIBLE) + sorted(NON_CONVERTIBLE)})."
    )


def check_normalization(normalization: str) -> Normalization:
    """The value itself, or raise. Mirrors `classify`'s contract for the other field."""
    if normalization not in NORMALIZATIONS:
        raise ValueError(
            f"unregistered normalization {normalization!r} "
            f"(known: {list(NORMALIZATION_ORDER)})."
        )
    return tp.cast(Normalization, normalization)


def preserves_unit(normalization: str) -> bool:
    """Whether a unit conversion is still valid under this regime (the interaction rule)."""
    return check_normalization(normalization) in _UNIT_PRESERVING


def is_reachable(source: str, target: str) -> bool:
    """Whether `target` can be produced from `source` by processing forward only.

    Ranks only, so equal ranks compare reachable. Two streams sharing a rank can still be
    incomparable in practice when their `processing` chains differ, which this cannot see.
    """
    src, dst = check_normalization(source), check_normalization(target)
    return _RANK[dst] >= _RANK[src]


def scale_factor(
    unit: str,
    target: TargetUnit | str,
    frequency: float,
    target_frequency: float | None = None,
) -> float:
    """Multiplier taking values from one unit and bin rate to another.

    Every conversion routes through the rate: multiply in by the source rate if the source
    is a count, divide out by the target rate if the target is a count.

    Parameters
    ----------
    unit : str
        The source unit, as the event declares it. Alias-resolved via `base_unit`.
    target : TargetUnit | str
        The unit to reach. `"native"` is a strict no-op that never inspects `unit`. Any
        other value must resolve to `"Hz"` or `"counts/bin"`.
    frequency : float
        Bin rate the source values are binned at, in Hz.
    target_frequency : float | None
        Bin rate the output is binned at, in Hz. Defaults to `frequency`, i.e. no change
        of bin width.

    Returns
    -------
    float
        The multiplier. `1.0` whenever nothing should change: `target="native"`, a
        non-convertible unit, or a conversion whose two ends already agree.
    """
    if target == "native":
        return 1.0
    base = base_unit(target)
    if base not in _TARGET_KIND:
        raise ValueError(
            f"target must be one of {['native', *_TARGET_KIND]}, got {target!r}"
        )
    dst = frequency if target_frequency is None else target_frequency
    kind, target_kind = classify(unit), _TARGET_KIND[base]
    if kind == "non_convertible":
        return 1.0
    needed = ([frequency] if kind == "count" else []) + (
        [dst] if target_kind == "count" else []
    )
    if not all(rate > 0 for rate in needed):
        raise ValueError(
            f"converting {unit!r} at {frequency!r} to {target!r} at {dst!r} needs a "
            f"positive bin rate at each count end."
        )
    if kind == "rate":
        return 1.0 if target_kind == "rate" else 1.0 / dst
    if target_kind == "rate":
        return frequency
    return 1.0 if frequency == dst else frequency / dst


__all__ = [
    "COUNT",
    "RATE",
    "ALIASES",
    "NON_CONVERTIBLE",
    "NORMALIZATIONS",
    "NORMALIZATION_ORDER",
    "Kind",
    "Normalization",
    "TargetUnit",
    "base_unit",
    "check_normalization",
    "classify",
    "is_reachable",
    "is_registered",
    "preserves_unit",
    "scale_factor",
]
