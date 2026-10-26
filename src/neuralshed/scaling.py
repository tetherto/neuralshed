"""Rescaling formulas and the loader JSON that carries them to read time.

A rescaled neural row is the same payload with a different *loader*: its
``filepath`` becomes a ``SpecialLoader`` JSON naming ``_load_rescaled`` on the
Study, so the fit and the division happen at read time, over the whole stored
array (extractors, segmenters and collators never see scaling).
"""

import json
import typing as tp

import numpy as np

GUARDS = ("eps", "zero_to_one")
EPS = 1e-8
METHOD = "_load_rescaled"

#: A bin range ``[start, end)`` in the payload's own bin grid.
StatRow = tp.Sequence[int]


def pooled_columns(n_bins: int, stat_rows: tp.Iterable[StatRow]) -> np.ndarray:
    """Column indices pooled by a fit: the union of the given bin ranges.

    Ranges are clamped to the array and overlapping ones merge, so a bin under
    two sibling windows enters the statistics exactly once. An empty
    ``stat_rows`` fits the whole array; ranges that all clamp away return
    empty, which is an error for the caller, not a silent fit-everything.
    """
    if not stat_rows:
        return np.arange(n_bins)
    merged: list[list[int]] = []
    for start, end in sorted(
        (max(int(a), 0), min(int(b), n_bins)) for a, b in stat_rows
    ):
        if start >= end:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    if not merged:
        return np.arange(0)
    return np.concatenate([np.arange(a, b) for a, b in merged])


def fit_rescale(
    x: np.ndarray, stat_rows: tp.Iterable[StatRow]
) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel ``(mean, std)`` over the pooled stat columns, ddof=0."""
    columns = pooled_columns(x.shape[-1], stat_rows)
    if columns.size == 0:
        raise ValueError(
            f"Stat rows {list(stat_rows)} select no columns of a {x.shape[-1]}-bin "
            "payload"
        )
    pooled = np.asfortranarray(x[..., columns].T)  # (stat_bins, channels), column-major
    mu = pooled.mean(axis=0, keepdims=True).T
    sd = pooled.std(axis=0, keepdims=True).T
    return mu, sd


def apply_rescale(
    x: np.ndarray,
    mu: np.ndarray,
    sd: np.ndarray | None = None,
    guard: str = "eps",
    divisor: tp.Sequence[float] | None = None,
) -> np.ndarray:
    """Subtract ``mu`` and divide, elementwise over the whole array. ``guard`` picks the zero-std treatment: ``"eps"`` divides by
    ``sd + 1e-8`` (Willet, Kunz), ``"zero_to_one"`` replaces a zero ``sd``
    with ``1.0`` (Wairagkar).
    """
    if guard not in GUARDS:
        raise ValueError(f"Unknown guard {guard!r}; have {list(GUARDS)}")
    if divisor is not None:
        div = np.asarray(divisor, dtype=x.dtype).reshape((-1,) + (1,) * (x.ndim - 1))
        if div.shape[0] != x.shape[0]:
            raise ValueError(
                f"Divisor has {div.shape[0]} channels, payload has {x.shape[0]}"
            )
    elif sd is None:
        raise ValueError(
            "Apply_rescale needs a fitted sd or an explicit divisor; with no "
            "std to guard there is no formula left to apply"
        )
    elif guard == "eps":
        div = sd + EPS
    else:
        div = np.where(sd == 0, 1.0, sd)
    return (x - mu) / div


def rescaled_loader_json(
    original_json: str,
    *,
    stat_rows: tp.Iterable[StatRow],
    guard: str = "eps",
    divisor: tp.Sequence[float] | None = None,
    mean: tp.Sequence[float] | None = None,
) -> str:
    """The loader JSON for a rescaled read of the row ``original_json`` loads.
    Wraps the raw loader untouched as ``inner`` so ``from_json``
    rebuilds the Study as it originally does.
    """
    if guard not in GUARDS:
        raise ValueError(f"Unknown guard {guard!r}; have {list(GUARDS)}")
    if mean is not None and divisor is None:
        raise ValueError(
            "Rescaled_loader_json: a baked mean requires a baked divisor; "
            "pairing a pre-fitted mean with a fitted std fits no audit formula"
        )
    head = json.loads(original_json)
    kwargs: dict[str, tp.Any] = {
        "inner": original_json,
        "stat_rows": [[int(a), int(b)] for a, b in stat_rows],
        "guard": guard,
    }
    if divisor is not None:
        kwargs["divisor"] = [float(v) for v in divisor]
    if mean is not None:
        kwargs["mean"] = [float(v) for v in mean]
    payload = {
        "cls": head["cls"],
        "timeline": head["timeline"],
        "method": METHOD,
        "kwargs": kwargs,
    }
    if "cls_kwargs" in head:
        payload["cls_kwargs"] = head["cls_kwargs"]
    return json.dumps(payload, sort_keys=True)


__all__ = [
    "EPS",
    "GUARDS",
    "METHOD",
    "apply_rescale",
    "fit_rescale",
    "pooled_columns",
    "rescaled_loader_json",
]
