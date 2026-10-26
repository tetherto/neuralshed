"""Normalization statistics, fitted on training trials only and carried in the checkpoint.

Why this is a separate artifact rather than a wrapper extractor. neuralset's fitted extractors
do not persist what they fit. ``ExtractorPCA`` keeps its PCA in a local variable inside
``prepare()`` and lets it be garbage-collected; its ``_uid`` is a private attribute excluded
from ``model_dump()``, and its cache is keyed by a hash of the exact event set it prepared on,
so a different split refits rather than reuses. The only serializable persistence anywhere in
the extractor stack is ``LabelEncoder.predefined_mapping``, a real pydantic field.

So the split is: *policy* is configuration (:class:`NormalizationPolicy`, which serializes with
the collection), *values* are a training artifact (:class:`BitPreprocessingState`, registered as
buffers on the read-in so they travel with the weights). Applying a stale or absent state is
then impossible to do quietly.
"""

import dataclasses
import typing as tp

import numpy as np
import torch

if tp.TYPE_CHECKING:
    from neuralshed.manifest import CollectionManifest

EPS = 1e-6


@dataclasses.dataclass(frozen=True)
class NormalizationPolicy:
    """How to normalize, as configuration. Contains no fitted values.

    ``per_channel`` z-scores each channel independently, which is what differing electrode
    gains call for. ``respect_already_normalized`` honours the recipe's list of members whose
    source already z-scored the data, as card_2024 does within session.
    """

    method: tp.Literal["zscore", "none"] = "zscore"
    per_channel: bool = True
    respect_already_normalized: bool = True


@dataclasses.dataclass
class ChannelStats:
    """Fitted mean and standard deviation for one layout, ``(C, 1)``."""

    mean: np.ndarray
    std: np.ndarray
    n_frames: int

    def apply(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)


class BitPreprocessingState:
    """Fitted statistics keyed by ``layout_key``.

    One key rather than a pair, because a layout *is* a stream: the layout key already names
    the event class it reads, so there is nothing left to disambiguate.

    Fitted from **training trials only**, so validation and test never leak into the scale the
    model sees. :meth:`fit` filters on the split assigned by the collection's split transform, which is
    why splits must exist before this runs.
    """

    def __init__(
        self,
        stats: dict[str, ChannelStats],
        policy: NormalizationPolicy,
        fitted_on: tuple[str, ...] = ("train",),
        target_stats: dict[str, ChannelStats] | None = None,
    ):
        self.stats = stats
        self.policy = policy
        self.fitted_on = fitted_on
        #: Statistics for a dense target, keyed by its own layout key. Empty for a text
        #: target, which has no scale to fit.
        self.target_stats = dict(target_stats or {})

    def __repr__(self) -> str:
        return (
            f"BitPreprocessingState({len(self.stats)} entries, "
            f"{len(self.target_stats)} target entries, "
            f"method={self.policy.method!r}, fitted_on={self.fitted_on})"
        )

    def __contains__(self, layout_key: str) -> bool:
        return layout_key in self.stats

    def get(self, layout_key: str) -> ChannelStats | None:
        return self.stats.get(layout_key)

    def fingerprint(self) -> str:
        """A stable digest of every fitted value.

        The leakage test compares this before and after removing validation trials from the
        input: an identical digest is direct evidence the fit saw only training data.
        """
        import hashlib

        h = hashlib.sha256()
        h.update(repr(dataclasses.asdict(self.policy)).encode())
        for source in (self.stats, self.target_stats):
            for key in sorted(source):
                s = source[key]
                h.update(key.encode())
                h.update(np.ascontiguousarray(s.mean, dtype=np.float64).tobytes())
                h.update(np.ascontiguousarray(s.std, dtype=np.float64).tobytes())
        return h.hexdigest()

    # -- fitting -------------------------------------------------------------
    @classmethod
    def fit(
        cls,
        trials: tp.Sequence,
        manifest: "CollectionManifest",
        policy: NormalizationPolicy = NormalizationPolicy(),
        *,
        splits: tuple[str, ...] = ("train",),
        already_normalized: tp.Iterable[str] = (),
    ) -> "BitPreprocessingState":
        """Accumulate per-channel statistics over ``splits``, one entry per input layout.

        ``already_normalized`` names collection members whose source z-scored the data
        (card_2024 does, within session). It is passed in rather than read off the manifest
        because it is a *declared* fact, part of the recipe, and the manifest holds only what
        was resolved from the data.

        Sums are accumulated in float64 across trials rather than concatenating them, so a
        long session costs no extra memory and the result does not depend on trial order.
        """
        stats: dict[str, ChannelStats] = {}
        if policy.method == "none":
            return cls(stats, policy, fitted_on=tuple(splits))
        skip_members = (
            set(already_normalized) if policy.respect_already_normalized else set()
        )

        for layout in manifest.array_layouts:
            if layout.member in skip_members:
                continue
            stream = layout.event_type
            arrays = [
                t.arrays[stream]
                for t in trials
                if t.split in splits
                and t.layouts.get(stream) == layout.layout_key
                and stream in t.arrays
            ]
            if not arrays:
                raise ValueError(
                    f"layout {layout.layout_key!r} has no trials in splits {splits}; it "
                    "cannot be normalized, and training it would use an unfitted scale"
                )
            acc = _accumulate(arrays, per_channel=policy.per_channel)
            if acc is not None:
                stats[layout.layout_key] = acc

        target_stats: dict[str, ChannelStats] = {}
        for layout in (
            s for s in manifest.layouts if s.is_array and s.role in ("target", "both")
        ):
            arrays = [
                t.target
                for t in trials
                if t.split in splits
                and t.group_key in layout.groups
                and t.target is not None
            ]
            if not arrays:
                continue
            # One scale for every channel, never per channel. A kinematic target's channels
            # are the axes of one vector, so scaling x and y by different numbers skews the
            # space and a direction error stops costing the same in every direction.
            acc = _accumulate(arrays, per_channel=False)
            if acc is not None:
                target_stats[layout.layout_key] = acc
        return cls(stats, policy, fitted_on=tuple(splits), target_stats=target_stats)

    # -- application ---------------------------------------------------------
    def transform(self, layout_key: str, x: np.ndarray) -> np.ndarray:
        """Normalize one ``(C, T)`` array. A missing entry is a pass-through, which is the
        correct behaviour for an already-normalized member and a no-op policy alike."""
        entry = self.stats.get(layout_key)
        return x if entry is None else entry.apply(x)

    def transform_target(self, layout_key: str, x: np.ndarray) -> np.ndarray:
        """Normalize one ``(C, T)`` target array. A missing entry is a pass-through."""
        entry = self.target_stats.get(layout_key)
        return x if entry is None else entry.apply(x)

    def target_buffers(self) -> dict[str, torch.Tensor]:
        """The target statistics, for the head to undo when reporting in source units."""
        out: dict[str, torch.Tensor] = {}
        for layout_key, s in self.target_stats.items():
            stem = _sanitize(layout_key)
            out[f"{stem}__mean"] = torch.from_numpy(s.mean.astype(np.float32))
            out[f"{stem}__std"] = torch.from_numpy(s.std.astype(np.float32))
        return out

    def as_buffers(self) -> dict[str, torch.Tensor]:
        """Flat ``name -> tensor`` map for registering on the read-in module.

        Keys are sanitized because ``nn.Module`` buffer names cannot contain ``.``; a layout
        key holds ``/`` separators by construction, never a dot, but sanitizing keeps an
        unusual key from failing at registration time.
        """
        out: dict[str, torch.Tensor] = {}
        for layout_key, s in self.stats.items():
            stem = _sanitize(layout_key)
            out[f"{stem}__mean"] = torch.from_numpy(s.mean.astype(np.float32))
            out[f"{stem}__std"] = torch.from_numpy(s.std.astype(np.float32))
        return out


def _sanitize(key: str) -> str:
    return key.replace(".", "_").replace("/", "__")


def _accumulate(
    arrays: tp.Iterable[np.ndarray], per_channel: bool
) -> ChannelStats | None:
    total = sq_total = None
    n = 0
    for a in arrays:
        a64 = np.asarray(a, dtype=np.float64)
        s = a64.sum(axis=1, keepdims=True)
        sq = (a64**2).sum(axis=1, keepdims=True)
        total = s if total is None else total + s
        sq_total = sq if sq_total is None else sq_total + sq
        n += a64.shape[1]
    if total is None or n == 0:
        return None
    n_channels = total.shape[0]
    if not per_channel:
        # one scalar statistic broadcast back over the channels
        total = np.full_like(total, total.sum() / n_channels)
        sq_total = np.full_like(sq_total, sq_total.sum() / n_channels)
    mean = total / n
    var = np.maximum(sq_total / n - mean**2, 0.0)
    return ChannelStats(
        mean=mean.astype(np.float32),
        std=(np.sqrt(var) + EPS).astype(np.float32),
        n_frames=n,
    )


__all__ = [
    "NormalizationPolicy",
    "ChannelStats",
    "BitPreprocessingState",
    "EPS",
]
