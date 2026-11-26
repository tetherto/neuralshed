import typing as tp

import numpy as np

from neuralset.base import StrCast
from neuralset.events import etypes
from pydantic import PositiveFloat

# ===========================================================================
# Language
# ===========================================================================


class Character(etypes.BaseText):
    """Single written/typed character — a-z, punctuation, or space (`text` is the glyph).

    Sibling of `Word`/`Phoneme`/`Keystroke` for handwriting / character-level typing tasks,
    where the decode target is one glyph rather than a whole word. Selection is by the
    `"Character"` type name.

    `char_class`
        the source's raw label for the glyph (e.g. `"greaterThan"` for space), for provenance.
    """

    char_class: str = ""


# ===========================================================================
# Shared base
# ===========================================================================


class DenseArrayEvent(etypes.BaseDataEvent):
    """Lazily-read dense `(n_features, n_samples)` array at `frequency` Hz — shared base for
    the neural and behavioral events below. Never select this base in an extractor; it
    would sweep both branches.

    `subject`
        recording-subject id (required).
    `feature`
        name of the stream, for channel names and provenance.
    `brain_region`
        optional event-level anatomical label.
    `n_features`
        required source-declared size of axis 0. For electrophysiology this is the emitted
        stream's channel count; for behavioral arrays it is the number of dimensions. This
        describes the event before extractor composition or projection.
    `unit`
        pre-normalization physical quantity, registered in `units.py`. A z-scored count
        stream declares `"counts/bin"`; `"a.u."` means the quantity is unknown.
    `normalization`
        which statistic the source removed, one of `units.NORMALIZATION_ORDER`. Nothing else.
    `processing`
        everything else the source's chain did, in order, `""` when it did nothing else.
        Free-form, but keep the corpus shape of semicolon-separated clauses, e.g.
        `"log-transformed before normalization; rolling 10 s statistics"`. Read by people
        and by pooling code deciding whether two streams at the same rank really match.
    `filepath`
        a `.npy`/`.npz` path, or a SpecialLoader JSON for in-memory arrays.
    """

    #: Fields an extractor's cached read depends on, so they belong in its cache key.
    CACHE_FIELDS: tp.ClassVar[tuple[str, ...]] = (
        "feature",
        "unit",
        "normalization",
        "frequency",
    )

    subject: StrCast = ""
    feature: str = ""
    brain_region: str = ""
    n_features: int
    unit: str = ""
    normalization: str = "raw"
    processing: str = ""

    def model_post_init(self, log__: tp.Any) -> None:
        self.subject = self.subject  # trigger StrCast coercion
        if self._missing_duration_or_frequency():
            raise ValueError(
                f"{type(self).__name__} needs non-zero frequency and duration; "
                f"got frequency={self.frequency}, duration={self.duration}."
            )
        if not self.subject:
            raise ValueError("Missing 'subject' field")
        if (
            not isinstance(self.n_features, int)
            or isinstance(self.n_features, bool)
            or self.n_features <= 0
        ):
            raise ValueError(
                f"{type(self).__name__}.n_features must be a positive integer; "
                f"got {self.n_features!r}"
            )
        super().model_post_init(log__)

    def read(self) -> np.ndarray:
        """Return the `(n_rows, n_samples)` array."""
        data = np.asarray(super().read())
        if data.ndim != 2:
            raise ValueError(
                f"{type(self).__name__} expects a 2-D (n_rows, n_samples) array; "
                f"got shape {data.shape!r}."
            )
        if data.shape[0] != self.n_features:
            raise ValueError(
                f"{type(self).__name__} declares n_features={self.n_features}, "
                f"but read shape {data.shape!r}."
            )
        return data

    def _read(self) -> np.ndarray:
        fp = str(self.filepath)
        if fp.endswith(".npy"):
            return np.load(fp)
        if fp.endswith(".npz"):
            with np.load(fp) as zf:
                return zf[self.feature if self.feature in zf.files else zf.files[0]]
        raise ValueError(
            f"{type(self).__name__}._read expects a .npy/.npz file or a SpecialLoader, "
            f"got {fp!r}."
        )


# ===========================================================================
# Electrophysiology
# ===========================================================================


class SpikeCounts(DenseArrayEvent):
    """Binned spike counts, `(n_channels, n_bins)`.

    `method`
        how counts were obtained — `"threshold_crossing"` (unsorted MUA proxy) or `"sorted"`.
    `threshold`
        detection threshold as the source states it, `""` when unpublished. Free-form, but
        keep the corpus shape `"<signed multiplier> x <RMS|SD>"`, e.g. `"-4.5 x RMS"`.
        A source giving a range keeps it: `"-3.5/-4.5 x RMS"`.
    """

    method: tp.Literal["threshold_crossing", "sorted"] = "threshold_crossing"
    threshold: str = ""


class BandPower(DenseArrayEvent):
    """Binned power in a frequency band, `(n_channels, n_bins)`.

    `lower_freq`, `upper_freq`
        band edges in Hz (optional; validated `0 <= lower < upper` when set). Spike-band
        power = `BandPower(lower_freq=250, upper_freq=5000)`. A band reaching down to DC
        keeps `lower_freq=0`, so only both edges at zero counts as unset.
    """

    lower_freq: float = 0.0
    upper_freq: float = 0.0

    def model_post_init(self, log__: tp.Any) -> None:
        lo, hi = self.lower_freq, self.upper_freq
        if (lo or hi) and not 0 <= lo < hi:
            raise ValueError(
                f"BandPower needs 0 <= lower_freq < upper_freq; got {lo}, {hi}."
            )
        super().model_post_init(log__)


# ===========================================================================
# Behavior
# ===========================================================================


class BehavioralSeries(DenseArrayEvent):
    """Continuous behavioral series (dims-first), `(n_dims, n_samples)` at `frequency` Hz.
    Prefer a concrete subclass so the pipeline can select by type.

    `dims`
        names for the axes; length must match the array's first dimension.
    `kind`
        free-form metadata for queries/provenance — not a selection key.
    `reference_frame`
        coordinate frame the values are expressed in.
    """

    dims: tuple[str, ...] = ()
    kind: str = ""
    reference_frame: str = ""

    def model_post_init(self, log__: tp.Any) -> None:
        if not self.dims:
            raise ValueError("`dims` must be non-empty")
        if self.n_features != len(self.dims):
            raise ValueError(
                f"{type(self).__name__}.n_features={self.n_features} does not match "
                f"len(dims)={len(self.dims)}"
            )
        super().model_post_init(log__)

    def read(self) -> np.ndarray:
        """Return the `(n_dims, n_samples)` array."""
        data = super().read()
        if data.shape[0] != len(self.dims):
            raise ValueError(
                f"{type(self).__name__} expects (len(dims)={len(self.dims)}, n_samples); "
                f"got {data.shape!r}."
            )
        return data


class PositionSeries(BehavioralSeries):
    """Spatial position (e.g. hand/cursor endpoint); `dims` like `(x, y, z)`."""


class OrientationSeries(BehavioralSeries):
    """Orientation / rotational DoFs (angles); `dims` like `(x,)` or roll/pitch/yaw."""


class GraspSeries(BehavioralSeries):
    """Grasp / hand-shape DoFs (often normalized 0..1); `dims` like `(g1, g2, g3)`."""


class TargetGraspSeries(BehavioralSeries):
    """Cued target hand shape over time, selectable independently from the observed one.

    The grasp counterpart of `TargetPositionSeries`: same dims and units as the
    `GraspSeries` it is the target for, so a decoder can read cue and outcome as two
    streams of the same shape. It is a **sibling** of `GraspSeries`, not a subclass,
    because `GraspSeries` is itself a concrete emitted type: subclassing it would make
    an extractor selecting `"GraspSeries"` sweep the target stream too and collide.
    """


class LinearVelocitySeries(BehavioralSeries):
    """Translational velocity (d/dt of position); `dims` like `(x, y, z)`."""


class AngularVelocitySeries(BehavioralSeries):
    """Angular velocity (d/dt of orientation, or a gyroscope); `dims` like `(x,)` or
    roll/pitch/yaw, or `(gyro_x, gyro_y, gyro_z)` for a 3-axis gyro."""


class GraspVelocitySeries(BehavioralSeries):
    """Grasp velocity (d/dt of grasp DoFs); `dims` like `(g1, g2, g3)`."""


class AccelerationSeries(BehavioralSeries):
    """Linear acceleration (e.g. a 3-axis accelerometer); `dims` like `(accel_x, accel_y, accel_z)`."""


class CursorPositionSeries(PositionSeries):
    """Observed 2-D cursor position, selectable independently from its target."""


class TargetPositionSeries(PositionSeries):
    """Cued 2-D target position over time."""


class CursorVelocitySeries(LinearVelocitySeries):
    """Velocity of the cursor, as the cursor actually moved.

    The plain name is the physical quantity, and every member can offer it: a member
    publishing positions only gets it from
    :class:`~neuralshed.transforms.DeriveStream`.
    """


class DecodedCursorVelocitySeries(LinearVelocitySeries):
    """Velocity an online decoder commanded for the cursor, as the source published it.

    A sibling of `CursorVelocitySeries` rather than the same type, because the two are not
    the same quantity wherever both exist: in `karpowicz_2025` the cursor follows the
    command inside movements but not while the task holds it at the centre or recentres
    it. Keeping the qualifier on this one means a stream that came out of a
    decoder can never be mistaken for a measurement.
    """


class AlignedCursorVelocitySeries(BehavioralSeries):
    """Cursor velocity along the direction to its cued target, one channel.

    The part of the cursor's motion the task's own intention accounts for. Its sibling
    `ResidualCursorVelocitySeries` is the part it cannot.
    """


class ResidualCursorVelocitySeries(BehavioralSeries):
    """Cursor velocity across the direction to its cued target, one channel, signed.

    Orthogonal to the intention by construction, so nothing about where the target sits
    predicts it. Whatever drove the cursor does: in a closed-loop block, the online decoder.
    """


class CursorIntentionSeries(BehavioralSeries):
    """Unit vector from the cursor to its cued target, per bin; `dims` like `(x, y)`.

    Dimensionless by construction, so it pools across members whose cursors move in pixels,
    screen heights and arbitrary units. Bins where the cursor sits on the target have no
    direction and are left at zero. Task-defined rather than decoder-emitted, which is what
    makes it a decode target no online decoder shaped.
    """


class CursorAssistSeries(BehavioralSeries):
    """Computer contribution to cursor velocity, usually in the range 0..1."""


class ClickAssistSeries(BehavioralSeries):
    """Binary state indicating whether target selection was computer-assisted."""


class ClickStateSeries(BehavioralSeries):
    """Binary click output over time, typically emitted by an online decoder."""


class AcousticFeatures(BehavioralSeries):
    """Acoustic feature vectors over time (e.g. LPCNet: 18 Bark-cepstral + 2 pitch/voicing),
    `(n_features, n_samples)`, often time-aligned to a neural timebase — a brain-to-voice
    decode target. `dims` names the features (e.g. `bfcc_0..17, pitch_period, pitch_corr`).
    """


# ===========================================================================
# Masks
# ===========================================================================


class BinaryMask(DenseArrayEvent):
    """Per-sample True/False flags, `(n_rows, n_samples)` at `frequency` Hz, e.g. the bins
    a benchmark scores or the samples an artifact covers. Declare `unit="binary"`.

    `true_means`
        what True marks, in the source's terms (`"scored"`, `"valid"`, `"artifact"`).
        Required, since a mask read with the wrong polarity selects the wrong samples.
    `dims`
        optional row names when there are several masks (e.g. one per channel); its
        length must match `n_features` when set.
    """

    true_means: str
    dims: tuple[str, ...] = ()

    def model_post_init(self, log__: tp.Any) -> None:
        if not self.true_means:
            raise ValueError(f"{type(self).__name__}.true_means must be non-empty")
        if self.dims and len(self.dims) != self.n_features:
            raise ValueError(
                f"{type(self).__name__}.n_features={self.n_features} does not match "
                f"len(dims)={len(self.dims)}"
            )
        super().model_post_init(log__)

    def read(self) -> np.ndarray:
        """Return the `(n_rows, n_samples)` bool array."""
        data = super().read()
        if not np.isin(data, (0, 1)).all():
            raise ValueError(f"{type(self).__name__} holds values other than 0 and 1")
        return data.astype(bool)


# ===========================================================================
# Inter-trial marker
# ===========================================================================


class ControlMode(etypes.CategoricalEvent):
    duration: PositiveFloat
    loop_type: tp.Literal["open", "assisted", "closed"]
    decoder: str


class Intertrial(etypes.CategoricalEvent):
    """Inter-trial interval — the rest/gap between structured task trials.

    A pure time-span marker (`start`/`duration` only), mirroring the built-in `Background`.
    Unlike `Action`/`Stimulus` it has no `code` — it is dead time, not a class to decode.

    `description`
        optional human-readable note, for provenance.
    """

    description: str = ""


class AutoMovement(etypes.CategoricalEvent):
    """A movement the task software makes on its own, with no cue and nothing for the
    participant to attempt, e.g. snapping a virtual hand onto an object or resetting a
    cursor to center.

    A time-span marker like `Intertrial`, with no `code`: there is no intention to decode.

    `description`
        the source's name for the movement, e.g. ``"SnapTo"``.
    """

    description: str = ""


__all__ = [
    "Character",
    "DenseArrayEvent",
    "SpikeCounts",
    "BandPower",
    "BehavioralSeries",
    "PositionSeries",
    "OrientationSeries",
    "GraspSeries",
    "TargetGraspSeries",
    "LinearVelocitySeries",
    "AngularVelocitySeries",
    "GraspVelocitySeries",
    "AccelerationSeries",
    "CursorPositionSeries",
    "TargetPositionSeries",
    "CursorVelocitySeries",
    "DecodedCursorVelocitySeries",
    "CursorIntentionSeries",
    "AlignedCursorVelocitySeries",
    "ResidualCursorVelocitySeries",
    "CursorAssistSeries",
    "ClickAssistSeries",
    "ClickStateSeries",
    "AcousticFeatures",
    "ControlMode",
    "BinaryMask",
    "Intertrial",
    "AutoMovement",
]
