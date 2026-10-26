"""Readers for Gusman et al. 2025 multi-gesture drag-and-drop data.

The release is three MATLAB **v7.3** files, so they are HDF5 and ``h5py`` reads them while
``scipy.io.loadmat`` does not. Two tasks:

* ``sesData_GH_T5.mat`` / ``sesData_GH_T11.mat`` -- Gesture Hero, the open-loop task where
  a cued gesture is attempted and held for 1, 2 or 4 s.
* ``sesData_DD_T11.mat`` -- Multi-Gesture Drag-and-Drop, the closed-loop 2D centre-out task.

Each file holds a MATLAB struct array with one entry per session. Fields are ``feat``
(``1344 x n_bins``), ``featInfo``, ``taskInfo``, ``blocks``, ``participant``,
``sessionNum``, and for drag-and-drop an ``outCL`` carrying the online decoder's output.

Unlike the trialized deposits, blocks are concatenated onto **one continuous 20 ms grid**
per session and the task tables index into it with bin ranges, so this behaves like a
recording rather than a bag of clips.

**The 1,344 columns are attributable.** Seven feature types for each of 192 electrodes,
with ``featInfo.featOffset`` giving the boundaries and ``featInfo.featNames`` naming them,
so unlike ``vargas_2025`` no identity is lost.

**MATLAB ``categorical`` fields decode from the file.** ``labels``, ``cuedGesture``,
``cuedDirection``, ``trialAttemptType`` and ``decodedState`` all appear as ``(1, 6)``
uint32 MCOS references. :func:`read_categorical` resolves them, needing only ``h5py``.

The paper is Gusman et al. 2025, *Journal of Neural Engineering* 22:026054; the reference
code is https://github.com/jgusman/drag-and-drop.
"""

import functools
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from neuralshed.datasets.utils import dataset_dir

N_ARRAYS = 2
N_CHANNELS = 192
N_FEATURE_TYPES = 7
N_FEATURES = N_CHANNELS * N_FEATURE_TYPES  # 1344

BIN_SIZE_MS = 20
FS = 50.0

#: The online feature chain, from the paper: non-causal threshold crossings, spike-band
#: power, then five short-time Fourier bands. Names are read from the file rather than
#: assumed, and this tuple is the expected order.
FEATURE_NAMES = (
    "ncTX",
    "spikePower",
    "STFT 0-8",
    "STFT 12-16",
    "STFT 20-31",
    "STFT 39-125",
    "STFT 129-250",
)

#: Spike-band power passband in Hz, from the paper's Methods.
SPIKE_BAND_HZ = (250.0, 5000.0)

#: The five local field potential bands in Hz. The paper's Methods define them as 0-11,
#: 12-19, 20-38, 39-128 and 129-250, while the file's own ``featNames`` label them 0-8,
#: 12-16, 20-31, 39-125 and 129-250. The labels sit on the 1000/256 = 3.90625 Hz grid of
#: the 256-sample FFT, so they read as the realized bin edges and the paper's numbers as
#: the band definitions. Source name -> (definition, source label).
LFP_BANDS = {
    "STFT 0-8": ((0.0, 11.0), (0.0, 8.0)),
    "STFT 12-16": ((12.0, 19.0), (12.0, 16.0)),
    "STFT 20-31": ((20.0, 38.0), (20.0, 31.0)),
    "STFT 39-125": ((39.0, 128.0), (39.0, 125.0)),
    "STFT 129-250": ((129.0, 250.0), (129.0, 250.0)),
}

#: Band edges in Hz as the paper's Methods define them.
LFP_BAND_EDGES = {name: defined for name, (defined, _) in LFP_BANDS.items()}

#: Band edges in Hz as the source file's own ``featNames`` label them.
LFP_SOURCE_EDGES = {name: labelled for name, (_, labelled) in LFP_BANDS.items()}

#: Source band name -> the ``feature`` a released event carries, e.g. ``"lfp_12_19"``.
#: Named from the definition, so the name and ``lower_freq``/``upper_freq`` agree.
LFP_FEATURE_NAMES = {
    name: "lfp_{:g}_{:g}".format(*edges) for name, edges in LFP_BAND_EDGES.items()
}

#: Every feature is z-scored against a rolling 3 minute window before release.
NORMALIZATION_WINDOW_S = 180

TASK_GESTURE_HERO = "gesture_hero"
TASK_DRAG_DROP = "drag_and_drop"

#: ``(task, participant) -> filename``. The drag-and-drop file holds T11 only.
FILES = {
    (TASK_GESTURE_HERO, "T5"): "sesData_GH_T5.mat",
    (TASK_GESTURE_HERO, "T11"): "sesData_GH_T11.mat",
    (TASK_DRAG_DROP, "T11"): "sesData_DD_T11.mat",
}

DEFAULT_DATA_DIR = dataset_dir("gusman_2025") / "data"


@dataclass(frozen=True)
class SessionRef:
    """One recording session inside one of the three files."""

    participant: str  # "T5" | "T11"
    task: str  # "gesture_hero" | "drag_and_drop"
    session: int  # the source's own sessionNum
    index: int  # position in the file's struct array
    path: Path

    @property
    def label(self) -> str:
        return f"{self.participant}_{self.task}_s{self.session}"


@functools.lru_cache(maxsize=4)
def _open(path: str) -> h5py.File:
    """Open a release file read-only, cached.

    ``feat`` is up to 2.3 GB per session, so nothing is read eagerly. Callers slice the
    dataset instead, which is why the handle stays open for the process lifetime.
    """
    return h5py.File(path, "r")


def _root(handle: h5py.File):
    """The single top-level struct of a release file."""
    name = next(k for k in handle.keys() if not k.startswith("#"))
    return handle[name]


def _field(handle: h5py.File, root, name: str, index: int):
    """One field of struct-array entry ``index``, dereferenced when it needs to be.

    A file holding one session stores fields directly; a file holding several stores an
    object reference per session, so both shapes have to be handled.
    """
    obj = root[name]
    if isinstance(obj, h5py.Dataset) and obj.dtype == h5py.ref_dtype:
        return handle[np.asarray(obj).ravel()[index]]
    return obj


def _text(handle: h5py.File, obj) -> str:
    """A MATLAB char array as a string."""
    return "".join(chr(c) for c in np.asarray(obj).ravel())


def read_categorical(handle: h5py.File, reference) -> tuple[list[str], np.ndarray]:
    """Decode a MATLAB ``categorical`` into its category names and 1-based codes.

    A ``categorical`` is stored as a ``(1, 6)`` uint32 MCOS reference whose fifth element
    is a 1-based object index. The payload lives in ``#subsystem#/MCOS``, laid out as a
    header followed by one pair per object: the category names at ``2 * index`` and the
    per-element codes at ``2 * index + 1``. Code 0 means undefined.

    Verified against all eight categoricals in the release, including ``decodedState``,
    which is one code per 20 ms bin rather than one per trial.
    """
    mcos = np.asarray(handle["#subsystem#/MCOS"]).ravel()
    index = int(np.asarray(reference).ravel()[4])
    names = [
        _text(handle, handle[ref])
        for ref in np.asarray(handle[mcos[2 * index]]).ravel()
    ]
    codes = np.asarray(handle[mcos[2 * index + 1]]).ravel().astype(int)
    return names, codes


def _decode_labels(handle: h5py.File, reference) -> np.ndarray:
    """A ``categorical`` as an array of strings, with ``""`` where the code is undefined."""
    names, codes = read_categorical(handle, reference)
    return np.array([names[c - 1] if c else "" for c in codes], dtype=object)


def list_sessions(data_dir: str | Path = DEFAULT_DATA_DIR) -> list[SessionRef]:
    """Every session across the three files, ordered by task then participant."""
    root_dir = Path(data_dir)
    refs = []
    for (task, participant), filename in FILES.items():
        path = root_dir / filename
        handle = _open(str(path))
        root = _root(handle)
        count = np.asarray(root["sessionNum"]).size
        for index in range(count):
            number = int(
                np.asarray(_field(handle, root, "sessionNum", index)).ravel()[0]
            )
            refs.append(SessionRef(participant, task, number, index, path))
    return sorted(refs, key=lambda r: (r.task, r.participant, r.session))


def _handle(ref: SessionRef):
    handle = _open(str(ref.path))
    return handle, _root(handle)


def feature_layout(ref: SessionRef) -> pd.DataFrame:
    """One row per feature type: its name and its half-open column range.

    This is what ``vargas_2025`` cannot provide. Every one of the 1,344 columns maps to a
    feature type and, within it, to an electrode of one of the two arrays.
    """
    handle, root = _handle(ref)
    info = _field(handle, root, "featInfo", ref.index)
    names = [_text(handle, handle[r]) for r in np.asarray(info["featNames"]).ravel()]
    offsets = np.asarray(info["featOffset"]).ravel().astype(int)
    stops = list(offsets[1:]) + [N_FEATURES]
    return pd.DataFrame({"feature": names, "start": offsets, "stop": stops})


def array_names(ref: SessionRef) -> list[str]:
    """The two array names the source uses, ``Lateral`` and ``Medial``."""
    handle, root = _handle(ref)
    info = _field(handle, root, "featInfo", ref.index)
    return [_text(handle, handle[r]) for r in np.asarray(info["arrayNames"]).ravel()]


def n_bins(ref: SessionRef) -> int:
    """Length of this session's continuous 20 ms grid."""
    handle, root = _handle(ref)
    return int(_field(handle, root, "feat", ref.index).shape[1])


def duration(ref: SessionRef) -> float:
    """Session length in seconds."""
    return n_bins(ref) / FS


def blocks(ref: SessionRef) -> np.ndarray:
    """The source's own block numbers for this session, not contiguous."""
    handle, root = _handle(ref)
    return np.asarray(_field(handle, root, "blocks", ref.index)).ravel().astype(int)


def get_features(
    ref: SessionRef, feature: str | None = None, bins: slice | None = None
) -> np.ndarray:
    """Neural features as ``(n_rows, n_bins)``, read lazily from the file.

    ``feature`` selects one of :data:`FEATURE_NAMES` and narrows the result to that
    type's 192 electrodes. ``bins`` slices the time axis. Reading a whole session's 1,344
    rows is up to 2.3 GB, so at least one of the two is usually wanted.
    """
    handle, root = _handle(ref)
    data = _field(handle, root, "feat", ref.index)
    rows = slice(0, N_FEATURES)
    if feature is not None:
        layout = feature_layout(ref)
        match = layout[layout["feature"] == feature]
        if match.empty:
            raise ValueError(
                f"unknown feature {feature!r}, expected one of {FEATURE_NAMES}"
            )
        rows = slice(int(match["start"].iloc[0]), int(match["stop"].iloc[0]))
    return np.asarray(data[rows, bins or slice(None)])


def epochs(ref: SessionRef) -> pd.DataFrame:
    """The session's task table, one row per trial or per stage.

    Gesture Hero gives one row per attempted gesture, with ``label`` naming it and
    ``duration_cued`` the 1, 2 or 4 s hold. Drag-and-drop gives one row per **stage** of a
    trial, so ``trial`` groups them and ``stage`` names the phase.

    ``start`` and ``stop`` are half-open bin indices into the continuous grid.
    """
    handle, root = _handle(ref)
    info = _field(handle, root, "taskInfo", ref.index)
    spans = np.asarray(info["startStops"]).astype(int)
    frame = pd.DataFrame(
        {
            "start": spans[0],
            "stop": spans[1] + 1,  # source ends are inclusive
            "block": np.asarray(info["blockNumber"]).ravel().astype(int),
        }
    )
    if ref.task == TASK_GESTURE_HERO:
        frame["label"] = _decode_labels(handle, info["labels"])
        # The source records the cued hold in bins (50, 100, 200); the paper quotes it
        # in seconds (1, 2, 4), which is what is exposed here.
        frame["hold_s"] = (
            np.asarray(info["trialDurationRounded"]).ravel().astype(int) / FS
        )
    else:
        frame["trial"] = np.asarray(info["trialNum"]).ravel().astype(int)
        frame["stage"] = [
            _text(handle, handle[r]) for r in np.asarray(info["trialStage"]).ravel()
        ]
        frame["label"] = _decode_labels(handle, info["cuedGesture"])
        frame["direction"] = _decode_labels(handle, info["cuedDirection"])
        frame["attempt_type"] = _decode_labels(handle, info["trialAttemptType"])
        for column, source in (
            ("is_drag", "isDragAttempt"),
            ("is_click", "isClickAttempt"),
            ("is_wait", "isWait"),
            ("kin_decoder_closed", "isKinDecoderCL"),
            ("gesture_decoder_closed", "isGestureDecoderCL"),
        ):
            frame[column] = np.asarray(info[source]).ravel().astype(bool)
        frame["kin_error_attenuation"] = np.asarray(info["kinErrorAttenuation"]).ravel()
    return frame


def decoded_state(ref: SessionRef) -> np.ndarray:
    """The online decoder's per-bin gesture output, drag-and-drop sessions only.

    One label per 20 ms bin over the same four states the task cues, so it is a stream
    rather than a per-trial value.
    """
    if ref.task != TASK_DRAG_DROP:
        raise ValueError(f"{ref.label}: an online decode exists for drag-and-drop only")
    handle, root = _handle(ref)
    return _decode_labels(
        handle, _field(handle, root, "outCL", ref.index)["decodedState"]
    )


def gestures(ref: SessionRef) -> list[str]:
    """The cued gestures of this session, excluding the no-action state."""
    labels = epochs(ref)["label"]
    return sorted({name for name in labels if name and name != "no_action"})
