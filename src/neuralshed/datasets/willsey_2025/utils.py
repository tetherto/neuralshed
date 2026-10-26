"""Readers for Willsey et al. 2025 finger-decoding and quadcopter ``.mat`` blocks.

The release ships one ``.mat`` per task block under ``Data/<YYYYMMDD>/RedisMat``
(seven closed-loop recording days, 2023-03-09 to 2023-04-13), two open-loop
cued-movement blocks under ``Data/OpenLoopData``, and one saved quadcopter flight
path under ``Data/20230712/RedisMat``. Blocks are stored as flat MATLAB v5 files
holding two redis streams recorded side by side:

* ``binnedNeural_*`` -- spike-band power in 50 ms bins, one row per bin;
* ``(binned_)taskOutput_stream_*`` -- the game state at roughly 1 kHz.

The power is **accumulated over the bin rather than averaged**, so it is extensive and
scales with the bin width. The 20 ms prefix described at :data:`N_STARTUP_BINS` is the
direct evidence, and the consequence is that rebinning has to sum, not average.

Both carry a ``redisClock`` in unix milliseconds, which is the only common time
base and the one the authors align on (``t5_parse.m`` matches each trial edge to
its nearest neural bin). ``binnedNeural_xpcClock`` is a block-relative ms counter
present in the closed-loop files only.

The paper is Willsey et al. 2025, *Nature Medicine* 31(1):96-104; the reference
code is https://github.com/WillseyBCILab/BCI_Finger_Decoding_Quadcopter.
"""

import datetime as dt
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import scipy.io as sio

from neuralshed.datasets.utils import dataset_dir

N_CHANNELS = 192
N_ARRAYS = 2
#: Every file stores 256 electrode columns; columns 192-255 are exactly zero in all
#: 60 neural blocks (the recording system's unpopulated front-end channels), so the
#: readers below trim them off.
N_CHANNELS_STORED = 256

NEURAL_BIN_SIZE_MS = 50
FS = 20.0

#: The first ~81 bins of every block (76 to 88, always contiguous from bin 0) are spaced
#: 20 ms rather than 50 ms. They are not transients: the power is **accumulated over the
#: bin**, so a 20 ms bin carries 40% of a 50 ms one. Measured across all 60 blocks the
#: raw prefix sits at 0.386 of the block mean, and rescaling it by 50/20 recovers the
#: following bins (per-channel correlation 0.97, level ratio 0.956). The authors drop 80
#: of them before z-scoring (``ManuscriptFigsFinal.ipynb``, "First 80 samples are
#: noisy"), which is the right call because the narrower bins bias the per-channel mean
#: and standard deviation, not because the samples are bad.
N_STARTUP_BINS = 80

#: The six columns of the finger position / target streams, per the source README.
#: The thumb carries two degrees of freedom, the other four fingers one each.
STORED_DIMS = ("thumb_flexion", "thumb_abduction", "index", "middle", "ring", "little")

#: The four independently controlled degrees of freedom and the stored columns that
#: carry them (the authors' ``FingSel = [0, 1, 2, 5]``). Index and middle move as one
#: group and so do ring and little, so columns 3 and 4 duplicate columns 2 and 5.
DECODED_DIMS = ("thumb_flexion", "thumb_abduction", "index_middle", "ring_little")
DECODED_COLUMNS = (0, 1, 2, 5)

#: Finger position is a fraction of the range of motion: 0 is full flexion, 1 is full
#: extension, 0.5 is the neutral posture the target stream rests at.
POSITION_FLEXED = 0.0
POSITION_NEUTRAL = 0.5
POSITION_EXTENDED = 1.0

#: Task parameters, stored per block as strings and identical across all 58
#: closed-loop blocks. The target spans ``+/- fn_hold_threshold`` around its centre,
#: so 0.1 is the 20%-of-range target width quoted in the paper.
HOLD_TIME_S = 0.5
TARGET_HALF_WIDTH = 0.1

#: Hand-colour codes in the closed-loop task stream. 1 is a ~10 ms flash at the moment
#: the held target is acquired and 2 is every other sample, so a trial runs from the
#: end of one flash to the start of the next (this is what ``t5_parse.m`` extracts).
COLOR_ACQUIRED = 1
COLOR_MOVING = 2

#: The nine cued conditions of the open-loop blocks: rest plus each decoded degree of
#: freedom driven to full flexion or full extension. Keyed by the target row.
OPEN_LOOP_CONDITIONS = {
    (0.5, 0.5, 0.5, 0.5): "rest",
    (0.0, 0.5, 0.5, 0.5): "thumb_flexion-flex",
    (1.0, 0.5, 0.5, 0.5): "thumb_flexion-extend",
    (0.5, 0.0, 0.5, 0.5): "thumb_abduction-flex",
    (0.5, 1.0, 0.5, 0.5): "thumb_abduction-extend",
    (0.5, 0.5, 0.0, 0.5): "index_middle-flex",
    (0.5, 0.5, 1.0, 0.5): "index_middle-extend",
    (0.5, 0.5, 0.5, 0.0): "ring_little-flex",
    (0.5, 0.5, 0.5, 1.0): "ring_little-extend",
}

#: The two closed-loop conditions. Targets strictly alternate between a randomly placed
#: posture and a return to the neutral one, so half the trials are the latter.
CLOSED_LOOP_CONDITIONS = ("return_to_neutral", "random_target")

#: Stable integer codes spanning both block kinds, so a code means the same thing across
#: the dataset. The open-loop conditions come first, in :data:`OPEN_LOOP_CONDITIONS` order.
CONDITION_CODES = tuple(OPEN_LOOP_CONDITIONS.values()) + CLOSED_LOOP_CONDITIONS


def condition_code(condition: str) -> int:
    """Index of a condition name in :data:`CONDITION_CODES`."""
    return CONDITION_CODES.index(condition)


KIND_CLOSED_LOOP = "closed_loop"
KIND_OPEN_LOOP = "open_loop"
KIND_QUADCOPTER = "quadcopter"

TASK_LABELS = {
    KIND_CLOSED_LOOP: "Finger target acquisition (closed loop)",
    KIND_OPEN_LOOP: "Cued single-degree-of-freedom movement (open loop)",
    KIND_QUADCOPTER: "Quadcopter flight path (no recording)",
}

#: The one block that holds a quadcopter flight path instead of a recording. It sits
#: under the same ``RedisMat`` folder as the finger blocks but shares none of their
#: fields, so it is named here rather than sniffed.
QUADCOPTER_BLOCK = "t5.2023.07.12_block14"

#: Per-block labels that exist only in the authors' analysis notebook, not in the
#: released files. ``decoder_dof`` is the number of degrees of freedom the online
#: decoder emitted, which is not recoverable from the data: a 4-DoF decoder driving a
#: 2-DoF task looks exactly like a 2-DoF decoder doing the same, because the game pins
#: the unused dimensions either way (use :func:`target_dof` for the task's own count).
#: ``new_targets`` is how many fingers were given a new target each trial.
#: Transcribed from ``ManuscriptFigsFinal.ipynb`` cells 18 and 24; blocks the notebook
#: does not label are absent.
BLOCK_LABELS: dict[str, dict] = {
    "t5.2023.04.06_block30": {"decoder_dof": 4},
    "t5.2023.04.06_block31": {"decoder_dof": 4},
    "t5.2023.04.06_block33": {"decoder_dof": 4},
    "t5.2023.04.06_block34": {"decoder_dof": 4},
    "t5.2023.04.06_block42": {"decoder_dof": 4},
    "t5.2023.04.06_block43": {"decoder_dof": 4},
    "t5.2023.04.06_block44": {"decoder_dof": 2},
    "t5.2023.04.06_block45": {"decoder_dof": 4},
    "t5.2023.04.06_block46": {"decoder_dof": 2},
    "t5.2023.04.13_block19": {"decoder_dof": 4},
    "t5.2023.04.13_block20": {"decoder_dof": 4},
    "t5.2023.04.13_block21": {"decoder_dof": 2},
    "t5.2023.04.13_block22": {"decoder_dof": 4},
    "t5.2023.04.13_block23": {"decoder_dof": 4},
    "t5.2023.04.13_block25": {"decoder_dof": 2},
    "t5.2023.04.13_block26": {"decoder_dof": 4},
    "t5.2023.04.13_block27": {"decoder_dof": 4},
    "t5.2023.04.13_block28": {"decoder_dof": 2},
    "t5.2023.04.13_block43": {"decoder_dof": 4, "new_targets": 2},
    "t5.2023.04.13_block44": {"decoder_dof": 4, "new_targets": 1},
    "t5.2023.04.13_block45": {"decoder_dof": 4, "new_targets": 2},
    "t5.2023.04.13_block46": {"decoder_dof": 4, "new_targets": 1},
    "t5.2023.04.13_block47": {"decoder_dof": 4, "new_targets": 1},
    "t5.2023.04.13_block48": {"decoder_dof": 4, "new_targets": 2},
    "t5.2023.04.13_block50": {"decoder_dof": 4, "new_targets": 1},
    "t5.2023.04.13_block51": {"decoder_dof": 4, "new_targets": 2},
}

#: The Study downloads into this class-named folder under ``data/``.
DATA_SUBDIR = "Willsey2025FingerDecoding"
DEFAULT_DATA_DIR = dataset_dir("willsey_2025") / "data" / DATA_SUBDIR / "Data"

_NAME_RE = re.compile(
    r"^t5_t5\.(?P<session>\d{4}\.\d{2}\.\d{2})_Data_RedisMat_\d{8}_"
    r"(?P<time>\d{6})_\((?P<block>\d+)\)$"
)


@dataclass(frozen=True)
class BlockRef:
    """One task block, i.e. one released ``.mat`` file."""

    session: str  # e.g. "t5.2023.03.09"
    block: int  # the source's own block number, from the filename suffix
    kind: str  # "closed_loop" | "open_loop" | "quadcopter"
    start_time: str  # wall-clock start "HHMMSS", from the filename
    path: Path

    @property
    def run(self) -> str:
        return f"block{self.block:02d}"

    @property
    def label(self) -> str:
        return f"{self.session}_{self.run}"

    @property
    def task_label(self) -> str:
        return TASK_LABELS[self.kind]


def session_date(session: str) -> dt.date:
    """The recording date behind a session label (``t5.2023.03.09`` -> 2023-03-09)."""
    return dt.date.fromisoformat(session.removeprefix("t5.").replace(".", "-"))


def list_blocks(data_dir: str | Path = DEFAULT_DATA_DIR) -> list[BlockRef]:
    """Every released block, in recording order."""
    root = Path(data_dir)
    refs = []
    for path in sorted(root.glob("**/*.mat")):
        if path.name.startswith("._") or path.parent.name == "Decoders":
            continue
        match = _NAME_RE.match(path.stem)
        if match is None:
            raise ValueError(f"unrecognized block filename: {path.name}")
        session, block = f"t5.{match['session']}", int(match["block"])
        if f"{session}_block{block:02d}" == QUADCOPTER_BLOCK:
            kind = KIND_QUADCOPTER
        elif path.parent.name == "OpenLoopData":
            kind = KIND_OPEN_LOOP
        else:
            kind = KIND_CLOSED_LOOP
        refs.append(
            BlockRef(
                session=session,
                block=block,
                kind=kind,
                start_time=match["time"],
                path=path,
            )
        )
    return sorted(refs, key=lambda ref: (ref.session, ref.start_time))


def list_sessions(data_dir: str | Path = DEFAULT_DATA_DIR) -> list[str]:
    """Recording days, oldest first."""
    return sorted({ref.session for ref in list_blocks(data_dir)})


def session_blocks(
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> dict[str, list[BlockRef]]:
    out: dict[str, list[BlockRef]] = {}
    for ref in list_blocks(data_dir):
        out.setdefault(ref.session, []).append(ref)
    return out


@lru_cache(maxsize=8)
def _load(path: str) -> dict:
    """Raw named arrays of one block file, with the stream names normalized.

    The closed-loop files prefix the task stream with ``binned_`` and the open-loop
    files do not, and only the closed-loop files carry the decoder's own output
    (``estimated``); everything else is shared.
    """
    raw = sio.loadmat(path)
    if "x" in raw:  # the quadcopter block holds only a flight path
        return {
            "flight_path": np.column_stack(
                [raw["x"].ravel(), raw["y"].ravel(), raw["z"].ravel()]
            ).astype(np.float64)
        }

    prefix = "binned_" if "binned_taskOutput_stream_target" in raw else ""
    power = np.asarray(raw["binnedNeural_hlfp"], dtype=np.float32)
    if power.shape[1] != N_CHANNELS_STORED:
        raise ValueError(f"{path}: expected {N_CHANNELS_STORED} columns")
    if power[:, N_CHANNELS:].any():
        raise ValueError(f"{path}: electrode columns {N_CHANNELS}-255 are not empty")

    # Closed-loop blocks store the colour as an RGBA-shaped ``(n, 4)`` array whose first
    # column carries the code and whose others are always zero; the open-loop blocks
    # store the code alone as ``(1, n)``.
    color = np.asarray(raw[f"{prefix}taskOutput_stream_hand_color"])
    color = color[:, 0] if color.ndim == 2 and color.shape[1] == 4 else color.ravel()

    data = {
        "spike_band_power": power[:, :N_CHANNELS],
        "neural_clock": np.asarray(raw["binnedNeural_redisClock"])
        .ravel()
        .astype(np.int64),
        "task_clock": np.asarray(raw[f"{prefix}taskOutput_stream_redis_clock"])
        .ravel()
        .astype(np.int64),
        "target": np.asarray(
            raw[f"{prefix}taskOutput_stream_target"], dtype=np.float64
        ),
        "hand_color": color.astype(np.int64),
    }
    if f"{prefix}taskOutput_stream_estimated" in raw:
        data["position"] = np.asarray(
            raw[f"{prefix}taskOutput_stream_estimated"], dtype=np.float64
        )
    if "binnedNeural_xpcClock" in raw:
        data["neural_xpc_clock"] = (
            np.asarray(raw["binnedNeural_xpcClock"]).ravel().astype(np.int64)
        )
    for name in ("holdTime", "fn_hold_threshold"):
        if name in raw:
            data[name] = float(np.asarray(raw[name]).ravel()[0])
    return data


def load_block(ref: BlockRef) -> dict:
    """Named arrays for one block."""
    return _load(str(ref.path))


def get_neural(ref: BlockRef) -> np.ndarray:
    """Spike-band power as ``(n_bins, 192)``, in the order the source stores it."""
    return load_block(ref)["spike_band_power"]


def bin_times(ref: BlockRef) -> np.ndarray:
    """Neural bin timestamps in seconds from the block's first bin.

    Derived from the redis clock rather than assumed regular: the leading
    :data:`N_STARTUP_BINS` are spaced 20 ms and the rest 50 ms with a few ms of
    network jitter.
    """
    clock = load_block(ref)["neural_clock"]
    return (clock - clock[0]) / 1000.0


def steady_state_start(ref: BlockRef) -> int:
    """Index of the first bin whose accumulation window is a full 50 ms.

    Everything before it belongs to the 20 ms prefix (see :data:`N_STARTUP_BINS`). A
    bin's timestamp marks the end of its window, so the boundary is one past the first
    gap above 35 ms: measured across all 60 blocks that bin sits at 0.97 of the steady
    level while its predecessor sits at 0.39, i.e. exactly 20/50.

    Anything that treats the stream as a uniform 20 Hz signal has to start here, since
    the prefix packs 76 to 88 bins into 1.5 to 1.8 s rather than 4 s. No trial starts
    before bin 192 in any block, so no trial is lost.
    """
    clock = load_block(ref)["neural_clock"]
    gaps = np.flatnonzero(np.diff(clock) > 35)
    if gaps.size == 0:
        raise ValueError(f"{ref.label}: no 50 ms bin found")
    return int(gaps[0]) + 1


def get_task_stream(ref: BlockRef, key: str) -> np.ndarray:
    """One raw ~1 kHz task-stream field (``target``, ``position``, ``hand_color``)."""
    data = load_block(ref)
    if key not in data:
        raise KeyError(f"{ref.label}: no such task-stream field {key!r}")
    return data[key]


def on_bins(ref: BlockRef, key: str) -> np.ndarray:
    """A task-stream field resampled onto the neural bin times, as ``(n_bins, n_dims)``.

    Linear interpolation over the shared redis clock, matching ``LoadData`` in the
    authors' notebook. Neural bins outside the task stream's span (the recording
    starts a few seconds early) are clipped to the nearest task sample.
    """
    data = load_block(ref)
    values = np.asarray(data[key], dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    clock = np.clip(data["neural_clock"], data["task_clock"][0], data["task_clock"][-1])
    return np.column_stack(
        [np.interp(clock, data["task_clock"], column) for column in values.T]
    )


def get_finger_position(ref: BlockRef) -> np.ndarray:
    """Decoded finger position per neural bin, as ``(n_bins, 4)`` over DECODED_DIMS."""
    return on_bins(ref, "position")[:, list(DECODED_COLUMNS)]


def get_finger_target(ref: BlockRef) -> np.ndarray:
    """Cued finger target per neural bin, as ``(n_bins, 4)`` over DECODED_DIMS."""
    return on_bins(ref, "target")[:, list(DECODED_COLUMNS)]


def _bin_span(
    ref: BlockRef, begin: np.ndarray, end: np.ndarray
) -> list[tuple[int, int]]:
    """Half-open neural-bin ranges covering each ``[begin, end]`` unix-ms interval.

    A bin's timestamp marks the **end** of its 50 ms accumulation window, so the first
    bin belonging to an interval is the first one timestamped after the interval opens,
    and the last is the last one timestamped at or before it closes.

    This is a deliberate one-bin divergence from ``t5_parse.m``, which matches each edge
    to its *nearest* bin. Nearest-matching puts the first bin of a trial on the window
    that mostly covered the previous target, because the acquisition flash is shorter
    than one bin: measured on the released blocks, the cued target changes at
    ``trial start + 1`` in 95 of 101 trials. It also makes consecutive trials share a
    bin, which an extractor rejects as overlapping coverage. The authors only used
    nearest-matching to average 2 s open-loop windows, where one bin is immaterial.
    """
    clock = load_block(ref)["neural_clock"]
    first = np.searchsorted(clock, begin, side="right")
    last = np.searchsorted(clock, end, side="right")
    return [(int(a), int(b)) for a, b in zip(first, last) if b > a]


def trial_windows(ref: BlockRef) -> list[tuple[int, int]]:
    """Half-open ``(start, stop)`` neural-bin ranges of a closed-loop block's trials.

    A trial runs from the end of the target-acquired flash that closed the previous
    trial to the start of the flash that closes this one, i.e. the whole period the
    participant was moving toward one target. See :func:`_bin_span` for how the edges
    land on bins.
    """
    if ref.kind != KIND_CLOSED_LOOP:
        raise ValueError(f"{ref.label}: trials are defined for closed-loop blocks only")
    data = load_block(ref)
    color, clock = data["hand_color"], data["task_clock"]
    starts, stops = _trial_edges(color)
    return _bin_span(ref, clock[starts + 1], clock[stops])


def _trial_edges(color: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Task-stream indices of the ``1 -> 2`` and ``2 -> 1`` colour edges, paired.

    Each acquisition flash is preceded by exactly one move onset, so pairing is a
    matter of dropping any flash that opens the block before a move has started.
    """
    starts = np.flatnonzero(
        (color[:-1] == COLOR_ACQUIRED) & (color[1:] == COLOR_MOVING)
    )
    stops = np.flatnonzero((color[:-1] == COLOR_MOVING) & (color[1:] == COLOR_ACQUIRED))
    stops = stops[stops > starts[0]] if starts.size else stops[:0]
    picked = np.searchsorted(starts, stops) - 1
    if picked.size and not np.array_equal(picked, np.arange(picked.size)):
        raise ValueError("hand colour does not alternate one flash per trial")
    return starts[picked], stops


def trial_targets(ref: BlockRef) -> np.ndarray:
    """The target each trial acquired, as ``(n_trials, 4)`` over DECODED_DIMS."""
    data = load_block(ref)
    _, stops = _trial_edges(data["hand_color"])
    return data["target"][np.ix_(stops, list(DECODED_COLUMNS))]


def target_dof(ref: BlockRef) -> tuple[str, ...]:
    """The degrees of freedom this block's task actually cued.

    Four of the closed-loop days interleave blocks whose targets move all four
    degrees of freedom with blocks that pin the thumb's abduction and the ring-little
    group at the neutral posture, leaving a two-dimensional task.
    """
    target = load_block(ref)["target"][:, list(DECODED_COLUMNS)]
    varies = [np.unique(target[:, i]).size > 1 for i in range(len(DECODED_DIMS))]
    return tuple(dim for dim, moves in zip(DECODED_DIMS, varies) if moves)


def cue_windows(ref: BlockRef) -> list[tuple[int, int, str]]:
    """``(start, stop, condition)`` neural-bin ranges of an open-loop block's go periods.

    The open-loop blocks alternate a 1 s preparation phase (``hand_color == 1``) with a
    2 s go phase (``hand_color == 2``) while the cued posture changes every 3 s, so the
    go periods are the 2 s windows the authors average over. ``condition`` is the
    matching :data:`OPEN_LOOP_CONDITIONS` name.
    """
    if ref.kind != KIND_OPEN_LOOP:
        raise ValueError(f"{ref.label}: cues are defined for open-loop blocks only")
    data = load_block(ref)
    color, clock, target = data["hand_color"], data["task_clock"], data["target"]
    edges = np.flatnonzero(np.diff(color) != 0) + 1
    bounds = np.concatenate(([0], edges, [color.size]))
    runs = [
        (start, stop)
        for start, stop in zip(bounds[:-1], bounds[1:])
        if color[start] == COLOR_MOVING
    ]
    spans = _bin_span(
        ref,
        clock[[start for start, _ in runs]],
        clock[[stop - 1 for _, stop in runs]],
    )
    conditions = [
        OPEN_LOOP_CONDITIONS[tuple(target[stop - 1, list(DECODED_COLUMNS)])]
        for _, stop in runs
    ]
    return [(a, b, condition) for (a, b), condition in zip(spans, conditions)]


def quadcopter_path(data_dir: str | Path = DEFAULT_DATA_DIR) -> np.ndarray:
    """The saved quadcopter flight path as ``(n_samples, 3)`` in ``x, y, z``.

    The file carries no clock, so the sample rate is not recoverable; the authors plot
    it against sample index. ``z`` points down, so elevation is ``-z`` (their
    ``PlotFlightPath.m`` plots ``x, y, -z`` against rings centred at ``z = +14.91``).
    """
    refs = [ref for ref in list_blocks(data_dir) if ref.kind == KIND_QUADCOPTER]
    if len(refs) != 1:
        raise ValueError(f"expected exactly one quadcopter block, found {len(refs)}")
    return load_block(refs[0])["flight_path"]
