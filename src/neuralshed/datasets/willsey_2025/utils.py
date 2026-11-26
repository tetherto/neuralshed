"""Readers for Willsey et al. 2025 finger-decoding and quadcopter ``.mat`` blocks.

The release ships one ``.mat`` per task block under ``Data/<YYYYMMDD>/RedisMat``
(seven closed-loop recording days, 2023-03-09 to 2023-04-13), two open-loop
cued-movement blocks under ``Data/OpenLoopData``, and one saved quadcopter flight
path under ``Data/20230712/RedisMat``. Blocks are stored as flat MATLAB v5 files
holding two redis streams recorded side by side:

* ``binnedNeural_*`` -- spike-band power in 50 ms bins, one row per bin;
* ``(binned_)taskOutput_stream_*`` -- the game state at roughly 1 kHz.

The Supplementary Methods describe the power as a sum of squared samples over the bin,
but the 50 ms bins sit on the scale of datasets that average (see the README), so the
sum was divided by a fixed count and the stream behaves as a mean square: rebinning
averages. The 20 ms prefix described at :data:`N_STARTUP_BINS` shows the count is fixed.

Both carry a ``redisClock`` in unix milliseconds, which is the only common time
base and the one the authors align on (``t5_parse.m`` matches each open-loop trial
edge to its nearest neural bin). ``binnedNeural_xpcClock`` is the real-time machine's ms
counter, not zeroed at block start, present in the closed-loop files only.

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
#: Every file stores 256 electrode columns; columns 192-255 are exactly zero in all
#: 60 neural blocks (the recording system's unpopulated front-end channels), so the
#: readers below trim them off.
N_CHANNELS_STORED = 256

#: On this day columns 0-95 and 96-191 are exchanged relative to every other session, as
#: on wilson_2025's swapped-cable day. No source states it. The data show it in all 20
#: blocks (``docs/channel_mapping/t5.md``), and :func:`_load` swaps them back.
SWAPPED_ARRAY_SESSIONS = frozenset({"t5.2023.04.13"})

FS = 20.0

#: The first ~81 bins of every block (76 to 88, always contiguous from bin 0) are spaced
#: 20 ms rather than 50 ms. They are not transients: every bin is divided by the same
#: count, sized for 50 ms (see the module docstring), so a 20 ms bin carries 40% of a
#: 50 ms one. Measured across all 60 blocks the raw prefix sits at 0.386 of the block
#: mean, and rescaling it by 50/20 recovers the following bins (per-channel correlation
#: 0.97, level ratio 0.956). The authors drop 80
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

#: The three fingers a closed-loop target moves, as indices into DECODED_DIMS. The
#: thumb's two degrees of freedom count as one finger, as in the authors' one- and
#: two-finger blocks.
FINGERS = {"thumb": (0, 1), "index_middle": (2,), "ring_little": (3,)}

#: Finger position is a fraction of the range of motion: 1 is full flexion (full
#: abduction for ``thumb_abduction``), 0 is full extension (adduction), 0.5 is the
#: neutral posture the target stream rests at. The paper's Methods give flexion and
#: abduction the positive sign and the authors' notebook takes it from the target
#: change, but ``RunAnalysis.m`` and the Extended Data Fig. 1 it draws name 0 flexion.
POSITION_NEUTRAL = 0.5

#: Task parameters, stored per block as strings and identical across all 58
#: closed-loop blocks. The target spans ``+/- fn_hold_threshold`` around its centre,
#: so 0.1 is the 20%-of-range target width quoted in the paper.
HOLD_TIME_S = 0.5
TARGET_HALF_WIDTH = 0.1

#: Hand-colour codes in the closed-loop task stream. 1 is a ~10 ms flash that closes a
#: trial, when the held target is acquired or at the 10 s timeout (:func:`trial_success`
#: tells the two apart), and 2 is every other sample, so a trial runs from the end of
#: one flash to the start of the next. The authors' notebook delimits the same trials by
#: target changes (``TranLog`` in ``CalculateTrialStats``).
COLOR_ACQUIRED = 1
COLOR_MOVING = 2

#: The nine cued conditions of the open-loop blocks: rest plus each decoded degree of
#: freedom driven to either end of its range. Keyed by the target row.
OPEN_LOOP_CONDITIONS = {
    (0.5, 0.5, 0.5, 0.5): "rest",
    (0.0, 0.5, 0.5, 0.5): "thumb_flexion-extend",
    (1.0, 0.5, 0.5, 0.5): "thumb_flexion-flex",
    (0.5, 0.0, 0.5, 0.5): "thumb_abduction-adduct",
    (0.5, 1.0, 0.5, 0.5): "thumb_abduction-abduct",
    (0.5, 0.5, 0.0, 0.5): "index_middle-extend",
    (0.5, 0.5, 1.0, 0.5): "index_middle-flex",
    (0.5, 0.5, 0.5, 0.0): "ring_little-extend",
    (0.5, 0.5, 0.5, 1.0): "ring_little-flex",
}

#: The two closed-loop conditions. Targets strictly alternate between a randomly placed
#: posture and a return to the neutral one, so half the trials are the latter.
CLOSED_LOOP_CONDITIONS = ("return_to_neutral", "random_target")

#: The open-loop go phases that bring a degree of freedom back to neutral, named after
#: the outward movement they undo. Keyed by the posture the phase starts from. The
#: authors' ``RunAnalysis.m`` names a phase by its end posture, which counts these as
#: rest, but only a phase that starts and ends at neutral is rest.
OPEN_LOOP_RETURNS = {
    posture: f"{name}_return"
    for posture, name in OPEN_LOOP_CONDITIONS.items()
    if name != "rest"
}

#: Stable integer codes spanning both block kinds, so a code means the same thing across
#: the dataset. The open-loop conditions come first, in :data:`OPEN_LOOP_CONDITIONS` order,
#: then the closed-loop ones, then the open-loop returns.
CONDITION_CODES = (
    tuple(OPEN_LOOP_CONDITIONS.values())
    + CLOSED_LOOP_CONDITIONS
    + tuple(OPEN_LOOP_RETURNS.values())
)


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

#: How many degrees of freedom the online decoder emitted on the 22 blocks of the
#: two-degree-of-freedom task. The data cannot tell: a 4-DoF decoder driving the 2-DoF
#: task looks exactly like a 2-DoF decoder doing the same, because the game pins the
#: unused dimensions either way. Taken from the authors' ``ManuscriptFigsFinal.ipynb``,
#: by the cell named above each group. The ``Decoders/`` folders confirm the ten blocks
#: they cover, with one ``myMean`` entry per decoder output. Cell 31 only groups its
#: blocks with the 2-DoF decoder ones, and for 2023-04-13 block 32 that makes four
#: 2-DoF decoder blocks that day where the paper's Table 2 lists three.
DECODER_DOF: dict[str, int] = {
    # Cell 11, the 2-DoF decoder population ("529 trials (3 days)" in the paper).
    "t5.2023.03.21_block18": 2,
    "t5.2023.03.21_block19": 2,
    "t5.2023.03.23_block07": 2,
    "t5.2023.03.23_block08": 2,
    "t5.2023.03.23_block09": 2,
    "t5.2023.04.06_block15": 2,
    "t5.2023.04.06_block16": 2,
    # Cell 18, and the Decoders/ folders from 2023-04-06 block 43 on.
    "t5.2023.04.06_block31": 4,
    "t5.2023.04.06_block34": 4,
    "t5.2023.04.06_block43": 4,
    "t5.2023.04.06_block44": 2,
    "t5.2023.04.06_block45": 4,
    "t5.2023.04.06_block46": 2,
    "t5.2023.04.13_block20": 4,
    "t5.2023.04.13_block21": 2,
    "t5.2023.04.13_block23": 4,
    "t5.2023.04.13_block25": 2,
    "t5.2023.04.13_block27": 4,
    "t5.2023.04.13_block28": 2,
    # Cell 31.
    "t5.2023.04.06_block41": 2,
    "t5.2023.04.06_block48": 2,
    "t5.2023.04.13_block32": 2,
}

#: The Study downloads into this class-named folder under ``data/``.
DATA_SUBDIR = "Willsey2025FingerDecoding"


def default_data_dir() -> Path:
    """This dataset's data folder under ``NEURALSHED_DATA_ROOT``."""
    return dataset_dir("willsey_2025") / "data" / DATA_SUBDIR / "Data"


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
    # Wall-clock "HHMMSS" from the filename. It is when the file was saved, 1 to 2 s
    # after the last neural bin, so it orders blocks but does not mark their start.
    saved_time: str
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


def block_ref(path: Path) -> BlockRef:
    """The block a released ``.mat`` file holds, read from its name and folder."""
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
    return BlockRef(
        session=session, block=block, kind=kind, saved_time=match["time"], path=path
    )


def list_blocks(data_dir: str | Path | None = None) -> list[BlockRef]:
    """Every released block, in recording order."""
    data_dir = default_data_dir() if data_dir is None else data_dir
    refs = [
        block_ref(path)
        for path in sorted(Path(data_dir).glob("**/*.mat"))
        if not path.name.startswith("._") and path.parent.name != "Decoders"
    ]
    return sorted(refs, key=lambda ref: (ref.session, ref.saved_time))


def list_sessions(data_dir: str | Path | None = None) -> list[str]:
    """Recording days, oldest first."""
    data_dir = default_data_dir() if data_dir is None else data_dir
    return sorted({ref.session for ref in list_blocks(data_dir)})


def session_blocks(
    data_dir: str | Path | None = None,
) -> dict[str, list[BlockRef]]:
    data_dir = default_data_dir() if data_dir is None else data_dir
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
    power = power[:, :N_CHANNELS]
    if block_ref(Path(path)).session in SWAPPED_ARRAY_SESSIONS:
        power = power[:, np.r_[96:192, 0:96]]

    # Closed-loop blocks store the colour as an RGBA-shaped ``(n, 4)`` array whose first
    # column carries the code and whose others are always zero; the open-loop blocks
    # store the code alone as ``(1, n)``.
    color = np.asarray(raw[f"{prefix}taskOutput_stream_hand_color"])
    color = color[:, 0] if color.ndim == 2 and color.shape[1] == 4 else color.ravel()

    data = {
        "spike_band_power": power,
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
    for name, expected in (
        ("holdTime", HOLD_TIME_S),
        ("fn_hold_threshold", TARGET_HALF_WIDTH),
    ):
        if name in raw:
            data[name] = float(np.asarray(raw[name]).ravel()[0])
            if data[name] != expected:
                raise ValueError(f"{path}: {name} is {data[name]}, expected {expected}")
    return data


def load_block(ref: BlockRef) -> dict:
    """Named arrays for one block."""
    return _load(str(ref.path))


def get_neural(ref: BlockRef) -> np.ndarray:
    """Spike-band power as ``(n_bins, 192)``, in source order except on
    :data:`SWAPPED_ARRAY_SESSIONS`, whose arrays are swapped back."""
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
    the prefix packs 76 to 88 bins into 1.5 to 1.8 s rather than 4 s. The earliest trial
    starts at bin 121 (open loop) or 192 (closed loop), so no trial is lost.
    """
    clock = load_block(ref)["neural_clock"]
    gaps = np.flatnonzero(np.diff(clock) > 35)
    if gaps.size == 0:
        raise ValueError(f"{ref.label}: no 50 ms bin found")
    return int(gaps[0]) + 1


def on_bins(ref: BlockRef, key: str) -> np.ndarray:
    """A task-stream field resampled onto the neural bin times, as ``(n_bins, n_dims)``.

    Linear interpolation over the shared redis clock, matching ``LoadData`` in the
    authors' notebook. Neural bins outside the task stream's span (the recording
    starts a few seconds early) are clipped to the nearest task sample, so anything
    emitted from this keeps to :func:`task_span`.
    """
    data = load_block(ref)
    values = np.asarray(data[key], dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    clock = np.clip(data["neural_clock"], data["task_clock"][0], data["task_clock"][-1])
    return np.column_stack(
        [np.interp(clock, data["task_clock"], column) for column in values.T]
    )


def task_span(ref: BlockRef) -> tuple[int, int]:
    """Half-open neural-bin range that the task stream covers, from steady state on.

    The neural recording starts several seconds before the task stream (about 7 s in
    the closed-loop blocks, 2 s in the open-loop ones) and can run up to 29 s past its
    end. A finger value for a bin outside the task stream would be made up, so the
    finger streams are emitted over this range only. It starts no earlier than
    :func:`steady_state_start`.
    """
    data = load_block(ref)
    clock, task_clock = data["neural_clock"], data["task_clock"]
    first = int(np.searchsorted(clock, task_clock[0], side="left"))
    stop = int(np.searchsorted(clock, task_clock[-1], side="right"))
    return max(first, steady_state_start(ref)), stop


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

    This is a deliberate one-bin divergence from ``t5_parse.m``, the authors' open-loop
    parser, which matches each edge to its *nearest* bin. Nearest-matching puts the
    first bin of a trial on the window that mostly covered the previous target, because
    the acquisition flash is shorter than one bin: measured on the released blocks, the
    cued target changes at ``trial start + 1`` in 95 of 101 trials. It also makes
    consecutive trials share a bin, which an extractor rejects as overlapping coverage.
    The authors only used nearest-matching to average 2 s open-loop windows, where one
    bin is immaterial.
    """
    clock = load_block(ref)["neural_clock"]
    first = np.searchsorted(clock, begin, side="right")
    last = np.searchsorted(clock, end, side="right")
    # The callers pair these spans with per-trial targets and conditions by position,
    # so a span covering no bin cannot be dropped without shifting every later label.
    if np.any(last <= first):
        raise ValueError(f"{ref.label}: a trial window covers no neural bin")
    return [(int(a), int(b)) for a, b in zip(first, last)]


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
    """The target each trial cued, as ``(n_trials, 4)`` over DECODED_DIMS."""
    data = load_block(ref)
    _, stops = _trial_edges(data["hand_color"])
    return data["target"][np.ix_(stops, list(DECODED_COLUMNS))]


def trial_success(ref: BlockRef) -> np.ndarray:
    """Whether each closed-loop trial acquired its target, as ``(n_trials,)`` bools.

    The flash that closes a trial also fires at the 10 s timeout, and the release does
    not record the outcome. A trial succeeded when every cued degree of freedom stayed
    within :data:`TARGET_HALF_WIDTH` of the target for the :data:`HOLD_TIME_S` before
    the flash, the test of the authors' ``CalculateTrialStats``. It runs on the
    full-rate task stream, not on neural bins as theirs does, because the neural
    recording of ``t5.2023.03.09_block18`` stops 155 ms before its last trial's flash.
    """
    data = load_block(ref)
    clock = data["task_clock"]
    position = data["position"][:, list(DECODED_COLUMNS)]
    target = data["target"][:, list(DECODED_COLUMNS)]
    cued = [DECODED_DIMS.index(dim) for dim in target_dof(ref)]
    _, stops = _trial_edges(data["hand_color"])
    begins = np.searchsorted(clock, clock[stops] - HOLD_TIME_S * 1000, side="right")
    ends = np.searchsorted(clock, clock[stops], side="right")
    return np.array(
        [
            np.all(np.abs(position[a:b, cued] - target[stop, cued]) < TARGET_HALF_WIDTH)
            for a, b, stop in zip(begins, ends, stops)
        ]
    )


def target_dof(ref: BlockRef) -> tuple[str, ...]:
    """The degrees of freedom this block's task actually cued.

    Four of the closed-loop days interleave blocks whose targets move all four
    degrees of freedom with blocks that pin the thumb's abduction and the ring-little
    group at the neutral posture, leaving a two-dimensional task.
    """
    target = load_block(ref)["target"][:, list(DECODED_COLUMNS)]
    varies = [np.unique(target[:, i]).size > 1 for i in range(len(DECODED_DIMS))]
    return tuple(dim for dim, moves in zip(DECODED_DIMS, varies) if moves)


def decoder_dof(ref: BlockRef) -> int:
    """How many degrees of freedom the online decoder of a closed-loop block emitted.

    All four on the four-degree-of-freedom task, which a two-output decoder cannot
    drive, and :data:`DECODER_DOF` on the two-degree-of-freedom task.
    """
    if len(target_dof(ref)) == len(DECODED_DIMS):
        return len(DECODED_DIMS)
    return DECODER_DOF[ref.label]


def new_targets(ref: BlockRef) -> int:
    """How many :data:`FINGERS` each outward trial of a closed-loop block moves.

    The return trial that follows brings the same fingers back to neutral. Two in every
    block except blocks 44, 46, 47 and 50 of 2023-04-13, where the authors cued one
    finger at a time.
    """
    off = trial_targets(ref) != POSITION_NEUTRAL
    moved = sum(off[:, list(dims)].any(axis=1) for dims in FINGERS.values())
    counts = np.unique(moved[moved > 0])
    if counts.size != 1:
        raise ValueError(f"{ref.label}: outward trials move {counts.tolist()} fingers")
    return int(counts[0])


def cue_windows(ref: BlockRef) -> list[tuple[int, int, str]]:
    """``(start, stop, condition)`` neural-bin ranges of an open-loop block's go periods.

    The open-loop blocks alternate 1 s with the fingers still (``hand_color == 1``)
    with a 2 s go phase (``hand_color == 2``) while the cued posture changes every 3 s,
    so the go periods are the 2 s windows the authors average over. ``condition`` is the
    :data:`OPEN_LOOP_CONDITIONS` name of the posture the phase ends at, or the
    :data:`OPEN_LOOP_RETURNS` name when it ends at neutral after an outward movement.
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
    conditions = []
    previous = (POSITION_NEUTRAL,) * len(DECODED_COLUMNS)
    for _, stop in runs:
        end = tuple(target[stop - 1, list(DECODED_COLUMNS)])
        condition = OPEN_LOOP_CONDITIONS[end]
        if condition == "rest" and previous != end:
            condition = OPEN_LOOP_RETURNS[previous]
        conditions.append(condition)
        previous = end
    return [(a, b, condition) for (a, b), condition in zip(spans, conditions)]


def quadcopter_path(data_dir: str | Path | None = None) -> np.ndarray:
    """The saved quadcopter flight path as ``(n_samples, 3)`` in ``x, y, z``.

    The file carries no clock, so the sample rate is not recoverable; the authors plot
    it against sample index. ``z`` points down, so elevation is ``-z`` (their
    ``PlotFlightPath.m`` plots ``x, y, -z`` against rings centred at ``z = +14.91``).
    """
    data_dir = default_data_dir() if data_dir is None else data_dir
    refs = [ref for ref in list_blocks(data_dir) if ref.kind == KIND_QUADCOPTER]
    if len(refs) != 1:
        raise ValueError(f"expected exactly one quadcopter block, found {len(refs)}")
    return load_block(refs[0])["flight_path"]
