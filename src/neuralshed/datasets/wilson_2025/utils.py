"""Readers for Wilson et al. 2025 closed-loop cursor-control ``.mat`` sessions.

The release ships one ``.mat`` per recording day under ``T5/raw_data/historical``
(2016-09-26 to 2020-02-26) and ``T5/raw_data/new`` (2021-04-26 to 2021-07-26), 103
days in total. A day holds several back-to-back task blocks at 50 Hz, and the clocks
restart at every block. This module exposes each block separately, since the task
label and the online decoder change from block to block.

The source stores everything in a MATLAB struct (``dataset[0][0]``) whose fields have
names, and this module reads them by position, so the field indices below are the only
place the layout is spelled out.
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
NEURAL_BIN_SIZE_MS = 20
FS = 50.0
POSITION_DIMS = ("x", "y")
#: Spike-detection threshold as a multiple of channel RMS. The paper's Methods say
#: -4.5, but every released session carries -3.5 in its own header (field 13).
RMS_THRESHOLD = -3.5

#: The neural stream is a firing **rate in Hz**, not a per-bin count: every released
#: value is an exact multiple of 50, i.e. the 20 ms bin count scaled by 1/bin size.
#: Divide by this to recover integer counts per bin.
SPIKE_RATE_SCALE = FS

DATA_SUBDIR = "Wilson2025CursorRecalibration"


def default_data_dir() -> Path:
    """This dataset's data folder under ``NEURALSHED_DATA_ROOT``."""
    return (
        dataset_dir("wilson_2025")
        / "data"
        / DATA_SUBDIR
        / "PRITpaper2025"
        / "T5"
        / "raw_data"
    )


#: The two release folders. The paper treats them as one five-year series.
GROUPS = ("historical", "new")

TASK_LABELS = {
    "cursor": "Center-out target acquisition",
    "keyboard": "Grid target acquisition",
    "fitts": "Fitts-style random target acquisition",
}

#: Screen position of each task's coordinate origin. ``cursor`` and ``fitts`` store
#: positions from the screen centre, but ``keyboard`` stores them from the top-left
#: corner of a 1920 x 1080 screen. :func:`get_series` subtracts these offsets from the
#: cursor and target positions so every task is screen-centred, as the reference
#: ``DataStruct(alignScreens=True)`` does with ``screen_realignments.npy``.
SCREEN_ORIGINS = {
    "cursor": (0.0, 0.0),
    "keyboard": (960.0, 540.0),
    "fitts": (-0.5, -0.5),
}
_POSITION_KEYS = frozenset({"cursor_position", "target_position"})

#: On this day the two pedestal cables were swapped, so channels 0-95 and 96-191 are
#: exchanged relative to every other session. The reference ``DataStruct`` undoes it and
#: so does :func:`load_session`.
SWAPPED_CABLE_SESSIONS = frozenset({"t5.2017.08.04"})

#: A step of the system clock (source field 4) longer than this between two consecutive
#: rows of a block is a recording pause: time passed with no rows stored. Every other step
#: inside a block is 20 or 21 ms.
PAUSE_MIN_MS = 25

#: Days whose block list repeats a block number. On ``t5.2019.01.14`` blocks 24, 27, 28
#: and 29 each appear twice, and the second copy is byte-identical to the first (same
#: counts, same cursor positions, same system clock), so roughly 20 minutes of the day
#: is a verbatim duplicate. Only the first occurrence of a block number is exposed, which
#: keeps the duplicate out of training data and out of any train/test split. The authors
#: dropped this day from their analyses too (it has no entry in :data:`ANALYSIS_BLOCKS`).
DUPLICATED_BLOCK_SESSIONS = frozenset({"t5.2019.01.14"})

_SESSION_RE = re.compile(r"^t5\.(?P<y>\d{4})\.(?P<m>\d{2})\.(?P<d>\d{2})$")

# Fields of the source struct by position. 9 (`spikePow`) is always empty and 16
# (`instructedDelays`) is always all-NaN, so neither is exposed. 14 (`decodingClick`) is
# not in the release README: it is a per-block click-enabled flag, verified to agree
# with the click stream on all 560 exposed blocks.
_BLOCK_LIST = 0
_TARGET_SIZE = 1
_CURSOR_SIZE = 2
_BLOCK_PER_BIN = 3
_SYS_CLOCK = 4
_NSP_CLOCK = 5
_CURSOR_POS = 6
_TARGET_POS = 7
_ON_TARGET = 8
_DECODED_VELOCITY = 10
_DECODED_CLICK = 11
_THRESHOLD_CROSSINGS = 12
_RMS_THRESHOLD = 13
_CLICK_ENABLED = 14
_TRIAL_EPOCHS = 15
_INTERTRIAL = 17
_BLOCK_NAMES = 18
_TRIAL_SUCCESS = 19

#: Per-session blocks kept by the authors' analyses, transcribed from
#: ``getBlockConstraints`` in github.com/guyhwilson/nonstationarities. Selecting the
#: sessions with at least two such blocks reproduces the paper's 73-session cohort
#: exactly; an empty tuple means the day contributes no analysis block.
# fmt: off
ANALYSIS_BLOCKS: dict[str, tuple[int, ...]] = {
    "t5.2016.09.26": (7, 12, 19, 21, 24, 29),
    "t5.2016.09.28": (4, 6, 7, 8, 9, 10, 16, 18, 19, 22, 24, 25, 29, 35),
    "t5.2016.10.03": (10, 14, 15, 27, 29, 30, 31),
    "t5.2016.10.05": (8, 9, 11, 12, 13, 14, 16, 17, 18, 21, 22),
    "t5.2016.10.07": (2, 3, 4, 5, 6, 8, 9, 10, 11, 14, 15, 16, 17, 18),
    "t5.2016.10.10": (5, 8, 10, 11, 12, 14, 15, 16, 17, 20, 21, 22, 24, 25, 26, 28, 29, 30),
    "t5.2016.10.12": (4, 10, 12, 13, 14, 17, 18, 19, 22, 23, 25, 26, 28, 29, 30, 31),
    "t5.2016.10.13": (2, 3, 5, 7, 8, 9, 13, 14, 15, 17, 18, 19, 21, 22, 24),
    "t5.2016.10.17": (2, 4),
    "t5.2016.10.19": (2, 4),
    "t5.2016.10.24": (3, 5, 17, 19, 26, 28, 34, 38, 41, 45),
    "t5.2016.10.26": (3, 5, 9, 13, 23, 28, 34),
    "t5.2016.10.31": (3, 5, 6, 7, 8, 9),
    "t5.2016.12.06": (3, 5),
    "t5.2016.12.08": (5, 7),
    "t5.2016.12.15": (4, 8, 11, 12, 13, 14, 17, 18, 19),
    "t5.2016.12.16": (6, 8),
    "t5.2016.12.19": (3, 5),
    "t5.2016.12.21": (3, 6, 12, 13),
    "t5.2017.01.04": (4, 7),
    "t5.2017.01.30": (27, 28, 29),
    "t5.2017.02.15": (19, 21, 22, 23, 24),
    "t5.2017.02.22": (5, 6, 7, 8, 9, 10),
    "t5.2017.03.30": (10, 11, 12, 14),
    "t5.2017.04.26": (15, 16, 17, 18),
    "t5.2017.05.24": (16, 17, 18, 19),
    "t5.2017.05.31": (3, 5),
    "t5.2017.07.07": (2, 3),
    "t5.2017.07.31": (18, 19, 20, 21),
    "t5.2017.08.04": (7, 8),
    "t5.2017.08.07": (6, 7, 8),
    "t5.2017.09.20": (4, 5, 6, 7, 8, 9, 10, 11, 12),
    "t5.2017.12.27": (4, 6, 13, 14, 15),
    "t5.2018.01.08": (0, 1, 2),
    "t5.2018.01.17": (2, 3),
    "t5.2018.01.19": (4,),
    "t5.2018.01.22": (7,),
    "t5.2018.01.24": (23, 24),
    "t5.2018.04.09": (4, 5, 6, 7, 8),
    "t5.2018.04.16": (),
    "t5.2018.04.18": (),
    "t5.2018.04.23": (),
    "t5.2018.05.14": (),
    "t5.2018.06.11": (),
    "t5.2018.06.25": (),
    "t5.2018.06.27": (),
    "t5.2018.07.02": (),
    "t5.2018.07.06": (),
    "t5.2018.07.11": (),
    "t5.2018.07.25": (5, 6, 7, 8, 9),
    "t5.2018.07.27": (),
    "t5.2018.08.06": (),
    "t5.2018.08.08": (),
    "t5.2018.08.22": (24, 25, 26, 27, 28, 29),
    "t5.2018.09.17": (24, 25, 27, 28, 30, 31),
    "t5.2018.10.24": (2, 3, 4, 5, 6),
    "t5.2018.11.21": (),
    "t5.2018.11.28": (),
    "t5.2018.12.12": (),
    "t5.2018.12.17": (),
    "t5.2018.12.19": (13, 14, 15, 16, 17),
    "t5.2019.01.09": (),
    "t5.2019.01.14": (),
    "t5.2019.01.16": (),
    "t5.2019.01.23": (16, 17, 18, 19, 21),
    "t5.2019.01.30": (),
    "t5.2019.02.22": (22, 23, 24, 25, 26),
    "t5.2019.03.13": (),
    "t5.2019.03.20": (27, 28, 29, 30, 32),
    "t5.2019.03.27": (11, 18, 20, 22, 25, 27, 29),
    "t5.2019.04.01": (),
    "t5.2019.04.03": (),
    "t5.2019.04.08": (),
    "t5.2019.04.29": (26, 27, 28, 29, 30),
    "t5.2019.05.29": (11, 12, 13, 14, 15),
    "t5.2019.06.19": (26, 27, 28, 29, 30),
    "t5.2019.07.01": (24, 25, 26, 27, 28),
    "t5.2019.09.18": (),  # array reference issue day
    "t5.2019.10.28": (2, 4, 5, 6, 7),
    "t5.2019.11.27": (31, 33, 36),
    "t5.2019.12.09": (22, 23, 24, 25, 26),
    "t5.2020.01.13": (26, 27, 28, 30),
    "t5.2020.02.26": (),
    "t5.2021.04.26": (5, 6, 7),
    "t5.2021.04.28": (2, 4, 5),
    "t5.2021.05.03": (6,),
    "t5.2021.05.05": (3, 4, 5),
    "t5.2021.05.17": (2, 3, 4),
    "t5.2021.05.19": (1, 2),
    "t5.2021.05.24": (2, 3, 4),
    "t5.2021.05.26": (1, 2, 3),
    "t5.2021.06.02": (2, 3, 4),
    "t5.2021.06.04": (16, 17, 18),
    "t5.2021.06.07": (15, 16, 17),
    "t5.2021.06.23": (1, 3),
    "t5.2021.06.28": (1, 2, 3),
    "t5.2021.06.30": (5, 7),
    "t5.2021.07.07": (1, 2, 3),
    "t5.2021.07.08": (2, 3, 4),
    "t5.2021.07.12": (4, 5, 6),
    "t5.2021.07.14": (1, 3, 4),
    "t5.2021.07.19": (1, 2, 3),
    "t5.2021.07.26": (2, 3, 4),
}
# fmt: on

#: Days the authors' ``getBlockConstraints`` marks as bad, with the reason its comment
#: gives. Every timeline of these days carries ``valid=False``.
INVALID_SESSIONS: dict[str, str] = {
    "t5.2019.09.18": "array reference issue (authors' analysis code)",
}

#: Blocks whose task held the cursor still after each target appeared and released it
#: later, like an instructed delay. The median trial of these blocks keeps the cursor at
#: its start position for 26 to 106 bins, against at most 14 on every other block. The
#: source records no cue times (field 16 is all NaN).
HELD_CURSOR_BLOCKS: dict[str, tuple[int, ...]] = {
    "t5.2019.04.01": (6, 8, 10, 12, 14, 16, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31),
    "t5.2021.06.04": (15,),
    "t5.2021.06.07": (14,),
}

#: Blocks where the computer moved the cursor (open loop). Field 10 is the cursor's own
#: step there, and every outward path is straight with the same speed profile. No
#: source labels loop type, so this is inferred from the cursor data (README).
OPEN_LOOP_BLOCKS: dict[str, tuple[int, ...]] = {
    "t5.2021.06.04": (15,),
    "t5.2021.06.07": (14,),
}


@dataclass(frozen=True)
class BlockRef:
    """One task block within a session file."""

    session: str  # e.g. "t5.2016.09.26"
    block: int  # the source's own block number (not contiguous)
    task: str  # "cursor" | "keyboard" | "fitts"
    group: str  # "historical" | "new"
    path: Path

    @property
    def run(self) -> str:
        return f"block{self.block:02d}"

    @property
    def label(self) -> str:
        return f"{self.session}_{self.run}"

    @property
    def task_label(self) -> str:
        return TASK_LABELS[self.task]

    @property
    def is_analysis_block(self) -> bool:
        """Whether the authors' analyses kept this block (see :data:`ANALYSIS_BLOCKS`)."""
        return self.block in ANALYSIS_BLOCKS.get(self.session, ())

    @property
    def loop_type(self) -> str:
        """``open`` on :data:`OPEN_LOOP_BLOCKS`, ``closed`` elsewhere."""
        return (
            "open" if self.block in OPEN_LOOP_BLOCKS.get(self.session, ()) else "closed"
        )


def session_date(session: str) -> dt.date | None:
    """``"t5.2016.09.26"`` -> ``date(2016, 9, 26)``."""
    match = _SESSION_RE.match(session)
    if match is None:
        return None
    return dt.date(int(match["y"]), int(match["m"]), int(match["d"]))


@lru_cache(maxsize=2)
def _load(path: str, session: str) -> dict[str, np.ndarray]:
    """Read one session file into named arrays (at most two held in memory).

    Files run 0.7 to 16.3 MB. Index conversions and the swapped-cable fix are applied
    here so every caller sees the same zero-based, half-open convention.
    """
    raw = sio.loadmat(path)["dataset"][0][0]

    counts = np.asarray(raw[_THRESHOLD_CROSSINGS])
    if counts.shape[1] != N_CHANNELS:
        raise ValueError(
            f"{session}: expected {N_CHANNELS} channels, got {counts.shape[1]}"
        )
    if session in SWAPPED_CABLE_SESSIONS:
        counts = counts[:, np.r_[96:192, 0:96]]

    # Source trial epochs are MATLAB 1-based with an inclusive end; make them
    # zero-based half-open. The same holds for the intertrial periods.
    epochs = np.atleast_2d(np.asarray(raw[_TRIAL_EPOCHS])).astype(np.int64) - 1
    epochs[:, 1] += 1
    # Some days carry all-NaN rows here, so drop them before the integer cast.
    intertrial = np.atleast_2d(np.asarray(raw[_INTERTRIAL], dtype=np.float64))
    intertrial = intertrial[~np.isnan(intertrial).any(axis=1)].astype(np.int64) - 1
    intertrial[:, 1] += 1

    block_names = [
        str(entry[0][0]) if len(entry[0]) else "" for entry in raw[_BLOCK_NAMES]
    ]
    return {
        "block_list": np.asarray(raw[_BLOCK_LIST]).ravel().astype(np.int64),
        "block_names": np.asarray(block_names, dtype=object),
        "click_enabled": np.asarray(raw[_CLICK_ENABLED]).ravel().astype(bool),
        "block_per_bin": np.asarray(raw[_BLOCK_PER_BIN]).ravel().astype(np.int64),
        "threshold_crossings": counts,
        "cursor_position": np.asarray(raw[_CURSOR_POS], dtype=np.float64),
        "target_position": np.asarray(raw[_TARGET_POS], dtype=np.float64),
        "cursor_decoder_output": np.asarray(raw[_DECODED_VELOCITY], dtype=np.float64),
        "click_decoder_output": np.asarray(
            raw[_DECODED_CLICK], dtype=np.float64
        ).reshape(-1, 1),
        "on_target": np.asarray(raw[_ON_TARGET], dtype=np.float64).reshape(-1, 1),
        "target_size": np.asarray(raw[_TARGET_SIZE]).ravel().astype(np.float64),
        "cursor_size": np.asarray(raw[_CURSOR_SIZE]).ravel().astype(np.float64),
        "system_clock": np.asarray(raw[_SYS_CLOCK]).ravel().astype(np.float64),
        "nsp_clock": np.asarray(raw[_NSP_CLOCK], dtype=np.float64),
        "trial_epochs": epochs,
        "trial_success": np.asarray(raw[_TRIAL_SUCCESS]).ravel().astype(bool),
        "intertrial": intertrial,
        "rms_threshold": float(np.asarray(raw[_RMS_THRESHOLD]).ravel()[0]),
    }


def load_session(ref_or_path: "BlockRef | str | Path", session: str = "") -> dict:
    """Named arrays for a whole session file."""
    if isinstance(ref_or_path, BlockRef):
        return _load(str(ref_or_path.path), ref_or_path.session)
    path = Path(ref_or_path)
    return _load(str(path), session or path.stem)


def list_session_files(
    data_dir: str | Path | None = None,
) -> list[tuple[str, str, Path]]:
    """``(session, group, path)`` for every released day, oldest first."""
    data_dir = default_data_dir() if data_dir is None else data_dir
    root = Path(data_dir)
    out = []
    for group in GROUPS:
        for path in (root / group).glob("t5.*.mat"):
            if path.name.startswith("._"):  # macOS resource forks on the data drive
                continue
            out.append((path.stem, group, path))
    return sorted(out, key=lambda item: item[0])


def list_sessions(data_dir: str | Path | None = None) -> list[str]:
    data_dir = default_data_dir() if data_dir is None else data_dir
    return [session for session, _, _ in list_session_files(data_dir)]


def list_blocks(data_dir: str | Path | None = None) -> list[BlockRef]:
    """Every task block across every released day, in recording order."""
    data_dir = default_data_dir() if data_dir is None else data_dir
    refs = []
    for session, group, path in list_session_files(data_dir):
        data = _load(str(path), session)
        seen: set[int] = set()
        for number, name in zip(data["block_list"], data["block_names"]):
            number = int(number)
            if number in seen:  # see DUPLICATED_BLOCK_SESSIONS
                continue
            seen.add(number)
            refs.append(
                BlockRef(
                    session=session,
                    block=number,
                    task=str(name),
                    group=group,
                    path=path,
                )
            )
    return refs


def session_blocks(
    data_dir: str | Path | None = None,
) -> dict[str, list[BlockRef]]:
    data_dir = default_data_dir() if data_dir is None else data_dir
    out: dict[str, list[BlockRef]] = {}
    for ref in list_blocks(data_dir):
        out.setdefault(ref.session, []).append(ref)
    return out


def block_window(ref: BlockRef) -> tuple[int, int]:
    """Half-open ``(start, stop)`` bin range of this block within its session.

    Every block occupies one contiguous run of bins, except on the days in
    :data:`DUPLICATED_BLOCK_SESSIONS` where a repeated block number produces a second,
    identical run. The first run is the one returned.
    """
    data = load_session(ref)
    idx = np.flatnonzero(data["block_per_bin"] == ref.block)
    if idx.size == 0:
        raise ValueError(f"{ref.label}: block absent from the per-bin block index")
    breaks = np.flatnonzero(np.diff(idx) != 1)
    first_run = idx[: breaks[0] + 1] if breaks.size else idx
    if breaks.size and ref.session not in DUPLICATED_BLOCK_SESSIONS:
        raise ValueError(f"{ref.label}: block bins are unexpectedly non-contiguous")
    return int(first_run[0]), int(first_run[-1]) + 1


def section_windows(ref: BlockRef) -> list[tuple[int, int]]:
    """Half-open, block-relative bin ranges between this block's recording pauses.

    The system clock advances 20 or 21 ms per bin except at a recording pause, where it
    advances by the pause length and the NSP clocks advance by the same amount. 310 of
    the 560 blocks hold 454 pauses (median 9.5 s, up to 193 s), so a block has one
    section more than it has pauses.
    """
    clock = get_series(ref, "system_clock").ravel()
    cuts = (np.flatnonzero(np.diff(clock) > PAUSE_MIN_MS) + 1).tolist()
    bounds = [0, *cuts, len(clock)]
    return list(zip(bounds[:-1], bounds[1:]))


def unrecorded_arrays(ref: BlockRef) -> np.ndarray:
    """``(n_bins, N_ARRAYS)`` booleans, True where that array's NSP clock reads 0.

    Column 0 is the NSP for channels 0-95 and column 1 the one for 96-191. A clock reads
    0 only at the start of ``t5.2018.08.08`` blocks 12, 16 and 18 (0.54, 6.12 and 9.48 s),
    always the first one, and channels 0-95 are then all zero on 96 to 100% of bins while
    96-191 record normally: that NSP delivered no data, and the zeros stand for it.
    """
    return get_series(ref, "nsp_clock") == 0


def n_bins(ref: BlockRef) -> int:
    start, stop = block_window(ref)
    return stop - start


def duration(ref: BlockRef) -> float:
    return n_bins(ref) / FS


def get_series(ref: BlockRef, key: str) -> np.ndarray:
    """One per-bin field sliced to this block, as ``(n_bins, n_dims)``."""
    data = load_session(ref)
    if key not in data:
        raise KeyError(f"{ref.label}: no such field {key!r}")
    values = np.asarray(data[key])
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError(f"{ref.label}:{key} expected 2-D, got {values.shape!r}")
    start, stop = block_window(ref)
    values = values[start:stop]
    if key in _POSITION_KEYS:
        values = values - np.asarray(SCREEN_ORIGINS[ref.task])
    return values


def click_enabled(ref: BlockRef) -> bool:
    """Whether a click decoder ran in this block (source field 14)."""
    data = load_session(ref)
    index = int(np.flatnonzero(data["block_list"] == ref.block)[0])
    return bool(data["click_enabled"][index])


def _windows_in_block(ref: BlockRef, spans: np.ndarray) -> list[tuple[int, int]]:
    """Spans whose start falls in this block, clipped to it and made block-relative.

    A trial is assigned to the block containing its first bin, matching the reference
    ``DataStruct``. The source's inclusive end puts the last trial one bin into the next
    block in 311 of the 560 exposed blocks and at the block's edge in the other 249, so
    clipping loses no data.
    """
    start, stop = block_window(ref)
    out = []
    for span_start, span_stop in spans:
        span_start, span_stop = int(span_start), int(span_stop)
        if not (start <= span_start < stop):
            continue
        span_stop = min(span_stop, stop)
        if span_stop > span_start:
            out.append((span_start - start, span_stop - start))
    return out


def trial_windows(ref: BlockRef) -> list[tuple[int, int]]:
    """Half-open, block-relative ``(start, stop)`` bin ranges for this block's trials.

    On 53.9% of trials the source's last bin already shows the next trial's target, and a
    target never changes earlier in a trial. That bin is dropped, so each window ends at
    the last bin of its own target.
    """
    targets = get_series(ref, "target_position")
    windows = _windows_in_block(ref, load_session(ref)["trial_epochs"])
    return [
        (start, stop - 1)
        if stop - start > 1 and (targets[stop - 1] != targets[start]).any()
        else (start, stop)
        for start, stop in windows
    ]


def trial_indices(ref: BlockRef) -> list[int]:
    """Session-level index of each trial returned by :func:`trial_windows`."""
    data = load_session(ref)
    start, stop = block_window(ref)
    return [
        index
        for index, (span_start, _) in enumerate(data["trial_epochs"])
        if start <= int(span_start) < stop
    ]


def trial_success(ref: BlockRef) -> list[bool]:
    """Success flag per trial returned by :func:`trial_windows`."""
    data = load_session(ref)
    return [bool(data["trial_success"][index]) for index in trial_indices(ref)]


#: Bins of the 2021 lockout: the decoder output is zero for 25 bins (0.5 s) from a
#: trial's first or second bin while the cursor cannot be driven (README).
LOCKOUT_BINS = 25


def lockout_end(ref: BlockRef, start: int, stop: int) -> int | None:
    """First bin of nonzero decoder output after a lockout opening the trial
    ``[start, stop)``, or None. Only 2021 closed-loop blocks have lockouts."""
    if ref.group != "new" or ref.loop_type != "closed":
        return None
    zero = (get_series(ref, "cursor_decoder_output")[start:stop] == 0).all(axis=1)
    for offset in (0, 1):
        run = zero[offset : offset + LOCKOUT_BINS]
        if len(run) == LOCKOUT_BINS and run.all():
            nonzero = np.flatnonzero(~zero[offset:])
            return start + offset + (int(nonzero[0]) if len(nonzero) else len(zero))
    return None


def analysis_sessions(data_dir: str | Path | None = None) -> list[str]:
    """Days with at least two analysis blocks: the paper's 73-session cohort."""
    data_dir = default_data_dir() if data_dir is None else data_dir
    return [
        session
        for session in list_sessions(data_dir)
        if len(ANALYSIS_BLOCKS.get(session, ())) >= 2
    ]


__all__ = [
    "N_CHANNELS",
    "N_ARRAYS",
    "NEURAL_BIN_SIZE_MS",
    "FS",
    "POSITION_DIMS",
    "RMS_THRESHOLD",
    "SPIKE_RATE_SCALE",
    "DATA_SUBDIR",
    "default_data_dir",
    "GROUPS",
    "TASK_LABELS",
    "SCREEN_ORIGINS",
    "SWAPPED_CABLE_SESSIONS",
    "PAUSE_MIN_MS",
    "DUPLICATED_BLOCK_SESSIONS",
    "ANALYSIS_BLOCKS",
    "INVALID_SESSIONS",
    "HELD_CURSOR_BLOCKS",
    "OPEN_LOOP_BLOCKS",
    "LOCKOUT_BINS",
    "BlockRef",
    "session_date",
    "load_session",
    "list_session_files",
    "list_sessions",
    "list_blocks",
    "session_blocks",
    "block_window",
    "section_windows",
    "unrecorded_arrays",
    "n_bins",
    "duration",
    "get_series",
    "click_enabled",
    "trial_windows",
    "trial_indices",
    "trial_success",
    "lockout_end",
    "analysis_sessions",
]
