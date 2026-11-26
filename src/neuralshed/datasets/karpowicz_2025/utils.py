"""Readers for the Karpowicz et al. 2025 human iBCI release (two NWB sessions).

Read through ``h5py``: both files carry a de-identified ``session_start_time`` whose
seconds-resolution UTC offset ``dateutil`` refuses, so pynwb before 4.2.0 (#2230) raises
before any data is reached. The layout is plain NWB 2.6 (six ``acquisition`` time series
on one shared clock, plus an ``intervals/trials`` table).

Two source quirks are corrected here rather than in ``study.py``:

* **Timestamps are absolute Unix nanoseconds**, not the seconds NWB nominally stores.
  :func:`bin_times` divides them out and rebases each block on its own first bin.
* **``cursor_vel`` is a displacement per 20 ms bin**, not the ``position/s`` its unit
  string claims, so :func:`get_stream` scales it by :data:`FS` to make the declared unit
  true. It is the decoder's command, not the cursor's motion. Inside the Simon movement
  windows the cursor follows it: against the next row's cursor displacement it correlates
  at 0.94 on each axis, with slope 0.98 and a path ratio of 1.01 (0.84 and 0.89, 0.99 and
  1.04 in calibration, which is assisted at first). While the task holds the cursor at
  the centre or recentres it, the cursor does not follow the command.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from neuralshed.datasets.utils import dataset_dir

N_CHANNELS = 192
FS = 50.0
POSITION_DIMS = ("x", "y")

#: Session ids are keyed by day since implant, which is what the paper publishes (153
#: and 427 days). The file names follow the paper's session numbers, not the date order.
SESSION_FILES = {
    "T11_day153": "session2.nwb",
    "T11_day427": "session1.nwb",
}
POST_IMPLANT_DAY = {"T11_day153": 153, "T11_day427": 427}
SESSIONS = tuple(SESSION_FILES)
SUBJECT = "T11"

#: The Study resolves its own class-named subfolder, which is also where ``_download``
#: lands the deposit, so the two NWB files sit one level below ``data/``.
DATA_SUBDIR = "Karpowicz2025CursorStability"


def default_data_dir() -> Path:
    """This dataset's data folder under ``NEURALSHED_DATA_ROOT``."""
    return dataset_dir("karpowicz_2025") / "data" / DATA_SUBDIR


TASK_CALIBRATION = "centeroutcalibration"
TASK_SIMON = "simonsequence"
TASK_LABELS = {
    TASK_CALIBRATION: "Radial-8 center-out, fits the session's Kalman filter",
    TASK_SIMON: "Simon memory-matching sequences of four cardinal targets",
}

#: ``cond_id`` of each Simon target, keyed by its direction. The same in every Simon block.
SIMON_CODES = {(1, 0): 1, (0, 1): 2, (-1, 0): 3, (0, -1): 4}
SIMON_NAMES = {1: "right", 2: "up", 3: "left", 4: "down"}
#: Rows each target of a sequence presentation holds in the target stream (0.78 s).
PRESENTATION_STEP = 39
#: Rows from one movement's stop to the next movement's start within a sequence, the
#: median on day 427 (80 ms). Used where the source gives no start.
ESTIMATED_START_LAG = 4
#: Calibration ``cond_id`` of the eight outer targets, the same in both calibration blocks,
#: and of the centre target shown after each of them.
CALIBRATION_NAMES = {
    1: "right",
    2: "up-right",
    3: "up",
    4: "up-left",
    5: "left",
    6: "down-left",
    7: "down",
    8: "down-right",
}
CENTRE_CODE = 0
#: A return to the centre is coded this plus the ``cond_id`` of the target it leaves.
RETURN_CODE_OFFSET = 10

#: Streams under ``acquisition``, all on the one 50 Hz clock.
STREAMS = (
    "binned_spikes",
    "cursor_pos",
    "cursor_vel",
    "target_pos",
    "block_num",
    "cond_id",
)


@dataclass(frozen=True)
class BlockRef:
    """One task block: the unit this dataset treats as a timeline."""

    session: str
    block: int
    is_calibration: bool
    path: Path

    @property
    def task(self) -> str:
        return TASK_CALIBRATION if self.is_calibration else TASK_SIMON

    @property
    def label(self) -> str:
        return f"{self.session}_block{self.block:02d}"


@lru_cache(maxsize=4)
def _file(path: str) -> h5py.File:
    """Open handle, kept for the process. Blocks are sliced out of it lazily."""
    return h5py.File(path, "r")


@lru_cache(maxsize=8)
def _block_bounds(path: str) -> dict[int, tuple[int, int]]:
    """Half-open ``[start, stop)`` index range of every block in a session file."""
    blocks = _file(path)["acquisition/block_num/data"][:, 0].astype(int)
    edges = np.flatnonzero(np.diff(blocks)) + 1
    bounds = {}
    for start, stop in zip(np.r_[0, edges], np.r_[edges, len(blocks)]):
        bounds[int(blocks[start])] = (int(start), int(stop))
    return bounds


def session_path(data_dir: Path, session: str) -> Path:
    return Path(data_dir) / SESSION_FILES[session]


def list_blocks(data_dir: Path) -> list[BlockRef]:
    """Every block of every present session, sessions in implant-day order."""
    refs = []
    for session in SESSIONS:
        path = session_path(data_dir, session)
        if not path.exists():
            continue
        trials = _trials(str(path))
        for block in sorted(_block_bounds(str(path))):
            in_block = trials["block_num"] == block
            refs.append(
                BlockRef(
                    session=session,
                    block=block,
                    is_calibration=bool(trials.loc[in_block, "is_calibration"].any()),
                    path=path,
                )
            )
    return refs


def n_bins(ref: BlockRef) -> int:
    start, stop = _block_bounds(str(ref.path))[ref.block]
    return stop - start


def duration(ref: BlockRef) -> float:
    """Block length on the emitted clock, which counts bins rather than wall time."""
    return n_bins(ref) / FS


def bin_times(ref: BlockRef) -> np.ndarray:
    """Source bin times in seconds, rebased on the block's first bin.

    Wall time, so this is what trial times are looked up against. It runs ahead of
    ``arange(n_bins) / FS`` wherever the source dropped a bin (see :func:`dropped_bins`).
    """
    start, stop = _block_bounds(str(ref.path))[ref.block]
    stamps = _file(str(ref.path))["acquisition/binned_spikes/timestamps"][start:stop]
    return (stamps - stamps[0]) / 1e9


def dropped_bins(ref: BlockRef) -> int:
    """Bins missing from the block's 20 ms grid, almost all in day 427's calibration."""
    gaps = np.diff(bin_times(ref))
    return int(np.round((gaps[gaps > 1.5 / FS] * FS - 1).sum()))


def get_stream(ref: BlockRef, key: str) -> np.ndarray:
    """A block's slice of one acquisition stream, ``(n_bins, n_dims)`` time-first."""
    start, stop = _block_bounds(str(ref.path))[ref.block]
    data = _file(str(ref.path))[f"acquisition/{key}/data"][start:stop]
    if data.ndim == 1:
        data = data[:, None]
    if key == "cursor_vel":
        data = data * FS  # per-bin displacement -> per second
    return data


@lru_cache(maxsize=8)
def _trials(path: str) -> pd.DataFrame:
    table = _file(path)["intervals/trials"]
    return pd.DataFrame(
        {
            "trial_id": table["id"][:],
            "start_time": table["start_time"][:],
            "stop_time": table["stop_time"][:],
            "cond_id": table["cond_id"][:].astype(int),
            "tgt_x": table["tgt_x"][:],
            "tgt_y": table["tgt_y"][:],
            "result": table["result"][:].astype(int),
            "is_calibration": table["is_calibration"][:].astype(int),
            "block_num": table["block_num"][:].astype(int),
        }
    )


def trials(ref: BlockRef) -> pd.DataFrame:
    """The block's trials, with bin indices resolved against the source timestamps.

    A trial is one movement, not one Simon sequence. Its window is the trial table's own,
    from the movement's start to the row where its target is replaced, with no 600 ms cap
    (that cap is the paper's analysis choice). ``start_bin``/``stop_bin`` are block-relative and half-open.
    ``start_estimated`` marks the four movements whose start is estimated (see the comment below).
    """
    table = _trials(str(ref.path))
    table = table.loc[table["block_num"] == ref.block].copy()
    start, stop = _block_bounds(str(ref.path))[ref.block]
    stamps = _file(str(ref.path))["acquisition/binned_spikes/timestamps"][start:stop]
    # Every trial time is a copy of one row's stamp, so match the stored values as they
    # are. Converting to seconds first rounds them by about 2e-7 s and lands about half
    # the edges one row late.
    for time, row in (("start_time", "start_bin"), ("stop_time", "stop_bin")):
        times = table[time].to_numpy()
        rows = np.searchsorted(stamps, times)
        if not np.array_equal(stamps[np.minimum(rows, len(stamps) - 1)], times):
            raise ValueError(f"{ref.label}: a trial {time} is not a row stamp")
        table[row] = rows
    # In day 427, four down movements (trials 131, 269, 283 and 347) carry the start time
    # of the up movement before them. Their start is estimated from that movement's stop.
    repeated = table["start_time"].duplicated(keep="first")
    previous_stop = table.groupby("start_time")["stop_bin"].transform("first")
    table["start_estimated"] = repeated
    table.loc[repeated, "start_bin"] = previous_stop[repeated] + ESTIMATED_START_LAG
    table.loc[repeated, "start_time"] = stamps[table.loc[repeated, "start_bin"]]
    return table.reset_index(drop=True)


def target_layout(ref: BlockRef) -> np.ndarray:
    """Distinct cued target centres in the block, sorted, as ``(n_targets, 2)``.

    ``cond_id`` is not comparable across block kinds: a calibration block spends it on
    eight radial targets at radius 0.5 (0.35 on the diagonals), while a Simon block uses
    only 1 to 4, on the four cardinal targets at radius 0.375.
    """
    table = trials(ref)
    return np.unique(table[["tgt_x", "tgt_y"]].to_numpy(), axis=0)


def presentations(ref: BlockRef) -> list[tuple[int, int, tuple[int, ...]]]:
    """A Simon block's sequence presentations, as ``(start_bin, stop_bin, codes)``.

    Before each sequence the task holds the cursor at the centre and steps the target
    stream through the sequence's four targets, one every 0.78 s, then switches it back to
    the first target as the response opens. ``stop_bin`` is that switch, and ``codes`` are
    the four targets' ``cond_id`` in the order shown. When the first target is the one
    already in the stream, its step shows no change, so it is placed one step before the
    second. A change more than two steps before the second target is the end of the
    previous movement, not a step.
    """
    if ref.is_calibration:
        return []
    target = get_stream(ref, "target_pos")
    table = trials(ref)
    gap_starts = np.r_[0, table["stop_bin"].to_numpy()[:-1]]
    found = []
    for start, stop in zip(gap_starts, table["start_bin"]):
        moved = np.any(np.diff(target[start:stop], axis=0) != 0, axis=1)
        changes = start + 1 + np.flatnonzero(moved)
        if len(changes) < 4:
            continue
        steps = list(changes[-5:-1])
        if len(steps) < 4 or steps[1] - steps[0] > 2 * PRESENTATION_STEP:
            steps = [changes[-4] - PRESENTATION_STEP, *changes[-4:-1]]
        directions = np.sign(np.round(target[steps], 2)).astype(int)
        codes = tuple(SIMON_CODES[tuple(direction)] for direction in directions)
        found.append((int(steps[0]), int(changes[-1]), codes))
    return found


def returns(ref: BlockRef) -> list[tuple[int, int, int, tuple[float, float]]]:
    """A calibration block's returns to the centre, as ``(start_bin, stop_bin, code, target)``.

    After each outer target the task shows the centre target (``cond_id`` 0) and the
    participant drives the cursor back to it. A return spans the rows the centre target is
    shown, ending where the next outward movement starts. ``code`` is the ``cond_id`` of the
    outer target shown before it, and ``target`` the centre target's position.
    """
    if not ref.is_calibration:
        return []
    cond = get_stream(ref, "cond_id")[:, 0].astype(int)
    target = get_stream(ref, "target_pos")
    starts = trials(ref)["start_bin"].to_numpy()
    edges = np.flatnonzero(np.diff(cond)) + 1
    found = []
    for start, stop in zip(np.r_[0, edges], np.r_[edges, len(cond)]):
        if start == 0 or cond[start] != CENTRE_CODE:
            continue
        later = starts[starts > start]
        if len(later):
            stop = min(stop, later[0])
        centre = (float(target[start, 0]), float(target[start, 1]))
        found.append((int(start), int(stop), int(cond[start - 1]), centre))
    return found


__all__ = [
    "N_CHANNELS",
    "FS",
    "POSITION_DIMS",
    "SESSION_FILES",
    "POST_IMPLANT_DAY",
    "SESSIONS",
    "SUBJECT",
    "DATA_SUBDIR",
    "default_data_dir",
    "TASK_CALIBRATION",
    "TASK_SIMON",
    "TASK_LABELS",
    "SIMON_CODES",
    "SIMON_NAMES",
    "PRESENTATION_STEP",
    "ESTIMATED_START_LAG",
    "CALIBRATION_NAMES",
    "CENTRE_CODE",
    "RETURN_CODE_OFFSET",
    "STREAMS",
    "BlockRef",
    "session_path",
    "list_blocks",
    "n_bins",
    "duration",
    "bin_times",
    "dropped_bins",
    "get_stream",
    "trials",
    "target_layout",
    "presentations",
    "returns",
]
