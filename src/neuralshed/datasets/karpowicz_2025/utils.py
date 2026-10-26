"""Readers for the Karpowicz et al. 2025 human iBCI release (two NWB sessions).

Read through ``h5py`` rather than ``pynwb``: both files carry a de-identified
``session_start_time`` of ``1900-10-01T00:00:00-05:50:36``, whose seconds-resolution UTC
offset ``dateutil`` refuses to parse, so ``NWBHDF5IO.read`` raises before any data is
reached. The layout is plain NWB 2.6 anyway (six ``acquisition`` time series on one
shared clock, plus an ``intervals/trials`` table). pynwb #2230 fixes the read upstream but
had not shipped as of 4.1.0.

Two source quirks are corrected here rather than in ``study.py``:

* **Timestamps are absolute Unix nanoseconds**, not the seconds NWB nominally stores.
  :func:`bin_times` divides them out and rebases each block on its own first bin.
* **``cursor_vel`` is a displacement per 20 ms bin**, not the ``position/s`` its unit
  string claims, so :func:`get_stream` scales it by :data:`FS` to make the declared unit
  true. It is the decoder's command, not the cursor's motion. A per-trial zero-intercept
  fit of ``diff(cursor_pos)`` against ``cursor_vel`` gives a slope of 0.99 (session day
  427) and 1.01 (day 153), but only in the calibration blocks. Across whole Simon blocks
  the cursor is frozen in about half the bins while the command keeps moving, and inside
  trial windows the two correlate at 0.66, with the cursor moving about twice as far as
  the command implies.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from neuralshed.datasets.utils import dataset_dir

N_CHANNELS = 192
N_ARRAYS = 2
NEURAL_BIN_SIZE_MS = 20
FS = 50.0
POSITION_DIMS = ("x", "y")

#: Session ids are keyed by day since implant, which is what the paper publishes (153
#: and 427 days). The file names carry no ordering: ``session2.nwb`` is the earlier one.
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
DEFAULT_DATA_DIR = dataset_dir("karpowicz_2025") / "data" / DATA_SUBDIR

TASK_CALIBRATION = "centeroutcalibration"
TASK_SIMON = "simonsequence"
TASK_LABELS = {
    TASK_CALIBRATION: "Radial-8 center-out, fits the session's Kalman filter",
    TASK_SIMON: "Simon memory-matching sequences of four cardinal targets",
}

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
            "start_time": table["start_time"][:] / 1e9,
            "stop_time": table["stop_time"][:] / 1e9,
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

    A trial is one movement, not one Simon sequence: the paper cuts it from the cursor's
    outward zero-crossing to 600 ms later or to target acquisition, whichever comes
    first. ``start_bin``/``stop_bin`` are block-relative and half-open, and trials that
    collide on a start bin are reduced to the first (see the comment below).
    """
    table = _trials(str(ref.path))
    table = table.loc[table["block_num"] == ref.block].copy()
    stamps = _file(str(ref.path))["acquisition/binned_spikes/timestamps"]
    start, _ = _block_bounds(str(ref.path))[ref.block]
    origin = stamps[start] / 1e9
    times = bin_times(ref)
    table["start_bin"] = np.searchsorted(times, table["start_time"] - origin)
    table["stop_bin"] = np.searchsorted(times, table["stop_time"] - origin)
    # Four pairs of trials in day 427 share a start time exactly, an artifact of the
    # authors' onset detection: for an up-then-down pair, the backward search for the
    # centre crossing landed on the same instant twice. The cursor follows the first of
    # each pair, so the second is dropped rather than re-anchored, which costs 4
    # movements of 1,116 and keeps trigger times unique.
    table = table.loc[~table["start_bin"].duplicated(keep="first")]
    return table.reset_index(drop=True)


def target_layout(ref: BlockRef) -> np.ndarray:
    """Distinct cued target centres in the block, sorted, as ``(n_targets, 2)``.

    ``cond_id`` is not comparable across block kinds: a calibration block spends it on
    eight radial targets at radius 0.5 (0.35 on the diagonals), while a Simon block uses
    only 1 to 4, on the four cardinal targets at radius 0.375.
    """
    table = trials(ref)
    return np.unique(table[["tgt_x", "tgt_y"]].to_numpy(), axis=0)


__all__ = [
    "N_CHANNELS",
    "N_ARRAYS",
    "NEURAL_BIN_SIZE_MS",
    "FS",
    "POSITION_DIMS",
    "SESSION_FILES",
    "POST_IMPLANT_DAY",
    "SESSIONS",
    "SUBJECT",
    "DATA_SUBDIR",
    "DEFAULT_DATA_DIR",
    "TASK_CALIBRATION",
    "TASK_SIMON",
    "TASK_LABELS",
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
]
