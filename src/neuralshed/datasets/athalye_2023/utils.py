"""Readers for Athalye et al. 2023, monkey 2D cursor BMI (DANDI 000404).

Thirteen NWB files, one per session, with everything under ``processing/behavior`` as
regularly sampled ``TimeSeries`` plus a start/stop-only trials table. Read through
``h5py`` (``nwb.load_h5``), which serves a local file or a DANDI streaming URL alike.

The two monkeys come from different rigs and decoders, so rates and units differ:

* ``monk_g``: 60 Hz task frames, Kalman filter updated at 10 Hz (every sixth frame,
  flagged by ``update_bmi``), ``spike_counts`` per frame for 35 to 151 decoder units,
  positions in cm.
* ``monk_j``: 200 Hz, point-process filter updated every 5 ms sample, binary
  ``spike_counts`` for 20 or 21 decoder units, positions in m. :func:`get_cursor` and
  :func:`get_decoder_velocity` scale J to cm and cm/s so the dataset emits one unit.

Each file concatenates several task entries (blocks) back to back, center-out first
whatever the recording order, and ``te_num`` is the only record of the seams. A block is
the unit of continuous recording, so it is the unit here (:class:`BlockRef`).

Four monkey J blocks end in zero-filled padding (no spikes, frozen cursor), left by the
writer's preallocated arrays. :data:`DEAD_TAIL_START` records where each starts, and
every accessor stops there. :func:`find_dead_tail` recomputes them from the data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from neuralshed.datasets import nwb
from neuralshed.datasets.utils import data_root_is_set, dataset_dir

DANDISET_ID = "000404"
DANDI_VERSION = "draft"
STUDY_CLASS = "Athalye2023CursorBMI"


#: Where ``_download`` lands the dandiset (``<data>/<StudyClass>/000404/sub-*/...``).
def default_data_dir() -> Path | None:
    """Local copy under ``NEURALSHED_DATA_ROOT``, or None to stream from DANDI."""
    if not data_root_is_set():
        return None
    return dataset_dir("athalye_2023") / "data" / STUDY_CLASS


SUBJECTS = ("monk_g", "monk_j")
RATE_HZ = {"monk_g": 60.0, "monk_j": 200.0}
#: Decoder update rate, for reference: G's Kalman filter steps every sixth frame.
DECODER_RATE_HZ = {"monk_g": 10.0, "monk_j": 200.0}
#: Factor from the source position unit to cm (G is in cm, J in m).
TO_CM = {"monk_g": 1.0, "monk_j": 100.0}
POSITION_DIMS = ("x", "y")

TASK_CENTER_OUT = "center_out"
TASK_OBSTACLE = "obstacle_avoidance"

_BEHAVIOR = "processing/behavior"

#: First padded sample of each block that ends in zero-filled padding, as a file-level
#: sample index: the sample after the last change in ``decoder_state``. Spikes stop 1 to
#: 2 samples earlier. Verified with :func:`find_dead_tail`; no monkey G block has one.
DEAD_TAIL_START: dict[tuple[str, str], int] = {
    ("sub-monk-j_ses-session0", "jeev082413e"): 482803,  # 1846.0 s of padding
    ("sub-monk-j_ses-session1", "jeev082413h"): 33867,  # 755.7 s
    ("sub-monk-j_ses-session1", "jeev082413g"): 436188,  # 544.0 s
    ("sub-monk-j_ses-session3", "jeev083113f"): 378626,  # 266.9 s
}


@dataclass(frozen=True)
class BlockRef:
    """One task entry inside one session file: the unit this dataset treats as a timeline.

    ``start`` and ``stop`` are file-level sample indices, half-open. ``live_stop`` is where
    the recorded data end, equal to ``stop`` unless the block ends in padding.
    """

    source: str
    session: str
    subject: str
    task_entry: str
    task: str
    start: int
    stop: int
    live_stop: int

    @property
    def rate(self) -> float:
        return RATE_HZ[self.subject]

    @property
    def n_samples(self) -> int:
        return self.live_stop - self.start

    @property
    def duration(self) -> float:
        return self.n_samples / self.rate

    @property
    def padding_s(self) -> float:
        return (self.stop - self.live_stop) / self.rate


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


def session_label(relpath: str | Path) -> str:
    """Session id, the file stem (``sub-monk-g_ses-session0``)."""
    return Path(relpath).stem


def subject_of(relpath: str | Path) -> str:
    """``monk_g`` or ``monk_j``, the NWB ``subject_id``, from the asset name."""
    match = re.match(r"sub-monk-([gj])_", Path(relpath).name)
    if match is None:
        raise ValueError(f"not an athalye_2023 asset name: {relpath}")
    return f"monk_{match.group(1)}"


def _sort_key(relpath: str) -> tuple[str, int]:
    number = re.search(r"ses-session(\d+)", relpath)
    return subject_of(relpath), int(number.group(1)) if number else -1


def list_assets(data_dir: Path | str | None = None) -> list[str]:
    """Relative asset paths (``000404/sub-monk-g/...nwb``), local when present.

    Looks under ``data_dir`` and falls back to listing the dandiset for streaming. The
    relative path is the same in both cases.
    """
    data_dir = default_data_dir() if data_dir is None else data_dir
    if data_dir is not None:
        root = Path(data_dir)
        local = sorted((root / DANDISET_ID).glob("sub-*/*.nwb"))
        local = [p for p in local if not p.name.startswith("._")]
        if local:
            return sorted((str(p.relative_to(root)) for p in local), key=_sort_key)
    remote = nwb.dandi_list_assets(DANDISET_ID, DANDI_VERSION)
    return sorted((f"{DANDISET_ID}/{a['path']}" for a in remote), key=_sort_key)


def resolve_source(relpath: str, data_dir: Path | str | None = None) -> str:
    """Local file path when the asset is on disk, else its DANDI streaming URL."""
    data_dir = default_data_dir() if data_dir is None else data_dir
    if data_dir is not None and (Path(data_dir) / relpath).is_file():
        return str(Path(data_dir) / relpath)
    asset = relpath.removeprefix(DANDISET_ID + "/")
    return nwb.dandi_asset_url(DANDISET_ID, asset, DANDI_VERSION)


@lru_cache(maxsize=16)
def _file(source: str):
    """Open handle, kept for the process. Blocks are sliced out of it lazily."""
    return nwb.load_h5(source)


def _series(source: str, name: str):
    return _file(source)[f"{_BEHAVIOR}/{name}/data"]


def _decode(values: np.ndarray) -> np.ndarray:
    if values.dtype == object:
        return np.array(
            [v.decode() if isinstance(v, bytes) else str(v) for v in values]
        )
    return values


def file_rate(source: str) -> float:
    return float(_file(source)[f"{_BEHAVIOR}/spike_counts/starting_time"].attrs["rate"])


def session_date(source: str) -> str:
    """ISO date of the session, from ``session_start_time`` (midnight, real dates)."""
    value = _file(source)["session_start_time"][()]
    value = value.decode() if isinstance(value, bytes) else str(value)
    return value[:10]


def n_units(source: str) -> int:
    """Decoder units released for this session (columns of ``spike_counts``)."""
    return int(_series(source, "spike_counts").shape[1])


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------


def _task_entry_label(value: object) -> str:
    """``te_num`` as text: G stores floats (4377.0), J strings (``jeev082413d``)."""
    if isinstance(value, (float, np.floating)):
        return str(int(value))
    return str(value)


@lru_cache(maxsize=16)
def blocks(relpath: str, data_dir: Path | str | None = None) -> tuple[BlockRef, ...]:
    """Every task entry of a session file, in file order."""
    source = resolve_source(relpath, data_dir)
    session = session_label(relpath)
    subject = subject_of(relpath)
    rate = file_rate(source)
    if rate != RATE_HZ[subject]:
        raise ValueError(f"{session}: expected {RATE_HZ[subject]} Hz, file says {rate}")
    te_num = _decode(_series(source, "te_num")[:])
    obstacle = _series(source, "obstacle_position")[:]
    edges = np.flatnonzero(te_num[1:] != te_num[:-1]) + 1
    refs = []
    for start, stop in zip(np.r_[0, edges], np.r_[edges, len(te_num)]):
        start, stop = int(start), int(stop)
        entry = _task_entry_label(te_num[start])
        # Center-out blocks carry NaN obstacle positions throughout, in both monkeys.
        task = (
            TASK_CENTER_OUT if np.isnan(obstacle[start:stop]).all() else TASK_OBSTACLE
        )
        live_stop = DEAD_TAIL_START.get((session, entry), stop)
        if not start < live_stop <= stop:
            raise ValueError(f"{session}/{entry}: dead tail {live_stop} outside block")
        refs.append(
            BlockRef(
                source=source,
                session=session,
                subject=subject,
                task_entry=entry,
                task=task,
                start=start,
                stop=stop,
                live_stop=live_stop,
            )
        )
    if len({ref.task_entry for ref in refs}) != len(refs):
        raise ValueError(f"{session}: a task entry appears in two separate runs")
    return tuple(refs)


def block(
    relpath: str, task_entry: str, data_dir: Path | str | None = None
) -> BlockRef:
    for ref in blocks(relpath, data_dir):
        if ref.task_entry == task_entry:
            return ref
    raise KeyError(f"no task entry {task_entry!r} in {relpath}")


def find_dead_tail(ref: BlockRef, min_seconds: float = 5.0) -> int:
    """Recompute where a block's trailing padding starts, ``ref.stop`` when it has none.

    The padding is the trailing run with no spike on any unit and a frozen decoder state.
    Its start is the sample after the last change of ``decoder_state``.
    """
    counts = _series(ref.source, "spike_counts")[ref.start : ref.stop]
    state = _series(ref.source, "decoder_state")[ref.start : ref.stop]
    last_spike = int(np.flatnonzero(counts.sum(axis=1) > 0)[-1]) + 1
    changes = np.flatnonzero(np.any(np.diff(state, axis=0) != 0, axis=1))
    last_change = int(changes[-1]) + 2
    live = max(last_spike, last_change)
    if (ref.stop - ref.start - live) / ref.rate < min_seconds:
        return ref.stop
    return ref.start + live


# ---------------------------------------------------------------------------
# Streams, all time-first and cut to the live part of the block
# ---------------------------------------------------------------------------


def get_spike_counts(ref: BlockRef) -> np.ndarray:
    """``(n_samples, n_units)`` decoder-unit spike counts per sample.

    G: counts per 60 Hz frame (``uint32``). J: 0 or 1 per 5 ms sample (``uint8``).
    """
    return _series(ref.source, "spike_counts")[ref.start : ref.live_stop]


def get_cursor(ref: BlockRef) -> np.ndarray:
    """``(n_samples, 2)`` cursor x, y in cm. Equals the decoder's position state."""
    data = _series(ref.source, "cursor")[ref.start : ref.live_stop]
    return data * TO_CM[ref.subject]


def get_decoder_state(ref: BlockRef) -> np.ndarray:
    """``(n_samples, 5)`` raw decoder state ``[pos_x, pos_y, vel_x, vel_y, 1]``, source units."""
    return _series(ref.source, "decoder_state")[ref.start : ref.live_stop]


def get_decoder_velocity(ref: BlockRef) -> np.ndarray:
    """``(n_samples, 2)`` decoder velocity state in cm/s, the velocity driving the cursor."""
    return get_decoder_state(ref)[:, 2:4] * TO_CM[ref.subject]


def get_update_bmi(ref: BlockRef) -> np.ndarray:
    """``(n_samples,)`` 1 on samples where the decoder updated (every sixth frame for G)."""
    return _series(ref.source, "update_bmi")[ref.start : ref.live_stop]


# ---------------------------------------------------------------------------
# Trials
# ---------------------------------------------------------------------------


@lru_cache(maxsize=16)
def _trial_table(source: str) -> pd.DataFrame:
    table = _file(source)["intervals/trials"]
    return pd.DataFrame(
        {
            "trial_id": table["id"][:].astype(int),
            "start_time": table["start_time"][:],
            "stop_time": table["stop_time"][:],
        }
    )


def _obstacle_size_cm(label: str) -> float:
    match = re.fullmatch(r"(\d+)_cm_sq", label)
    return float(match.group(1)) if match else np.nan


def trials(ref: BlockRef) -> pd.DataFrame:
    """The block's rewarded trials, go cue to reward, with block-relative times.

    Columns: ``trial_id`` (the file's trial id, unique within a session), ``start`` and
    ``stop`` in seconds from the block start, and for monkey G obstacle trials
    ``obstacle_x``, ``obstacle_y`` (cm) and ``obstacle_size_cm``, NaN elsewhere. Monkey
    J's obstacle streams are misaligned by the writer and are not read.
    """
    table = _trial_table(ref.source)
    rate = ref.rate
    start_sample = np.round(table["start_time"].to_numpy() * rate).astype(int)
    stop_sample = np.round(table["stop_time"].to_numpy() * rate).astype(int)
    inside = (start_sample >= ref.start) & (start_sample < ref.stop)
    if np.any(stop_sample[inside] > ref.live_stop):
        raise ValueError(
            f"{ref.session}/{ref.task_entry}: a trial runs past the live data"
        )
    out = pd.DataFrame(
        {
            "trial_id": table["trial_id"].to_numpy()[inside],
            "start": (start_sample[inside] - ref.start) / rate,
            "stop": (stop_sample[inside] - ref.start) / rate,
        }
    )
    out["obstacle_x"] = np.nan
    out["obstacle_y"] = np.nan
    out["obstacle_size_cm"] = np.nan
    if ref.subject == "monk_g" and ref.task == TASK_OBSTACLE:
        at = start_sample[inside]
        position = _series(ref.source, "obstacle_position")[ref.start : ref.stop]
        details = _decode(_series(ref.source, "obs_details")[ref.start : ref.stop])
        local = at - ref.start
        out["obstacle_x"] = position[local, 0]
        out["obstacle_y"] = position[local, 1]
        out["obstacle_size_cm"] = [_obstacle_size_cm(d) for d in details[local]]
    return out.reset_index(drop=True)


__all__ = [
    "DANDISET_ID",
    "DANDI_VERSION",
    "DEAD_TAIL_START",
    "DECODER_RATE_HZ",
    "default_data_dir",
    "POSITION_DIMS",
    "RATE_HZ",
    "STUDY_CLASS",
    "SUBJECTS",
    "TASK_CENTER_OUT",
    "TASK_OBSTACLE",
    "TO_CM",
    "BlockRef",
    "block",
    "blocks",
    "file_rate",
    "find_dead_tail",
    "get_cursor",
    "get_decoder_state",
    "get_decoder_velocity",
    "get_spike_counts",
    "get_update_bmi",
    "list_assets",
    "n_units",
    "resolve_source",
    "session_date",
    "session_label",
    "subject_of",
    "trials",
]
