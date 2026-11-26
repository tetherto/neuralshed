"""Readers for Even-Chen et al. 2019 (DANDI 000121), two monkeys reaching center-out-and-back.

Twelve NWB files, one per session, all with the same layout: a ``units`` table of 192
threshold-crossing channels (spike times), four position series under
``processing/behavior/Position`` (``Hand``, ``Cursor``, ``Eye``, ``DecodePos``) and a trials
table. Read through ``h5py`` for speed. Every reader takes a ``source`` (a local path or a
DANDI URL) and opens it through ``datasets/nwb.py``.

Source quirks handled here, all verified against the twelve files:

* **No data between trials.** The session clock starts at the first trial, and behavior
  samples and spike times exist only inside ``[start_time, stop_time]`` of each trial. The
  gaps are 220 ms (JenkinsC) or 40 ms (Reggie, with pauses up to 11 minutes).
* **Behavior is 60 Hz sample-and-hold on a 1 kHz grid.** ``Hand`` and ``Cursor`` change every
  16 to 17 ms, ``Eye`` every 8 to 9 ms. :func:`binned_behavior` averages the 1 kHz samples into
  the same 20 ms bins as :func:`binned_spike_counts`.
* **Positions are millimeters** relative to screen center, although the files say meters.
* **Tracking loss is a sentinel row**, not NaN: ``Hand`` jumps to one of
  :data:`TRACKING_LOSS_ROWS` and ``Cursor`` copies it. ``Eye`` dropouts are all-zero rows.
  Both are set to NaN before binning.
* **Brain control is flagged only by ``DecodePos``.** It is NaN everywhere except Reggie
  2017-01-19 trials 974 to 1107, where it is filled on every sample and the cursor is the
  decoder output. :func:`get_trials` flags those trials from ``DecodePos``.
* **``units/electrode_group`` contradicts the electrodes table.** Units 0 to 95 link to
  electrodes 0 to 95, which the electrodes table and the LFP series put on the M1 array, but
  the units' own ``electrode_group`` column says PMd. The electrodes table wins: Reggie's 29
  silent channels would otherwise leave 67 live PMd channels, fewer than the 80 the paper
  reports as tuned. See :data:`ARRAYS`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from neuralshed.datasets import nwb
from neuralshed.datasets.utils import data_root_is_set, dataset_dir

DANDISET_ID = "000121"
VERSION = "0.220124.2156"

#: Folder the Study resolves (``<data>/<StudyClass>``). The dandiset lands below it as
#: ``000121/sub-*/*.nwb``, mirroring the archive layout.
STUDY_CLASS = "EvenChen2019DelayedReach"


def default_data_dir() -> Path | None:
    """Local copy under ``NEURALSHED_DATA_ROOT``, or None to stream from DANDI."""
    if not data_root_is_set():
        return None
    return dataset_dir("even_chen_2019") / "data" / STUDY_CLASS


SUBJECTS = ("JenkinsC", "Reggie")
N_CHANNELS = 192

#: Channel blocks in the order of the ``units`` table (see the module docstring).
ARRAYS = (
    ("M1", 0, 96),
    ("PMd", 96, 192),
)

#: Threshold as the paper states it, per monkey.
THRESHOLD = {"JenkinsC": "-4.5 x RMS", "Reggie": "-4.0 x RMS"}

BIN_SIZE_S = 0.02
FS = 1.0 / BIN_SIZE_S  # 50 Hz, shared by the neural and behavioral streams
SOURCE_FS = 1000.0  # the release's behavior grid
_SAMPLES_PER_BIN = round(BIN_SIZE_S * SOURCE_FS)

POSITION_KEYS = {
    "Hand": ("x", "y", "z"),
    "Cursor": ("x", "y"),
    "Eye": ("x", "y"),
}

#: ``Hand`` rows written while the optical tracker lost the bead. Three distinct rows occur,
#: one per rig configuration: JenkinsC 2015, JenkinsC 2016, Reggie 2017.
TRACKING_LOSS_ROWS = np.array(
    [[70.0, 1278.5, -219.0], [70.0, 1258.5, -219.0], [70.0, 1258.5, -149.0]]
)

#: Beyond this many millimeters from screen center a cursor sample is off the display.
WORKSPACE_LIMIT_MM = 500.0


@dataclass(frozen=True)
class SessionRef:
    """One NWB file, which is one session and one timeline."""

    subject: str
    path: str  # relative to the Study folder, e.g. ``000121/sub-Reggie/...nwb``

    @property
    def label(self) -> str:
        return session_label(self.path)

    @property
    def date(self) -> str:
        return session_date(self.path)


def session_label(path: str | Path) -> str:
    """Canonical session id: the NWB file stem, e.g. ``sub-Reggie_ses-20170119T123128_behavior+ecephys``."""
    return Path(path).stem


def session_date(path: str | Path) -> str:
    """ISO date from the ``ses-`` entity of the file name."""
    ses = Path(path).stem.split("_ses-")[1][:8]
    return f"{ses[:4]}-{ses[4:6]}-{ses[6:8]}"


def subject_of(path: str | Path) -> str:
    return Path(path).stem.split("_")[0].removeprefix("sub-")


def list_sessions(data_dir: Path | str | None = None) -> list[SessionRef]:
    """Local sessions under ``data_dir`` when present, else the dandiset's asset list."""
    data_dir = default_data_dir() if data_dir is None else data_dir
    local = []
    if data_dir is not None:
        data_dir = Path(data_dir)
        local = sorted(data_dir.glob(f"{DANDISET_ID}/sub-*/*.nwb"))
        local = [p for p in local if not p.name.startswith("._")]
    if local:
        rels = [str(p.relative_to(data_dir)) for p in local]
    else:
        rels = sorted(
            f"{DANDISET_ID}/{a['path']}"
            for a in nwb.dandi_list_assets(DANDISET_ID, VERSION, glob="*.nwb")
        )
    return [SessionRef(subject=subject_of(rel), path=rel) for rel in rels]


def resolve_source(data_dir: Path | str, rel: str) -> str:
    """A local path when the file exists under ``data_dir``, else its DANDI streaming URL."""
    local = Path(data_dir) / rel
    if local.is_file():
        return str(local)
    return nwb.dandi_asset_url(
        DANDISET_ID, rel.removeprefix(DANDISET_ID + "/"), VERSION
    )


# --------------------------------------------------------------------------- #
# Trials
# --------------------------------------------------------------------------- #
def _ragged(table, name: str) -> list[np.ndarray]:
    data = table[name][:]
    stops = table[f"{name}_index"][:].astype(int)
    starts = np.r_[0, stops[:-1]]
    return [data[a:b] for a, b in zip(starts, stops)]


def _trial_sample_bounds(
    start: np.ndarray, stop: np.ndarray, n_samples: int
) -> np.ndarray:
    """First behavior-sample index of every trial, plus the total as a last entry.

    Each trial holds one 1 kHz sample per millisecond of ``[start_time, stop_time]``, both
    ends included, and the trials follow one another in the behavior arrays.
    """
    per_trial = np.round((stop - start) * SOURCE_FS).astype(int) + 1
    bounds = np.r_[0, np.cumsum(per_trial)]
    if bounds[-1] != n_samples:
        raise ValueError(
            f"trial table implies {bounds[-1]} behavior samples, the file has {n_samples}"
        )
    return bounds


@lru_cache(maxsize=16)
def get_trials(source: str) -> pd.DataFrame:
    """The trials table, one row per trial, with the quantities the Study emits.

    Times are seconds on the session clock. ``reach_limit``, ``hold_limit`` and
    ``fail_limit`` are the source's ``reach_time``, ``target_hold_time`` and ``fail_time``
    with ``start_time`` taken back out, since the source stores them as limits plus the
    trial start rather than as events. ``brain_control`` is True where ``DecodePos`` is
    filled (the decoder drives the cursor).
    """
    with nwb.open_h5(source) as f:
        tr = f["intervals/trials"]
        start = tr["start_time"][:]
        stop = tr["stop_time"][:]
        n = len(start)
        target_pos = tr["target_pos"][:].reshape(n, 3)
        target_size = tr["target_size"][:].reshape(n, 3)
        acquire = _ragged(tr, "target_acquire_time")
        table = pd.DataFrame(
            {
                "trial_id": tr["id"][:].astype(int),
                "start_time": start,
                "stop_time": stop,
                "success": tr["is_successful"][:].astype(bool),
                "target_x": target_pos[:, 0],
                "target_y": target_pos[:, 1],
                "target_size": target_size[:, 0],
                "target_shown_time": tr["target_shown_time"][:],
                "go_cue_time": tr["go_cue_time"][:],
                "target_held_time": tr["target_held_time"][:],
                "first_acquire_time": [
                    float(a[0]) if len(a) and np.isfinite(a[0]) else np.nan
                    for a in acquire
                ],
                "reach_limit": tr["reach_time"][:] - start,
                "hold_limit": tr["target_hold_time"][:] - start,
                "fail_limit": tr["fail_time"][:] - start,
                "task_id": tr["task_id"][:].astype(int),
                "version_id": tr["version_id"][:].astype(int),
            }
        )
        positions = f["processing/behavior/Position"]
        bounds = _trial_sample_bounds(start, stop, positions["DecodePos/data"].shape[0])
        # DecodePos is either filled on every sample of a trial or on none, so one sample
        # per trial decides it without reading the whole array.
        middle = (bounds[:-1] + bounds[1:]) // 2
        decoded = positions["DecodePos/data"][middle, 0]

    table["brain_control"] = np.isfinite(decoded)
    table["is_return"] = (table["target_x"] == 0) & (table["target_y"] == 0)
    table["delay"] = table["go_cue_time"] - table["target_shown_time"]
    return table


@lru_cache(maxsize=16)
def cursor_off_screen(source: str) -> np.ndarray:
    """Per trial, whether most cursor samples lie beyond :data:`WORKSPACE_LIMIT_MM` without
    being the tracking-loss sentinel. True only for Reggie 2017-01-19 trials 1108 to 1116,
    right after the brain-control block. Reads the whole ``Cursor`` series."""
    trials = get_trials(source)
    with nwb.open_h5(source) as f:
        cursor = f["processing/behavior/Position/Cursor/data"][:]
    bounds = _trial_sample_bounds(
        trials["start_time"].to_numpy(), trials["stop_time"].to_numpy(), len(cursor)
    )
    off = (np.abs(cursor) > WORKSPACE_LIMIT_MM).any(axis=1) & ~_tracking_lost(cursor, 2)
    return np.add.reduceat(off.astype(float), bounds[:-1]) / np.diff(bounds) > 0.5


def target_codes(trials: pd.DataFrame) -> np.ndarray:
    """Session-local target index: rank of ``(target_x, target_y)`` among the session's
    distinct targets, sorted by x then y. The center target is one of them."""
    xy = trials[["target_x", "target_y"]].to_numpy()
    _, codes = np.unique(xy, axis=0, return_inverse=True)
    return codes.ravel().astype(int)


def target_layout(trials: pd.DataFrame) -> tuple[str, int]:
    """Name of the session's target layout and its number of outward targets.

    Derived from ``target_pos``, since ``task_id`` and ``version_id`` are constant per
    monkey. A line layout is named by its axis in the file, which for Reggie is the
    opposite of the paper's labels. Reggie 2017-01-17 adds six 80 mm targets to its
    vertical line in the first trials, hence the 90% rule.
    """
    out = trials.loc[~trials["is_return"], ["target_x", "target_y"]].to_numpy()
    n_targets = len(np.unique(out, axis=0))
    if np.mean(out[:, 0] == 0) >= 0.9:
        return "vertical_line", n_targets
    if np.mean(out[:, 1] == 0) >= 0.9:
        return "horizontal_line", n_targets
    radii = np.unique(np.round(np.hypot(out[:, 0], out[:, 1])))
    if len(radii) == 1:
        return "ring", n_targets
    if len(radii) == 3:
        return "three_rings", n_targets
    raise ValueError(f"unrecognized target layout with radii {radii}")


def n_bins(source: str) -> int:
    """20 ms bins from 0 through the last behavior sample of the session."""
    last_ms = int(round(get_trials(source)["stop_time"].iloc[-1] * SOURCE_FS))
    return last_ms // _SAMPLES_PER_BIN + 1


# --------------------------------------------------------------------------- #
# Dense streams, all on the 50 Hz grid starting at 0
# --------------------------------------------------------------------------- #
def binned_spike_counts(source: str, n_bins: int) -> np.ndarray:
    """Threshold crossings per 20 ms bin, ``(192, n_bins)`` float32, in ``units`` order."""
    counts = np.zeros((N_CHANNELS, n_bins), dtype=np.float32)
    with nwb.open_h5(source) as f:
        stops = f["units/spike_times_index"][:].astype(int)
        if len(stops) != N_CHANNELS:
            raise ValueError(f"expected {N_CHANNELS} units, found {len(stops)}")
        times = f["units/spike_times"]
        for channel, (a, b) in enumerate(zip(np.r_[0, stops[:-1]], stops)):
            if b == a:
                continue
            bins = np.floor(times[a:b] / BIN_SIZE_S).astype(np.int64)
            counts[channel] = np.bincount(
                np.clip(bins, 0, n_bins - 1), minlength=n_bins
            )
    return counts


def _tracking_lost(data: np.ndarray, n_dims: int) -> np.ndarray:
    lost = np.zeros(len(data), dtype=bool)
    for row in np.unique(TRACKING_LOSS_ROWS[:, :n_dims], axis=0):
        lost |= (data[:, :n_dims] == row).all(axis=1)
    return lost


def _bin_samples(samples: np.ndarray, ms_index: np.ndarray, n_bins: int) -> np.ndarray:
    """Average 1 kHz samples into 20 ms bins, ignoring NaN; ``(n_dims, n_bins)`` float32."""
    grid = np.full(
        (n_bins * _SAMPLES_PER_BIN, samples.shape[1]), np.nan, dtype=np.float32
    )
    grid[ms_index] = samples
    grid = grid.reshape(n_bins, _SAMPLES_PER_BIN, samples.shape[1])
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore", category=RuntimeWarning
        )  # all-NaN bins stay NaN
        binned = np.nanmean(grid, axis=1)
    return np.ascontiguousarray(binned.T, dtype=np.float32)


def binned_behavior(source: str, key: str, n_bins: int) -> np.ndarray:
    """One ``processing/behavior/Position`` series on the 50 Hz grid, ``(n_dims, n_bins)``.

    Tracking-loss rows (``Hand``, ``Cursor``) and all-zero rows (``Eye``) become NaN before
    averaging, as do the gaps between trials. A bin holding only NaN samples is NaN.
    """
    n_dims = len(POSITION_KEYS[key])
    with nwb.open_h5(source) as f:
        series = f[f"processing/behavior/Position/{key}"]
        data = series["data"][:, :n_dims].astype(np.float32)
        stamps = series["timestamps"][:]
    if key == "Eye":
        data[(data == 0).all(axis=1)] = np.nan
    else:
        data[_tracking_lost(data, n_dims)] = np.nan
    ms_index = np.round(stamps * SOURCE_FS).astype(np.int64)
    return _bin_samples(data, ms_index, n_bins)


def binned_target(source: str, n_bins: int) -> np.ndarray:
    """Cued target ``(x, y)`` from its appearance to the trial end, NaN elsewhere, ``(2, n_bins)``."""
    trials = get_trials(source)
    shown = trials.dropna(subset=["target_shown_time"])
    first = np.round(shown["target_shown_time"].to_numpy() * SOURCE_FS).astype(np.int64)
    last = np.round(shown["stop_time"].to_numpy() * SOURCE_FS).astype(np.int64)
    lengths = last - first + 1
    ms_index = np.repeat(first, lengths) + (
        np.arange(lengths.sum()) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    )
    xy = np.repeat(
        shown[["target_x", "target_y"]].to_numpy(np.float32), lengths, axis=0
    )
    return _bin_samples(xy, ms_index, n_bins)


__all__ = [
    "DANDISET_ID",
    "VERSION",
    "STUDY_CLASS",
    "default_data_dir",
    "SUBJECTS",
    "N_CHANNELS",
    "ARRAYS",
    "THRESHOLD",
    "BIN_SIZE_S",
    "FS",
    "SOURCE_FS",
    "POSITION_KEYS",
    "TRACKING_LOSS_ROWS",
    "WORKSPACE_LIMIT_MM",
    "SessionRef",
    "session_label",
    "session_date",
    "subject_of",
    "list_sessions",
    "resolve_source",
    "get_trials",
    "cursor_off_screen",
    "target_codes",
    "target_layout",
    "n_bins",
    "binned_spike_counts",
    "binned_behavior",
    "binned_target",
]
