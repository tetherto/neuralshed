"""Source-reading utilities for FALCON H1 (DANDI 000954).

The accessors take an open ``pynwb.NWBFile``, from a downloaded file or streamed with
``nwb.open_nwb`` (see ``datasets/nwb.py``).

The H1 NWB layout:

* Neural data: a ``Units`` table (``nwb.units``) whose 176 rows are channels of
  unsorted threshold crossings, not sorted units, with no link to an electrode.
  Spike times are not pre-binned.
  Kinematic sample ``k`` sits at ``k * 0.02`` s and closes the 20 ms bin
  ``[k * 0.02 - 0.02, k * 0.02)``, so spike times start at -0.02 s. ``bin_spikes``
  bins this way, as FALCON's ``bin_units`` does.
* Kinematics: ``nwb.acquisition["OpenLoopKinematics"]`` -- a ``TimeSeries`` of
  shape ``(n_bins, 7)``; the 7 DoF labels are in ``.description``
  (``tx,ty,tz,rx,g1,g2,g3``). Velocity in ``OpenLoopKinematicsVelocity``.
* Trial id per bin: ``nwb.acquisition["TrialNum"]`` shape ``(n_bins,)``.
* FALCON eval mask: ``nwb.acquisition["eval_mask"]`` -- bool ``(n_bins,)``;
  True marks bins scored by the FALCON benchmark.
* Intervals: ``nwb.intervals["epochs"]`` -- one row per task phase, with a single tag.

Timebase quirk: the kinematics ``TimeSeries`` stores ``rate = 0.02``. That value
is the **bin period in seconds** (20 ms -> 50 Hz), not a sample rate in Hz.
Sample ``k`` sits at ``k * 0.02`` s. Use :data:`BIN_SIZE_S` rather than the raw
``.rate`` field.
"""

from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pynwb

from neuralshed.datasets import nwb as _nwb
from neuralshed.datasets.utils import data_root_is_set, dataset_dir

# --- Constants (H1) -------------------------------------------------------
DANDISET_ID = "000954"
VERSION = "draft"
STUDY_CLASS = "Karpowicz2024FalconH1"
SUBJECT = "HumanPitt"
N_CHANNELS = 176  # Units-table rows, one channel each, in every released file
BIN_SIZE_S = 0.02  # 20 ms bins -> 50 Hz (stored, misleadingly, in TimeSeries.rate)
BIN_RATE_HZ = 50.0
# 3 translation, 1 roll, 3 grasp
DOF_LABELS = ("tx", "ty", "tz", "rx", "g1", "g2", "g3")
_SPLITS = ("held-in-calib", "held-in-minival", "held-out-calib")
# Each held-in-minival file is an exact copy of the first bins of the held-in-calib
# file with the same timestamp, so the Study leaves it out.
MINIVAL_SPLIT = "held-in-minival"


def default_data_dir() -> Path | None:
    """The Study's folder under ``NEURALSHED_DATA_ROOT``, or None to stream from DANDI."""
    if not data_root_is_set():
        return None
    return dataset_dir("karpowicz_2024_h1") / "data" / STUDY_CLASS


def split_of(path: str) -> str:
    """The FALCON split (held-in/out) embedded in an asset path."""
    for s in _SPLITS:
        if s in path:
            return s
    raise ValueError(f"no FALCON split in {path!r}")


def recordings(paths: Iterable[str]) -> List[Dict]:
    """One row per recording, minival copies left out, sorted by start time.

    ``block`` numbers the recordings of a day by start time from 1, and ``session`` is
    ``<day>_block<block>`` (e.g. ``19250101_block1``). It is the join key between
    ``study.iter_timelines()`` and the ``sessions`` table.
    """
    rows = []
    for path in paths:
        split = split_of(path)
        if split == MINIVAL_SPLIT:
            continue
        stamp = Path(path).stem.split("_ses-")[1]  # e.g. 19250101T111740
        rows.append({"path": path, "split": split, "day": stamp[:8], "time": stamp[9:]})
    rows.sort(key=lambda r: (r["day"], r["time"]))
    blocks: Dict[str, int] = {}
    for row in rows:
        blocks[row["day"]] = blocks.get(row["day"], 0) + 1
        row["block"] = blocks[row["day"]]
        row["session"] = f"{row['day']}_block{row['block']}"
    return rows


def list_assets() -> List[Dict]:
    """List H1 assets with ``path``, ``size`` (bytes) and ``split``.

    Returns
    -------
    list of dict
        One dict per asset: ``{"path", "asset_id", "size", "split"}`` where
        ``split`` is one of ``held-in-calib``/``held-in-minival``/
        ``held-out-calib``.
    """
    return [
        {**a, "split": split_of(a["path"])}
        for a in _nwb.dandi_list_assets(DANDISET_ID, VERSION)
    ]


# --- Accessors ------------------------------------------------------------
def get_spike_times(nwb: pynwb.NWBFile) -> List[np.ndarray]:
    """Per-channel spike-time arrays (seconds), one per Units-table row.

    Parameters
    ----------
    nwb : pynwb.NWBFile

    Returns
    -------
    list of numpy.ndarray
        ``len == n_channels``; each entry is a 1-D array of spike times (s).
    """
    units = nwb.units
    return [np.asarray(units["spike_times"][i]) for i in range(len(units.id))]


def bin_edges(n_bins: int) -> np.ndarray:
    """FALCON H1 spike-bin edges, ``n_bins + 1`` of them.

    Bin ``k`` ends at kinematic sample ``k`` (``k * 0.02`` s), as in
    ``falcon_challenge.dataloaders.bin_units``.
    """
    ends = np.arange(n_bins) * BIN_SIZE_S
    return np.concatenate(([ends[0] - BIN_SIZE_S], ends))


def bin_centers(nwb: pynwb.NWBFile) -> np.ndarray:
    """Centers (s) of the spike bins, ``(k - 0.5) * BIN_SIZE_S``, shape ``(n_bins,)``."""
    edges = bin_edges(nwb.acquisition["OpenLoopKinematics"].data.shape[0])
    return (edges[:-1] + edges[1:]) / 2


def bin_spike_times(spike_times: List[np.ndarray], n_bins: int) -> np.ndarray:
    """Spike counts on the kinematics grid, ``(n_channels, n_bins)`` float32."""
    edges = bin_edges(n_bins)
    counts = np.zeros((len(spike_times), n_bins), dtype=np.float32)
    for channel, times in enumerate(spike_times):
        counts[channel] = np.histogram(times, bins=edges)[0]
    return counts


def bin_spikes(nwb: pynwb.NWBFile) -> np.ndarray:
    """Spike counts on the kinematics grid, ``(n_bins, n_channels)``."""
    n_bins = nwb.acquisition["OpenLoopKinematics"].data.shape[0]
    return bin_spike_times(get_spike_times(nwb), n_bins).T


def get_kinematics(nwb: pynwb.NWBFile, velocity: bool = False) -> np.ndarray:
    """Return the 7-DoF kinematics (or velocity) matrix.

    Parameters
    ----------
    nwb : pynwb.NWBFile
    velocity : bool, optional
        If True return ``OpenLoopKinematicsVelocity`` instead of position.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_bins, 7)``; columns ordered as :data:`DOF_LABELS`.
    """
    key = "OpenLoopKinematicsVelocity" if velocity else "OpenLoopKinematics"
    return np.asarray(nwb.acquisition[key].data[:])


def get_trial_num(nwb: pynwb.NWBFile) -> np.ndarray:
    """Per-bin trial index, shape ``(n_bins,)``."""
    return np.asarray(nwb.acquisition["TrialNum"].data[:])


def phase_trial_ids(
    trial_num: np.ndarray,
    starts: Iterable[float],
    stops: Iterable[float],
) -> np.ndarray:
    """Trial id of each phase row, the ``TrialNum`` value over most of the row's bins.

    ``TrialNum`` steps to the next trial a few samples before a ``Release`` row ends, so
    the value at a row's start or end is not enough.
    """
    ids = []
    for start, stop in zip(starts, stops):
        first = int(round(start / BIN_SIZE_S))
        last = max(int(round(stop / BIN_SIZE_S)), first + 1)
        values, counts = np.unique(trial_num[first:last], return_counts=True)
        ids.append(int(values[np.argmax(counts)]))
    return np.array(ids, dtype=int)


def get_eval_mask(nwb: pynwb.NWBFile) -> np.ndarray:
    """FALCON per-bin boolean eval mask, shape ``(n_bins,)``.

    True marks the bins FALCON trains on and scores, as its loader reads it (the NWB
    description, "Timesteps to ignore covariates", states the opposite). False covers
    every ``SnapTo`` bin, the leading ``Intertrial`` rows and a few ``Orient`` bins.
    """
    return np.asarray(nwb.acquisition["eval_mask"].data[:]).astype(bool)


def get_epochs(nwb: pynwb.NWBFile):
    """The ``intervals/epochs`` table as a DataFrame, one row per task phase.

    Columns ``start_time``, ``stop_time`` and ``tags`` (one tag per row). Group phases
    into trials with :func:`phase_trial_ids`.
    """
    return nwb.intervals["epochs"].to_dataframe()
