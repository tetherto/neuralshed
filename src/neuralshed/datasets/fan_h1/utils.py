"""Streaming NWB source-reading utilities for FALCON H1 (DANDI 000954).

NWB files are read in place over the network with ``remfile`` + ``h5py`` + ``pynwb``
no bulk download, no ``simplecache``.
``remfile`` performs lazy byte-range reads with a small chunk
cache, so only the bytes actually touched cross the wire.

The H1 NWB layout:

* Neural data: a ``Units`` table (``nwb.units``) of multi-unit **spike times**
  (one ragged ``spike_times`` row per unit; 176 units = 2 x 96-ch Utah arrays,
  hand-knob motor cortex). Spikes are NOT pre-binned -- bin them yourself at the
  20 ms / 50 Hz cadence used by the kinematics.
* Kinematics: ``nwb.acquisition["OpenLoopKinematics"]`` -- a ``TimeSeries`` of
  shape ``(n_bins, 7)``; the 7 DoF labels are in ``.description``
  (``tx,ty,tz,rx,g1,g2,g3``). Velocity in ``OpenLoopKinematicsVelocity``.
* Trial id per bin: ``nwb.acquisition["TrialNum"]`` shape ``(n_bins,)``.
* FALCON eval mask: ``nwb.acquisition["eval_mask"]`` -- bool ``(n_bins,)``;
  True marks bins scored by the FALCON benchmark.
* Intervals: ``nwb.intervals["epochs"]`` -- per-trial start/stop times.

Timebase quirk: the kinematics ``TimeSeries`` stores ``rate = 0.02``. That value
is the **bin period in seconds** (20 ms -> 50 Hz), not a sample rate in Hz; the
total duration ``n_bins * 0.02`` matches the epochs span. Use
``bin_centers()`` / :data:`BIN_SIZE_S` rather than the raw ``.rate`` field.
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pynwb

from neuralshed.datasets import nwb as _nwb

# --- Constants (H1) -------------------------------------------------------
DANDISET_ID = "000954"
VERSION = "draft"
ARRAYS = "2 x 96-channel Utah arrays"
BRAIN_AREA = "hand-knob motor cortex (precentral gyrus)"
CHANNEL_COUNT = 192
N_UNITS = 176  # multi-units, consistent across the released files
BIN_SIZE_S = 0.02  # 20 ms bins -> 50 Hz (stored, misleadingly, in TimeSeries.rate)
BIN_RATE_HZ = 50.0
DOF_LABELS = ("tx", "ty", "tz", "rx", "g1", "g2", "g3")  # 3 translation, 1 rot, 3 grasp
N_DOF = 7
_SPLITS = ("held-in-calib", "held-in-minival", "held-out-calib")


def split_of(path: str) -> str:
    """The FALCON split (held-in/out) embedded in an asset path."""
    for s in _SPLITS:
        if s in path:
            return s
    return "unknown"


def session_label(path: str) -> str:
    """Canonical session id: the NWB filename stem (one NWB file = one session).

    The join key between ``study.iter_timelines()`` (``timeline["session"]``) and the
    ``sessions`` table (``label``).
    """
    return Path(path).stem


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
    """Return per-unit spike-time arrays (seconds) from the Units table.

    Parameters
    ----------
    nwb : pynwb.NWBFile

    Returns
    -------
    list of numpy.ndarray
        ``len == n_units``; each entry is a 1-D array of spike times (s).
    """
    units = nwb.units
    return [np.asarray(units["spike_times"][i]) for i in range(len(units.id))]


def bin_centers(nwb: pynwb.NWBFile) -> np.ndarray:
    """Bin-center times (s) for the kinematics / mask timebase.

    Parameters
    ----------
    nwb : pynwb.NWBFile

    Returns
    -------
    numpy.ndarray
        Shape ``(n_bins,)``; ``starting_time + (i + 0.5) * BIN_SIZE_S``.
    """
    ts = nwb.acquisition["OpenLoopKinematics"]
    n = ts.data.shape[0]
    t0 = ts.starting_time or 0.0
    return t0 + (np.arange(n) + 0.5) * BIN_SIZE_S


def bin_spikes(nwb: pynwb.NWBFile) -> np.ndarray:
    """Bin spike times into the kinematics 20 ms timebase.

    Parameters
    ----------
    nwb : pynwb.NWBFile

    Returns
    -------
    numpy.ndarray
        Spike-count matrix of shape ``(n_bins, n_units)`` (int).
    """
    n_bins = nwb.acquisition["OpenLoopKinematics"].data.shape[0]
    t0 = nwb.acquisition["OpenLoopKinematics"].starting_time or 0.0
    edges = t0 + np.arange(n_bins + 1) * BIN_SIZE_S
    spikes = get_spike_times(nwb)
    counts = np.zeros((n_bins, len(spikes)), dtype=np.int32)
    for j, st in enumerate(spikes):
        counts[:, j] = np.histogram(st, bins=edges)[0]
    return counts


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


def get_eval_mask(nwb: pynwb.NWBFile) -> np.ndarray:
    """FALCON per-bin boolean eval mask, shape ``(n_bins,)``.

    True marks bins included in FALCON scoring (transients / pre-trial bins are
    masked out as False).
    """
    return np.asarray(nwb.acquisition["eval_mask"].data[:]).astype(bool)


def get_trials(nwb: pynwb.NWBFile):
    """Per-trial interval table (``intervals['epochs']``) as a DataFrame.

    Parameters
    ----------
    nwb : pynwb.NWBFile

    Returns
    -------
    pandas.DataFrame or None
        Columns include ``start_time``/``stop_time``. None if absent.
    """
    if nwb.intervals and "epochs" in nwb.intervals:
        return nwb.intervals["epochs"].to_dataframe()
    return None
