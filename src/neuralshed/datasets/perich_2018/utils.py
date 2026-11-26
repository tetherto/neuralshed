"""Readers for the Perich, Miller, Azabou and Dyer reaching release (DANDI 000688).

One NWB file per session, 111 in total, named
``sub-<X>/sub-<X>_ses-<CO|RT>-<YYYYMMDD>_behavior+ecephys.nwb`` for monkeys C, J, M and T
and the center-out (CO) or random-target (RT) task. Everything is read through ``h5py``
by key, so the same accessors work on a local file and on a streamed one.

Layout, identical across files except for the trials table:

* ``units``: sorted units with ragged ``spike_times`` (s) and an ``electrodes`` index. Units
  are stored in electrode order, M1 electrodes before PMd ones.
* ``general/extracellular_ephys/electrodes``: 96 rows (M1 only) or 192 (M1 then PMd), with
  ``group_name`` ``electrode_group_M1`` / ``electrode_group_PMd``.
* ``processing/behavior/{Position/cursor_pos, Velocity/cursor_vel, Acceleration/cursor_acc}``:
  ``(n, 2)`` in cm, cm/s and cm/s^2, x right and y up, origin at the screen centre, with
  explicit timestamps on a 10 ms grid.
* ``intervals/trials``: CO files carry ``target_on_time``, ``go_cue_time``, ``target_dir``;
  RT files carry ``go_cue_time_array`` ``(n, 4)`` and ``target_{1..4}_{x,y}_position``.

Timebase: the cursor timestamps are rounded to 10 ms and in two sessions (C 2016-09-23 and
2016-10-11) a pair of samples swaps a 0 ms step for a 20 ms one, but no sample sits more
than one bin off ``t0 + i * 0.01``. :func:`grid` therefore treats every behavior stream as
a uniform 100 Hz grid starting at the first timestamp, and :func:`bin_spikes` bins the
units onto the same grid.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

DANDISET_ID = "000688"
VERSION = "draft"

FS = 100.0
BIN_S = 0.01
POSITION_DIMS = ("x", "y")

#: CO targets: eight 2 cm squares on an 8 cm circle in every session (checked against the
#: centres of ``target_corners``). ``target_dir`` is 0 to the right, counter-clockwise, and
#: ``target_id`` 0 is the upward target, stepping clockwise by 45 degrees.
TARGET_RADIUS_CM = 8.0
TARGET_SIZE_CM = 2.0
N_CO_TARGETS = 8

TASKS = {"CO": "center_out", "RT": "random_target"}
SUBJECTS = ("C", "J", "M", "T")

#: Electrode groups in storage order, and the region label each stands for.
REGIONS = {"electrode_group_M1": "M1", "electrode_group_PMd": "PMd"}

BEHAVIOR_KEYS = {
    "cursor_pos": "processing/behavior/Position/cursor_pos",
    "cursor_vel": "processing/behavior/Velocity/cursor_vel",
    "cursor_acc": "processing/behavior/Acceleration/cursor_acc",
}

#: Monkey C sessions whose ``session_description`` ends in a truncated "Warning: This
#: session with Monkey" and whose cursor position and velocity sit at about 0.53 of the
#: target geometry (rewarded reaches end at a median radius of 4.2 cm against 8 cm targets).
HALF_SCALE_SESSIONS = frozenset(
    f"sub-C_ses-CO-{day}_behavior+ecephys"
    for day in (
        "20160909",
        "20160912",
        "20160914",
        "20160915",
        "20160919",
        "20160921",
        "20160923",
    )
)
HALF_SCALE_DESCRIPTION = (
    "cursor position and velocity at about half the scale of the targets; the source's "
    "session_description carries a truncated warning"
)

#: CO sessions whose ``target_corners`` row i+1 holds trial i's target. The Study never
#: reads ``target_corners``: target positions come from ``target_dir``, correct everywhere.
SHIFTED_CORNER_SESSIONS = frozenset(
    {
        "sub-C_ses-CO-20131203_behavior+ecephys",
        "sub-C_ses-CO-20131204_behavior+ecephys",
        "sub-C_ses-CO-20131219_behavior+ecephys",
        "sub-M_ses-CO-20140203_behavior+ecephys",
        "sub-M_ses-CO-20140218_behavior+ecephys",
        "sub-M_ses-CO-20140626_behavior+ecephys",
        "sub-M_ses-CO-20140627_behavior+ecephys",
        "sub-T_ses-CO-20130819_behavior+ecephys",
    }
)

#: RT target coordinates outside this bound are float garbage (about 1e33 to 1e37).
_MAX_TARGET_CM = 50.0

_NAME = re.compile(r"sub-(?P<subject>[A-Z])_ses-(?P<task>CO|RT)-(?P<date>\d{8})_")


@dataclass(frozen=True)
class SessionRef:
    """One NWB file, the unit this dataset treats as a timeline."""

    subject: str
    task_code: str
    date: str  # ISO
    path: str  # relative to the Study folder, e.g. "000688/sub-C/....nwb"

    @property
    def label(self) -> str:
        return session_label(self.path)

    @property
    def task(self) -> str:
        return TASKS[self.task_code]


def session_label(path: str | Path) -> str:
    """Session id: the NWB file stem, shared by ``metadata.sessions()`` and the Study."""
    return Path(path).stem


def parse_path(path: str | Path) -> SessionRef:
    match = _NAME.search(Path(path).name)
    if match is None:
        raise ValueError(f"not a perich_2018 session file name: {path}")
    day = match["date"]
    return SessionRef(
        subject=match["subject"],
        task_code=match["task"],
        date=f"{day[:4]}-{day[4:6]}-{day[6:]}",
        path=str(path),
    )


def list_sessions(root: Path) -> list[SessionRef]:
    """Local session files under ``root``, as paths relative to it."""
    root = Path(root)
    files = sorted(p for p in root.rglob("sub-*.nwb") if not p.name.startswith("._"))
    return [parse_path(str(p.relative_to(root))) for p in files]


# --- accessors on an open h5py.File --------------------------------------------------


def session_description(h5: h5py.File) -> str:
    return h5["session_description"][()].decode()


def grid(h5: h5py.File) -> tuple[float, int]:
    """``(t0, n)`` of the 100 Hz behavior grid, checked against the stored timestamps."""
    stamps = h5[BEHAVIOR_KEYS["cursor_pos"] + "/timestamps"][:]
    t0, n = float(stamps[0]), len(stamps)
    drift = np.abs(stamps - (t0 + np.arange(n) * BIN_S)).max()
    if drift > BIN_S + 1e-6:
        raise ValueError(f"cursor timestamps drift {drift:.3f} s off the 100 Hz grid")
    return t0, n


def get_behavior(h5: h5py.File, key: str) -> np.ndarray:
    """One behavior stream, ``(n, 2)`` time-first, in the source's unit."""
    return np.asarray(h5[BEHAVIOR_KEYS[key] + "/data"][:])


def unit_regions(h5: h5py.File) -> np.ndarray:
    """Region label (``M1`` or ``PMd``) of every unit, in storage order."""
    groups = h5["general/extracellular_ephys/electrodes/group_name"][:].astype(str)
    electrodes = h5["units/electrodes"][:]
    return np.array([REGIONS[g] for g in groups[electrodes]])


def unit_order(h5: h5py.File) -> np.ndarray:
    """Unit indices with M1 before PMd, stable within a region (the catalog's array order)."""
    rank = {region: i for i, region in enumerate(REGIONS.values())}
    regions = unit_regions(h5)
    return np.argsort([rank[r] for r in regions], kind="stable")


def get_spike_times(h5: h5py.File) -> list[np.ndarray]:
    """Per-unit spike times (s), in storage order."""
    times = h5["units/spike_times"][:]
    stops = h5["units/spike_times_index"][:].astype(np.int64)
    starts = np.r_[0, stops[:-1]]
    return [times[a:b] for a, b in zip(starts, stops)]


def bin_spikes(h5: h5py.File, t0: float, n: int) -> np.ndarray:
    """Spike counts on the behavior grid, ``(n_units, n)`` float32, M1 units first.

    Bin ``i`` counts spikes in ``[t0 + i * 0.01, t0 + (i + 1) * 0.01)``. Spikes before the
    first cursor sample or after the last bin (up to about 2 s in some sessions) are dropped.
    About 0.1% of spikes fall exactly on a bin edge, and the small offset puts them in the
    bin that starts there rather than leaving it to float rounding.
    """
    spikes = get_spike_times(h5)
    order = unit_order(h5)
    counts = np.zeros((len(spikes), n), dtype=np.float32)
    for row, unit in enumerate(order):
        index = np.floor((spikes[unit] - t0) / BIN_S + 1e-6).astype(np.int64)
        index = index[(index >= 0) & (index < n)]
        counts[row] = np.bincount(index, minlength=n)
    return counts


def _table(h5: h5py.File) -> h5py.Group:
    return h5["intervals/trials"]


def target_xy(direction: np.ndarray) -> np.ndarray:
    """CO target centre (cm) from ``target_dir`` (rad), NaN where the direction is."""
    return TARGET_RADIUS_CM * np.stack([np.cos(direction), np.sin(direction)], axis=-1)


def target_code(direction: np.ndarray) -> np.ndarray:
    """CO target index in the source's ``target_id`` convention, -100 where unknown.

    Derived from ``target_dir`` because ``target_id`` holds out-of-range values (8, 12) in
    a few trials.
    """
    code = np.full(len(direction), -100, dtype=np.int64)
    known = ~np.isnan(direction)
    steps = np.round((np.pi / 2 - direction[known]) / (np.pi / 4)).astype(np.int64)
    code[known] = np.mod(steps, N_CO_TARGETS)
    return code


def co_trials(h5: h5py.File) -> pd.DataFrame:
    """CO trials with target geometry derived from ``target_dir``."""
    table = _table(h5)
    direction = table["target_dir"][:]
    xy = target_xy(direction)
    trials = pd.DataFrame(
        {
            "trial_id": table["id"][:].astype(int),
            "start_time": table["start_time"][:],
            "stop_time": table["stop_time"][:],
            "target_on_time": table["target_on_time"][:],
            "go_cue_time": table["go_cue_time"][:],
            "target_dir": direction,
            "target_x": xy[:, 0],
            "target_y": xy[:, 1],
            "code": target_code(direction),
            "result": table["result"][:].astype(str),
        }
    )
    has_go = trials["go_cue_time"].notna()
    if (has_go & trials["target_on_time"].isna()).any():
        raise ValueError("CO trial with a go cue but no target onset")
    return trials


def rt_trials(h5: h5py.File) -> pd.DataFrame:
    """RT trials, with ``go_k`` and ``target_k_{x,y}`` columns for ``k`` in 0..3.

    Target coordinates beyond 50 cm are float garbage in the source and become NaN. So do
    targets at exactly (0, 0), a placeholder: successful reaches to them end a median
    5.5 cm away, against 0.5 cm for every other target (757 reaches, most in C 2015-03-16,
    C 2015-03-17 and T 2013-09-04).
    """
    table = _table(h5)
    go = table["go_cue_time_array"][:]
    trials = pd.DataFrame(
        {
            "trial_id": table["id"][:].astype(int),
            "start_time": table["start_time"][:],
            "stop_time": table["stop_time"][:],
            "result": table["result"][:].astype(str),
        }
    )
    for k in range(go.shape[1]):
        trials[f"go_{k}"] = go[:, k]
        xy = np.stack(
            [table[f"target_{k + 1}_{axis}_position"][:] for axis in POSITION_DIMS],
            axis=1,
        ).astype(float)
        xy[(np.abs(xy) > _MAX_TARGET_CM).any(axis=1)] = np.nan
        xy[(xy == 0).all(axis=1)] = np.nan
        for column, axis in enumerate(POSITION_DIMS):
            trials[f"target_{k}_{axis}"] = xy[:, column]
    return trials


__all__ = [
    "DANDISET_ID",
    "VERSION",
    "FS",
    "BIN_S",
    "POSITION_DIMS",
    "TARGET_RADIUS_CM",
    "TARGET_SIZE_CM",
    "N_CO_TARGETS",
    "TASKS",
    "SUBJECTS",
    "REGIONS",
    "BEHAVIOR_KEYS",
    "HALF_SCALE_SESSIONS",
    "HALF_SCALE_DESCRIPTION",
    "SHIFTED_CORNER_SESSIONS",
    "SessionRef",
    "session_label",
    "parse_path",
    "list_sessions",
    "session_description",
    "grid",
    "get_behavior",
    "unit_regions",
    "unit_order",
    "get_spike_times",
    "bin_spikes",
    "target_xy",
    "target_code",
    "co_trials",
    "rt_trials",
]
