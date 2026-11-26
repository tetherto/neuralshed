"""Source-reading utilities for the Deo et al. 2024 bimanual-cursor dataset.

Reads the raw Dryad source directly: one ``.mat`` file per experiment day
(``scipy.io.loadmat`` dict), each holding a whole-day concatenated stream of 20 ms-binned
threshold crossings plus the two cursors' positions, their targets, and per-trial timing.
Single BrainGate2 participant **T5** (C4 spinal cord injury, tetraplegia; two 96-channel
Utah arrays = 192 channels in hand-knob motor cortex, left precentral gyrus). The task is
delayed bimanual / unimanual control of two on-screen cursors.

On-disk layout (Dryad deposit ``doi:10.5061/dryad.sn02v6xbb``)::

    <NEURALSHED_DATA_ROOT>/deo_2024/
    └── data/
        └── Deo2024Bimanual/                    Study folder
            ├── readme.txt                      variable + per-block condition tables
            └── t5_MM_DD_YYYY.mat               one experiment day per file (13 files)

Per-day ``.mat`` fields (verified vs ``t5_06_28_2021.mat`` and siblings). Numeric dtypes below
are the saved array dtypes returned by the default ``scipy.io.loadmat``:

    tx                   uint8    (192, T)     threshold-crossing counts, 20 ms bins, -3.5xRMS
    Cp                   float64  (4, T)       binned cursor positions [Rx, Ry, Lx, Ly]
    Tp                   float64  (4, T)       binned target positions [Rx, Ry, Lx, Ly]
    blockNum             uint8    (T,)         block index per bin (recording gaps at block changes)
    goCue                int32    (R,)         per-trial GO index into the time axis (MATLAB 1-based)
    moveType             uint8    (R,)         1=UniR, 2=UniL, 3=Bi
    moveTypeName         cell     (3,)         ['UniR', 'UniL', 'Bi']
    durationBinnedDelay  uint8    (R,)         per-trial delay-period length (bins before GO)
    durationBinnedMove   uint8/16 (R,)         per-trial move-period length (bins after GO)
    Notes                char                  free-text description of the day

``tx`` has declared MATLAB class ``double``, but its saved count array is ``uint8``.
The Study converts those counts to ``float32`` without changing their values.
``durationBinnedMove`` is ``uint8`` on June 2 and June 4, and ``uint16`` otherwise.

``T`` is the number of 20 ms bins in the day and ``R`` the number of trials. The neural /
kinematic streams are dims-first (channels or cursor-dims x time); ``goCue`` is a **1-based**
MATLAB index into the time axis, so accessors convert it to a 0-based bin index. Open-loop
(OL) vs closed-loop (CL / CLR) is a per-block property that lives only in ``readme.txt``,
transcribed here into :data:`SESSION_BLOCKS`; the ``.mat`` itself does not carry it.
"""

from pathlib import Path

import numpy as np
import scipy.io

from neuralshed.datasets.utils import dataset_dir

# --- Constants -----------------------------------------------------------------
# The Study base appends its class-name folder to ``.../data``.
STUDY_FOLDER = "Deo2024Bimanual"


def default_data_dir() -> Path:
    """This dataset's data folder under ``NEURALSHED_DATA_ROOT``."""
    return dataset_dir("deo_2024") / "data" / STUDY_FOLDER


SUBJECT = "T5"

N_ARRAYS = 2  # two 96-channel Utah arrays
CHANNELS_PER_ARRAY = 96
N_CHANNELS = 192  # both arrays, concatenated
NEURAL_BIN_SIZE_MS = 20  # 20 ms non-overlapping bins
FS_NEURAL = 50  # Hz
THRESHOLD_RMS = -3.5  # threshold-crossing threshold, x per-electrode RMS
# durationBinnedMove of a closed-loop trial that ran to the 10 s timeout.
TIMEOUT_BINS = 499

# Cursor / target dimension order (rows of Cp and Tp), per readme.txt.
CURSOR_DIMS = ("right_x", "right_y", "left_x", "left_y")
# Per-hand cursor/target dims (one 2-D CursorPositionSeries per hand in study.py).
POSITION_DIMS = ("x", "y")

# moveType code -> short name (moveTypeName) -> human label.
MOVE_TYPE = {1: "UniR", 2: "UniL", 3: "Bi"}
MOVE_TYPE_LABEL = {
    "UniR": "unimanual right",
    "UniL": "unimanual left",
    "Bi": "bimanual",
}

# Per-block conditions, transcribed from the readme.txt "Experiment Block Conditions"
# tables (authoritative, not stored in the .mat). Every released block is listed.
#   open_loop          OL:  instructed-delay, cursors move autonomously to targets
#   closed_loop_recal  CLR: assisted closed-loop recalibration ("lock mode" + assist)
#   closed_loop        CL:  real-time neural control of both cursors (RNN or linear decoder)
_OL, _CLR, _CL = "open_loop", "closed_loop_recal", "closed_loop"
SESSION_BLOCKS: dict[str, dict[int, str]] = {
    "t5_06_02_2021": {b: _OL for b in range(12, 29)},
    "t5_06_04_2021": {b: _OL for b in range(2, 7)},
    "t5_06_23_2021": {b: _OL for b in range(6, 10)},
    "t5_06_28_2021": {
        **{b: _OL for b in range(5, 10)},
        **{b: _CLR for b in range(11, 15)},
        **{b: _CL for b in range(15, 21)},
    },
    "t5_06_30_2021": {
        **{b: _OL for b in range(9, 14)},
        **{b: _CLR for b in range(15, 19)},
        **{b: _CL for b in range(20, 28)},
    },
    "t5_07_12_2021": {
        **{b: _OL for b in range(8, 13)},
        **{b: _CLR for b in range(14, 18)},
        **{b: _CL for b in range(18, 22)},
    },
    "t5_07_14_2021": {
        **{b: _OL for b in (7, 8, 14, 15)},
        **{b: _CL for b in (16, 17, 18, 20, 21)},
    },
    "t5_09_13_2021": {
        **{b: _OL for b in range(1, 5)},
        **{b: _CLR for b in range(6, 10)},
        **{b: _CL for b in (12, 13, 14, 15, 16, 17, 18, 19, 20, 21)},
    },
    "t5_09_15_2021": {
        **{b: _OL for b in (1, 2, 3, 5)},
        **{b: _CLR for b in range(6, 10)},
        **{b: _CL for b in (16, 17, 18, 19, 20, 21, 22, 23)},
        **{b: _CL for b in (24, 25, 26, 27, 28, 29, 30, 31, 32)},
    },
    "t5_09_27_2021": {
        **{b: _OL for b in (1, 2, 3, 4)},
        **{b: _OL for b in range(12, 18)},
        **{b: _OL for b in range(22, 28)},
        **{b: _CL for b in (7, 8, 9, 10, 18, 19, 20, 21, 28, 29, 30, 31)},
    },
    "t5_09_29_2021": {
        **{b: _OL for b in range(1, 8)},
        **{b: _CL for b in (8, 9, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22)},
    },
    "t5_10_11_2021": {
        **{b: _OL for b in (1, 4, 6, 7, 8, 9, 19, 28)},
        **{b: _CL for b in (22, 23, 25, 29, 30, 31, 32, 33)},
    },
    "t5_10_13_2021": {
        **{b: _OL for b in (2, 4, 5, 6, 7, 8, 9, 27)},
        **{b: _CL for b in (20, 22, 23, 24, 25, 26, 28, 29, 31, 32)},
    },
}


DECODER_TYPES: dict[str, dict[int, str]] = {
    "t5_06_28_2021": {b: "recurrent_neural_network" for b in range(11, 21)},
    "t5_06_30_2021": {
        b: "recurrent_neural_network" for b in (*range(15, 19), *range(20, 28))
    },
    "t5_07_12_2021": {b: "recurrent_neural_network" for b in range(14, 22)},
    "t5_07_14_2021": {b: "recurrent_neural_network" for b in (16, 17, 18, 20, 21)},
    "t5_09_13_2021": {
        **{b: "linear" for b in (6, 8, 12, 14, 16, 18, 20, 21)},
        **{b: "recurrent_neural_network" for b in (7, 9, 13, 15, 17, 19)},
    },
    "t5_09_15_2021": {
        **{b: "linear" for b in (6, 7, 17, 19, 21, 23, 25, 27, 29, 31)},
        **{
            b: "recurrent_neural_network"
            for b in (8, 9, 16, 18, 20, 22, 24, 26, 28, 30, 32)
        },
    },
    "t5_09_27_2021": {
        b: "recurrent_neural_network"
        for b in (7, 8, 9, 10, 18, 19, 20, 21, 28, 29, 30, 31)
    },
    "t5_09_29_2021": {
        b: "recurrent_neural_network"
        for b in (8, 9, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22)
    },
    "t5_10_11_2021": {
        b: "recurrent_neural_network" for b in (22, 23, 25, 29, 30, 31, 32, 33)
    },
    "t5_10_13_2021": {
        b: "recurrent_neural_network" for b in (20, 22, 23, 24, 25, 26, 28, 29, 31, 32)
    },
}

DECODER_GAINS: dict[str, dict[int, float]] = {
    "t5_09_13_2021": {
        **dict(zip((12, 14, 16, 18, 20, 21), (0.5, 0.4, 0.3, 0.8, 1.0, 1.4))),
        **dict(zip((13, 15, 17, 19), (0.3, 0.4, 0.2, 0.5))),
    },
    "t5_09_15_2021": {
        **dict(zip((17, 19, 21, 23), (1.0, 0.8, 1.1, 0.9))),
        **dict(zip((16, 18, 20, 22), (0.5, 0.7, 0.4, 0.6))),
        **dict(zip((25, 27, 29, 31), (1.1, 0.9, 0.8, 1.2))),
        **dict(zip((24, 26, 28, 30, 32), (0.7, 0.5, 0.6, 0.4, 0.8))),
    },
}

DECODER_TRAINING_DATA: dict[str, dict[int, str]] = {
    "t5_09_27_2021": {
        **{b: "augmented" for b in (7, 9, 19, 21, 29, 31)},
        **{b: "raw" for b in (8, 10, 18, 20, 28, 30)},
    },
    "t5_09_29_2021": {
        **{b: "augmented" for b in (9, 13, 15, 17, 19, 21)},
        **{b: "raw" for b in (8, 12, 14, 16, 20, 22)},
    },
}

MOVEMENT_STRATEGIES: dict[str, dict[int, str]] = {
    "t5_10_11_2021": {
        **{b: "sequential_unimanual" for b in (22, 25, 30, 32)},
        **{b: "true_bimanual" for b in (23, 29, 31, 33)},
    },
    "t5_10_13_2021": {
        **{b: "sequential_unimanual" for b in (22, 24, 26, 29, 32)},
        **{b: "true_bimanual" for b in (20, 23, 25, 28, 31)},
    },
}

UNIMANUAL_ONLY_BLOCKS: dict[str, set[int]] = {
    "t5_09_13_2021": {1, 2, 3, 4, 6, 7, 8, 9},
    "t5_09_15_2021": {1, 2, 3, 5, 6, 7, 8, 9, *range(24, 33)},
    "t5_09_27_2021": {1, 2, 3, 4, 7, 8, 9, 10, *range(12, 32)},
    "t5_09_29_2021": {*range(1, 10), *range(12, 18), 19, 20, 21, 22},
    "t5_10_11_2021": {1, 4, 6, 8},
    "t5_10_13_2021": {2, 4, 6, 8},
}

# Each day's purpose, from the paper's Supplementary Table 2.
SESSION_DESCRIPTIONS: dict[str, str] = {
    "t5_06_02_2021": "Cued unimanual and bimanual movements, open loop only",
    "t5_06_04_2021": "Pilot open-loop data used to first calibrate the RNN decoders",
    "t5_06_23_2021": "Pilot open-loop data used to first calibrate the RNN decoders",
    "t5_06_28_2021": "Closed-loop control of both cursors with the RNN decoder",
    "t5_06_30_2021": "Closed-loop control of both cursors with the RNN decoder",
    "t5_07_12_2021": "Closed-loop control of both cursors with the RNN decoder",
    "t5_07_14_2021": "Closed-loop control of both cursors with the RNN decoder",
    "t5_09_13_2021": "Comparison of RNN and linear decoders",
    "t5_09_15_2021": (
        "Comparison of RNN and linear decoders, with a unimanual-only task variant"
    ),
    "t5_09_27_2021": (
        "Comparison of decoders trained on raw and on temporally augmented data"
    ),
    "t5_09_29_2021": (
        "Comparison of decoders trained on raw and on temporally augmented data"
    ),
    "t5_10_11_2021": (
        "Comparison of sequential unimanual and simultaneous bimanual strategies, "
        "each with its own RNN decoder"
    ),
    "t5_10_13_2021": (
        "Comparison of sequential unimanual and simultaneous bimanual strategies, "
        "each with its own RNN decoder"
    ),
}


# --- Session discovery ---------------------------------------------------------
def list_sessions(data_dir: str | Path | None = None) -> list[str]:
    """All session names (``t5_MM_DD_YYYY``), one per experiment-day ``.mat``, sorted by date."""
    data_dir = default_data_dir() if data_dir is None else data_dir
    mats = [p for p in Path(data_dir).glob("t5_*.mat") if not p.name.startswith("._")]
    return sorted((p.stem for p in mats), key=session_date)


def session_path(session: str, data_dir: str | Path | None = None) -> Path:
    """Path to a session's ``.mat`` file."""
    data_dir = default_data_dir() if data_dir is None else data_dir
    return Path(data_dir) / f"{session}.mat"


def session_date(session: str) -> str:
    """ISO date from a session name (``t5_06_02_2021`` -> ``2021-06-02``)."""
    _, mm, dd, yyyy = session.split("_")
    return f"{yyyy}-{mm}-{dd}"


def loop_type_of(session: str, block: int) -> str:
    """Control loop for a (session, block) from the readme tables."""
    return SESSION_BLOCKS[session][int(block)]


# --- Per-session readers -------------------------------------------------------
def read_session(path: str | Path, fields: list[str] | None = None) -> dict:
    """Load a session ``.mat`` (optionally only ``fields``), squeezed.

    Pass ``fields`` to skip the large ``tx`` / ``Cp`` / ``Tp`` arrays when only trial
    timing or labels are needed.
    """
    kw = {"squeeze_me": True, "struct_as_record": False}
    if fields is not None:
        kw["variable_names"] = list(fields)
    return scipy.io.loadmat(str(path), **kw)


def get_threshold_crossings(path: str | Path) -> np.ndarray:
    """Threshold-crossing counts as ``(n_bins, 192)``, time-first (uint8).

    Source ``tx`` is ``(192, T)`` channels-first; returned transposed to match the
    time-first neural convention. Raw counts, not z-scored.
    """
    tx = np.asarray(read_session(path, fields=["tx"])["tx"])
    return tx.T


def get_cursor_positions(path: str | Path) -> np.ndarray:
    """Binned cursor positions as ``(4, n_bins)`` dims-first, rows in :data:`CURSOR_DIMS` order."""
    return np.asarray(read_session(path, fields=["Cp"])["Cp"], dtype=np.float64)


def get_target_positions(path: str | Path) -> np.ndarray:
    """Binned target positions as ``(4, n_bins)`` dims-first, rows in :data:`CURSOR_DIMS` order."""
    return np.asarray(read_session(path, fields=["Tp"])["Tp"], dtype=np.float64)


def get_block_numbers(path: str | Path) -> np.ndarray:
    """Per-bin block index, shape ``(n_bins,)``. Value changes mark (unrecorded) gaps."""
    return np.asarray(read_session(path, fields=["blockNum"])["blockNum"]).ravel()


def block_bin_range(block_numbers: np.ndarray, block: int) -> tuple[int, int]:
    """Half-open bin range ``(start, stop)`` of a block within the day's stream.

    A block's bins are a contiguous run in the stored array, so
    this is ``(first_index, last_index + 1)`` of ``block_numbers == block``.
    """
    idx = np.nonzero(np.asarray(block_numbers) == block)[0]
    if idx.size == 0:
        raise ValueError(f"block {block} not present in the stream")
    return int(idx[0]), int(idx[-1]) + 1


def get_notes(path: str | Path) -> str:
    """The day's free-text ``Notes`` string."""
    return str(read_session(path, fields=["Notes"]).get("Notes", "")).strip()


# --- Trial table ---------------------------------------------------------------
def get_trials(path: str | Path):
    """Per-trial table (one row per cued movement) as a DataFrame.

    Columns: ``block``, ``loop_type`` (open_loop / closed_loop_recal / closed_loop),
    ``move_type`` (1/2/3), ``move_type_name`` (UniR/UniL/Bi), ``go_cue_bin`` (0-based),
    ``delay_bins``, ``move_bins``, ``delay_start_bin`` (GO - delay), ``move_end_bin``
    (GO + move). Bin indices are into the day's ``(n_bins,)`` time axis.
    """
    import pandas as pd

    session = Path(path).stem
    d = read_session(
        path,
        fields=[
            "goCue",
            "moveType",
            "durationBinnedDelay",
            "durationBinnedMove",
            "blockNum",
        ],
    )
    go_cue_bin = (
        np.atleast_1d(d["goCue"]).astype(np.int64) - 1
    )  # MATLAB 1-based -> 0-based
    move_type = np.atleast_1d(d["moveType"]).astype(np.int64)
    delay_bins = np.atleast_1d(d["durationBinnedDelay"]).astype(np.int64)
    move_bins = np.atleast_1d(d["durationBinnedMove"]).astype(np.int64)
    block_per_bin = np.asarray(d["blockNum"]).ravel()

    block = block_per_bin[go_cue_bin].astype(np.int64)
    return pd.DataFrame(
        {
            "block": block,
            "loop_type": [loop_type_of(session, b) for b in block],
            "move_type": move_type,
            "move_type_name": [MOVE_TYPE.get(int(m), "unknown") for m in move_type],
            "go_cue_bin": go_cue_bin,
            "delay_bins": delay_bins,
            "move_bins": move_bins,
            "delay_start_bin": go_cue_bin - delay_bins,
            "move_end_bin": go_cue_bin + move_bins,
        }
    )
