"""Source-reading utilities for Fan et al. 2023 (CORP handwriting, participant T5).

Two deposits hold the 26 recording days:

* **Dryad** ``doi:10.5061/dryad.hqbzkh1p6``, ``CORP_data_release/``: one ``.mat`` per day and
  block kind for the 21 days up to 2023-02-28. ``seed_model_training_data/mat`` holds the 11
  open-loop seed days, ``online_evaluation_data/{recalibration,no_recalibration}/mat`` the
  two block kinds of the 10 closed-loop days. Each file has ``tx_feats`` (one
  ``(n_bins, 192)`` float32 array of 20 ms threshold crossings per trial, go period only),
  ``sentences``, ``blocks``, and on online days ``rnn_decoded`` and ``lm_decoded``.
* **DANDI** 000950 (FALCON H2): NWB files cut from the Dryad arrays. Only the 5 later days
  (``held-out-calib``, 3 trials each) are not on Dryad. Each NWB lays its trial clips end
  to end with a synthetic 10 s spacer, so trials are located by their timestamps.

Cues write a space as ``>`` and a period as ``~``.
"""

import functools
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

# --- Constants --------------------------------------------------------------
DRYAD_DOI = "10.5061/dryad.hqbzkh1p6"
DRYAD_ROOT = "CORP_data_release"
DANDISET_ID = "000950"
VERSION = "0.241029.1403"  # published version, DOI 10.48324/dandi.000950/0.241029.1403

SUBJECT = "T5"
BRAIN_AREA = "hand knob, motor cortex (left precentral gyrus)"
N_ARRAYS = 2
CHANNELS_PER_ARRAY = 96
N_CHANNELS = N_ARRAYS * CHANNELS_PER_ARRAY  # 192
BIN_SIZE_S = 0.02  # 20 ms bins
BIN_RATE_HZ = 1.0 / BIN_SIZE_S  # 50 Hz

# Fan et al. 2023, section 3.1 and supplement 2.4
THRESHOLD = "-4.5 x SD"
PROCESSING = (
    "0.3 Hz-7.5 kHz analog filter, 30 kHz sampling; non-causal 300 Hz highpass; "
    "-4.5 x SD thresholds; crossings counted per 20 ms bin; go period only"
)

CUE_SPACE = ">"
CUE_PERIOD = "~"

# block kind -> (folder under the Study path, loop type)
DRYAD_FOLDERS: dict[str, tuple[str, str]] = {
    "seed": (f"{DRYAD_ROOT}/seed_model_training_data/mat", "open"),
    "recalibration": (
        f"{DRYAD_ROOT}/online_evaluation_data/recalibration/mat",
        "closed",
    ),
    "no_recalibration": (
        f"{DRYAD_ROOT}/online_evaluation_data/no_recalibration/mat",
        "closed",
    ),
}
# FALCON's converter read the recalibration files (its default is a held-out date with
# ``online_recalibration_data``), and the released held-in recalibration trials equal them
HELD_OUT_FOLDER = f"{DANDISET_ID}/sub-T5-held-out-calib"
FALCON_EVAL_FRACTION = 0.4
# The Dryad tfrecords hold the last 10 trials of each seed day as test, online days all train
CORP_TEST_TRIALS = 10

# Days whose firing rates depart from every other day (audit of 2026-10-14). No source
# calls them bad, and Fan et al. 2023 decoded all three online.
INVALID_DAYS: dict[str, str] = {
    "2023-02-28": "a tenth of channels fire at 37x or more their held-in median rate (audit)",
    "2023-08-16": "a tenth of channels fire at 37x or more their held-in median rate (audit)",
    "2023-10-09": "firing rates about 12x the other days, channel profile uncorrelated with them (audit)",
}


def default_root() -> Path:
    """The Study's data folder, ``<NEURALSHED_DATA_ROOT>/fan_2023/data/Fan2023Handwriting``."""
    from neuralshed.datasets.utils import dataset_dir

    return dataset_dir("fan_2023") / "data" / "Fan2023Handwriting"


_DATE_RE = re.compile(r"(\d{4})[.-]?(\d{2})[.-]?(\d{2})")


def date_of(path: str | Path) -> str:
    """ISO date of a source file (``t5.2022.05.18.mat`` or ``...ses-20230417.nwb``)."""
    m = _DATE_RE.search(Path(path).name)
    if m is None:
        raise ValueError(f"no date in {path!r}")
    return f"{m[1]}-{m[2]}-{m[3]}"


def session_label(date: str) -> str:
    """The session id of a recording day, ``t5.YYYY.MM.DD`` as the Dryad files name it."""
    return "t5." + date.replace("-", ".")


def decode_cue(cue: str) -> str:
    """Translate a raw cue to plain text (``>`` -> space, ``~`` -> period)."""
    return cue.replace(CUE_SPACE, " ").replace(CUE_PERIOD, ".")


def falcon_split(trial_id: int, n_trials: int) -> str:
    """FALCON's label for a held-in trial: its converter kept the first 60% for calibration."""
    n_calib = n_trials - int(FALCON_EVAL_FRACTION * n_trials)
    return "held-in-calib" if trial_id < n_calib else "held-in-eval"


def corp_split(block_kind: str, trial_id: int, n_trials: int) -> str:
    """The authors' label for a Dryad trial, as their tfrecords partition it."""
    if block_kind == "seed" and trial_id >= n_trials - CORP_TEST_TRIALS:
        return "test"
    return "train"


def list_sources(root: Path | None = None) -> list[dict]:
    """Every file the Study reads, as ``{path, date, block_kind, loop_type}`` dicts.

    ``path`` is relative to ``root``. Raises if a source folder is missing.
    """
    root = default_root() if root is None else Path(root)
    rows = []
    for kind, (folder, loop) in DRYAD_FOLDERS.items():
        files = sorted((root / folder).glob("t5.*.mat"))
        if not files:
            raise FileNotFoundError(f"no Dryad files under {root / folder}")
        rows += [
            dict(
                path=str(p.relative_to(root)),
                date=date_of(p),
                block_kind=kind,
                loop_type=loop,
            )
            for p in files
        ]
    held_out = sorted(
        p for p in (root / HELD_OUT_FOLDER).glob("*.nwb") if not p.name.startswith("._")
    )
    if not held_out:
        raise FileNotFoundError(
            f"no DANDI held-out files under {root / HELD_OUT_FOLDER}"
        )
    rows += [
        dict(
            path=str(p.relative_to(root)),
            date=date_of(p),
            block_kind="recalibration",
            loop_type="closed",
        )
        for p in held_out
    ]
    return sorted(rows, key=lambda r: (r["date"], r["block_kind"]))


def count_trials(path: str | Path) -> int:
    """Number of trials in a source file, without reading the neural arrays."""
    path = Path(path)
    if path.suffix == ".mat":
        return len(sio.loadmat(str(path), variable_names=["blocks"])["blocks"][0])
    import h5py

    with h5py.File(path, "r") as h5:
        return len(h5["intervals/trials/id"])


# --- Dryad .mat -----------------------------------------------------------------
def _text(value) -> str:
    a = np.asarray(value).squeeze()
    return str(a) if a.size else ""


@functools.lru_cache(maxsize=2)
def read_mat(path: str | Path) -> dict:
    """One Dryad file: ``tx_feats`` (list of ``(n_bins, 192)``), ``sentences``, ``blocks``
    and, on online days, ``rnn_decoded`` / ``lm_decoded``. Cached, a day is parsed once."""
    m = sio.loadmat(str(path))
    out = {
        "tx_feats": [np.asarray(x, dtype=np.float32) for x in m["tx_feats"][0]],
        "sentences": [_text(s) for s in m["sentences"][0]],
        "blocks": [int(np.asarray(b).squeeze()) for b in m["blocks"][0]],
    }
    for key in ("rnn_decoded", "lm_decoded"):
        if key in m:
            out[key] = [_text(s) for s in m[key][0]]
    n = len(out["tx_feats"])
    if not all(len(v) == n for v in out.values()):
        raise ValueError(f"{path}: fields disagree on the number of trials")
    if any(x.ndim != 2 or x.shape[1] != N_CHANNELS for x in out["tx_feats"]):
        raise ValueError(f"{path}: a trial is not (n_bins, {N_CHANNELS})")
    return out


# --- DANDI NWB --------------------------------------------------------------------
def read_trials(h5) -> pd.DataFrame:
    """The ``intervals/trials`` table of an open ``h5py.File``, indexed by ``id``."""
    group = h5["intervals/trials"]
    return pd.DataFrame(
        {
            "start_time": group["start_time"][:],
            "stop_time": group["stop_time"][:],
            "cue": [
                c.decode() if isinstance(c, bytes) else str(c) for c in group["cue"][:]
            ],
            "block_num": group["block_num"][:].astype(int),
        },
        index=pd.Index(group["id"][:].astype(int), name="id"),
    )


def trial_bins(timestamps: np.ndarray, trials: pd.DataFrame) -> pd.DataFrame:
    """Each trial's ``[bin_start, bin_stop)`` rows in a FALCON NWB array.

    Raises unless the file is what the converter writes: consecutive clips that start and
    stop on a bin timestamp, cover every bin, and are evenly spaced at 20 ms inside.
    """
    i0 = np.searchsorted(timestamps, trials.start_time.to_numpy() - 1e-6)
    i1 = np.searchsorted(timestamps, trials.stop_time.to_numpy() + 1e-6)
    t = np.append(timestamps, np.nan)
    on_start = np.allclose(t[i0], trials.start_time, atol=1e-6)
    on_stop = np.allclose(t[i1 - 1], trials.stop_time, atol=1e-6)
    contiguous = (
        i0[0] == 0 and i1[-1] == len(timestamps) and np.array_equal(i0[1:], i1[:-1])
    )
    if not (on_start and on_stop and contiguous):
        raise ValueError("trials do not tile the stored bins one clip per trial")
    for a, b in zip(i0, i1):
        if not np.allclose(np.diff(timestamps[a:b]), BIN_SIZE_S):
            raise ValueError(f"bins {a}-{b} are not evenly spaced at {BIN_SIZE_S} s")
    return pd.DataFrame({"bin_start": i0, "bin_stop": i1}, index=trials.index)


def read_nwb(path: str | Path) -> dict:
    """A FALCON NWB file in the shape of :func:`read_mat` (``tx_feats``, ``sentences``, ``blocks``)."""
    import h5py

    with h5py.File(path, "r") as h5:
        timestamps = h5["acquisition/binned_spikes/timestamps"][:]
        data = h5["acquisition/binned_spikes/data"][:]
        trials = read_trials(h5)
    bins = trial_bins(timestamps, trials)
    return {
        "tx_feats": [data[a:b] for a, b in zip(bins.bin_start, bins.bin_stop)],
        "sentences": trials.cue.tolist(),
        "blocks": trials.block_num.tolist(),
    }


def read_source(root: Path, rel: str) -> dict:
    """The trials of one source file, ``.mat`` or ``.nwb``."""
    path = Path(root) / rel
    return read_mat(path) if path.suffix == ".mat" else _read_nwb_cached(path)


@functools.lru_cache(maxsize=2)
def _read_nwb_cached(path: Path) -> dict:
    return read_nwb(path)
