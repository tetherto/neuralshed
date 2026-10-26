"""Source-reading utilities for the Willett et al. 2023 speech neuroprosthesis dataset.

Reads the raw MATLAB ``.mat`` files distributed on Dryad
(doi:10.5061/dryad.x69p8czpq) for subject T12. The curated scope is the *sentence-production* data
(``sentences/*.mat``). Every function that touches the drive takes the study folder
``root`` (``source_root()`` on the data drive, ``Study.path`` inside the Study);
``list_sessions`` and ``session_summary`` also scan the out-of-scope directories.

All ``.mat`` files are loaded with ``scipy.io.loadmat(..., simplify_cells=True)``,
which yields plain numpy arrays / Python scalars (no MATLAB struct wrappers).
"""

import functools
from pathlib import Path

import numpy as np
import scipy.io as sio

from neuralshed.datasets.utils import dataset_dir
from neuralshed.text import PhonemeTable

STUDY_FOLDER = "Willett2023Sentences"


def source_root() -> Path:
    """Study folder on the data drive, ``<NEURALSHED_DATA_ROOT>/willet_2023/data/Willett2023Sentences``.

    The Study base appends the class-name subfolder to its ``path``, so this is also
    ``Willett2023Sentences(path=dataset_dir("willet_2023") / "data").path``.
    """
    return dataset_dir("willet_2023") / "data" / STUDY_FOLDER


# Neural sampling: all binned neural features use 20 ms bins (50 Hz).
BIN_SIZE_S = 0.02
SAMPLING_RATE_HZ = 50.0

PARTICIPANT_ID = "braingate2/T12"  # datasets/participants.py

# Blocks whose released neural data are known to be invalid, as
# {session: (blocks, reason)}. On 2022.06.14 the 128 area 44 channels (NSP2) collapse
# to near-zero power with sporadic 1e9 spikes from block 6 to the end of the session,
# while area 6v stays normal. The Study flags these timelines with valid=False.
INVALID_BLOCKS: dict[str, tuple[tuple[int, ...], str]] = {
    "t12.2022.06.14": ((6, 7, 8, 10, 11, 12, 13, 16, 18), "area 44 recordings invalid"),
}

# --------------------------------------------------------------------------- #
# Electrode array layout
# --------------------------------------------------------------------------- #
# Four 64-electrode arrays, 256 columns. "File order" is the column order of the
# source .mat matrices, "array order" the order the Study emits (one contiguous block
# of 64 per array). The 8x8 grids below are transcribed from the deposit readmes
# (row-major, superior row first, anterior column first) and hold the 0-based file
# column of each electrode. Only the two superior arrays are contiguous in file
# order; the two inferior ones are split around them.
ARRAY_GRIDS: dict[str, np.ndarray] = {
    "dorsal_6v": np.array(
        [
            [62, 51, 43, 35, 94, 87, 79, 78],
            [60, 53, 41, 33, 95, 86, 77, 76],
            [63, 54, 47, 44, 93, 84, 75, 74],
            [58, 55, 48, 40, 92, 85, 73, 72],
            [59, 45, 46, 38, 91, 82, 71, 70],
            [61, 49, 42, 36, 90, 83, 69, 68],
            [56, 52, 39, 34, 89, 81, 67, 66],
            [57, 50, 37, 32, 88, 80, 65, 64],
        ]
    ),
    "ventral_6v": np.array(
        [
            [125, 126, 112, 103, 31, 28, 11, 8],
            [123, 124, 110, 102, 29, 26, 9, 5],
            [121, 122, 109, 101, 27, 19, 18, 4],
            [119, 120, 108, 100, 25, 15, 12, 6],
            [117, 118, 107, 99, 23, 13, 10, 3],
            [115, 116, 106, 97, 21, 20, 7, 2],
            [113, 114, 105, 98, 17, 24, 14, 0],
            [127, 111, 104, 96, 30, 22, 16, 1],
        ]
    ),
    "dorsal_44": np.array(
        [
            [192, 193, 208, 216, 160, 165, 178, 185],
            [194, 195, 209, 217, 162, 167, 180, 184],
            [196, 197, 211, 218, 164, 170, 177, 189],
            [198, 199, 210, 219, 166, 174, 173, 187],
            [200, 201, 213, 220, 168, 176, 183, 186],
            [202, 203, 212, 221, 172, 175, 182, 191],
            [204, 205, 214, 223, 161, 169, 181, 188],
            [206, 207, 215, 222, 163, 171, 179, 190],
        ]
    ),
    "ventral_44": np.array(
        [
            [129, 144, 150, 158, 224, 232, 239, 255],
            [128, 142, 152, 145, 226, 233, 242, 241],
            [130, 135, 148, 149, 225, 234, 244, 243],
            [131, 138, 141, 151, 227, 235, 246, 245],
            [134, 140, 143, 153, 228, 236, 248, 247],
            [132, 146, 147, 155, 229, 237, 250, 249],
            [133, 137, 154, 157, 230, 238, 252, 251],
            [136, 139, 156, 159, 231, 240, 254, 253],
        ]
    ),
}

ARRAY_DESCRIPTIONS = {
    "dorsal_6v": "Area 6v (ventral precentral gyrus), superior array.",
    "ventral_6v": "Area 6v (ventral precentral gyrus), inferior array.",
    "dorsal_44": "Area 44 (inferior frontal gyrus, Broca's area), superior array.",
    "ventral_44": "Area 44 (inferior frontal gyrus, Broca's area), inferior array.",
}

# Order of the arrays in the Study's neural rows.
ARRAY_ORDER = tuple(ARRAY_GRIDS)

# File columns of each array, ascending.
ARRAY_CHANNELS: dict[str, np.ndarray] = {
    name: np.sort(grid.ravel()) for name, grid in ARRAY_GRIDS.items()
}

# File column of each array-order column: the arrays in ARRAY_ORDER, each with its
# file columns ascending, so every array is one contiguous block of 64 columns in
# array order (see STUDY_ARRAY_SLICES).
CHANNEL_ORDER = np.concatenate([ARRAY_CHANNELS[name] for name in ARRAY_ORDER])

STUDY_ARRAY_SLICES = {
    name: slice(64 * i, 64 * (i + 1)) for i, name in enumerate(ARRAY_ORDER)
}

assert all(grid.shape == (8, 8) for grid in ARRAY_GRIDS.values())
assert np.array_equal(np.sort(CHANNEL_ORDER), np.arange(256))


def to_array_order(x: np.ndarray) -> np.ndarray:
    """Convert a ``(T, 256)`` matrix from file order to array order."""
    return np.asarray(x)[:, CHANNEL_ORDER]


# --------------------------------------------------------------------------- #
# Neural signal types (all 256 channels, 20 ms bins)
# --------------------------------------------------------------------------- #
SIGNAL_TYPES = {
    "spikePow": "Spike band power -- mean of squared high-pass filtered voltages (250 Hz cutoff), uV^2.",
    "tx1": "Threshold crossings at -3.5xRMS (binned counts, 20 ms bins).",
    "tx2": "Threshold crossings at -4.5xRMS (binned counts, 20 ms bins).",
    "tx3": "Threshold crossings at -5.5xRMS (binned counts, 20 ms bins).",
    "tx4": "Threshold crossings at -6.5xRMS (binned counts, 20 ms bins).",
}


# --------------------------------------------------------------------------- #
# Session discovery
# --------------------------------------------------------------------------- #
def list_sessions(root: Path, data_type: str = "sentences") -> list[str]:
    """Sorted session ids of one source directory under the study folder ``root``.

    ``sentences`` and ``diagnosticBlocks`` give ``t12.YYYY.MM.DD`` (the ``_<data_type>``
    suffix stripped), ``tuningTasks`` the full file stems (each carries a task suffix),
    ``competitionData`` the union of stems across train/test/competitionHoldOut.
    """
    subdir = Path(root) / data_type
    if data_type == "competitionData":
        stems = {
            f.stem
            for split in ("train", "test", "competitionHoldOut")
            for f in (subdir / split).glob("*.mat")
            if not f.name.startswith("._")
        }
        return sorted(stems)
    stems = [f.stem for f in subdir.glob("*.mat") if not f.name.startswith("._")]
    if data_type == "tuningTasks":
        return sorted(stems)
    return sorted(stem.replace(f"_{data_type}", "") for stem in stems)


def session_date(session: str) -> str:
    """``t12.2022.04.28`` -> ``2022-04-28``."""
    _, y, m, d = session.split(".")
    return f"{y}-{m}-{d}"


def sentences_file(root: Path, session: str) -> Path:
    """Path of a session's ``sentences/*.mat`` under the study folder ``root``."""
    return Path(root) / "sentences" / f"{session}_sentences.mat"


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def load_sentences_session(root: Path, session: str) -> dict:
    """Every variable of a session's ``sentences/*.mat`` (about 700 MB).

    Continuous arrays (``spikePow``, ``tx1``-``tx4``, ``audioFeatures``) are ``(T, F)``
    over 20 ms bins, per-trial arrays (``sentences``, ``goTrialEpochs``, ...) have one
    entry per trial. ``load_variables`` reads a subset.
    """
    return sio.loadmat(str(sentences_file(root, session)), simplify_cells=True)


def load_variables(path: Path, names: list[str]) -> dict:
    """Load only ``names`` from a ``.mat`` file (skips the large neural arrays)."""
    return sio.loadmat(str(path), variable_names=names, simplify_cells=True)


# --------------------------------------------------------------------------- #
# Published competition partition
# --------------------------------------------------------------------------- #
# The paper's own split, as ``session -> (train blocks, test blocks)``, read once from
# ``competitionData/{train,test}/`` and transcribed here so the split is available without
# downloading that 6.5 GB directory (``_download`` fetches ``sentences`` only). These are
# the ``blockNum`` values of the sentences files, so they join straight onto the Sentence
# events.
#
# Per competitionData_readme.txt the last competition-eligible large-vocabulary block of
# a day is test and the rest are train, but fifty-word-set blocks and repeated switchboard
# blocks are in no partition, so a session's listed blocks do not cover all of its blocks
# and the test block is not always the literal last one. The third partition, ``competitionHoldOut``, is not represented:
# its block numbers appear in no sentences file, so nothing we load belongs to it.
COMPETITION_BLOCKS: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {
    "t12.2022.04.28": ((3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), (17,)),
    "t12.2022.05.05": (
        (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18),
        (19,),
    ),
    "t12.2022.05.17": (
        (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 23, 24),
        (25,),
    ),
    "t12.2022.05.19": ((5, 6, 7, 8, 9, 10, 11, 14, 15), (16,)),
    "t12.2022.05.24": ((4, 5, 7, 8, 9, 10, 11, 12, 13), (14,)),
    "t12.2022.05.26": ((5, 6, 7, 8, 9, 10, 11, 12, 13), (14,)),
    "t12.2022.06.02": ((4, 5, 6, 7, 8, 9, 10, 11, 12, 14), (15,)),
    "t12.2022.06.07": ((4, 5, 6, 7, 8, 9, 10, 11, 12), (13,)),
    "t12.2022.06.14": ((3, 4, 6, 7, 8, 10, 11, 12), (13,)),
    "t12.2022.06.16": ((8, 9, 10, 11, 12, 13, 14, 15), (18,)),
    "t12.2022.06.21": ((3, 4, 5, 6, 7, 8, 9, 10), (12,)),
    "t12.2022.06.23": ((5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 19), (20,)),
    "t12.2022.06.28": ((3, 4, 5, 6, 7, 9, 10, 11, 12), (14,)),
    "t12.2022.07.05": ((5, 6, 7, 8, 11, 13, 14, 15, 16), (17,)),
    "t12.2022.07.14": ((3, 4, 5, 6, 7, 9, 10, 11, 12, 13), (14,)),
    "t12.2022.07.21": ((3, 4, 5, 6, 10, 11, 12, 13, 14, 16), (17,)),
    "t12.2022.07.27": ((3, 4, 5, 6, 8, 9, 10, 11, 12, 15), (16,)),
    "t12.2022.07.29": ((3, 4, 5, 6, 7), (8,)),
    "t12.2022.08.02": ((3, 4, 5, 6, 8, 9, 11, 12, 13, 15), (16,)),
    "t12.2022.08.11": ((3, 4, 5, 6, 13, 14, 15, 17), (18,)),
    "t12.2022.08.13": ((3, 4, 5, 6, 8, 9, 10, 12), (13,)),
    "t12.2022.08.18": ((2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 16), (17,)),
    "t12.2022.08.23": ((1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 16), (17,)),
    "t12.2022.08.25": ((1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 16), (17,)),
}

COMPETITION_SPLITS = ("train", "test")


def competition_split_by_block(session: str) -> dict[int, str]:
    """``{block_num: "train" | "test"}`` for one session.

    Blocks the competition partition leaves out are simply absent from the mapping.
    """
    return {
        block: split
        for split, blocks in zip(
            COMPETITION_SPLITS, COMPETITION_BLOCKS.get(session, ((), ()))
        )
        for block in blocks
    }


# --------------------------------------------------------------------------- #
# Per-trial extraction
# --------------------------------------------------------------------------- #
def extract_trials(data: dict, data_type: str = "sentences") -> list[dict]:
    """Extract per-trial information from a loaded continuous-format session.

    Works for ``sentences``, ``diagnosticBlocks`` and ``tuningTasks`` (all of which
    share the ``goTrialEpochs`` / ``delayTrialEpochs`` layout). Not applicable to
    ``competitionData`` (already per-trial; use the loaded arrays directly).

    Parameters
    ----------
    data : dict
        Output of :func:`load_sentences_session`, or of :func:`load_variables` with the
        trial fields (``delayTrialEpochs``, ``goTrialEpochs``, ``trialDelayTimes``,
        ``sentences``, ``sentenceDurations``).
    data_type : {"sentences", "diagnosticBlocks", "tuningTasks"}
        Selects which type-specific fields are attached.

    Returns
    -------
    list of dict
        One dict per trial with keys:

        - ``delay_start``, ``delay_end`` : delay-period bins, 0-based with an exclusive
          end (the source's 1-based inclusive epochs converted), so
          ``data[delay_start:delay_end]`` is the delay period.
        - ``go_start``, ``go_end`` : go-period (production) bins, same convention.
        - ``delay_time`` : nominal delay-period duration in seconds.
        - ``sentence``, ``duration_bins`` : sentence text and go-period length in
          bins (``sentences`` only).
        - ``cue_id``, ``cue`` : 1-based cue index and its text from ``cueList``
          (``diagnosticBlocks`` / ``tuningTasks`` only).
    """
    go = np.asarray(data["goTrialEpochs"])
    delay = np.asarray(data["delayTrialEpochs"])
    delay_times = np.asarray(data["trialDelayTimes"])
    n_trials = go.shape[0]

    trials = []
    for i in range(n_trials):
        trial = {
            "delay_start": int(delay[i, 0]) - 1,
            "delay_end": int(delay[i, 1]),
            "go_start": int(go[i, 0]) - 1,
            "go_end": int(go[i, 1]),
            "delay_time": float(delay_times[i]),
        }
        if data_type == "sentences":
            trial["sentence"] = str(data["sentences"][i])
            trial["duration_bins"] = int(data["sentenceDurations"][i])
        elif data_type in ("diagnosticBlocks", "tuningTasks"):
            cue_idx = int(np.asarray(data["trialCues"])[i])  # MATLAB 1-based
            trial["cue_id"] = cue_idx
            trial["cue"] = str(data["cueList"][cue_idx - 1])
        trials.append(trial)
    return trials


def get_array_signal(data: dict, signal: str, array: str) -> np.ndarray:
    """Select one electrode array from one neural signal of a loaded session.

    Parameters
    ----------
    data : dict
        A loaded continuous-format session (sentences / diagnostic / tuning).
    signal : str
        A key in :data:`SIGNAL_TYPES` (``"spikePow"``, ``"tx1"`` .. ``"tx4"``).
    array : str
        An array name from :data:`ARRAY_GRIDS` (e.g. ``"dorsal_6v"``).

    Returns
    -------
    numpy.ndarray
        ``(T, 64)`` columns of that array, in ascending file column order.
    """
    if signal not in SIGNAL_TYPES:
        raise ValueError(
            f"Unknown signal {signal!r}; expected one of {list(SIGNAL_TYPES)}"
        )
    if array not in ARRAY_CHANNELS:
        raise ValueError(f"Unknown array {array!r}; expected one of {ARRAY_ORDER}")
    return np.asarray(data[signal])[:, ARRAY_CHANNELS[array]]


# --------------------------------------------------------------------------- #
# Summaries
# --------------------------------------------------------------------------- #
def _session_file(root: Path, session: str, data_type: str) -> Path:
    """Path to a session's ``.mat`` for a continuous-format ``data_type``."""
    if data_type == "tuningTasks":  # session already carries its task suffix
        return Path(root) / data_type / f"{session}.mat"
    return Path(root) / data_type / f"{session}_{data_type}.mat"


def session_summary(root: Path, data_type: str = "sentences") -> list[dict]:
    """Build a summary table over all sessions of a data type under ``root``.

    Parameters
    ----------
    data_type : {"sentences", "diagnosticBlocks", "tuningTasks"}
        Which continuous-format directory to summarize.

    Returns
    -------
    list of dict
        One row per session: ``session``, ``n_trials``, ``n_timesteps``,
        ``duration_min`` (recording length in minutes), ``n_blocks`` and, for
        sentences, ``speaking_mode``. Suitable for ``pandas.DataFrame(...)``.
    """
    if data_type not in ("sentences", "diagnosticBlocks", "tuningTasks"):
        raise ValueError(f"session_summary does not support data_type={data_type!r}")

    rows = []
    for session in list_sessions(root, data_type):
        path = _session_file(root, session, data_type)
        # Shapes only (whosmat) -- avoids loading the 700 MB+ neural/audio arrays.
        shapes = {name: shp for name, shp, _dt in sio.whosmat(str(path))}
        n_timesteps = int(shapes["spikePow"][0])
        row = {
            "session": session,
            "n_trials": int(shapes["goTrialEpochs"][0]),
            "n_timesteps": n_timesteps,
            "duration_min": round(n_timesteps * BIN_SIZE_S / 60.0, 1),
            "n_blocks": int(shapes["blockList"][0]),
        }
        if data_type == "sentences":
            sm = sio.loadmat(
                str(path), variable_names=["speakingMode"], simplify_cells=True
            )
            row["speaking_mode"] = str(sm["speakingMode"])
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Block-level access (used by the Study, which treats one block as one timeline)
# --------------------------------------------------------------------------- #
def list_blocks(data: dict) -> list[tuple[int, str]]:
    """``(block, block_type)`` pairs from a session's ``blockList`` / ``blockTypes``."""
    blocks = np.atleast_1d(np.asarray(data["blockList"])).ravel()
    types = np.atleast_1d(np.asarray(data["blockTypes"], dtype=object)).ravel()
    return [(int(b), str(t)) for b, t in zip(blocks, types)]


def block_bins(block_num: np.ndarray, block: int) -> tuple[int, int]:
    """0-based half-open bin range ``[start, stop)`` of ``block`` in the session grid."""
    idx = np.flatnonzero(np.asarray(block_num).ravel() == block)
    if idx.size == 0:
        raise ValueError(f"block {block} has no bins")
    start, stop = int(idx[0]), int(idx[-1]) + 1
    if stop - start != idx.size:
        raise ValueError(f"block {block} is not contiguous in blockNum")
    return start, stop


@functools.lru_cache(maxsize=2)
def read_stream(path: str, key: str) -> np.ndarray:
    """One neural stream ``(T, 256)`` of a session file, as stored.

    Cached so that reading a session's blocks one after the other loads the 300 MB
    matrix once rather than once per block.
    """
    return np.asarray(sio.loadmat(path, variable_names=[key], simplify_cells=True)[key])


# assets/cue_phonemes.json: cue text -> ARPAbet target, the cue's dictionary
# pronunciation (text.g2p_phonemes) over every sentence cue of the release. The
# transcripts write some contractions with a space before the apostrophe ("I 'll");
# the table keeps them split (``join_split_apostrophes=False``), so the detached
# piece is read as a word of its own.
CUE_PHONEMES = PhonemeTable(
    Path(__file__).resolve().parent / "assets" / "cue_phonemes.json"
)


def build_cue_phonemes(
    root: Path | str | None = None, path: str | Path | None = None
) -> PhonemeTable:
    """Regenerate ``assets/cue_phonemes.json`` (or ``path``) over every sentence cue of
    the 24 ``sentences/*.mat`` files under the study folder ``root`` (default
    :func:`source_root`), stripped as the Study emits them. Needs the ``g2p`` extra."""
    root = Path(root) if root is not None else source_root()
    cues: set[str] = set()
    for session in list_sessions(root):
        texts = load_variables(sentences_file(root, session), ["sentences"])["sentences"]
        cues.update(t for t in (str(s).strip() for s in np.atleast_1d(texts)) if t)
    return PhonemeTable.build(
        cues,
        path or CUE_PHONEMES.path,
        description="Dictionary pronunciation of each sentence cue, not a measurement "
        "of what T12 produced. g2p_en over the cue with punctuation stripped, stress "
        "removed, '|' at every word boundary and at the end; contractions the "
        "transcript writes with a space before the apostrophe (\"I 'll\") stay split "
        "(join_split_apostrophes=false). Symbols are PhonemeVocab v1 "
        "(neuralshed.text.ARPABET_V1).",
        join_split_apostrophes=False,
    )
