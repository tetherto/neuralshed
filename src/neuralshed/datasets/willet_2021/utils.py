"""Readers for the Willett et al. 2021 handwriting BCI ``.mat`` files on Dryad.

Binned threshold-crossing counts (10 ms bins, 192 channels from two 96-channel Utah
arrays in the hand knob of the left precentral gyrus) recorded while BrainGate2
participant T5 attempted to handwrite single characters, straight lines and sentences.

Every reader takes ``root``, the ``handwritingBCIData`` folder, and falls back to
:func:`source_root` on the data drive.

References
----------
- Willett, F.R. et al. (2021). High-performance brain-to-text communication via
  handwriting. Nature 593, 249-254. https://doi.org/10.1038/s41586-021-03506-2
- Dryad data: https://doi.org/10.5061/dryad.wh70rxwmv
- Code repo: https://github.com/fwillett/handwritingBCI
"""

import functools
from pathlib import Path

import numpy as np
import scipy.io as sio

from neuralshed.datasets import participants
from neuralshed.datasets.utils import dataset_dir

# The Study base appends the class-name subfolder to its ``path``.
STUDY_FOLDER = "Willett2021Handwriting"


def source_root() -> Path:
    """``handwritingBCIData`` on the data drive, the default ``root`` of every reader."""
    return dataset_dir("willet_2021") / "data" / STUDY_FOLDER / "handwritingBCIData"


N_CHANNELS = 192

# Two 96-channel Utah microelectrode arrays in the hand-knob area of the left
# precentral gyrus. Per the dataset readme: the first 96 channels (0-95) are
# array 1 (the more lateral array), the last 96 channels (96-191) are array 2
# (the more medial array). Electrodes are spaced 0.4 mm apart.
ARRAY_GROUPS = [
    (
        "lateral_array",
        "Array 1 (lateral), hand-knob area of left precentral gyrus, channels 0-95.",
        slice(0, 96),
    ),
    (
        "medial_array",
        "Array 2 (medial), hand-knob area of left precentral gyrus, channels 96-191.",
        slice(96, 192),
    ),
]

# ``arrayGeometryMap``, identical in all 21 files: the electrode number (1-96, 0 where
# there is none) at each position of the 10 x 10 grid, 0.4 mm pitch. Rows move toward
# the central sulcus, columns move medially (deposit readme). The deposit gives one map
# for both arrays and does not state how electrode numbers map to channel columns.
ARRAY_GEOMETRY_MAP = np.array(
    [
        [0, 79, 77, 75, 73, 71, 69, 67, 65, 0],
        [81, 80, 78, 76, 74, 72, 70, 68, 66, 2],
        [83, 84, 82, 45, 41, 39, 37, 35, 33, 1],
        [85, 86, 49, 47, 43, 40, 38, 36, 34, 3],
        [88, 87, 53, 51, 44, 42, 48, 5, 7, 4],
        [90, 89, 55, 56, 46, 50, 15, 17, 9, 6],
        [92, 91, 57, 58, 52, 54, 19, 13, 11, 8],
        [93, 94, 59, 60, 62, 21, 25, 23, 12, 10],
        [96, 63, 61, 64, 31, 29, 27, 20, 16, 14],
        [0, 95, 32, 30, 28, 26, 24, 22, 18, 0],
    ]
)

PARTICIPANT_ID = "braingate2/T5"  # datasets/participants.py

# Session purposes, from the deposit readme's session table.
SESSION_DESCRIPTIONS = {
    "t5.2019.05.08": "Sentence and character writing pilot day (no real-time decoding)",
    "t5.2019.06.03": "Attempted handwriting of straight lines",
    "t5.2019.11.25": "Real-time decoding pilot day",
    "t5.2019.12.09": "Real-time decoding pilot day",
    "t5.2019.12.11": "Copy-typing evaluation",
    "t5.2019.12.18": "Copy-typing evaluation",
    "t5.2019.12.20": "Copy-typing evaluation",
    "t5.2020.01.06": "Copy-typing evaluation",
    "t5.2020.01.08": "Copy-typing evaluation",
    "t5.2020.01.13": "Free-answer evaluation",
    "t5.2020.01.15": "Free-answer evaluation",
}

# The 31 character classes used in the handwriting task. These match the suffix
# names of the ``neuralActivityCube_{x}`` variables in singleLetters.mat.
CHAR_LIST = [
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
    "greaterThan",
    "comma",
    "apostrophe",
    "tilde",
    "questionMark",
]

# The source text strings (``sentencePrompt``, ``intendedText``) write space as ``>``
# and period as ``~``.
_TEXT_ABBREVIATIONS = {">": " ", "~": "."}

# Raw character class (``characterCues``, ``neuralActivityCube_{x}`` suffix) -> glyph.
CHAR_CLASS_TO_GLYPH = {c: c for c in CHAR_LIST[:26]} | {
    "greaterThan": " ",
    "comma": ",",
    "apostrophe": "'",
    "tilde": ".",
    "questionMark": "?",
}

BIN_SIZE_S = 0.01  # 10 ms bins
RATE_HZ = 100.0  # 1 / BIN_SIZE_S

# Cube trials are aligned so that 0-indexed time step 50 is the go cue.
CUBE_GO_BIN = 50

TASK_TYPES = ("sentences", "singleLetters", "straightLines")

REST_CUE = "doNothing"  # rest condition, pilot session t5.2019.05.08 only
LINE_DISTANCES = ("short", "medium", "long")
LINE_CONDITIONS = tuple(
    f"straightLine_{d}_{a}" for d in LINE_DISTANCES for a in range(1, 17)
)


def condition_code(cue: str) -> int:
    """Dataset-wide class id of a cue: rest 0, characters 1-31 in :data:`CHAR_LIST`
    order, lines 32-79 as ``32 + 16 * distance + angle - 1`` (distance 0, 1, 2 for
    short, medium, long, angle 1-16).
    """
    if cue == REST_CUE:
        return 0
    if cue in CHAR_LIST:
        return CHAR_LIST.index(cue) + 1
    return 32 + LINE_CONDITIONS.index(cue)


def decode_text(text: str) -> str:
    """Source text with ``>`` as space and ``~`` as period."""
    for abbr, glyph in _TEXT_ABBREVIATIONS.items():
        text = text.replace(abbr, glyph)
    return text


def session_date(session: str) -> str:
    """ISO date of a session label, ``"t5.2019.05.08"`` -> ``"2019-05-08"``."""
    _, year, month, day = session.split(".")
    return f"{year}-{month}-{day}"


def post_implant_day(session: str) -> int:
    """Days from the registry implant date to the session."""
    return participants.post_implant_day(PARTICIPANT_ID, session_date(session))


# Variables read for trial rows, per task. Everything else (the neural matrix, the
# cubes, the real-time ``rnn_*`` outputs) stays on disk.
TRIAL_VARIABLES = {
    "singleLetters": [
        "goPeriodOnsetTimeBin",
        "delayPeriodOnsetTimeBin",
        "characterCues",
        "blockNumsTimeSeries",
    ],
    "straightLines": [
        "goPeriodOnsetTimeBin",
        "delayPeriodOnsetTimeBin",
        "movementCues",
        "blockNumsTimeSeries",
    ],
    "sentences": [
        "goCueOnsetTimeBin",
        "delayCueOnsetTimeBin",
        "sentenceEndTimeBin",
        "numTimeBinsPerSentence",
        "sentencePrompt",
        "intendedText",
        "sentenceCondition",
        "sentenceBlockNums",
        "excludedSentences",
        "blockNumsTimeSeries",
    ],
}


# ---------------------------------------------------------------------------
# Listing and loading source files
# ---------------------------------------------------------------------------


def _root(root: Path | str | None) -> Path:
    return Path(root) if root is not None else source_root()


def list_sessions(root: Path | str | None = None) -> list[str]:
    """Sorted session folder names, e.g. ``["t5.2019.05.08", ...]``."""
    datasets = _root(root) / "Datasets"
    return sorted(d.name for d in datasets.iterdir() if d.is_dir())


def list_task_files(session: str, root: Path | str | None = None) -> dict[str, Path]:
    """``{task: path}`` for the task files present in one session."""
    session_dir = _root(root) / "Datasets" / session
    paths = {task: session_dir / f"{task}.mat" for task in TASK_TYPES}
    return {task: path for task, path in paths.items() if path.exists()}


def n_time_bins(path: Path | str) -> int:
    """Number of bins in a task file's ``neuralActivityTimeSeries``, read from the header."""
    for name, shape, _ in sio.whosmat(str(path)):
        if name == "neuralActivityTimeSeries":
            if shape[1] != N_CHANNELS:
                raise ValueError(f"{path}: {shape[1]} channels, expected {N_CHANNELS}")
            return int(shape[0])
    raise KeyError(f"neuralActivityTimeSeries not in {path}")


def load_variables(path: Path | str, names: list[str]) -> dict:
    """Load only ``names`` from a ``.mat`` file. Raises if any is missing."""
    data = sio.loadmat(str(path), variable_names=names, simplify_cells=True)
    missing = set(names) - set(data)
    if missing:
        raise KeyError(f"{path}: missing {sorted(missing)}")
    return data


def load_trial_variables(path: Path | str, task: str) -> dict:
    """The :data:`TRIAL_VARIABLES` of one task file."""
    return load_variables(path, TRIAL_VARIABLES[task])


def load_single_letters(session: str, root: Path | str | None = None) -> dict:
    """Every variable of a session's ``singleLetters.mat``, cubes included."""
    path = _root(root) / "Datasets" / session / "singleLetters.mat"
    return sio.loadmat(str(path), simplify_cells=True)


def load_sentences(session: str, root: Path | str | None = None) -> dict:
    """Every variable of a session's ``sentences.mat``, ``rnn_*`` outputs included."""
    path = _root(root) / "Datasets" / session / "sentences.mat"
    return sio.loadmat(str(path), simplify_cells=True)


@functools.lru_cache(maxsize=2)
def read_neural(path: str) -> np.ndarray:
    """A task file's ``neuralActivityTimeSeries`` ``(T, 192)`` uint8, as stored.

    Cached so that reading a file's blocks one after the other loads it once.
    """
    key = "neuralActivityTimeSeries"
    return np.asarray(load_variables(path, [key])[key])


def block_bins(block_nums: np.ndarray, block: int) -> tuple[int, int]:
    """0-based half-open bin range ``[start, stop)`` of ``block`` in a file's grid."""
    idx = np.flatnonzero(np.asarray(block_nums).ravel() == block)
    if idx.size == 0:
        raise ValueError(f"block {block} has no bins")
    start, stop = int(idx[0]), int(idx[-1]) + 1
    if stop - start != idx.size:
        raise ValueError(f"block {block} is not contiguous in blockNumsTimeSeries")
    return start, stop


# ---------------------------------------------------------------------------
# Extracting per-trial metadata
# ---------------------------------------------------------------------------


def _text(value) -> str:
    """A cell string, with an empty cell (an empty array) as ``""``."""
    if isinstance(value, np.ndarray):
        if value.size:
            raise ValueError(f"expected a string, got an array {value!r}")
        return ""
    return str(value)


def _same_length(**arrays: np.ndarray) -> int:
    lengths = {name: len(a) for name, a in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"per-trial arrays differ in length: {lengths}")
    return next(iter(lengths.values()))


def extract_single_letter_trials(data: dict) -> list[dict]:
    """Per-trial ``delay_onset_bin``, ``go_onset_bin`` (1-based), ``cue``, ``block_num``.

    Works for ``singleLetters`` (``characterCues``) and ``straightLines``
    (``movementCues``).
    """
    lines = "movementCues" in data
    cues = np.atleast_1d(data["movementCues" if lines else "characterCues"])
    go_bins = np.atleast_1d(data["goPeriodOnsetTimeBin"]).astype(int)
    delay_bins = np.atleast_1d(data["delayPeriodOnsetTimeBin"]).astype(int)
    block_ts = np.atleast_1d(data["blockNumsTimeSeries"]).astype(int)
    _same_length(cues=cues, go=go_bins, delay=delay_bins)

    known = set(LINE_CONDITIONS) if lines else {*CHAR_LIST, REST_CUE}
    unknown = set(map(str, cues)) - known
    if unknown:
        raise ValueError(f"unknown conditions {sorted(unknown)}")
    if not (delay_bins < go_bins).all():
        raise ValueError("a delay period does not precede its go cue")

    return [
        {
            "delay_onset_bin": int(delay),
            "go_onset_bin": int(go),
            "cue": str(cue),
            "block_num": int(block_ts[go - 1]),
        }
        for delay, go, cue in zip(delay_bins, go_bins, cues)
    ]


def extract_sentence_trials(data: dict) -> list[dict]:
    """Per-trial onsets, text, condition, block and exclusion flag of a sentences file."""
    go_bins = np.atleast_1d(data["goCueOnsetTimeBin"]).astype(int)
    delay_bins = np.atleast_1d(data["delayCueOnsetTimeBin"]).astype(int)
    end_bins = np.atleast_1d(data["sentenceEndTimeBin"]).astype(int)
    num_bins = np.atleast_1d(data["numTimeBinsPerSentence"]).astype(int)
    prompts = np.atleast_1d(data["sentencePrompt"])
    intended = np.atleast_1d(data["intendedText"])
    conditions = np.atleast_1d(data["sentenceCondition"])
    block_nums = np.atleast_1d(data["sentenceBlockNums"]).astype(int)
    excluded = np.atleast_1d(data["excludedSentences"]).astype(bool)
    _same_length(
        go=go_bins,
        delay=delay_bins,
        end=end_bins,
        num_bins=num_bins,
        prompts=prompts,
        intended=intended,
        conditions=conditions,
        blocks=block_nums,
        excluded=excluded,
    )
    if not (delay_bins < go_bins).all():
        raise ValueError("a delay period does not precede its go cue")

    return [
        {
            "delay_onset_bin": int(delay_bins[i]),
            "go_onset_bin": int(go_bins[i]),
            "end_bin": int(end_bins[i]),
            "num_time_bins": int(num_bins[i]),
            "sentence_prompt": _text(prompts[i]),
            "intended_text": _text(intended[i]),
            "sentence_condition": str(conditions[i]),
            "block_num": int(block_nums[i]),
            "excluded": bool(excluded[i]),
        }
        for i in range(len(go_bins))
    ]


# ---------------------------------------------------------------------------
# Published train/test partitions
# ---------------------------------------------------------------------------

#: The two partitions shipped in ``RNNTrainingSteps/``. ``HeldOutBlocks`` holds out whole
#: blocks, ``HeldOutTrials`` holds out individual trials; the paper's RNN pipeline
#: (Step4-Step7) uses ``HeldOutTrials``.
PARTITIONS = ("HeldOutBlocks", "HeldOutTrials")


def load_partition(
    partition: str, root: Path | str | None = None
) -> dict[str, dict[int, str]]:
    """``{session: {trial_index: "train" | "test"}}`` for one published partition.

    Trial indices are 0-based positions in the session's sentence arrays, so they index
    :func:`extract_sentence_trials` directly. Neither partition is a full cover: excluded
    sentences are in neither set, and so are the free-answer trials of the two
    ``2020.01.13``/``2020.01.15`` sessions.
    """
    if partition not in PARTITIONS:
        raise ValueError(
            f"unknown partition {partition!r}, expected one of {PARTITIONS}"
        )
    mat = sio.loadmat(
        str(_root(root) / "RNNTrainingSteps" / f"trainTestPartitions_{partition}.mat"),
        simplify_cells=True,
    )
    out: dict[str, dict[int, str]] = {}
    for key, idx in mat.items():
        if key.startswith("__") or "_" not in key:
            continue
        session, _, split = key.rpartition("_")
        if split not in ("train", "test"):
            continue
        out.setdefault(session, {}).update(
            {int(i): split for i in np.atleast_1d(idx).ravel()}
        )
    return out


# ---------------------------------------------------------------------------
# Convenience accessors
# ---------------------------------------------------------------------------


def list_cube_keys(data: dict) -> list[str]:
    """Sorted condition suffixes of the ``neuralActivityCube_{x}`` variables in ``data``."""
    prefix = "neuralActivityCube_"
    return sorted(k[len(prefix) :] for k in data if k.startswith(prefix))


def get_condition_cube(data: dict, condition: str) -> np.ndarray:
    """The ``(n_trials, 201, 192)`` uint8 cube of one condition."""
    return data[f"neuralActivityCube_{condition}"]


def session_summary(root: Path | str | None = None) -> list[dict]:
    """One row per session: description, implant day, tasks, bins and trials per task."""
    rows = []
    for session in list_sessions(root):
        tasks = list_task_files(session, root)
        row = {
            "session": session,
            "description": SESSION_DESCRIPTIONS[session],
            "post_implant_day": post_implant_day(session),
            "tasks": list(tasks),
        }
        for task, path in tasks.items():
            go_key = TRIAL_VARIABLES[task][0]
            go_bins = load_variables(path, [go_key])[go_key]
            row[f"{task}_n_timesteps"] = n_time_bins(path)
            row[f"{task}_n_trials"] = len(np.atleast_1d(go_bins))
        rows.append(row)
    return rows
