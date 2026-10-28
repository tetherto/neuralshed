"""Source-reading utilities for the Kunz et al. 2025 inner-speech dataset.

Reads the Dryad ``.mat`` source. Every file is a MATLAB v5/v7 file
(``scipy.io.loadmat``) holding the blocks of one task for one participant on one date,
concatenated with the unrecorded time between blocks removed, so the two feature matrices
index a uniform bin grid (not wall-clock time). :func:`block_segments` gives the blocks.

Per-file variables (verified against the real files)
-----------------------------------------------------
- ``binnedTX``  ``(T, F)``: binned threshold-crossing counts (-3.5 or -4.5 x SD,
  LRR-denoised).
- ``spikePow``  ``(T, F)``: binned spike-band power (mean sq. voltage >250 Hz, µV²).
- ``delayTrialEpochs`` / ``goTrialEpochs`` ``(N, 2)``: 1-based **inclusive** [start, stop]
  bin of each trial's delay (cue) / go (execution) period, into the T axis.
- ``binSize``: bin width in ms, **varies by file** (10, 20, 25 or 33 stored, and 40 read for
  the five T16 seqRecall files that store 20, see :data:`T16_SEQ_BIN_MS`).
- ``chanSets`` + ``chanSetNames``: per-array channel indices (1-based) and HCP-area labels.
- ``blockNum`` ``(T,)``: block id per bin (absent in the seqRecall families, whose blocks
  come from :data:`RECOVERED_SEQ_BLOCK_STOPS`).
- family metadata: ``trialCues``+``cueList`` (word-cue & sequence families),
  ``sentenceText``+``sentenceDurations`` (sentence families), ``numTargets`` (counting).

Two normalization regimes (read ``mean_subtracted`` / the dtype, don't assume):
- **raw families** (isolated, interleaved, sentence, chitty, counting): binnedTX = uint8
  counts (>=0), spikePow = float64 µV² (>0). NOT mean-subtracted, NOT z-scored: the
  paper's block-mean-subtraction + SD-normalization are applied in *analysis* code only.
- **seqRecall families**: block-wise **mean-subtracted** in the file (both features carry
  negatives; binnedTX is int8/float64, spikePow float32/float64). Not directly comparable
  to the raw count/power scale. Three T12 seqRecall files carry a ``uint8`` class tag on
  signed ``binnedTX`` bytes; :func:`_load_stream` reinterprets them as ``int8``.

Channel layout: every stored column is a recorded electrode. The in-file ``chanSets`` are
incomplete for the seven T12 seqRecall files (two of the four stored arrays), so the
array layout comes from :data:`ARRAY_LAYOUTS`, truncated to the stored width, never from
``chanSets``. Every stream is checked against its declared normalization on load.
"""

import re
import typing as tp
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import scipy.io as sio

from neuralshed.datasets.utils import dataset_dir
from neuralshed.text import PhonemeTable

if tp.TYPE_CHECKING:
    import pandas as pd

# --- Constants -----------------------------------------------------------------
# Data is staged in the framework study-folder (.../data/Kunz2025InnerSpeech/); the Study
# base appends that class-name subfolder, so utils mirrors the exact same path.
DATA_DIR = dataset_dir("kunz_2025") / "data" / "Kunz2025InnerSpeech"

# Array layout per participant in stored column order, (name, n_channels). Names follow the
# sibling datasets of the same participant (T12 willet_2023, T15 card_2024, T17
# jude_2026_speech). T12 stores two arrays (128 columns) in its 2023 word and sentence files
# and all four (256) in the seqRecall and December 2024 files, so its layout is truncated to
# the stored width. The in-file ``chanSetNames`` are source labels only: the T12 December
# 2024 files say i6v/M1/55b/s6v, most T16 files merge columns 128-255 into one ``6d`` group
# (the seqRecall files name i6d and s6d), and T17 merges its two 55b arrays into one ``55b``
# group. Channel correlations split that group at column 192, and its ventral-then-dorsal
# order is assumed from jude_2026_speech, not published.
ARRAY_LAYOUTS: dict[str, tuple[tuple[str, int], ...]] = {
    "T12": (
        ("ventral_6v", 64),
        ("dorsal_6v", 64),
        ("ventral_44", 64),
        ("dorsal_44", 64),
    ),
    "T15": (("ventral_6v", 64), ("4", 64), ("55b", 64), ("dorsal_6v", 64)),
    "T16": (("pef_55b", 64), ("6v", 64), ("ventral_6d", 64), ("dorsal_6d", 64)),
    "T17": (
        ("ventral_6v", 64),
        ("dorsal_6v", 64),
        ("ventral_55b", 64),
        ("dorsal_55b", 64),
    ),
}

# Source variable behind each neural feature stream.
STREAM_KEYS = {"threshold_crossings": "binnedTX", "spike_band_power": "spikePow"}

# The 7 single-syllable cue words (non-overlapping phonemes, similar duration) + rest.
WORDS = ("ban", "choice", "day", "feel", "kite", "though", "were")

# Trial-metadata family each experiment belongs to (drives trial parsing).
FAMILY_WORD_CUE = "word_cue"  # isolated, interleaved: single-word cues -> Word events
FAMILY_SENTENCE = "sentence"  # sentenceData, chitty: sentenceText  -> Sentence events
FAMILY_COUNTING = "counting"  # conjunctiveCounting: numTargets      -> Stimulus events
FAMILY_SEQUENCE = (
    "sequence"  # all seqRecall*: direction-sequence cues -> Stimulus events
)

# Per-experiment metadata. ``task_condition`` is the canonical vocab label (per-trial for
# the word-cue families, so it is resolved from the cue there); ``mean_subtracted`` flags
# the in-file block-mean-subtraction of the seqRecall families.
EXPERIMENTS: dict[str, dict] = {
    "isolatedVerbalBehaviors": dict(
        family=FAMILY_WORD_CUE,
        mean_subtracted=False,
        description="Seven verbal behaviors in separate blocks (attempted vocalized, "
        "mimed, motoric/auditory inner speech, imagined/passive listening, silent reading).",
    ),
    "interleavedVerbalBehaviors": dict(
        family=FAMILY_WORD_CUE,
        mean_subtracted=False,
        description="Same 7-word set with attempted/imagined/listening cue types "
        "intermixed within a block.",
    ),
    "sentenceData": dict(
        family=FAMILY_SENTENCE,
        mean_subtracted=False,
        description="Instructed-delay sentence production: attempt or imagine speaking a "
        "displayed sentence (50-word or ~125k-word vocab; *_eval = closed-loop).",
    ),
    "chittychittybangbang": dict(
        family=FAMILY_SENTENCE,
        mean_subtracted=False,
        description="Inner-speech sentence production with a keyword lock/unlock: a '*' "
        "prefix means imagine the keyword 'chittychittybangbang' first.",
    ),
    "conjunctiveCounting": dict(
        family=FAMILY_COUNTING,
        mean_subtracted=False,
        description="Silently count a cued shape-color combo in a grid (10-20), then "
        "vocalize the count. Probes uninstructed inner speech.",
    ),
    "seqRecallUninstructed3ElementArrow": dict(
        family=FAMILY_SEQUENCE,
        mean_subtracted=True,
        description="3-element arrow sequences, uninstructed mental strategy.",
    ),
    "seqRecallUninstructed3ElementLines": dict(
        family=FAMILY_SEQUENCE,
        mean_subtracted=True,
        description="3-element line-drawing sequences (motor control, no inner speech).",
    ),
    "seqRecallUninstructedSingleArrow": dict(
        family=FAMILY_SEQUENCE,
        mean_subtracted=True,
        description="Single-arrow joystick movement (arrow-cue motor control).",
    ),
    "seqRecallVerbalMemory": dict(
        family=FAMILY_SEQUENCE,
        mean_subtracted=True,
        description="Instructed verbal (inner-speech) memory strategy for sequence recall.",
    ),
    "seqRecallVisualMemory": dict(
        family=FAMILY_SEQUENCE,
        mean_subtracted=True,
        description="Instructed visual memory strategy for sequence recall.",
    ),
    "seqRecallSpeech": dict(
        family=FAMILY_SEQUENCE,
        mean_subtracted=True,
        description="Attempted speech of direction words (compares to verbal-memory recall).",
    ),
}

# Source behavior spelling (lower-cased) -> canonical ``behavior``. The verbatim cue stays
# in ``cue_text``.
BEHAVIOR_CANONICAL = {
    "attempted": "attempted",
    "mouthed": "mimed",
    "mimed": "mimed",
    "imagined1stmotor": "imagined_motor",
    "imagined1stauditory": "imagined_auditory",
    "imaginedlistening": "imagined_listening",
    "listening": "listening",
    "passivelistening": "listening",
    "listen": "listening",
    "reading": "reading",
}
# The interleaved files cue a bare "imagined" whose strategy is per participant, and T12's
# listening cue carries no label at all (authors' Fig5_ABCD.ipynb, behaviorNamesP).
INTERLEAVED_IMAGINED = {"T12": "imagined_motor", "T16": "imagined_auditory"}
REST = "rest"

# Canonical behavior -> task_condition (vocab.TASK_CONDITIONS).
CONDITION_OF_BEHAVIOR = {
    "attempted": "attempted",
    "mimed": "mimed",
    "imagined_motor": "imagined",
    "imagined_auditory": "imagined",
    "imagined_listening": "imagined",
    "imagined_hybrid": "imagined",
    "listening": "listening",
    "reading": "read",
    REST: "na",
}

# Inner-speech strategy of the imagined sentence files (paper STAR Methods, "Participant
# Inner Speech Behavior Strategy"). T15 switched to a hybrid of motoric inner speech and
# imagined listening in the 125k-word sessions.
SENTENCE_IMAGINED = {
    "T12": "imagined_motor",
    "T15": "imagined_motor",
    "T16": "imagined_auditory",
}
SENTENCE_IMAGINED_BY_DATE = {
    ("T15", "2024.12.15"): "imagined_hybrid",
    ("T15", "2025.01.05"): "imagined_hybrid",
}

# The imagined keyword that opens the chitty trials cued with ``*``, spelled as one token as
# in the authors' decoder corpus.
KEYWORD = "chittychittybangbang"

# seqRecall experiment -> memory strategy of the delay period. The go period is an attempted
# movement (joystick, drawing) or attempted speech in every one of them.
SEQ_STRATEGY = {
    "seqRecallSpeech": "speech",
    "seqRecallVerbalMemory": "verbal",
    "seqRecallVisualMemory": "visual",
    "seqRecallUninstructed3ElementArrow": "uninstructed",
    "seqRecallUninstructed3ElementLines": "uninstructed",
    "seqRecallUninstructedSingleArrow": "uninstructed",
}

# Spike-detection threshold per participant. The paper says "-3.5 or -4.5 times the standard
# deviation" by clinical-site default without naming the sites. The values are the ones the
# whole-body-mapping paper (Deo et al. 2026, Nature, same rigs) states for each participant.
THRESHOLD_SD = {"T12": -4.5, "T15": -4.5, "T16": -3.5, "T17": -3.5}

# Delay of the non-causal 250 Hz high-pass filter, per participant (paper STAR Methods).
HIGHPASS_DELAY_MS = {"T12": 4, "T15": 1, "T16": 4, "T17": 1}

# seqRecall go-period action. T12 moved a joystick in the arrow tasks and drew with a stylus
# in the others. T16 attempted to draw in every movement task (paper Figure S3).
SEQ_ACTION_KIND = {
    "seqRecallSpeech": "speech",
    "seqRecallVerbalMemory": "drawing",
    "seqRecallVisualMemory": "drawing",
    "seqRecallUninstructed3ElementArrow": "joystick",
    "seqRecallUninstructed3ElementLines": "drawing",
    "seqRecallUninstructedSingleArrow": "joystick",
}

# Isolated listening and reading blocks have no go cue: the word is presented over the
# source's delay epoch and its go epoch is the intertrial interval (paper Table S4, and the
# authors' Fig1D / Fig1EF scripts, which analyse those behaviors over the delay window).
# T12's interleaved listening trials played the word at delay onset too (Fig5_ABCD.ipynb).
PASSIVE_BEHAVIORS = ("listening", "reading")

# The gap after a go period of these T12 files is the 1.5 s return of the pen tip to its
# start (paper STAR Methods, 3-element lines, and the memory tasks of the same design).
RETURN_PERIOD_FILES = {
    "seqRecallUninstructed3ElementLines",
    "seqRecallVerbalMemory",
    "seqRecallVisualMemory",
}

# The five T16 seqRecall files store ``binSize=20`` but run on a 40 ms clock: only at 40 ms
# do their blocks reproduce the paper's Table S4 trial (2 s delay = the 50 bins from block
# start to the first go cue, 4 s go = the 100 bins from the last go cue to block end, 7 s
# cycle), their per-channel count variance is ~1.8-1.9x the mean count of T16's same-day
# 20 ms file (1.05 within that file), and the paper gives the same decoding window as 0.75 s
# (Methods, 20 ms label) and 1.5 s (Figure S3). A curation inference, not an
# author-confirmed correction (README.md, Notes / Scope).
T16_SEQ_BIN_MS = 40
T16_SEQ_DELAY_BINS = 50  # 2 s
T16_SEQ_GO_BINS = 100  # 4 s

# Exclusive stop bin of every block of the seqRecall files, which ship no ``blockNum``.
# Recovered offline with :func:`recover_block_stops` and checked in, so enumerating
# timelines never loads a feature matrix. Block ids are positions in the file.
RECOVERED_SEQ_BLOCK_STOPS: dict[str, tuple[int, ...]] = {
    "seqRecallSpeech/t12.2023.09.28_binnedTX_spikePow_20ms_speak.mat": (
        20033,
        36062,
        52089,
        68127,
    ),
    "seqRecallSpeech/t16.2024.07.17_binnedTX_spikePow_20ms.mat": (10478,),
    "seqRecallUninstructed3ElementArrow/t12.2023.05.16_binnedTX_spikePow_20ms_joystickSeq.mat": (
        11944,
        23899,
        35832,
        47725,
        59684,
        71600,
        83503,
        95412,
        107309,
        119197,
        131156,
        143089,
        155034,
        167004,
        179325,
        191563,
        203847,
        216154,
        228449,
        240783,
        253082,
        265352,
        277613,
        289921,
        302219,
        314528,
        326825,
        339146,
    ),
    "seqRecallUninstructed3ElementArrow/t16.2024.04.08_binnedTX_spikePow_20ms.mat": (
        6977,
        13954,
        20931,
        27908,
    ),
    "seqRecallUninstructed3ElementLines/t12.2023.09.19_binnedTX_spikePow_20ms_lineSequenes.mat": (
        20381,
        40626,
        60980,
        81246,
        101527,
        121846,
        142232,
        162600,
        182917,
        203260,
        223638,
        243941,
        264288,
        284566,
        304854,
        325192,
        345487,
    ),
    "seqRecallUninstructed3ElementLines/t16.2024.04.08_binnedTX_spikePow_20ms_lines.mat": (
        6977,
        13954,
        20931,
        27908,
    ),
    "seqRecallUninstructedSingleArrow/t12.2023.05.16_binnedTX_spikePow_20ms_joystickSingle.mat": (
        13271,
        26543,
        39801,
    ),
    "seqRecallVerbalMemory/t12.2023.09.28_binnedTX_spikePow_20ms_arrow.mat": (
        18053,
        36092,
        54147,
        72209,
    ),
    "seqRecallVerbalMemory/t12.2023.10.12_binnedTX_spikePow_20ms_verbalMemory.mat": (
        18078,
        36161,
        54229,
        72247,
    ),
    "seqRecallVerbalMemory/t16.2024.07.17_binnedTX_spikePow_20ms_verbal.mat": (
        6977,
        13954,
        20931,
        27908,
    ),
    "seqRecallVisualMemory/t12.2023.10.12_binnedTX_spikePow_20ms_visualMemory.mat": (
        18069,
        36138,
        54210,
        72305,
    ),
    "seqRecallVisualMemory/t16.2024.07.17_binnedTX_spikePow_20ms_visual.mat": (
        6977,
        13954,
        20931,
        27908,
    ),
}

_PARTICIPANT_RE = re.compile(r"^(t\d+)\.", re.IGNORECASE)
_DATE_RE = re.compile(r"^t\d+\.(\d{4}\.\d{2}\.\d{2})", re.IGNORECASE)


# --- File discovery ------------------------------------------------------------
def experiment_of_path(path: Path | str) -> str:
    """Experiment-type (folder) name for a ``.mat`` file."""
    return Path(path).parent.name


def participant_of_path(path: Path | str) -> str:
    """Participant id (``T12``..``T17``) parsed from the file name."""
    m = _PARTICIPANT_RE.match(Path(path).name)
    return m.group(1).upper() if m else "?"


def session_of_path(path: Path | str) -> str:
    """Session date (``2023.08.15``) parsed from the file name (``""`` if absent)."""
    m = _DATE_RE.match(Path(path).name)
    return m.group(1) if m else ""


def session_date(path: Path | str) -> str | None:
    """Recording date as ISO ``2023-08-15`` (``None`` if the file name has none)."""
    date = session_of_path(path)
    return date.replace(".", "-") if date else None


def session_label(path: Path | str) -> str:
    """Canonical session id of a file: its stem (unique per file).

    Shared by the block timelines of the file (``timeline["session"]``) and the join key
    with the ``sessions`` table (``label``, one row per file). Distinct from
    :func:`session_of_path` (the recording *date*, non-unique).
    """
    return Path(path).stem


def family_of_path(path: Path | str) -> str:
    """Trial-metadata family of a file (``FAMILY_*``), from its experiment folder."""
    exp = experiment_of_path(path)
    if exp not in EXPERIMENTS:
        raise ValueError(f"{path}: unknown experiment folder {exp!r}")
    return EXPERIMENTS[exp]["family"]


def list_mat_files(data_dir: Path | str = DATA_DIR) -> list[Path]:
    """Every dataset ``.mat`` file (sorted; skips macOS ``._`` shadow files)."""
    return sorted(
        p for p in Path(data_dir).rglob("*.mat") if not p.name.startswith("._")
    )


def list_timeline_files(data_dir: Path | str = DATA_DIR) -> list[Path]:
    """The ``.mat`` files the Study reads, one timeline per block of each.

    Every file, EXCEPT the 7 per-behavior ``isolatedVerbalBehaviors`` subsets: only the
    combined ``*_allBehaviors_raw.mat`` is kept (their exact concatenation, with the behavior
    encoded in the cue text), to avoid double-counting trials.
    """
    out = []
    for p in list_mat_files(data_dir):
        if (
            experiment_of_path(p) == "isolatedVerbalBehaviors"
            and "allBehaviors" not in p.name
        ):
            continue
        out.append(p)
    return out


# --- Loaders -------------------------------------------------------------------
# Trial/channel metadata is tiny; the binnedTX/spikePow matrices are hundreds of MB, so
# they are only ever loaded one stream at a time (variable_names) and bounded in RAM.
_META_VARS = [
    "binSize",
    "chanSets",
    "chanSetNames",
    "blockNum",
    "delayTrialEpochs",
    "goTrialEpochs",
    "trialCues",
    "cueList",
    "sentenceText",
    "sentenceDurations",
    "numTargets",
]


@lru_cache(maxsize=32)
def load_meta(path: str) -> dict:
    """Load just the light per-file metadata (no feature matrices), simplified."""
    present = [n for n, _, _ in sio.whosmat(path)]
    want = [v for v in _META_VARS if v in present]
    raw = sio.loadmat(path, variable_names=want, simplify_cells=True)
    meta = {k: v for k, v in raw.items() if not k.startswith("__")}
    meta["binSizeStored"] = int(np.atleast_1d(meta["binSize"]).ravel()[0])
    if bin_size_is_inferred(path):
        meta["binSize"] = T16_SEQ_BIN_MS
    return meta


def bin_size_is_inferred(path: str | Path) -> bool:
    """True for the T16 seqRecall files, whose bin size is :data:`T16_SEQ_BIN_MS`."""
    return (
        family_of_path(path) == FAMILY_SEQUENCE and participant_of_path(path) == "T16"
    )


def is_mean_subtracted(path: str | Path) -> bool:
    """Whether the file's experiment ships block-wise mean-subtracted features."""
    return bool(EXPERIMENTS.get(experiment_of_path(path), {}).get("mean_subtracted"))


def reinterpreted_uint8(path: str | Path) -> bool:
    """Whether ``binnedTX`` carries a ``uint8`` class tag on mean-subtracted (signed) data."""
    return stored_dtype(path, "binnedTX") == "uint8" and is_mean_subtracted(path)


def band_power_invalid_reason(path: str | Path) -> str:
    """Why the file's ``spikePow`` is unusable, ``""`` when it is fine.

    One file (``t12.2024.04.11_interleavedVerbalBehaviors_raw.mat``) stores ``spikePow``
    as ``uint8`` and byte-identical to ``binnedTX``.
    """
    if stored_dtype(path, "spikePow") == "uint8":
        return "spikePow is stored as uint8 and is a byte-identical copy of binnedTX"
    return ""


def _check_stream(arr: np.ndarray, key: str, mean_subtracted: bool, path: str) -> None:
    """Raise when a stream contradicts its declared normalization."""
    if mean_subtracted:
        if not (arr < 0).any() or abs(float(arr.mean())) > 0.5:
            raise ValueError(
                f"{path}: {key} is declared mean_subtracted but has no negatives or a "
                f"mean of {float(arr.mean()):.2f}"
            )
    else:
        if float(arr.min()) < 0:
            raise ValueError(f"{path}: {key} is declared raw but has negative values")
        if key == "binnedTX" and not np.all(arr == np.round(arr)):
            raise ValueError(
                f"{path}: binnedTX is declared raw counts but is not integer"
            )


@lru_cache(maxsize=2)
def _load_stream(path: str, feature: str) -> np.ndarray:
    """Load one full feature matrix ``(T, F)`` as float32 (heavy; bounded to 2 in RAM).

    A ``uint8`` ``binnedTX`` in a mean-subtracted experiment is a signed array with the
    wrong class tag and is viewed as ``int8`` before the cast.
    """
    key = STREAM_KEYS[feature]
    arr = sio.loadmat(path, variable_names=[key], simplify_cells=True)[key]
    ms = is_mean_subtracted(path)
    if key == "binnedTX" and arr.dtype == np.uint8 and ms:
        arr = arr.view(np.int8)
    arr = np.asarray(arr, dtype=np.float32)
    _check_stream(arr, key, ms, path)
    return arr


def _as_str_list(arr) -> list[str]:
    if arr is None:
        return []
    if isinstance(arr, (str, np.str_)):
        return [str(arr).strip()]
    return [str(x).strip() for x in np.atleast_1d(arr).ravel()]


def get_channel_sets(meta: dict) -> list[tuple[str, np.ndarray]]:
    """Per-array ``(source label, 0-based channel indices)`` as the file declares them.

    Source labels only: incomplete for the T12 seqRecall files and inconsistent across
    sessions. The layout the Study uses is :func:`array_layout`. Handles both storage
    variants: an object array of per-array index vectors (raw families) and a 2-D
    ``(A, 64)`` integer array (seqRecall). Indices are converted 1-based -> 0-based.
    """
    names = _as_str_list(meta.get("chanSetNames"))
    cs = meta.get("chanSets")
    groups: list[np.ndarray] = []
    if isinstance(cs, np.ndarray) and cs.dtype == object:
        groups = [np.asarray(g).ravel() for g in cs]
    elif isinstance(cs, np.ndarray) and cs.ndim == 2:
        groups = [cs[i].ravel() for i in range(cs.shape[0])]
    elif isinstance(cs, (list, tuple)):
        groups = [np.asarray(g).ravel() for g in cs]
    out = []
    for i, g in enumerate(groups):
        name = names[i] if i < len(names) else f"array{i}"
        out.append((name, g.astype(int) - 1))
    return out


def region_labels(meta: dict) -> list[str]:
    """The file's own array labels (``chanSetNames``), see :func:`get_channel_sets`."""
    return [name for name, _ in get_channel_sets(meta)]


def array_layout(path: str | Path) -> list[tuple[str, slice]]:
    """``(array name, column slice)`` per array, :data:`ARRAY_LAYOUTS` cut to the stored width."""
    width = n_channels(path)
    out: list[tuple[str, slice]] = []
    start = 0
    for name, n in ARRAY_LAYOUTS[participant_of_path(path)]:
        if start >= width:
            break
        out.append((name, slice(start, start + n)))
        start += n
    if start != width:
        raise ValueError(f"{path}: {width} stored columns do not match ARRAY_LAYOUTS")
    return out


def array_names(path: str | Path) -> list[str]:
    """Array names of a file in stored column order, see :func:`array_layout`."""
    return [name for name, _ in array_layout(path)]


def bin_size_ms(meta: dict) -> int:
    """Bin width in ms of a :func:`load_meta` record (the inferred 40 for T16 seqRecall)."""
    return int(np.atleast_1d(meta["binSize"]).ravel()[0])


def rate_hz(meta: dict) -> float:
    """Feature sampling rate (Hz) = 1000 / bin size."""
    return 1000.0 / bin_size_ms(meta)


@lru_cache(maxsize=128)
def _header(path: str, key: str) -> tuple[tuple[int, ...], str]:
    """``(shape, MATLAB class)`` of one variable from the file header (no array load)."""
    for name, shape, cls in sio.whosmat(path):
        if name == key:
            return tuple(int(x) for x in shape), str(cls)
    raise KeyError(f"{key} not in {path}")


def n_bins(path: str | Path) -> int:
    """Number of time bins (T) of a recording, from the binnedTX header."""
    return _header(str(path), "binnedTX")[0][0]


def n_channels(path: str | Path) -> int:
    """Number of stored channels (F) of a recording, from the binnedTX header."""
    return _header(str(path), "binnedTX")[0][1]


def stored_dtype(path: str | Path, key: str) -> str:
    """MATLAB class of a stored variable (``uint8``, ``int8``, ``single``, ``double``)."""
    return _header(str(path), key)[1]


def get_neural_stream(
    path: str | Path,
    feature: str,
    start_bin: int = 0,
    stop_bin: int | None = None,
) -> np.ndarray:
    """One neural feature stream as ``(n_channels, n_bins)`` float32, every stored column.

    ``feature`` is ``"threshold_crossings"`` (binnedTX) or ``"spike_band_power"``
    (spikePow). ``start_bin`` / ``stop_bin`` select one block (see :func:`block_segments`).
    """
    arr = _load_stream(str(path), feature)  # (T, F)
    return np.ascontiguousarray(arr[start_bin:stop_bin].T)  # (F, T)


# --- Blocks ---------------------------------------------------------------------
@dataclass(frozen=True)
class BlockSegment:
    """One continuously recorded block inside a file, ``[start_bin, stop_bin)``."""

    block: int
    start_bin: int
    stop_bin: int
    source: str  # "blockNum" (source ids) or "recovered" (seqRecall, positional ids)


def block_segments(path: str | Path, meta: dict | None = None) -> list[BlockSegment]:
    """The continuously recorded blocks of a file, in order, tiling ``[0, T)``.

    Neural activity between blocks was not recorded, so a block is the largest continuous
    span. ``blockNum`` gives the blocks of the raw families. The seqRecall files ship none,
    and their blocks come from :data:`RECOVERED_SEQ_BLOCK_STOPS` (block ids are positions,
    not source numbers).
    """
    path = Path(path)
    if meta is None:
        meta = load_meta(str(path))
    total = n_bins(path)
    if "blockNum" in meta:
        block = np.atleast_1d(meta["blockNum"]).ravel()
        if len(block) != total:
            raise ValueError(
                f"{path}: blockNum has {len(block)} entries for {total} bins"
            )
        edges = np.flatnonzero(np.diff(block) != 0) + 1
        starts, stops = np.r_[0, edges], np.r_[edges, total]
        ids = [int(block[a]) for a in starts]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{path}: a block number recurs after another block")
        return [
            BlockSegment(b, int(a), int(z), "blockNum")
            for b, a, z in zip(ids, starts, stops)
        ]
    key = "/".join(path.parts[-2:])
    if key not in RECOVERED_SEQ_BLOCK_STOPS:
        raise ValueError(
            f"{path}: no blockNum and no entry in RECOVERED_SEQ_BLOCK_STOPS. Recover the "
            "blocks with recover_block_stops() and add them."
        )
    stops = RECOVERED_SEQ_BLOCK_STOPS[key]
    if stops[-1] != total or list(stops) != sorted(set(stops)):
        raise ValueError(f"{path}: recovered stops {stops} do not tile {total} bins")
    starts = (0, *stops[:-1])
    return [
        BlockSegment(k + 1, int(a), int(z), "recovered")
        for k, (a, z) in enumerate(zip(starts, stops))
    ]


def recover_block_stops(
    path: str | Path,
    min_bins: int = 500,
    ratio: float = 0.005,
    chunk: int = 20000,
) -> tuple[int, ...]:
    """Block stops of a seqRecall file, recovered from its block-wise mean subtraction.

    Offline analysis behind :data:`RECOVERED_SEQ_BLOCK_STOPS`, never called on load. The
    float ``spikePow`` was centred block by block, so its cumulative sum returns to about
    zero at the end of every block. The score is the RMS over channels of that sum, per
    channel standard deviation. A block ends at the first minimum, at least ``min_bins``
    in, below ``ratio`` times the median score, and the sum restarts there.
    """
    power = sio.loadmat(str(path), variable_names=["spikePow"])["spikePow"]  # (T, F)
    total = len(power)
    scale = power.std(axis=0, dtype=np.float64)
    keep = np.isfinite(scale) & (scale > 0)
    stops: list[int] = []
    start = 0
    while start < total:
        running = np.zeros(int(keep.sum()))
        score = np.empty(total - start)
        for a in range(start, total, chunk):
            part = power[a : a + chunk, keep].astype(np.float64) / scale[keep]
            sums = running + np.cumsum(part, axis=0)
            score[a - start : a - start + len(part)] = np.sqrt((sums**2).mean(axis=1))
            running = sums[-1]
        below = np.flatnonzero(score[min_bins:] < ratio * np.median(score)) + min_bins
        if below.size == 0:
            raise ValueError(f"{path}: no centring minimum after bin {start}")
        run = below[below - below[0] < 50]
        stop = start + int(run[np.argmin(score[run])]) + 1
        stops.append(stop)
        start = stop
    return tuple(stops)


# --- Trial parsing -------------------------------------------------------------
def _is_rest(cue: str) -> bool:
    """Whether a word or sequence cue is the DO_NOTHING rest condition."""
    return re.sub(r"[^a-z]", "", cue.lower()).startswith("donothing")


def _parse_word_cue(cue: str) -> tuple[str, str]:
    """Split a word-cue string into ``(word, behavior)``.

    Tolerates the participant-specific encodings across files, e.g. ``"ban-attempted"``
    (isolated), ``"ban (imaginedlistening)"`` / ``"ban,(mimed)"`` / ``"ban\\n(attempted)$1"``
    / ``"ban$0"`` (interleaved). Returns ``("", "DO_NOTHING")`` for every rest cue.
    """
    s = re.sub(r"\$\d+", "", cue.replace("\\n", " ").replace("\n", " ")).strip()
    if _is_rest(s) or not s:
        return "", "DO_NOTHING"
    toks = [t for t in re.split(r"[(),\s\-]+", s) if t]
    word = next((t.lower() for t in toks if t.lower() in WORDS), "")
    if not word:
        return "", "DO_NOTHING"
    behavior = next((t for t in toks if t.lower() not in WORDS), "")
    return word, behavior


def _canonical_behavior(raw: str, word: str, path: Path) -> str:
    """Canonical ``behavior`` of a word cue (see :data:`BEHAVIOR_CANONICAL`)."""
    if not word:
        return REST
    key = raw.lower()
    if key == "imagined":
        return INTERLEAVED_IMAGINED[participant_of_path(path)]
    if not key:
        # a per-behavior isolated file names its behavior only in the file name
        if (from_name := _file_behavior(path)) is not None:
            return from_name
        # T12 interleaved listening: an audio cue with no on-screen label
        if experiment_of_path(path) != "interleavedVerbalBehaviors":
            raise ValueError(f"{path}: word cue {word!r} without a behavior label")
        return "listening"
    return BEHAVIOR_CANONICAL[key]


def _file_behavior(path: Path) -> str | None:
    """Canonical behavior of a per-behavior ``isolatedVerbalBehaviors`` file (its name, e.g.
    ``t12.2023.08.15_mouthed_raw``), ``None`` for any other file."""
    if experiment_of_path(path) != "isolatedVerbalBehaviors":
        return None
    return BEHAVIOR_CANONICAL.get(path.stem.split("_")[1].lower())


def get_trials(path: str | Path, meta: dict | None = None) -> "pd.DataFrame":
    """Per-trial table for one file as a pandas DataFrame.

    Common columns (all families): ``trial_id`` (0-based row of the source epochs, unique
    within the file), ``block`` (:func:`block_segments`), ``go_start_bin``/``go_stop_bin``
    and ``delay_start_bin``/``delay_stop_bin`` (0-based half-open file bins, clamped to the
    trial's block), the same spans in seconds (``*_time`` from the file origin, ``*_duration``),
    ``next_start_bin`` (the next trial's delay start in the block, ``go_stop_bin`` on a
    block's last trial), ``passive`` (an isolated listening or reading trial, or a T12
    interleaved listening trial, whose word sits in the delay epoch and whose go epoch is
    the intertrial interval, :data:`PASSIVE_BEHAVIORS`), ``task_condition`` and ``cue_text``
    (verbatim source cue). The T16 seqRecall files store analysis windows (100 bins before,
    150 after each go cue), so their task periods are rebuilt as the 50 bins before and the
    100 bins after the go cue (:data:`T16_SEQ_BIN_MS`). Every go span ends at the next delay
    start of its block or the block end, whichever comes first: the source's inclusive go
    stop index equals the next delay start in the chitty and T16 word files (one shared bin).

    Family-specific columns:
    - word_cue: ``word`` (``""`` on rest trials), ``behavior`` (canonical, from the file name
      in the 28 per-behavior isolated files, whose cues carry the word only),
      ``text_modality``.
    - sentence: ``sentence_text`` (the chitty ``*`` replaced by :data:`KEYWORD`), ``keyword``,
      ``behavior`` (:data:`SENTENCE_IMAGINED`), ``text_modality``, ``sentence_duration_bins``.
    - counting: ``num_targets``. The go span is the attempted vocalization of the count.
    - sequence: ``cue_id`` (index into the file's own ``cueList``, 0 on rest trials),
      ``strategy`` (:data:`SEQ_STRATEGY`), ``action_kind`` (:data:`SEQ_ACTION_KIND`),
      ``rest``, ``stimulus_modality``.
    """
    import pandas as pd

    path = Path(path)
    if meta is None:
        meta = load_meta(str(path))
    exp = experiment_of_path(path)
    family = family_of_path(path)
    participant = participant_of_path(path)
    bin_s = bin_size_ms(meta) / 1000.0

    delay = np.atleast_2d(meta["delayTrialEpochs"]).astype(int)
    go = np.atleast_2d(meta["goTrialEpochs"]).astype(int)
    n_trials = go.shape[0]
    segments = block_segments(path, meta)
    seg_stops = np.array([seg.stop_bin for seg in segments])
    seg_of = np.searchsorted(seg_stops, go[:, 0] - 1, side="right")
    # T16 seqRecall: the stored epochs are analysis windows, the task periods are Table S4's
    inferred = bin_size_is_inferred(path)
    cue_list = _as_str_list(meta.get("cueList"))
    trial_cues = (
        np.atleast_1d(meta["trialCues"]).astype(int).ravel()
        if "trialCues" in meta
        else None
    )
    sent_text = _as_str_list(meta.get("sentenceText"))
    sent_dur = (
        np.atleast_1d(meta["sentenceDurations"]).astype(int).ravel()
        if "sentenceDurations" in meta
        else None
    )
    num_targets = (
        np.atleast_1d(meta["numTargets"]).astype(int).ravel()
        if "numTargets" in meta
        else None
    )
    # default per-file condition (sentence files encode it in the file name)
    file_condition = file_task_condition(path)
    for name, vec in (
        ("delayTrialEpochs", delay),
        ("trialCues", trial_cues),
        ("sentenceText", sent_text if sent_text else None),
        ("numTargets", num_targets),
    ):
        if vec is not None and len(vec) != n_trials:
            raise ValueError(
                f"{path}: {name} has {len(vec)} rows for {n_trials} trials"
            )

    def _cue(i: int) -> str:
        k = trial_cues[i] - 1
        if not 0 <= k < len(cue_list):
            raise ValueError(f"{path}: trial {i} cue index {k + 1} not in cueList")
        return cue_list[k]

    rows = []
    for i in range(n_trials):
        seg = segments[seg_of[i]]
        # 1-based inclusive -> 0-based half-open, clamped to the trial's block
        ds = max(seg.start_bin, int(delay[i, 0]) - 1)
        gs = max(seg.start_bin, int(go[i, 0]) - 1)
        d1 = min(int(delay[i, 1]), seg.stop_bin)
        g1 = min(int(go[i, 1]), seg.stop_bin)
        if i + 1 < n_trials and seg_of[i + 1] == seg_of[i]:
            g1 = min(g1, int(delay[i + 1, 0]) - 1)
        if inferred:
            ds = max(seg.start_bin, gs - T16_SEQ_DELAY_BINS)
            d1 = gs
            g1 = min(gs + T16_SEQ_GO_BINS, seg.stop_bin)
        if not (seg.start_bin <= ds < d1 <= gs < g1 <= seg.stop_bin):
            raise ValueError(
                f"{path}: trial {i} epochs delay [{ds}, {d1}) go [{gs}, {g1}) do not fit "
                f"in block {seg.block} [{seg.start_bin}, {seg.stop_bin})"
            )
        if rows and rows[-1]["go_stop_bin"] > ds:
            raise ValueError(f"{path}: trial {i} delay starts inside trial {i - 1}")
        rec = dict(
            trial_id=i,
            block=seg.block,
            delay_start_bin=ds,
            delay_stop_bin=d1,
            go_start_bin=gs,
            go_stop_bin=g1,
            delay_start_time=ds * bin_s,
            delay_duration=(d1 - ds) * bin_s,
            go_start_time=gs * bin_s,
            go_duration=(g1 - gs) * bin_s,
        )
        if family == FAMILY_WORD_CUE and trial_cues is not None and cue_list:
            cue = _cue(i)
            word, raw_behavior = _parse_word_cue(cue)
            behavior = _canonical_behavior(raw_behavior, word, path)
            rec.update(
                cue_text=cue,
                word=word,
                behavior=behavior,
                task_condition=CONDITION_OF_BEHAVIOR[behavior],
                text_modality=_text_modality(CONDITION_OF_BEHAVIOR[behavior]),
            )
        elif family == FAMILY_SENTENCE and sent_text:
            text = sent_text[i].strip()
            keyword = text.startswith("*")
            text = text.lstrip("*").strip()
            if file_condition == "imagined":
                behavior = SENTENCE_IMAGINED_BY_DATE.get(
                    (participant, session_of_path(path)), SENTENCE_IMAGINED[participant]
                )
            else:
                behavior = file_condition
            rec.update(
                cue_text=sent_text[i],
                sentence_text=f"{KEYWORD} {text}" if keyword else text,
                keyword=keyword,
                behavior=behavior,
                sentence_duration_bins=(
                    int(sent_dur[i]) if sent_dur is not None else None
                ),
                task_condition=file_condition,
                text_modality=_text_modality(file_condition),
            )
        elif family == FAMILY_COUNTING and num_targets is not None:
            rec.update(
                cue_text=str(int(num_targets[i])),
                num_targets=int(num_targets[i]),
                task_condition="attempted",
            )
        elif family == FAMILY_SEQUENCE and trial_cues is not None and cue_list:
            cue = _cue(i)
            rest = _is_rest(cue)
            audio = exp == "seqRecallSpeech" and participant == "T12"
            rec.update(
                cue_text=cue,
                cue_id=0 if rest else int(trial_cues[i]),
                rest=rest,
                strategy=SEQ_STRATEGY[exp],
                action_kind=(
                    "drawing"
                    if participant == "T16" and exp != "seqRecallSpeech"
                    else SEQ_ACTION_KIND[exp]
                ),
                task_condition="na" if rest else "attempted",
                stimulus_modality="audio" if audio else "visual",
            )
        else:
            raise ValueError(f"{path}: no trial metadata for family {family!r}")
        rows.append(rec)
    trials = pd.DataFrame(rows)
    following = trials.groupby("block").delay_start_bin.shift(-1)
    trials["next_start_bin"] = following.fillna(trials.go_stop_bin).astype(int)
    trials["passive"] = False
    if exp == "isolatedVerbalBehaviors":
        passive_blocks = trials.loc[trials.behavior.isin(PASSIVE_BEHAVIORS), "block"]
        trials["passive"] = trials.block.isin(set(passive_blocks))
    elif exp == "interleavedVerbalBehaviors" and participant == "T12":
        trials["passive"] = trials.behavior == "listening"
    return trials


def _text_modality(task_condition: str) -> str:
    """``BaseText.modality`` of a cued word or sentence: how the text reached or left the
    participant."""
    return {"listening": "heard", "imagined": "imagined"}.get(task_condition, "read")


def file_task_condition(path: Path | str) -> str | None:
    """The ``task_condition`` every trial of a file shares, ``None`` for the word-cue files,
    which mix behaviors. Sentence files encode it in the file name."""
    family = family_of_path(path)
    if family in (FAMILY_COUNTING, FAMILY_SEQUENCE):
        return "attempted"
    if family != FAMILY_SENTENCE:
        return None
    name = Path(path).name.lower()
    if "attempted" in name:
        return "attempted"
    if "imagined" in name or experiment_of_path(path) == "chittychittybangbang":
        return "imagined"
    raise ValueError(f"{path}: no task condition in the file name")


def original_split(path: Path | str) -> str:
    """The deposit's partition of the sentence families: ``eval`` (the ``_eval`` closed-loop
    files) or ``train``. ``""`` for every other experiment, which publishes no split."""
    if family_of_path(path) != FAMILY_SENTENCE:
        return ""
    return "eval" if is_eval_file(path) else "train"


def loop_type(path: Path | str) -> str:
    """``closed`` for the real-time decoded ``_eval`` sentence files, ``open`` otherwise."""
    return "closed" if is_eval_file(path) else "open"


def is_eval_file(path: Path | str) -> bool:
    """Closed-loop (real-time decoded) evaluation file: the name ends in ``_eval``."""
    return Path(path).stem.endswith("_eval")


def sentence_vocab(path: Path | str) -> str:
    """``"50w"`` / ``"125k"`` for sentence files (from the file name), else ``"na"``."""
    name = Path(path).name.lower()
    if "125k" in name:
        return "125k"
    if "50word" in name or "50words" in name:
        return "50w"
    return "na"


# --- Summary tables ------------------------------------------------------------
@dataclass
class FileSummary:
    path: Path
    experiment: str
    participant: str
    session: str
    family: str
    n_trials: int
    n_bins: int
    n_channels: int
    bin_size_ms: int
    regions: list[str]
    source_labels: list[str]
    n_blocks: int
    mean_subtracted: bool
    is_eval: bool


def summarize_file(path: Path | str) -> FileSummary:
    """Light summary of one file (headers and trial metadata, no feature matrix)."""
    path = Path(path)
    meta = load_meta(str(path))
    exp = experiment_of_path(path)
    go = np.atleast_2d(meta["goTrialEpochs"])
    n_blocks = len(block_segments(path, meta))
    return FileSummary(
        path=path,
        experiment=exp,
        participant=participant_of_path(path),
        session=session_of_path(path),
        family=family_of_path(path),
        n_trials=int(go.shape[0]),
        n_bins=n_bins(path),
        n_channels=n_channels(path),
        bin_size_ms=bin_size_ms(meta),
        regions=array_names(path),
        source_labels=region_labels(meta),
        n_blocks=n_blocks,
        mean_subtracted=EXPERIMENTS.get(exp, {}).get("mean_subtracted", False),
        is_eval=is_eval_file(path),
    )


def summary_table(data_dir: Path | str = DATA_DIR, timelines_only: bool = False):
    """Per-file summary DataFrame (one row per ``.mat`` file)."""
    import pandas as pd

    files = (
        list_timeline_files(data_dir) if timelines_only else list_mat_files(data_dir)
    )
    rows = []
    for p in files:
        s = summarize_file(p)
        rows.append(
            dict(
                experiment=s.experiment,
                participant=s.participant,
                date=s.session,
                file=s.path.name,
                family=s.family,
                n_trials=s.n_trials,
                n_channels=s.n_channels,
                bin_size_ms=s.bin_size_ms,
                n_blocks=s.n_blocks,
                mean_subtracted=s.mean_subtracted,
                is_eval=s.is_eval,
                regions="|".join(s.regions),
                source_labels="|".join(s.source_labels),
            )
        )
    return pd.DataFrame(rows)


def experiment_overview(data_dir: Path | str = DATA_DIR):
    """Experiment-type x participants x file/trial counts DataFrame."""
    import pandas as pd

    df = summary_table(data_dir)
    g = df.groupby("experiment")
    rows = []
    for exp, sub in g:
        rows.append(
            dict(
                experiment=exp,
                family=EXPERIMENTS.get(exp, {}).get("family", "?"),
                n_files=len(sub),
                participants="|".join(sorted(sub.participant.unique())),
                n_trials=int(sub.n_trials.sum()),
                bin_sizes_ms="|".join(str(b) for b in sorted(sub.bin_size_ms.unique())),
                mean_subtracted=EXPERIMENTS.get(exp, {}).get("mean_subtracted", False),
            )
        )
    return pd.DataFrame(rows).sort_values("experiment").reset_index(drop=True)


# assets/cue_phonemes.json: cue text -> ARPAbet target, the cue's dictionary
# pronunciation (text.g2p_phonemes) over every sentence cue of the release. One
# table covers both T12 and T15.
CUE_PHONEMES = PhonemeTable(
    Path(__file__).resolve().parent / "assets" / "cue_phonemes.json"
)


def build_cue_phonemes(
    data_dir: Path | str = DATA_DIR, path: str | Path | None = None
) -> PhonemeTable:
    """Regenerate ``assets/cue_phonemes.json`` (or ``path``) over the ``sentence_text``
    of every sentence-family file the Study reads (:func:`list_timeline_files`). Needs
    the ``g2p`` extra."""
    cues: set[str] = set()
    for p in list_timeline_files(data_dir):
        if family_of_path(p) == FAMILY_SENTENCE:
            cues.update(t for t in get_trials(p).sentence_text if isinstance(t, str) and t)
    return PhonemeTable.build(
        cues,
        path or CUE_PHONEMES.path,
        description="Dictionary pronunciation of each sentence cue (T12 and T15), not a "
        "measurement of what was produced. The speechBCI g2p_en recipe "
        "(neuralshed.text.g2p_phonemes), stress removed, '|' at every word "
        "boundary and at the end. Symbols are PhonemeVocab v1 "
        "(neuralshed.text.ARPABET_V1).",
    )


if __name__ == "__main__":
    print(f"Data dir: {DATA_DIR}  exists={DATA_DIR.exists()}")
    files = list_mat_files()
    tls = list_timeline_files()
    print(
        f"{len(files)} .mat files -> {len(tls)} timelines "
        f"({len(files) - len(tls)} per-behavior isolated subsets skipped)"
    )
    print(experiment_overview().to_string(index=False))
