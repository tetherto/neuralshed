"""Source-reading utilities for the Jude et al. 2026 bimanual QWERTY-typing dataset.

Reads the Dryad per-block ``.mat`` files (one block per file, one ``scipy.io.loadmat``
dict) holding 10 ms-binned threshold crossings and spike-band power, the cued text or
finger movement, and trial timing. Two BrainGate2 participants: T17 (ALS, six arrays in
left precentral gyrus, areas 6v, 55b and 6d, microphone recorded) and T18 (spinal-cord
injury, six arrays in area 6d, four left and two right hemisphere, no microphone).

On-disk layout (fetched by ``study.Jude2026Typing.download`` via the shared Dryad backend)::

    <NEURALSHED_DATA_ROOT>/jude_2026_typing/data/Jude2026Typing/   DATA_DIR
    ├── README.md                                  Dryad description + block tables
    └── Data/                                      BLOCKS_ROOT
        ├── t17/t17.YYYY.MM.DD/<Typing|brainToText>/RedisMat/
        │   ├── YYYYMMDD_HHMMSS_(<block>).mat      one block per file
        │   └── YYYYMMDD_HHMMSS_(<block>).png      single-trial feature preview
        └── t18/t18.YYYY.MM.DD/<Typing|brainToText|Handwriting>/RedisMat/...

Per-block ``.mat`` fields:

    binned_neural_threshold_crossings  int16   (n_bins, 384)    -3.5 x SD counts per 10 ms bin
    binned_neural_spike_band_power     single  (n_bins, 384)    sum sq voltage per 10 ms bin
    binned_neural_nsp_timestamp        double  (1, n_bins)      NSP time (ns) per bin
    binned_neural_redis_clock          uint64  (1, n_bins)      BRAND/Redis time (ms) per bin
    cue                                char    (n_trials,)      cued sentence, movement or key
    go_cue_nsp_neural_time             double  (1, n_trials)    trial go (ns), neural clock
    trial_end_nsp_neural_time          double  (1, n_trials)    trial end (ns)
    trial_start_nsp_neural_time        double  (1, n_trials)    pre-cue trial start (ns)
    trial_paused_by_CNRA, trial_timed_out  int64 (1, n_trials)  zero on every trial
    decoder_logit_output               single  (n_rows, 41)     RNN character logits (closed loop)
    ngram_decoder_final_output         cell                     online 5-gram LM text (closed loop)
    microphone_data / *_nsp_analog_time                         T17 only

Both participants store 384 channels (six 64-channel arrays, :data:`ARRAYS`). The online
decoder used a feature subset (:data:`N_DECODER_FEATURES`), the saved arrays are always
384 channels. Task family comes from the Dryad block tables (:data:`SESSION_BLOCKS`).
Each file's ``block_description`` ("Runs a closed loop brianToText block ...", "Do a finger
sweep.") and ``graph_name`` separate the families too, and give the loop type
(:func:`described_loop_type`), which the Dryad README misstates for sentence copy.
"""

import re
from pathlib import Path

import numpy as np
import scipy.io

from neuralshed.datasets import participants
from neuralshed.datasets.utils import dataset_dir

# --- Constants -----------------------------------------------------------------
# Data lives under the StudyClassName wrapper so the neuralset Study base (which
# resolves self.path = <data>/Jude2026Typing) and these accessors read the same tree.
DATA_DIR = (
    dataset_dir("jude_2026_typing") / "data" / "Jude2026Typing"
)  # holds README.md + Data/
BLOCKS_ROOT = DATA_DIR / "Data"  # holds t17/ + t18/
SUBJECTS = ("T17", "T18")
DRYAD_DOI = "10.5061/dryad.cz8w9gjjk"
# Version 2; Data.zip unpacks to Data/. LanguageModels.zip (the 5-gram LM) is out of scope.
DRYAD_FILES = ("Data.zip", "README.md")

CHANNELS_PER_ARRAY = 64
N_CHANNELS = 384  # six 64-channel arrays, both participants
NEURAL_BIN_SIZE_MS = 10  # 10 ms non-overlapping bins
FS_NEURAL = 100  # Hz (downstream training compresses to 20 ms / 50 Hz)
AUDIO_FS = 30000  # Hz (microphone; T17 only)
THRESHOLD_SD = -3.5  # ncTX threshold, x per-electrode SD
# Online decoder features (crossings and power of the electrodes used): T17 256 on
# 2024-06-03 (the two 6d arrays) and 512 afterwards (6v and 6d, 55b omitted), T18 768.
N_DECODER_FEATURES = {"T17": 512, "T18": 768}

# Arrays in file column order, one block of 64 columns each, as (name, hemisphere, area).
# Column groups from the authors' FVAF notebooks: T17 v6v, d6v, v55b, d55b, v6d, d6d (the
# Study order of jude_2026_speech), T18 mv6d, v6d, d6d, md6d, Rl6d, Rm6d. Names follow
# metadata.py.
ARRAYS = {
    "T17": (
        ("ventral_6v", "left", "6v"),
        ("dorsal_6v", "left", "6v"),
        ("ventral_55b", "left", "55b"),
        ("dorsal_55b", "left", "55b"),
        ("ventral_6d", "left", "6d"),
        ("dorsal_6d", "left", "6d"),
    ),
    "T18": (
        ("mid_ventral_6d", "left", "6d"),
        ("ventral_6d", "left", "6d"),
        ("dorsal_6d", "left", "6d"),
        ("mid_dorsal_6d", "left", "6d"),
        ("lateral_6d", "right", "6d"),
        ("medial_6d", "right", "6d"),
    ),
}

# Neural context before the go cue (bins) the authors' loader prepends to each trial
# (formatSessionData start_offset). The emitted unit rows start at the go cue.
PRE_GO_BINS = {"T17": 450, "T18": 250}

# Participant registry keys (datasets/participants.py), which hold the implant dates.
PARTICIPANT_IDS = {"T17": "braingate2/T17", "T18": "braingate2/T18"}

# Block -> task family, transcribed from the Dryad README block tables, plus four
# sentence-copy blocks the tables omit (t18.2025.01.21 7, 01.22 4 and 7, 02.05 7), which
# the authors' T18_SessionArgs.py trains on. Families:
#   finger_movement  isolated attempted finger movement / single-key cue (Action target)
#   calibration      open-loop calibration sentence copy (Sentence target)
#   sentence_copy    closed-loop sentence copy, decoded keys shown online (Sentence target)
#   free_typing      closed-loop unconstrained typing (Sentence target, online decode)
#   handwriting      handwriting calibration sentence, open loop except one block
# Every other file on disk is in EXCLUDED_BLOCKS or DUPLICATE_FILES.
_FM, _CAL, _COPY, _FREE, _HW = (
    "finger_movement",
    "calibration",
    "sentence_copy",
    "free_typing",
    "handwriting",
)
SESSION_BLOCKS: dict[str, dict[int, str]] = {
    # ---- T17 ----
    "t17.2024.04.24": {b: _FM for b in (1, 2, 3, 6, 7, 8, 9, 10)},
    "t17.2024.04.25": {b: _CAL for b in (1, 3, 4, 5)},
    "t17.2024.06.03": {
        **{b: _CAL for b in (4, 5, 7, 8, 9, 11)},
        12: _COPY,
    },
    "t17.2024.06.04": {
        **{b: _CAL for b in (1, 2, 3, 5)},
        6: _COPY,
        **{b: _FREE for b in (8, 9, 10, 11)},
    },
    "t17.2024.06.05": {1: _CAL, 2: _COPY, 3: _FREE, 5: _CAL, 6: _CAL},
    "t17.2024.06.13": {7: _CAL, 8: _CAL},
    "t17.2024.07.09": {
        **{b: _CAL for b in (1, 2, 3, 6)},
        **{b: _FREE for b in (4, 7, 8, 10, 12, 13)},
    },
    "t17.2024.07.10": {1: _CAL, 2: _CAL, 3: _FREE, 4: _FREE},
    "t17.2024.07.22": {1: _CAL, 2: _CAL, 3: _CAL, 4: _FREE, 5: _FREE},
    # ---- T18 ----
    "t18.2024.12.03": {b: _FM for b in (1, 2, 3, 4, 5, 6)},
    "t18.2024.12.04": {2: _FM, 3: _FM, 5: _CAL, 6: _CAL, 7: _CAL},
    "t18.2024.12.05": {1: _CAL, 2: _CAL, 3: _COPY, 4: _COPY, 5: _FREE, 6: _COPY},
    "t18.2025.01.14": {1: _CAL, 2: _CAL, **{b: _COPY for b in range(3, 12)}},
    "t18.2025.01.15": {
        1: _CAL,
        2: _COPY,
        3: _CAL,
        5: _CAL,
        6: _CAL,
        **{b: _COPY for b in (7, 8, 9, 10, 11, 12, 13, 14)},
    },
    "t18.2025.01.21": {
        1: _CAL,
        2: _CAL,
        3: _CAL,
        **{b: _COPY for b in range(4, 12)},
    },
    "t18.2025.01.22": {
        1: _CAL,
        2: _CAL,
        3: _CAL,
        **{b: _COPY for b in range(4, 14)},
    },
    "t18.2025.02.04": {
        1: _CAL,
        2: _CAL,
        3: _CAL,
        4: _COPY,
        5: _FREE,
        **{b: _COPY for b in (6, 7, 8, 9, 10, 11, 12, 13, 14)},
    },
    "t18.2025.02.05": {
        1: _CAL,
        2: _CAL,
        3: _CAL,
        **{b: _COPY for b in range(4, 17)},
    },
    "t18.2025.04.01": {b: _HW for b in (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)},
}

# Listed in the Dryad README block tables but absent from Data.zip (the paper's six
# further finger-movement repetitions of trial day 95, "not used").
NOT_DEPOSITED: dict[str, tuple[int, ...]] = {"t17.2024.06.03": (1, 2, 3)}

# Deposited blocks the Study does not emit, one reason each.
_REFERENCE = "reference block, one DO_NOTHING trial, used to fit LRR and thresholds"
EXCLUDED_BLOCKS: dict[str, dict[int, str]] = {
    "t17.2024.06.04": {7: "free-typing block with no recorded sentences"},
    "t18.2024.12.03": {0: _REFERENCE, 7: _REFERENCE, 8: _REFERENCE},
}

# A second file of an already listed block, skipped by list_blocks.
DUPLICATE_FILES: dict[str, dict[str, str]] = {
    "t18.2024.12.04": {
        "20241204_164727_(7).mat": "re-export of 20241204_163713_(7).mat, identical arrays"
    },
}

for _session, _blocks in EXCLUDED_BLOCKS.items():
    _both = set(_blocks) & set(SESSION_BLOCKS.get(_session, {}))
    if _both:
        raise ValueError(
            f"{_session} blocks {sorted(_both)} are catalogued and excluded"
        )

# Task-family -> (cued-unit neuralset event type, loop_type). The loop type follows each
# file's block_description, not the Dryad README, which calls sentence copy open loop.
TASK_FAMILY_INFO = {
    _FM: ("Action", "open"),
    _CAL: ("Sentence", "open"),
    _COPY: ("Sentence", "closed"),
    _FREE: ("Sentence", "closed"),
    _HW: ("Sentence", "open"),
}
# Blocks whose loop type differs from their family's.
LOOP_TYPE_OVERRIDES: dict[str, dict[int, str]] = {
    "t18.2025.04.01": {13: "closed"},  # T18_Handwriting_ClosedLoop.yaml, 2 trials
}


def loop_type(session: str, block: int) -> str:
    """``open`` or ``closed`` for a catalogued (session, block)."""
    override = LOOP_TYPE_OVERRIDES.get(session, {}).get(block)
    return override or TASK_FAMILY_INFO[SESSION_BLOCKS[session][block]][1]


def described_loop_type(path: str | Path) -> str | None:
    """Loop type stated by a block file's ``block_description``, ``None`` when it states none.

    "Runs a closed loop ..." and "Runs a personal use ..." (free typing) are ``closed``,
    "Runs a open loop ..." is ``open``. Finger sweeps and reference blocks state none.
    """
    text = str(read_block(path, fields=["block_description"])["block_description"])
    if text.startswith(("Runs a closed loop", "Runs a personal use")):
        return "closed"
    if text.startswith("Runs a open loop"):
        return "open"
    return None


# QWERTY layout (paper Fig. 3a): each finger owns a column, the movement direction picks
# the row (extension = top, straight-down flexion = middle, curl into the palm = bottom).
# Confirmed on T18: the key-cue sweep (2024.12.04) matches the movement sweep (2024.12.03)
# condition by condition on 28 of 31 cues, the other 3 by an adjacent finger.
FINGERS = (
    "LEFT PINKY",
    "LEFT RING",
    "LEFT MIDDLE",
    "LEFT INDEX",
    "LEFT THUMB",
    "RIGHT THUMB",
    "RIGHT INDEX",
    "RIGHT MIDDLE",
    "RIGHT RING",
    "RIGHT PINKY",
)
DIRECTIONS = ("Extend Upwards", "Straight Downwards", "In to Palm")
KEY_ROWS = ("QWERTYUIOP", "ASDFGHJKL?", "ZXCVB NM,.")
REST_CUES = ("DO NOTHING", "DO_NOTHING")  # movement sweeps, key sweep
_KEY_CUE = {" ": "SPACE", ",": ", (COMMA)", ".": ". (PERIOD)", "?": "? (QUESTION MARK)"}

# Dataset-wide class id of a finger-movement cue: 0 = rest, 1-30 = keys in keyboard
# order (Q = 1 ... P = 10, A = 11 ... ? = 20, Z = 21 ... . = 30). A movement cue and the
# key it types share the id.
CUE_CODES: dict[str, int] = {cue: 0 for cue in REST_CUES}
MOVEMENTS: dict[int, str] = {0: "DO NOTHING"}  # id -> movement, the unit's description
for _r, _row in enumerate(KEY_ROWS):
    for _c, _key in enumerate(_row):
        _code = 1 + 10 * _r + _c
        MOVEMENTS[_code] = f"{FINGERS[_c]} - {DIRECTIONS[_r]}"
        CUE_CODES[MOVEMENTS[_code]] = _code
        CUE_CODES[_KEY_CUE.get(_key, _key)] = _code


def cue_code(cue: str) -> int:
    """Dataset-wide class id of a finger-movement or key cue (``KeyError`` if unknown)."""
    return CUE_CODES[cue]


# Trial-level (one value per trial) fields loaded by get_trials.
_TRIAL_FIELDS = (
    "cue",
    "block_number",
    "trial_start_nsp_neural_time",
    "go_cue_nsp_neural_time",
    "trial_end_nsp_neural_time",
    "delay_duration_ms",
    "inter_trial_duration_ms",
    "trial_paused_by_CNRA",
    "trial_timed_out",
)


# --- Session / block discovery -------------------------------------------------
def subject_of(session: str) -> str:
    """Subject label (``T17``/``T18``) from a session name (``t17.2024.04.24`` -> ``T17``)."""
    return session.split(".")[0].upper()


def session_date(session: str) -> str:
    """ISO date from a session name (``t17.2024.04.24`` -> ``2024-04-24``)."""
    return session.split(".", 1)[1].replace(".", "-")


def post_implant_day(session: str) -> int:
    """Days from the participant's implant (registry) to a session's date.

    Equals the Dryad README's "Trial day" for every session except ``t18.2025.04.01``,
    which the README labels 163 and the implant date puts at 162.
    """
    pid = PARTICIPANT_IDS[subject_of(session)]
    day = participants.post_implant_day(pid, session_date(session))
    if day is None:
        raise ValueError(f"no implant date for {pid}")
    return day


def list_sessions(blocks_root: str | Path = BLOCKS_ROOT) -> list[str]:
    """All session names (``t1?.YYYY.MM.DD``) across both participants, sorted."""
    root = Path(blocks_root)
    out = []
    for part in ("t17", "t18"):
        pdir = root / part
        if pdir.is_dir():
            out += [
                p.name
                for p in pdir.iterdir()
                if p.is_dir() and p.name.startswith(f"{part}.2")
            ]
    return sorted(out)


def session_dir(session: str, blocks_root: str | Path = BLOCKS_ROOT) -> Path:
    """The session's ``RedisMat`` folder (under its single Typing/brainToText/Handwriting task dir)."""
    sdir = Path(blocks_root) / session.split(".")[0] / session
    for rm in sorted(sdir.glob("*/RedisMat")):
        return rm
    raise FileNotFoundError(f"no */RedisMat under {sdir}")


def session_modality(session: str, blocks_root: str | Path = BLOCKS_ROOT) -> str:
    """Recording-task folder name for a session (``Typing``/``brainToText``/``Handwriting``)."""
    return session_dir(session, blocks_root).parent.name


def _block_num(path: Path) -> int:
    """Block number parsed from a ``..._(<n>).mat`` filename (authoritative vs the field)."""
    m = re.search(r"\((\d+)\)\.mat$", path.name)
    if m is None:
        raise ValueError(f"no block number in {path.name}")
    return int(m.group(1))


def _block_files(session: str, blocks_root: str | Path) -> list[Path]:
    """Every ``.mat`` file in a session's ``RedisMat`` folder, duplicates included."""
    rmdir = session_dir(session, blocks_root)
    return sorted(p for p in rmdir.glob("*.mat") if not p.name.startswith("._"))


def list_blocks(
    session: str, blocks_root: str | Path = BLOCKS_ROOT
) -> list[tuple[int, Path]]:
    """``(block_number, path)`` for every block in a session, sorted by block.

    Skips the files in ``DUPLICATE_FILES``, so each block number appears once.
    """
    skip = DUPLICATE_FILES.get(session, {})
    blocks = sorted(
        (_block_num(p), p)
        for p in _block_files(session, blocks_root)
        if p.name not in skip
    )
    numbers = [b for b, _ in blocks]
    if len(numbers) != len(set(numbers)):
        raise ValueError(f"{session}: two files share a block number, {numbers}")
    return blocks


def block_path(session: str, block: int, blocks_root: str | Path = BLOCKS_ROOT) -> Path:
    """Path to one block's ``.mat`` file."""
    for b, p in list_blocks(session, blocks_root):
        if b == block:
            return p
    raise FileNotFoundError(f"block {block} not found in {session}")


def task_of(session: str, block: int) -> str:
    """Task family of a catalogued (session, block), ``"unknown"`` otherwise."""
    return SESSION_BLOCKS.get(session, {}).get(block, "unknown")


def exclusion_reason(session: str, block: int) -> str | None:
    """Why a deposited (session, block) is not catalogued, ``None`` when it is."""
    return EXCLUDED_BLOCKS.get(session, {}).get(block)


# --- Deposit verification ------------------------------------------------------
def expected_blocks() -> dict[str, set[int]]:
    """Every block the deposit holds, catalogued or excluded, per session."""
    out: dict[str, set[int]] = {}
    for table in (SESSION_BLOCKS, EXCLUDED_BLOCKS):
        for session, blocks in table.items():
            out.setdefault(session, set()).update(blocks)
    return out


def verify_tree(
    data_dir: str | Path = DATA_DIR, readme_digest: str | None = None
) -> list[str]:
    """Problems with the extracted deposit under ``data_dir``, empty when it is complete.

    Checks the README (against ``readme_digest`` when given), that every session holds
    exactly the blocks of ``expected_blocks()`` plus its ``DUPLICATE_FILES``, and that
    each block file opens with ``scipy.io.whosmat``.
    """
    import hashlib

    data_dir = Path(data_dir)
    problems: list[str] = []
    readme = data_dir / "README.md"
    if not readme.is_file():
        problems.append("README.md missing")
    elif (
        readme_digest
        and hashlib.sha256(readme.read_bytes()).hexdigest() != readme_digest
    ):
        problems.append("README.md differs from the Dryad copy")
    blocks_root = data_dir / "Data"
    for session, blocks in sorted(expected_blocks().items()):
        try:
            files = _block_files(session, blocks_root)
        except FileNotFoundError:
            problems.append(f"{session} missing")
            continue
        found = [(_block_num(p), p) for p in files]
        numbers = {b for b, _ in found}
        if missing := sorted(blocks - numbers):
            problems.append(f"{session} missing blocks {missing}")
        if extra := sorted(numbers - blocks):
            problems.append(f"{session} has uncatalogued blocks {extra}")
        duplicates = set(DUPLICATE_FILES.get(session, {}))
        for name in sorted(duplicates - {p.name for p in files}):
            problems.append(f"{session} missing duplicate file {name}")
        kept = [b for b, p in found if p.name not in duplicates]
        if len(kept) != len(set(kept)):
            problems.append(f"{session} has an unlisted second file for a block")
        for _, p in found:
            try:
                scipy.io.whosmat(str(p))
            except Exception as error:
                problems.append(f"{session} {p.name} unreadable: {error}")
    return problems


# --- Per-block readers ---------------------------------------------------------
def n_channels(path: str | Path) -> int:
    """Recorded channel count of a block (384) without loading the arrays."""
    for name, shape, _ in scipy.io.whosmat(str(path)):
        if name == "binned_neural_threshold_crossings":
            return int(shape[1])
    raise KeyError("binned_neural_threshold_crossings not in file")


def array_channels(subject: str) -> dict[str, tuple[int, int]]:
    """Array -> half-open column range of a subject's 384 channels."""
    return {
        name: (i * CHANNELS_PER_ARRAY, (i + 1) * CHANNELS_PER_ARRAY)
        for i, (name, _hemisphere, _area) in enumerate(ARRAYS[subject.upper()])
    }


def group_channels(subject: str, by: str = "area") -> dict[str, np.ndarray]:
    """Column indices per ``area`` (``6v``, ``55b``, ``6d``) or per ``hemisphere``."""
    key = {"area": 2, "hemisphere": 1}[by]
    out: dict[str, list[int]] = {}
    for i, array in enumerate(ARRAYS[subject.upper()]):
        out.setdefault(array[key], []).extend(
            range(i * CHANNELS_PER_ARRAY, (i + 1) * CHANNELS_PER_ARRAY)
        )
    return {k: np.asarray(v) for k, v in out.items()}


def read_block(path: str | Path, fields: list[str] | None = None) -> dict:
    """Load a block ``.mat`` (optionally only ``fields``), squeezed.

    Pass ``fields`` to avoid loading the large ``binned_neural_*`` / ``microphone_*``
    arrays when you only need timing or labels.
    """
    kw = {"squeeze_me": True, "struct_as_record": False}
    if fields is not None:
        kw["variable_names"] = list(fields)
    return scipy.io.loadmat(str(path), **kw)


def get_neural(path: str | Path):
    """Return ``(threshold_crossings, spike_band_power)``, each ``(n_bins, n_ch)``.

    threshold_crossings is int16 counts; spike_band_power is float32. Neither is z-scored.
    """
    d = read_block(
        path,
        fields=["binned_neural_threshold_crossings", "binned_neural_spike_band_power"],
    )
    return (
        np.asarray(d["binned_neural_threshold_crossings"]),
        np.asarray(d["binned_neural_spike_band_power"]),
    )


def get_bin_times_ns(path: str | Path) -> np.ndarray:
    """Per-bin NSP timestamps (nanoseconds), shape ``(n_bins,)``."""
    d = read_block(path, fields=["binned_neural_nsp_timestamp"])
    return np.asarray(d["binned_neural_nsp_timestamp"], dtype=np.float64).ravel()


def get_audio(path: str | Path):
    """Return ``(audio, fs)``: 1-D int16 microphone samples and :data:`AUDIO_FS` (T17 only).

    Returns ``(None, AUDIO_FS)`` for T18 blocks, which carry no microphone.
    """
    avail = {n for n, _, _ in scipy.io.whosmat(str(path))}
    if "microphone_data" not in avail:
        return None, AUDIO_FS
    d = read_block(path, fields=["microphone_data"])
    return np.asarray(d["microphone_data"]).ravel(), AUDIO_FS


def clean_cue(cue: str, modality: str = "typing") -> str:
    """Strip a cue to its plain text. Handwriting cues use ``>`` (space) and ``~`` (end)
    tokens (e.g. ``"they > were > him ~"`` -> ``"they were him"``)."""
    cue = str(cue).strip()
    if modality == "handwriting":
        cue = re.sub(r"\s*[>~]\s*", " ", cue)
    return cue.strip()


def get_trials(path: str | Path, modality: str = "typing"):
    """Per-trial table (one row per cue) as a DataFrame.

    Columns: ``cue`` (cleaned), ``go_cue_ns``, ``trial_end_ns``, ``trial_start_ns``,
    ``delay_ms``, ``iti_ms``, ``paused``, ``timed_out``.
    """
    import pandas as pd

    avail = {n for n, _, _ in scipy.io.whosmat(str(path))}
    want = [f for f in _TRIAL_FIELDS if f in avail]
    d = read_block(path, fields=want)

    cue = np.atleast_1d(d.get("cue"))
    out = {"cue": [clean_cue(c, modality) for c in cue]}
    n = len(out["cue"])

    def col(key):
        return (
            np.atleast_1d(d[key]).astype(np.float64).ravel()[:n]
            if key in d
            else np.full(n, np.nan)
        )

    out["go_cue_ns"] = col("go_cue_nsp_neural_time")
    out["trial_end_ns"] = col("trial_end_nsp_neural_time")
    out["trial_start_ns"] = col("trial_start_nsp_neural_time")
    out["delay_ms"] = col("delay_duration_ms")
    out["iti_ms"] = col("inter_trial_duration_ms")
    out["paused"] = col("trial_paused_by_CNRA").astype(bool)
    out["timed_out"] = col("trial_timed_out").astype(bool)
    return pd.DataFrame(out)


# --- Trial segmentation --------------------------------------------------------
def clock_breaks(bin_times_ns: np.ndarray) -> np.ndarray:
    """Bins after which the NSP clock jumps (a recording pause), as sorted indices ``k``.

    11 blocks (10 free-typing blocks and handwriting block 3) hold 49 pauses of 22 ms-420 s
    with no bins, so bin ``k + 1`` is not 10 ms after bin ``k``.
    """
    step_ns = 1e9 / FS_NEURAL
    return np.flatnonzero(np.diff(bin_times_ns) > 1.5 * step_ns)


def bin_index(bin_times_ns: np.ndarray, t_ns: float) -> int:
    """Index of the neural bin nearest an NSP timestamp (``argmin|t - bin_times|``)."""
    return int(np.argmin(np.abs(bin_times_ns - t_ns)))


def trial_window(
    bin_times_ns: np.ndarray,
    go_cue_ns: float,
    trial_end_ns: float,
    pre_go_bins: int = 0,
) -> tuple[int, int]:
    """``(start_bin, end_bin)`` for a trial: GO cue (minus ``pre_go_bins``) to trial end.

    The companion loader uses ``pre_go_bins`` = 450 (T17) / 250 (T18) of pre-GO context.
    """
    start = bin_index(bin_times_ns, go_cue_ns) - pre_go_bins
    end = bin_index(bin_times_ns, trial_end_ns)
    return max(start, 0), end


if __name__ == "__main__":
    sessions = list_sessions()
    print(f"{len(sessions)} sessions across {len(SUBJECTS)} participants")
    for s in sessions:
        blocks = list_blocks(s)
        fams = sorted({task_of(s, b) for b, _ in blocks})
        print(
            f"  {s} (day {post_implant_day(s):>3}) {session_modality(s):11s} "
            f"{len(blocks):>2} blocks :: {', '.join(fams)}"
        )

    s0, b0 = "t18.2024.12.04", 5
    p = next(p for b, p in list_blocks(s0) if b == b0)
    print(
        f"\n{s0} block {b0} [{task_of(s0, b0)}]  subject={subject_of(s0)}  n_ch={n_channels(p)}"
    )
    tx, sbp = get_neural(p)
    print(
        f"  threshold_crossings {tx.shape} {tx.dtype} | spike_band_power {sbp.shape} {sbp.dtype}"
    )
    trials = get_trials(p)
    print(f"  {len(trials)} trials; cue[0]={trials.cue[0]!r}")
