"""Source-reading utilities for the Jude et al. 2026 anarthria intended-speech dataset.

Reads the raw Dryad source directly (no NWB): per-block ``.mat`` files (one block =
one ``scipy.io.loadmat`` dict) holding 10 ms-binned threshold crossings + spike-band
power, the cued language unit, trial timing, and 30 kHz microphone audio. Single
BrainGate2 participant T17 (advanced ALS, anarthria, locked-in).

On-disk layout (populated by ``study.py``'s ``_download`` via the shared Dryad backend)::

    <NEURALSHED_DATA_ROOT>/jude_2026_speech/
    └── data/
        └── Jude2026IntendedSpeech/                     <- DATA_DIR (study subfolder)
            ├── README.md                              Dryad description + block tables
            └── t17/
                └── t17.YYYY.MM.DD/                     14 session folders
                    └── brainToText/RedisMat/
                        ├── YYYYMMDD_HHMMSS_(<block>).mat   one block per file
                        └── YYYYMMDD_HHMMSS_(<block>).png   single-trial feature preview

Per-block ``.mat`` fields (verified vs ``t17.2024.04.17/...(10).mat`` and siblings):

    binned_neural_threshold_crossings  int16   (n_bins, n_ch)   -3.5xSD counts / 10 ms bin
    binned_neural_spike_band_power     float32 (n_bins, n_ch)   sum sq voltage / 10 ms bin
    binned_neural_nsp_timestamp        float64 (1, n_bins)      NSP time (ns) per bin
    cue                                char    (n_trials,)      cued language unit (padded)
    go_cue_nsp_neural_time             float64 (1, n_trials)    trial GO (ns), neural clock
    trial_end_nsp_neural_time          float64 (1, n_trials)    trial end (ns)
    trial_start_nsp_neural_time        float64 (1, n_trials)    pre-cue trial start (ns)
    trial_paused_by_CNRA, trial_timed_out  int64 (1, n_trials)  drop-trial flags
    microphone_data                    int16   (n_samples, 1)   30 kHz audio
    microphone_nsp_time                uint64  (n_samples, 1)   audio NSP time (ns)
    decoder_logit_output               float32 (n_frames, 41)   RNN phoneme logits (CL only)
    ngram_decoder_final_output, ...    cell                     online LM output (CL only)

``n_ch`` is **per session**: 384 (all six arrays) on 11 sessions, 256 (the 2x6v + 2x55b
speech-decoder subset, 6d dropped at acquisition) on 2024.04.15/04.16/04.17. Read it
from the file; never hardcode. The file column order of the six arrays differs between
the three March and early April sessions and every later one (``FILE_GROUPS``), so the
Study permutes every block into one Study order (``ARRAY_ORDER``, :func:`to_study_order`).
The phased-task session (2024.05.23) additionally carries ``cue_presentation`` /
``imagined_go_period`` / ``vocalized_go_cue`` ``_nsp_neural_time`` phase markers (see
:func:`phase_windows`).
"""

from pathlib import Path

import numpy as np
import scipy.io

from neuralshed.datasets import participants
from neuralshed.datasets.utils import dataset_dir
from neuralshed.text import ARPABET_V1, PhonemeTable

# --- Constants -----------------------------------------------------------------
# Data lands under data/<StudyClassName>/ (the neuralset Study resolves its own
# subfolder on download); mirror it here so metadata.sessions() reads the same tree.
DATA_DIR = dataset_dir("jude_2026_speech") / "data" / "Jude2026IntendedSpeech"
T17_DIR = DATA_DIR / "t17"

N_ARRAYS = 6  # six 64-channel Utah arrays, two per region
CHANNELS_PER_ARRAY = 64
N_ELECTRODES_FULL = 384  # all six arrays (6d + 6v + 55b)
N_ELECTRODES_DECODER = 256  # 2x6v + 2x55b speech-decoder subset (6d excluded)
NEURAL_BIN_SIZE_MS = 10  # 10 ms non-overlapping bins
FS_NEURAL = 100  # Hz (downstream training compresses to 20 ms / 50 Hz)
AUDIO_FS = 30000  # Hz (microphone, measured 29,999.995 Hz)
THRESHOLD_SD = -3.5  # ncTX threshold, x per-electrode SD

# Study column order of the neural rows: one contiguous block of 64 per array. It is the
# file order of every session from 2024.05.06 on and the order of the authors' phased-task
# notebooks (v6v, d6v, v55b, d55b, v6d, d6d). 256-channel sessions hold the first four.
ARRAY_ORDER = (
    "ventral_6v",
    "dorsal_6v",
    "ventral_55b",
    "dorsal_55b",
    "ventral_6d",
    "dorsal_6d",
)
ARRAY_REGION = {name: name.split("_", 1)[1] for name in ARRAY_ORDER}
ARRAY_CHANNELS = {
    name: (i * CHANNELS_PER_ARRAY, (i + 1) * CHANNELS_PER_ARRAY)
    for i, name in enumerate(ARRAY_ORDER)
}
AREA_CHANNELS = {"6v": (0, 128), "55b": (128, 256), "6d": (256, 384)}  # Study columns

# 64-column group of each array in the source file, per era. The early layout is the one
# of the authors' Figure 1 notebooks (6d, 6v, 55b), the late layout the one of their
# phased-task notebooks. The ventral/dorsal identity of the early files follows the
# measured group correspondence (README, Channel order). 256-channel sessions store the
# 6v and 55b arrays in Study order (6d dropped at acquisition).
EARLY_SESSIONS = {"t17.2024.03.25", "t17.2024.03.26", "t17.2024.04.02"}
SESSIONS_256 = {"t17.2024.04.15", "t17.2024.04.16", "t17.2024.04.17"}
FILE_GROUPS = {
    "early": {
        "ventral_6v": 3,
        "dorsal_6v": 2,
        "ventral_55b": 5,
        "dorsal_55b": 4,
        "ventral_6d": 0,
        "dorsal_6d": 1,
    },
    "late": {name: i for i, name in enumerate(ARRAY_ORDER)},
}

PARTICIPANT_ID = "braingate2/T17"  # participants.py registry key, implant 2024-02-29
DRYAD_DOI = "10.5061/dryad.vq83bk481"
DRYAD_FILES = ("Data.zip", "README.md")  # version 8, Data.zip unpacks to t17/

# Block -> task family, transcribed from the Dryad README block tables (the .mat
# block_description field is a generic string). Block 6 of 2024.05.23 is absent from the
# tables: a second phased short-sentence block over the same ten cues as block 3, with a
# read phase of about 4 s instead of 8 s. Every other file on disk is in EXCLUDED_BLOCKS.
SESSION_BLOCKS = {
    "t17.2024.03.25": {
        **{b: "phoneme" for b in range(5, 14)},
        **{b: "orofacial" for b in range(14, 19)},
    },
    "t17.2024.03.26": {
        **{b: "word" for b in (10, 14, 15, 16, 17, 19, 20, 21, 22)},
        **{b: "sentence_50w_ol" for b in (23, 24, 25, 26)},
        27: "sentence_50w_cl",
    },
    "t17.2024.04.02": {
        **{b: "sentence_50w_ol" for b in (1, 2, 3, 4)},
        5: "sentence_50w_cl",
        **{b: "sentence_125k_ol" for b in (6, 7, 8, 9, 10, 11, 12)},
        13: "sentence_125k_cl",
    },
    "t17.2024.04.15": {b: "sentence_125k_ol" for b in range(5, 14)},
    "t17.2024.04.16": {b: "sentence_125k_ol" for b in range(9, 14)},
    "t17.2024.04.17": {
        **{b: "sentence_125k_ol" for b in (1, 2, 3, 4, 5, 6)},
        **{b: "sentence_125k_cl" for b in (7, 8, 10)},
    },
    "t17.2024.05.06": {b: "sentence_125k_ol" for b in (1, 2, 4, 5, 6, 7, 8, 9, 10)},
    "t17.2024.05.07": {
        **{b: "sentence_125k_ol" for b in (1, 2, 3, 4, 5)},
        **{b: "sentence_125k_cl" for b in (7, 8, 9, 10, 11)},
    },
    "t17.2024.05.13": {11: "sentence_125k_ol", 12: "sentence_125k_ol"},
    "t17.2024.05.14": {
        1: "sentence_125k_ol",
        2: "sentence_125k_ol",
        **{b: "sentence_125k_cl" for b in (3, 6, 7, 8, 9)},
    },
    "t17.2024.05.23": {
        1: "phased_phoneme",
        2: "phased_word",
        3: "phased_short_sentence",
        5: "phased_sentence",
        6: "phased_short_sentence",
    },
    "t17.2024.05.28": {
        **{b: "sentence_125k_ol" for b in (1, 2, 3, 4, 6, 7)},
        8: "sentence_125k_cl",
        9: "sentence_125k_cl",
    },
    "t17.2024.08.12": {
        **{b: "sentence_50w_ol" for b in (1, 3, 4, 5, 6, 7)},
        **{b: "sentence_50w_cl" for b in (8, 9, 10)},
    },
    "t17.2024.08.13": {
        **{b: "sentence_50w_ol" for b in (1, 2, 3, 5)},
        **{b: "sentence_50w_cl" for b in (6, 7, 8)},
    },
}

# Files on disk the Study does not catalogue, with the reason. Every .mat under t17/ is
# in SESSION_BLOCKS or here, the Study raises on one that is in neither.
_REFERENCE = "reference block, one DO_NOTHING trial, used to fit LRR and thresholds"
_DIAGNOSTIC = "diagnostic speech block over 8 reference cues, not a task block"
_UNREADABLE = "unreadable, truncated file"
EXCLUDED_BLOCKS = {
    "t17.2024.03.26": {18: "aborted word block, 2 trials"},
    "t17.2024.04.02": {0: _DIAGNOSTIC},
    "t17.2024.04.17": {0: _DIAGNOSTIC, 9: _DIAGNOSTIC},
    "t17.2024.05.06": {0: _REFERENCE, 3: _REFERENCE},
    "t17.2024.05.07": {0: _REFERENCE, 6: _REFERENCE},
    "t17.2024.05.14": {0: _REFERENCE, 4: _REFERENCE},
    "t17.2024.05.23": {0: _REFERENCE, 4: _REFERENCE, 7: _UNREADABLE},
    "t17.2024.05.28": {0: _REFERENCE, 5: _REFERENCE},
    "t17.2024.08.12": {0: _REFERENCE, 2: _REFERENCE},
    "t17.2024.08.13": {0: _REFERENCE, 4: _REFERENCE},
}
for _session, _blocks in EXCLUDED_BLOCKS.items():
    _both = set(_blocks) & set(SESSION_BLOCKS.get(_session, {}))
    if _both:
        raise ValueError(
            f"{_session} blocks {sorted(_both)} are catalogued and excluded"
        )

# 40-class phoneme target set (ARPAbet + SIL), PHONE_DEF_SIL in the companion loader.
# The speech RNN emits a 41-d logit = these 40 + a CTC blank (index 0).
PHONE_DEF_SIL = (
    "AA",
    "AE",
    "AH",
    "AO",
    "AW",
    "AY",
    "B",
    "CH",
    "D",
    "DH",
    "EH",
    "ER",
    "EY",
    "F",
    "G",
    "HH",
    "IH",
    "IY",
    "JH",
    "K",
    "L",
    "M",
    "N",
    "NG",
    "OW",
    "OY",
    "P",
    "R",
    "S",
    "SH",
    "T",
    "TH",
    "UH",
    "UW",
    "V",
    "W",
    "Y",
    "Z",
    "ZH",
    "SIL",
)

# Screen cue -> target phoneme of the isolated-phoneme blocks (the authors' GNB notebook
# map). Consonants were cued as consonant + AA syllables, vowels inside a carrier word
# with the vowel capitalised. ``Ah`` and ``Oy`` occur only in the phased session.
PHONEME_CUES = {
    "Pah": "P",
    "Bah": "B",
    "Mah": "M",
    "Fah": "F",
    "Vah": "V",
    "Thah": "TH",
    "THe": "DH",
    "Tah": "T",
    "Dah": "D",
    "Nah": "N",
    "Sah": "S",
    "Zah": "Z",
    "Chah": "CH",
    "Jah": "JH",
    "Shah": "SH",
    "Zhah": "ZH",
    "Ngah": "NG",
    "Kah": "K",
    "Gah": "G",
    "Yah": "Y",
    "Lah": "L",
    "Rah": "R",
    "Wah": "W",
    "Hah": "HH",
    "chOIce": "OY",
    "gOAt": "OW",
    "thOUGHt": "AO",
    "lOt": "AA",
    "mOUth": "AW",
    "prIce": "AY",
    "trAp": "AE",
    "drEss": "EH",
    "fAce": "EY",
    "flEEce": "IY",
    "kIt": "IH",
    "strUt": "AH",
    "gOOse": "UW",
    "LettER": "ER",
    "fOOt": "UH",
    "Ah": "AA",
    "Oy": "OY",
}
# Rest cue of the phoneme blocks and of the orofacial blocks, emitted as an ``Action``
# with ``action_kind="rest"``, never as a phoneme.
REST_CUES = ("DO_NOTHING", "Do Nothing")
# The 12 orofacial movements, in the authors' order, plus the four syllable cues the
# orofacial blocks interleave (the paper's 13-class analysis is the 12 plus rest).
OROFACIAL_CUES = (
    "Clench jaw",
    "Open jaw",
    "Frown",
    "Smile",
    "Pucker lips",
    "Tuck lips",
    "Hum a HIGH frequency",
    "Hum a LOW frequency",
    "Push your tongue behind your LOWER teeth",
    "Push your tongue behind your UPPER teeth",
    "Push tongue to FRONT roof of your mouth",
    "Push tongue to BACK roof of your mouth",
)
SAY_CUES = {"Say Dah": "D AA", "Say Fah": "F AA", "Say Jah": "JH AA", "Say Yah": "Y AA"}
# The 50-word vocabulary (the cues of every word block, and the 50w sentence blocks).
WORD_CUES = (
    "AM", "ARE", "BAD", "BRING", "CLEAN", "CLOSER", "COMFORTABLE", "COMING", "COMPUTER",
    "DO", "FAITH", "FAMILY", "FEEL", "GLASSES", "GOING", "GOOD", "GOODBYE", "HAVE", "HELLO",
    "HELP", "HERE", "HOPE", "HOW", "HUNGRY", "I", "IS", "IT", "LIKE", "MUSIC", "MY", "NEED",
    "NO", "NOT", "NURSE", "OKAY", "OUTSIDE", "PLEASE", "RIGHT", "SUCCESS", "TELL", "THAT",
    "THEY", "THIRSTY", "TIRED", "UP", "VERY", "WHAT", "WHERE", "YES", "YOU",
)  # fmt: skip

# Dataset-wide class ids of the cued conditions (``code`` on Action and Stimulus rows):
# 1-39 the target phoneme (its id in ``text.PhonemeVocab.canonical()``), 40 rest, 41-52 the
# orofacial movements, 53-56 the syllable cues, 101-150 the 50-word vocabulary. Sentence
# cues have no class and keep the neuralset default.
NO_CODE = -100
CUE_CODES: dict[str, int] = {}
CUE_CODES.update({cue: ARPABET_V1.index(sym) + 1 for cue, sym in PHONEME_CUES.items()})
CUE_CODES.update({cue: 40 for cue in REST_CUES})
CUE_CODES.update({cue: 41 + i for i, cue in enumerate(OROFACIAL_CUES)})
CUE_CODES.update({cue: 53 + i for i, cue in enumerate(SAY_CUES)})
CUE_CODES.update({cue: 101 + i for i, cue in enumerate(WORD_CUES)})


def cue_code(cue: str) -> int:
    """Dataset-wide class id of a screen cue, ``NO_CODE`` for a sentence."""
    return CUE_CODES.get(cue, CUE_CODES.get(cue.upper(), NO_CODE))


# Phased-task (2024.05.23) phase windows: (start_marker, end_marker) nsp_neural_time fields.
# Each trial also carries ``imagined_delay`` -> ``imagined_go_period`` and
# ``vocalized_delay_start`` -> ``vocalized_go_cue`` delays (2 s each) and an
# ``intertrial_start`` marker 2 s before ``cue_presentation``.
# --- Phoneme targets for sentence and word cues ----------------------------------------
# assets/cue_phonemes.json: cue text -> ARPAbet target, the prompt's dictionary
# pronunciation (text.g2p_phonemes, the authors' g2p_en recipe with "|" boundaries).
CUE_PHONEMES = PhonemeTable(
    Path(__file__).resolve().parent / "assets" / "cue_phonemes.json"
)


def build_cue_phonemes(data_dir: str | Path = T17_DIR) -> PhonemeTable:
    """Regenerate ``assets/cue_phonemes.json`` over every sentence and word cue of the
    catalogued blocks. Needs the ``g2p`` extra (``uv pip install -e ".[g2p]"``)."""
    cues: set[str] = set()
    for session, blocks in SESSION_BLOCKS.items():
        for block, fam in blocks.items():
            if "sentence" in fam or "word" in fam:
                cues.update(get_trials(block_path(session, block, data_dir)).cue)
    return PhonemeTable.build(
        cues,
        CUE_PHONEMES.path,
        description="Dictionary pronunciation of each sentence and word cue, not a "
        "measurement of what T17 produced. The authors' g2p_en recipe "
        "(mat_to_tfrecord.py), stress removed, '|' at every word boundary and at the "
        "end (their SIL). Symbols are PhonemeVocab v1 (neuralshed.text.ARPABET_V1).",
    )


PHASE_WINDOWS = {
    "read": ("cue_presentation_nsp_neural_time", "imagined_delay_nsp_neural_time"),
    "imagined": (
        "imagined_go_period_nsp_neural_time",
        "vocalized_delay_start_nsp_neural_time",
    ),
    "attempted": ("vocalized_go_cue_nsp_neural_time", "trial_end_nsp_neural_time"),
}

# Trial-level (one value per trial) fields loaded by get_trials, besides the
# ``*_nsp_neural_time`` markers (three in every block, nine in the phased session).
_TRIAL_FIELDS = ("cue", "delay_duration_ms", "inter_trial_duration_ms")
_TRIAL_FLAGS = ("trial_paused_by_CNRA", "trial_timed_out")
_MARKER_SUFFIX = "_nsp_neural_time"


# --- Session / block discovery -------------------------------------------------
def list_sessions(data_dir: str | Path = T17_DIR) -> list[str]:
    """All session names (``t17.YYYY.MM.DD``), sorted chronologically."""
    data_dir = Path(data_dir)
    return sorted(
        p.name for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("t17.2")
    )


def session_date(session: str) -> str:
    """ISO date from a session name (``t17.2024.04.17`` -> ``2024-04-17``)."""
    return session.removeprefix("t17.").replace(".", "-")


def post_implant_day(session: str) -> int:
    """Days from T17's implant (participant registry) to a session's date."""
    day = participants.post_implant_day(PARTICIPANT_ID, session_date(session))
    if day is None:
        raise ValueError(f"no implant date for {PARTICIPANT_ID}")
    return day


def session_dir(session: str, data_dir: str | Path = T17_DIR) -> Path:
    """Path to a session's ``RedisMat`` folder (where the ``.mat`` blocks live)."""
    return Path(data_dir) / session / "brainToText" / "RedisMat"


def _block_num(path: Path) -> int:
    """Block number parsed from a ``..._(<n>).mat`` filename."""
    return int(path.stem.rsplit("(", 1)[1].rstrip(")"))


def list_blocks(session: str, data_dir: str | Path = T17_DIR) -> list[tuple[int, Path]]:
    """``(block_number, path)`` for every ``.mat`` block in a session, sorted by block."""
    rmdir = session_dir(session, data_dir)
    mats = [p for p in rmdir.glob("*.mat") if not p.name.startswith("._")]
    return sorted(((_block_num(p), p) for p in mats), key=lambda t: t[0])


def block_path(session: str, block: int, data_dir: str | Path = T17_DIR) -> Path:
    """Path to one specific block's ``.mat`` file."""
    for b, p in list_blocks(session, data_dir):
        if b == block:
            return p
    raise FileNotFoundError(f"block {block} not found in {session}")


def task_of(session: str, block: int) -> str:
    """Task family of a catalogued (session, block), ``"unknown"`` for an excluded one."""
    return SESSION_BLOCKS.get(session, {}).get(block, "unknown")


def exclusion_reason(session: str, block: int) -> str | None:
    """Why a (session, block) is not catalogued, ``None`` when it is."""
    return EXCLUDED_BLOCKS.get(session, {}).get(block)


# --- Deposit verification ------------------------------------------------------
def expected_blocks() -> dict[str, set[int]]:
    """Every block file the deposit holds, catalogued or excluded, per session."""
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
    exactly the block files in ``SESSION_BLOCKS`` and ``EXCLUDED_BLOCKS``, and that each
    file except the one catalogued as unreadable opens with ``scipy.io.whosmat``.
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
    t17 = data_dir / "t17"
    for session, blocks in sorted(expected_blocks().items()):
        if not session_dir(session, t17).is_dir():
            problems.append(f"{session} missing")
            continue
        found = dict(list_blocks(session, t17))
        if missing := sorted(blocks - set(found)):
            problems.append(f"{session} missing blocks {missing}")
        if extra := sorted(set(found) - blocks):
            problems.append(f"{session} has uncatalogued blocks {extra}")
        for block in sorted(blocks & set(found)):
            if exclusion_reason(session, block) == _UNREADABLE:
                continue
            try:
                scipy.io.whosmat(str(found[block]))
            except Exception as error:
                problems.append(f"{session} block {block} unreadable: {error}")
    return problems


# --- Per-block readers ---------------------------------------------------------
def n_channels(path: str | Path) -> int:
    """Recorded channel count of a block (384 or 256) without loading the arrays."""
    for name, shape, _ in scipy.io.whosmat(str(path)):
        if name == "binned_neural_threshold_crossings":
            return int(shape[1])
    raise KeyError("binned_neural_threshold_crossings not in file")


def session_n_channels(session: str) -> int:
    """Channel count for a session (256 for 04.15/04.16/04.17, else 384)."""
    return N_ELECTRODES_DECODER if session in SESSIONS_256 else N_ELECTRODES_FULL


def session_arrays(session: str) -> tuple[str, ...]:
    """Arrays recorded in a session, in Study order (four on the 256-channel days)."""
    return ARRAY_ORDER[:4] if session in SESSIONS_256 else ARRAY_ORDER


def array_channels(session: str) -> dict[str, tuple[int, int]]:
    """Array -> half-open Study column range for the arrays a session recorded."""
    return {name: ARRAY_CHANNELS[name] for name in session_arrays(session)}


def area_channels(session: str) -> dict[str, tuple[int, int]]:
    """Area -> half-open Study column range for the areas a session recorded."""
    areas = ("6v", "55b") if session in SESSIONS_256 else ("6v", "55b", "6d")
    return {area: AREA_CHANNELS[area] for area in areas}


def channel_order(session: str) -> np.ndarray:
    """File column of each Study column, so ``data[:, channel_order(s)]`` is Study order."""
    groups = FILE_GROUPS["early" if session in EARLY_SESSIONS else "late"]
    order = [
        groups[name] * CHANNELS_PER_ARRAY + k
        for name in session_arrays(session)
        for k in range(CHANNELS_PER_ARRAY)
    ]
    return np.asarray(order)


def to_study_order(data: np.ndarray, session: str) -> np.ndarray:
    """Permute a ``(n_bins, n_ch)`` file-order array into Study column order."""
    order = channel_order(session)
    if data.shape[1] != order.size:
        raise ValueError(
            f"{session}: {data.shape[1]} columns, expected {order.size} for its arrays"
        )
    return data[:, order]


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
    """Return ``(audio, fs)``: 1-D int16 microphone samples and :data:`AUDIO_FS`."""
    d = read_block(path, fields=["microphone_data"])
    return np.asarray(d["microphone_data"]).ravel(), AUDIO_FS


def get_trials(path: str | Path):
    """Per-trial table (one row per cue) as a DataFrame.

    Columns: ``cue`` (stripped), ``delay_ms``, ``iti_ms``, ``paused``, ``timed_out``,
    and one ``<marker>_ns`` column per ``*_nsp_neural_time`` field in the file
    (``trial_start_ns``, ``go_cue_ns``, ``trial_end_ns`` everywhere, plus the six phased
    markers on 2024.05.23). The two flag fields must be present.
    """
    import pandas as pd

    avail = {n for n, _, _ in scipy.io.whosmat(str(path))}
    missing = [f for f in (*_TRIAL_FIELDS, *_TRIAL_FLAGS) if f not in avail]
    if missing:
        raise KeyError(f"{path}: missing trial fields {missing}")
    markers = sorted(n for n in avail if n.endswith(_MARKER_SUFFIX))
    d = read_block(path, fields=[*_TRIAL_FIELDS, *_TRIAL_FLAGS, *markers])

    cue = np.atleast_1d(d["cue"])
    out = {"cue": [str(c).strip() for c in cue]}
    n = len(out["cue"])

    def col(key):
        return np.atleast_1d(d[key]).astype(np.float64).ravel()[:n]

    out["delay_ms"] = col("delay_duration_ms")
    out["iti_ms"] = col("inter_trial_duration_ms")
    out["paused"] = col("trial_paused_by_CNRA").astype(bool)
    out["timed_out"] = col("trial_timed_out").astype(bool)
    for f in markers:
        out[f.removesuffix(_MARKER_SUFFIX) + "_ns"] = col(f)
    return pd.DataFrame(out)


# --- Trial / phase segmentation ------------------------------------------------
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

    The companion sentence loader uses ``pre_go_bins=450`` (4.5 s of pre-GO context).
    """
    start = bin_index(bin_times_ns, go_cue_ns) - pre_go_bins
    end = bin_index(bin_times_ns, trial_end_ns)
    return max(start, 0), end


def phase_windows(
    trials, bin_times_ns: np.ndarray, trial_idx: int
) -> dict[str, tuple[int, int]]:
    """Phased-task (2024.05.23) per-trial bin windows for read/imagined/attempted phases.

    ``trials`` is the :func:`get_trials` DataFrame; requires the phase-marker columns.
    """
    row = trials.iloc[trial_idx]
    out = {}
    for phase, (start_f, end_f) in PHASE_WINDOWS.items():
        s = row.get(start_f.replace("_nsp_neural_time", "_ns"))
        e = row.get(end_f.replace("_nsp_neural_time", "_ns"))
        if s is None or e is None or np.isnan(s) or np.isnan(e):
            continue
        out[phase] = (bin_index(bin_times_ns, s), bin_index(bin_times_ns, e))
    return out


if __name__ == "__main__":
    sessions = list_sessions()
    print(
        f"{len(sessions)} sessions, {sum(len(list_blocks(s)) for s in sessions)} blocks"
    )
    for s in sessions:
        blocks = list_blocks(s)
        tasks = sorted({task_of(s, b) for b, _ in blocks})
        print(
            f"  {s} (day {post_implant_day(s):>3}) {len(blocks):>2} blocks, "
            f"{session_n_channels(s)} ch :: {', '.join(tasks)}"
        )

    s0, b0 = "t17.2024.04.17", 4
    p = block_path(s0, b0)
    print(f"\n{s0} block {b0} [{task_of(s0, b0)}]  n_ch={n_channels(p)}")
    tx, sbp = (to_study_order(x, s0) for x in get_neural(p))
    print(
        f"  threshold_crossings {tx.shape} {tx.dtype} | spike_band_power {sbp.shape} {sbp.dtype}"
    )
    trials = get_trials(p)
    print(f"  {len(trials)} trials; cue[0]={trials.cue[0]!r}")
    audio, fs = get_audio(p)
    print(f"  audio {audio.shape} @ {fs} Hz")
