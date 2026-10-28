"""Source-reading utilities for the Wairagkar et al. 2025 brain-to-voice dataset.

This reads the **`t15_neural_data`** release (the official, full dataset: 37 sessions,
~8,500 attempted-speech trials, open vocabulary) directly from the raw Dryad ``.mat``
files (no NWB). Each session day folder holds a triplet:

* ``t15_dayXXX_neural_data.mat``      — neural features (spikepow + threshcross), cues,
  per-trial ``trial_description``, and (evaluation sessions only) the closed-loop
  synthesized audio + the decoder name;
* ``t15_dayXXX_speech_indices.mat``   — word & syllable onset/offset indices (100 Hz);
* ``t15_dayXXX_target_speech_feats.mat`` — the **LPCNet acoustic-feature target**
  (20-dim), time-aligned to the neural bins; word- and syllable-aligned variants.

Nine later "evaluation" sessions also ship a per-session causal ``brain2voice_*.h5``
decoder (trained on all prior sessions). A top-level ``test_idx.mat`` lists the
human-listener intelligibility benchmark trials (global, 0-based, sorted-day order).

The smaller 50-word ``t15_control_experiment`` release (single session,
with stethoscope mic + IMU but **no** LPCNet target) is read by the sibling
``control_study.Wairagkar2025ControlExperiment``; its accessors are at the
bottom of this module.

Verified against ``t15_neural_data/t15_day025/*`` (and siblings).
"""

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import scipy.io as sio

from neuralshed.datasets.utils import dataset_dir
from neuralshed.text import PhonemeTable

# --- Constants -----------------------------------------------------------------
N_ELECTRODES = 256  # 4 Utah arrays x 64 electrodes
ARRAY_ORDER = ("ventral_6v", "4", "55b", "dorsal_6v")  # order of the 256-wide axis
NEURAL_BIN_SIZE_MS = 10  # 10 ms non-overlapping bins
FS_NEURAL = 100  # Hz (neural feature sampling rate)

# LPCNet acoustic target: 18 Bark-freq cepstral coefficients + 2 pitch/voicing features.
LPCNET_FEATURE_DIM = 20
LPCNET_DIMS = tuple(f"bfcc_{i}" for i in range(18)) + ("pitch_period", "pitch_corr")
SYNTH_AUDIO_FS = 16000  # Hz (closed-loop synthesized voice; eval sessions only)

DATA_SUBDIR = "t15_neural_data"
# Data lives under the Study's class-name folder (the Study base appends it to `path`),
# so utils and the Study resolve the same location (card_2024 / kunz_2025 convention).
DEFAULT_DATA_DIR = (
    dataset_dir("wairagkar_2025") / "data" / "Wairagkar2025Brain2Voice" / DATA_SUBDIR
)

# The control release arrives inside the *main* study's download folder (one
# Dryad download, two subfolders), hence the Wairagkar2025Brain2Voice parent.
CONTROL_SUBDIR = "t15_control_experiment"
DEFAULT_CONTROL_DIR = (
    dataset_dir("wairagkar_2025") / "data" / "Wairagkar2025Brain2Voice" / CONTROL_SUBDIR
)

_DAY_RE = re.compile(r"t15_day(?P<day>\d+)$")
_DECODER_RE = re.compile(r"brain2voice_t15_day(?P<day>\d+)\.h5$")


# --- Cached loaders (one heavy neural file per session at a time) --------------
@lru_cache(maxsize=2)
def _load_neural(path: str) -> dict:
    """Full neural_data ``.mat`` (heavy: 0.6–1.1 GB); bounded to 2 sessions in RAM."""
    return sio.loadmat(path, squeeze_me=True, struct_as_record=False)


@lru_cache(maxsize=8)
def _load_meta(path: str) -> dict:
    """Just the light metadata variables from a neural_data ``.mat``."""
    return sio.loadmat(
        path,
        variable_names=["session", "cues", "trial_description", "brain2voice_decoder"],
        squeeze_me=True,
    )


@lru_cache(maxsize=4)
def _load_feats(path: str) -> dict:
    """Target LPCNet features ``.mat`` (medium: ~50–90 MB)."""
    return sio.loadmat(path, squeeze_me=True, struct_as_record=False)


@lru_cache(maxsize=8)
def _load_indices(path: str) -> dict:
    """Word/syllable speech-index ``.mat`` (tiny)."""
    return sio.loadmat(path, squeeze_me=True, struct_as_record=False)


# --- Session discovery ---------------------------------------------------------
@dataclass(frozen=True)
class SessionRef:
    """One recording session (day) on disk and its companion files."""

    day: int
    dir: Path

    @property
    def session(self) -> str:
        return f"t15_day{self.day:03d}"

    @property
    def neural_path(self) -> Path:
        return self.dir / f"{self.session}_neural_data.mat"

    @property
    def indices_path(self) -> Path:
        return self.dir / f"{self.session}_speech_indices.mat"

    @property
    def feats_path(self) -> Path:
        return self.dir / f"{self.session}_target_speech_feats.mat"

    @property
    def decoder_path(self) -> Path | None:
        """The per-session causal ``brain2voice_*.h5`` (evaluation sessions only)."""
        hits = [
            p
            for p in self.dir.glob("brain2voice_t15_day*.h5")
            if not p.name.startswith("._")
        ]
        return hits[0] if hits else None

    @property
    def is_eval(self) -> bool:
        return self.decoder_path is not None


def list_sessions(data_dir: str | Path = DEFAULT_DATA_DIR) -> list[SessionRef]:
    """All session folders, sorted by post-implant day."""
    data_dir = Path(data_dir)
    out = []
    for d in data_dir.iterdir():
        m = _DAY_RE.match(d.name)
        if d.is_dir() and m:
            out.append(SessionRef(day=int(m.group("day")), dir=d))
    return sorted(out, key=lambda s: s.day)


def load_test_idx(data_dir: str | Path = DEFAULT_DATA_DIR) -> dict:
    """Benchmark trial indices (global, 0-based, sorted-day order).

    Returns ``{"random": ndarray, "video": ndarray}`` — the human-listener intelligibility
    test set and the paper's video-example trials.
    """
    m = sio.loadmat(str(Path(data_dir) / "test_idx.mat"), squeeze_me=True)
    return {
        "random": np.atleast_1d(m["random_test_idx"]).astype(int),
        "video": np.atleast_1d(m["video_test_idx"]).astype(int),
    }


# --- Per-session / per-trial accessors -----------------------------------------
def get_cues(ref: SessionRef) -> list[str]:
    """Text cue (sentence) shown on each trial, trimmed."""
    return [
        str(c).strip() for c in np.atleast_1d(_load_meta(str(ref.neural_path))["cues"])
    ]


def get_trial_descriptions(ref: SessionRef) -> list[str]:
    """Per-trial ``trial_description`` (e.g. 'Training data', 'Evaluation trial')."""
    return [
        str(d).strip()
        for d in np.atleast_1d(_load_meta(str(ref.neural_path))["trial_description"])
    ]


def n_trials(ref: SessionRef) -> int:
    return len(get_cues(ref))


def get_neural_features(
    ref: SessionRef, trial: int, feature: str = "both"
) -> np.ndarray:
    """Neural features for one trial: ``[N x 256]`` (single stream) or ``[N x 512]`` (both)."""
    nd = _load_neural(str(ref.neural_path))
    sp = np.asarray(np.atleast_1d(nd["spikepow_trials"])[trial])
    tc = np.asarray(np.atleast_1d(nd["threshcross_trials"])[trial])
    if feature == "spikepow":
        return sp
    if feature == "threshcross":
        return tc
    if feature == "both":
        return np.concatenate([sp, tc], axis=1)
    raise ValueError(f"unknown feature {feature!r}")


def get_target_feats(
    ref: SessionRef, trial: int, alignment: str = "word"
) -> np.ndarray:
    """LPCNet acoustic-feature target for one trial, ``[N x 20]`` (aligned to neural bins).

    ``alignment`` selects word- or syllable-aligned synthesis timing.
    """
    key = f"target_speech_feats_{alignment}"
    return np.asarray(np.atleast_1d(_load_feats(str(ref.feats_path))[key])[trial])


def get_word_indices(ref: SessionRef, trial: int) -> np.ndarray:
    """Per-word ``[n_words x 2]`` onset/offset indices (100 Hz) of attempted speech."""
    return np.asarray(
        np.atleast_1d(_load_indices(str(ref.indices_path))["word_indices"])[trial]
    )


def get_syllable_indices(ref: SessionRef, trial: int) -> np.ndarray:
    """Per-syllable ``[n_syl x 2]`` onset/offset indices (100 Hz)."""
    return np.asarray(
        np.atleast_1d(_load_indices(str(ref.indices_path))["syl_indices"])[trial]
    )


def get_pred_audio(ref: SessionRef, trial: int) -> np.ndarray | None:
    """Closed-loop synthesized voice (16 kHz) for an evaluation trial, else ``None``.

    Present only in evaluation sessions, and empty for the session's training trials.
    """
    nd = _load_neural(str(ref.neural_path))
    if "brain2voice_pred_audio_16kHz" not in nd:
        return None
    wf = np.asarray(np.atleast_1d(nd["brain2voice_pred_audio_16kHz"])[trial]).ravel()
    return wf if wf.size else None


def split_of(trial_description: str) -> str:
    """Coarse split from ``trial_description``: ``'eval'`` for evaluation trials else ``'train'``."""
    return "eval" if "evaluation" in trial_description.lower() else "train"


# assets/cue_phonemes.json: cue text -> ARPAbet target for the 240 cues of the
# control experiment, which ``control_study.Wairagkar2025ControlExperiment``
# attaches to its Sentence rows. g2p_en runs on each word of the cue on its own,
# punctuation kept (``per_word=True``), so a trailing "?" or "." adds a second "|".
CUE_PHONEMES = PhonemeTable(
    Path(__file__).resolve().parent / "assets" / "cue_phonemes.json"
)


def build_cue_phonemes(
    data_dir: str | Path = DEFAULT_CONTROL_DIR, path: str | Path | None = None
) -> PhonemeTable:
    """Regenerate ``assets/cue_phonemes.json`` (or ``path``) over every cue of the
    control-experiment block files in ``data_dir``, stripped as the Study emits them.
    Needs the ``g2p`` extra."""
    cues = {s for ref in list_control_blocks(data_dir) for s in get_control_sentences(ref)}
    return PhonemeTable.build(
        {s for s in cues if s},
        path or CUE_PHONEMES.path,
        description="Dictionary pronunciation of each cue of the t15_control_experiment "
        "release, not a measurement of what T15 produced. g2p_en run on each word of "
        "the cue separately with punctuation kept (per_word=true), so a trailing '?' "
        "or '.' adds a second '|'; stress removed, '|' after every word. Symbols are "
        "PhonemeVocab v1 (neuralshed.text.ARPABET_V1).",
        per_word=True,
        join_split_apostrophes=False,
    )


# --- Control experiment (t15_control_experiment) ---------------------------------------
# One session (t15_day482), 10 blocks x 24 trials, 50-word vocabulary per block file.
# The neural features use the same online pipeline and the same variable names as the
# full release, but each block additionally records T15's residual speech with a
# stethoscopic microphone (30 kHz) and a 6-axis IMU (~242 Hz as shipped, 300 Hz
# nominal).

_CONTROL_FILE_RE = re.compile(r"(?P<session>t15_day\d+)_block(?P<block>\d+)\.mat")


@lru_cache(maxsize=2)
def _load_control_block(path: str) -> dict:
    """One control-experiment block ``.mat`` (heavy: ~100 MB); at most 2 in RAM."""
    return sio.loadmat(path, squeeze_me=True, struct_as_record=False)


@dataclass(frozen=True)
class BlockRef:
    """One recording block of the control experiment on disk."""

    session: str  # e.g. ``t15_day482``
    block: int
    path: Path

    @property
    def file_stem(self) -> str:
        return f"{self.session}_block{self.block:02d}"

    @property
    def decoder_path(self) -> Path | None:
        """The ``brain2voice_*.h5`` trained on the *other* blocks with this one held
        out (the online system's output, not a decode target)."""
        hits = [
            p
            for p in self.path.parent.glob(f"brain2voice_{self.file_stem}*.h5")
            if not p.name.startswith("._")
        ]
        return hits[0] if hits else None


def list_control_blocks(
    data_dir: str | Path = DEFAULT_CONTROL_DIR,
) -> list[BlockRef]:
    """All block files of the release folder, sorted by session then block number."""
    out = []
    for p in sorted(Path(data_dir).glob("t15_day*_block*.mat")):
        m = _CONTROL_FILE_RE.fullmatch(p.name)
        if m:
            out.append(
                BlockRef(
                    session=m.group("session"), block=int(m.group("block")), path=p
                )
            )
    return sorted(out, key=lambda r: (r.session, r.block))


def get_control_sentences(ref: BlockRef) -> list[str]:
    """Text cue displayed on each trial, trimmed (the shipped strings pad with spaces)."""
    return [
        str(s).strip()
        for s in np.atleast_1d(_load_control_block(str(ref.path))["sentences"])
    ]


def get_block_description(ref: BlockRef) -> str:
    """The block's verbatim ``block_description`` as shipped in the file."""
    return str(_load_control_block(str(ref.path))["block_description"])


def get_control_features(ref: BlockRef, trial: int, key: str) -> np.ndarray:
    """One trial's stored feature array, verbatim (no re-normalization here).

    ``key`` selects the stream: ``spikepow_trials`` / ``threshcross_trials``
    (``[N x 256]`` at ``Fs_neural`` = 100 Hz), ``stethoscope_trials``
    (``[T x 1]`` at ``Fs_steth`` = 30 kHz) or ``imu_trials`` (``[t x 6]`` at
    ``Fs_imu``, ~242 Hz as shipped).
    """
    return np.asarray(np.atleast_1d(_load_control_block(str(ref.path))[key])[trial])


def get_stethoscope(ref: BlockRef, trial: int) -> np.ndarray:
    """T15's residual speech from the stethoscopic microphone, ``[T]`` at 30 kHz."""
    return get_control_features(ref, trial, "stethoscope_trials").ravel()


def get_imu(ref: BlockRef, trial: int) -> np.ndarray:
    """6-axis head motion ``[t x 6]`` (~242 Hz): accelerometer XYZ, gyroscope XYZ."""
    return np.atleast_2d(get_control_features(ref, trial, "imu_trials"))


if __name__ == "__main__":
    sessions = list_sessions()
    print(
        f"{len(sessions)} sessions | eval sessions: {sum(s.is_eval for s in sessions)}"
    )
    total = sum(n_trials(s) for s in sessions)
    print(f"total trials: {total}")
    s0 = sessions[0]
    print(f"\n{s0.session}: {n_trials(s0)} trials | eval={s0.is_eval}")
    print("  cue[0]      :", repr(get_cues(s0)[0]))
    print("  descr[0]    :", get_trial_descriptions(s0)[0])
    print("  neural[0]   :", get_neural_features(s0, 0).shape, "(N x 512)")
    print("  target[0]   :", get_target_feats(s0, 0).shape, "(N x 20)")
    print("  word_idx[0] :", get_word_indices(s0, 0).shape, "(n_words x 2)")
    ti = load_test_idx()
    print("  test_idx    :", {k: v.shape for k, v in ti.items()})
