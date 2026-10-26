from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from neuralshed.datasets import participants
from neuralshed.datasets.utils import dataset_dir

# --- Constants -----------------------------------------------------------------
# Data lands in the framework study-folder (.../data/Card2024Brain2Text/); the Dryad
# neural-data zip nests the session dirs one level deeper (see sessions_dir).
DATA_DIR = dataset_dir("card_2024") / "data" / "Card2024Brain2Text"
PARTICIPANT_ID = "braingate2/T15"  # datasets/participants.py

N_ELECTRODES = 256  # 4 Utah arrays x 64 electrodes
N_ARRAYS = 4
CHANNELS_PER_ARRAY = 64
# Per-array order repeated within each 256-wide feature block ("area 4" == M1; same
# four T15 arrays as wairagkar_2025).
ARRAY_ORDER = ("ventral 6v", "area 4", "55b", "dorsal 6v")

# input_features (512 cols) = 256 threshold crossings (0-255) + 256 spike-band power (256-511).
N_THRESHOLD_CROSSINGS = 256
N_SPIKE_BAND_POWER = 256
N_FEATURES = N_THRESHOLD_CROSSINGS + N_SPIKE_BAND_POWER  # 512
NEURAL_BIN_SIZE_MS = 20  # 20 ms non-overlapping bins
FS_NEURAL = 50  # Hz
SPLITS = ("train", "val", "test")

# seq_class_ids: 0 = BLANK/pad, 1-39 = ARPAbet phonemes, 40 = word boundary "|".
# Verified against LOGIT_TO_PHONEME in nejm-brain-to-text/model_training/evaluate_model_helpers.py.
LOGIT_TO_PHONEME = (
    "BLANK",
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
    "|",
)
PHONEME_LABELS = {i: p for i, p in enumerate(LOGIT_TO_PHONEME) if i >= 1}  # 1..40


def feature_channel_layout() -> dict:
    """Column ranges of ``input_features`` and the per-array channel ordering.

    cols 0-255 = threshold crossings, 256-511 = spike-band power; within each block the
    four arrays (64 ch each) repeat in ``ARRAY_ORDER``. Ranges are half-open (slice-style).
    """
    return {
        "threshold_crossings": (0, N_THRESHOLD_CROSSINGS),
        "spike_band_power": (N_THRESHOLD_CROSSINGS, N_FEATURES),
        "arrays": ARRAY_ORDER,
        "channels_per_array": CHANNELS_PER_ARRAY,
        "bin_size_ms": NEURAL_BIN_SIZE_MS,
        "rate_hz": FS_NEURAL,
    }


# --- Session discovery ---------------------------------------------------------
@dataclass(frozen=True)
class SessionRef:
    """One copy-task session on disk (e.g. ``t15.2023.08.13``) and its split files."""

    session: str
    dir: Path

    @property
    def date(self) -> str:
        """ISO date parsed from the name (``t15.2023.08.13`` -> ``2023-08-13``)."""
        return self.session.removeprefix("t15.").replace(".", "-")

    @property
    def post_implant_day(self) -> int:
        """Days since the implant date in the participant registry."""
        return participants.post_implant_day(PARTICIPANT_ID, self.date)

    def split_path(self, split: str) -> Path:
        return self.dir / f"data_{split}.hdf5"

    @property
    def splits(self) -> tuple[str, ...]:
        """Splits present on disk (train always; val/test on 41/45 sessions)."""
        return tuple(s for s in SPLITS if self.split_path(s).exists())


def sessions_dir(base: str | Path = DATA_DIR) -> Path:
    """Dir holding the ``t15.*`` session folders. The Dryad neural-data zip wraps
    them in ``hdf5_data_final/``; fall back to ``base`` when already flattened."""
    base = Path(base)
    nested = base / "hdf5_data_final"
    return nested if nested.is_dir() else base


def list_sessions(data_dir: str | Path = DATA_DIR) -> list[SessionRef]:
    """All session folders, sorted chronologically by name."""
    data_dir = sessions_dir(data_dir)
    refs = [
        SessionRef(session=p.name, dir=p)
        for p in data_dir.iterdir()
        if p.is_dir() and p.name.startswith("t15.")
    ]
    return sorted(refs, key=lambda r: r.session)


# --- Per-session / per-trial accessors -----------------------------------------
def n_trials(ref: SessionRef, split: str = "train") -> int:
    """Trial count in one split of a session (0 if that split file is absent)."""
    path = ref.split_path(split)
    if not path.exists():
        return 0
    with h5py.File(path, "r") as f:
        return len(f.keys())


@dataclass(frozen=True)
class TrialRef:
    """One trial's source identity: ``block_num``, ``trial_num`` and its HDF5 group name.

    ``(session, block, trial)`` is unique across the release, the group name is not
    (each split file restarts at ``trial_0000``).
    """

    block: int
    trial: int
    key: str


def list_trials(ref: SessionRef, split: str = "train") -> list[TrialRef]:
    """Trials of one split in file order ([] if the split file is absent)."""
    path = ref.split_path(split)
    if not path.exists():
        return []
    with h5py.File(path, "r") as f:
        return [
            TrialRef(int(f[k].attrs["block_num"]), int(f[k].attrs["trial_num"]), k)
            for k in sorted(f.keys())
        ]


def _group(f: h5py.File, block: int, trial: int) -> h5py.Group:
    """The group whose ``block_num``/``trial_num`` attrs match, scanning in file order."""
    for k in sorted(f.keys()):
        a = f[k].attrs
        if int(a["block_num"]) == block and int(a["trial_num"]) == trial:
            return f[k]
    raise KeyError(
        f"no trial with block_num={block}, trial_num={trial} in {f.filename}"
    )


def _trial_from_group(g) -> dict:
    out = {"input_features": g["input_features"][:], "attrs": dict(g.attrs)}
    for k in ("seq_class_ids", "transcription"):  # train/val only
        if k in g:
            out[k] = g[k][:]
    return out


def read_trial(ref: SessionRef, split: str, block: int, trial: int) -> dict:
    """One trial: ``input_features`` (n_time_steps, 512) + ``attrs`` (+ labels on train/val)."""
    with h5py.File(ref.split_path(split), "r") as f:
        return _trial_from_group(_group(f, block, trial))


def read_input_features(
    ref: SessionRef, split: str, block: int, trial: int
) -> np.ndarray:
    """Just the ``input_features`` array of one trial, ``(n_time_steps, 512)`` float32."""
    with h5py.File(ref.split_path(split), "r") as f:
        return _group(f, block, trial)["input_features"][:]


def read_labels(ref: SessionRef, split: str, block: int, trial: int) -> dict:
    """Per-trial labels/attrs + ``n_bins``, without loading ``input_features``.

    Returns ``attrs``, ``n_bins`` (= n_time_steps), and — for train/val — ``seq_class_ids``
    and ``transcription``.
    """
    with h5py.File(ref.split_path(split), "r") as f:
        g = _group(f, block, trial)
        out = {"attrs": dict(g.attrs), "n_bins": int(g["input_features"].shape[0])}
        for k in ("seq_class_ids", "transcription"):  # train/val only
            if k in g:
                out[k] = g[k][:]
    return out


def iter_trials(ref: SessionRef, split: str = "train"):
    """Yield each trial dict in a split (opens the file once)."""
    path = ref.split_path(split)
    if not path.exists():
        return
    with h5py.File(path, "r") as f:
        for key in sorted(f.keys()):
            yield _trial_from_group(f[key])


def session_summary(data_dir: str | Path = DATA_DIR) -> list[dict]:
    """One row per existing ``data_{split}.hdf5``: ``session``, ``split``, ``n_trials``."""
    rows = []
    for ref in list_sessions(data_dir):
        for split in ref.splits:
            rows.append(
                {
                    "session": ref.session,
                    "split": split,
                    "n_trials": n_trials(ref, split),
                }
            )
    return rows


# --- Label decoding ------------------------------------------------------------
def decode_phonemes(seq_class_ids, seq_len: int | None = None) -> str:
    """Space-separated phoneme string from ``seq_class_ids`` (``|`` = word boundary)."""
    ids = np.asarray(seq_class_ids)
    if seq_len is not None:
        ids = ids[:seq_len]
    return " ".join(PHONEME_LABELS[int(i)] for i in ids if int(i) != 0)


def decode_transcription(transcription) -> str:
    """Decode the 0-padded int32 ASCII ``transcription`` array to text."""
    arr = np.asarray(transcription)
    nz = np.nonzero(arr)[0]
    return "".join(chr(int(c)) for c in arr[: nz[-1] + 1]) if len(nz) else ""


if __name__ == "__main__":
    refs = list_sessions()
    n_val = sum("val" in r.splits for r in refs)
    n_test = sum("test" in r.splits for r in refs)
    print(f"{len(refs)} sessions | with val: {n_val} | with test: {n_test}")
    total = sum(n_trials(r, s) for r in refs for s in r.splits)
    print(f"total trials (all splits): {total}")

    r0 = refs[0]
    print(f"\n{r0.session} ({r0.date}) splits={r0.splits}")
    t0 = list_trials(r0, "train")[0]
    t = read_trial(r0, "train", t0.block, t0.trial)
    print("  input_features:", t["input_features"].shape, t["input_features"].dtype)
    print("  sentence_label:", repr(t["attrs"]["sentence_label"]))
    print(
        "  phonemes      :",
        decode_phonemes(t["seq_class_ids"], int(t["attrs"]["seq_len"])),
    )
    print("  transcription :", repr(decode_transcription(t["transcription"])))
