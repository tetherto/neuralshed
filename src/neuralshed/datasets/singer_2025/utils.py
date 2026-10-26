"""Readers for Singer-Clark et al. 2025 cursor-control ``.mat`` blocks.

The public release contains 51 cursor-task blocks from post-implant days 39, 202,
and 468. Each file is one 3 to 5 minute block whose neural and behavioral streams
share a 10 ms time base. The separate ``t15_day00202_words.mat`` speech-only file
is useful as a comparison but is outside this cursor-control Study.
"""

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import scipy.io as sio

from neuralshed.datasets.utils import dataset_dir

N_ELECTRODES = 256
N_ARRAYS = 4
NEURAL_BIN_SIZE_MS = 10
FS = 100.0
POSITION_DIMS = ("x", "y")
ARRAY_ORDER = ("v6v", "4", "55b", "d6v")

DATA_SUBDIR = "Singer2025CursorControl"
DEFAULT_DATA_DIR = dataset_dir("singer_2025") / "data" / DATA_SUBDIR

_BLOCK_RE = re.compile(
    r"t15_day(?P<day>\d{5})_block(?P<block>\d{2})_(?P<task>.+)_task\.mat$"
)

TASK_LABELS = {
    "radial8_calibration": "Radial8 Calibration Task",
    "simultaneous_speech_and_cursor": "Simultaneous Speech and Cursor Task",
    "grid_evaluation": "Grid Evaluation Task",
}

CURSOR_IMAGERY_BY_DAY = {
    39: "varied hand, paintbrush, and tongue motor imagery",
    202: "right hand mouse imagery",
    468: "effector-agnostic intuition",
}


@dataclass(frozen=True)
class BlockRef:
    """One source block on disk."""

    day: int
    block: int
    task: str
    path: Path

    @property
    def session(self) -> str:
        return f"t15_day{self.day:05d}"

    @property
    def run(self) -> str:
        return f"block{self.block:02d}"

    @property
    def label(self) -> str:
        return self.path.stem

    @property
    def task_label(self) -> str:
        return TASK_LABELS[self.task]


def block_ref(path: str | Path) -> BlockRef:
    """Parse a released cursor-task filename."""
    path = Path(path)
    match = _BLOCK_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"not a Singer cursor-task block: {path.name!r}")
    return BlockRef(
        day=int(match.group("day")),
        block=int(match.group("block")),
        task=match.group("task"),
        path=path,
    )


def list_blocks(data_dir: str | Path = DEFAULT_DATA_DIR) -> list[BlockRef]:
    """All downloaded cursor-task blocks in canonical day/block order."""
    root = Path(data_dir)
    if not root.exists():
        return []
    refs = []
    for path in root.glob("t15_day*_block*_task.mat"):
        if path.name.startswith("._"):
            continue
        try:
            refs.append(block_ref(path))
        except ValueError:
            continue
    return sorted(refs, key=lambda ref: (ref.day, ref.block, ref.task))


@lru_cache(maxsize=2)
def _load(path: str) -> dict:
    """Load one block, keeping at most two approximately 28 MB files in memory."""
    return sio.loadmat(path, squeeze_me=True, struct_as_record=False)


def load_block(ref: BlockRef) -> dict:
    return _load(str(ref.path))


def available_fields(ref: BlockRef) -> set[str]:
    return {key for key in load_block(ref) if not key.startswith("__")}


def get_series(ref: BlockRef, key: str) -> np.ndarray:
    """Return one per-bin field as ``(n_bins, n_dims)``."""
    data = load_block(ref)
    if key not in data:
        raise KeyError(f"{ref.label}: missing field {key!r}")
    out = np.asarray(data[key])
    if out.ndim == 1:
        out = out[:, None]
    if out.ndim != 2:
        raise ValueError(f"{ref.label}:{key} expected 2-D, got {out.shape!r}")
    return out


def n_bins(ref: BlockRef) -> int:
    return int(get_series(ref, "timestamp_sec").shape[0])


def duration(ref: BlockRef) -> float:
    return n_bins(ref) / FS


def trial_starts(ref: BlockRef) -> np.ndarray:
    """Zero-based source indices at which trials begin."""
    return np.atleast_1d(load_block(ref)["trial_start_bin"]).astype(int)


def trial_windows(ref: BlockRef) -> list[tuple[int, int]]:
    """Half-open ``(start, stop)`` bin ranges for every trial."""
    starts = trial_starts(ref)
    data = load_block(ref)
    if "trial_end_bin" in data:
        stops = np.atleast_1d(data["trial_end_bin"]).astype(int) + 1
    else:
        stops = np.r_[starts[1:], n_bins(ref)]
    return [
        (int(start), min(int(stop), n_bins(ref)))
        for start, stop in zip(starts, stops)
        if 0 <= start < stop
    ]


def trial_field(ref: BlockRef, key: str, default=None) -> np.ndarray | None:
    """A source per-trial field, or ``default`` when this task does not have it."""
    value = load_block(ref).get(key, default)
    if value is default:
        return default
    return np.atleast_1d(value)


def scalar_field(ref: BlockRef, key: str, default=None):
    """A source per-block scalar with NumPy wrappers removed."""
    value = load_block(ref).get(key, default)
    if value is default:
        return default
    arr = np.asarray(value).squeeze()
    return arr.item() if arr.ndim == 0 else arr


def speech_prompts(ref: BlockRef) -> list[str]:
    values = trial_field(ref, "speech_prompt", [])
    return [str(value).strip() for value in np.atleast_1d(values)]


def geometry(ref: BlockRef) -> dict[str, float | int | None]:
    """Per-block task geometry and selection rule, ``None`` where the task lacks it.

    ``cursor_radius``/``target_radius`` give the circular-target touching criterion
    (distance < cursor_radius + target_radius). ``dwell_requirement_sec`` is the
    selection rule on the dwell sessions (days 39 and 202); the click sessions
    (day 468) have no dwell and use the click decoder instead. ``grid_num_rows`` and
    ``grid_total_height`` describe the square-target grid.
    """
    return {
        key: scalar_field(ref, key)
        for key in (
            "cursor_radius",
            "target_radius",
            "dwell_requirement_sec",
            "grid_num_rows",
            "grid_total_height",
        )
    }


def grid_cell_index(ref: BlockRef, position: np.ndarray) -> int | None:
    """Row-major cell index of a grid target, or ``None`` outside the grid task.

    The grid is centered on the screen, so a cell's center sits at
    ``-H/2 + (i + 0.5) * H/rows`` along each axis.
    """
    rows = scalar_field(ref, "grid_num_rows")
    height = scalar_field(ref, "grid_total_height")
    if rows is None or height is None:
        return None
    rows = int(rows)
    cell = float(height) / rows
    col, row = (
        int(np.round((float(value) + float(height) / 2 - cell / 2) / cell))
        for value in (position[0], position[1])
    )
    if not (0 <= col < rows and 0 <= row < rows):
        return None
    return row * rows + col


def array_labels(ref: BlockRef) -> tuple[str, ...]:
    """One anatomical array label per electrode, in feature-column order."""
    raw = np.atleast_1d(load_block(ref)["array_label_by_electrode"])
    return tuple(str(value).strip() for value in raw)


def session_blocks(
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> dict[str, list[BlockRef]]:
    out: dict[str, list[BlockRef]] = {}
    for ref in list_blocks(data_dir):
        out.setdefault(ref.session, []).append(ref)
    return out


__all__ = [
    "N_ELECTRODES",
    "N_ARRAYS",
    "NEURAL_BIN_SIZE_MS",
    "FS",
    "POSITION_DIMS",
    "ARRAY_ORDER",
    "DATA_SUBDIR",
    "DEFAULT_DATA_DIR",
    "TASK_LABELS",
    "CURSOR_IMAGERY_BY_DAY",
    "BlockRef",
    "block_ref",
    "list_blocks",
    "load_block",
    "available_fields",
    "get_series",
    "n_bins",
    "duration",
    "trial_starts",
    "trial_windows",
    "trial_field",
    "scalar_field",
    "speech_prompts",
    "geometry",
    "grid_cell_index",
    "array_labels",
    "session_blocks",
]
