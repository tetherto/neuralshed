"""neuralset Study for Deo et al. 2024 bimanual cursor control.

One timeline is one task **block** (a contiguous run under a single control loop). The
day's ``.mat`` concatenates its blocks with recording gaps between them, so blocks -- not
days -- are the continuous, single-condition recording units; :meth:`iter_timelines`
slices each day by ``blockNum``. Neural threshold crossings and the two cursors' and
targets' ``(x, y)`` positions become whole-block lazy streams (the two hands split into
separate homogeneous ``CursorPositionSeries`` / ``TargetPositionSeries``); each trial
becomes a ``Stimulus`` (the cued movement type) and an ``Action`` (the attempted move).
"""

import typing as tp

import numpy as np
import pandas as pd

from neuralset.events import study

from neuralshed.datasets.study import NeuralshedStudy
from neuralshed.events import (  # noqa: F401  registers custom types
    CursorPositionSeries,
    SpikeCounts,
    TargetPositionSeries,
)

from . import utils

_SUBJECT = "T5"
_RATE_HZ = utils.FS_NEURAL

# (source accessor, row indices in Cp/Tp) for each per-hand position stream. Cp/Tp rows
# are [right_x, right_y, left_x, left_y]; the two cursors/targets are emitted as separate
# homogeneous series so a model can select one hand.
_POSITION_SOURCE: dict[str, tuple[str, tuple[int, int]]] = {
    "right_cursor_position": ("cursor", (0, 1)),
    "left_cursor_position": ("cursor", (2, 3)),
    "right_target_position": ("target", (0, 1)),
    "left_target_position": ("target", (2, 3)),
}


class Deo2024Bimanual(NeuralshedStudy):
    """Bimanual cursor control from T5's hand-knob motor cortex arrays."""

    dataset_id: tp.ClassVar[str] = "deo_2024"
    aliases: tp.ClassVar[tuple[str, ...]] = ("deo_2024", "bimanual-cursor")
    bibtex: tp.ClassVar[str] = """
@article{deo2024bimanual,
  title = {Brain control of bimanual movement enabled by recurrent neural networks},
  author = {Deo, Darrel R. and Willett, Francis R. and Avansino, Donald T. and Hochberg, Leigh R. and Henderson, Jaimie M. and Shenoy, Krishna V.},
  journal = {Scientific Reports},
  volume = {14},
  year = {2024},
  doi = {10.1038/s41598-024-51617-3},
  url = {https://doi.org/10.1038/s41598-024-51617-3},
}
"""
    url: tp.ClassVar[str] = "https://doi.org/10.5061/dryad.sn02v6xbb"
    licence: tp.ClassVar[str] = "CC0-1.0"
    description: tp.ClassVar[str] = (
        "Delayed bimanual and unimanual control of two on-screen cursors by BrainGate2 "
        "participant T5 (C4 spinal cord injury), using two 96-channel Utah arrays in "
        "hand-knob motor cortex. Released 50 Hz streams are 20 ms-binned threshold "
        "crossings and the two cursors' and targets' positions, across open-loop "
        "instructed-delay blocks and real-time closed-loop decoder control."
    )
    requirements: tp.ClassVar[tuple[str, ...]] = ("scipy",)

    # One timeline per block (204 across the 13 days). Every timeline emits the same
    # event families, so timeline 0 exercises all emission paths.
    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=204,
        num_subjects=1,
        query="timeline_index == 0",
        num_events_in_query=161,
        event_types_in_query={
            "SpikeCounts",
            "CursorPositionSeries",
            "TargetPositionSeries",
            "Stimulus",
            "Action",
        },
        frequency=50.0,
    )

    def _download(self) -> None:
        """Download the Dryad release into this Study's folder (needs NEURALFETCH_DRYAD_TOKEN)."""
        from neuralshed.datasets.dryad import Dryad

        Dryad(
            study="deo_2024",
            dset_dir=self.path.parent,
            folder=self.path.name,
            doi="10.5061/dryad.sn02v6xbb",
            include=["bimanualData.zip", "README.md"],
        ).download()

    # --- timelines: one per (session, block) -----------------------------------
    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        for session in utils.list_sessions(self.path):
            path = utils.session_path(session, self.path)
            block_numbers = utils.get_block_numbers(path)
            trials = utils.get_trials(path)
            for block in sorted(trials["block"].unique()):
                loop_type = utils.loop_type_of(session, int(block))
                if loop_type == "unknown":  # transition blocks; no declared condition
                    continue
                start_bin, stop_bin = utils.block_bin_range(block_numbers, int(block))
                yield {
                    "subject": _SUBJECT,
                    "session": session,
                    "block": int(block),
                    "loop_type": loop_type,
                    "date": utils.session_date(session),
                    "path": path.name,
                    "start_bin": start_bin,
                    "stop_bin": stop_bin,
                }

    # --- lazy signal loading ---------------------------------------------------
    def _load_series(self, timeline: dict[str, tp.Any], key: str) -> np.ndarray:
        """SpecialLoader target: one block's stream as dims-first float32."""
        path = self.path / timeline["path"]
        a, b = timeline["start_bin"], timeline["stop_bin"]
        if key == "threshold_crossings":
            arr = utils.get_threshold_crossings(path)[a:b]  # (block_bins, 192)
            return np.ascontiguousarray(arr.T, dtype=np.float32)  # (192, block_bins)
        source, rows = _POSITION_SOURCE[key]
        full = (
            utils.get_cursor_positions(path)
            if source == "cursor"
            else utils.get_target_positions(path)
        )  # (4, n_bins)
        return np.ascontiguousarray(
            full[rows, a:b], dtype=np.float32
        )  # (2, block_bins)

    def _continuous_rows(self, timeline: dict[str, tp.Any]) -> list[dict[str, tp.Any]]:
        duration = (timeline["stop_bin"] - timeline["start_bin"]) / _RATE_HZ

        def row(*, type_: str, key: str, dims=(), unit: str = "", **extra):
            out = {
                "type": type_,
                "feature": key,
                "start": 0.0,
                "duration": duration,
                "frequency": _RATE_HZ,
                "unit": unit,
                "filepath": study.SpecialLoader(
                    method=self._load_series, timeline=timeline, key=key
                ).to_json(),
                **extra,
            }
            if dims:
                out["dims"] = dims
                out["n_features"] = len(dims)
            return out

        rows = [
            row(
                type_="SpikeCounts",
                key="threshold_crossings",
                n_features=utils.N_CHANNELS,
                unit="counts/bin",
                normalization="raw",
                method="threshold_crossing",
                threshold=f"{utils.THRESHOLD_RMS} x RMS",
            )
        ]
        for key, (source, _) in _POSITION_SOURCE.items():
            effector = "right" if key.startswith("right") else "left"
            type_ = (
                "CursorPositionSeries" if source == "cursor" else "TargetPositionSeries"
            )
            rows.append(
                row(
                    type_=type_,
                    key=key,
                    dims=utils.POSITION_DIMS,
                    unit="a.u.",
                    kind=f"{source}_position",
                    effector=effector,
                    reference_frame="screen_center",
                )
            )
        return rows

    # --- trial markers ---------------------------------------------------------
    def _trial_rows(self, timeline: dict[str, tp.Any]) -> list[dict[str, tp.Any]]:
        path = self.path / timeline["path"]
        start_bin = timeline["start_bin"]
        trials = utils.get_trials(path)
        trials = trials[trials["block"] == timeline["block"]]

        rows: list[dict[str, tp.Any]] = []
        for trial_id, t in enumerate(trials.itertuples(index=False)):
            delay_start_s = (t.delay_start_bin - start_bin) / _RATE_HZ
            go_s = (t.go_cue_bin - start_bin) / _RATE_HZ
            move_end_s = (t.move_end_bin - start_bin) / _RATE_HZ
            rows.append(
                {
                    "type": "Stimulus",
                    "start": delay_start_s,
                    "duration": move_end_s - delay_start_s,
                    "code": int(t.move_type),
                    "description": t.move_type_name,
                    "modality": "visual",
                    "stimulus_kind": "movement_cue",
                    "trial_id": trial_id,
                }
            )
            rows.append(
                {
                    "type": "Action",
                    "start": go_s,
                    "duration": move_end_s - go_s,
                    "code": int(t.move_type),
                    "description": f"attempted {utils.MOVE_TYPE_LABEL.get(t.move_type_name, '')} movement",
                    "task_condition": "attempted",
                    "action_kind": "cursor_control",
                    "trial_id": trial_id,
                }
            )
        return rows

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        rows = self._continuous_rows(timeline)
        rows.extend(self._trial_rows(timeline))
        return pd.DataFrame(rows)


__all__ = ["Deo2024Bimanual"]
