"""neuralset Study for Wilson et al. 2025 closed-loop cursor control.

One timeline is one task block. A released ``.mat`` holds a whole recording day, but
the task label and the online decoder change from block to block, so the block is the
smallest unit with invariant task entities. Neural counts, cursor and target
trajectories, and the online decoder outputs stay whole-block lazy streams; trial
timing becomes a visual ``Stimulus`` and an attempted-movement ``Action`` per trial.
"""

import typing as tp

import numpy as np
import pandas as pd

from neuralset.events import study

from neuralshed.datasets.study import NeuralshedStudy
from neuralshed.events import (  # noqa: F401, registers custom types
    ClickStateSeries,
    CursorPositionSeries,
    DecodedCursorVelocitySeries,
    SpikeCounts,
    TargetPositionSeries,
)

from . import utils

_SUBJECT = "T5"
_RATE_HZ = utils.FS


class Wilson2025CursorRecalibration(NeuralshedStudy):
    """Five years of closed-loop cursor control from T5's hand-knob arrays."""

    dataset_id: tp.ClassVar[str] = "wilson_2025"
    aliases: tp.ClassVar[tuple[str, ...]] = (
        "wilson_2025",
        "PRI-T",
        "cursor-recalibration",
    )
    bibtex: tp.ClassVar[str] = """
@article{wilson2025recalibration,
  title = {Long-term unsupervised recalibration of cursor-based intracortical brain-computer interfaces using a hidden Markov model},
  author = {Wilson, Guy H. and Stein, Elias A. and Kamdar, Foram and Avansino, Donald T. and Pun, Tsam Kiu and Gross, Ronnie and Hosman, Tommy and Singer-Clark, Tyler and Kapitonava, Anastasia and Hochberg, Leigh R. and Simeral, John D. and Shenoy, Krishna V. and Druckmann, Shaul and Henderson, Jaimie M. and Willett, Francis R.},
  journal = {Nature Biomedical Engineering},
  volume = {10},
  pages = {1466--1484},
  year = {2026},
  doi = {10.1038/s41551-025-01536-z},
  url = {https://doi.org/10.1038/s41551-025-01536-z},
}
"""
    url: tp.ClassVar[str] = "https://doi.org/10.5061/dryad.1jwstqk6g"
    licence: tp.ClassVar[str] = "CC0-1.0"
    description: tp.ClassVar[str] = (
        "Closed-loop two-dimensional cursor control by BrainGate2 participant T5 using "
        "two 96-channel Utah arrays in the hand knob of left dorsal motor cortex, across "
        "103 recording days spanning 2016 to 2021. The released 50 Hz streams include "
        "threshold crossings, cursor and target positions, and the online cursor and "
        "click decoder outputs."
    )
    requirements: tp.ClassVar[tuple[str, ...]] = ("scipy",)

    # Four timelines covering every emission path and every task: a radial-8 cursor
    # block, a grid (keyboard) block, and two click-enabled blocks that add the
    # ClickStateSeries row, one of them the fitts task.
    # NB: `neuralfetch.utils.update_source_info` computes the counts from this `query`
    # but omits `query` and `frequency` when it rewrites the block, so re-add both by
    # hand after running it, or the counts guard the wrong query.
    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=560,
        num_subjects=1,
        query="timeline_index in [0, 8, 27, 123]",
        num_events_in_query=730,
        event_types_in_query={
            "Action",
            "ClickStateSeries",
            "CursorPositionSeries",
            "DecodedCursorVelocitySeries",
            "SpikeCounts",
            "Stimulus",
            "TargetPositionSeries",
        },
        frequency=50.0,
    )

    def _download(self) -> None:
        """Download the Dryad release into this Study's folder.

        Dryad requires ``NEURALFETCH_DRYAD_TOKEN``. The deposit is a single 1.7 GB
        tarball that unpacks to roughly 30 GB, most of which is method-sweep output.
        """
        from neuralshed.datasets.dryad import Dryad

        Dryad(
            study="wilson_2025",
            dset_dir=self.path.parent,
            folder=self.path.name,
            doi="10.5061/dryad.1jwstqk6g",
        ).download()

    @property
    def _raw_dir(self):
        return self.path / "PRITpaper2025" / "T5" / "raw_data"

    def _ref(self, timeline: dict[str, tp.Any]) -> utils.BlockRef:
        return utils.BlockRef(
            session=timeline["session"],
            block=timeline["block"],
            task=timeline["task"],
            group=timeline["original_split"],
            path=self._raw_dir
            / timeline["original_split"]
            / f"{timeline['session']}.mat",
        )

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        implant = _implant_date()
        for ref in utils.list_blocks(self._raw_dir):
            date = utils.session_date(ref.session)
            yield {
                "subject": _SUBJECT,
                "session": ref.session,
                "run": ref.run,
                "task": ref.task,
                "block": ref.block,
                # The release folder doubles as the paper's own grouping of the
                # 2016-2020 series versus the 2021 sessions.
                "original_split": ref.group,
                "date": date.isoformat() if date else None,
                "post_implant_day": (date - implant).days if date and implant else None,
                # The authors' own block curation; keeping only sessions with at least
                # two such blocks reproduces the paper's 73-session cohort.
                "analysis_block": ref.is_analysis_block,
                "click_enabled": utils.click_enabled(ref),
            }

    def _load_series(self, timeline: dict[str, tp.Any], key: str) -> np.ndarray:
        """SpecialLoader target: source time-first data as dims-first float32."""
        data = utils.get_series(self._ref(timeline), key)
        return np.ascontiguousarray(data.T, dtype=np.float32)

    def _continuous_rows(
        self, timeline: dict[str, tp.Any], ref: utils.BlockRef
    ) -> list[dict[str, tp.Any]]:
        duration = utils.duration(ref)

        def row(*, type_: str, key: str, dims=(), unit: str = "", **extra):
            out = {
                "type": type_,
                "feature": key,
                "start": 0.0,
                "duration": duration,
                "frequency": _RATE_HZ,
                "unit": unit,
                "filepath": study.SpecialLoader(
                    method=self._load_series,
                    timeline=timeline,
                    key=key,
                ).to_json(),
                **extra,
            }
            if dims:
                out["dims"] = dims
                out["n_features"] = len(dims)
            return out

        rows = [
            # The source scales the 20 ms bin count to a rate, so the unit is Hz rather
            # than the counts/bin other pre-binned datasets emit (see SPIKE_RATE_SCALE).
            row(
                type_="SpikeCounts",
                key="threshold_crossings",
                n_features=utils.N_CHANNELS,
                unit="Hz",
                normalization="raw",
                method="threshold_crossing",
                threshold=f"{utils.RMS_THRESHOLD} x RMS",
            ),
            row(
                type_="CursorPositionSeries",
                key="cursor_position",
                dims=utils.POSITION_DIMS,
                unit="px",
                kind="cursor_position",
                reference_frame="screen_center",
            ),
            row(
                type_="TargetPositionSeries",
                key="target_position",
                dims=utils.POSITION_DIMS,
                unit="px",
                kind="target_position",
                reference_frame="screen_center",
            ),
            # Not px/s: one unit moves the cursor about 20 px per bin on 163 blocks and
            # about 1e5 px per bin on the other 389, so the scale is per block (README).
            row(
                type_="DecodedCursorVelocitySeries",
                key="decoded_velocity",
                dims=utils.POSITION_DIMS,
                unit="a.u.",
                kind="online_decoder_output",
                reference_frame="screen",
            ),
        ]
        if utils.click_enabled(ref):
            rows.append(
                row(
                    type_="ClickStateSeries",
                    key="decoded_click",
                    dims=("click",),
                    unit="binary",
                    kind="online_decoder_output",
                )
            )
        return rows

    def _trial_rows(self, ref: utils.BlockRef) -> list[dict[str, tp.Any]]:
        windows = utils.trial_windows(ref)
        if not windows:
            return []
        targets = utils.get_series(ref, "target_position")
        sizes = utils.get_series(ref, "target_size").ravel()
        success = utils.trial_success(ref)
        trial_ids = utils.trial_indices(ref)

        # Target codes are block-local: the index into this block's sorted set of
        # distinct target centres. Most `cursor` blocks are radial-8 plus a centre
        # target, but some use randomly placed targets and the grid task has up to 55
        # cells, so a fixed spoke code would not hold. `target_x`/`target_y` carry the
        # geometry when comparing across blocks.
        cued = np.array([targets[start] for start, _ in windows])
        layout = np.unique(cued, axis=0)
        codes = [
            int(np.flatnonzero((layout == target).all(axis=1))[0]) for target in cued
        ]

        rows: list[dict[str, tp.Any]] = []
        for index, (start, stop) in enumerate(windows):
            target = cued[index]
            common = {
                "start": start / _RATE_HZ,
                "duration": (stop - start) / _RATE_HZ,
                "code": codes[index],
                "trial_id": trial_ids[index],
                "target_x": float(target[0]),
                "target_y": float(target[1]),
                "target_size": float(sizes[start]),
                "n_targets": len(layout),
            }
            rows.append(
                {
                    "type": "Stimulus",
                    "modality": "visual",
                    "description": "cued cursor target",
                    "stimulus_kind": "target_cue",
                    **common,
                }
            )
            rows.append(
                {
                    "type": "Action",
                    "description": "attempted hand movement driving the cursor",
                    "task_condition": "attempted",
                    "action_kind": "cursor_control",
                    "success": success[index],
                    **common,
                }
            )
        return rows

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        ref = self._ref(timeline)
        rows = self._continuous_rows(timeline, ref)
        rows.extend(self._trial_rows(ref))
        return pd.DataFrame(rows)


def _implant_date():
    from neuralshed.datasets import participants

    return participants.implant_date("braingate2/T5")


__all__ = ["Wilson2025CursorRecalibration"]
