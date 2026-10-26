"""neuralset Study for Willsey et al. 2025 finger decoding.

One timeline is one task block, because the block is the smallest unit with invariant
task entities: whether the task cues two or four degrees of freedom, and which online
decoder drove the hand, change from block to block within a day.

Two block kinds share the Study. Closed-loop blocks emit the decoder's own finger
posture alongside the cued target, open-loop blocks emit the cued posture only, since
the hand was driven by the task rather than by a decoder. The quadcopter block is not a
timeline at all: it holds a saved flight path with no neural data and no clock.

Every stream starts at :func:`utils.steady_state_start` rather than at bin 0. The first
76 to 88 bins of a block are 20 ms wide instead of 50 ms, so treating them as part of a
uniform 20 Hz signal would misplace roughly 1.6 s of data. No trial starts before bin
192, so nothing is lost.
"""

import typing as tp

import numpy as np
import pandas as pd

from neuralset.events import study

from neuralshed.datasets.study import NeuralshedStudy
from neuralshed.events import (  # noqa: F401, registers custom types
    BandPower,
    GraspSeries,
    TargetGraspSeries,
)

from . import utils

_SUBJECT = "T5"
_RATE_HZ = utils.FS
_TASKS = {
    utils.KIND_CLOSED_LOOP: "fingertargets",
    utils.KIND_OPEN_LOOP: "cuedmovement",
}


class Willsey2025FingerDecoding(NeuralshedStudy):
    """Closed-loop four-degree-of-freedom finger control from T5's hand-knob arrays."""

    dataset_id: tp.ClassVar[str] = "willsey_2025"
    aliases: tp.ClassVar[tuple[str, ...]] = (
        "willsey_2025",
        "finger-decoding",
        "quadcopter",
    )
    bibtex: tp.ClassVar[str] = """
@article{willsey2025finger,
  title = {A high-performance brain-computer interface for finger decoding and quadcopter game control in an individual with paralysis},
  author = {Willsey, Matthew S. and Shah, Nishal P. and Avansino, Donald T. and Hahn, Nick V. and Jamiolkowski, Ryan M. and Kamdar, Foram B. and Hochberg, Leigh R. and Willett, Francis R. and Henderson, Jaimie M.},
  journal = {Nature Medicine},
  volume = {31},
  number = {1},
  pages = {96--104},
  year = {2025},
  doi = {10.1038/s41591-024-03341-8},
  url = {https://doi.org/10.1038/s41591-024-03341-8},
}
"""
    url: tp.ClassVar[str] = "https://doi.org/10.5061/dryad.1jwstqk4f"
    licence: tp.ClassVar[str] = "CC0-1.0"
    description: tp.ClassVar[str] = (
        "Closed-loop finger control by BrainGate2 participant T5 using two 96-channel "
        "Utah arrays in the hand knob of left precentral gyrus. Three finger groups "
        "give four degrees of freedom, the thumb contributing flexion and abduction. "
        "The released 20 Hz streams are spike-band power on 192 electrodes, the online "
        "decoder's finger posture, and the cued target posture."
    )
    requirements: tp.ClassVar[tuple[str, ...]] = ("scipy",)

    # Three timelines covering every emission path: an open-loop block (no GraspSeries,
    # nine cued conditions), a closed-loop block whose task cued two degrees of freedom
    # while a four-dimensional decoder ran, and a fully labelled four-dimensional block.
    # NB: `neuralfetch.utils.update_source_info` computes the counts from this `query`
    # but omits `query` and `frequency` when it rewrites the block, so re-add both by
    # hand after running it, or the counts guard the wrong query.
    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=60,
        num_subjects=1,
        query="timeline_index in [25, 30, 53]",
        num_events_in_query=404,
        event_types_in_query={
            "Action",
            "BandPower",
            "GraspSeries",
            "Stimulus",
            "TargetGraspSeries",
        },
        frequency=20.0,
    )

    def _download(self) -> None:
        """Download the Dryad release into this Study's folder.

        Dryad requires ``NEURALFETCH_DRYAD_TOKEN``. The deposit is a single 181 MB zip.
        """
        from neuralshed.datasets.dryad import Dryad

        Dryad(
            study="willsey_2025",
            dset_dir=self.path.parent,
            folder=self.path.name,
            doi="10.5061/dryad.1jwstqk4f",
        ).download()

    @property
    def _raw_dir(self):
        # `self.path` is the class-named folder the framework appends under the
        # dataset's `data/`, the same place `_download` lands the deposit.
        return self.path / "Data"

    def _ref(self, timeline: dict[str, tp.Any]) -> utils.BlockRef:
        return utils.BlockRef(
            session=timeline["session"],
            block=timeline["block"],
            kind=timeline["block_kind"],
            start_time=timeline["start_time"],
            path=self._raw_dir / timeline["relative_path"],
        )

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        implant = _implant_date()
        for ref in utils.list_blocks(self._raw_dir):
            if ref.kind == utils.KIND_QUADCOPTER:
                continue  # a saved flight path, not a recording
            date = utils.session_date(ref.session)
            labels = utils.BLOCK_LABELS.get(ref.label, {})
            cued = utils.target_dof(ref) if ref.kind == utils.KIND_CLOSED_LOOP else ()
            yield {
                "subject": _SUBJECT,
                "session": ref.session,
                "run": ref.run,
                "task": _TASKS[ref.kind],
                "block": ref.block,
                "block_kind": ref.kind,
                "start_time": ref.start_time,
                "relative_path": str(ref.path.relative_to(self._raw_dir)),
                # The framework broadcasts every timeline key to that block's rows as a
                # string, so these three are emitted as strings with "" for "not
                # applicable" rather than as ints that would arrive as "None".
                #
                # How many degrees of freedom the task cued, read off the target stream.
                # Empty on the open-loop blocks, which cue one posture at a time.
                "task_dof": str(len(cued)) if cued else "",
                # How many the online decoder emitted. Not recoverable from the files:
                # see utils.BLOCK_LABELS. Empty on the blocks the authors did not label.
                "decoder_dof": str(labels.get("decoder_dof", "")),
                "new_targets": str(labels.get("new_targets", "")),
                "date": date.isoformat(),
                "post_implant_day": (date - implant).days,
            }

    def _load_neural(self, timeline: dict[str, tp.Any]) -> np.ndarray:
        """SpecialLoader target: spike-band power as dims-first ``(192, n_bins)``."""
        ref = self._ref(timeline)
        power = utils.get_neural(ref)[utils.steady_state_start(ref) :]
        return np.ascontiguousarray(power.T, dtype=np.float32)

    def _load_grasp(self, timeline: dict[str, tp.Any], key: str) -> np.ndarray:
        """SpecialLoader target: a finger stream as dims-first ``(4, n_bins)``."""
        ref = self._ref(timeline)
        series = utils.on_bins(ref, key)[utils.steady_state_start(ref) :]
        return np.ascontiguousarray(
            series[:, list(utils.DECODED_COLUMNS)].T, dtype=np.float32
        )

    def _continuous_rows(
        self,
        timeline: dict[str, tp.Any],
        ref: utils.BlockRef,
        n_bins: int,
    ) -> list[dict[str, tp.Any]]:
        duration = n_bins / _RATE_HZ
        rows = [
            {
                "type": "BandPower",
                "feature": "spike_band_power",
                "start": 0.0,
                "duration": duration,
                "frequency": _RATE_HZ,
                "n_features": utils.N_CHANNELS,
                # The Supplementary Methods give the whole chain: a 250 Hz digital
                # high-pass, common average referencing, then the sum of the squared
                # magnitude per 50 ms window. So the quantity is a squared voltage,
                # summed rather than averaged, which makes it extensive. The 20 ms
                # prefix of every block confirms it independently by sitting at 20/50
                # of the steady level.
                "unit": "uV^2",
                "normalization": "raw",
                "processing": (
                    "250 Hz digital high-pass; common average referencing; "
                    "sum of squared magnitude per 50 ms bin"
                ),
                "filepath": study.SpecialLoader(
                    method=self._load_neural, timeline=timeline
                ).to_json(),
            },
            {
                "type": "TargetGraspSeries",
                "feature": "finger_target",
                "start": 0.0,
                "duration": duration,
                "frequency": _RATE_HZ,
                "dims": utils.DECODED_DIMS,
                "n_features": len(utils.DECODED_DIMS),
                "unit": "proportion",
                "kind": "target_posture",
                "reference_frame": "range_of_motion",
                "filepath": study.SpecialLoader(
                    method=self._load_grasp, timeline=timeline, key="target"
                ).to_json(),
            },
        ]
        if ref.kind == utils.KIND_CLOSED_LOOP:
            rows.append(
                {
                    "type": "GraspSeries",
                    "feature": "finger_position",
                    "start": 0.0,
                    "duration": duration,
                    "frequency": _RATE_HZ,
                    "dims": utils.DECODED_DIMS,
                    "n_features": len(utils.DECODED_DIMS),
                    "unit": "proportion",
                    "kind": "online_decoder_output",
                    "reference_frame": "range_of_motion",
                    "filepath": study.SpecialLoader(
                        method=self._load_grasp, timeline=timeline, key="position"
                    ).to_json(),
                }
            )
        return rows

    def _closed_loop_trials(
        self,
        ref: utils.BlockRef,
        offset: int,
    ) -> list[tuple[int, int, str, np.ndarray]]:
        targets = utils.trial_targets(ref)
        out = []
        for index, (start, stop) in enumerate(utils.trial_windows(ref)):
            target = targets[index]
            neutral = bool(np.all(target == utils.POSITION_NEUTRAL))
            condition = utils.CLOSED_LOOP_CONDITIONS[0 if neutral else 1]
            out.append((start - offset, stop - offset, condition, target))
        return out

    def _open_loop_trials(
        self, ref: utils.BlockRef, offset: int
    ) -> list[tuple[int, int, str, np.ndarray]]:
        postures = {
            name: np.array(row) for row, name in utils.OPEN_LOOP_CONDITIONS.items()
        }
        return [
            (start - offset, stop - offset, condition, postures[condition])
            for start, stop, condition in utils.cue_windows(ref)
        ]

    def _trial_rows(self, ref: utils.BlockRef, offset: int) -> list[dict[str, tp.Any]]:
        trials = (
            self._closed_loop_trials(ref, offset)
            if ref.kind == utils.KIND_CLOSED_LOOP
            else self._open_loop_trials(ref, offset)
        )
        rows: list[dict[str, tp.Any]] = []
        for index, (start, stop, condition, target) in enumerate(trials):
            if start < 0:
                raise ValueError(
                    f"{ref.label}: trial {index} starts inside the 20 ms prefix"
                )
            common = {
                "start": start / _RATE_HZ,
                "duration": (stop - start) / _RATE_HZ,
                "code": utils.condition_code(condition),
                "condition": condition,
                "trial_id": index,
                **{
                    f"target_{dim}": float(value)
                    for dim, value in zip(utils.DECODED_DIMS, target)
                },
            }
            rows.append(
                {
                    "type": "Stimulus",
                    "modality": "visual",
                    "description": "cued finger posture",
                    "stimulus_kind": "target_cue",
                    **common,
                }
            )
            rows.append(
                {
                    "type": "Action",
                    "description": "attempted finger movement toward the cued posture",
                    "task_condition": "attempted",
                    "action_kind": "finger_control",
                    **common,
                }
            )
        return rows

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        ref = self._ref(timeline)
        offset = utils.steady_state_start(ref)
        n_bins = utils.get_neural(ref).shape[0] - offset
        rows = self._continuous_rows(timeline, ref, n_bins)
        rows.extend(self._trial_rows(ref, offset))
        return pd.DataFrame(rows)


def _implant_date():
    from neuralshed.datasets import participants

    return participants.implant_date("braingate2/T5")


__all__ = ["Willsey2025FingerDecoding"]
