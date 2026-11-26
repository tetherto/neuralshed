"""neuralset Study for Willsey et al. 2025 finger decoding.

One timeline is one task block, because each block is a separate file on its own
stretch of the shared clock, 39 s to 78 min from the next, and a timeline is the
largest span on one continuous clock.

Two block kinds share the Study. Closed-loop blocks emit the decoder's own finger
posture alongside the cued target, open-loop blocks emit the cued posture only, since
the hand was driven by the task rather than by a decoder. The quadcopter block is not a
timeline at all: it holds a saved flight path with no neural data and no clock.

Every stream starts at :func:`utils.steady_state_start` rather than at bin 0. The first
76 to 88 bins of a block are 20 ms wide instead of 50 ms, so treating them as part of a
uniform 20 Hz signal would misplace roughly 1.6 s of data. The earliest trial starts at
bin 121 (open loop) or 192 (closed loop), after the narrow bins end (bin 89 at the
latest), so nothing is lost.

The finger streams cover only :func:`utils.task_span`, the bins the task stream reaches.
The neural recording starts seconds before the task stream and can run on after it, and
finger values there would be made up.
"""

import typing as tp

import numpy as np
import pandas as pd

from neuralset.events import study

from neuralshed.datasets import participants
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
_LOOP_TYPES = {utils.KIND_CLOSED_LOOP: "closed", utils.KIND_OPEN_LOOP: "open"}


class Willsey2025FingerDecoding(NeuralshedStudy):
    """Closed-loop four-degree-of-freedom finger control from T5's hand-knob arrays."""

    dataset_id: tp.ClassVar[str] = "willsey_2025"
    aliases: tp.ClassVar[tuple[str, ...]] = (
        "willsey_2025",
        "finger-decoding",
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
    # nine cued postures), a closed-loop block whose task cued two degrees of freedom
    # while a four-dimensional decoder ran, and a fully labelled four-dimensional block.
    # NB: `neuralfetch.utils.update_source_info` computes the counts from this `query`
    # but omits `query` and `frequency` when it rewrites the block, so re-add both by
    # hand after running it, or the counts guard the wrong query.
    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=60,
        num_subjects=1,
        query=(
            "(session == 't5.2023.04.06' and block in [5, 31])"
            " or (session == 't5.2023.04.13' and block == 44)"
        ),
        num_events_in_query=306,
        event_types_in_query={
            "Action",
            "BandPower",
            "GraspSeries",
            "Stimulus",
            "TargetGraspSeries",
        },
        frequency=20.0,
    )

    def _download(self, overwrite: bool = False) -> None:
        """Download the Dryad release into this Study's folder.

        Dryad requires ``NEURALFETCH_DRYAD_TOKEN``. The deposit is a single 181 MB zip.
        """
        from neuralshed.datasets.dryad import Dryad

        Dryad(
            study="willsey_2025",
            dset_dir=self.path.parent,
            folder=self.path.name,
            doi="10.5061/dryad.1jwstqk4f",
        ).download(overwrite=overwrite)

    @property
    def _raw_dir(self):
        # `self.path` is the class-named folder the framework appends under the
        # dataset's `data/`, the same place `_download` lands the deposit.
        return self.path / "Data"

    def _ref(self, timeline: dict[str, tp.Any]) -> utils.BlockRef:
        return utils.block_ref(self._raw_dir / timeline["path"])

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        for ref in utils.list_blocks(self._raw_dir):
            if ref.kind == utils.KIND_QUADCOPTER:
                continue  # a saved flight path, not a recording
            date = utils.session_date(ref.session)
            cued = utils.target_dof(ref) if ref.kind == utils.KIND_CLOSED_LOOP else ()
            yield {
                "subject": _SUBJECT,
                "session": ref.session,
                "run": ref.run,
                "task": _TASKS[ref.kind],
                "block": ref.block,
                "loop_type": _LOOP_TYPES[ref.kind],
                "path": str(ref.path.relative_to(self._raw_dir)),
                # How many degrees of freedom the task cued, read off the target stream.
                # None on the open-loop blocks, which cue one posture at a time.
                "task_dof": len(cued) if cued else None,
                # How many the online decoder emitted, and how many fingers each outward
                # trial moved. None on the open-loop blocks.
                "decoder_dof": utils.decoder_dof(ref) if cued else None,
                "new_targets": utils.new_targets(ref) if cued else None,
                "date": date.isoformat(),
                "post_implant_day": participants.post_implant_day(
                    "braingate2/T5", date.isoformat()
                ),
            }

    def _load_neural(self, timeline: dict[str, tp.Any]) -> np.ndarray:
        """SpecialLoader target: spike-band power as dims-first ``(192, n_bins)``."""
        ref = self._ref(timeline)
        power = utils.get_neural(ref)[utils.steady_state_start(ref) :]
        return np.ascontiguousarray(power.T, dtype=np.float32)

    def _load_grasp(self, timeline: dict[str, tp.Any], key: str) -> np.ndarray:
        """SpecialLoader target: a finger stream as dims-first ``(4, n_bins)``."""
        ref = self._ref(timeline)
        first, stop = utils.task_span(ref)
        series = utils.on_bins(ref, key)[first:stop]
        return np.ascontiguousarray(
            series[:, list(utils.DECODED_COLUMNS)].T, dtype=np.float32
        )

    def _continuous_rows(
        self,
        timeline: dict[str, tp.Any],
        ref: utils.BlockRef,
        n_bins: int,
        offset: int,
    ) -> list[dict[str, tp.Any]]:
        duration = n_bins / _RATE_HZ
        first, stop = utils.task_span(ref)
        finger_start = (first - offset) / _RATE_HZ
        finger_duration = (stop - first) / _RATE_HZ
        rows = [
            {
                "type": "BandPower",
                "feature": "spike_band_power",
                "start": 0.0,
                "duration": duration,
                "frequency": _RATE_HZ,
                "n_features": utils.N_CHANNELS,
                # The Supplementary Methods give the whole chain for these seven
                # days: the amplifier's 0.3 Hz-7.5 kHz analog filter at 30 kHz, common
                # average referencing, the 250 Hz digital high-pass, then the sum of
                # the squared magnitude per 50 ms window. The analog low-pass is the
                # band's top edge. The 50 ms values sit on the scale of a mean, so the
                # sum was divided by a fixed count, which the 20 ms prefix of every
                # block confirms by sitting at 20/50 of the steady level.
                "lower_freq": 250.0,
                "upper_freq": 7500.0,
                "unit": "uV^2",
                "normalization": "raw",
                "processing": (
                    "0.3 Hz-7.5 kHz analog filter, 30 kHz sampling, 250 nV resolution; "
                    "common average referencing; 250 Hz digital high-pass; "
                    "sum of squared magnitude per 50 ms bin over a fixed count "
                    "(mean scale)"
                ),
                "filepath": study.SpecialLoader(
                    method=self._load_neural, timeline=timeline
                ).to_json(),
            },
            {
                "type": "TargetGraspSeries",
                "feature": "finger_target",
                "start": finger_start,
                "duration": finger_duration,
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
                    "start": finger_start,
                    "duration": finger_duration,
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
        neutral = postures["rest"]
        postures.update({name: neutral for name in utils.OPEN_LOOP_RETURNS.values()})
        return [
            (start - offset, stop - offset, condition, postures[condition])
            for start, stop, condition in utils.cue_windows(ref)
        ]

    def _trial_rows(self, ref: utils.BlockRef, offset: int) -> list[dict[str, tp.Any]]:
        closed_loop = ref.kind == utils.KIND_CLOSED_LOOP
        trials = (
            self._closed_loop_trials(ref, offset)
            if closed_loop
            else self._open_loop_trials(ref, offset)
        )
        success = utils.trial_success(ref) if closed_loop else None
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
                "period": "go",
                **{
                    f"target_{dim}": float(value)
                    for dim, value in zip(utils.DECODED_DIMS, target)
                },
            }
            # A closed-loop target appears when its trial starts, with no instructed
            # delay, so the Action already carries its onset and position.
            if not closed_loop:
                rows.append(
                    {
                        "type": "Stimulus",
                        "modality": "visual",
                        "description": "cued finger posture",
                        "stimulus_kind": "target_cue",
                        "task_condition": "na",
                        **common,
                    }
                )
            if condition == "rest":
                action = {
                    "description": "rest at the neutral posture",
                    "task_condition": "na",
                    "action_kind": "rest",
                }
            else:
                action = {
                    "description": "attempted finger movement toward the cued posture",
                    "task_condition": "attempted",
                    "action_kind": "finger_control",
                }
            if closed_loop:
                action["success"] = bool(success[index])
            rows.append({"type": "Action", **action, **common})
        return rows

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        ref = self._ref(timeline)
        offset = utils.steady_state_start(ref)
        n_bins = utils.get_neural(ref).shape[0] - offset
        rows = self._continuous_rows(timeline, ref, n_bins, offset)
        rows.extend(self._trial_rows(ref, offset))
        return pd.DataFrame(rows)


__all__ = ["Willsey2025FingerDecoding"]
