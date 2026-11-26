"""neuralset Study for Athalye et al. 2023, monkey 2D cursor BMI (DANDI 000404).

One timeline is one task entry (block). A session file concatenates its blocks back to
back, center-out first whatever the recording order, so the file clock jumps at every
``te_num`` seam, while a block is one continuous recording on one grid. Timelines are
emitted in file order.

Everything is closed-loop brain control with a decoder fixed for the session, and the
cursor is the decoder's position state. Streams are emitted at the source rate, 60 Hz for
monkey G and 200 Hz for monkey J, and the extractor resamples. Monkey J's positions are
scaled from m to cm on read, so both monkeys declare cm and cm/s. Four monkey J blocks end
in zero-filled padding left by the writer, and every row of those timelines stops where
the recorded data stop (``utils.DEAD_TAIL_START``).

No target stream is emitted: monkey G's ``target_state`` is all zeros, and monkey J's
``target_state`` and ``obs_details`` are written against the wrong index. Monkey G's
obstacle geometry, which is aligned, rides the ``Action`` rows of obstacle trials.
"""

import typing as tp

import numpy as np
import pandas as pd
from neuralset.events import study

from neuralshed.datasets.study import NeuralshedStudy
from neuralshed.events import (  # noqa: F401, registers custom types
    CursorPositionSeries,
    DecodedCursorVelocitySeries,
    Intertrial,
    SpikeCounts,
)

from . import utils


class Athalye2023CursorBMI(NeuralshedStudy):
    """Two macaques moving a 2D cursor under brain control, center-out and obstacle tasks."""

    dataset_id: tp.ClassVar[str] = "athalye_2023"
    aliases: tp.ClassVar[tuple[str, ...]] = (
        "athalye_2023",
        "dandi_000404",
        "monkey-2d-cursor-bmi",
    )
    url: tp.ClassVar[str] = "https://dandiarchive.org/dandiset/000404"
    bibtex: tp.ClassVar[str] = """
@article{athalye2023invariant,
  title = {Invariant neural dynamics drive commands to control different movements},
  author = {Athalye, Vivek R. and Khanna, Preeya and Gowda, Suraj and Orsborn, Amy L. and Costa, Rui M. and Carmena, Jose M.},
  journal = {Current Biology},
  volume = {33},
  number = {14},
  pages = {2962--2976.e15},
  year = {2023},
  doi = {10.1016/j.cub.2023.06.027},
  url = {https://doi.org/10.1016/j.cub.2023.06.027},
}
"""
    licence: tp.ClassVar[str] = "CC-BY-4.0"
    description: tp.ClassVar[str] = (
        "Two rhesus macaques (G and J) move a 2D cursor under closed-loop brain control "
        "with a decoder fixed within each session, on center-out and obstacle-avoidance "
        "tasks. Chronic microwire arrays in M1 and PMd of both hemispheres. Released as "
        "the spike counts of the decoder's own units, the cursor and the decoder state: "
        "monkey G at 60 Hz (Kalman filter updated at 10 Hz), monkey J at 200 Hz "
        "(point-process filter)."
    )
    requirements: tp.ClassVar[tuple[str, ...]] = ("dandi", "h5py", "remfile")

    # One monkey G obstacle block and one monkey J center-out block that ends in padding,
    # so both rates, both tasks, the obstacle attributes and the padding cut are exercised.
    # No `frequency`: the query mixes 60 Hz and 200 Hz timelines.
    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=34,
        num_subjects=2,
        query="timeline_index in [1, 28]",
        num_events_in_query=242,
        event_types_in_query={
            "Action",
            "CursorPositionSeries",
            "DecodedCursorVelocitySeries",
            "Intertrial",
            "SpikeCounts",
        },
    )

    def _download(self) -> None:
        from neuralfetch.download import Dandi

        # folder="" mirrors the dandiset under self.path: self.path/000404/sub-*/...nwb
        Dandi(
            study=utils.DANDISET_ID,
            version=utils.DANDI_VERSION,
            dset_dir=self.path,
            folder="",
        ).download()

    def _ref(self, timeline: dict[str, tp.Any]) -> utils.BlockRef:
        return utils.block(timeline["path"], timeline["task_entry"], self.path)

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        for relpath in utils.list_assets(self.path):
            for ref in utils.blocks(relpath, self.path):
                yield {
                    "subject": ref.subject,
                    "session": ref.session,
                    "task_entry": ref.task_entry,
                    "task": ref.task,
                    "loop_type": "closed",
                    "path": relpath,
                }

    def _load_stream(self, timeline: dict[str, tp.Any], key: str) -> np.ndarray:
        """SpecialLoader target: one stream of the block, dims-first float32."""
        ref = self._ref(timeline)
        reader = {
            "spike_counts": utils.get_spike_counts,
            "cursor": utils.get_cursor,
            "decoder_velocity": utils.get_decoder_velocity,
        }[key]
        return np.ascontiguousarray(reader(ref).T, dtype=np.float32)

    def _stream_rows(
        self, timeline: dict[str, tp.Any], ref: utils.BlockRef
    ) -> list[dict[str, tp.Any]]:
        def row(type_: str, key: str, feature: str, **extra: tp.Any) -> dict:
            return {
                "type": type_,
                "feature": feature,
                "start": 0.0,
                "duration": ref.duration,
                "frequency": ref.rate,
                "filepath": study.SpecialLoader(
                    method=self._load_stream, timeline=timeline, key=key
                ).to_json(),
                **extra,
            }

        kinematics = {
            "dims": utils.POSITION_DIMS,
            "n_features": len(utils.POSITION_DIMS),
            # G's center target sits at the origin, J's at about (4.1, 14.0) cm.
            "reference_frame": "workspace",
        }
        return [
            row(
                "SpikeCounts",
                "spike_counts",
                "sorted_spike_counts",
                n_features=utils.n_units(ref.source),
                unit="counts/bin",
                normalization="raw",
                processing="",
                method="sorted",
                threshold="",
            ),
            row(
                "CursorPositionSeries",
                "cursor",
                "cursor_position",
                unit="cm",
                kind="cursor_position",
                **kinematics,
            ),
            # The decoder's velocity state, which drives the cursor.
            row(
                "DecodedCursorVelocitySeries",
                "decoder_velocity",
                "cursor_velocity",
                unit="cm/s",
                kind="cursor_velocity",
                **kinematics,
            ),
        ]

    def _task_rows(self, ref: utils.BlockRef) -> list[dict[str, tp.Any]]:
        rows: list[dict[str, tp.Any]] = []
        previous_stop = 0.0

        def gap(start: float, stop: float) -> None:
            if stop > start:
                rows.append(
                    {
                        "type": "Intertrial",
                        "start": start,
                        "duration": stop - start,
                        "description": "between rewarded trials, unlabeled",
                    }
                )

        for trial in utils.trials(ref).itertuples():
            gap(previous_stop, trial.start)
            rows.append(
                {
                    "type": "Action",
                    "start": trial.start,
                    "duration": trial.stop - trial.start,
                    "description": "brain-controlled cursor movement, go cue to reward",
                    "task_condition": "other",
                    "action_kind": "cursor_control",
                    "trial_id": int(trial.trial_id),
                    "success": True,  # the release labels rewarded trials only
                    "obstacle_x": float(trial.obstacle_x),
                    "obstacle_y": float(trial.obstacle_y),
                    "obstacle_size_cm": float(trial.obstacle_size_cm),
                }
            )
            previous_stop = trial.stop
        gap(previous_stop, ref.duration)
        return rows

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        ref = self._ref(timeline)
        return pd.DataFrame(self._stream_rows(timeline, ref) + self._task_rows(ref))
