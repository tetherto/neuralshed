"""neuralset Study for Karpowicz et al. 2025 human iBCI cursor control.

One timeline is one task block, following ``wilson_2025``. A session file holds only the
windows the authors kept, so its blocks are separated by gaps of up to 70 minutes and
the block numbering skips; a block, by contrast, is one near-continuous 20 ms grid with
invariant task entities (the calibration block cues eight radial targets, every other
block cues the four cardinal Simon targets).

The emitted clock counts bins rather than wall time. The source drops the odd bin (246
across both sessions, 243 of them in day 427's calibration block), so trial times are
resolved against the source timestamps in ``utils.trials`` and then expressed in bins,
which keeps every task row aligned with the dense streams it indexes into.

No ``Stimulus`` row is emitted: the release publishes no cue timing. A trial begins at
the cursor's outward zero-crossing, by which point the target has already been
illuminated, so the cued target rides the ``Action`` rows and the continuous
``TargetPositionSeries`` instead of an invented cue window.
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
    TargetPositionSeries,
)

from . import utils

_RATE_HZ = utils.FS


class Karpowicz2025CursorStability(NeuralshedStudy):
    """Two T11 cursor sessions 274 days apart, for decoder-stability work."""

    dataset_id: tp.ClassVar[str] = "karpowicz_2025"
    aliases: tp.ClassVar[tuple[str, ...]] = (
        "karpowicz_2025",
        "nomad",
        "cursor-stability",
    )
    bibtex: tp.ClassVar[str] = """
@article{karpowicz2025stabilizing,
  title = {Stabilizing brain-computer interfaces through alignment of latent dynamics},
  author = {Karpowicz, Brianna M. and Ali, Yahia H. and Wimalasena, Lahiru N. and Sedler, Andrew R. and Keshtkaran, Mohammad Reza and Bodkin, Kevin and Ma, Xuan and Rubin, Daniel B. and Williams, Ziv M. and Cash, Sydney S. and Hochberg, Leigh R. and Miller, Lee E. and Pandarinath, Chethan},
  journal = {Nature Communications},
  volume = {16},
  pages = {4662},
  year = {2025},
  doi = {10.1038/s41467-025-59652-y},
  url = {https://doi.org/10.1038/s41467-025-59652-y},
}
"""
    url: tp.ClassVar[str] = "https://doi.org/10.5061/dryad.q83bk3jtp"
    licence: tp.ClassVar[str] = "CC0-1.0"
    description: tp.ClassVar[str] = (
        "Closed-loop two-dimensional cursor control by BrainGate2 participant T11 using "
        "two 96-channel Utah arrays in the left precentral gyrus, on two sessions 153 and "
        "427 days after implant. Each session opens with a radial-8 calibration block and "
        "continues with Simon memory-matching blocks in which the participant imagines "
        "right-hand movements to drive the cursor through a sequence of four cardinal "
        "targets. The released 50 Hz streams are threshold crossings on 192 channels plus "
        "cursor position, cursor velocity and target position."
    )
    requirements: tp.ClassVar[tuple[str, ...]] = ("h5py",)

    # Two timelines covering both sessions and both tasks: day 153's Simon block 5 and
    # day 427's calibration block, the one carrying nearly every dropped bin.
    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=20,
        num_subjects=1,
        query="timeline_index in [1, 11]",
        num_events_in_query=226,
        event_types_in_query={
            "Action",
            "CursorPositionSeries",
            "DecodedCursorVelocitySeries",
            "Intertrial",
            "SpikeCounts",
            "TargetPositionSeries",
        },
        frequency=50.0,
    )

    def _download(self) -> None:
        """Download the Dryad release into this Study's folder.

        Dryad requires ``NEURALFETCH_DRYAD_TOKEN``. The deposit is two NWB files, 262 MB.
        """
        from neuralshed.datasets.dryad import Dryad

        Dryad(
            study="karpowicz_2025",
            dset_dir=self.path.parent,
            folder=self.path.name,
            doi="10.5061/dryad.q83bk3jtp",
        ).download()

    def _ref(self, timeline: dict[str, tp.Any]) -> utils.BlockRef:
        return utils.BlockRef(
            session=timeline["session"],
            block=timeline["block"],
            is_calibration=timeline["is_calibration"],
            path=utils.session_path(self.path, timeline["session"]),
        )

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        for ref in utils.list_blocks(self.path):
            yield {
                "subject": utils.SUBJECT,
                "session": ref.session,
                "block": ref.block,
                "task": ref.task,
                "is_calibration": ref.is_calibration,
                # The paper gives days since implant and no calendar dates, and the
                # deposit de-identifies its own, so this is the only session clock.
                "post_implant_day": utils.POST_IMPLANT_DAY[ref.session],
                "n_targets": len(utils.target_layout(ref)),
                "dropped_bins": utils.dropped_bins(ref),
            }

    def _load_stream(self, timeline: dict[str, tp.Any], key: str) -> np.ndarray:
        """SpecialLoader target: source time-first data as dims-first float32."""
        data = utils.get_stream(self._ref(timeline), key)
        return np.ascontiguousarray(data.T, dtype=np.float32)

    def _stream_rows(
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
                    method=self._load_stream,
                    timeline=timeline,
                    key=key,
                ).to_json(),
                **extra,
            }
            if dims:
                out["dims"] = dims
                out["n_features"] = len(dims)
            return out

        return [
            row(
                type_="SpikeCounts",
                key="binned_spikes",
                n_features=utils.N_CHANNELS,
                unit="counts/bin",
                normalization="raw",
                method="threshold_crossing",
            ),
            # The workspace is normalized and the source names the unit "position", so
            # the quantity behind these numbers is unknown: the cued targets sit at
            # radius 0.375 (0.5 in calibration) and the cursor stays inside 0.63.
            row(
                type_="CursorPositionSeries",
                key="cursor_pos",
                dims=utils.POSITION_DIMS,
                unit="a.u.",
                kind="cursor_position",
                reference_frame="screen_center",
            ),
            row(
                type_="TargetPositionSeries",
                key="target_pos",
                dims=utils.POSITION_DIMS,
                unit="a.u.",
                kind="target_position",
                reference_frame="screen_center",
            ),
            # Scaled by the bin rate in `utils.get_stream`, since the source ships a
            # per-bin displacement under a "position/s" label.
            row(
                type_="DecodedCursorVelocitySeries",
                key="cursor_vel",
                dims=utils.POSITION_DIMS,
                unit="a.u./s",
                kind="online_decoder_output",
                reference_frame="screen_center",
            ),
        ]

    def _task_rows(self, ref: utils.BlockRef) -> list[dict[str, tp.Any]]:
        table = utils.trials(ref)
        rows: list[dict[str, tp.Any]] = []
        previous_stop = 0

        for trial in table.itertuples():
            start_bin, stop_bin = int(trial.start_bin), int(trial.stop_bin)
            if stop_bin <= start_bin:
                continue
            if start_bin > previous_stop:
                rows.append(
                    {
                        "type": "Intertrial",
                        "start": previous_stop / _RATE_HZ,
                        "duration": (start_bin - previous_stop) / _RATE_HZ,
                        "description": "cursor recentred, between movements",
                    }
                )
            target = np.array([trial.tgt_x, trial.tgt_y])
            rows.append(
                {
                    "type": "Action",
                    "start": start_bin / _RATE_HZ,
                    "duration": (stop_bin - start_bin) / _RATE_HZ,
                    "description": "imagined right-hand movement driving the cursor",
                    "task_condition": "imagined",
                    "action_kind": "cursor_control",
                    "trial_id": int(trial.trial_id),
                    # Block-local: `cond_id` numbers different targets in a calibration
                    # block than in a Simon block, so the geometry travels with it.
                    "code": int(trial.cond_id),
                    "target_x": float(trial.tgt_x),
                    "target_y": float(trial.tgt_y),
                    "target_angle_deg": float(np.degrees(np.arctan2(*target[::-1]))),
                    # 1 when every one of the sequence's four targets was acquired.
                    # `n_targets` and `is_calibration` stay off these rows: both are
                    # block-level, so they ride the timeline dict instead.
                    "success": bool(trial.result),
                }
            )
            previous_stop = stop_bin

        total = utils.n_bins(ref)
        if total > previous_stop:
            rows.append(
                {
                    "type": "Intertrial",
                    "start": previous_stop / _RATE_HZ,
                    "duration": (total - previous_stop) / _RATE_HZ,
                    "description": "cursor recentred, between movements",
                }
            )
        return rows

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        ref = self._ref(timeline)
        rows = self._stream_rows(timeline, ref) + self._task_rows(ref)
        return pd.DataFrame(rows)
