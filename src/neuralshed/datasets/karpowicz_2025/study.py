"""neuralset Study for Karpowicz et al. 2025 human iBCI cursor control.

One timeline is one task block, following ``wilson_2025``. A session file holds only the
windows the authors kept, so its blocks are separated by gaps of up to 70 minutes and
the block numbering skips; a block, by contrast, is one near-continuous 20 ms grid with
invariant task entities (the calibration block cues eight radial targets, every other
block cues the four cardinal Simon targets).

The emitted clock counts bins rather than wall time. The source drops the odd bin (251
across both sessions, 243 of them in day 427's calibration block), so trial times are
resolved against the source timestamps in ``utils.trials`` and then expressed in bins,
which keeps every task row aligned with the dense streams it indexes into.

Each Simon sequence is shown before it is reproduced: with the cursor held at the centre,
the target stream steps through the sequence's four targets, one every 0.78 s. The whole
presentation is one ``Stimulus`` (``stimulus_kind="movement_cue"``), since what instructs
the four movements is the sequence, as in ``kunz_2025``'s sequence-recall tasks. The paper
lights each target for 400 ms with a tone, somewhere inside its 0.78 s step, and the
release does not say where.
"""

import typing as tp
from pathlib import Path

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


def _intertrial(start: int, stop: int, description: str) -> dict[str, tp.Any]:
    return {
        "type": "Intertrial",
        "start": start / _RATE_HZ,
        "duration": (stop - start) / _RATE_HZ,
        "description": description,
        "period": "intertrial",
    }


def _lead_in_or_tail(row: dict[str, tp.Any], total: int) -> bool:
    if row["type"] != "Intertrial":
        return False
    start = round(row["start"] * _RATE_HZ)
    stop = round((row["start"] + row["duration"]) * _RATE_HZ)
    return start == 0 or stop == total


def _return_rows(
    start: int, stop: int, back: list[tuple[int, int, int, tuple[float, float]]]
) -> list[dict[str, tp.Any]]:
    rows, cursor = [], start
    for back_start, back_stop, origin, centre in back:
        if back_start > cursor:
            rows.append(_intertrial(cursor, back_start, "between movements"))
        rows.append(
            {
                "type": "Action",
                "start": back_start / _RATE_HZ,
                "duration": (back_stop - back_start) / _RATE_HZ,
                "description": (
                    f"return to the centre from {utils.CALIBRATION_NAMES[origin]}"
                ),
                "task_condition": "imagined",
                "action_kind": "cursor_control",
                "period": "go",
                # 11 to 18, so returns from eight directions keep eight labels. The
                # source gives no trial id or outcome for them.
                "code": utils.RETURN_CODE_OFFSET + origin,
                "target_x": centre[0],
                "target_y": centre[1],
            }
        )
        cursor = back_stop
    if stop > cursor:
        rows.append(_intertrial(cursor, stop, "between movements"))
    return rows


def _gap_rows(
    start: int,
    stop: int,
    shown: list[tuple[int, int, tuple[int, ...]]],
    returned: list[tuple[int, int, int, tuple[float, float]]],
    idle: str,
) -> list[dict[str, tp.Any]]:
    """Rows between two movements, splitting out a presentation or a return there."""
    back = [row for row in returned if start <= row[0] < stop]
    if back:
        return _return_rows(start, stop, back)
    inside = [presentation for presentation in shown if start <= presentation[0] < stop]
    if not inside:
        return [_intertrial(start, stop, idle)]
    cue_start, cue_stop, codes = inside[0]
    order = ", ".join(utils.SIMON_NAMES[code] for code in codes)
    rows = []
    if cue_start > start:
        rows.append(_intertrial(start, cue_start, "before the next sequence is shown"))
    rows.append(
        {
            "type": "Stimulus",
            "start": cue_start / _RATE_HZ,
            "duration": (cue_stop - cue_start) / _RATE_HZ,
            "description": f"sequence shown: {order}",
            "stimulus_kind": "movement_cue",
            "modality": "visual",
            "period": "delay",
            # The four targets' `cond_id` as digits in the order shown, so 3142 is
            # left, right, down, up. The same in every Simon block.
            "code": int("".join(str(code) for code in codes)),
        }
    )
    rows.append(
        _intertrial(
            cue_stop, stop, "sequence shown, before the cursor leaves the centre"
        )
    )
    return rows


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
  month = may,
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
    timeline_descriptions: tp.ClassVar[dict[str, dict[tp.Any, str]]] = {
        "task": {
            utils.TASK_CALIBRATION: (
                "In the radial-8 calibration task, eight outer targets alternate with the "
                "centre target, and the participant drives the cursor out to each and "
                "back to the centre. The block fits the session's Kalman filter."
            ),
            utils.TASK_SIMON: (
                "In the Simon task, a sequence of four cardinal targets is shown one at a "
                "time while the cursor is held at the centre, and the participant then "
                "drives the cursor to each in that order. The cursor recentres "
                "automatically after each target is acquired."
            ),
        },
        "loop_type": {
            "assisted": (
                "In this block, computer assistance steps down to none over the first "
                "90 s, after which the decoder alone moves the cursor."
            ),
            "closed": "In this block the decoder alone moves the cursor.",
        },
    }
    requirements: tp.ClassVar[tuple[str, ...]] = ("h5py",)

    # Two timelines covering both sessions and both tasks: day 153's Simon block 5 and
    # day 427's calibration block, the one carrying nearly every dropped bin.
    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=20,
        num_subjects=1,
        query="timeline_index in [1, 11]",
        num_events_in_query=303,
        event_types_in_query={
            "Action",
            "CursorPositionSeries",
            "DecodedCursorVelocitySeries",
            "Intertrial",
            "SpikeCounts",
            "Stimulus",
            "TargetPositionSeries",
        },
        frequency=50.0,
    )

    def _download(self, overwrite: bool = False) -> None:
        """Download the Dryad release into this Study's folder.

        Dryad requires ``NEURALFETCH_DRYAD_TOKEN``. The deposit is two NWB files, 247.2 MB.
        """
        from neuralshed.datasets.dryad import Dryad

        Dryad(
            study="karpowicz_2025",
            dset_dir=self.path.parent,
            folder=self.path.name,
            doi="10.5061/dryad.q83bk3jtp",
        ).download(overwrite=overwrite)

    def _ref(self, timeline: dict[str, tp.Any]) -> utils.BlockRef:
        return utils.BlockRef(
            session=timeline["session"],
            block=timeline["block"],
            is_calibration=timeline["task"] == utils.TASK_CALIBRATION,
            path=Path(self.path) / timeline["path"],
        )

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        for ref in utils.list_blocks(self.path):
            yield {
                "subject": utils.SUBJECT,
                "session": ref.session,
                "block": ref.block,
                "task": ref.task,
                # Calibration starts computer-assisted (Rubin et al. 2022, Fig. 1B).
                "loop_type": (
                    "assisted" if ref.task == utils.TASK_CALIBRATION else "closed"
                ),
                # The paper gives days since implant and no calendar dates, and the
                # deposit de-identifies its own, so this is the only session clock.
                "post_implant_day": utils.POST_IMPLANT_DAY[ref.session],
                "path": ref.path.name,
            }

    def _load_stream(self, timeline: dict[str, tp.Any], key: str) -> np.ndarray:
        """SpecialLoader target: source time-first data as dims-first float32."""
        data = utils.get_stream(self._ref(timeline), key)
        return np.ascontiguousarray(data.T, dtype=np.float32)

    def _stream_rows(
        self, timeline: dict[str, tp.Any], ref: utils.BlockRef
    ) -> list[dict[str, tp.Any]]:
        duration = utils.duration(ref)

        def row(
            *, type_: str, feature: str, key: str, dims=(), unit: str = "", **extra
        ):
            out = {
                "type": type_,
                "feature": feature,
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
                feature="threshold_crossings",
                key="binned_spikes",
                n_features=utils.N_CHANNELS,
                unit="counts/bin",
                normalization="raw",
                method="threshold_crossing",
                # The chain both papers publish. Neither gives the filter band or the
                # threshold, so `threshold` stays empty.
                processing=(
                    "30 kHz sampling; custom real-time signal processing with digital "
                    "downsampling; threshold crossings counted per 20 ms bin"
                ),
            ),
            # The workspace is normalized and the source names the unit "position", so
            # the quantity behind these numbers is unknown: the cued targets sit at
            # radius 0.375 (0.5 in calibration) and the cursor stays inside 0.63 in all
            # but one block.
            row(
                type_="CursorPositionSeries",
                feature="cursor_position",
                key="cursor_pos",
                dims=utils.POSITION_DIMS,
                unit="a.u.",
                kind="cursor_position",
                reference_frame="screen_center",
            ),
            row(
                type_="TargetPositionSeries",
                feature="target_position",
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
                feature="cursor_decoder_output",
                key="cursor_vel",
                dims=utils.POSITION_DIMS,
                unit="a.u./s",
                kind="online_decoder_output",
                reference_frame="screen_center",
            ),
        ]

    def _task_rows(self, ref: utils.BlockRef) -> list[dict[str, tp.Any]]:
        table = utils.trials(ref)
        shown = utils.presentations(ref)
        returned = utils.returns(ref)
        # Only Simon blocks recentre the cursor. In calibration the participant returns it.
        idle = "between movements"
        if not ref.is_calibration:
            idle = "cursor recentred, between movements"
        rows: list[dict[str, tp.Any]] = []
        previous_stop = 0

        for trial in table.itertuples():
            start_bin, stop_bin = int(trial.start_bin), int(trial.stop_bin)
            if stop_bin <= start_bin:
                continue
            if start_bin > previous_stop:
                rows += _gap_rows(previous_stop, start_bin, shown, returned, idle)
            target = np.array([trial.tgt_x, trial.tgt_y])
            rows.append(
                {
                    "type": "Action",
                    "start": start_bin / _RATE_HZ,
                    "duration": (stop_bin - start_bin) / _RATE_HZ,
                    "description": "imagined right-hand movement driving the cursor"
                    + (" (start estimated)" if trial.start_estimated else ""),
                    "task_condition": "imagined",
                    "action_kind": "cursor_control",
                    "period": "go",
                    "trial_id": int(trial.trial_id),
                    # Block-local: `cond_id` numbers different targets in a calibration
                    # block than in a Simon block, so the geometry travels with it.
                    "code": int(trial.cond_id),
                    "target_x": float(trial.tgt_x),
                    "target_y": float(trial.tgt_y),
                    "target_angle_deg": float(np.degrees(np.arctan2(*target[::-1]))),
                    # The sequence's outcome, the source's `result`, shared by every
                    # movement of the sequence.
                    "success": bool(trial.result),
                }
            )
            previous_stop = stop_bin

        total = utils.n_bins(ref)
        if total > previous_stop:
            rows += _gap_rows(previous_stop, total, shown, returned, idle)
        # Before the first trial and after the last is a block lead-in and tail, not a
        # gap between two trials, so it stays unlabelled, as in singer_2025.
        return [row for row in rows if not _lead_in_or_tail(row, total)]

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        ref = self._ref(timeline)
        rows = self._stream_rows(timeline, ref) + self._task_rows(ref)
        return pd.DataFrame(rows)
