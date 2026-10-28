"""neuralset Study for Singer-Clark et al. 2025 cursor control and click.

One timeline is one released 3 to 5 minute task block. Neural features, cursor
kinematics, online decoder outputs, and assist state remain whole-block lazy
streams. Trial timing becomes visual ``Stimulus`` and imagined ``Action`` rows.
Simultaneous speech blocks additionally emit word prompts, beep cues, attempted
speech actions, and intertrial gaps.
"""

import typing as tp

import numpy as np
import pandas as pd

from neuralset.events import study

from neuralshed.datasets.study import NeuralshedStudy
from neuralshed.events import (  # noqa: F401, registers custom types
    BandPower,
    ClickAssistSeries,
    ClickStateSeries,
    CursorAssistSeries,
    CursorPositionSeries,
    DecodedCursorVelocitySeries,
    Intertrial,
    SpikeCounts,
    TargetPositionSeries,
)

from . import utils

_SUBJECT = "T15"
_RATE_HZ = utils.FS
_BIN_SIZE_S = 1.0 / _RATE_HZ


class Singer2025CursorControl(NeuralshedStudy):
    """Motor-imagery cursor control from T15's speech motor cortex arrays."""

    dataset_id: tp.ClassVar[str] = "singer_2025"
    aliases: tp.ClassVar[tuple[str, ...]] = (
        "singer_2025",
        "speech-motor-cortex-cursor",
    )
    bibtex: tp.ClassVar[str] = """
@article{singerclark2025cursor,
  title = {Speech motor cortex enables BCI cursor control and click},
  author = {Singer-Clark, Tyler and Hou, Xianda and Card, Nicholas S. and Wairagkar, Maitreyee and Iacobacci, Carrina and Peracha, Hamza and Hochberg, Leigh R. and Stavisky, Sergey D. and Brandman, David M.},
  journal = {Journal of Neural Engineering},
  volume = {22},
  number = {3},
  pages = {036015},
  year = {2025},
  doi = {10.1088/1741-2552/add0e5},
  url = {https://doi.org/10.1088/1741-2552/add0e5},
}
"""
    url: tp.ClassVar[str] = "https://doi.org/10.5061/dryad.prr4xgxzq"
    licence: tp.ClassVar[str] = "CC0-1.0"
    description: tp.ClassVar[str] = (
        "Two-dimensional cursor control and click by BrainGate2 participant T15 using "
        "motor imagery and four 64-channel Utah arrays in left ventral precentral gyrus. "
        "The released 100 Hz streams include threshold crossings, spike-band power, "
        "cursor and target positions, assist state, and online cursor/click decoder output."
    )
    requirements: tp.ClassVar[tuple[str, ...]] = ("scipy",)

    # One timeline per task family, so the guard covers every emission path: the
    # click streams (grid), and the word/beep/intertrial rows (speech).
    # NB: `neuralfetch.utils.update_source_info` computes the counts from this
    # `query` but then omits `query` and `frequency` when it rewrites the block, so
    # re-add both by hand after running it, or the counts guard the wrong query.
    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=51,
        num_subjects=1,
        query="timeline_index in [0, 8, 36]",
        num_events_in_query=400,
        event_types_in_query={
            "Action",
            "BandPower",
            "ClickAssistSeries",
            "ClickStateSeries",
            "CursorAssistSeries",
            "CursorPositionSeries",
            "DecodedCursorVelocitySeries",
            "Intertrial",
            "SpikeCounts",
            "Stimulus",
            "TargetPositionSeries",
            "Word",
        },
        frequency=100.0,
    )

    def _download(self) -> None:
        """Download the current Dryad release into this Study's folder.

        Dryad requires ``NEURALFETCH_DRYAD_TOKEN``. The speech-only comparison file is
        downloaded for reference but excluded from :meth:`iter_timelines`.
        """
        from neuralshed.datasets.dryad import Dryad

        Dryad(
            study="singer_2025",
            dset_dir=self.path.parent,
            folder=self.path.name,
            doi="10.5061/dryad.prr4xgxzq",
            include=["README.md", "t15_day*_task.mat", "t15_day00202_words.mat"],
        ).download()

    def _ref(self, timeline: dict[str, tp.Any]) -> utils.BlockRef:
        return utils.block_ref(self.path / timeline["path"])

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        for ref in utils.list_blocks(self.path):
            fields = utils.available_fields(ref)
            if ref.task == "radial8_calibration":
                split = "calibration"
            elif ref.task == "grid_evaluation":
                split = "evaluation"
            else:
                split = "interference"
            yield {
                "subject": _SUBJECT,
                "session": ref.session,
                "run": ref.run,
                "task": ref.task,
                "path": ref.path.name,
                "post_implant_day": ref.day,
                "original_split": split,
                "cursor_imagery": utils.CURSOR_IMAGERY_BY_DAY.get(
                    ref.day, "motor imagery"
                ),
                "click_enabled": "click_decoder_output" in fields,
                "speech_condition": self._speech_condition(ref),
                **utils.geometry(ref),
            }

    @staticmethod
    def _speech_condition(ref: utils.BlockRef) -> str:
        if ref.task != "simultaneous_speech_and_cursor":
            return "none"
        return (
            "control"
            if bool(utils.scalar_field(ref, "is_control_block", False))
            else "verbal"
        )

    def _load_series(self, timeline: dict[str, tp.Any], key: str) -> np.ndarray:
        """SpecialLoader target: source time-first data as dims-first float32."""
        data = utils.get_series(self._ref(timeline), key)
        return np.ascontiguousarray(data.T, dtype=np.float32)

    def _continuous_rows(
        self, timeline: dict[str, tp.Any], ref: utils.BlockRef
    ) -> list[dict[str, tp.Any]]:
        duration = utils.duration(ref)
        fields = utils.available_fields(ref)

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
            row(
                type_="SpikeCounts",
                key="threshold_crossings",
                n_features=utils.N_ELECTRODES,
                unit="counts/bin",
                normalization="raw",
                method="threshold_crossing",
                threshold="-4.5 x RMS",
            ),
            row(
                type_="BandPower",
                key="spike_band_power",
                n_features=utils.N_ELECTRODES,
                unit="uV^2",
                normalization="raw",
                lower_freq=250.0,
                upper_freq=5000.0,
            ),
            row(
                type_="CursorPositionSeries",
                key="cursor_position",
                dims=utils.POSITION_DIMS,
                unit="screen_height",
                kind="cursor_position",
                reference_frame="screen_center",
            ),
            row(
                type_="TargetPositionSeries",
                key="target_position",
                dims=utils.POSITION_DIMS,
                unit="screen_height",
                kind="target_position",
                reference_frame="screen_center",
            ),
            row(
                type_="DecodedCursorVelocitySeries",
                key="cursor_decoder_output",
                dims=utils.POSITION_DIMS,
                unit="screen_height/s",
                kind="online_decoder_output",
                reference_frame="screen",
            ),
            row(
                type_="CursorAssistSeries",
                key="assist_amount",
                dims=("cursor_assist",),
                unit="proportion",
                kind="computer_assist",
            ),
        ]
        if "click_assist" in fields:
            rows.append(
                row(
                    type_="ClickAssistSeries",
                    key="click_assist",
                    dims=("click_assist",),
                    unit="binary",
                    kind="computer_assist",
                )
            )
        if "click_decoder_output" in fields:
            rows.append(
                row(
                    type_="ClickStateSeries",
                    key="click_decoder_output",
                    dims=("click",),
                    unit="binary",
                    kind="online_decoder_output",
                )
            )
        return rows

    @staticmethod
    def _target_code(ref: utils.BlockRef, position: np.ndarray) -> int | None:
        """Radial8: 0-7 by 45-degree spoke, 8 for the center target.

        Grid: the row-major cell index (0 to ``grid_num_rows ** 2 - 1``).
        """
        if ref.task == "grid_evaluation":
            return utils.grid_cell_index(ref, position)
        if np.allclose(position, 0.0):
            return 8
        angle = float(np.arctan2(position[1], position[0])) % (2 * np.pi)
        return int(np.round(angle / (np.pi / 4))) % 8

    def _trial_rows(self, ref: utils.BlockRef) -> list[dict[str, tp.Any]]:
        data = utils.load_block(ref)
        windows = utils.trial_windows(ref)
        targets = utils.get_series(ref, "target_position")
        presentation = utils.trial_field(ref, "target_presentation_bin")
        cursor_go = utils.trial_field(ref, "cursor_go_cue_bin")
        speech_go = utils.trial_field(ref, "speech_go_cue_bin")
        prompts = utils.speech_prompts(ref)

        # Only the speech task separates trial start from target presentation, so only
        # it has a real intertrial period. Elsewhere the gap before the first trial is
        # a block lead-in, not an intertrial, and is left unlabelled.
        has_intertrial = presentation is not None
        rows: list[dict[str, tp.Any]] = []
        previous_stop = windows[0][0] if windows else 0
        for trial, (start, stop) in enumerate(windows):
            present = int(presentation[trial]) if presentation is not None else start
            go = int(cursor_go[trial]) if cursor_go is not None else start
            present = min(max(present, start), stop)
            go = min(max(go, present), stop)
            target = targets[min(go, len(targets) - 1)]
            code = self._target_code(ref, target)

            if has_intertrial and present > previous_stop:
                rows.append(
                    {
                        "type": "Intertrial",
                        "start": previous_stop / _RATE_HZ,
                        "duration": (present - previous_stop) / _RATE_HZ,
                        "description": "gap before target presentation",
                        "trial_id": trial,
                    }
                )

            rows.append(
                {
                    "type": "Stimulus",
                    "start": present / _RATE_HZ,
                    "duration": (stop - present) / _RATE_HZ,
                    "code": code,
                    "modality": "visual",
                    "description": "cued cursor target",
                    "stimulus_kind": "target_cue",
                    "trial_id": trial,
                    "target_x": float(target[0]),
                    "target_y": float(target[1]),
                }
            )
            rows.append(
                {
                    "type": "Action",
                    "start": go / _RATE_HZ,
                    "duration": (stop - go) / _RATE_HZ,
                    "code": code,
                    "description": "imagined cursor movement toward target",
                    "task_condition": "imagined",
                    "action_kind": "cursor_control",
                    "trial_id": trial,
                    "target_x": float(target[0]),
                    "target_y": float(target[1]),
                }
            )

            if prompts and trial < len(prompts) and prompts[trial]:
                rows.append(
                    {
                        "type": "Word",
                        "start": present / _RATE_HZ,
                        "duration": max(_BIN_SIZE_S, (go - present) / _RATE_HZ),
                        "text": prompts[trial],
                        "language": "en",
                        "modality": "read",
                        "trial_id": trial,
                    }
                )
            if speech_go is not None and trial < len(speech_go):
                beep = int(speech_go[trial])
                if beep >= 0 and beep < stop:
                    rows.append(
                        {
                            "type": "Stimulus",
                            "start": beep / _RATE_HZ,
                            "duration": _BIN_SIZE_S,
                            "code": 1,
                            "modality": "audio",
                            "description": "speech go cue beep",
                            "stimulus_kind": "speech_go_cue",
                            "trial_id": trial,
                        }
                    )
                    if not bool(data.get("is_control_block", False)):
                        rows.append(
                            {
                                "type": "Action",
                                "start": beep / _RATE_HZ,
                                "duration": (stop - beep) / _RATE_HZ,
                                "code": 1,
                                "description": "attempted prompted speech",
                                "task_condition": "attempted",
                                "action_kind": "speech",
                                "trial_id": trial,
                            }
                        )
            previous_stop = max(previous_stop, stop)
        return rows

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        ref = self._ref(timeline)
        rows = self._continuous_rows(timeline, ref)
        rows.extend(self._trial_rows(ref))
        return pd.DataFrame(rows)


__all__ = ["Singer2025CursorControl"]
