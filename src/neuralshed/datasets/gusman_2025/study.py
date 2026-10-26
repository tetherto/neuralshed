"""neuralset Study for Gusman et al. 2025 multi-gesture drag-and-drop.

One timeline is one **session**, following ``kunz_2025``: the source concatenates a
session's blocks onto a single continuous 20 ms grid with the inter-block gaps removed,
and the task tables index into that grid, so a session reads as one recording.

Each timeline emits seven whole-session neural streams and one row per task epoch. The
streams are threshold crossings, spike-band power, and the five local-field-potential
bands, one event each. Every band carries the edges the paper's Methods define, so a
band is selected by
``query="feature == 'lfp_12_19'"`` instead of by slicing a stacked array.

Task epochs differ by task. Gesture Hero gives one trial per cued gesture, with the hold
duration on the row. Drag-and-drop gives one **stage** per row, so a trial spans several,
and the closed-loop flags are per stage and per decoder rather than per block.
"""

import typing as tp

import numpy as np
import pandas as pd

from neuralset.events import study

from neuralshed.datasets.study import NeuralshedStudy
from neuralshed.events import (  # noqa: F401, registers custom types
    BandPower,
    Intertrial,
    SpikeCounts,
)

from . import utils

_RATE_HZ = utils.FS
_LFP_BANDS = utils.FEATURE_NAMES[2:]  # the five STFT bands
_INTERTRIAL = "Intertrial"


class Gusman2025DragAndDrop(NeuralshedStudy):
    """Sustained hand gestures held during 2D cursor control, T11 and T5."""

    dataset_id: tp.ClassVar[str] = "gusman_2025"
    aliases: tp.ClassVar[tuple[str, ...]] = (
        "gusman_2025",
        "drag-and-drop",
        "latch-decoder",
    )
    bibtex: tp.ClassVar[str] = """
@article{gusman2025draganddrop,
  title = {Multi-gesture drag-and-drop decoding in a 2D iBCI control task},
  author = {Gusman, Jacob T. and Hosman, Tommy and Crawford, Jacob and Singer-Clark, Tyler and Kapitonava, Anastasia and Kelemen, Jessica N. and Hahn, Nick and Henderson, Jaimie M. and Hochberg, Leigh R. and Simeral, John D. and Vargas-Irwin, Carlos E.},
  journal = {Journal of Neural Engineering},
  volume = {22},
  number = {2},
  pages = {026054},
  year = {2025},
  doi = {10.1088/1741-2552/adb180},
  url = {https://doi.org/10.1088/1741-2552/adb180},
}
"""
    url: tp.ClassVar[str] = "https://doi.org/10.5061/dryad.98sf7m0v1"
    licence: tp.ClassVar[str] = "CC0-1.0"
    description: tp.ClassVar[str] = (
        "Sustained hand gestures held during 2D cursor control by BrainGate2 "
        "participants T11 and T5, two 96-channel Utah arrays each in the hand knob of "
        "left precentral gyrus. An open-loop Gesture Hero task measures how a gesture "
        "representation decays over 1, 2 and 4 second holds, and a closed-loop "
        "drag-and-drop task uses a latch decoder to hold a gesture while the cursor "
        "moves. Seven feature types per electrode at 50 Hz, five of them LFP bands."
    )
    requirements: tp.ClassVar[tuple[str, ...]] = ("h5py",)

    # Two timelines covering every emission path: a drag-and-drop session, which is the
    # only one carrying the online decode and the Intertrial rows, and a Gesture Hero
    # session, which is the only one carrying a cued hold duration.
    # NB: `update_source_info` drops `query` and `frequency` when it rewrites this block,
    # so re-add both by hand after running it.
    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=5,
        num_subjects=2,
        query="timeline_index in [0, 2]",
        num_events_in_query=2700,
        event_types_in_query={
            "Action",
            "BandPower",
            "BehavioralSeries",
            "Intertrial",
            "SpikeCounts",
            "Stimulus",
        },
        frequency=50.0,
    )

    def _download(self) -> None:
        """Download the Dryad release into this Study's folder.

        Dryad requires ``NEURALFETCH_DRYAD_TOKEN``. The deposit unpacks to three MATLAB
        v7.3 files totalling 7.7 GB.
        """
        from neuralshed.datasets.dryad import Dryad

        Dryad(
            study="gusman_2025",
            dset_dir=self.path.parent,
            folder=self.path.name,
            doi="10.5061/dryad.98sf7m0v1",
        ).download()

    def _ref(self, timeline: dict[str, tp.Any]) -> utils.SessionRef:
        return utils.SessionRef(
            participant=timeline["subject"],
            task=timeline["task_kind"],
            session=timeline["session_number"],
            index=timeline["file_index"],
            path=self.path / timeline["file"],
        )

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        for ref in utils.list_sessions(self.path):
            yield {
                "subject": ref.participant,
                "session": ref.label,  # matches metadata.sessions()'s label
                "task": (
                    "gesturehero"
                    if ref.task == utils.TASK_GESTURE_HERO
                    else "draganddrop"
                ),
                "task_kind": ref.task,
                "session_number": ref.session,
                "file_index": ref.index,
                "file": ref.path.name,
                # Whole-session labels worth filtering on. The loop type is a property of
                # the task here; within drag-and-drop it varies per stage instead, and
                # those flags ride the Action rows.
                "loop_type": (
                    "open" if ref.task == utils.TASK_GESTURE_HERO else "closed"
                ),
                "n_gestures": len(utils.gestures(ref)),
            }

    def _load_stream(self, timeline: dict[str, tp.Any], feature: str) -> np.ndarray:
        """SpecialLoader target: one feature type as ``(192, n_bins)`` float32."""
        data = utils.get_features(self._ref(timeline), feature=feature)
        return np.ascontiguousarray(data, dtype=np.float32)

    def _load_decoded_state(self, timeline: dict[str, tp.Any]) -> np.ndarray:
        """SpecialLoader target: the latch decoder's per-bin class index, ``(1, n_bins)``."""
        ref = self._ref(timeline)
        labels = utils.decoded_state(ref)
        classes = sorted(set(labels))
        lookup = {name: index for index, name in enumerate(classes)}
        return np.array([[lookup[name] for name in labels]], dtype=np.float32)

    def _neural_rows(
        self, timeline: dict[str, tp.Any], duration: float
    ) -> list[dict[str, tp.Any]]:
        def loader(method, **kwargs):
            return study.SpecialLoader(
                method=method, timeline=timeline, **kwargs
            ).to_json()

        common = {
            "start": 0.0,
            "duration": duration,
            "frequency": _RATE_HZ,
            # One event per feature type, so each carries the electrode count rather than
            # the 1,344 columns the source stacks them into.
            "n_features": utils.N_CHANNELS,
            # Every feature is z-scored against a rolling 3 minute window before release,
            # so the values are dimensionless and no bin-rate conversion applies. `unit`
            # names the quantity the source's chain took as input, per standardization.md.
            "normalization": "z_scored",
            "processing": f"z-scored against a rolling {utils.NORMALIZATION_WINDOW_S} s window",
        }
        return [
            {
                "type": "SpikeCounts",
                "feature": "threshold_crossings",
                "unit": "counts/bin",
                "method": "threshold_crossing",
                "filepath": loader(self._load_stream, feature="ncTX"),
                **common,
            },
            {
                "type": "BandPower",
                "feature": "spike_band_power",
                "unit": "uV^2",
                "lower_freq": utils.SPIKE_BAND_HZ[0],
                "upper_freq": utils.SPIKE_BAND_HZ[1],
                "filepath": loader(self._load_stream, feature="spikePower"),
                **common,
            },
            *[
                {
                    "type": "BandPower",
                    "feature": utils.LFP_FEATURE_NAMES[band],
                    "unit": "uV^2",
                    "lower_freq": utils.LFP_BAND_EDGES[band][0],
                    "upper_freq": utils.LFP_BAND_EDGES[band][1],
                    "filepath": loader(self._load_stream, feature=band),
                    **common,
                }
                for band in _LFP_BANDS
            ],
        ]

    def _epoch_rows(
        self, timeline: dict[str, tp.Any], ref: utils.SessionRef
    ) -> list[dict[str, tp.Any]]:
        table = utils.epochs(ref)
        gesture_hero = ref.task == utils.TASK_GESTURE_HERO
        rows: list[dict[str, tp.Any]] = []
        for index, row in enumerate(table.itertuples(index=False)):
            span = {
                "start": row.start / _RATE_HZ,
                "duration": (row.stop - row.start) / _RATE_HZ,
                "block": int(row.block),
                "epoch_id": index,
                "gesture": row.label,
            }
            if gesture_hero:
                rows.append(
                    {
                        "type": "Stimulus",
                        "modality": "visual",
                        "description": "falling instruction box naming the gesture to hold",
                        "stimulus_kind": "gesture_cue",
                        **span,
                    }
                )
                rows.append(
                    {
                        "type": "Action",
                        "description": "attempted gesture, held for the cued duration",
                        "task_condition": "attempted",
                        "action_kind": "gesture_hold",
                        "hold_s": float(row.hold_s),
                        **span,
                    }
                )
                continue

            span |= {
                "trial": int(row.trial),
                "stage": row.stage,
                "direction": row.direction,
                "attempt_type": row.attempt_type,
            }
            if row.stage == _INTERTRIAL:
                rows.append({"type": "Intertrial", "description": _INTERTRIAL, **span})
                continue
            if row.stage.startswith("Prepare"):
                rows.append(
                    {
                        "type": "Stimulus",
                        "modality": "visual",
                        "description": row.stage,
                        "stimulus_kind": "trial_cue",
                        **span,
                    }
                )
            rows.append(
                {
                    "type": "Action",
                    "description": row.stage,
                    "task_condition": "attempted",
                    "action_kind": "drag_and_drop",
                    # Closed-loop status is per stage and per decoder, so it belongs on
                    # the row rather than on the timeline.
                    "kin_decoder_closed": bool(row.kin_decoder_closed),
                    "gesture_decoder_closed": bool(row.gesture_decoder_closed),
                    "kin_error_attenuation": float(row.kin_error_attenuation),
                    "is_drag": bool(row.is_drag),
                    "is_click": bool(row.is_click),
                    **span,
                }
            )
        return rows

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        ref = self._ref(timeline)
        duration = utils.duration(ref)
        rows = self._neural_rows(timeline, duration)
        if ref.task == utils.TASK_DRAG_DROP:
            rows.append(
                {
                    "type": "BehavioralSeries",
                    "feature": "online_gesture_decode",
                    "start": 0.0,
                    "duration": duration,
                    "frequency": _RATE_HZ,
                    "dims": ("decoded_state",),
                    "n_features": 1,
                    "unit": "a.u.",
                    "kind": "online_decoder_output",
                    "processing": "latch decoder class index, alphabetical over the session's states",
                    "filepath": study.SpecialLoader(
                        method=self._load_decoded_state, timeline=timeline
                    ).to_json(),
                }
            )
        rows.extend(self._epoch_rows(timeline, ref))
        return pd.DataFrame(rows)


__all__ = ["Gusman2025DragAndDrop"]
