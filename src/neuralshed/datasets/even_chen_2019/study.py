"""neuralset Study for Even-Chen et al. 2019, instructed-delay center-out-and-back reaching.

One timeline is one NWB session. The session clock is continuous (trial times, spike times
and behavior timestamps all sit on it), but the release keeps data only inside trials, so
the gaps between trials hold zero spike counts and NaN behavior. Those gaps are emitted as
``Intertrial`` rows so a reader can see them.

Dense streams share one 50 Hz grid starting at 0: ``SpikeCounts`` binned from the units
table's spike times, and hand, cursor, eye and target positions averaged from the source's
1 kHz sample-and-hold grid into the same 20 ms bins (see ``utils.py`` for why).

Each trial gives a ``Stimulus`` (the target on screen, from its appearance to the trial
end) and an ``Action`` (from the go cue to the trial end). Trials whose target never
appeared (always failures) get an ``Action`` over the whole trial and no ``Stimulus``.
Every trial row carries ``loop_type``: ``hand_control`` where the arm drives the cursor,
``closed`` for the 134 brain-control trials of Reggie 2017-01-19, detected from
``DecodePos``. ``loop_type`` stays off the dense rows because one session mixes both.
"""

import typing as tp

import numpy as np
import pandas as pd
from neuralset.events import study

from neuralshed.datasets.study import NeuralshedStudy
from neuralshed.events import (  # noqa: F401, registers custom types
    CursorPositionSeries,
    Intertrial,
    PositionSeries,
    SpikeCounts,
    TargetPositionSeries,
)

from . import utils

_RATE_HZ = utils.FS
_BINNING = "averaged from the 1 kHz grid into 20 ms bins"

# Behavior rows: event type, source key (``target`` for the trial-table target), feature, unit,
# processing.
_BEHAVIOR: tuple[tuple[str, str, str, str, str], ...] = (
    (
        "PositionSeries",
        "Hand",
        "hand_position",
        "mm",
        "60 Hz optical tracker held on a 1 kHz grid; tracking-loss sentinel set to NaN; "
        + _BINNING,
    ),
    (
        "CursorPositionSeries",
        "Cursor",
        "cursor_position",
        "mm",
        "hand x, y one 1 ms sample later under hand control, decoder output under brain "
        "control; tracking-loss sentinel set to NaN; " + _BINNING,
    ),
    (
        "PositionSeries",
        "Eye",
        "eye_position",
        "a.u.",
        "uncalibrated eye tracker units, about 120 Hz held on a 1 kHz grid; all-zero "
        "dropout samples set to NaN; " + _BINNING,
    ),
    (
        "TargetPositionSeries",
        "target",
        "target_position",
        "mm",
        "trial-table target_pos from target_shown_time to stop_time, NaN elsewhere; "
        + _BINNING,
    ),
)


class EvenChen2019DelayedReach(NeuralshedStudy):
    """Two macaques reaching center-out-and-back to 24 to 48 targets (DANDI 000121)."""

    dataset_id: tp.ClassVar[str] = "even_chen_2019"
    aliases: tp.ClassVar[tuple[str, ...]] = ("even_chen_2019", "000121")
    bibtex: tp.ClassVar[str] = """
@article{evenchen2019structure,
  title = {Structure and variability of delay activity in premotor cortex},
  author = {Even-Chen, Nir and Sheffer, Blue and Vyas, Saurabh and Ryu, Stephen I. and Shenoy, Krishna V.},
  journal = {PLoS Computational Biology},
  volume = {15},
  number = {2},
  pages = {e1006808},
  year = {2019},
  doi = {10.1371/journal.pcbi.1006808},
  url = {https://doi.org/10.1371/journal.pcbi.1006808},
}
"""
    url: tp.ClassVar[str] = "https://dandiarchive.org/dandiset/000121/0.220124.2156"
    licence: tp.ClassVar[str] = "CC-BY-4.0"
    description: tp.ClassVar[str] = (
        "Two rhesus macaques make instructed-delay center-out-and-back arm reaches that "
        "move a cursor to targets on a line, a ring or three rings, recorded with a "
        "96-channel Utah array in M1 and one in PMd. Twelve sessions with 192 "
        "threshold-crossing channels and hand, cursor and eye position. One session holds "
        "a 134-trial brain-control block."
    )
    requirements: tp.ClassVar[tuple[str, ...]] = ("dandi", "pynwb", "h5py", "remfile")

    # JenkinsC 2015-10-15 (hand control only) and Reggie 2017-01-19 (brain-control block,
    # off-screen cursor trials, aborted trials without a target).
    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=12,
        num_subjects=2,
        query="timeline_index in [2, 9]",
        num_events_in_query=18735,
        event_types_in_query={
            "Action",
            "CursorPositionSeries",
            "Intertrial",
            "PositionSeries",
            "SpikeCounts",
            "Stimulus",
            "TargetPositionSeries",
        },
        frequency=_RATE_HZ,
    )

    def _download(self, overwrite: bool = False) -> None:
        """Download the dandiset (about 38 GB) so files land at ``self.path/000121/sub-*/``."""
        from neuralfetch.download import Dandi

        Dandi(
            study=utils.DANDISET_ID,
            version=utils.VERSION,
            dset_dir=self.path,
            folder="",
        ).download(overwrite=overwrite)

    def _source(self, timeline: dict[str, tp.Any]) -> str:
        return utils.resolve_source(self.path, timeline["path"])

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        for ref in utils.list_sessions(self.path):
            trials = utils.get_trials(utils.resolve_source(self.path, ref.path))
            layout, n_targets = utils.target_layout(trials)
            yield {
                "subject": ref.subject,
                "session": ref.label,
                "date": ref.date,
                "task": "center_out_and_back",
                "target_layout": layout,
                "n_targets": n_targets,
                "path": ref.path,
            }

    # -- loaders ---------------------------------------------------------------
    def _load_spike_counts(self, timeline: dict[str, tp.Any]) -> np.ndarray:
        """SpecialLoader target: ``(192, n_bins)`` threshold-crossing counts."""
        source = self._source(timeline)
        return utils.binned_spike_counts(source, utils.n_bins(source))

    def _load_behavior(self, timeline: dict[str, tp.Any], key: str) -> np.ndarray:
        """SpecialLoader target: one position stream as ``(n_dims, n_bins)``."""
        source = self._source(timeline)
        if key == "target":
            return utils.binned_target(source, utils.n_bins(source))
        return utils.binned_behavior(source, key, utils.n_bins(source))

    # -- rows ------------------------------------------------------------------
    def _stream_rows(self, timeline: dict[str, tp.Any], duration: float) -> list[dict]:
        rows: list[dict[str, tp.Any]] = [
            {
                "type": "SpikeCounts",
                "feature": "threshold_crossings",
                "start": 0.0,
                "duration": duration,
                "frequency": _RATE_HZ,
                "n_features": utils.N_CHANNELS,
                "unit": "counts/bin",
                "normalization": "raw",
                "processing": "",
                "method": "threshold_crossing",
                "threshold": utils.THRESHOLD[timeline["subject"]],
                "filepath": study.SpecialLoader(
                    method=self._load_spike_counts, timeline=timeline
                ).to_json(),
            }
        ]
        for type_, key, feature, unit, processing in _BEHAVIOR:
            dims = utils.POSITION_KEYS.get(key, ("x", "y"))
            rows.append(
                {
                    "type": type_,
                    "feature": feature,
                    "start": 0.0,
                    "duration": duration,
                    "frequency": _RATE_HZ,
                    "dims": dims,
                    "n_features": len(dims),
                    "unit": unit,
                    "normalization": "raw",
                    "processing": processing,
                    "kind": feature,
                    "reference_frame": "screen_center",
                    "filepath": study.SpecialLoader(
                        method=self._load_behavior, timeline=timeline, key=key
                    ).to_json(),
                }
            )
        return rows

    @staticmethod
    def _trial_rows(
        trials: pd.DataFrame, off_screen: np.ndarray
    ) -> list[dict[str, tp.Any]]:
        codes = utils.target_codes(trials)
        rows: list[dict[str, tp.Any]] = []
        previous_stop = None
        for trial, code, off in zip(trials.itertuples(index=False), codes, off_screen):
            start, stop = float(trial.start_time), float(trial.stop_time)
            if previous_stop is not None and start > previous_stop:
                rows.append(
                    {
                        "type": "Intertrial",
                        "start": previous_stop,
                        "duration": start - previous_stop,
                        "description": "no neural or behavioral data released between trials",
                    }
                )
            previous_stop = stop

            brain = bool(trial.brain_control)
            common: dict[str, tp.Any] = {
                "trial_id": int(trial.trial_id),
                "code": int(code),  # session-local, see utils.target_codes
                "target_x": float(trial.target_x),
                "target_y": float(trial.target_y),
                "trial_kind": "return" if trial.is_return else "center_out",
                # NaN when the target never appeared.
                "delay_duration": float(trial.delay),
                "loop_type": "closed" if brain else "hand_control",
                "success": bool(trial.success),
            }
            if off:
                common["valid"] = False
                common["valid_description"] = (
                    "cursor off the screen for most of the trial"
                )

            shown = float(trial.target_shown_time)
            if np.isnan(shown):
                rows.append(
                    {
                        "type": "Action",
                        "start": start,
                        "duration": stop - start,
                        "description": "trial aborted before the target appeared",
                        "task_condition": "other" if brain else "executed",
                        "action_kind": "cursor_control",
                        **common,
                    }
                )
                continue
            go = float(trial.go_cue_time)
            rows.append(
                {
                    "type": "Stimulus",
                    "start": shown,
                    "duration": stop - shown,
                    "modality": "visual",
                    "stimulus_kind": "target_cue",
                    "description": "reach target on screen",
                    **common,
                }
            )
            rows.append(
                {
                    "type": "Action",
                    "start": go,
                    "duration": stop - go,
                    "description": (
                        "decoder-driven cursor movement to the target"
                        if brain
                        else "arm reach moving the cursor to the target"
                    ),
                    # Under brain control the source does not say whether the arm moved.
                    "task_condition": "other" if brain else "executed",
                    "action_kind": "cursor_control",
                    **common,
                }
            )
        return rows

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        source = self._source(timeline)
        trials = utils.get_trials(source)
        duration = utils.n_bins(source) / _RATE_HZ
        off_screen = utils.cursor_off_screen(source)
        rows = self._stream_rows(timeline, duration) + self._trial_rows(
            trials, off_screen
        )
        return pd.DataFrame(rows)


__all__ = ["EvenChen2019DelayedReach"]
