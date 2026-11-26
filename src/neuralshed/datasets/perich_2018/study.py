"""neuralset Study for the Perich, Miller, Azabou and Dyer monkey reaching release (DANDI 000688).

One timeline is one NWB file: a single session of one monkey (C, J, M or T) on one task,
center-out (CO) or random-target (RT), with neural and behavioral data on one continuous
clock. Every dense stream sits on the 100 Hz cursor grid, starting at the first cursor
timestamp (0 to 0.31 s into the file's clock), and trial rows keep the source's times.

Per timeline the Study emits:

* one ``SpikeCounts`` row, the sorted units binned by the Study onto the cursor grid, M1
  units before PMd ones (``n_units_m1`` / ``n_units_pmd`` give the split);
* ``CursorPositionSeries``, ``CursorVelocitySeries`` and ``AccelerationSeries`` rows, the
  released cursor streams in cm, cm/s and cm/s^2. The cursor follows the manipulandum
  handle, so its velocity is how the hand moved rather than a decoder's command;
* trial rows. A CO trial becomes a center-hold ``Intertrial``, a ``Stimulus`` over the
  instructed delay (``period="delay"``), a one-bin auditory go cue ``Stimulus`` and an
  ``Action`` from the go cue to the trial end (``period="go"``). An RT trial becomes one
  ``Action`` per reach, from its go cue to the next one, each preceded by a one-bin target
  ``Stimulus``. Gaps between trials are ``Intertrial`` rows. Failed and aborted trials keep
  their ``result`` and ``success=False``.

The seven Monkey C sessions with the cursor at about half scale carry ``valid=False``.
"""

import typing as tp

import numpy as np
import pandas as pd
from neuralset.events import study

from neuralshed.datasets import nwb
from neuralshed.datasets.study import NeuralshedStudy
from neuralshed.events import (  # noqa: F401, registers the custom event types
    AccelerationSeries,
    CursorPositionSeries,
    CursorVelocitySeries,
    Intertrial,
    SpikeCounts,
)

from . import utils

_RATE_HZ = utils.FS
_DANDISET_ID = utils.DANDISET_ID
_VERSION = utils.VERSION

#: (event type, source key, feature, unit) of the three released cursor streams.
_BEHAVIOR: tuple[tuple[str, str, str, str], ...] = (
    ("CursorPositionSeries", "cursor_pos", "cursor_position", "cm"),
    ("CursorVelocitySeries", "cursor_vel", "cursor_velocity", "cm/s"),
    ("AccelerationSeries", "cursor_acc", "cursor_acceleration", "cm/s^2"),
)


class Perich2018Reaching(NeuralshedStudy):
    """Four monkeys reaching with a planar manipulandum, M1 and PMd Utah arrays."""

    dataset_id: tp.ClassVar[str] = "perich_2018"
    aliases: tp.ClassVar[tuple[str, ...]] = ("perich_2018", "dandi-000688")
    bibtex: tp.ClassVar[str] = """
@article{perich2018neural,
  title = {A Neural Population Mechanism for Rapid Learning},
  author = {Perich, Matthew G. and Gallego, Juan A. and Miller, Lee E.},
  journal = {Neuron},
  volume = {100},
  number = {4},
  pages = {964--976.e7},
  year = {2018},
  doi = {10.1016/j.neuron.2018.09.030},
  url = {https://doi.org/10.1016/j.neuron.2018.09.030},
}
"""
    url: tp.ClassVar[str] = "https://dandiarchive.org/dandiset/000688"
    licence: tp.ClassVar[str] = "CC-BY-4.0"
    description: tp.ClassVar[str] = (
        "Monkeys C, J, M and T move a cursor with the handle of a planar manipulandum in "
        "a center-out task with an instructed delay or a random-target task of four "
        "sequential reaches. 111 sessions from 2013 to 2016, each with sorted units from "
        "96-channel Utah arrays in M1 and, in about half the sessions, PMd, and the "
        "cursor's position, velocity and acceleration at 100 Hz."
    )
    requirements: tp.ClassVar[tuple[str, ...]] = ("dandi", "pynwb", "h5py", "remfile")

    # A zero-delay CO session (M1 only), a half-scale CO session with both arrays
    # (valid=False) and an RT session.
    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=111,
        num_subjects=4,
        query="timeline_index in [5, 39, 53]",
        num_events_in_query=3886,
        event_types_in_query={
            "AccelerationSeries",
            "Action",
            "CursorPositionSeries",
            "CursorVelocitySeries",
            "Intertrial",
            "SpikeCounts",
            "Stimulus",
        },
        frequency=_RATE_HZ,
    )

    # -- data access ------------------------------------------------------------------
    def _download(self, overwrite: bool = False) -> None:
        from neuralfetch.download import Dandi

        # folder="" lays the files out as self.path/000688/sub-*/...nwb
        Dandi(
            study=_DANDISET_ID, version=_VERSION, dset_dir=self.path, folder=""
        ).download(overwrite=overwrite)

    def _session_refs(self) -> list[utils.SessionRef]:
        """Local files when present, else the dandiset's assets (streamed)."""
        local = utils.list_sessions(self.path)
        if local:
            return local
        assets = nwb.dandi_list_assets(_DANDISET_ID, _VERSION)
        return [
            utils.parse_path(f"{_DANDISET_ID}/{a['path']}")
            for a in assets
            if a["path"].endswith(".nwb")
        ]

    def _resolve_source(self, timeline_path: str) -> str:
        local = self.path / timeline_path
        if local.is_file():
            return str(local)
        asset_path = timeline_path.removeprefix(_DANDISET_ID + "/")
        return nwb.dandi_asset_url(_DANDISET_ID, asset_path, _VERSION)

    # -- timelines --------------------------------------------------------------------
    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        for ref in self._session_refs():
            timeline: dict[str, tp.Any] = {
                "subject": ref.subject,
                "session": ref.label,
                "task": ref.task,
                "date": ref.date,
                "path": ref.path,
            }
            if ref.label in utils.HALF_SCALE_SESSIONS:
                timeline["valid"] = False
                timeline["valid_description"] = utils.HALF_SCALE_DESCRIPTION
            yield timeline

    # -- SpecialLoader targets --------------------------------------------------------
    def _load_spike_counts(self, timeline: dict[str, tp.Any]) -> np.ndarray:
        """``(n_units, n_bins)`` float32 on the cursor grid, M1 units first."""
        with nwb.open_h5(self._resolve_source(timeline["path"])) as h5:
            t0, n = utils.grid(h5)
            return utils.bin_spikes(h5, t0, n)

    def _load_behavior(self, timeline: dict[str, tp.Any], key: str) -> np.ndarray:
        """``(2, n_bins)`` float32, dims-first."""
        with nwb.open_h5(self._resolve_source(timeline["path"])) as h5:
            data = utils.get_behavior(h5, key)
        return np.ascontiguousarray(data.T, dtype=np.float32)

    # -- events -----------------------------------------------------------------------
    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        with nwb.open_h5(self._resolve_source(timeline["path"])) as h5:
            t0, n = utils.grid(h5)
            regions = utils.unit_regions(h5)
            warned = "Warning:" in utils.session_description(h5)
            is_co = timeline["task"] == utils.TASKS["CO"]
            trials = utils.co_trials(h5) if is_co else utils.rt_trials(h5)
        if warned != (timeline["session"] in utils.HALF_SCALE_SESSIONS):
            raise ValueError(
                f"{timeline['session']}: session_description warning does not match "
                "utils.HALF_SCALE_SESSIONS"
            )
        stop = t0 + n / _RATE_HZ
        rows = self._stream_rows(timeline, t0, n, regions, warned)
        task_rows = self._co_rows(trials) if is_co else self._rt_rows(trials)
        rows += task_rows + self._gap_rows(trials, t0, stop)
        return pd.DataFrame(rows)

    def _stream_rows(
        self,
        timeline: dict[str, tp.Any],
        t0: float,
        n: int,
        regions: np.ndarray,
        warned: bool,
    ) -> list[dict[str, tp.Any]]:
        span = {"start": t0, "duration": n / _RATE_HZ, "frequency": _RATE_HZ}
        rows: list[dict[str, tp.Any]] = [
            {
                "type": "SpikeCounts",
                "feature": "sorted_spikes",
                **span,
                "n_features": len(regions),
                "unit": "counts/bin",
                "normalization": "raw",
                "processing": (
                    "high-pass 750 Hz; spikes sorted into units by the source; units "
                    "binned from spike times onto the 10 ms cursor grid by the Study"
                ),
                "method": "sorted",
                "threshold": "",
                "n_units_m1": int((regions == "M1").sum()),
                "n_units_pmd": int((regions == "PMd").sum()),
                "filepath": study.SpecialLoader(
                    method=self._load_spike_counts, timeline=timeline
                ).to_json(),
            }
        ]
        for etype, key, feature, unit in _BEHAVIOR:
            rows.append(
                {
                    "type": etype,
                    "feature": feature,
                    **span,
                    "dims": utils.POSITION_DIMS,
                    "n_features": len(utils.POSITION_DIMS),
                    "unit": unit,
                    "kind": feature,
                    "reference_frame": "screen_center",
                    # The source ships cursor_vel and cursor_acc as numerical
                    # derivatives of cursor_pos.
                    "processing": (
                        "" if key == "cursor_pos" else "derivative of cursor_pos"
                    ),
                    "filepath": study.SpecialLoader(
                        method=self._load_behavior, timeline=timeline, key=key
                    ).to_json(),
                }
            )
        return rows

    @staticmethod
    def _co_rows(trials: pd.DataFrame) -> list[dict[str, tp.Any]]:
        rows: list[dict[str, tp.Any]] = []
        for tr in trials.itertuples(index=False):
            ids = {"trial_id": int(tr.trial_id), "result": tr.result}
            if np.isnan(tr.target_on_time):
                description = (
                    "rewarded trial with no logged target"
                    if tr.result == "R"
                    else "center hold, trial ended before target onset"
                )
                rows.append(_intertrial(tr.start_time, tr.stop_time, description, ids))
                continue
            target = {
                "code": int(tr.code),
                "target_x": float(tr.target_x),
                "target_y": float(tr.target_y),
            }
            if tr.target_on_time > tr.start_time:
                rows.append(
                    _intertrial(
                        tr.start_time,
                        tr.target_on_time,
                        "center hold before target onset",
                        ids,
                    )
                )
            has_go = not np.isnan(tr.go_cue_time)
            delay_stop = tr.go_cue_time if has_go else tr.stop_time
            if delay_stop > tr.target_on_time:
                rows.append(
                    {
                        "type": "Stimulus",
                        "start": tr.target_on_time,
                        "duration": delay_stop - tr.target_on_time,
                        "description": "outer target shown, instructed delay",
                        "modality": "visual",
                        "stimulus_kind": "target_cue",
                        "period": "delay",
                        "task_condition": "na",
                        **target,
                        **ids,
                    }
                )
            if not has_go or tr.stop_time <= tr.go_cue_time:
                continue
            rows.append(
                {
                    "type": "Stimulus",
                    "start": tr.go_cue_time,
                    "duration": min(utils.BIN_S, tr.stop_time - tr.go_cue_time),
                    "description": "auditory go cue",
                    "modality": "audio",
                    "stimulus_kind": "movement_go_cue",
                    "period": "go",
                    "task_condition": "na",
                    **target,
                    **ids,
                }
            )
            rows.append(
                {
                    "type": "Action",
                    "start": tr.go_cue_time,
                    "duration": tr.stop_time - tr.go_cue_time,
                    "description": "reach to the outer target and hold",
                    "action_kind": "cursor_control",
                    "task_condition": "executed",
                    "period": "go",
                    "reach_index": 0,
                    "delay": float(tr.go_cue_time - tr.target_on_time),
                    "success": tr.result == "R",
                    **target,
                    **ids,
                }
            )
        return rows

    @staticmethod
    def _rt_rows(trials: pd.DataFrame) -> list[dict[str, tp.Any]]:
        rows: list[dict[str, tp.Any]] = []
        for tr in trials.to_dict("records"):
            ids = {"trial_id": int(tr["trial_id"]), "result": tr["result"]}
            reaches = [k for k in range(4) if not np.isnan(tr[f"go_{k}"])]
            if not reaches:
                if tr["result"] == "R":
                    # 15 to 96 rewarded trials per session log neither targets nor go
                    # cues. The reaches happened, so the trial stays one Action.
                    rows.append(
                        {
                            "type": "Action",
                            "start": tr["start_time"],
                            "duration": tr["stop_time"] - tr["start_time"],
                            "description": "random-target trial with no logged targets",
                            "action_kind": "cursor_control",
                            "task_condition": "executed",
                            "period": "go",
                            "reach_index": -1,
                            "code": -100,
                            "target_x": np.nan,
                            "target_y": np.nan,
                            "success": True,
                            **ids,
                        }
                    )
                else:
                    rows.append(
                        _intertrial(
                            tr["start_time"],
                            tr["stop_time"],
                            "hold, trial ended before the first target",
                            ids,
                        )
                    )
                continue
            first_go = tr[f"go_{reaches[0]}"]
            if first_go > tr["start_time"]:
                rows.append(
                    _intertrial(
                        tr["start_time"], first_go, "hold before the first target", ids
                    )
                )
            for position, k in enumerate(reaches):
                go = tr[f"go_{k}"]
                last = position == len(reaches) - 1
                end = tr["stop_time"] if last else tr[f"go_{reaches[position + 1]}"]
                if end <= go:
                    continue
                target = {
                    "code": -100,  # targets are placed at random, there is no class
                    "target_x": float(tr[f"target_{k}_x"]),
                    "target_y": float(tr[f"target_{k}_y"]),
                    "reach_index": k,
                }
                rows.append(
                    {
                        "type": "Stimulus",
                        "start": go,
                        "duration": min(utils.BIN_S, end - go),
                        "description": "next target shown, no instructed delay",
                        "modality": "visual",
                        "stimulus_kind": "target_cue",
                        "period": "go",
                        "task_condition": "na",
                        **target,
                        **ids,
                    }
                )
                rows.append(
                    {
                        "type": "Action",
                        "start": go,
                        "duration": end - go,
                        "description": f"reach to target {k + 1} of 4",
                        "action_kind": "cursor_control",
                        "task_condition": "executed",
                        "period": "go",
                        # A reach the monkey moved on from was acquired. The last one
                        # succeeded only if the trial was rewarded.
                        "success": tr["result"] == "R" or not last,
                        **target,
                        **ids,
                    }
                )
        return rows

    @staticmethod
    def _gap_rows(
        trials: pd.DataFrame, start: float, stop: float
    ) -> list[dict[str, tp.Any]]:
        """``Intertrial`` rows over the gaps between trials, and before the first and
        after the last one within the dense span."""
        edges = [start]
        for tr in trials.itertuples(index=False):
            edges += [tr.start_time, tr.stop_time]
        edges.append(stop)
        rows = []
        for gap_start, gap_stop in zip(edges[::2], edges[1::2]):
            if gap_stop - gap_start > 1e-6:
                rows.append(_intertrial(gap_start, gap_stop, "intertrial interval", {}))
        return rows


def _intertrial(
    start: float, stop: float, description: str, ids: dict[str, tp.Any]
) -> dict[str, tp.Any]:
    return {
        "type": "Intertrial",
        "start": start,
        "duration": stop - start,
        "description": description,
        "period": "intertrial",
        "task_condition": "na",
        **ids,
    }


__all__ = ["Perich2018Reaching"]
