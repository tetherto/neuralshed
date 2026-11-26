"""neuroai ``Study`` for FALCON Benchmark H1 (DANDI 000954).

Each released recording is one :term:`timeline`. For each we emit:

* one ``SpikeCounts`` event, the 176 threshold-crossing channels (the rows of the NWB
  ``Units`` table, see the README) binned at 50 Hz on the kinematics grid through a ``SpecialLoader``. Neural sample ``k`` counts the 20 ms
  ending at kinematic sample ``k`` (``k * 0.02`` s), as FALCON bins them
  (``utils.bin_edges``), so both streams start at 0 and share their indices;
* one ``BinaryMask`` event, FALCON's ``eval_mask`` on the same grid, True on the bins
  FALCON trains on and scores (``true_means="scored"``);
* one event per ``intervals/epochs`` row -- each row is a single *task phase*
  tagged with exactly one of 16 labels. The phases tile each trial contiguously
  (``Reach -> Orient -> Shape -> Grasp -> Carry -> Orient2 -> Release``, see
  ``datasets/karpowicz_2024_h1/README.md``). We split them four ways by tag:

  - **``Stimulus``** -- the ``Presentation*`` *cue* phases, during which the
    effector is held still while the next target is shown to the subject. The
    number is positional (which movement it cues), so each cue's ``code`` is set
    to the ``code`` of the ``Action`` it precedes; ``modality`` is left unset
    (the sources describe the cues differently, see the README).
  - **``Action``** -- the attempted-movement phases (``Reach``, ``Orient``,
    ``Shape``, ``Grasp``, ``Carry``, ``Orient2``, ``Release``), one ``code`` per
    phase. ``Orient`` turns the hand before the grasp, ``Orient2`` turns the held
    object after ``Carry``.
  - **``AutoMovement``** -- ``SnapTo``, an uncued move of the virtual hand onto the
    object before the grasp (see the README).
  - **``Intertrial``** -- the rest before the first trial (custom event type). The
    ``Intertrial`` tag that follows each file's last ``Release`` cue is that trial's
    ``Release`` and is emitted as one.

  These are open-loop data: an autocontroller moves a virtual arm while the
  participant attempts each movement, hence ``task_condition="attempted"`` and
  ``loop_type="open"`` on every timeline.

NWB layout (see ``datasets/karpowicz_2024_h1/README.md``):

* ``units`` -- Units table, 176 rows, each one channel of unsorted threshold crossings
  (not a sorted unit), ragged ``spike_times`` (seconds).
* ``acquisition/OpenLoopKinematics`` -- ``TimeSeries (n_bins, 7)``; the 7 DoF mix
  3 translation (``tx,ty,tz``), 1 roll (``rx``), 3 grasp (``g1,g2,g3``).
* ``intervals/epochs`` -- one row per task phase, single ``tags`` entry plus
  ``start_time``/``stop_time``

The 7-DoF kinematics are split by quantity into homogeneous events, at both the
position level (``OpenLoopKinematics``) and the velocity level
(``OpenLoopKinematicsVelocity``): ``Position``/``Orientation``/``Grasp`` Series
(x,y,z / roll / pinch,scoop,thumb_abduction) and ``Linear``/``Angular``/``Grasp``
VelocitySeries. Each is
loaded from the NWB via a ``SpecialLoader`` and read by ``BehavioralSeriesExtractor``.

## Caching:
Cache keys are shared between local and streaming. Both embed the relative timeline path
(e.g. sub-HumanPitt-…/…nwb), which is identical in both modes, so a disk cache built locally
is reused when streaming, and vice versa.
"""

import typing as tp

import numpy as np
import pandas as pd

from neuralset.events import study

from neuralshed.datasets import nwb
from neuralshed.datasets.study import NeuralshedStudy
from neuralshed.events import (  # noqa: F401 — registers the custom event types
    AngularVelocitySeries,
    AutoMovement,
    BinaryMask,
    GraspSeries,
    GraspVelocitySeries,
    Intertrial,
    LinearVelocitySeries,
    OrientationSeries,
    PositionSeries,
    SpikeCounts,
)

from . import utils

# Decompose the 7-DoF kinematics (cols tx,ty,tz,rx,g1,g2,g3) into homogeneous events,
# at both the position level (``OpenLoopKinematics``) and velocity level
# (``OpenLoopKinematicsVelocity``). Grasp dims are the source's pinch, scoop and thumb
# abduction (FALCON A.3.5). Velocity is kept per 20 ms bin, as FALCON's targets.
_XYZ = ("x", "y", "z")
_GRASP_DIMS = ("pinch", "scoop", "thumb_abduction")
_VELOCITY_PROCESSING = (
    "smoothed temporal derivative of position per 20 ms bin; kernel unpublished, "
    "close to FALCON's 490 ms Gaussian followed by np.gradient"
)
_KINEMATICS: tuple[dict[str, tp.Any], ...] = (
    dict(
        type="PositionSeries",
        feature="hand_position",
        key="OpenLoopKinematics",
        cols=(0, 1, 2),
        dims=_XYZ,
        unit="a.u.",
        reference_frame="",
        processing="",
    ),
    dict(
        type="OrientationSeries",
        feature="hand_orientation",
        key="OpenLoopKinematics",
        cols=(3,),
        dims=("roll",),
        unit="a.u.",
        reference_frame="",
        processing="",
    ),
    dict(
        type="GraspSeries",
        feature="grasp",
        key="OpenLoopKinematics",
        cols=(4, 5, 6),
        dims=_GRASP_DIMS,
        unit="proportion",
        reference_frame="range_of_motion",
        processing="",
    ),
    dict(
        type="LinearVelocitySeries",
        feature="hand_linear_velocity",
        key="OpenLoopKinematicsVelocity",
        cols=(0, 1, 2),
        dims=_XYZ,
        unit="a.u.",
        reference_frame="",
        processing=_VELOCITY_PROCESSING,
    ),
    dict(
        type="AngularVelocitySeries",
        feature="hand_angular_velocity",
        key="OpenLoopKinematicsVelocity",
        cols=(3,),
        dims=("roll",),
        unit="a.u.",
        reference_frame="",
        processing=_VELOCITY_PROCESSING,
    ),
    dict(
        type="GraspVelocitySeries",
        feature="grasp_velocity",
        key="OpenLoopKinematicsVelocity",
        cols=(4, 5, 6),
        dims=_GRASP_DIMS,
        unit="a.u.",
        reference_frame="range_of_motion",
        processing=_VELOCITY_PROCESSING,
    ),
)

# -- Epoch-tag taxonomy (see datasets/karpowicz_2024_h1/README.md "Task phases / events") ---
# Each epochs row carries exactly one of these 16 tags. The participant's
# attempted-movement phases become ``Action`` events, one ``code`` per phase in
# task order (Reach .. Release).
_ACTION_CODES: dict[str, int] = {
    "Reach": 0,
    "Orient": 1,
    "Shape": 2,
    "Grasp": 3,
    "Carry": 4,
    "Orient2": 5,  # turns the held object after Carry
    "Release": 6,
}
# Uncued moves the task software makes on its own, emitted as ``AutoMovement``.
_AUTO_MOVEMENTS: frozenset[str] = frozenset({"SnapTo"})
# What each movement phase moves, from the per-phase DoF speeds (README).
_ACTION_KINDS: dict[str, str] = {
    "Reach": "arm_movement",
    "Carry": "arm_movement",
    "Orient": "wrist_movement",
    "Orient2": "wrist_movement",
    "Shape": "hand_movement",
    "Grasp": "hand_movement",
    "Release": "hand_movement",
}
# The ``Presentation*`` phases are cue periods, the number is positional.
# Each maps to the ``Action`` it precedes, and its ``Stimulus.code``
# is set to that action's code so a cue and the movement it cues share a code.
_CUE_TO_ACTION: dict[str, str] = {
    "Presentation": "Reach",
    "Presentation2": "Orient",
    "Presentation3": "Shape",
    "Presentation4": "Grasp",
    "Presentation5": "Carry",
    "Presentation6": "Orient2",
    "Presentation7": "Release",
}


class Karpowicz2024FalconH1(NeuralshedStudy):
    """Karpowicz et al. 2024 (FALCON H1): human 7-DoF reach & grasp iBCI (DANDI 000954).

    Open-loop 7-DoF reach and grasp by a single iBCI participant with two
    96-channel Utah arrays in hand-knob motor cortex. One timeline per released
    recording, 27 across the held-in-calib and held-out-calib FALCON splits. The
    held-in-minival files copy the start of held-in-calib files and are left out.
    """

    dataset_id: tp.ClassVar[str] = "karpowicz_2024_h1"
    aliases: tp.ClassVar[tuple[str, ...]] = (
        "karpowicz_2024_h1",
        "fan_h1",
        "FALCON Benchmark H1",
        "FALCON-H1",
    )
    bibtex: tp.ClassVar[str] = """
@inproceedings{karpowicz2024falcon,
  title = {Few-shot Algorithms for Consistent Neural Decoding (FALCON) Benchmark},
  author = {Karpowicz, Brianna M. and Ye, Joel and Fan, Chaofei and Tostado-Marcos, Pablo and Rizzoglio, Fabio and Washington, Clay and Scodeler, Thiago and de Lucena, Diogo and Nason-Tomaszewski, Samuel R. and Mender, Matthew and Ma, Xuan and Arneodo, Ezequiel Matias and Hochberg, Leigh R. and Chestek, Cynthia and Henderson, Jaimie M. and Gentner, Timothy Q. and Gilja, Vikash and Miller, Lee and Rouse, Adam G. and Gaunt, Robert and Collinger, Jennifer L. and Pandarinath, Chethan},
  booktitle = {Advances in Neural Information Processing Systems},
  volume = {37},
  pages = {76578--76615},
  year = {2024},
  doi = {10.52202/079017-2439},
  url = {https://proceedings.neurips.cc/paper_files/paper/2024/hash/8c2e6bb15be1894b8fb4e0f9bcad1739-Abstract-Datasets_and_Benchmarks_Track.html},
}
"""
    timeline_descriptions: tp.ClassVar[dict[str, dict[tp.Any, str]]] = {
        "original_split": {
            "held-in-calib": (
                "A recording from one of the 6 held-in days, without the last 20% "
                "FALCON keeps for its hidden evaluation."
            ),
            "held-out-calib": (
                "The first 20% of a recording from one of the 7 later held-out days, "
                "released for few-shot recalibration."
            ),
        },
    }
    url: tp.ClassVar[str] = "https://dandiarchive.org/dandiset/000954"
    licence: tp.ClassVar[str] = "CC-BY-4.0"
    description: tp.ClassVar[str] = (
        "Open-loop 7-DoF reach and grasp by a human iBCI participant with two "
        "96-channel Utah arrays in hand-knob motor cortex (FALCON H1 cross-day "
        "decoding-stability benchmark, DANDI 000954)."
    )
    requirements: tp.ClassVar[tuple[str, ...]] = ("dandi", "pynwb", "h5py", "remfile")

    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=27,
        num_subjects=1,
        # timeline 0 starts mid-Release, 13 is the first held-out-calib file
        query="timeline_index in [0, 13]",
        num_events_in_query=247,  # 8 dense rows each + 187 and 44 phase rows
        event_types_in_query={
            "SpikeCounts",
            "BinaryMask",
            "PositionSeries",
            "OrientationSeries",
            "GraspSeries",
            "LinearVelocitySeries",
            "AngularVelocitySeries",
            "GraspVelocitySeries",
            "Stimulus",
            "Action",
            "AutoMovement",
            "Intertrial",
        },
        frequency=utils.BIN_RATE_HZ,
    )

    # -- data access ------------------------------------------------------
    def _download(self, overwrite: bool = False) -> None:
        from neuralfetch.download import Dandi

        dandi_client = Dandi(
            study=utils.DANDISET_ID,
            version=utils.VERSION,
            dset_dir=self.path,
            folder="",
        )
        dandi_client.download(overwrite=overwrite)

    def _resolve_source(self, timeline_path: str) -> str:
        """The local file when downloaded, else its DANDI streaming URL."""
        return nwb.resolve_source(
            timeline_path, utils.DANDISET_ID, utils.VERSION, root=self.path
        )

    # -- timelines + events ----------------------------------------------
    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        """One timeline per recording, held-in-minival copies left out."""
        paths = nwb.dandi_nwb_paths(utils.DANDISET_ID, utils.VERSION, root=self.path)
        for rec in utils.recordings(paths):
            yield dict(
                subject=utils.SUBJECT,
                session=rec["session"],
                block=rec["block"],
                original_split=rec["split"],
                task="reach_grasp",
                # the participant attempts each movement the virtual arm makes, and no
                # decoder drives the arm
                task_condition="attempted",
                loop_type="open",
                path=rec["path"],
            )

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        with nwb.open_nwb(self._resolve_source(timeline["path"])) as nwbfile:
            n_bins = int(nwbfile.acquisition["OpenLoopKinematics"].data.shape[0])
            duration_s = n_bins * utils.BIN_SIZE_S
            trial_num = np.asarray(nwbfile.acquisition["TrialNum"].data[:])
            epochs = nwbfile.intervals["epochs"].to_dataframe()

        rows: list[dict[str, tp.Any]] = [
            dict(
                type="SpikeCounts",
                feature="threshold_crossings",
                start=0.0,
                duration=float(duration_s),
                frequency=utils.BIN_RATE_HZ,
                n_features=utils.N_CHANNELS,
                unit="counts/bin",
                normalization="raw",
                processing="",
                method="threshold_crossing",
                threshold="-4.5 x RMS",
                filepath=study.SpecialLoader(
                    method=self._load_spike_counts,
                    timeline=timeline,
                ).to_json(),
            )
        ]
        rows.append(
            dict(
                type="BinaryMask",
                feature="falcon_eval_mask",
                start=0.0,
                duration=float(duration_s),
                frequency=utils.BIN_RATE_HZ,
                n_features=1,
                unit="binary",
                normalization="raw",
                processing="",
                true_means="scored",
                filepath=study.SpecialLoader(
                    method=self._load_eval_mask,
                    timeline=timeline,
                ).to_json(),
            )
        )
        rows += self._kinematics_events(timeline, duration_s)
        trial_ids = utils.phase_trial_ids(
            trial_num, epochs.start_time, epochs.stop_time
        )
        previous = None
        for ep, trial_id in zip(epochs.itertuples(index=False), trial_ids):
            if len(ep.tags) != 1:
                raise ValueError(
                    f"{timeline['path']}: epochs row at {ep.start_time} s has tags "
                    f"{list(ep.tags)}, expected exactly one"
                )
            tag = str(ep.tags[0])
            row = self._phase_event(tag, ep.start_time, ep.stop_time, previous)
            rows.append({**row, "trial_id": int(trial_id)})
            previous = tag
        return pd.DataFrame(rows)

    def _kinematics_events(
        self,
        timeline: dict[str, tp.Any],
        duration_s: float,
    ) -> list[dict[str, tp.Any]]:
        """Six homogeneous kinematics rows (position + velocity; subject added by standardize_events)."""
        out: list[dict[str, tp.Any]] = []
        for stream in _KINEMATICS:
            loader = study.SpecialLoader(
                method=self._load_kinematics,
                timeline=timeline,
                key=stream["key"],
                cols=list(stream["cols"]),
            ).to_json()
            out.append(
                dict(
                    type=stream["type"],
                    feature=stream["feature"],
                    start=0.0,
                    duration=float(duration_s),
                    frequency=utils.BIN_RATE_HZ,
                    dims=stream["dims"],
                    n_features=len(stream["dims"]),
                    unit=stream["unit"],
                    normalization="raw",
                    processing=stream["processing"],
                    reference_frame=stream["reference_frame"],
                    filepath=loader,
                )
            )
        return out

    @staticmethod
    def _phase_event(
        tag: str,
        start_time: tp.Any,
        stop_time: tp.Any,
        previous_tag: str | None = None,
    ) -> dict[str, tp.Any]:
        """Map one epoch phase (single ``tag``) to its event row:

        - ``Presentation*`` -> ``Stimulus`` (code = the cued action's code);
        - attempted-movement phases -> ``Action`` (one code per phase);
        - ``SnapTo`` -> ``AutoMovement``;
        - ``Intertrial`` after the ``Release`` cue -> the trial's ``Release``, cut short
            by the file end, with ``source_tag="Intertrial"`` (see the README);
        - other ``Intertrial`` -> ``Intertrial``. Unknown tags raise (the released tag
            set is fully enumerated, so a surprise means the taxonomy is stale).
        """
        start = float(start_time)
        duration = float(stop_time) - start
        if tag == "Intertrial" and previous_tag == "Presentation7":
            return dict(
                type="Action",
                start=start,
                duration=duration,
                code=_ACTION_CODES["Release"],
                description="Release",
                period="go",
                action_kind=_ACTION_KINDS["Release"],
                source_tag=tag,
            )
        if tag in _CUE_TO_ACTION:
            cued = _CUE_TO_ACTION[tag]
            return dict(
                type="Stimulus",
                start=start,
                duration=duration,
                code=_ACTION_CODES[cued],  # share the cued action's code
                description=f"cue: {cued}",  # the phase this presentation precedes
                period="delay",  # the arm holds still while the next movement is cued
                stimulus_kind="movement_cue",
                cue_tag=tag,  # extra: keep the raw positional tag
            )
        if tag in _ACTION_CODES:
            return dict(
                type="Action",
                start=start,
                duration=duration,
                code=_ACTION_CODES[tag],
                description=tag,
                period="go",
                action_kind=_ACTION_KINDS[tag],
            )
        if tag in _AUTO_MOVEMENTS:
            return dict(
                type="AutoMovement",
                start=start,
                duration=duration,
                description=tag,
            )
        if tag == "Intertrial":
            return dict(
                type="Intertrial",
                start=start,
                duration=duration,
                description=tag,
                period="intertrial",
            )
        raise ValueError(
            f"Unexpected karpowicz_2024_h1 epoch tag {tag!r}; known tags are "
            f"{sorted({*_ACTION_CODES, *_CUE_TO_ACTION, *_AUTO_MOVEMENTS, 'Intertrial'})}."
        )

    # SpecialLoaders -------------------------------------------------
    def _load_spike_counts(self, timeline: dict[str, tp.Any]) -> np.ndarray:
        """SpecialLoader target: ``(176, n_bins)`` counts on the kinematics grid."""
        with nwb.open_h5(self._resolve_source(timeline["path"])) as f:
            n_bins = f["acquisition/OpenLoopKinematics/data"].shape[0]
            stops = f["units/spike_times_index"][:].astype(np.int64)
            times = f["units/spike_times"][:]
        if len(stops) != utils.N_CHANNELS:
            raise ValueError(
                f"expected {utils.N_CHANNELS} channels, found {len(stops)}"
            )
        return utils.bin_spike_times(np.split(times, stops[:-1]), n_bins)

    def _load_eval_mask(self, timeline: dict[str, tp.Any]) -> np.ndarray:
        """SpecialLoader target: FALCON's ``eval_mask`` as ``(1, n_bins)`` bool, True on
        scored bins."""
        with nwb.open_h5(self._resolve_source(timeline["path"])) as f:
            mask = f["acquisition/eval_mask/data"][:]
        return mask.astype(bool)[np.newaxis]

    def _load_kinematics(
        self,
        timeline: dict[str, tp.Any],
        key: str,
        cols: tp.Sequence[int],
    ) -> np.ndarray:
        """SpecialLoader target: ``acquisition[key]`` ``cols`` as ``(len(cols), n_bins)``."""
        with nwb.open_nwb(self._resolve_source(timeline["path"])) as nwbfile:
            data = np.asarray(nwbfile.acquisition[key].data[:])
        return data[:, list(cols)].T.astype(np.float32)  # (len(cols), n_bins)
