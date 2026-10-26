"""neuroai ``Study`` for FALCON Benchmark H1 (DANDI 000954).

Each released NWB session is one :term:`timeline`. For each we emit:

* one ``Spikes`` event -- the multi-unit spike-times ``Units`` table, opened
  (local or streamed, via a ``SpecialLoader``) and binned by ``neuralset``
  ``SpikesExtractor`` at the FALCON H1 cadence (50 Hz / 20 ms);
* one event per ``intervals/epochs`` row -- each row is a single *task phase*
  tagged with exactly one of 16 labels. The phases tile each trial contiguously
  in the canonical Pittsburgh reach-grasp-transport sequence
  ``Reach -> Orient -> Grasp -> Carry -> Reorient -> Release`` (see ``datasets/fan_h1/README.md``).
  We split them three ways by tag:

  - **``Stimulus``** -- the ``Presentation*`` *cue* phases, during which the
    effector is held still while the next target is shown to the subject. The
    number is positional (which movement it cues), so each cue's ``code`` is set
    to the ``code`` of the ``Action`` it precedes; ``modality`` is left unset
    (cues were mixed visual LED/VR + auditory verbal across paradigms).
  - **``Action``** -- the attempted-movement phases (``Reach``, ``Orient``,
    ``SnapTo``, ``Shape``, ``Grasp``, ``Carry``, ``Orient2``, ``Release``), one
    ``code`` per phase. ``Orient`` (pre-grasp) and ``Orient2`` (the papers'
    "Reorient" phase, post-carry) are genuinely distinct movements.
  - **``Intertrial``** -- the rest gap between trials (custom event type).

  These are open-loop *observation* data: the kinematics are computer-driven
  target trajectories the (tetraplegic) subject watched and *attempted* to
  perform, hence ``task_condition="attempted"`` on every ``Action``.

NWB layout (see ``datasets/fan_h1/README.md``):

* ``units`` -- Units table, 176 multi-units, ragged ``spike_times`` (seconds).
* ``acquisition/OpenLoopKinematics`` -- ``TimeSeries (n_bins, 7)``; the 7 DoF mix
  3 translation (``tx,ty,tz``), 1 wrist rotation (``rx``), 3 grasp (``g1,g2,g3``).
* ``intervals/epochs`` -- one row per task phase, single ``tags`` entry plus
  ``start_time``/``stop_time``

The 7-DoF kinematics are split by quantity into homogeneous events, at both the
position level (``OpenLoopKinematics``) and the velocity level
(``OpenLoopKinematicsVelocity``): ``Position``/``Orientation``/``Grasp`` Series
(x,y,z / x / g1,g2,g3) and ``Linear``/``Angular``/``Grasp`` VelocitySeries. Each is
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
    GraspSeries,
    GraspVelocitySeries,
    Intertrial,
    LinearVelocitySeries,
    OrientationSeries,
    PositionSeries,
)

from . import utils

# FALCON H1 constants (see datasets/fan_h1/utils.py / README.md).
_SUBJECT = "HumanPitt"
_DANDISET_ID = "000954"
_VERSION = "draft"
_BIN_SIZE_S = 0.02  # 20 ms bins
_BIN_RATE_HZ = 50.0  # = 1 / _BIN_SIZE_S; also the Spikes binning frequency

# Decompose the 7-DoF kinematics (cols tx,ty,tz,rx,g1,g2,g3) into homogeneous events,
# at both the position level (``OpenLoopKinematics``) and velocity level
# (``OpenLoopKinematicsVelocity``): (event type, NWB key, column indices, dim names).
_KINEMATICS: tuple[tuple[str, str, tuple[int, ...], tuple[str, ...]], ...] = (
    ("PositionSeries", "OpenLoopKinematics", (0, 1, 2), ("x", "y", "z")),
    ("OrientationSeries", "OpenLoopKinematics", (3,), ("x",)),
    ("GraspSeries", "OpenLoopKinematics", (4, 5, 6), ("g1", "g2", "g3")),
    ("LinearVelocitySeries", "OpenLoopKinematicsVelocity", (0, 1, 2), ("x", "y", "z")),
    ("AngularVelocitySeries", "OpenLoopKinematicsVelocity", (3,), ("x",)),
    (
        "GraspVelocitySeries",
        "OpenLoopKinematicsVelocity",
        (4, 5, 6),
        ("g1", "g2", "g3"),
    ),
)

# -- Epoch-tag taxonomy (see datasets/fan_h1/README.md "Task phases / events") ---
# Each epochs row carries exactly one of these 16 tags. The participant's
# attempted-movement phases become ``Action`` events, one ``code`` per phase in
# canonical task order (Reach .. Release; ``Orient2`` is the "Reorient" phase and
# is distinct from ``Orient``).
_ACTION_CODES: dict[str, int] = {
    "Reach": 0,
    "Orient": 1,
    "SnapTo": 2,
    "Shape": 3,
    "Grasp": 4,
    "Carry": 5,
    "Orient2": 6,  # the references' "Reorient" phase (rotate the grasped object)
    "Release": 7,
}
# The ``Presentation*`` phases are cue periods, the number is positional.
# Each maps to the ``Action`` it precedes, and its ``Stimulus.code``
# is set to that action's code so a cue and the movement it cues share a code.
# (``SnapTo`` follows ``Orient`` directly and has no preceding cue.)
_CUE_TO_ACTION: dict[str, str] = {
    "Presentation": "Reach",
    "Presentation2": "Orient",
    "Presentation3": "Shape",
    "Presentation4": "Grasp",
    "Presentation5": "Carry",
    "Presentation6": "Orient2",
    "Presentation7": "Release",
}


def _tag_of(tags: tp.Any) -> str:
    """The single phase label from an epochs-row ``tags`` cell.

    Every fan_h1 epochs row carries exactly one tag, we read element 0 and stringify it.
    """
    return str(list(tags)[0])


class FanFalconH1(NeuralshedStudy):
    """FALCON Benchmark H1: human 7-DoF reach & grasp iBCI (DANDI 000954).

    Open-loop 7-DoF reach and grasp by a single iBCI participant with two
    96-channel Utah arrays in hand-knob motor cortex. One timeline per released
    NWB session (40 across the held-in/held-out FALCON splits).
    """

    dataset_id: tp.ClassVar[str] = "fan_h1"
    aliases: tp.ClassVar[tuple[str, ...]] = ("FALCON Benchmark H1", "FALCON-H1")
    bibtex: tp.ClassVar[str] = """
@article{karpowicz2024falcon,
  title = {Few-shot Algorithms for Consistent Neural Decoding (FALCON) Benchmark},
  author = {Karpowicz, Brianna M. and Ye, Joel and Fan, Chaofei and Tostado-Marcos, Pablo and Rizzoglio, Fabio and Washington, Clay and Scodeler, Thiago and de Lucena, Diogo and Nason-Tomaszewski, Samuel R. and Mender, Matthew J. and Ma, Xuan and Arneodo, Ezequiel Matias and Hochberg, Leigh R. and Chestek, Cynthia A. and Henderson, Jaimie M. and Gentner, Timothy Q. and Gilja, Vikash and Miller, Lee E. and Rouse, Adam G. and Gaunt, Robert A. and Collinger, Jennifer L. and Pandarinath, Chethan},
  journal = {bioRxiv},
  year = {2024},
  month = sep,
  doi = {10.1101/2024.09.15.613126},
  url = {https://www.biorxiv.org/content/10.1101/2024.09.15.613126v1},
}
"""
    url: tp.ClassVar[str] = "https://dandiarchive.org/dandiset/000954"
    licence: tp.ClassVar[str] = "CC-BY-4.0"
    description: tp.ClassVar[str] = (
        "Open-loop 7-DoF reach and grasp by a human iBCI participant with two "
        "96-channel Utah arrays in hand-knob motor cortex (FALCON H1 cross-day "
        "decoding-stability benchmark, DANDI 000954)."
    )
    requirements: tp.ClassVar[tuple[str, ...]] = ("dandi", "pynwb", "h5py", "remfile")

    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=40,
        num_subjects=1,
        query="timeline_index < 1",  # default; first session only
        num_events_in_query=194,  # 7 signal events + 187 epoch rows (session 0)
        event_types_in_query={
            "Spikes",
            "PositionSeries",
            "OrientationSeries",
            "GraspSeries",
            "LinearVelocitySeries",
            "AngularVelocitySeries",
            "GraspVelocitySeries",
            "Stimulus",
            "Action",
            "Intertrial",
        },
    )

    # -- data access ------------------------------------------------------
    def _download(self) -> None:
        from neuralfetch.download import Dandi

        dandi_client = Dandi(
            study=_DANDISET_ID,
            version=_VERSION,
            dset_dir=self.path,
            folder="",
        )
        dandi_client.download()

    def _nwb_files_paths(self) -> list[str]:
        """Local ``.nwb`` session files when present (excludes macOS ``._`` sidecars).
        Otherwise enumerate the dandiset and stream (see _resolve_source)."""
        local = [p for p in self.path.rglob("*.nwb") if not p.name.startswith("._")]
        rels = (
            [str(p.relative_to(self.path)) for p in local]
            if local
            else [
                _DANDISET_ID + "/" + a["path"]
                for a in nwb.dandi_list_assets(_DANDISET_ID, _VERSION)
            ]
        )
        return sorted(rels)

    def _resolve_source(self, timeline_path: str) -> str:
        """Resolves the session's NWB source: a file path when a local file exists
        (e.g. after ``_download``), else its DANDI streaming URL."""
        local = self.path / timeline_path
        if local.is_file():
            return str(local)
        asset_path = timeline_path.removeprefix(_DANDISET_ID + "/")
        return nwb.dandi_asset_url(_DANDISET_ID, asset_path, _VERSION)

    # -- timelines + events ----------------------------------------------
    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        """One timeline per NWB session."""
        nwb_files_paths = self._nwb_files_paths()
        for rel in nwb_files_paths:
            yield dict(
                subject=_SUBJECT,
                session=utils.session_label(rel),
                original_split=utils.split_of(rel),
                task="reach_grasp",
                path=rel,
            )

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        with nwb.open_nwb(self._resolve_source(timeline["path"])) as nwbfile:
            n_bins = int(nwbfile.acquisition["OpenLoopKinematics"].data.shape[0])
            duration_s = n_bins * _BIN_SIZE_S
            if nwbfile.intervals is not None and "epochs" in nwbfile.intervals:
                epochs = nwbfile.intervals["epochs"].to_dataframe()
            else:
                epochs = pd.DataFrame(columns=["start_time", "stop_time", "tags"])

        rows: list[dict[str, tp.Any]] = [
            dict(
                type="Spikes",
                start=0.0,
                duration=float(duration_s),
                frequency=_BIN_RATE_HZ,
                filepath=study.SpecialLoader(
                    method=self._load_spikes,
                    timeline=timeline,
                ).to_json(),
            )
        ]
        rows += self._kinematics_events(timeline, duration_s)
        for ep in epochs.itertuples(index=False):
            rows.append(
                self._phase_event(
                    _tag_of(ep.tags),
                    ep.start_time,
                    ep.stop_time,
                )
            )
        return pd.DataFrame(rows)

    def _kinematics_events(
        self,
        timeline: dict[str, tp.Any],
        duration_s: float,
    ) -> list[dict[str, tp.Any]]:
        """Six homogeneous kinematics rows (position + velocity; subject added by standardize_events)."""
        out: list[dict[str, tp.Any]] = []
        for etype, key, cols, dims in _KINEMATICS:
            loader = study.SpecialLoader(
                method=self._load_kinematics,
                timeline=timeline,
                key=key,
                cols=list(cols),
            ).to_json()
            out.append(
                dict(
                    type=etype,
                    start=0.0,
                    duration=float(duration_s),
                    frequency=_BIN_RATE_HZ,
                    dims=dims,
                    n_features=len(dims),
                    unit="arbitrary",  # OpenLoopKinematics is normalized (NWB unit)
                    reference_frame="open-loop target kinematics (Pittsburgh reach-grasp-transport)",
                    filepath=loader,
                )
            )
        return out

    @staticmethod
    def _phase_event(
        tag: str,
        start_time: tp.Any,
        stop_time: tp.Any,
    ) -> dict[str, tp.Any]:
        """Map one epoch phase (single ``tag``) to its event row:

        - ``Presentation*`` -> ``Stimulus`` (code = the cued action's code);
        - attempted-movement phases -> ``Action`` (one code per phase);
        - ``Intertrial`` -> ``Intertrial``. Unknown tags raise (the released tag set
            is fully enumerated, so a surprise means the taxonomy is stale).
        """
        start = float(start_time)
        duration = max(0.0, float(stop_time) - start)
        if tag in _CUE_TO_ACTION:
            cued = _CUE_TO_ACTION[tag]
            return dict(
                type="Stimulus",
                start=start,
                duration=duration,
                code=_ACTION_CODES[cued],  # share the cued action's code
                description=f"cue: {cued}",  # the phase this presentation precedes
                cue_tag=tag,  # extra: keep the raw positional tag
            )
        if tag in _ACTION_CODES:
            return dict(
                type="Action",
                start=start,
                duration=duration,
                code=_ACTION_CODES[tag],
                description=tag,
                task_condition="attempted",  # extra: open-loop observation data
            )
        if tag == "Intertrial":
            return dict(
                type="Intertrial",
                start=start,
                duration=duration,
                description=tag,
            )
        raise ValueError(
            f"Unexpected fan_h1 epoch tag {tag!r}; known tags are "
            f"{sorted({*_ACTION_CODES, *_CUE_TO_ACTION, 'Intertrial'})}."
        )

    # SpecialLoaders -------------------------------------------------
    def _load_spikes(self, timeline: dict[str, tp.Any]) -> tp.Any:
        """SpecialLoader target: the session NWB as a live ``h5py.File`` (local or
        streamed) for ``SpikesExtractor`` to bin; it closes the handle."""
        return nwb.load_h5(self._resolve_source(timeline["path"]))

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
