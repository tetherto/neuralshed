_M1 = "Primary Motor Cortex"
_PMD = "Dorsal Premotor Cortex"

#: Region labels as the NWB electrodes table names them. Both arrays are reported only
#: as M1 and PMd, so no somatotopic subregion is asserted.
_BRAIN_AREAS = {
    _M1: {
        "raw_label": _M1,
        "homba_id": "HOMBA_10163",  # primary motor cortex (area 4)
        "uberon_id": "UBERON_0001384",  # primary motor cortex
        "brodmann": "4",
    },
    _PMD: {
        "raw_label": _PMD,
        "homba_id": "HOMBA_10169",  # laterodorsal division of area 6 (PMd)
        "uberon_id": "UBERON_0016634",  # premotor cortex
        "brodmann": "6",
    },
}


def _array(name: str, hemisphere: str | None, start: int) -> dict:
    # channel_start/stop index the NWB electrodes table (M1 rows first). The emitted
    # SpikeCounts row holds sorted units, not electrodes, so its width varies by session.
    return {
        "name": name,
        "hemisphere": hemisphere,
        "device_type": "utah_array",
        "n_channels": 96,
        "channel_start": start,
        "channel_stop": start + 95,
        "brain_area": _M1 if name == "M1" else _PMD,
    }


DATASET_METADATA = {
    "id": "perich_2018",
    "title": "Long-term recordings of motor and premotor cortical spiking activity during reaching in monkeys",
    "description": (
        "Monkeys C, J, M and T move a cursor with the handle of a two-link planar "
        "manipulandum, in a center-out task (eight targets at 8 cm, instructed delay, "
        "auditory go cue) or a random-target task (four sequential targets per trial, no "
        "delay). 111 sessions recorded between 2013 and 2016, about 43 hours and 35,410 "
        "trials, with sorted units from 96-channel Utah arrays in M1 and, in 52 sessions, "
        "PMd, and cursor position, velocity and acceleration at 100 Hz. The release holds "
        "unperturbed reaching only: the force-field and rotation epochs of Perich et al. "
        "2018 are not in it."
    ),
    "source_data_platform": "dandi",
    "source_id": "000688",
    "source_data_url": "https://dandiarchive.org/dandiset/000688",
    "status": "study_wip",
    "macro_task": "nhp_motor",
    "decode_target": "movement",
    "study_class": "Perich2018Reaching",
    "bibtex": """
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
@article{glaser2018population,
  title = {Population coding of conditional probability distributions in dorsal premotor cortex},
  author = {Glaser, Joshua I. and Perich, Matthew G. and Ramkumar, Pavan and Miller, Lee E. and Kording, Konrad P.},
  journal = {Nature Communications},
  volume = {9},
  pages = {1788},
  year = {2018},
  doi = {10.1038/s41467-018-04062-6},
  url = {https://doi.org/10.1038/s41467-018-04062-6},
}
""",
    "publication_doi": "10.1016/j.neuron.2018.09.030",
    "code_repo": None,
    "licence": "CC-BY-4.0",
    "notes": (
        "A pool of sessions from Perich 2018, Glaser 2018, Perich and Miller 2017 and "
        "Gallego 2020, assembled for the POYO pretraining set (Azabou et al. 2023). Only "
        "unperturbed reaching is released. The seven Monkey C CO sessions of 2016-09-09 "
        "to 2016-09-23 carry a truncated 'Warning:' in session_description and a cursor "
        "at about 0.53 of the target geometry; the Study marks them valid=False. "
        "target_corners is shifted by one row in eight CO sessions, so target positions "
        "come from target_dir (correct everywhere) at the 8 cm radius. RT target "
        "coordinates hold NaN and float garbage near 1e37, and 15 to 96 rewarded RT "
        "trials per session log no targets or go cues. Nine Monkey C CO sessions have a "
        "zero instructed delay. The dandiset says one subject is threshold crossings "
        "without naming it; the Study declares every subject as sorted. Monkey T's files "
        "declare an M1 array that Glaser 2018 and Gallego 2020 do not mention. The "
        "local dandiset.yaml asset summary (99 files, 3 subjects) is stale."
    ),
    "subjects": [
        {
            "label": "C",
            "participant_id": "miller/C",
            "age": "P5Y7M",  # date_of_birth 2008-03-01, first session 2013-10-03
            "time_since_implant": None,
            "clinical_condition": [],
            # Gallego 2020: a right-M1 array first (2013 to 2015 sessions, 96 electrodes),
            # later left M1 and left PMd arrays (2016 sessions, 192 electrodes). The files
            # do not state the hemisphere.
            "arrays": [
                _array("M1", "right", 0),
                _array("M1", "left", 0),
                _array("PMd", "left", 96),
            ],
            "brain_areas": [_BRAIN_AREAS[_M1], _BRAIN_AREAS[_PMD]],
        },
        {
            "label": "J",
            "participant_id": "miller/J",
            "age": None,  # the files give only the range P8Y/P15Y
            "time_since_implant": None,
            "clinical_condition": [],
            "arrays": [_array("M1", "left", 0)],  # Gallego 2020: left M1
            "brain_areas": [_BRAIN_AREAS[_M1]],
        },
        {
            "label": "M",
            "participant_id": "miller/M",
            "age": "P5Y10M",  # date_of_birth 2008-03-01, first session 2014-01-14
            "time_since_implant": None,
            "clinical_condition": [],
            # Hemisphere inferred: the ma_2023 release says Monkey M reached with the left
            # hand and the arrays were contralateral to the working arm.
            "arrays": [_array("M1", "right", 0), _array("PMd", "right", 96)],
            "brain_areas": [_BRAIN_AREAS[_M1], _BRAIN_AREAS[_PMD]],
        },
        {
            "label": "T",
            "participant_id": "miller/T",
            "age": "P8Y8M",  # date_of_birth 2004-12-01, first session 2013-08-19
            "time_since_implant": None,
            "clinical_condition": [],
            # The files declare an M1 array (3 to 12 units per session); Glaser 2018 and
            # Gallego 2020 describe a PMd array only.
            "arrays": [_array("M1", None, 0), _array("PMd", None, 96)],
            "brain_areas": [_BRAIN_AREAS[_M1], _BRAIN_AREAS[_PMD]],
        },
    ],
    "events": [
        {
            "name": "units",
            "category": "data",
            "event_class": "SpikeCounts",
            "signal_type": "sorted_spikes",
            "modality": "intracortical_microelectrode",
            "bin_size_ms": 10.0,
            "rate_hz": 100.0,
            "dtype": "float64",
            "description": (
                "Sorted units (Units table spike times, high-pass 750 Hz), 18 to 353 per "
                "session, binned by the Study onto the 10 ms cursor grid, M1 units first."
            ),
        },
        {
            "name": "cursor_pos",
            "category": "data",
            "event_class": "CursorPositionSeries",
            "signal_type": "kinematics",
            "modality": "behavior",
            "bin_size_ms": 10.0,
            "rate_hz": 100.0,
            "dtype": "float64",
            "description": "Cursor x and y in cm, origin at the screen centre, following the manipulandum handle.",
        },
        {
            "name": "cursor_vel",
            "category": "data",
            "event_class": "CursorVelocitySeries",
            "signal_type": "kinematics",
            "modality": "behavior",
            "bin_size_ms": 10.0,
            "rate_hz": 100.0,
            "dtype": "float64",
            "description": "Cursor velocity in cm/s, the numerical derivative of cursor_pos.",
        },
        {
            "name": "cursor_acc",
            "category": "data",
            "event_class": "AccelerationSeries",
            "signal_type": "kinematics",
            "modality": "behavior",
            "bin_size_ms": 10.0,
            "rate_hz": 100.0,
            "dtype": "float64",
            "description": "Cursor acceleration in cm/s^2, the numerical derivative of cursor_vel.",
        },
        {
            "name": "target_cue",
            "category": "categorical",
            "event_class": "Stimulus",
            "task_condition": "na",
            "description": (
                "CO: the outer target over the instructed delay. RT: a one-bin marker when "
                "each target appears. Carries the target position in cm."
            ),
        },
        {
            "name": "go_cue",
            "category": "categorical",
            "event_class": "Stimulus",
            "task_condition": "na",
            "description": "CO auditory go cue, one 10 ms bin.",
        },
        {
            "name": "reach",
            "category": "categorical",
            "event_class": "Action",
            "task_condition": "executed",
            "description": (
                "CO: go cue to trial end. RT: one row per reach, go cue to the next go cue. "
                "Carries target position, code (CO target index), result and success."
            ),
        },
        {
            "name": "intertrial",
            "category": "categorical",
            "event_class": "Intertrial",
            "task_condition": "na",
            "description": "Gaps between trials, and the hold before the first target of a trial.",
        },
    ],
}


def sessions() -> list[dict]:
    """One row per local NWB file."""
    import h5py

    from neuralshed.datasets.utils import dataset_dir

    from . import utils

    root = dataset_dir("perich_2018")
    rows = []
    for ref in utils.list_sessions(root):
        with h5py.File(root / ref.path, "r") as h5:
            n_trials = len(h5["intervals/trials/id"])
        rows.append(
            {
                "label": ref.label,
                "subject_label": ref.subject,
                "date": ref.date,
                "task": ref.task,
                "task_condition": "executed",
                "n_trials": n_trials,
            }
        )
    return rows


def get_metadata() -> dict:
    """Full dataset metadata: catalog (DATASET_METADATA) + per-session rows.

    Sessions need the data drive; omitted (empty list) when unavailable.
    """
    meta = DATASET_METADATA.copy()
    try:
        meta["sessions"] = sessions()
    except Exception:
        meta["sessions"] = []
    return meta


__all__ = ["DATASET_METADATA", "sessions", "get_metadata"]
