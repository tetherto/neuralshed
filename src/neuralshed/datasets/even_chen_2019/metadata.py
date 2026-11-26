_BIBTEX = """
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

# Both monkeys carry a 96-channel Utah array in M1 and one in caudal PMd of the left
# hemisphere. Channels 0-95 are M1 and 96-191 PMd, following the electrodes table rather
# than the units table's own electrode_group column, which says the opposite (see utils.py).
_BRAIN_AREAS = [
    {
        "raw_label": "M1 (primary motor cortex)",
        "homba_id": "HOMBA_10163",  # primary motor cortex (area 4); placed by landmarks, no somatotopic claim
        "uberon_id": "UBERON_0001384",  # primary motor cortex
        "brodmann": "4",
    },
    {
        "raw_label": "PMd (caudal dorsal premotor cortex)",
        "homba_id": "HOMBA_10169",  # laterodorsal division of area 6 (A6ld); dorsal premotor
        "uberon_id": "UBERON_0016634",  # premotor cortex (no dorsal subdivision in UBERON)
        "brodmann": "6",
    },
]


def _arrays() -> list[dict]:
    return [
        {
            "name": "M1",
            "hemisphere": "left",
            "device_type": "utah_array",
            "n_channels": 96,
            "channel_start": 0,
            "channel_stop": 95,
            "brain_area": "M1 (primary motor cortex)",
        },
        {
            "name": "PMd",
            "hemisphere": "left",
            "device_type": "utah_array",
            "n_channels": 96,
            "channel_start": 96,
            "channel_stop": 191,
            "brain_area": "PMd (caudal dorsal premotor cortex)",
        },
    ]


DATASET_METADATA = {
    "id": "even_chen_2019",
    "title": "Structure and variability of delay activity in premotor cortex",
    "description": (
        "Two rhesus macaques (JenkinsC and Reggie) make instructed-delay center-out-and-back "
        "arm reaches that move a cursor to one of 24 to 48 targets laid out as a line, a "
        "ring or three rings, with a 96-channel Utah array in M1 and one in PMd of the left "
        "hemisphere. Twelve sessions (5 JenkinsC, 7 Reggie), 56,616 trials, with 192 "
        "threshold-crossing channels, hand, cursor and eye position, and the trial table. "
        "One Reggie session contains an unflagged 134-trial brain-control block."
    ),
    "source_data_platform": "dandi",
    "source_id": "000121",
    "source_data_url": "https://dandiarchive.org/dandiset/000121/0.220124.2156",
    "status": "study_wip",
    "macro_task": "nhp_motor",
    "decode_target": "movement",
    "study_class": "EvenChen2019DelayedReach",  # datasets/even_chen_2019/study.py
    "bibtex": _BIBTEX,
    "publication_doi": "10.1371/journal.pcbi.1006808",
    "code_repo": None,
    "licence": "CC-BY-4.0",
    "notes": (
        "NWB 2.2.5, one file per session, read with h5py. Positions are millimeters from "
        "screen center although the files declare meters. Behavior is a 60 Hz tracker held "
        "on a 1 kHz grid, and neither behavior nor spikes exist between trials (220 ms gaps "
        "for JenkinsC, 40 ms to 11 min for Reggie). Hand tracking loss is written as a "
        "fixed sentinel row that the cursor copies, and eye dropouts as all-zero rows. "
        "Reggie 2017-01-19 trials 974 to 1107 are brain control, flagged only by DecodePos "
        "being non-NaN, and the next 9 trials have the cursor off the screen. task_id and "
        "version_id are constant per monkey, so the target layout is derived from "
        "target_pos, and the paper's line labels for Reggie are swapped relative to the "
        "files. The units table's electrode_group column contradicts the electrodes table; "
        "the electrodes table is followed (0-95 M1, 96-191 PMd). Reggie has 30 silent "
        "channels in every session (29 on M1, 1 on PMd). The LFP segments have NaN start "
        "times and cannot be aligned, juice_reward holds undocumented values, and DecodePos "
        "is filled only in the brain-control block; none of the three is emitted."
    ),
    "subjects": [
        {
            "label": "JenkinsC",
            # Likely the Jenkins of churchland_2012.
            "participant_id": "shenoy/Jenkins",
            "age": None,
            # The paper: arrays implanted 74 months before this study (late 2015).
            "time_since_implant": "P74M",
            "clinical_condition": [],
            "arrays": _arrays(),
            "brain_areas": _BRAIN_AREAS,
        },
        {
            "label": "Reggie",
            "participant_id": "shenoy/Reggie",
            "age": None,
            "time_since_implant": "P64M",  # the paper, 64 months before this study
            "clinical_condition": [],
            "arrays": _arrays(),
            "brain_areas": _BRAIN_AREAS,
        },
    ],
    "events": [
        {
            "name": "units",
            "category": "data",
            "event_class": "SpikeCounts",
            "signal_type": "threshold_crossings",
            "modality": "intracortical_microelectrode",
            "bin_size_ms": 20.0,
            "rate_hz": 50.0,
            "dtype": "float64",
            "description": (
                "Threshold crossings on 192 channels (one units row per electrode, not "
                "sorted units), released as spike times and binned at 20 ms by the Study. "
                "Thresholds -4.5 x RMS (JenkinsC) and -4.0 x RMS (Reggie)."
            ),
        },
        {
            "name": "Hand",
            "category": "data",
            "event_class": "PositionSeries",
            "signal_type": "kinematics",
            "modality": "behavior",
            "bin_size_ms": 20.0,
            "rate_hz": 50.0,
            "dtype": "float64",
            "description": (
                "Hand x, y, z in mm from an optical bead tracker at 60 frames/s, held on a "
                "1 kHz grid in the source and averaged into 20 ms bins by the Study."
            ),
        },
        {
            "name": "Cursor",
            "category": "data",
            "event_class": "CursorPositionSeries",
            "signal_type": "kinematics",
            "modality": "behavior",
            "bin_size_ms": 20.0,
            "rate_hz": 50.0,
            "dtype": "float64",
            "description": (
                "Cursor x, y in mm. The hand's x, y one sample later under hand control, "
                "the decoder output during the brain-control block."
            ),
        },
        {
            "name": "Eye",
            "category": "data",
            "event_class": "PositionSeries",
            "signal_type": "kinematics",
            "modality": "behavior",
            "bin_size_ms": 20.0,
            "rate_hz": 50.0,
            "dtype": "float64",
            "description": "Eye x, y in raw uncalibrated units (0 to about 5700), updated at about 120 Hz.",
        },
        {
            "name": "target_pos",
            "category": "data",
            "event_class": "TargetPositionSeries",
            "signal_type": "kinematics",
            "modality": "derived",
            "bin_size_ms": 20.0,
            "rate_hz": 50.0,
            "dtype": "float64",
            "description": "Cued target x, y in mm from the trial table, from target appearance to trial end.",
        },
        {
            "name": "target_cue",
            "category": "categorical",
            "event_class": "Stimulus",
            "description": "Reach target on screen, from target_shown_time to the trial end.",
        },
        {
            "name": "reach",
            "category": "categorical",
            "event_class": "Action",
            "task_condition": "executed",
            "description": (
                "Cursor movement to the target from the go cue to the trial end, by arm "
                "reach under hand control or by the decoder in the brain-control block. "
                "Carries success, target geometry, delay and loop_type."
            ),
        },
        {
            "name": "intertrial",
            "category": "categorical",
            "event_class": "Intertrial",
            "description": "Gap between trials, where the release holds no neural or behavioral data.",
        },
    ],
}


def sessions() -> list[dict]:
    """One row per NWB session, from the local files (trial counts need the data drive)."""
    from . import utils

    data_dir = utils.default_data_dir()
    rows = []
    for ref in utils.list_sessions(data_dir):
        row = {
            "label": ref.label,
            "subject_label": ref.subject,
            "date": ref.date,
            "post_implant_day": None,  # the paper gives the implant age in months only
            "task": "center_out_and_back",
            "task_condition": "executed",
        }
        if data_dir is not None and (data_dir / ref.path).is_file():
            trials = utils.get_trials(str(data_dir / ref.path))
            row["n_trials"] = len(trials)
            row["task"] = f"center_out_and_back ({utils.target_layout(trials)[0]})"
        rows.append(row)
    return rows


def get_metadata() -> dict:
    """Full dataset metadata: catalog (DATASET_METADATA) + per-session rows.

    Sessions need the data drive or the DANDI API; omitted (empty list) when unavailable.
    """
    meta = DATASET_METADATA.copy()
    try:
        meta["sessions"] = sessions()
    except Exception:
        meta["sessions"] = []
    return meta


__all__ = ["DATASET_METADATA", "get_metadata", "sessions"]
