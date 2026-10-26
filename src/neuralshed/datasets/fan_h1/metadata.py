DATASET_METADATA = {
    "id": "fan_h1",
    "title": "FALCON Benchmark H1 - Human 7-DoF Reach and Grasp Motor BCI",
    "description": """Open-loop 7-DoF reach and grasp (3 translation, 1 rotation, 3
        grasp) by a human iBCI participant with two 96-channel Utah arrays in
        hand-knob motor cortex. FALCON cross-day decoding-stability benchmark.""",
    "source_data_platform": "dandi",
    "source_id": "000954",
    "source_data_url": "https://dandiarchive.org/dandiset/000954",
    "status": "ml_ready",
    "macro_task": "human_motor",
    "decode_target": "movement",
    "study_class": "FanFalconH1",  # datasets/fan_h1/study.py
    "bibtex": """
@article{karpowicz2024falcon,
  title = {Few-shot Algorithms for Consistent Neural Decoding (FALCON) Benchmark},
  author = {Karpowicz, Brianna M. and Ye, Joel and Fan, Chaofei and Tostado-Marcos, Pablo and Rizzoglio, Fabio and Washington, Clay and Scodeler, Thiago and de Lucena, Diogo and Nason-Tomaszewski, Samuel R. and Mender, Matthew J. and Ma, Xuan and Arneodo, Ezequiel Matias and Hochberg, Leigh R. and Chestek, Cynthia A. and Henderson, Jaimie M. and Gentner, Timothy Q. and Gilja, Vikash and Miller, Lee E. and Rouse, Adam G. and Gaunt, Robert A. and Collinger, Jennifer L. and Pandarinath, Chethan},
  journal = {bioRxiv},
  year = {2024},
  month = sep,
  doi = {10.1101/2024.09.15.613126},
  url = {https://www.biorxiv.org/content/10.1101/2024.09.15.613126v1},
}
""",
    "publication_doi": "10.1101/2024.09.15.613126",
    "code_repo": "https://github.com/snel-repo/falcon-challenge",
    "licence": "CC-BY-4.0",
    "notes": """FALCON Benchmark H1. Spike times in a Units table (NOT pre-binned); 
        bin at 20 ms / 50 Hz to match kinematics. Kinematics sampled at 20 ms bins. 
        Downloaded locally (~102 MB, 40 files). Electrodes / electrode_groups / devices tables
        are empty in the released files; array identity documented only in metadata 
        text. Session timestamps anonymized to 1925.""",
    "subjects": [
        {
            "label": "HumanPitt",
            "participant_id": "pitt/HumanPitt",  # datasets/participants.py
            "age": None,
            "time_since_implant": None,
            "clinical_condition": [
                "Redacted/anonymized by the data owners (human iBCI participant)",
            ],
            "arrays": [
                {
                    "name": "array_1",
                    "hemisphere": None,  # not stated by the source
                    "device_type": "utah_array",
                    "n_channels": 96,
                    "channel_start": 0,
                    "channel_stop": 95,
                    "brain_area": "hand-knob motor cortex (precentral gyrus)",
                },
                {
                    "name": "array_2",
                    "hemisphere": None,  # not stated by the source
                    "device_type": "utah_array",
                    "n_channels": 96,
                    "channel_start": 96,
                    "channel_stop": 191,
                    "brain_area": "hand-knob motor cortex (precentral gyrus)",
                },
            ],
            "brain_areas": [
                {
                    "raw_label": "hand-knob motor cortex (precentral gyrus)",
                    "homba_id": "HOMBA_10165",  # upper limb regions of M1 (hand area)
                    "uberon_id": "UBERON_0001384",  # primary motor cortex (whole M1)
                    "brodmann": "4",
                },
            ],
        }
    ],
    "events": [
        {
            "name": "sorted_spikes",
            "category": "data",
            "event_class": "Spikes",
            "signal_type": "sorted_spikes",
            "modality": "intracortical_microelectrode",
            "bin_size_ms": 20.0,
            "rate_hz": 50.0,
            "dtype": "float64",
            "description": "Multi-unit spike times (Units table), 176 units across 2 x 96-ch Utah arrays; ragged spike_times, binned at 20 ms / 50 Hz by SpikesExtractor.",
        },
        {
            "name": "hand_position",
            "category": "data",
            "event_class": "PositionSeries",
            "signal_type": "kinematics",
            "modality": "behavior",
            "bin_size_ms": 20.0,
            "rate_hz": 50.0,
            "dtype": "float64",
            "description": "3-DoF hand translation (tx, ty, tz) from OpenLoopKinematics; open-loop attempted target trajectory.",
        },
        {
            "name": "hand_orientation",
            "category": "data",
            "event_class": "OrientationSeries",
            "signal_type": "kinematics",
            "modality": "behavior",
            "bin_size_ms": 20.0,
            "rate_hz": 50.0,
            "dtype": "float64",
            "description": "1-DoF wrist rotation (rx) from OpenLoopKinematics.",
        },
        {
            "name": "grasp",
            "category": "data",
            "event_class": "GraspSeries",
            "signal_type": "kinematics",
            "modality": "behavior",
            "bin_size_ms": 20.0,
            "rate_hz": 50.0,
            "dtype": "float64",
            "description": "3-DoF grasp aperture (g1, g2, g3) from OpenLoopKinematics.",
        },
        {
            "name": "hand_linear_velocity",
            "category": "data",
            "event_class": "LinearVelocitySeries",
            "signal_type": "kinematics",
            "modality": "behavior",
            "bin_size_ms": 20.0,
            "rate_hz": 50.0,
            "dtype": "float32",
            "description": "Velocity of hand translation (x, y, z) from OpenLoopKinematicsVelocity.",
        },
        {
            "name": "hand_angular_velocity",
            "category": "data",
            "event_class": "AngularVelocitySeries",
            "signal_type": "kinematics",
            "modality": "behavior",
            "bin_size_ms": 20.0,
            "rate_hz": 50.0,
            "dtype": "float32",
            "description": "Velocity of wrist rotation (x) from OpenLoopKinematicsVelocity.",
        },
        {
            "name": "grasp_velocity",
            "category": "data",
            "event_class": "GraspVelocitySeries",
            "signal_type": "kinematics",
            "modality": "behavior",
            "bin_size_ms": 20.0,
            "rate_hz": 50.0,
            "dtype": "float32",
            "description": "Velocity of grasp (g1, g2, g3) from OpenLoopKinematicsVelocity.",
        },
        {
            "name": "cue",
            "category": "categorical",
            "event_class": "Stimulus",
            "task_condition": "attempted",
            "description": "Presentation* cue phase: next target shown while the effector is held still; Stimulus.code = the cued Action code (modality unset — mixed visual/auditory cues).",
        },
        {
            "name": "movement_phase",
            "category": "categorical",
            "event_class": "Action",
            "task_condition": "attempted",
            "description": "Attempted-movement phase (Reach, Orient, SnapTo, Shape, Grasp, Carry, Orient2/Reorient, Release); one code per phase. Open-loop: subject attempts the computer-driven target.",
        },
        {
            "name": "intertrial",
            "category": "categorical",
            "event_class": "Intertrial",
            "task_condition": "na",
            "description": "Rest gap between trials.",
        },
    ],
}


def sessions() -> list[dict]:
    """Per-session rows from ``utils.list_assets()`` (one row per NWB asset).

    Requires network access (DANDI API).
    """
    from neuralshed.datasets.fan_h1 import utils

    rows = []
    for asset in utils.list_assets():
        path = asset["path"]
        rows.append(
            {
                "label": utils.session_label(path),
                "subject_label": "HumanPitt",
                # No real date: the source anonymizes session timestamps to 1925.
                "date": None,
                "original_split": asset["split"],
                "task": "reach_grasp",
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
