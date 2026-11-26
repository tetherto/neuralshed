DATASET_METADATA = {
    "id": "karpowicz_2024_h1",
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
    "study_class": "Karpowicz2024FalconH1",  # datasets/karpowicz_2024_h1/study.py
    "bibtex": """
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
""",
    "publication_doi": "10.52202/079017-2439",
    "code_repo": "https://github.com/snel-repo/falcon-challenge",
    "licence": "CC-BY-4.0",
    "notes": """FALCON Benchmark H1. The NWB Units table holds 176 channels of unsorted
        threshold crossings, not sorted units, with no electrode link. The Study bins them
        on the 20 ms kinematics grid with FALCON's bin edges. 
        Downloaded locally (~102 MB, 40 files). Electrodes / electrode_groups / devices tables
        are empty in the released files; array identity documented only in metadata 
        text. Session timestamps anonymized to 1925. The data owners withhold the
        participant's clinical condition, age and implant date.""",
    "subjects": [
        {
            "label": "HumanPitt",
            "participant_id": "pitt/HumanPitt",  # datasets/participants.py
            "age": None,
            "time_since_implant": None,
            "clinical_condition": [],
            "arrays": [
                {
                    "name": "array_1",
                    "hemisphere": None,  # not stated by the source
                    "device_type": "utah_array",
                    "n_channels": 96,
                    "channel_start": None,  # no channel-to-array link released
                    "channel_stop": None,
                    "brain_area": "hand-knob motor cortex (precentral gyrus)",
                },
                {
                    "name": "array_2",
                    "hemisphere": None,  # not stated by the source
                    "device_type": "utah_array",
                    "n_channels": 96,
                    "channel_start": None,  # no channel-to-array link released
                    "channel_stop": None,
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
            "name": "threshold_crossings",
            "category": "data",
            "event_class": "SpikeCounts",
            "signal_type": "threshold_crossings",
            "modality": "intracortical_microelectrode",
            "bin_size_ms": 20.0,
            "rate_hz": 50.0,
            "dtype": "float64",
            "description": "Unsorted threshold crossings on 176 channels, released as the rows of the NWB Units table and binned at 20 ms by the Study. Which electrode each channel comes from is not released. Bin k counts the 20 ms ending at kinematic sample k (k x 0.02 s), as FALCON bins them.",
        },
        {
            "name": "eval_mask",
            "category": "data",
            "event_class": "BinaryMask",
            "signal_type": "mask",
            "modality": "other",
            "bin_size_ms": 20.0,
            "rate_hz": 50.0,
            "dtype": "bool",
            "description": "FALCON's per-bin eval_mask (acquisition/eval_mask), True on the bins FALCON trains on and scores. False covers every SnapTo bin, the leading Intertrial rows and a few Orient bins. The NWB description ('Timesteps to ignore covariates') states the opposite polarity.",
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
            "description": "1-DoF wrist roll (rx) from OpenLoopKinematics.",
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
            "description": "3-DoF grasp (g1, g2, g3: pinch, scoop, thumb abduction) from OpenLoopKinematics, each 0-1 between a neutral and a predefined grasp pose.",
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
            "description": "Velocity of hand translation (x, y, z) from OpenLoopKinematicsVelocity, per 20 ms bin (FALCON's decoding target).",
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
            "description": "Velocity of wrist roll (rx) from OpenLoopKinematicsVelocity, per 20 ms bin (FALCON's decoding target).",
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
            "description": "Velocity of grasp (pinch, scoop, thumb abduction) from OpenLoopKinematicsVelocity, per 20 ms bin (FALCON's decoding target).",
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
            "description": "Attempted-movement phase (Reach, Orient, Shape, Grasp, Carry, Orient2, Release); one code per phase. Open-loop: the participant attempts each movement of an autocontrolled virtual arm.",
        },
        {
            "name": "snap_to",
            "category": "categorical",
            "event_class": "AutoMovement",
            "task_condition": "na",
            "description": "SnapTo: an uncued, short move of the virtual hand onto the object between Orient and the Shape cue, one per trial, unscored by FALCON. Not described by the source.",
        },
        {
            "name": "intertrial",
            "category": "categorical",
            "event_class": "Intertrial",
            "task_condition": "na",
            "description": "Rest before the first trial of a recording.",
        },
    ],
}


def sessions() -> list[dict]:
    """One row per recording, listed as the Study lists them: the local files when
    downloaded, else the DANDI assets (network). Minival copies left out.

    ``n_trials`` counts the distinct source ``TrialNum`` values of the recording.
    """
    import numpy as np

    from neuralshed.datasets import nwb
    from neuralshed.datasets.karpowicz_2024_h1 import utils

    root = utils.default_data_dir()
    paths = nwb.dandi_nwb_paths(utils.DANDISET_ID, utils.VERSION, root=root)
    rows = []
    for rec in utils.recordings(paths):
        source = nwb.resolve_source(
            rec["path"], utils.DANDISET_ID, utils.VERSION, root=root
        )
        with nwb.open_h5(source) as f:
            trial_num = f["acquisition/TrialNum/data"][:]
        rows.append(
            {
                "label": rec["session"],
                "subject_label": utils.SUBJECT,
                # No real date: the source anonymizes session timestamps to 1925.
                "date": None,
                "n_trials": int(np.unique(trial_num).size),
                "original_split": rec["split"],
                "task": "reach_grasp",
                "task_condition": "attempted",
            }
        )
    return rows


def get_metadata() -> dict:
    """Full dataset metadata: catalog (DATASET_METADATA) + per-session rows.

    Sessions need the data drive or the DANDI API, and a failure raises.
    """
    meta = DATASET_METADATA.copy()
    meta["sessions"] = sessions()
    return meta
