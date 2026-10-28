DATASET_METADATA = {
    "id": "deo_2024",
    "title": "Brain control of bimanual movement enabled by recurrent neural networks",
    "description": "Delayed bimanual and unimanual control of two on-screen cursors by "
    "BrainGate2 participant T5 (C4 spinal cord injury, tetraplegia). On each trial one of "
    "three movement types is cued (unimanual right, unimanual left, or bimanual), with a "
    "1-2 s instructed delay before a GO cue. Data span open-loop instructed-delay blocks "
    "(cursors move autonomously) and real-time closed-loop blocks where RNN or linear "
    "decoders drive the two cursors from neural activity. Neural features are 20 ms-binned "
    "threshold crossings from two 96-channel Utah arrays (192 channels) in hand-knob motor "
    "cortex (left precentral gyrus). Behavioral streams are the two cursors' and targets' "
    "(x, y) positions.",
    "source_data_platform": "dryad",
    "source_id": "doi:10.5061/dryad.sn02v6xbb",
    "source_data_url": "https://datadryad.org/dataset/doi:10.5061/dryad.sn02v6xbb",
    "status": "ml_ready",
    "macro_task": "human_motor",
    "decode_target": "movement",
    "study_class": "Deo2024Bimanual",  # datasets/deo_2024/study.py (Phase 2, tentative)
    "bibtex": """
@article{deo2024bimanual,
  title = {Brain control of bimanual movement enabled by recurrent neural networks},
  author = {Deo, Darrel R. and Willett, Francis R. and Avansino, Donald T. and Hochberg, Leigh R. and Henderson, Jaimie M. and Shenoy, Krishna V.},
  journal = {Scientific Reports},
  volume = {14},
  year = {2024},
  month = jan,
  doi = {10.1038/s41598-024-51617-3},
  url = {https://doi.org/10.1038/s41598-024-51617-3},
}
""",
    "publication_doi": "10.1038/s41598-024-51617-3",
    "code_repo": "https://github.com/d-r-deo/bimanualBCI",
    "licence": "CC0-1.0",
    "notes": "13 experiment days (2021-06-02..2021-10-13), one .mat per day, ~135 MB total. "
    "Each day is a whole-day concatenated stream of 20 ms / 50 Hz binned threshold crossings "
    "(uint8, -3.5xRMS) with recording gaps at block changes (blockNum marks the active block "
    "per bin). Threshold crossings only, no spike-band power. Two 96-channel Utah arrays (192 "
    "channels) in hand-knob motor cortex; the source does not label which 96 channels belong to "
    "which array, so the 0-95 / 96-191 split follows the willet_2021 convention for the same "
    "participant. moveType (1=UniR, 2=UniL, 3=Bi) is per-trial in the file; open-loop (OL) vs "
    "closed-loop (CL / CLR recalibration) is per-block and lives only in readme.txt, transcribed "
    "to utils.SESSION_BLOCKS. In closed-loop blocks the cursor positions are decoder output "
    "(neural-driven), not the autonomous instructed trajectory of open-loop blocks. goCue is a "
    "MATLAB 1-based index into the time axis. Same participant (T5) and lab (NPTL, Stanford) as "
    "willet_2021 and fan_2023. Clinical trial NCT00912041.",
    "subjects": [
        {
            "label": "T5",
            "participant_id": "braingate2/T5",  # datasets/participants.py
            # Age and AIS grade from Deo 2024 Methods. fan_2023 also says P69Y but for a 2022
            # first session, so the two agree only if T5's birthday is in late May.
            "age": "P69Y",
            "time_since_implant": "P1750D",  # first session 2021-06-02, implant 2016-08-17
            "clinical_condition": [
                "Spinal cord injury (C4 AIS-C)",
                "Tetraplegia",
            ],
            # Two 96-channel Utah arrays (192 electrodes) in the hand knob of left motor cortex.
            # Deo 2024 does not publish the per-array channel split; names/ranges mirror
            # willet_2021 (same participant, same NPTL implant). No stereotaxic coordinates.
            "arrays": [
                {
                    "name": "lateral_array",
                    "hemisphere": "left",
                    "device_type": "utah_array",
                    "n_channels": 96,
                    "channel_start": 0,
                    "channel_stop": 95,
                    "brain_area": "hand knob, motor cortex (left precentral gyrus)",
                },
                {
                    "name": "medial_array",
                    "hemisphere": "left",
                    "device_type": "utah_array",
                    "n_channels": 96,
                    "channel_start": 96,
                    "channel_stop": 191,
                    "brain_area": "hand knob, motor cortex (left precentral gyrus)",
                },
            ],
            # Region ontology (subject-owned BrainArea), keyed by each array's brain_area.
            # Same hand-knob / precentral-gyrus mapping as willet_2021's T5.
            "brain_areas": [
                {
                    "raw_label": "hand knob, motor cortex (left precentral gyrus)",
                    "homba_id": "HOMBA_10169",
                    "uberon_id": "UBERON_0016634",
                    "brodmann": "6",  # Willett et al. 2020 (Cell) places T5's arrays in area 6, not 4
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
            "dtype": "uint8",
            "description": "Binned threshold-crossing counts (-3.5xRMS), 192 channels, 20 ms "
            "bins. Whole-day stream (tx, (192, T) in source); recording gaps occur at block "
            "changes. Raw counts, not z-scored.",
        },
        {
            "name": "cursor_position",
            "category": "data",
            "event_class": "CursorPositionSeries",
            "modality": "behavior",
            "signal_type": "kinematics",
            "bin_size_ms": 20.0,
            "rate_hz": 50.0,
            "dtype": "float64",
            "description": "Binned (x, y) cursor positions (Cp). The Study splits the two hands "
            "into separate per-hand CursorPositionSeries (right = Cp rows 0-1, left = rows 2-3), "
            "each tagged with an 'effector' attribute. In open-loop blocks these are the autonomous "
            "instructed trajectory; in closed-loop blocks they are decoder output.",
        },
        {
            "name": "target_position",
            "category": "data",
            "event_class": "TargetPositionSeries",
            "modality": "behavior",
            "signal_type": "kinematics",
            "bin_size_ms": 20.0,
            "rate_hz": 50.0,
            "dtype": "float64",
            "description": "Binned (x, y) target positions (Tp), split per hand into right/left "
            "TargetPositionSeries in the same order as cursor_position.",
        },
        {
            "name": "movement_type",
            "category": "categorical",
            "event_class": "Stimulus",
            "task_condition": "attempted",
            "description": "Per-trial cued movement type: unimanual right (UniR), unimanual "
            "left (UniL), or bimanual (Bi). One event per trial over the delay-to-move span.",
        },
        {
            "name": "movement_epoch",
            "category": "categorical",
            "event_class": "Action",
            "task_condition": "attempted",
            "description": "Per-trial motor epochs from the trial timing (durationBinnedDelay "
            "preparatory period before the GO cue, durationBinnedMove move period after it). "
            "Marks the delay vs move phases around each goCue.",
        },
    ],
    "quality_flags": [
        {
            "kind": "recording_gaps",
            "description": "Neural activity between blocks was not recorded; the day's stream is "
            "concatenated across blocks and blockNum marks the active block per bin.",
        },
        {
            "kind": "loop_type_from_readme",
            "description": "Open-loop (OL) vs closed-loop (CL) vs closed-loop recalibration (CLR) "
            "is a per-block property present only in readme.txt, transcribed to utils.SESSION_BLOCKS; "
            "the .mat carries no loop-type field. Unlisted blocks resolve to 'unknown'.",
        },
        {
            "kind": "closed_loop_cursor",
            "description": "In closed-loop blocks the cursor positions are decoder output "
            "(neural-driven), not the autonomous instructed trajectory of open-loop blocks. The "
            "cursor_position stream means different things across loop types.",
        },
        {
            "kind": "matlab_1based_gocue",
            "description": "goCue is a MATLAB 1-based index into the time axis; accessors convert "
            "it to a 0-based bin index.",
        },
        {
            "kind": "array_split_assumed",
            "description": "The source does not label which of the 192 channels belong to which "
            "96-channel array; the 0-95 / 96-191 split follows the willet_2021 convention for the "
            "same participant.",
        },
    ],
}


#: Single paradigm across all 13 days.
SESSION_TASK = "delayed unimanual and bimanual cursor control"

#: T5 has very limited voluntary arm motion, so all task movement is attempted.
SESSION_TASK_CONDITION = "attempted"


def sessions() -> list[dict]:
    """Per-session rows from ``utils`` (requires the data drive mounted).

    One row per experiment day, carrying only fields the ``sessions`` table stores.
    ``post_implant_day`` reproduces the paper's own trial-day numbering (1750-1883), which
    is how every figure in the publication identifies its sessions.
    """
    from neuralshed.datasets import participants

    from . import utils

    rows = []
    for ref in utils.list_sessions():
        trials = utils.get_trials(utils.session_path(ref))
        date = utils.session_date(ref)
        rows.append(
            {
                "label": ref,
                "subject_label": utils.SUBJECT,
                "date": date,
                "post_implant_day": participants.post_implant_day(
                    "braingate2/T5", date
                ),
                "n_trials": int(len(trials)),
                "task": SESSION_TASK,
                "task_condition": SESSION_TASK_CONDITION,
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
