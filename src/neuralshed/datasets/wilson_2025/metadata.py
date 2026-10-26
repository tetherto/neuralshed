DATASET_METADATA = {
    "id": "wilson_2025",
    "title": (
        "Long-term unsupervised recalibration of cursor-based intracortical "
        "brain-computer interfaces using a hidden Markov model"
    ),
    "description": (
        "Closed-loop two-dimensional cursor control by BrainGate2 participant T5 using "
        "two 96-channel Utah arrays in the hand knob of left dorsal motor cortex. The "
        "release holds 103 recording days spanning 2016 to 2021 (560 task blocks, 38.2 "
        "hours, 68,327 trials) of radial-8, grid, and Fitts-style target acquisition, at "
        "50 Hz threshold crossings with cursor and target trajectories and the online "
        "cursor and click decoder outputs."
    ),
    "source_data_platform": "dryad",
    "source_id": "doi:10.5061/dryad.1jwstqk6g",
    "source_data_url": "https://doi.org/10.5061/dryad.1jwstqk6g",
    "status": "ml_ready",
    "macro_task": "human_motor",
    "decode_target": "movement",
    "study_class": "Wilson2025CursorRecalibration",
    "bibtex": """
@article{wilson2025recalibration,
  title = {Long-term unsupervised recalibration of cursor-based intracortical brain-computer interfaces using a hidden Markov model},
  author = {Wilson, Guy H. and Stein, Elias A. and Kamdar, Foram and Avansino, Donald T. and Pun, Tsam Kiu and Gross, Ronnie and Hosman, Tommy and Singer-Clark, Tyler and Kapitonava, Anastasia and Hochberg, Leigh R. and Simeral, John D. and Shenoy, Krishna V. and Druckmann, Shaul and Henderson, Jaimie M. and Willett, Francis R.},
  journal = {Nature Biomedical Engineering},
  volume = {10},
  pages = {1466--1484},
  year = {2026},
  month = jul,
  doi = {10.1038/s41551-025-01536-z},
  url = {https://doi.org/10.1038/s41551-025-01536-z},
}
""",
    "publication_doi": "10.1038/s41551-025-01536-z",
    "code_repo": "https://github.com/guyhwilson/nonstationarities",
    "licence": "CC0-1.0",
    "notes": (
        "Spikes were downsampled to 1000 Hz then binned in non-overlapping 20 ms bins "
        "(50 Hz, 192 channels). The detection threshold is -3.5 times RMS in every "
        "released session header, not the -4.5 stated in the paper's Methods. "
        "t5.2017.08.04 shipped with the two pedestal cables swapped, so channels 0-95 and "
        "96-191 are exchanged and utils.py undoes it. On t5.2019.01.14 blocks 24, 27, 28 "
        "and 29 each appear twice with byte-identical data, so only the first copy is "
        "exposed. Selecting days with at least two of the authors' analysis blocks "
        "reproduces the paper's 73-session cohort. Out of scope: the 2022 one_month_recal "
        "closed-loop blocks (a separate experiment with a different pickle schema and "
        "normalized screen coordinates), the T11 personal-use recordings from Fig. 6 "
        "(never released, possible PHI), and the simulator and method-sweep .npy outputs "
        "(analysis results, not recordings)."
    ),
    "subjects": [
        {
            "label": "T5",
            "participant_id": "braingate2/T5",  # datasets/participants.py
            "age": "P69Y",
            "time_since_implant": "P40D",  # first session 2016-09-26, implant 2016-08-17
            "clinical_condition": [
                "Spinal cord injury (C4 AIS-C)",
                "Tetraplegia",
            ],
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
            "dtype": "uint16",
            "description": (
                "Crossings below -3.5 times RMS, 192 electrodes, binned at 20 ms. The "
                "source scales each bin count to a firing rate, so values are exact "
                "multiples of 50 Hz and the unit is Hz rather than counts per bin."
            ),
        },
        {
            "name": "cursor_position",
            "category": "data",
            "event_class": "CursorPositionSeries",
            "signal_type": "kinematics",
            "modality": "behavior",
            "rate_hz": 50.0,
            "dtype": "float64",
            "description": "Observed cursor x and y position on the task screen.",
        },
        {
            "name": "target_position",
            "category": "data",
            "event_class": "TargetPositionSeries",
            "signal_type": "kinematics",
            "modality": "behavior",
            "rate_hz": 50.0,
            "dtype": "int16",
            "description": "Cued target-center x and y position on the task screen.",
        },
        {
            "name": "cursor_decoder_output",
            "category": "data",
            "event_class": "DecodedCursorVelocitySeries",
            "signal_type": "kinematics",
            "modality": "derived",
            "rate_hz": 50.0,
            "dtype": "float32",
            "description": (
                "Online linear decoder's two-dimensional cursor velocity output, which "
                "drove the cursor during closed-loop control."
            ),
        },
        {
            "name": "click_decoder_output",
            "category": "data",
            "event_class": "ClickStateSeries",
            "signal_type": "other",
            "modality": "derived",
            "rate_hz": 50.0,
            "dtype": "uint8",
            "description": (
                "Binary output of the online neural click decoder, emitted on the 195 "
                "blocks that ran one (source field 14)."
            ),
        },
        {
            "name": "target_cue",
            "category": "categorical",
            "event_class": "Stimulus",
            "task_condition": "attempted",
            "description": (
                "Visual presentation of the target to acquire, one per trial. Its code is "
                "the block-local index into that block's distinct target centres, since "
                "the layout varies (radial-8 plus centre, a grid of up to 55 cells, or "
                "randomly placed targets)."
            ),
        },
        {
            "name": "cursor_control",
            "category": "categorical",
            "event_class": "Action",
            "task_condition": "attempted",
            "description": (
                "Attempted hand movement, as though using a joystick, driving the cursor "
                "toward the cued target. Carries the trial's success or failure outcome."
            ),
        },
    ],
}


def sessions() -> list[dict]:
    """One row per released recording day, derived from the downloaded session files."""
    from . import utils

    rows = []
    for label, blocks in utils.session_blocks().items():
        date = utils.session_date(label)
        implant = _implant_date()
        tasks = sorted({block.task_label for block in blocks})
        rows.append(
            {
                "label": label,
                "subject_label": "T5",
                "date": date.isoformat() if date else None,
                "post_implant_day": (date - implant).days if date and implant else None,
                "n_trials": sum(len(utils.trial_windows(block)) for block in blocks),
                "task": ", ".join(tasks),
            }
        )
    return rows


def _implant_date():
    from neuralshed.datasets import participants

    return participants.implant_date("braingate2/T5")


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
