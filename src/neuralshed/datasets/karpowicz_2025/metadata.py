DATASET_METADATA = {
    "id": "karpowicz_2025",
    "title": "Stabilizing brain-computer interfaces through alignment of latent dynamics",
    "description": (
        "Closed-loop two-dimensional cursor control by BrainGate2 participant T11 using "
        "two 96-channel Utah arrays in the left precentral gyrus, on two sessions 153 and "
        "427 days after implant. Each session opens with a radial-8 calibration block and "
        "continues with Simon memory-matching blocks in which the participant imagines "
        "right-hand movements to drive the cursor through a sequence of four cardinal "
        "targets. The release holds 20 blocks and 1,116 movements at 50 Hz (1,112 of them "
        "emitted), as threshold crossings on 192 channels with cursor position, cursor "
        "velocity and target position."
    ),
    "source_data_platform": "dryad",
    "source_id": "doi:10.5061/dryad.q83bk3jtp",
    "source_data_url": "https://doi.org/10.5061/dryad.q83bk3jtp",
    "status": "ml_ready",
    "macro_task": "human_motor",
    "decode_target": "movement",
    "study_class": "Karpowicz2025CursorStability",
    "bibtex": """
@article{karpowicz2025stabilizing,
  title = {Stabilizing brain-computer interfaces through alignment of latent dynamics},
  author = {Karpowicz, Brianna M. and Ali, Yahia H. and Wimalasena, Lahiru N. and Sedler, Andrew R. and Keshtkaran, Mohammad Reza and Bodkin, Kevin and Ma, Xuan and Rubin, Daniel B. and Williams, Ziv M. and Cash, Sydney S. and Hochberg, Leigh R. and Miller, Lee E. and Pandarinath, Chethan},
  journal = {Nature Communications},
  volume = {16},
  pages = {4662},
  year = {2025},
  month = may,
  doi = {10.1038/s41467-025-59652-y},
  url = {https://doi.org/10.1038/s41467-025-59652-y},
}
""",
    "publication_doi": "10.1038/s41467-025-59652-y",
    "code_repo": "https://github.com/snel-repo/nomad",
    "licence": "CC0-1.0",
    "notes": (
        "Both NWB files carry a de-identified session_start_time of 1900-10-01 whose "
        "seconds-resolution UTC offset dateutil cannot parse, so pynwb raises on read and "
        "utils.py goes through h5py. Timestamps are absolute Unix nanoseconds rather than "
        "seconds, and cursor_vel is a displacement per 20 ms bin rather than the "
        "position/s its unit string claims (a per-trial fit against diff(cursor_pos) gives "
        "0.99 and 1.01 for the two sessions), so utils.py scales it by the bin rate. "
        "cond_id is block-local: it numbers eight radial targets in a calibration block "
        "and four cardinal ones in a Simon block, so the target geometry travels with "
        "every trial. Only the task blocks are released, so a session file skips block "
        "numbers and holds gaps of up to 70 minutes; 246 individual 20 ms bins are missing "
        "from the grid, 243 of them in day 427's calibration block. Four day-427 trials share "
        "a start time with their predecessor, an onset-detection artifact, and are dropped "
        "so trigger times stay unique. Out of scope: the "
        "monkey isometric-force and reaching datasets analyzed in the same paper, which "
        "are a separate Dryad deposit (10.5061/dryad.cvdncjt7n)."
    ),
    "subjects": [
        {
            "label": "T11",
            "participant_id": "braingate2/T11",  # datasets/participants.py
            "age": "P36Y",
            # The paper gives 153 and 427 days since implant for the two sessions.
            "time_since_implant": "P153D",
            "clinical_condition": [
                "Spinal cord injury (cervical)",
                "Tetraplegia",
            ],
            "arrays": [
                {
                    "name": "array_1",
                    "hemisphere": "left",
                    "device_type": "utah_array",
                    "n_channels": 96,
                    "channel_start": 0,
                    "channel_stop": 95,
                    "brain_area": "motor cortex (left precentral gyrus)",
                },
                {
                    "name": "array_2",
                    "hemisphere": "left",
                    "device_type": "utah_array",
                    "n_channels": 96,
                    "channel_start": 96,
                    "channel_stop": 191,
                    "brain_area": "motor cortex (left precentral gyrus)",
                },
            ],
            "brain_areas": [
                {
                    "raw_label": "motor cortex (left precentral gyrus)",
                    "homba_id": "HOMBA_10165",
                    "uberon_id": "UBERON_0001384",
                    "brodmann": "4",
                },
            ],
        }
    ],
    "events": [
        {
            "name": "binned_spikes",
            "category": "data",
            "event_class": "SpikeCounts",
            "signal_type": "threshold_crossings",
            "modality": "intracortical_microelectrode",
            "bin_size_ms": 20.0,
            "rate_hz": 50.0,
            "dtype": "float64",
            "description": (
                "Threshold crossings on 192 electrodes aggregated in 20 ms bins. Values "
                "are integral counts stored as float64, up to 15 per bin."
            ),
        },
        {
            "name": "cursor_pos",
            "category": "data",
            "event_class": "CursorPositionSeries",
            "signal_type": "kinematics",
            "modality": "behavior",
            "rate_hz": 50.0,
            "dtype": "float64",
            "description": (
                "Cursor x and y position in the normalized workspace, centre at the "
                "origin. The source names the unit 'position' and publishes no scale."
            ),
        },
        {
            "name": "target_pos",
            "category": "data",
            "event_class": "TargetPositionSeries",
            "signal_type": "kinematics",
            "modality": "behavior",
            "rate_hz": 50.0,
            "dtype": "float64",
            "description": (
                "Cued target centre, at radius 0.375 on the cardinal axes in Simon "
                "blocks and 0.5 (0.35 diagonal) in the calibration block."
            ),
        },
        {
            "name": "cursor_vel",
            "category": "data",
            "event_class": "DecodedCursorVelocitySeries",
            "signal_type": "kinematics",
            "modality": "derived",
            "rate_hz": 50.0,
            "dtype": "float64",
            "description": (
                "Cursor velocity driving the closed-loop cursor, released as a "
                "displacement per 20 ms bin and rescaled to per second on read."
            ),
        },
        {
            "name": "cursor_control",
            "category": "categorical",
            "event_class": "Action",
            "task_condition": "imagined",
            "description": (
                "Imagined right-hand movement driving the cursor to the cued target, cut "
                "from the outward zero-crossing to 600 ms later or to target acquisition. "
                "Carries the target geometry and whether its four-target sequence was "
                "completed successfully."
            ),
        },
        {
            "name": "intertrial",
            "category": "categorical",
            "event_class": "Intertrial",
            "description": (
                "Gap between movements, covering the automatic recentring of the cursor "
                "and the cue for the next target."
            ),
        },
    ],
}


def sessions() -> list[dict]:
    """One row per released session, derived from the downloaded NWB files."""
    from . import utils

    blocks = utils.list_blocks(utils.DEFAULT_DATA_DIR)
    rows = []
    for session in utils.SESSIONS:
        in_session = [ref for ref in blocks if ref.session == session]
        if not in_session:
            continue
        rows.append(
            {
                "label": session,
                "subject_label": utils.SUBJECT,
                "date": None,  # the deposit de-identifies it and the paper omits it
                "post_implant_day": utils.POST_IMPLANT_DAY[session],
                "n_trials": sum(len(utils.trials(ref)) for ref in in_session),
                "task": ", ".join(
                    sorted({utils.TASK_LABELS[ref.task] for ref in in_session})
                ),
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
