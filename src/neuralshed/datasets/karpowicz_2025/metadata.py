DATASET_METADATA = {
    "id": "karpowicz_2025",
    "title": "Stabilizing brain-computer interfaces through alignment of latent dynamics",
    "description": (
        "Closed-loop two-dimensional cursor control by BrainGate2 participant T11 using "
        "two 96-channel Utah arrays in the left precentral gyrus, on two sessions 153 and "
        "427 days after implant. Each session opens with a radial-8 calibration block and "
        "continues with Simon memory-matching blocks in which the participant imagines "
        "right-hand movements to drive the cursor through a sequence of four cardinal "
        "targets. The release holds 20 blocks and 1,116 movements at 50 Hz, as threshold "
        "crossings on 192 channels with cursor position, cursor velocity and target "
        "position."
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
        "Both NWB files carry a de-identified session_start_time whose seconds-resolution "
        "UTC offset dateutil cannot parse, so pynwb before 4.2.0 raises on read and "
        "utils.py goes through h5py. Timestamps are absolute Unix nanoseconds rather than "
        "seconds, and cursor_vel is a displacement per 20 ms bin rather than the "
        "position/s its unit string claims (a per-trial fit against diff(cursor_pos) gives "
        "0.99 and 1.01 for the two sessions), so utils.py scales it by the bin rate. "
        "cond_id is block-local: it numbers eight radial targets in a calibration block "
        "and four cardinal ones in a Simon block, so the target geometry travels with "
        "every trial. Only the task blocks are released, so a session file skips block "
        "numbers and holds gaps of up to 70 minutes; 251 individual 20 ms bins are missing "
        "from the grid (5 on day 153, 246 on day 427), 243 of them in day 427's calibration block. Four day-427 down "
        "movements carry the start time of the up movement before them, so their start "
        "is estimated as 80 ms after that movement ends. Out of scope: the "
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
            # The paper says only "cervical". Gusman et al. 2025 grades the same
            # participant's injury, which predates both sessions, as C4 AIS-B.
            "clinical_condition": [
                "Spinal cord injury (C4 AIS-B)",
                "Tetraplegia",
            ],
            # The deposit names no arrays. 0-95 is taken as the lateral array and 96-191
            # as the medial one, the names gusman_2025 uses for T11, by inference from
            # Rubin et al. 2022 on these sessions: the lateral array is sparser, and 39 of
            # the online decoder's 40 features came from the medial array. Here 0-95 fire
            # at a third of the rate of 96-191, and the released decoder output is
            # predicted from 96-191 only (held-out R2 0.58 and 0.37, against 0.005, 0.03).
            "arrays": [
                {
                    "name": "lateral_array",
                    "hemisphere": "left",
                    "device_type": "utah_array",
                    "n_channels": 96,
                    "channel_start": 0,
                    "channel_stop": 95,
                    "brain_area": "motor cortex (left precentral gyrus)",
                },
                {
                    "name": "medial_array",
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
                    "homba_id": "HOMBA_10169",  # area 6d by MRI/CT coregistration (Gusman et al. 2026, PNAS)
                    "uberon_id": "UBERON_0016634",  # premotor cortex
                    "brodmann": "6",
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
            "description": (
                "Threshold crossings on 192 electrodes aggregated in 20 ms bins. Values "
                "are integral counts stored as float64, up to 17 per bin."
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
            "description": (
                "Cursor x and y position in the normalized workspace, centre at the "
                "origin. The source names the unit 'position' and publishes no scale."
            ),
        },
        {
            "name": "target_position",
            "category": "data",
            "event_class": "TargetPositionSeries",
            "signal_type": "kinematics",
            "modality": "behavior",
            "rate_hz": 50.0,
            "dtype": "float64",
            "description": (
                "Cued target centre, at radius 0.375 on the cardinal axes in Simon "
                "blocks and 0.5 (0.35 diagonal) in the calibration block, where the "
                "centre target (0.0012, 0) is shown for each return."
            ),
        },
        {
            "name": "cursor_decoder_output",
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
                "Imagined right-hand movement driving the cursor to the cued target, from "
                "the movement's start to the row where its target is replaced (the trial "
                "table's window, with no 600 ms cap). Carries the target geometry and the "
                "outcome of its sequence, which every movement of the sequence shares."
            ),
        },
        {
            "name": "return_to_centre",
            "category": "categorical",
            "event_class": "Action",
            "task_condition": "imagined",
            "description": (
                "Calibration blocks only: the imagined movement back to the centre target "
                "(cond_id 0) after each outer target, 81 rows. code is 10 plus the "
                "cond_id of the outer target it leaves (11 to 18). The source gives no "
                "trial id or outcome for returns."
            ),
        },
        {
            "name": "sequence_cue",
            "category": "categorical",
            "event_class": "Stimulus",
            "task_condition": "na",
            "description": (
                "stimulus_kind='movement_cue': the presentation of a Simon sequence, the "
                "four targets shown one every 0.78 s while the cursor is held at the "
                "centre (288 rows, 16 per Simon block). code is the four targets' cond_id "
                "in the order shown (3142 is left, right, down, up)."
            ),
        },
        {
            "name": "intertrial",
            "category": "categorical",
            "event_class": "Intertrial",
            "description": (
                "Gap between movements: in Simon blocks the automatic recentring of the "
                "cursor, the wait before a sequence is shown, and the time after it "
                "before the cursor leaves the centre. In calibration blocks the row between "
                "an outward movement and its return. The block lead-in and tail carry no "
                "task row."
            ),
        },
    ],
}


def sessions() -> list[dict]:
    """One row per released session, derived from the downloaded NWB files."""
    from . import utils

    blocks = utils.list_blocks(utils.default_data_dir())
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
