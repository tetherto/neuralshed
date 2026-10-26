"""Catalog metadata for Singer-Clark et al. 2025."""

import datetime as dt

DATASET_METADATA = {
    "id": "singer_2025",
    "title": "Speech motor cortex enables BCI cursor control and click",
    "description": (
        "Two-dimensional cursor control and target selection by BrainGate2 participant "
        "T15 using motor imagery and four 64-channel Utah arrays in left ventral "
        "precentral gyrus. The release contains 100 Hz threshold crossings, spike-band "
        "power, cursor and target trajectories, online decoder outputs, and task timing."
    ),
    "source_data_platform": "dryad",
    "source_id": "doi:10.5061/dryad.prr4xgxzq",
    "source_data_url": "https://doi.org/10.5061/dryad.prr4xgxzq",
    "status": "ml_ready",
    "macro_task": "human_motor",
    "decode_target": "movement",
    "study_class": "Singer2025CursorControl",
    "bibtex": """
@article{singerclark2025cursor,
  title = {Speech motor cortex enables BCI cursor control and click},
  author = {Singer-Clark, Tyler and Hou, Xianda and Card, Nicholas S. and Wairagkar, Maitreyee and Iacobacci, Carrina and Peracha, Hamza and Hochberg, Leigh R. and Stavisky, Sergey D. and Brandman, David M.},
  journal = {Journal of Neural Engineering},
  volume = {22},
  number = {3},
  pages = {036015},
  year = {2025},
  month = may,
  doi = {10.1088/1741-2552/add0e5},
  url = {https://doi.org/10.1088/1741-2552/add0e5},
}
""",
    "publication_doi": "10.1088/1741-2552/add0e5",
    "code_repo": "https://github.com/Neuroprosthetics-Lab/BCI-cursor-control-from-speech-motor-cortex",
    "licence": "CC0-1.0",
    "notes": (
        "The paper reports six sessions from post-implant days 39 to 468. Dryad version "
        "5 releases 51 cursor-task blocks from days 39, 202, and 468 plus one speech-only "
        "comparison file: 2,521 cursor trials and 2.52 hours across the released blocks."
    ),
    "subjects": [
        {
            "label": "T15",
            "participant_id": "braingate2/T15",
            "age": "P45Y",
            "time_since_implant": "P39D",
            "clinical_condition": [
                "Amyotrophic lateral sclerosis (ALS)",
                "Tetraplegia",
                "Severe dysarthria",
            ],
            "arrays": [
                {
                    "name": "ventral_6v",
                    "hemisphere": "left",
                    "device_type": "utah_array",
                    "n_channels": 64,
                    "channel_start": 0,
                    "channel_stop": 63,
                    "brain_area": "ventral 6v, left precentral gyrus",
                },
                {
                    "name": "4",
                    "hemisphere": "left",
                    "device_type": "utah_array",
                    "n_channels": 64,
                    "channel_start": 64,
                    "channel_stop": 127,
                    "brain_area": "area 4, left precentral gyrus",
                },
                {
                    "name": "55b",
                    "hemisphere": "left",
                    "device_type": "utah_array",
                    "n_channels": 64,
                    "channel_start": 128,
                    "channel_stop": 191,
                    "brain_area": "area 55b, left precentral gyrus",
                },
                {
                    "name": "dorsal_6v",
                    "hemisphere": "left",
                    "device_type": "utah_array",
                    "n_channels": 64,
                    "channel_start": 192,
                    "channel_stop": 255,
                    "brain_area": "dorsal 6v, left precentral gyrus",
                },
            ],
            "brain_areas": [
                {
                    "raw_label": "ventral 6v, left precentral gyrus",
                    "homba_id": "HOMBA_10170",
                    "uberon_id": "UBERON_0016634",
                    "brodmann": "6",
                },
                {
                    "raw_label": "dorsal 6v, left precentral gyrus",
                    "homba_id": "HOMBA_10170",
                    "uberon_id": "UBERON_0016634",
                    "brodmann": "6",
                },
                {
                    "raw_label": "area 4, left precentral gyrus",
                    "homba_id": "HOMBA_10163",
                    "uberon_id": "UBERON_0001384",
                    "brodmann": "4",
                },
                {
                    "raw_label": "area 55b, left precentral gyrus",
                    "homba_id": "HOMBA_AA30011",
                    "uberon_id": "UBERON_0016634",
                    "brodmann": None,  # HCP parcel with no Brodmann equivalent
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
            "bin_size_ms": 10.0,
            "rate_hz": 100.0,
            "dtype": "int16",
            "description": "Counts below -4.5 times RMS, 256 electrodes, summed in 10 ms bins.",
        },
        {
            "name": "spike_band_power",
            "category": "data",
            "event_class": "BandPower",
            "signal_type": "spike_band_power",
            "modality": "intracortical_microelectrode",
            "bin_size_ms": 10.0,
            "rate_hz": 100.0,
            "dtype": "float32",
            "description": "Mean squared 250 to 5000 Hz signal in each 10 ms bin, 256 electrodes.",
        },
        {
            "name": "cursor_position",
            "category": "data",
            "event_class": "CursorPositionSeries",
            "signal_type": "kinematics",
            "modality": "behavior",
            "rate_hz": 100.0,
            "dtype": "float32",
            "description": "Observed cursor x and y position in screen-height units.",
        },
        {
            "name": "target_position",
            "category": "data",
            "event_class": "TargetPositionSeries",
            "signal_type": "kinematics",
            "modality": "behavior",
            "rate_hz": 100.0,
            "dtype": "float32",
            "description": "Cued target-center x and y position in screen-height units.",
        },
        {
            "name": "cursor_decoder_output",
            "category": "data",
            "event_class": "DecodedCursorVelocitySeries",
            "signal_type": "kinematics",
            "modality": "derived",
            "rate_hz": 100.0,
            "dtype": "float32",
            "description": "Online neural decoder's two-dimensional cursor velocity output.",
        },
        {
            "name": "assist_amount",
            "category": "data",
            "event_class": "CursorAssistSeries",
            "signal_type": "other",
            "modality": "derived",
            "rate_hz": 100.0,
            "dtype": "float64",
            "description": "Computer contribution to velocity, 1 for open-loop and 0 for closed-loop.",
        },
        {
            "name": "click_assist",
            "category": "data",
            "event_class": "ClickAssistSeries",
            "signal_type": "other",
            "modality": "derived",
            "rate_hz": 100.0,
            "dtype": "float32",
            "description": "Whether target selection was computer-assisted, when click was enabled. Source is uint8 0/1, read as float32.",
        },
        {
            "name": "click_decoder_output",
            "category": "data",
            "event_class": "ClickStateSeries",
            "signal_type": "other",
            "modality": "derived",
            "rate_hz": 100.0,
            "dtype": "float32",
            "description": "Binary output of the online neural click decoder. Source is uint8 0/1, read as float32.",
        },
        {
            "name": "target_cue",
            "category": "categorical",
            "event_class": "Stimulus",
            "task_condition": "imagined",
            "description": (
                "Visual presentation of the target to acquire. Radial8 and speech blocks "
                "code it 0-7 by 45-degree spoke and 8 for center; grid blocks code it by "
                "row-major grid cell index."
            ),
        },
        {
            "name": "cursor_control",
            "category": "categorical",
            "event_class": "Action",
            "task_condition": "imagined",
            "description": "Motor imagery used to move the cursor toward the cued target.",
        },
        {
            "name": "speech_prompt",
            "category": "text",
            "event_class": "Word",
            "task_condition": "read",
            "description": "Word shown and played before movement in simultaneous speech blocks.",
        },
        {
            "name": "speech_go_cue",
            "category": "categorical",
            "event_class": "Stimulus",
            "task_condition": "attempted",
            "description": "Auditory beep instructing attempted speech on selected trials.",
        },
        {
            "name": "speech_attempt",
            "category": "categorical",
            "event_class": "Action",
            "task_condition": "attempted",
            "description": "Attempted prompted word after the beep in verbal blocks.",
        },
        {
            "name": "intertrial",
            "category": "categorical",
            "event_class": "Intertrial",
            "task_condition": "na",
            "description": "One-second gap between simultaneous speech and cursor trials. Emitted only for that task, which is the only one with a separate target-presentation index.",
        },
    ],
}


def sessions() -> list[dict]:
    """One row per released post-implant day, derived from downloaded block files."""
    from neuralshed.datasets import participants

    from . import utils

    rows = []
    for label, blocks in utils.session_blocks().items():
        day = blocks[0].day
        date = participants.implant_date("braingate2/T15")
        session_date = (date + dt.timedelta(days=day)).isoformat() if date else None
        tasks = sorted({block.task_label for block in blocks})
        rows.append(
            {
                "label": label,
                "subject_label": "T15",
                "date": session_date,
                "post_implant_day": day,
                "n_trials": sum(len(utils.trial_windows(block)) for block in blocks),
                "task": ", ".join(tasks),
            }
        )
    return rows


def get_metadata() -> dict:
    meta = DATASET_METADATA.copy()
    try:
        meta["sessions"] = sessions()
    except Exception:
        meta["sessions"] = []
    return meta


__all__ = ["DATASET_METADATA", "sessions", "get_metadata"]
