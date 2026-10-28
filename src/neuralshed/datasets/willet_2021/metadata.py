DATASET_METADATA = {
    "id": "willet_2021",
    "title": "High-performance brain-to-text communication via handwriting",
    "description": "Attempted handwriting of characters, straight lines, and full "
    "sentences by BrainGate2 participant T5 (instructed-delay paradigm).",
    "source_data_platform": "dryad",
    "source_id": "doi:10.5061/dryad.wh70rxwmv",
    "source_data_url": "https://datadryad.org/dataset/doi:10.5061/dryad.wh70rxwmv",
    "status": "ml_ready",
    "macro_task": "human_communication",
    "decode_target": "text",
    "study_class": "Willett2021Handwriting",  # datasets/willet_2021/study.py
    "bibtex": """
@article{willett2021handwriting,
  title = {High-performance brain-to-text communication via handwriting},
  author = {Willett, Francis R. and Avansino, Donald T. and Hochberg, Leigh R. and Henderson, Jaimie M. and Shenoy, Krishna V.},
  journal = {Nature},
  volume = {593},
  number = {7858},
  pages = {249–254},
  year = {2021},
  month = may,
  doi = {10.1038/s41586-021-03506-2},
  url = {https://doi.org/10.1038/s41586-021-03506-2},
}
""",
    "publication_doi": "10.1038/s41586-021-03506-2",
    "code_repo": "https://github.com/fwillett/handwritingBCI",
    "licence": "CC0-1.0",
    "notes": "10 ms bins / 100 Hz. Threshold-crossing counts only (no spike-band power). "
    "One timeline per recording block (130 blocks in 10 singleLetters, 10 sentences and 1 "
    "straightLines files). RNNTrainingSteps (except the train/test partition), BigramLM, "
    "computerMouseTemplates.mat and the rnn_* decoder outputs are out of scope.",
    "subjects": [
        {
            "label": "T5",
            "participant_id": "braingate2/T5",  # datasets/participants.py
            # Supplementary Methods 1.1: "65 years old at the time of data collection"
            "age": "P65Y",
            "time_since_implant": "P994D",
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
            "bin_size_ms": 10.0,
            "rate_hz": 100.0,
            "dtype": "uint8",
            "description": "Binned threshold-crossing counts (-3.5 x RMS), 192 channels. "
            "Emitted as one SpikeCounts stream over the whole block per timeline.",
        },
        {
            "name": "character",
            "category": "text",
            "event_class": "Character",
            "task_condition": "attempted",
            "description": "Attempted handwritten character over the go period, to the next "
            "trial's delay onset (1.0 s), singleLetters. 31 classes, `code` 1-31 in "
            "utils.CHAR_LIST order, `text` the glyph (a-z, space, comma, apostrophe, period, "
            "question mark), `char_class` the source label.",
        },
        {
            "name": "sentence",
            "category": "text",
            "event_class": "Sentence",
            "task_condition": "attempted",
            "description": "Attempted handwritten sentence from the go cue to the head turn "
            "that ended it (999 rows, intendedText). `original_split` is the HeldOutBlocks "
            "partition. The 32 sentences the source excludes carry valid=False. "
            "`sentence_condition` and `loop_type` ride the timeline.",
        },
        {
            "name": "straight_line",
            "category": "categorical",
            "event_class": "Action",
            "task_condition": "attempted",
            "description": "Attempted straight-line stroke over the go period, to the next "
            "trial's delay onset (1.5 s), straightLines, one session (action_kind "
            "'drawing'). 48 conditions, `code` 32-79 as 32 + 16 x distance + angle - 1 "
            "(distance short, medium, long; angle 1-16).",
        },
        {
            "name": "rest",
            "category": "categorical",
            "event_class": "Action",
            "task_condition": "na",
            "description": "The doNothing condition of the pilot session t5.2019.05.08 over "
            "the go period (action_kind 'rest', code 0, 27 rows).",
        },
        {
            "name": "cue_stimulus",
            "category": "categorical",
            "event_class": "Stimulus",
            "task_condition": "na",
            "description": "On-screen cue of each trial over the delay (stimulus_kind "
            "character_cue, movement_cue, rest_cue or sentence_cue). Character, line and rest "
            "cues carry the unit's code, sentence cues the prompt as `description`.",
        },
        {
            "name": "cue_go",
            "category": "categorical",
            "event_class": "Stimulus",
            "task_condition": "na",
            "description": "One-bin go marker of each trial (stimulus_kind movement_go_cue).",
        },
    ],
}


def sessions() -> list[dict]:
    """Per-session rows from ``utils`` (requires the data drive mounted)."""
    import numpy as np

    from . import utils

    rows = []
    for label in utils.list_sessions():
        tasks = utils.list_task_files(label)
        n_trials = 0
        for task, path in tasks.items():
            go_key = utils.TRIAL_VARIABLES[task][0]
            go_bins = utils.load_variables(path, [go_key])[go_key]
            n_trials += len(np.atleast_1d(go_bins))
        rows.append(
            {
                "label": label,
                "subject_label": "T5",
                "date": utils.session_date(label),
                "post_implant_day": utils.post_implant_day(label),
                "n_trials": n_trials,
                "task": ", ".join(tasks),
                "task_condition": "attempted",
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
