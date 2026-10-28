DATASET_METADATA = {
    "id": "fan_2023",
    "title": "Fan et al. 2023 - One year of attempted handwriting by iBCI participant T5 (CORP)",
    "description": "Attempted handwriting of prompted English sentences by BrainGate2 "
    "participant T5 (C4 spinal cord injury) over 26 days in a year, from two 96-channel "
    "Utah arrays in the hand knob. 20 ms threshold-crossing counts per trial (go period) "
    "with the cued sentence: open-loop seed days to train a decoder, then closed-loop days "
    "with and without online recalibration (CORP).",
    "source_data_platform": "dryad",
    "source_id": "10.5061/dryad.hqbzkh1p6",
    "source_data_url": "https://doi.org/10.5061/dryad.hqbzkh1p6",
    "status": "ml_ready",
    "macro_task": "human_communication",
    "decode_target": "text",
    "study_class": "Fan2023Handwriting",  # datasets/fan_2023/study.py
    "bibtex": """
@inproceedings{fan2023plugandplay,
  title = {Plug-and-Play Stability for Intracortical Brain-Computer Interfaces: A One-Year Demonstration of Seamless Brain-to-Text Communication},
  author = {Fan, Chaofei and Hahn, Nick and Kamdar, Foram and Avansino, Donald and Wilson, Guy H. and Hochberg, Leigh and Shenoy, Krishna V. and Henderson, Jaimie M. and Willett, Francis R.},
  booktitle = {Advances in Neural Information Processing Systems},
  volume = {36},
  pages = {42258--42270},
  year = {2023},
  url = {https://proceedings.neurips.cc/paper_files/paper/2023/hash/83a14a36de4502bac5b580db36e81858-Abstract-Conference.html},
}
""",
    "publication_doi": None,  # NeurIPS 2023 issues no DOI, see bibtex
    "code_repo": "https://github.com/cffan/CORP",
    "licence": "CC0-1.0 (Dryad), CC-BY-4.0 (DANDI)",
    "notes": "26 days (May 2022 - Oct 2023), 1,330 trials, one timeline each. 20 ms bins / "
    "50 Hz, go period only, raw threshold-crossing counts (no spike-band power). 21 days from "
    "Dryad: 11 open-loop seed days (739 trials) and 10 closed-loop days with recalibration "
    "(377) and no-recalibration (199) blocks. 5 later days from DANDI 000950 (FALCON H2 "
    "held-out-calib, 3 trials each). FALCON's held-in files are byte copies of the first 60% "
    "of each Dryad file and are not read. original_split carries FALCON's labels "
    "(671 held-in-calib / 445 held-in-eval / 15 held-out-calib, empty on no-recalibration "
    "blocks). corp_split carries the authors' tfrecord partition (the last 10 trials of each "
    "seed day are test). 2023-02-28, 2023-08-16 and 2023-10-09 are flagged valid=False for shifted "
    "firing rates. The tfrecords, pretrained seed model and language model are not read.",
    "subjects": [
        {
            "label": "T5",
            "participant_id": "braingate2/T5",  # datasets/participants.py
            "age": "P69Y",
            "time_since_implant": "P2100D",  # first session 2022-05-18
            "clinical_condition": [
                "Spinal cord injury (C4 AIS-C)",
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
                    "brain_area": "hand knob, motor cortex (left precentral gyrus)",
                },
                {
                    "name": "array_2",
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
                    "homba_id": "HOMBA_10169",  # laterodorsal division of area 6 (A6ld); dorsal premotor
                    "uberon_id": "UBERON_0016634",  # premotor cortex (no dorsal subdivision in UBERON)
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
            "dtype": "float32",
            "description": "Threshold-crossing counts at -4.5 x SD, 192 channels = 2 x 96-ch Utah arrays, 20 ms bins, raw counts, go period of each trial (Dryad tx_feats, FALCON binned_spikes). Read as (192, n_bins) by NeuralFeatureExtractor.",
        },
        {
            "name": "sentence_cue",
            "category": "text",
            "event_class": "Sentence",
            "task_condition": "attempted",
            "description": "Prompted sentence the participant attempted to handwrite, one per trial, over the go period. `text` is decoded to plain English (raw `cue` encodes space as `>` and period as `~`). On closed-loop Dryad days it also carries the online decoder outputs (`rnn_decoded`, `lm_decoded`). Sentence-level only (no character-level timing).",
        },
    ],
}


def sessions() -> list[dict]:
    """One row per recording day, from the local download (needs the data drive)."""
    from neuralshed.datasets import participants

    from . import utils

    root = utils.default_root()
    days: dict[str, int] = {}
    for src in utils.list_sources(root):
        days[src["date"]] = days.get(src["date"], 0) + utils.count_trials(
            root / src["path"]
        )
    return [
        {
            "label": utils.session_label(date),
            "subject_label": utils.SUBJECT,
            "date": date,
            "post_implant_day": participants.post_implant_day("braingate2/T5", date),
            "n_trials": n_trials,
            "original_split": "held-in" if date <= "2023-02-28" else "held-out",
            "task": "attempted_handwriting",
            "task_condition": "attempted",
        }
        for date, n_trials in sorted(days.items())
    ]


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
