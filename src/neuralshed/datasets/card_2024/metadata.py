DATASET_METADATA = {
    "id": "card_2024",
    "title": "An accurate and rapidly calibrating speech neuroprosthesis",
    "description": "Attempted (vocalized or silent) speech of cued sentences by "
    "BrainGate2 participant T15 (ALS / severe dysarthria); copy-task only. Neural "
    "features are 20 ms-binned threshold crossings and spike-band power from four "
    "64-channel Utah arrays in left precentral (speech motor) cortex.",
    "source_data_platform": "dryad",
    "source_id": "doi:10.5061/dryad.dncjsxm85",
    "source_data_url": "https://doi.org/10.5061/dryad.dncjsxm85",
    "status": "ml_ready",
    "macro_task": "human_communication",
    "decode_target": "text",
    "study_class": "Card2024Brain2Text",  # datasets/card_2024/study.py
    "bibtex": """
@article{card2024speech,
  title = {An accurate and rapidly calibrating speech neuroprosthesis},
  author = {Card, Nicholas S. and Wairagkar, Maitreyee and Iacobacci, Carrina and Hou, Xianda and Singer-Clark, Tyler and Willett, Francis R. and Kunz, Erin M. and Fan, Chaofei and Vahdati Nia, Maryam and Deo, Darrel R. and Srinivasan, Aparna and Choi, Eun Young and Glasser, Matthew F. and Hochberg, Leigh R. and Henderson, Jaimie M. and Shahlaie, Kiarash and Stavisky, Sergey D. and Brandman, David M.},
  journal = {New England Journal of Medicine},
  volume = {391},
  number = {7},
  pages = {609–618},
  year = {2024},
  month = aug,
  doi = {10.1056/NEJMoa2314132},
  url = {https://doi.org/10.1056/NEJMoa2314132},
}
""",
    "publication_doi": "10.1056/NEJMoa2314132",
    "code_repo": "https://github.com/Neuroprosthetics-Lab/nejm-brain-to-text",
    "licence": "CC0-1.0",
    "notes": "45 sessions (Aug 2023 - Apr 2025), 10,948 copy-task trials (8,072 train / "
    "1,426 val / 1,450 test; 41/45 sessions have val+test). 20 ms bins / 50 Hz. 512 "
    "input_features = 256 threshold crossings (-4.5xRMS) + 256 spike-band power, z-scored "
    "within session from the preceding ~20 trials. test-split trials carry no labels "
    "(reserved for the Brain-to-Text '25 challenge). Copy task only; conversation-mode / "
    "personal-use data (t15_personalUse.pkl) excluded.",
    "subjects": [
        {
            "label": "T15",
            "participant_id": "braingate2/T15",  # datasets/participants.py
            "age": "P45Y",
            "time_since_implant": "P25D",
            "clinical_condition": [
                "Amyotrophic lateral sclerosis (ALS)",
                "Tetraparesis",
                "Severe dysarthria",
            ],
            # Array centers from the NEJM supplementary appendix S1.12 (MNI2009b asymmetric,
            # registration-dependent estimate).
            "arrays": [
                {
                    "name": "ventral_6v",
                    "hemisphere": "left",
                    "device_type": "utah_array",
                    "stereotaxic_coordinates": {
                        "space": "MNI152NLin2009bAsym",
                        "unit": "mm",
                        "x": -60.8,
                        "y": 0.2,
                        "z": 40.6,
                    },
                    "n_channels": 64,
                    "channel_start": 0,
                    "channel_stop": 63,
                    "brain_area": "ventral 6v, left precentral gyrus",
                },
                {
                    "name": "4",
                    "hemisphere": "left",
                    "device_type": "utah_array",
                    "stereotaxic_coordinates": {
                        "space": "MNI152NLin2009bAsym",
                        "unit": "mm",
                        "x": -63.6,
                        "y": 3.6,
                        "z": 29.3,
                    },
                    "n_channels": 64,
                    "channel_start": 64,
                    "channel_stop": 127,
                    "brain_area": "area 4, left precentral gyrus",
                },
                {
                    "name": "55b",
                    "hemisphere": "left",
                    "device_type": "utah_array",
                    "stereotaxic_coordinates": {
                        "space": "MNI152NLin2009bAsym",
                        "unit": "mm",
                        "x": -57.2,
                        "y": -1.1,
                        "z": 48.5,
                    },
                    "n_channels": 64,
                    "channel_start": 128,
                    "channel_stop": 191,
                    "brain_area": "area 55b, left precentral gyrus",
                },
                {
                    "name": "dorsal_6v",
                    "hemisphere": "left",
                    "device_type": "utah_array",
                    "stereotaxic_coordinates": {
                        "space": "MNI152NLin2009bAsym",
                        "unit": "mm",
                        "x": -64.5,
                        "y": 5.3,
                        "z": 21.6,
                    },
                    "n_channels": 64,
                    "channel_start": 192,
                    "channel_stop": 255,
                    "brain_area": "dorsal 6v, left precentral gyrus",
                },
            ],
            "brain_areas": [
                {
                    "raw_label": "ventral 6v, left precentral gyrus",
                    "homba_id": "HOMBA_10170",  # lateroventral division of area 6 (ventral premotor); no HCP 6v leaf in HOMBA
                    "uberon_id": "UBERON_0016634",  # premotor cortex (UBERON has no ventral subdivision)
                    "brodmann": "6",
                },
                {
                    "raw_label": "dorsal 6v, left precentral gyrus",
                    "homba_id": "HOMBA_10170",  # same HCP 6v parcel, dorsal placement within it
                    "uberon_id": "UBERON_0016634",  # premotor cortex
                    "brodmann": "6",
                },
                {
                    "raw_label": "area 4, left precentral gyrus",
                    "homba_id": "HOMBA_10163",  # primary motor cortex (area 4); whole M1, not a somatotopic subregion
                    "uberon_id": "UBERON_0001384",  # primary motor cortex (Brodmann area 4)
                    "brodmann": "4",
                },
                {
                    "raw_label": "area 55b, left precentral gyrus",
                    "homba_id": "HOMBA_AA30011",  # area 55b (HCP), placed under laterodorsal division of area 6
                    "uberon_id": "UBERON_0016634",  # premotor cortex (UBERON has no 55b)
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
            "bin_size_ms": 20.0,
            "rate_hz": 50.0,
            "dtype": "float32",
            "description": 'Binned threshold-crossing counts (-4.5xRMS), 256 channels (input_features cols 0-255), z-scored within session. Listed before spike_band_power so aggregation="cat" yields the source\'s native 512-d order.',
        },
        {
            "name": "spike_band_power",
            "category": "data",
            "event_class": "BandPower",
            "signal_type": "spike_band_power",
            "modality": "intracortical_microelectrode",
            "bin_size_ms": 20.0,
            "rate_hz": 50.0,
            "dtype": "float32",
            "description": "Spike-band power, 256 channels (input_features cols 256-511), z-scored within session.",
        },
        {
            "name": "sentence",
            "category": "text",
            "event_class": "Sentence",
            "task_condition": "attempted",
            "description": "Cued sentence over the whole trial: orthographic `text` + ARPAbet `phonemes` target (seq_class_ids: 1-39 phonemes, 40=word boundary '|'; see utils.PHONEME_LABELS). Phoneme target rides the Sentence (no per-phoneme timing). Train/val only; test labels held out for the Brain-to-Text '25 challenge.",
        },
    ],
}


def sessions() -> list[dict]:
    """Per-session rows from ``utils`` (requires the data drive mounted).

    One row per session, with ``n_trials`` summed across the available train/val/test
    splits. ``post_implant_day`` is derived from the participant's implant date.
    """
    from . import utils

    rows = []
    for ref in utils.list_sessions():
        rows.append(
            {
                "label": ref.session,
                "subject_label": "T15",
                "date": ref.date,
                "post_implant_day": ref.post_implant_day,
                "n_trials": sum(utils.n_trials(ref, s) for s in ref.splits),
                "task": "attempted speech",
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
