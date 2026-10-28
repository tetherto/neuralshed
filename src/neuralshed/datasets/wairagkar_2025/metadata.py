DATASET_METADATA = {
    "id": "wairagkar_2025",
    "title": "An instantaneous voice-synthesis neuroprosthesis",
    "description": "Brain-to-voice speech neuroprosthesis: intracortical neural "
    "activity from BrainGate2 participant T15 (ALS, severe dysarthria) is decoded "
    "into continuously synthesized voice during attempted, open-vocabulary cued speech. "
    "The full t15_neural_data release ships the pre-binned neural features, the "
    "time-aligned LPCNet acoustic target, word/syllable speech timings, and (for the "
    "evaluation sessions) the closed-loop synthesized voice and causal decoders.",
    "source_data_platform": "dryad",
    "source_id": "doi:10.5061/dryad.2280gb64f",
    "source_data_url": "https://datadryad.org/dataset/doi:10.5061/dryad.2280gb64f",
    "status": "ml_ready",
    "macro_task": "human_communication",
    "decode_target": "voice",
    "study_class": "Wairagkar2025Brain2Voice",  # datasets/wairagkar_2025/study.py
    "bibtex": """
@article{wairagkar2025voice,
  title = {An instantaneous voice-synthesis neuroprosthesis},
  author = {Wairagkar, Maitreyee and Card, Nicholas S. and Singer-Clark, Tyler and Hou, Xianda and Iacobacci, Carrina and Miller, Lee M. and Hochberg, Leigh R. and Brandman, David M. and Stavisky, Sergey D.},
  journal = {Nature},
  volume = {644},
  number = {8075},
  pages = {145–153},
  year = {2025},
  month = jun,
  doi = {10.1038/s41586-025-09127-3},
  url = {https://doi.org/10.1038/s41586-025-09127-3},
}
""",
    "publication_doi": "10.1038/s41586-025-09127-3",
    "code_repo": "https://github.com/Neuroprosthetics-Lab/brain-to-voice-2025",
    "licence": "CC0-1.0",  # Dryad data release (CC0)
    "notes": "Voice-synthesis paradigm (brain-to-voice attempted speech). 10 ms bins / "
    "100 Hz neural features. Working dataset = t15_neural_data: 37 sessions, 8,489 trials, "
    "8,235 unique sentences (open vocabulary, about 3,500 words), with the time-aligned "
    "20-dim LPCNet target in word- and syllable-aligned variants. 9 evaluation sessions add "
    "closed-loop synthesized audio + a causal decoder; test_idx.mat marks the trials whose "
    "synthesized voice was scored by human listeners. The smaller 50-word "
    "t15_control_experiment release (1 session, with stethoscope mic + IMU but no LPCNet "
    "target) is documented in README.md but is not the working dataset.",
    "subjects": [
        {
            "label": "T15",
            "participant_id": "braingate2/T15",  # datasets/participants.py
            "age": "P45Y",
            "time_since_implant": "P25D",
            "clinical_condition": [
                "Amyotrophic lateral sclerosis (ALS)",
                "Severe dysarthria (retained some orofacial movement and could "
                "vocalize but could not produce intelligible speech)",
            ],
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
            # Region ontology (subject-owned BrainArea), keyed by each array's brain_area.
            # Same T15 implant as card_2024, so array names, labels, ontology ids, Brodmann
            # areas and MNI centers (NEJM 2024 appendix S1.12) are identical to it: both 6v
            # arrays map to the lateroventral division of area 6 (HCP has no ventral/dorsal
            # 6v leaf in HOMBA), area 4 to primary motor cortex, 55b to the HCP 55b parcel.
            # UBERON has no premotor subdivision, so 6v/55b map up to premotor cortex.
            "brain_areas": [
                {
                    "raw_label": "ventral 6v, left precentral gyrus",
                    "homba_id": "HOMBA_10170",  # lateroventral division of area 6 (ventral premotor)
                    "uberon_id": "UBERON_0016634",  # premotor cortex (no ventral subdivision in UBERON)
                    "brodmann": "6",
                },
                {
                    "raw_label": "area 4, left precentral gyrus",
                    "homba_id": "HOMBA_10163",  # primary motor cortex (area 4), whole area
                    "uberon_id": "UBERON_0001384",  # primary motor cortex (Brodmann area 4)
                    "brodmann": "4",
                },
                {
                    "raw_label": "area 55b, left precentral gyrus",
                    "homba_id": "HOMBA_AA30011",  # area 55b (HCP), under laterodorsal division of area 6
                    "uberon_id": "UBERON_0016634",  # premotor cortex (UBERON has no 55b)
                    "brodmann": None,  # HCP parcel with no Brodmann equivalent
                },
                {
                    "raw_label": "dorsal 6v, left precentral gyrus",
                    "homba_id": "HOMBA_10170",  # same HCP 6v parcel, dorsal placement within it
                    "uberon_id": "UBERON_0016634",  # premotor cortex
                    "brodmann": "6",
                },
            ],
        }
    ],
    # Catalog of the events `study.py` emits per timeline (see `_load_timeline_events`):
    # 3 `data` streams (BandPower + SpikeCounts neural features, the AcousticFeatures target
    # emitted twice, word- and syllable-aligned) and 2 `text` events (the Sentence cue + per-
    # word Word). The closed-loop synthesized audio and causal decoders are the online
    # system's output, surfaced via `utils`, not emitted as events (see README.md, Notes /
    # Scope). Whole-trial labels (original_split, trial_description, task_condition,
    # benchmark) ride the timeline. `event_class` is the concrete neuroai Event subclass.
    "events": [
        {
            "name": "spike_band_power",
            "category": "data",
            "event_class": "BandPower",
            "signal_type": "spike_band_power",
            "modality": "intracortical_microelectrode",
            "bin_size_ms": 10.0,
            "rate_hz": 100.0,
            "dtype": "float32",
            "description": "Binned spike-band power in the 250-5000 Hz band, [N x 256] (4 Utah arrays x 64 ch, ordered ventral_6v, 4, 55b, dorsal_6v). Declared unit is the chain's input, the mean of squared band-passed samples (uV^2, clipped at 50,000 uV^2) after linear regression referencing; that is then log-transformed, normalized by rolling 10 s mean/SD, and causally smoothed with a 1.5 s sigmoid kernel, so the shipped values are dimensionless.",
        },
        {
            "name": "threshold_crossings",
            "category": "data",
            "event_class": "SpikeCounts",
            "signal_type": "threshold_crossings",
            "modality": "intracortical_microelectrode",
            "bin_size_ms": 10.0,
            "rate_hz": 100.0,
            "dtype": "float32",
            "description": "Binned threshold-crossing counts (-4.5 x RMS on the 250-5000 Hz band after linear regression referencing), [N x 256] (4 Utah arrays x 64 ch, ordered ventral_6v, 4, 55b, dorsal_6v). Declared unit is the chain's input, the raw per-bin crossing counts; those are then log-transformed, normalized by rolling 10 s mean/SD, and causally smoothed with a 1.5 s sigmoid kernel, so the shipped values are dimensionless.",
        },
        {
            "name": "lpcnet_acoustic_target",
            "category": "data",
            "event_class": "AcousticFeatures",
            "signal_type": "acoustic_features",
            "modality": "derived",
            "bin_size_ms": 10.0,
            "rate_hz": 100.0,
            "dtype": "float32",
            "description": "Synthetic LPCNet acoustic-feature target (18 Bark-frequency cepstral coefficients + 2 pitch/voicing), [N x 20] in the original LPCNet range, generated offline by text-to-speech of the cue and time-aligned to the neural bins. Emitted twice per trial, feature lpcnet_word_aligned and lpcnet_syllable_aligned; the paper's decoders trained on the syllable-aligned variant on every session but the first (t15_day025, word-aligned).",
        },
        {
            "name": "sentence",
            "category": "text",
            "event_class": "Sentence",
            "task_condition": "attempted",
            "description": "Cued sentence spanning the whole go period (open vocabulary); T15 attempts to speak it (mimed on 59 trials). The source's trial_description rides the timeline.",
        },
        {
            "name": "word",
            "category": "text",
            "event_class": "Word",
            "task_condition": "attempted",
            "description": "Per-word attempted-speech onset/offset (100 Hz speech_indices). Measured from a microphone on t15_day025 only; on every later session inferred from the envelope of the decoder's synthesized output. Syllable-level timings ship in the source but are not emitted.",
        },
    ],
}


def sessions() -> list[dict]:
    """Per-session rows from ``utils`` (requires the data drive mounted).

    ``date`` is derived as the participant's implant date + post_implant_day (the
    ``t15_dayNNN`` folder number is days since implant); the source ships no explicit
    session date.
    """
    import datetime as dt

    from neuralshed.datasets import participants

    from . import utils

    implant = participants.implant_date("braingate2/T15")
    rows = []
    for ref in utils.list_sessions():
        rows.append(
            {
                "label": ref.session,
                "subject_label": "T15",
                "date": (implant + dt.timedelta(days=ref.day)).isoformat(),
                "post_implant_day": ref.day,
                "n_trials": utils.n_trials(ref),
                "task": "brain-to-voice attempted speech",
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
