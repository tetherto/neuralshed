DATASET_METADATA = {
    "id": "willet_2023",
    "title": "A high-performance speech neuroprosthesis",
    "description": "Attempted speech (sentence production) by BrainGate2 participant "
    "T12, an ALS patient, recorded from four 64-channel microelectrode arrays in the "
    "left hemisphere, two in area 6v (ventral precentral gyrus) and two in area 44 "
    "(inferior frontal gyrus). Instructed-delay task: read a "
    "sentence, wait through a delay, then on the go cue attempt to speak it "
    "(vocalized or mouthed). Sentences only (no decoder/competition data).",
    "source_data_platform": "dryad",
    "source_id": "doi:10.5061/dryad.x69p8czpq",
    "source_data_url": "https://datadryad.org/dataset/doi:10.5061/dryad.x69p8czpq",
    "status": "ml_ready",
    "macro_task": "human_communication",
    "decode_target": "text",
    "study_class": "Willett2023Sentences",  # datasets/willet_2023/study.py
    "bibtex": """
@article{willett2023speech,
  title = {A high-performance speech neuroprosthesis},
  author = {Willett, Francis R. and Kunz, Erin M. and Fan, Chaofei and Avansino, Donald T. and Wilson, Guy H. and Choi, Eun Young and Kamdar, Foram and Glasser, Matthew F. and Hochberg, Leigh R. and Druckmann, Shaul and Shenoy, Krishna V. and Henderson, Jaimie M.},
  journal = {Nature},
  volume = {620},
  number = {7976},
  pages = {1031–1036},
  year = {2023},
  month = aug,
  doi = {10.1038/s41586-023-06377-x},
  url = {https://doi.org/10.1038/s41586-023-06377-x},
}
""",
    "publication_doi": "10.1038/s41586-023-06377-x",
    "code_repo": "https://github.com/fwillett/speechBCI",
    "licence": "CC0-1.0",
    "notes": "20 ms bins / 50 Hz, 256 channels (4 x 64-ch 1.5 mm SIROF Blackrock arrays "
    "implanted 2022-03-30). Spike-band power + threshold-crossing counts at 4 thresholds "
    "(tx1-tx4) + audio. Sentence production only; diagnosticBlocks / tuningTasks / "
    "competitionData are out of scope. "
    "The Study emits the neural rows in array order, one contiguous block of 64 per "
    "array (utils.CHANNEL_ORDER), not in the file order of the source .mat matrices; "
    "the channel ranges below are array-order columns.",
    "subjects": [
        {
            "label": "T12",
            "participant_id": "braingate2/T12",  # datasets/participants.py
            "age": "P67Y",  # supplement 1.1: 67 at the time of data collection
            "time_since_implant": "P29D",  # first session 2022-04-28; last is day 148
            "clinical_condition": [
                "Slowly progressive bulbar-onset amyotrophic lateral sclerosis (ALS), "
                "diagnosed at age 59, ALSFRS-R 26 at enrollment",
                "Severe dysarthria for nearly 8 years: vocalizes while attempting to "
                "speak with some differentiable vowels, no intelligible consonants or "
                "sentences",
                "Partial use of the limbs, communicates through a writing board or tablet",
            ],
            "arrays": [
                {
                    "name": "dorsal_6v",
                    "hemisphere": "left",
                    "device_type": "utah_array",
                    "n_channels": 64,
                    "channel_start": 0,
                    "channel_stop": 63,
                    "brain_area": "area 6v, left ventral precentral gyrus",
                },
                {
                    "name": "ventral_6v",
                    "hemisphere": "left",
                    "device_type": "utah_array",
                    "n_channels": 64,
                    "channel_start": 64,
                    "channel_stop": 127,
                    "brain_area": "area 6v, left ventral precentral gyrus",
                },
                {
                    "name": "dorsal_44",
                    "hemisphere": "left",
                    "device_type": "utah_array",
                    "n_channels": 64,
                    "channel_start": 128,
                    "channel_stop": 191,
                    "brain_area": "area 44, left inferior frontal gyrus (Broca's area)",
                },
                {
                    "name": "ventral_44",
                    "hemisphere": "left",
                    "device_type": "utah_array",
                    "n_channels": 64,
                    "channel_start": 192,
                    "channel_stop": 255,
                    "brain_area": "area 44, left inferior frontal gyrus (Broca's area)",
                },
            ],
            "brain_areas": [
                {
                    "raw_label": "area 6v, left ventral precentral gyrus",
                    "homba_id": "HOMBA_10170",
                    "uberon_id": "UBERON_0016634",
                },
                {
                    "raw_label": "area 44, left inferior frontal gyrus (Broca's area)",
                    "homba_id": "HOMBA_10186",
                    "uberon_id": "UBERON_0002980",
                },
            ],
        }
    ],
    "events": [
        {
            "name": "spike_band_power",
            "category": "data",
            "event_class": "BandPower",
            "signal_type": "spike_band_power",
            "modality": "intracortical_microelectrode",
            "bin_size_ms": 20.0,
            "rate_hz": 50.0,
            "dtype": "float32",
            "description": "Spike band power (`spikePow`), uV^2, 256 channels, emitted raw. Chain: non-causal 250 Hz high-pass (4 ms delay; supplement 1.4 says 300 Hz), linear regression referencing refit per session from its diagnostic block, mean of squared voltages per 20 ms bin, unclipped (outliers to 1e7 uV^2).",
        },
        {
            "name": "threshold_crossings",
            "category": "data",
            "event_class": "SpikeCounts",
            "signal_type": "threshold_crossings",
            "modality": "intracortical_microelectrode",
            "bin_size_ms": 20.0,
            "rate_hz": 50.0,
            "dtype": "uint8",
            "description": "Binned threshold-crossing counts (`tx2`, -4.5 x RMS), 256 channels, emitted raw. Same high-pass and referencing chain as spike_band_power, thresholds refit per session from its diagnostic block. The source also ships tx1 (-3.5), tx3 (-5.5) and tx4 (-6.5 x RMS), not emitted.",
        },
        # The three audio streams are source only: out of scope, not emitted by the Study.
        {
            "name": "audio",
            "category": "data",
            "event_class": None,
            "signal_type": "audio",
            "modality": "audio",
            "bin_size_ms": None,
            "rate_hz": 30000.0,
            "dtype": "int16",
            "description": "Source only, not emitted: raw microphone waveform at 30 kHz, one snippet per block (`audio`). Near silent on the four mimed days; in closed-loop blocks the microphone also records the computer reading the decoded text aloud.",
        },
        {
            "name": "audio_envelope",
            "category": "data",
            "event_class": None,
            "signal_type": "audio",
            "modality": "audio",
            "bin_size_ms": 20.0,
            "rate_hz": 50.0,
            "dtype": "float64",
            "description": "Source only, not emitted: audio volume envelope per 20 ms bin (`audioEnvelope`), aligned to the neural bins.",
        },
        {
            "name": "audio_features",
            "category": "data",
            "event_class": None,
            "signal_type": "acoustic_features",
            "modality": "audio",
            "bin_size_ms": 20.0,
            "rate_hz": 50.0,
            "dtype": "float64",
            "description": "Source only, not emitted: 42-dimensional MFCC features per 20 ms bin (`audioFeatures`), aligned to the neural bins.",
        },
        {
            "name": "sentence_cue",
            "category": "categorical",
            "event_class": "Stimulus",
            "task_condition": "na",
            "description": 'Sentence shown above a red square over the randomized delay period (`delayTrialEpochs`, trialState 0); `stimulus_kind="sentence_cue"`, `description` is the sentence.',
        },
        {
            "name": "speech_go_cue",
            "category": "categorical",
            "event_class": "Stimulus",
            "task_condition": "attempted",
            "description": 'Go cue, the square turns green (first bin of `goTrialEpochs`, trialState 1); `stimulus_kind="speech_go_cue"`, one 20 ms bin.',
        },
        {
            "name": "sentence_onset",
            "category": "text",
            "event_class": "Sentence",
            "task_condition": "attempted",
            "description": "Onset of attempted sentence production. Speaking modes: `attempted speaking` (vocalized) and `attempted nonvocal speaking` (mouthed/mimed).",
        },
    ],
}


def sessions() -> list[dict]:
    """Per-session rows from ``utils`` (requires the data drive mounted).

    One row per ``sentences/*.mat`` file: ``n_trials`` from the trial arrays and
    ``task_condition`` from ``speakingMode``, the same labels the Study's timelines carry.
    """
    from neuralshed.datasets import participants

    from . import utils

    rows = []
    for row in utils.session_summary(utils.source_root()):
        label = row["session"]
        date = utils.session_date(label)
        mimed = "nonvocal" in row["speaking_mode"].lower()
        rows.append(
            {
                "label": label,
                "subject_label": "T12",
                "date": date,
                "post_implant_day": participants.post_implant_day(
                    utils.PARTICIPANT_ID, date
                ),
                "n_trials": row["n_trials"],
                "task": "attempted speech",
                "task_condition": "mimed" if mimed else "attempted",
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
