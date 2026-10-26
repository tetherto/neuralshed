DATASET_METADATA = {
    "id": "jude_2026_typing",
    "title": "Restoring rapid natural bimanual typing with a neuroprosthesis after paralysis",
    "description": "Bimanual QWERTY typing decoded from attempted finger movements by two "
    "BrainGate2 participants with tetraplegia (T17, ALS; T18, spinal cord injury). Tasks span "
    "isolated attempted finger movements, open-loop calibration sentences, and closed-loop "
    "sentence copy and free typing. T18 also has a handwriting session. Neural features are "
    "10 ms-binned threshold crossings and spike-band power from six 64-channel Utah arrays per "
    "participant in precentral gyrus: T17's span areas 6v, 55b and 6d (left hemisphere), T18's "
    "are all area 6d (four left, two right). Online typing reached about 110 characters per "
    "minute (22 words per minute).",
    "source_data_platform": "dryad",
    "source_id": "doi:10.5061/dryad.cz8w9gjjk",
    "source_data_url": "https://datadryad.org/dataset/doi:10.5061/dryad.cz8w9gjjk",
    "status": "ml_ready",
    "macro_task": "human_communication",
    "decode_target": "text",
    "study_class": "Jude2026Typing",  # datasets/jude_2026_typing/study.py
    "bibtex": """
@article{jude2026typing,
  title = {Restoring rapid natural bimanual typing with a neuroprosthesis after paralysis},
  author = {Jude, Justin J. and Levi-Aharoni, Hadar and Acosta, Alexander J. and Allcroft, Shane B. and Nicolas, Claire and Lacayo, Bayardo E. and Card, Nicholas S. and Wairagkar, Maitreyee and Levin, Alisa D. and Brandman, David M. and Stavisky, Sergey D. and Willett, Francis R. and Williams, Ziv M. and Simeral, John D. and Hochberg, Leigh R. and Rubin, Daniel B.},
  journal = {Nature Neuroscience},
  year = {2026},
  month = mar,
  doi = {10.1038/s41593-026-02218-y},
  url = {https://doi.org/10.1038/s41593-026-02218-y},
}
""",
    "publication_doi": "10.1038/s41593-026-02218-y",
    "code_repo": "https://github.com/justin-jude/typingBCI",
    "licence": "CC0-1.0",
    "notes": "19 sessions across 2 participants (T17 Trial Days 55-144, 2024-04-24 to "
    "2024-07-22, T18 Trial Days 43-162, 2024-12-03 to 2025-04-01), 165 .mat files (34.9 GB): "
    "160 task blocks (2,649 trials), 3 reference blocks, 1 free-typing block with no recorded "
    "sentences and 1 re-exported copy of a block (utils.EXCLUDED_BLOCKS, utils.DUPLICATE_FILES). "
    "Three T17 finger-movement blocks listed in the Dryad README (2024-06-03 blocks 1-3) are "
    "not in the deposit. Raw bins are 10 ms (100 Hz), the authors' training compresses them to "
    "20 ms. Both participants store 384 channels, six 64-channel arrays in file order, and "
    "T17's order is the Study order of jude_2026_speech. Task family comes from the Dryad block "
    "tables (utils.SESSION_BLOCKS), the loop type from each file's block_description, which "
    "shows sentence copy is closed loop where the Dryad README says open. 11 blocks pause the "
    "recording (49 pauses of 22 ms-420 s with no bins, utils.clock_breaks). Cued sentences come from the "
    "Switchboard corpus. Code via the typingBCI repo, clinical trial NCT00912041. "
    "LanguageModels.zip (5-gram language model) is out of scope.",
    "subjects": [
        {
            "label": "T17",
            "participant_id": "braingate2/T17",  # datasets/participants.py
            # This paper's Methods and kunz_2025 give 33. jude_2026_speech keeps its own
            # paper's 34 for recordings from day 25 of the same implant.
            "age": "P33Y",
            "time_since_implant": "P55D",
            "clinical_condition": [
                "Amyotrophic lateral sclerosis (ALS)",
                "Anarthria",
                "Tetraplegia",
                "Locked-in syndrome",
                "Ventilator-dependent",
            ],
            # Six 64-channel Utah arrays in the left precentral gyrus, two per area, in file
            # column order. The authors' FVAF notebook titles the column groups v6v, d6v, v55b,
            # d55b, v6d, d6d, the order jude_2026_speech uses for the same implant (README,
            # Channel order). No stereotaxic coordinates are published.
            "arrays": [
                {
                    "name": "ventral_6v",
                    "hemisphere": "left",
                    "device_type": "utah_array",
                    "n_channels": 64,
                    "channel_start": 0,
                    "channel_stop": 63,
                    "brain_area": "ventral precentral gyrus (area 6v), left hemisphere",
                },
                {
                    "name": "dorsal_6v",
                    "hemisphere": "left",
                    "device_type": "utah_array",
                    "n_channels": 64,
                    "channel_start": 64,
                    "channel_stop": 127,
                    "brain_area": "ventral precentral gyrus (area 6v), left hemisphere",
                },
                {
                    "name": "ventral_55b",
                    "hemisphere": "left",
                    "device_type": "utah_array",
                    "n_channels": 64,
                    "channel_start": 128,
                    "channel_stop": 191,
                    "brain_area": "middle precentral gyrus (area 55b), left hemisphere",
                },
                {
                    "name": "dorsal_55b",
                    "hemisphere": "left",
                    "device_type": "utah_array",
                    "n_channels": 64,
                    "channel_start": 192,
                    "channel_stop": 255,
                    "brain_area": "middle precentral gyrus (area 55b), left hemisphere",
                },
                {
                    "name": "ventral_6d",
                    "hemisphere": "left",
                    "device_type": "utah_array",
                    "n_channels": 64,
                    "channel_start": 256,
                    "channel_stop": 319,
                    "brain_area": "dorsal precentral gyrus (area 6d), left hemisphere",
                },
                {
                    "name": "dorsal_6d",
                    "hemisphere": "left",
                    "device_type": "utah_array",
                    "n_channels": 64,
                    "channel_start": 320,
                    "channel_stop": 383,
                    "brain_area": "dorsal precentral gyrus (area 6d), left hemisphere",
                },
            ],
            # Region ontology (subject-owned BrainArea), keyed by each array's brain_area.
            # HOMBA has no HCP 6d/6v leaves, so these map to the orientation-based divisions
            # of area 6, ids identical to jude_2026_speech and card_2024. UBERON has no
            # premotor subdivision, so all three map up to premotor cortex.
            "brain_areas": [
                {
                    "raw_label": "ventral precentral gyrus (area 6v), left hemisphere",
                    "homba_id": "HOMBA_10170",  # lateroventral division of area 6 (A6lv); ventral premotor
                    "uberon_id": "UBERON_0016634",  # premotor cortex
                },
                {
                    "raw_label": "middle precentral gyrus (area 55b), left hemisphere",
                    "homba_id": "HOMBA_AA30011",  # area 55b (HCP), under the laterodorsal division of area 6
                    "uberon_id": "UBERON_0016634",  # premotor cortex (UBERON has no 55b)
                },
                {
                    "raw_label": "dorsal precentral gyrus (area 6d), left hemisphere",
                    "homba_id": "HOMBA_10169",  # laterodorsal division of area 6 (A6ld); dorsal premotor
                    "uberon_id": "UBERON_0016634",  # premotor cortex (no dorsal subdivision in UBERON)
                },
            ],
        },
        {
            "label": "T18",
            "participant_id": "braingate2/T18",  # datasets/participants.py
            "age": "P48Y",
            "time_since_implant": "P43D",
            "clinical_condition": [
                "Spinal cord injury (C4 ASIA A)",
                "Tetraplegia",
            ],
            # Six 64-channel Utah arrays, all in area 6d: four in the left hemisphere and two
            # in the right, in file column order. The authors' FVAF and PlotArrayCERs notebooks
            # label the column groups mv6d, v6d, d6d, md6d, Rl6d, Rm6d. Reading the left "m" as
            # mid (between the ventral and dorsal arrays) and the right "l"/"m" as lateral and
            # medial is ours. No stereotaxic coordinates are published.
            "arrays": [
                {
                    "name": "mid_ventral_6d",
                    "hemisphere": "left",
                    "device_type": "utah_array",
                    "n_channels": 64,
                    "channel_start": 0,
                    "channel_stop": 63,
                    "brain_area": "dorsal precentral gyrus (area 6d), left hemisphere",
                },
                {
                    "name": "ventral_6d",
                    "hemisphere": "left",
                    "device_type": "utah_array",
                    "n_channels": 64,
                    "channel_start": 64,
                    "channel_stop": 127,
                    "brain_area": "dorsal precentral gyrus (area 6d), left hemisphere",
                },
                {
                    "name": "dorsal_6d",
                    "hemisphere": "left",
                    "device_type": "utah_array",
                    "n_channels": 64,
                    "channel_start": 128,
                    "channel_stop": 191,
                    "brain_area": "dorsal precentral gyrus (area 6d), left hemisphere",
                },
                {
                    "name": "mid_dorsal_6d",
                    "hemisphere": "left",
                    "device_type": "utah_array",
                    "n_channels": 64,
                    "channel_start": 192,
                    "channel_stop": 255,
                    "brain_area": "dorsal precentral gyrus (area 6d), left hemisphere",
                },
                {
                    "name": "lateral_6d",
                    "hemisphere": "right",
                    "device_type": "utah_array",
                    "n_channels": 64,
                    "channel_start": 256,
                    "channel_stop": 319,
                    "brain_area": "dorsal precentral gyrus (area 6d), right hemisphere",
                },
                {
                    "name": "medial_6d",
                    "hemisphere": "right",
                    "device_type": "utah_array",
                    "n_channels": 64,
                    "channel_start": 320,
                    "channel_stop": 383,
                    "brain_area": "dorsal precentral gyrus (area 6d), right hemisphere",
                },
            ],
            # All six arrays are area 6d (the hand knob), so both hemispheres share T17's 6d ids.
            "brain_areas": [
                {
                    "raw_label": "dorsal precentral gyrus (area 6d), left hemisphere",
                    "homba_id": "HOMBA_10169",  # laterodorsal division of area 6 (A6ld); dorsal premotor
                    "uberon_id": "UBERON_0016634",  # premotor cortex
                },
                {
                    "raw_label": "dorsal precentral gyrus (area 6d), right hemisphere",
                    "homba_id": "HOMBA_10169",  # laterodorsal division of area 6 (A6ld); dorsal premotor
                    "uberon_id": "UBERON_0016634",  # premotor cortex
                },
            ],
        },
    ],
    "events": [
        {
            "name": "binned_neural_threshold_crossings",
            "category": "data",
            "event_class": "SpikeCounts",
            "signal_type": "threshold_crossings",
            "modality": "intracortical_microelectrode",
            "bin_size_ms": 10.0,
            "rate_hz": 100.0,
            "dtype": "int16",
            "description": "Non-causal threshold-crossing counts (-3.5 x SD, after linear "
            "regression referencing) per 10 ms bin, (n_bins, 384) for both participants. Raw "
            "counts, not z-scored.",
        },
        {
            "name": "binned_neural_spike_band_power",
            "category": "data",
            "event_class": "BandPower",
            "signal_type": "spike_band_power",
            "modality": "intracortical_microelectrode",
            "bin_size_ms": 10.0,
            "rate_hz": 100.0,
            "dtype": "float32",
            "description": "Spike-band power (sum of squared 250 Hz-5 kHz voltages, capped at "
            "50,000 uV^2) per 10 ms bin, same (n_bins, 384) shape as the crossings. Raw, not "
            "z-scored.",
        },
        {
            "name": "cue_sentence",
            "category": "text",
            "event_class": "Sentence",
            "task_condition": "attempted",
            "description": "Sentence the participant attempts to type (calibration, sentence "
            "copy, free typing, modality 'typed') or to handwrite (modality 'written'), over the "
            "go window. Free-typing sentences were entered after the block and confirmed with "
            "the participant. Handwriting cues have their '>' (space) and '~' (end) tokens "
            "removed. The text keeps the source's case and punctuation, including apostrophes "
            "the 30-key keyboard cannot type.",
        },
        {
            "name": "finger_movement",
            "category": "categorical",
            "event_class": "Action",
            "task_condition": "attempted",
            "description": "Attempted isolated finger movement over the go window "
            "(action_kind 'finger_movement'), code 1-30 from the QWERTY layout (utils.CUE_CODES): "
            "10 fingers by extension, straight-down flexion or curl into the palm. The key-cue "
            "sweep (2024-12-04) cues the key a movement types and shares its code.",
        },
        {
            "name": "rest",
            "category": "categorical",
            "event_class": "Action",
            "task_condition": "na",
            "description": "The finger-movement blocks' DO NOTHING condition (action_kind "
            "'rest', code 0).",
        },
        {
            "name": "cue_stimulus",
            "category": "categorical",
            "event_class": "Stimulus",
            "task_condition": "na",
            "description": "On-screen cue of each trial over the delay, trial start to go "
            "(stimulus_kind movement_cue, rest_cue or sentence_cue, code as the unit). Free "
            "typing is uncued and has none.",
        },
        {
            "name": "cue_go",
            "category": "categorical",
            "event_class": "Stimulus",
            "task_condition": "na",
            "description": "One-bin go marker of each trial (stimulus_kind movement_go_cue).",
        },
        {
            "name": "intertrial",
            "category": "categorical",
            "event_class": "Intertrial",
            "task_condition": "na",
            "description": "Gaps of 0.1 s or more from one trial's end to the next trial's "
            "start, except across a recording pause.",
        },
        {
            "name": "microphone_audio",
            "category": "data",
            "event_class": None,
            "signal_type": "audio",
            "modality": "audio",
            "bin_size_ms": None,
            "rate_hz": 30000.0,
            "dtype": "int16",
            "description": "Source only, not emitted: single-channel 30 kHz microphone audio "
            "per block (microphone_data, microphone_nsp_time), T17 only. T17 is anarthric, so "
            "it records breathing and effort, not speech. Read with utils.get_audio.",
        },
        {
            "name": "decoder_logit_output",
            "category": "data",
            "event_class": None,
            "signal_type": "other",
            "modality": "derived",
            "bin_size_ms": None,
            "rate_hz": None,
            "dtype": "float32",
            "description": "Source only, not emitted: the online RNN's character logits in "
            "the closed-loop blocks (sentence copy, free typing, handwriting block 13), "
            "(n_rows, 41) of which the first 31 are used (blank, space, comma, question mark, "
            "period, A-Z). About half the rows are zero padding, and the non-zero rows arrive "
            "every 80-160 ms. Open-loop blocks store a (1, 41) placeholder.",
        },
        {
            "name": "ngram_decoder_output",
            "category": "text",
            "event_class": None,
            "description": "Source only, not emitted: the 5-gram language model's partial and "
            "final strings per logit row in the closed-loop blocks.",
        },
        {
            "name": "norm_channel_stats",
            "category": "data",
            "event_class": None,
            "signal_type": "other",
            "modality": "derived",
            "bin_size_ms": None,
            "rate_hz": None,
            "dtype": "float32",
            "description": "Source only, not emitted: the online decoder's per-feature "
            "normalization means and standard deviations in the closed-loop blocks, updated per "
            "trial. T18 uses all 768 features. T17 uses 256 on 2024-06-03 (crossings and power "
            "of the two 6d arrays) and 512 afterwards (6v and 6d, 55b omitted).",
        },
    ],
}


def sessions() -> list[dict]:
    """Per-session rows from ``utils`` (requires the data drive mounted).

    One row per session: subject, ISO date, post-implant (Trial) day, the trial count and
    task families of its catalogued blocks (``utils.SESSION_BLOCKS``, the Study's
    timelines), and ``task_condition`` ``attempted`` (every unit is an attempted movement,
    rest aside).
    """
    import scipy.io

    from . import utils

    rows = []
    for ref in utils.list_sessions():
        families = utils.SESSION_BLOCKS[ref]
        fams = sorted(set(families.values()))
        n_trials = 0
        for block in families:
            p = utils.block_path(ref, block)
            for name, shape, _ in scipy.io.whosmat(str(p)):
                if name == "cue":
                    n_trials += int(shape[0]) if shape else 0
                    break
        rows.append(
            {
                "label": ref,
                "subject_label": utils.subject_of(ref),
                "date": utils.session_date(ref),
                "post_implant_day": utils.post_implant_day(ref),
                "n_trials": n_trials,
                "task": ", ".join(fams),
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
