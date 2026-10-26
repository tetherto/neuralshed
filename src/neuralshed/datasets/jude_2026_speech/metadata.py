DATASET_METADATA = {
    "id": "jude_2026_speech",
    "title": "Decoding intended speech with an intracortical brain-computer interface "
    "in a person with longstanding anarthria and locked-in syndrome",
    "description": "Cued intended/attempted speech by BrainGate2 participant T17 (ALS, "
    "anarthria, locked-in). Tasks span isolated orofacial movements, phonemes, words, "
    "and sentences (50-word and 125k-word vocab, open/closed loop), plus a phased "
    "reading vs internal vs attempted-speech task. Neural features are 10 ms-binned "
    "threshold crossings and spike-band power from six 64-channel Utah arrays in left "
    "precentral gyrus (areas 6d, 6v, 55b); 30 kHz microphone audio is recorded per block.",
    "source_data_platform": "dryad",
    "source_id": "doi:10.5061/dryad.vq83bk481",
    "source_data_url": "https://datadryad.org/dataset/doi:10.5061/dryad.vq83bk481",
    "status": "ml_ready",
    "macro_task": "human_communication",
    "decode_target": "text",
    "study_class": "Jude2026IntendedSpeech",  # datasets/jude_2026_speech/study.py
    "bibtex": """
@article{jude2026anarthria,
  title = {Decoding intended speech with an intracortical brain-computer interface in a person with long-standing anarthria and locked-in syndrome},
  author = {Jude, Justin J. and Haro, Stephanie and Levi-Aharoni, Hadar and Hashimoto, Hiroaki and Acosta, Alexander J. and Card, Nicholas S. and Wairagkar, Maitreyee and Brandman, David M. and Stavisky, Sergey D. and Williams, Ziv M. and Cash, Sydney S. and Simeral, John D. and Hochberg, Leigh R. and Rubin, Daniel B.},
  journal = {Cell Reports},
  volume = {45},
  number = {4},
  pages = {117162},
  year = {2026},
  month = apr,
  doi = {10.1016/j.celrep.2026.117162},
  url = {https://doi.org/10.1016/j.celrep.2026.117162},
}
""",
    "publication_doi": "10.1016/j.celrep.2026.117162",
    "code_repo": "https://github.com/justin-jude/anarthria-speech-bci",
    "licence": "CC0-1.0",
    "notes": "14 sessions, Trial Days 25-166 (2024-03-25 .. 2024-08-13), 140 .mat files "
    "(~38.6 GB) of which 121 are task blocks (5,114 trials) and 19 are reference, "
    "diagnostic, aborted or unreadable (utils.EXCLUDED_BLOCKS). Raw bins are 10 ms / "
    "100 Hz (the authors' training pipeline compresses to 20 ms). Channel count is "
    "per-session: 384 (all six arrays) on 11 sessions, 256 (6v and 55b only, 6d dropped "
    "at acquisition) on 2024.04.15/16/17. The Study emits channels in utils.ARRAY_ORDER "
    "(6v, 55b, 6d, one block of 64 per array), permuting the three March and early April "
    "sessions whose files order the arrays 6d, 6v, 55b. Threshold at -3.5xSD; spike-band "
    "power = sum of squared voltages per 10 ms bin, capped at 50,000. The online decoder "
    "of the closed-loop blocks ran on 64, 128 or 147 electrodes depending on the day "
    "(README, Online decoder configuration), never on all 256 or 384. Phased "
    "read/imagined/attempted task is session 2024.05.23 only. Code Zenodo DOI "
    "10.5281/zenodo.18599809; trial NCT00912041.",
    "subjects": [
        {
            "label": "T17",
            "participant_id": "braingate2/T17",  # datasets/participants.py
            "age": "P34Y",
            "time_since_implant": "P25D",
            "clinical_condition": [
                "Amyotrophic lateral sclerosis (ALS)",
                "Anarthria",
                "Tetraplegia",
                "Locked-in syndrome",
                "Ventilator-dependent",
            ],
            # Six 64-channel Utah arrays (384 electrodes) in the left precentral gyrus, two
            # per region. Channel ranges are Study columns (utils.ARRAY_ORDER): the Study
            # permutes each file into this order, see README, Channel order. The 256-channel
            # sessions (2024.04.15/16/17) hold the first four arrays. No stereotaxic
            # coordinates are published (targets set from HCP-pipeline fMRI parcellation).
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
            "brain_areas": [
                {
                    "raw_label": "dorsal precentral gyrus (area 6d), left hemisphere",
                    "homba_id": "HOMBA_10169",  # laterodorsal division of area 6 (A6ld); dorsal premotor
                    "uberon_id": "UBERON_0016634",  # premotor cortex (no dorsal subdivision in UBERON)
                },
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
            ],
        }
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
            "description": "Non-causal threshold-crossing counts (-3.5xSD) per 10 ms bin, "
            "(n_bins, n_channels) where n_channels is 384 (all arrays) or 256 (6v and 55b "
            "only) per session. Raw counts, not z-scored, emitted in Study column order.",
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
            "description": "Spike-band power (sum of squared 250 Hz-5 kHz voltages, capped "
            "at 50,000 uV^2) per 10 ms bin, same (n_bins, n_channels) shape as "
            "threshold_crossings. Raw, emitted in Study column order.",
        },
        {
            "name": "cue_sentence",
            "category": "text",
            "event_class": "Sentence",
            "task_condition": "attempted",
            "description": "Sentence-task cue (50-word or 125k-word vocabulary, open or "
            "closed loop) as orthographic `text` (language 'en', modality 'read') plus the "
            "ARPAbet `phonemes` target (the prompt's dictionary pronunciation from "
            "assets/cue_phonemes.json, PhonemeVocab v1 with '|' word boundaries, not a "
            "measurement of what T17 produced, no per-phoneme timing) over the go window, "
            "with `trial_id` and `success`. Phased-session sentence trials emit three rows, "
            "read/imagined/attempted on `task_condition` and `phase`.",
        },
        {
            "name": "cue_word",
            "category": "text",
            "event_class": "Word",
            "task_condition": "attempted",
            "description": "Word-task cue (50-word vocabulary) as `text` plus the ARPAbet "
            "`phonemes` target (assets/cue_phonemes.json, as for sentences) over the go "
            "window. Phased-session word trials emit read/imagined/attempted rows.",
        },
        {
            "name": "cue_phoneme",
            "category": "text",
            "event_class": "Phoneme",
            "task_condition": "attempted",
            "description": "Phoneme-task trial: the ARPAbet target of the cued syllable "
            "(`utils.PHONEME_CUES`, 39 symbols of `PhonemeVocab.canonical()`) as `text` "
            "and `phonemes`, the screen cue as `description`, over the go window. The rest cue is an "
            "`Action` (`action_kind='rest'`), never a phoneme. Phased-session trials emit "
            "read/imagined/attempted rows.",
        },
        {
            "name": "cue_orofacial",
            "category": "categorical",
            "event_class": "Action",
            "task_condition": "attempted",
            "description": "Orofacial-block trial over the go window: the 12 cued movements "
            "(`action_kind='orofacial_movement'`), the four 'Say ...' syllables "
            "(`action_kind='speech'`) and rest (`action_kind='rest'`, `task_condition` "
            "'na'), with the cue as `description` and the dataset-wide class id as `code` "
            "(`utils.CUE_CODES`).",
        },
        {
            "name": "cue_stimulus",
            "category": "categorical",
            "event_class": "Stimulus",
            "description": "Visual cue of every trial over the delay (trial start to go, "
            "`stimulus_kind` phoneme/word/sentence/movement/syllable/rest_cue, `code` from "
            "`utils.CUE_CODES`, sentences -100) and a one-bin go marker "
            "(`speech_go_cue` or `movement_go_cue`). Phased trials carry a "
            "`speech_delay_cue` and a go marker before the imagined and attempted phases.",
        },
        {
            "name": "intertrial",
            "category": "categorical",
            "event_class": "Intertrial",
            "description": "Gaps the source clocks delimit: trial end to next trial start "
            "in closed-loop blocks (decoder output and text-to-speech playback, 2-40 s) "
            "and the 2 s `intertrial_start` to `cue_presentation` window of each phased "
            "trial. Open-loop trials abut, so those blocks have no rows.",
        },
        # Source-only streams: in the files, not emitted by the Study.
        {
            "name": "microphone_audio",
            "category": "data",
            "event_class": None,
            "signal_type": "audio",
            "modality": "audio",
            "bin_size_ms": None,
            "rate_hz": 30000.0,
            "dtype": "int16",
            "description": "Source only, not emitted: single-channel 30 kHz microphone "
            "audio per block (microphone_data + microphone_nsp_time). T17 is anarthric, so "
            "it holds breathing and effort, and in closed-loop blocks the text-to-speech "
            "playback of the decoded sentence. Readable with utils.get_audio.",
        },
        {
            "name": "decoder_logit_output",
            "category": "data",
            "event_class": None,
            "signal_type": "other",
            "modality": "derived",
            "bin_size_ms": 80.0,
            "rate_hz": 12.5,
            "dtype": "float32",
            "description": "Source only, not emitted: the online RNN's 41 phoneme logits "
            "per row in the closed-loop blocks (decoder_logit_output, "
            "decoder_output_redis_clock). Rows are written every few milliseconds and a "
            "non-zero row appears every 80 ms. The reference code carries two class "
            "orderings for these logits, so the map is not verified.",
        },
        {
            "name": "ngram_decoder_output",
            "category": "text",
            "event_class": None,
            "description": "Source only, not emitted: the 5-gram language model's partial "
            "and final strings per logit row in the closed-loop blocks "
            "(ngram_decoder_partial_output, ngram_decoder_final_output).",
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
            "normalization means and standard deviations, one row per update "
            "(norm_channel_means, norm_channel_stds, norm_redis_times), 128, 256 or 294 "
            "wide depending on the day, the electrodes the decoder ran on.",
        },
    ],
}


def sessions() -> list[dict]:
    """Per-session rows from ``utils`` (requires the data drive mounted).

    One row per session: ISO date, post-implant (Trial) day, ``n_trials`` summed over the
    session's catalogued blocks (``utils.SESSION_BLOCKS``, the Study's timelines), ``task``
    as the task families present and ``task_condition`` as the rows carry it:
    ``attempted`` everywhere except the phased session, whose rows split into read,
    imagined and attempted phases (``other``).
    """
    from . import utils

    rows = []
    for ref in utils.list_sessions():
        families = utils.SESSION_BLOCKS[ref]
        n_trials = sum(
            len(utils.get_trials(utils.block_path(ref, block))) for block in families
        )
        phased = all(fam.startswith("phased") for fam in families.values())
        rows.append(
            {
                "label": ref,
                "subject_label": "T17",
                "date": utils.session_date(ref),
                "post_implant_day": utils.post_implant_day(ref),
                "n_trials": n_trials,
                "task": ", ".join(sorted(set(families.values()))),
                "task_condition": "other" if phased else "attempted",
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
