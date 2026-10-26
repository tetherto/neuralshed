"""Catalog metadata for the Kunz et al. 2025 inner-speech dataset.

Standalone (no DB import): ``DATASET_METADATA`` is the static catalog record,
``sessions()`` adds per-file rows (needs the data drive), ``get_metadata()`` is the
accessor the DB layer consumes. Validate with::

    python -m neuralshed.database.validate --datasets kunz_2025

Subject details are from the paper STAR Methods (Kunz et al. 2025, Cell 188). The array
layout follows ``utils.ARRAY_LAYOUTS``, and ``channel_start`` / ``channel_stop`` are stored
columns. The in-file ``chanSetNames`` disagree with it in places (see README.md, Channel layout).
"""

DATASET_METADATA = {
    "id": "kunz_2025",
    "title": "Inner speech in motor cortex and implications for speech neuroprostheses",
    "description": "Instructed-delay verbal-behavior and sentence-production tasks probing "
    "the neural representation of inner (imagined) speech in motor cortex, across four "
    "BrainGate2 participants with severe dysarthria/anarthria. Neural features are binned "
    "threshold crossings and spike-band power (10-40 ms bins) from 64-channel Utah arrays in "
    "the left precentral gyrus and, for T12, the left inferior frontal gyrus. A real-time RNN "
    "decodes imagined sentences from a 125k-word vocabulary. Tasks span seven isolated verbal "
    "behaviors (attempted/mimed/inner speech, listening, reading), sentence production, "
    "conjunctive counting, and sequence recall.",
    "source_data_platform": "dryad",
    "source_id": "doi:10.5061/dryad.gf1vhhn1j",
    "source_data_url": "https://datadryad.org/dataset/doi:10.5061/dryad.gf1vhhn1j",
    "status": "ml_ready",
    "macro_task": "human_communication",
    "decode_target": "text",
    "study_class": "Kunz2025InnerSpeech",  # datasets/kunz_2025/study.py
    "bibtex": """
@article{kunz2025innerspeech,
  title = {Inner speech in motor cortex and implications for speech neuroprostheses},
  author = {Kunz, Erin M. and Abramovich Krasa, Benyamin and Kamdar, Foram and Avansino, Donald T. and Hahn, Nick and Yoon, Seonghyun and Singh, Akansha and Nason-Tomaszewski, Samuel R. and Card, Nicholas S. and Jude, Justin J. and Jacques, Brandon G. and Bechefsky, Payton H. and Iacobacci, Carrina and Hochberg, Leigh R. and Rubin, Daniel B. and Williams, Ziv M. and Brandman, David M. and Stavisky, Sergey D. and AuYong, Nicholas and Pandarinath, Chethan and Druckmann, Shaul and Henderson, Jaimie M. and Willett, Francis R.},
  journal = {Cell},
  volume = {188},
  number = {17},
  pages = {4658–4673.e17},
  year = {2025},
  month = aug,
  doi = {10.1016/j.cell.2025.06.015},
  url = {https://doi.org/10.1016/j.cell.2025.06.015},
}
""",
    "publication_doi": "10.1016/j.cell.2025.06.015",
    "code_repo": "https://github.com/nptl-stanford/inner_speech",
    "licence": "CC0-1.0",  # Dryad data deposit (CC0)
    "notes": "BrainGate2 trial NCT00912041. 69 .mat files across 11 task types / 4 "
    "participants, 243 Study timelines, one per recorded block of the 41 timeline files "
    "(isolatedVerbalBehaviors keeps only the combined *_allBehaviors file, dropping its 7 "
    "per-behavior subsets). Each file concatenates its blocks with the unrecorded inter-block "
    "gaps removed, so features index a uniform bin grid within a block. Bin size varies by "
    "file (10/20/25/33 ms, and 40 ms inferred for the five T16 seqRecall files, which store "
    "20). The word, sentence and counting families ship un-normalized features (binnedTX "
    "uint8 counts, spikePow float64 µV²), the seqRecall families ship block-wise "
    "mean-subtracted features. The decoder output (decodedSentenceText) and the go-aligned "
    "sentenceFeatures copy are out of scope. Joystick and stylus trajectories, counting "
    "correctness, audio waveforms, the counting grids, the prompted-thought data of Figure S5 "
    "and the decoder training sessions of Table S7 are not in the deposit.",
    "subjects": [
        {
            "label": "T12",
            "participant_id": "braingate2/T12",  # datasets/participants.py
            "age": "P68Y",
            "time_since_implant": "P412D",  # paper: post-implant days 412-995
            # Same wording as willet_2023, which the paper's participant section repeats.
            "clinical_condition": [
                "Slowly progressive bulbar-onset amyotrophic lateral sclerosis (ALS), "
                "diagnosed at age 59, ALSFRS-R 26 at enrollment",
                "Severe dysarthria for nearly 8 years: vocalizes while attempting to speak "
                "with some differentiable vowels, no intelligible consonants or sentences",
                "Partial use of the limbs, communicates through a writing board or tablet",
            ],
            # Same implant, names and ontology ids as willet_2023, in this Study's column order
            # (inferior array first in each pair). 46 of the 127 T12 timelines store only the
            # two 6v arrays (columns 0-127). The 81 seqRecall and December 2024 timelines store
            # all four. The December 2024 in-file labels (M1, 55b) are a relabel of the same
            # columns, see README.md, Channel layout.
            "arrays": [
                {
                    "name": "ventral_6v",
                    "hemisphere": "left",
                    "device_type": "utah_array",
                    "n_channels": 64,
                    "channel_start": 0,
                    "channel_stop": 63,
                    "brain_area": "area 6v, left ventral precentral gyrus",
                },
                {
                    "name": "dorsal_6v",
                    "hemisphere": "left",
                    "device_type": "utah_array",
                    "n_channels": 64,
                    "channel_start": 64,
                    "channel_stop": 127,
                    "brain_area": "area 6v, left ventral precentral gyrus",
                },
                {
                    "name": "ventral_44",
                    "hemisphere": "left",
                    "device_type": "utah_array",
                    "n_channels": 64,
                    "channel_start": 128,
                    "channel_stop": 191,
                    "brain_area": "area 44, left inferior frontal gyrus (Broca's area)",
                },
                {
                    "name": "dorsal_44",
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
                    "brodmann": "6",
                },
                {
                    "raw_label": "area 44, left inferior frontal gyrus (Broca's area)",
                    "homba_id": "HOMBA_10186",
                    "uberon_id": "UBERON_0002980",
                    "brodmann": "44",
                },
            ],
        },
        {
            "label": "T15",
            "participant_id": "braingate2/T15",  # datasets/participants.py
            "age": "P45Y",
            "time_since_implant": "P230D",  # paper: post-implant days 230-538
            # Same wording as card_2024.
            "clinical_condition": [
                "Amyotrophic lateral sclerosis (ALS)",
                "Tetraparesis",
                "Severe dysarthria",
            ],
            # Same implant, names, column order and ontology ids as card_2024. Array centers
            # from the NEJM 2024 supplementary appendix S1.12, as in card_2024.
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
                    "brodmann": None,
                },
            ],
        },
        {
            "label": "T16",
            "participant_id": "braingate2/T16",  # datasets/participants.py
            "age": "P52Y",
            "time_since_implant": "P88D",  # paper: post-implant days 88-377
            "clinical_condition": [
                "Pontine stroke (about 19 years before enrollment)",
                "Tetraplegia",
                "Dysarthria",
            ],
            # Paper: two hand-knob (6d) arrays, one ventral premotor (6v), one on the border
            # of PEF and speech-related 55b. Most files label columns 128-255 as one `6d`
            # group. The seqRecall files name the two arrays, i6d then s6d. The paper found no
            # speech decoding on pef_55b and the two 6d arrays and excluded them (Figure 1C).
            "arrays": [
                {
                    "name": "pef_55b",
                    "hemisphere": "left",
                    "device_type": "utah_array",
                    "n_channels": 64,
                    "channel_start": 0,
                    "channel_stop": 63,
                    "brain_area": "premotor eye field / border area 55b, left precentral gyrus",
                },
                {
                    "name": "6v",
                    "hemisphere": "left",
                    "device_type": "utah_array",
                    "n_channels": 64,
                    "channel_start": 64,
                    "channel_stop": 127,
                    "brain_area": "ventral area 6v, left precentral gyrus",
                },
                {
                    "name": "ventral_6d",
                    "hemisphere": "left",
                    "device_type": "utah_array",
                    "n_channels": 64,
                    "channel_start": 128,
                    "channel_stop": 191,
                    "brain_area": "hand-knob area 6d, left precentral gyrus",
                },
                {
                    "name": "dorsal_6d",
                    "hemisphere": "left",
                    "device_type": "utah_array",
                    "n_channels": 64,
                    "channel_start": 192,
                    "channel_stop": 255,
                    "brain_area": "hand-knob area 6d, left precentral gyrus",
                },
            ],
            # The border array maps to the laterodorsal division of area 6 (HOMBA_10169), the
            # parent of both the frontal eye field (HOMBA_AA30010) and 55b (HOMBA_AA30011):
            # the source places it on their border, not inside either.
            "brain_areas": [
                {
                    "raw_label": "premotor eye field / border area 55b, left precentral gyrus",
                    "homba_id": "HOMBA_10169",
                    "uberon_id": "UBERON_0016634",
                    "brodmann": None,
                },
                {
                    "raw_label": "ventral area 6v, left precentral gyrus",
                    "homba_id": "HOMBA_10170",
                    "uberon_id": "UBERON_0016634",
                    "brodmann": "6",
                },
                {
                    "raw_label": "hand-knob area 6d, left precentral gyrus",
                    "homba_id": "HOMBA_10169",
                    "uberon_id": "UBERON_0016634",
                    "brodmann": "6",
                },
            ],
        },
        {
            "label": "T17",
            "participant_id": "braingate2/T17",  # datasets/participants.py
            # Paper: 33 years of age at the time of this study. jude_2026_speech and
            # jude_2026_typing state P34Y for the same participant.
            "age": "P33Y",
            "time_since_implant": "P284D",  # paper: post-implant days 284-287
            # Same wording as jude_2026_speech.
            "clinical_condition": [
                "Amyotrophic lateral sclerosis (ALS)",
                "Anarthria",
                "Tetraplegia",
                "Locked-in syndrome",
                "Ventilator-dependent",
            ],
            # Six arrays implanted (384 channels). This release stores the two 6v arrays and
            # the two 55b arrays (256 columns), so the two hand-knob 6d arrays are not listed.
            # Names and region strings follow jude_2026_speech. The files label columns
            # 128-255 as one `55b` group. Channel correlations split it at column 192, and the
            # ventral-then-dorsal order of the two halves is assumed from jude_2026_speech's
            # order, not published. The paper found no speech decoding and very little spiking
            # on the two 55b arrays and excluded them (Figure 1C).
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
            ],
            "brain_areas": [
                {
                    "raw_label": "ventral precentral gyrus (area 6v), left hemisphere",
                    "homba_id": "HOMBA_10170",
                    "uberon_id": "UBERON_0016634",
                    "brodmann": "6",
                },
                {
                    "raw_label": "middle precentral gyrus (area 55b), left hemisphere",
                    "homba_id": "HOMBA_AA30011",
                    "uberon_id": "UBERON_0016634",
                    "brodmann": None,
                },
            ],
        },
    ],
    # Events emitted by study.py per timeline: two neural `data` streams and, per trial, a
    # delay Stimulus, a one-bin go Stimulus, the cued unit (Word / Sentence / Action) and an
    # Intertrial over any gap before the next trial. One entry per `task_condition` or
    # `stimulus_kind`. Counts are rows over the 243 timelines.
    "events": [
        {
            "name": "threshold_crossings",
            "category": "data",
            "event_class": "SpikeCounts",
            "signal_type": "threshold_crossings",
            "modality": "intracortical_microelectrode",
            "bin_size_ms": None,  # varies by file, see description
            "rate_hz": None,
            "dtype": "uint8/int8/float64",
            "description": "binnedTX: threshold-crossing counts (-4.5 x SD for T12/T15, -3.5 x SD "
            "for T16/T17, LRR-denoised), (n_channels, n_bins) over one block. Listed before "
            'spike_band_power so aggregation="cat" yields the paper\'s native [binnedTX | '
            "spikePow] order. Bin size by file: 20 ms (21 files), 10 ms (13), 40 ms (5, the T16 "
            "seqRecall files, inferred, stored as 20), 25 ms (1), 33 ms (1). By timeline: 50 Hz "
            "(149), 100 Hz (74), 25 Hz (17), 40 Hz (2), 30.3 Hz (1). The word, sentence and "
            "counting families are un-normalized uint8 counts. The seqRecall families are "
            "block-wise mean-subtracted (int8 for T12, three of them tagged uint8 in the file "
            "and read as int8, float64 for T16).",
        },
        {
            "name": "spike_band_power",
            "category": "data",
            "event_class": "BandPower",
            "signal_type": "spike_band_power",
            "modality": "intracortical_microelectrode",
            "bin_size_ms": None,  # same bin grid as threshold_crossings
            "rate_hz": None,
            "dtype": "uint8/float32/float64",
            "description": "spikePow: spike-band power (mean squared >250 Hz voltage per bin, "
            "µV², LRR-denoised), (n_channels, n_bins) over one block, on the bin grid of "
            "threshold_crossings. The word, sentence and counting families are un-normalized, "
            "the seqRecall families are mean-subtracted. The t12.2024.04.11 interleaved file "
            "stores a copy of binnedTX here, and its rows carry valid=False.",
        },
        {
            "name": "word_attempted",
            "category": "text",
            "event_class": "Word",
            "task_condition": "attempted",
            "description": "Attempted vocalized word over the go period (952 rows). One of 7 cued "
            "single-syllable words (ban/choice/day/feel/kite/though/were), isolated and "
            "interleaved verbal-behavior tasks. Every Word carries `text`, the canonical "
            "`behavior`, `modality` (heard, read or imagined), `cue_text`, `trial_id` and "
            "`period`.",
        },
        {
            "name": "word_mimed",
            "category": "text",
            "event_class": "Word",
            "task_condition": "mimed",
            "description": "Mimed (mouthed, no sound) word over the go period (665 rows).",
        },
        {
            "name": "word_imagined",
            "category": "text",
            "event_class": "Word",
            "task_condition": "imagined",
            "description": "Inner-speech word over the go period (2,786 rows, "
            "`modality='imagined'`). `behavior` separates imagined_motor, imagined_auditory and "
            "imagined_listening.",
        },
        {
            "name": "word_listening",
            "category": "text",
            "event_class": "Word",
            "task_condition": "listening",
            "description": "Word heard (1,141 rows, `modality='heard'`). In the isolated "
            "listening blocks and for T12's interleaved listening trials the word spans the "
            "source's delay epoch, where it was played (`period='presentation'`).",
        },
        {
            "name": "word_read",
            "category": "text",
            "event_class": "Word",
            "task_condition": "read",
            "description": "Word read silently (525 rows), over the source's delay epoch like the "
            "isolated listening words (`period='presentation'`).",
        },
        {
            "name": "sentence_attempted",
            "category": "text",
            "event_class": "Sentence",
            "task_condition": "attempted",
            "description": "Cued sentence the participant attempted to speak, over the go period "
            "(520 rows, sentenceData, `modality='read'`). Every Sentence carries `text`, "
            "`behavior`, `vocab` (50w or 125k), `loop_type` and `original_split`.",
        },
        {
            "name": "sentence_imagined",
            "category": "text",
            "event_class": "Sentence",
            "task_condition": "imagined",
            "description": "Cued sentence produced in inner speech, over the go period (1,830 "
            "rows, sentenceData and chittychittybangbang, `modality='imagined'`). `behavior` is "
            "the participant's strategy: imagined_motor (T12, T15 on 2024-03-03), "
            "imagined_auditory (T16) or imagined_hybrid (T15's 125k sessions). `keyword=True` "
            "marks the 344 chitty trials that open with the imagined unlock keyword, whose "
            "`text` starts with 'chittychittybangbang'.",
        },
        {
            "name": "action_sequence",
            "category": "categorical",
            "event_class": "Action",
            "task_condition": "attempted",
            "description": "Go period of a seqRecall trial (3,162 rows): `action_kind` joystick "
            "(1,016), drawing (1,936) or speech (210), `description` = the cued sequence, "
            "`code` = file-local cue index, `strategy` = the delay-period memory strategy.",
        },
        {
            "name": "action_count",
            "category": "categorical",
            "event_class": "Action",
            "task_condition": "attempted",
            "description": "Self-paced count of conjunctiveCounting reported aloud "
            "(`action_kind='speech'`, 60 rows), with `num_targets` (10-20).",
        },
        {
            "name": "action_rest",
            "category": "categorical",
            "event_class": "Action",
            "task_condition": "na",
            "description": "DO_NOTHING rest trial of any family over its go period "
            "(`action_kind='rest'`, `code=0`, 805 rows).",
        },
        {
            "name": "cue_word",
            "category": "categorical",
            "event_class": "Stimulus",
            "task_condition": "na",
            "description": "`stimulus_kind='word_cue'`: the word and behavior shown over the "
            "delay period of an active word trial (4,830 rows, `period='delay'`), `code` 1-7 "
            "into the word list (ban, choice, day, feel, kite, though, were).",
        },
        {
            "name": "cue_sentence",
            "category": "categorical",
            "event_class": "Stimulus",
            "task_condition": "na",
            "description": "`stimulus_kind='sentence_cue'`: the sentence shown over the delay "
            "period (2,350 rows, `period='delay'`). In the T16 33 ms eval file the 50 delays are "
            "0.13-0.20 s stubs, and the cue onset is not recorded.",
        },
        {
            "name": "cue_speech_delay",
            "category": "categorical",
            "event_class": "Stimulus",
            "task_condition": "na",
            "description": "`stimulus_kind='speech_delay_cue'`: the direction sequence to be "
            "spoken in seqRecallSpeech, played to T12 and shown to T16 (210 rows, "
            "`period='delay'`).",
        },
        {
            "name": "cue_movement",
            "category": "categorical",
            "event_class": "Stimulus",
            "task_condition": "na",
            "description": "`stimulus_kind='movement_cue'`: the arrow or line sequence shown over "
            "the delay period of a seqRecall movement trial (2,472 rows, `period='delay'`).",
        },
        {
            "name": "cue_movement_verbal_memory",
            "category": "categorical",
            "event_class": "Stimulus",
            "task_condition": "imagined",
            "description": "`movement_cue` of seqRecallVerbalMemory (480 rows): the participant "
            "was told to hold the sequence in inner speech during the delay.",
        },
        {
            "name": "cue_counting_grid",
            "category": "categorical",
            "event_class": "Stimulus",
            "task_condition": "na",
            "description": "`stimulus_kind='counting_grid'`: the shape grid of a "
            "conjunctiveCounting trial (60 rows), with `num_targets`. No go marker follows.",
        },
        {
            "name": "cue_rest",
            "category": "categorical",
            "event_class": "Stimulus",
            "task_condition": "na",
            "description": "`stimulus_kind='rest_cue'`: the DO_NOTHING cue over the delay period "
            "of a rest trial (648 rows).",
        },
        {
            "name": "cue_go",
            "category": "categorical",
            "event_class": "Stimulus",
            "task_condition": "na",
            "description": "One-bin go marker at the start of the go period (10,990 rows, "
            "`period='go'`): `speech_go_cue` (7,917) or `movement_go_cue` (3,073), visual except "
            "the audio go cue of seqRecallUninstructed3ElementLines.",
        },
        {
            "name": "intertrial",
            "category": "categorical",
            "event_class": "Intertrial",
            "task_condition": "na",
            "description": "Every gap between two trials of a block (8,217 rows, "
            "`period='intertrial'`). `description='return period'` on the 1,344 gaps of T12's "
            "lines and memory files (1.5 s to bring the pen tip back), 'intertrial interval' "
            "elsewhere. In passive trials it starts at the source's go epoch. 31 sentence-block "
            "intertrials last over 30 s (up to 195 s).",
        },
    ],
}


def sessions() -> list[dict]:
    """Per-file (timeline) rows from ``utils`` (requires the data drive mounted)."""
    from neuralshed.datasets import participants

    from . import utils

    rows = []
    for p in utils.list_timeline_files():
        s = utils.summarize_file(p)
        date = utils.session_date(p)
        rows.append(
            {
                "label": utils.session_label(p),
                "subject_label": s.participant,
                "date": date,
                "post_implant_day": participants.post_implant_day(
                    f"braingate2/{s.participant}", date
                ),
                "n_trials": s.n_trials,
                "task": s.experiment,
                "task_condition": utils.file_task_condition(p),
                "original_split": utils.original_split(p) or None,
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
    except Exception:  # data drive not mounted or unreadable
        meta["sessions"] = []
    return meta
