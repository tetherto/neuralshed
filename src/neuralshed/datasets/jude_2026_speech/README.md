# Jude et al. 2026: Intended speech iBCI in anarthria & locked-in syndrome

Intracortical BCI study of intended speech in one BrainGate2 participant (**T17**, advanced ALS, anarthric, locked-in, ventilator-dependent). Six 64-channel Utah arrays in the left precentral gyrus record 10 ms-binned threshold crossings and spike-band power while T17 is cued to attempt or intend orofacial movements, phonemes, words, and sentences (50-word and 125,000-word vocabularies, open and closed loop), plus a phased read/internal/attempted-speech task. The aim is to decode intended speech into text and synthesized voice for communication.

## References

- **Paper (Cell Reports, 2026):** Jude, Haro, Levi-Aharoni, Hashimoto, Acosta, Card, Wairagkar, Brandman, Stavisky, Cash, Williams, Simeral, Hochberg, Rubin. "Decoding intended speech with an intracortical brain-computer interface in a person with longstanding anarthria and locked-in syndrome." *Cell Reports* 45, 117162. doi:[10.1016/j.celrep.2026.117162](https://doi.org/10.1016/j.celrep.2026.117162)
- **Source data (Dryad):** doi:[10.5061/dryad.vq83bk481](https://datadryad.org/dataset/doi:10.5061/dryad.vq83bk481), data license CC0 1.0
- **Code repository:** <https://github.com/justin-jude/anarthria-speech-bci>, archived as Zenodo doi:[10.5281/zenodo.18599809](https://doi.org/10.5281/zenodo.18599809). The speech RNN builds on <https://github.com/fwillett/speechBCI>.
- **Clinical trial:** BrainGate2, NCT00912041.

## Species & Subjects

- **Species:** human (*Homo sapiens*).
- **Subject T17:** male, 34 yo, right-handed. **ALS** diagnosed ~3 years before the study. Tetraplegia, anarthria (has not vocalized in >2 years), and ventilator dependence. Only remaining volitional motor control is over the extraocular muscles (locked-in). The typing paper ([`jude_2026_typing`](../jude_2026_typing/README.md), recordings from day 55 of the same implant) and the [`kunz_2025`](../kunz_2025/README.md) paper (day 284) both give the age as 33. This package keeps this paper's 34.
- **Implant:** **six 64-channel Utah arrays = 384 electrodes** (Blackrock, 1.5 mm shanks) in the **left precentral gyrus**, two per region: dorsal (**area 6d**), ventral (**area 6v**), middle (**area 55b**). Targets set from HCP-pipeline fMRI parcellation. No stereotaxic coordinates are published.
- **Channel order.** The Study emits the channels in array order, one block of 64 per array (`ventral_6v, dorsal_6v, ventral_55b, dorsal_55b, ventral_6d, dorsal_6d`, `utils.ARRAY_ORDER`). The source files use two different orders, see [Channel order](#channel-order). The same participant appears in [`jude_2026_typing`](../jude_2026_typing/README.md) (the same array order, coded there for T17) and [`kunz_2025`](../kunz_2025/README.md) (256 channels, 6v then 55b). No electrode-level correspondence across the three datasets is published.

## Behavioral Task

Cued intended/attempted speech (instructed-delay). Because T17 is anarthric, "speech" is **intended/attempted** (and, in the phased task, also **internal/covert**), with no overt vocalization. Task families (per the Dryad block tables, see `utils.SESSION_BLOCKS`):

- **Isolated orofacial movements**: 17 cues, 10 repetitions each: 12 movements (`utils.OROFACIAL_CUES`, e.g. "Clench jaw"), a rest cue ("Do Nothing") and four syllables ("Say Dah", "Say Fah", "Say Jah", "Say Yah"). The paper's 13-class analysis is the 12 movements plus rest.
- **Isolated phonemes**: 40 cues, 18 repetitions each: 39 target phonemes and a rest cue (`DO_NOTHING`). Consonants were cued as consonant + AA syllables ("Bah"), vowels inside a carrier word with the vowel capitalised ("thOUGHt"). `utils.PHONEME_CUES` maps each cue to its ARPAbet symbol, the authors' own map.
- **Individual words**: the 50-word vocabulary (`utils.WORD_CUES`), one trial per word per block.
- **Sentences**: 50-word and 125,000-word vocabularies, each **open-loop** (no feedback) and **closed-loop** (online RNN→5-gram-LM→TTS feedback). The RNN decodes phonemes every 80 ms. Speaking rate ~50-60 wpm.
- **Phased Reading / Internal / Attempted Speech task** (session **2024.05.23** only): each trial cues a language unit (phoneme / word / short sentence / sentence), then three timed phases, **read**, **internal (covert) speech**, and **attempted (vocalized) speech**, delimited by per-trial NSP timestamp markers (`utils.PHASE_WINDOWS`). Five blocks: phonemes (block 1), words (2), short sentences (3 and 6) and sentences (5), 50 trials each over ten cues, five repetitions each. The phoneme cues are ten of the isolated set plus "Ah" and "Oy", the words are eight of the 50-word vocabulary plus two nonsense words ("Cheldgup", "Zelwog"), the short sentences are built from the 50-word vocabulary and the sentences are unrestricted. Block 6 repeats the ten short sentences of block 3 with a read phase of about 4 s instead of 8 s and is absent from the deposit's block table.

The 140 files on disk hold 121 task blocks (`utils.SESSION_BLOCKS`) and 19 others (`utils.EXCLUDED_BLOCKS`, one reason each): 14 reference blocks with a single `DO_NOTHING` trial (the block the session's LRR coefficients and thresholds were fit on), 3 diagnostic speech blocks with 64, 64 and 24 cued trials over 8 reference cues (`04.02` block 0, `04.17` blocks 0 and 9), an aborted 2-trial word block (`03.26` block 18) and one unreadable file (`05.23` block 7). Reported results: phonemes/words/orofacial decode well above chance (orofacial 91.5%, phonemes 50.4%, words 51.1% in area 6v). Sentence WER stayed above the threshold for independent communication, below that of dysarthric (non-anarthric) speakers.

### Sessions

Trial counts are per task block from the files (`utils.get_trials`), the same numbers `metadata.sessions()` reports. The channel column is per session, see [Data Modalities](#data-modalities).

| session | date | Trial Day | channels | task blocks | trials by task family |
|---|---|---:|---:|---:|---|
| `t17.2024.03.25` | 2024-03-25 | 25 | 384 | 14 | phoneme 720, orofacial 170 |
| `t17.2024.03.26` | 2024-03-26 | 26 | 384 | 14 | word 450, sentence_50w_ol 160, sentence_50w_cl 40 |
| `t17.2024.04.02` | 2024-04-02 | 33 | 384 | 13 | sentence_50w_ol 160, sentence_50w_cl 40, sentence_125k_ol 252, sentence_125k_cl 11 |
| `t17.2024.04.15` | 2024-04-15 | 46 | 256 | 9 | sentence_125k_ol 360 |
| `t17.2024.04.16` | 2024-04-16 | 47 | 256 | 5 | sentence_125k_ol 200 |
| `t17.2024.04.17` | 2024-04-17 | 48 | 256 | 9 | sentence_125k_ol 240, sentence_125k_cl 33 |
| `t17.2024.05.06` | 2024-05-06 | 67 | 384 | 9 | sentence_125k_ol 330 |
| `t17.2024.05.07` | 2024-05-07 | 68 | 384 | 10 | sentence_125k_ol 200, sentence_125k_cl 200 |
| `t17.2024.05.13` | 2024-05-13 | 74 | 384 | 2 | sentence_125k_ol 80 |
| `t17.2024.05.14` | 2024-05-14 | 75 | 384 | 7 | sentence_125k_ol 80, sentence_125k_cl 178 |
| `t17.2024.05.23` | 2024-05-23 | 84 | 384 | 5 | phased_phoneme 50, phased_word 50, phased_short_sentence 100, phased_sentence 50 |
| `t17.2024.05.28` | 2024-05-28 | 89 | 384 | 8 | sentence_125k_ol 240, sentence_125k_cl 80 |
| `t17.2024.08.12` | 2024-08-12 | 165 | 384 | 9 | sentence_50w_ol 240, sentence_50w_cl 120 |
| `t17.2024.08.13` | 2024-08-13 | 166 | 384 | 7 | sentence_50w_ol 160, sentence_50w_cl 120 |

In total 5,114 trials: 720 phonemes, 170 orofacial, 450 words, 3,524 non-phased sentences (720 and 320 over the 50-word vocabulary open and closed loop, 1,982 and 502 over the 125,000-word vocabulary) and 250 phased trials.

- **Prompt overlap.** `loop_type` alone does not give prompt-disjoint splits: 80 of the 311 distinct closed-loop 50-word prompts also occur in open-loop blocks, and 77 of the 489 closed-loop 125k prompts do. Hold out whole blocks or sessions, or split on the prompt text.
- **The authors' held-out convention** (`T17_SessionArgs.py` in the reference code) trains the offline sentence decoder on the open-loop 125k blocks of `04.15`, `04.16`, `04.17`, `05.06` and `05.07` and tests on the last trial of each of those sessions, except `05.07` where the last 50 trials are held out.

## Data Modalities

| Modality | Source location | Type / shape |
|---|---|---|
| Threshold crossings | `.mat` `binned_neural_threshold_crossings` | int16 `(n_bins, n_ch)`, 10 ms bins / 100 Hz, −3.5×SD counts |
| Spike-band power | `.mat` `binned_neural_spike_band_power` | float32 `(n_bins, n_ch)`, 10 ms bins, sum-sq voltage |
| Bin NSP timestamps | `.mat` `binned_neural_nsp_timestamp` | float64 `(n_bins,)`, ns |
| Cue (label) | `.mat` `cue` | char `(n_trials,)`, whitespace-padded language unit |
| Trial timing | `.mat` `*_nsp_neural_time` | float64 `(n_trials,)`, ns (trial_start / go_cue / trial_end) |
| Microphone audio | `.mat` `microphone_data` (+ `microphone_nsp_time`) | int16 `(n_samples, 1)` @ **30 kHz** |
| Online decoder (closed loop only) | `.mat` `decoder_logit_output`, `decoder_output_redis_clock`, `ngram_decoder_partial_output`, `ngram_decoder_final_output` | float32 `(n_rows, 41)` logits, ms clock, LM strings per row. Not emitted, see [Online decoder configuration](#online-decoder-configuration) |
| Decoder normalization (closed loop only) | `.mat` `norm_channel_means`, `norm_channel_stds`, `norm_redis_times` | float32 `(n_updates, 128, 256 or 294)`. Not emitted, see [Online decoder configuration](#online-decoder-configuration) |

- **`n_ch` is per session:** **384** (all six arrays) on 11 sessions, **256** (the 2×6v + 2×55b speech-decoder subset, 6d dropped at acquisition) on **2024.04.15 / 04.16 / 04.17**. Read it from the file, never hardcode (`utils.n_channels`).
- **Channel to array map, Study order:** 6v = cols 0-127, 55b = 128-255, 6d = 256-383 (`utils.AREA_CHANNELS`, `utils.ARRAY_CHANNELS` per array). The 256-channel sessions hold the first 256. The Study permutes each file into this order (`utils.to_study_order`), the file orders are in [Channel order](#channel-order).
- Features are **raw** (not z-scored). What the authors' training pipeline does to them is in [Downstream preprocessing](#downstream-preprocessing-in-the-authors-code). Audio is recorded but T17 is anarthric (effort/breathing, not speech).

## Source Data Format & Layout

`Data.zip` expands to a `t17/` tree, one `.mat` per session block, with a single-trial feature-preview `.png` alongside each:

```
<NEURALSHED_DATA_ROOT>/jude_2026_speech/data/Jude2026IntendedSpeech/    <- study subfolder (= self.path)
    README.md                                  Dryad description + block tables
    dryad_jude_2026_speech_success_download.txt       download marker (what was verified, and when)
    t17/
        t17.2024.03.25/brainToText/RedisMat/
            20240325_133029_(5).mat            block 5 (one block per file)
            20240325_133036_(5).png
            ...
        ...                                    14 sessions, 140 .mat blocks, ~38.6 GB
```

Per-block `.mat` fields are documented in `utils.py`'s module docstring and in the Dryad `README.md`. Block files range from 8 MB (the aborted block) to 1.14 GB (`05.23` block 5), 38.6 GB in total. `04.02` block 13 is named `20250218_112835_(13).mat`, a 2025 re-export of the 2024-04-02 recording (its internal session name and timestamps are 2024-04-02).

## Study, timelines and Events

`study.py` (`Jude2026IntendedSpeech`). Each `.mat` block is a *continuous* recording with many cued trials inside, so **one block = one timeline** (121 catalogued task blocks, the 19 files in `utils.EXCLUDED_BLOCKS` skipped, and the Study raises on a file in neither table). `task_family`, `vocab` (`50w`, `125k` or `na`), `loop_type` (`open` or `closed`), `date` and `post_implant_day` (from the participant registry, implant 2024-02-29) ride the timeline and broadcast to every row. Select neural vs cue by event **type**, slice conditions by attribute (`task_condition`, `phase`, `trial_id`, ...) at query time. Event windows use neural-**bin indices** (the nearest bin to each NSP marker) so cues align exactly to the 100 Hz feature grid.

Every trial emits three rows with the block-local 0-based `trial_id`: the visual cue as a `Stimulus` over the instructed delay (trial start to go, 2.5 s for isolated cues, 3.4-9.8 s for sentences), a one-bin `Stimulus` go marker, and the cued unit over the go window (go to trial end). The unit carries `success` (the source's `trial_paused_by_CNRA` and `trial_timed_out` flags, both zero in every released trial) and the screen cue as `description`. `code` on `Action` and `Stimulus` rows is the dataset-wide class id in `utils.CUE_CODES`: 1-39 the target phoneme (its id in `PhonemeVocab.canonical()`), 40 rest, 41-52 the orofacial movements, 53-56 the four syllables, 101-150 the 50-word vocabulary. Sentence cues have no class and keep -100. Two trials the recording cuts short (`03.25` block 8 trial 79, 0.01 s of the intended 2 s, and block 14 trial 33, 0.66 s) carry `valid=False` and the reason in `valid_description`.

| Event type | Source | Shape / rate | Notes |
|---|---|---|---|
| `SpikeCounts` (threshold crossings) | `binned_neural_threshold_crossings` | `(n_ch, T)` @ 100 Hz, whole block | n_ch 384 or 256 per session, channels in Study order, SpikeCounts first so `cat` → native order |
| `BandPower` (spike-band power) | `binned_neural_spike_band_power` | `(n_ch, T)` @ 100 Hz, whole block | channels in Study order, concatenate w/ TX (`aggregation="cat"`) for the decoder input |
| `Stimulus` (cue) | `cue`, trial start and go markers | per trial, over the delay | `modality="visual"`, `stimulus_kind` `phoneme_cue` / `word_cue` / `sentence_cue` / `movement_cue` / `syllable_cue` / `rest_cue`, cue as `description`, class id as `code` |
| `Stimulus` (go marker) | go marker | per trial, one bin | `stimulus_kind` `speech_go_cue` or `movement_go_cue` |
| `Sentence` / `Word` / `Phoneme` (unit) | `cue` | per trial, go to trial end | type by task family, `text` (the ARPAbet target for phonemes), `phonemes` (the ARPAbet target sequence, see Phoneme targets), `language="en"`, `modality="read"`, `task_condition="attempted"`, `trial_id`, `success` |
| `Action` (orofacial unit, rest) | `cue` | per trial, go to trial end | `action_kind` `orofacial_movement` / `speech` (the "Say" syllables) / `rest` (both rest cues, `task_condition="na"`), `code` |
| `Intertrial` | trial end and next trial start, `intertrial_start` | closed-loop gaps, phased pre-cue window | 2-40 s in closed-loop blocks (decoder output and text-to-speech playback), 2 s before each phased cue. Open-loop trials abut, no rows |
| phased trial | `cue` + phase markers | per trial | `Intertrial`, the unit over the `read` phase, then a `speech_delay_cue` `Stimulus` (2 s), a go marker and the unit for the `imagined` and `attempted` phases, on `task_condition` and `phase` (2024.05.23). The imagined unit carries `modality="imagined"` |

Reuses the package `SpikeCounts`/`BandPower`/`Intertrial` events + `NeuralFeatureExtractor`, and the built-in neuralset `Sentence`/`Word`/`Phoneme`/`Stimulus`/`Action`, with no new event or extractor classes. The neural rows read `n_features` from the file and declare the acquisition chain in `processing` (250 Hz-5 kHz Butterworth, per-array linear regression referencing fit on the session's reference block, -3.5 x SD thresholds, spike-band power as the sum of squared voltages per bin, capped at 50,000 uV^2). The 30 kHz mic audio is anarthric and surfaced via `utils.get_audio`, not as a decode-target event.

**Phoneme targets.** Every `Sentence`, `Word` and `Phoneme` row carries `phonemes`, a space-separated ARPAbet sequence in `PhonemeVocab.canonical()` (39 symbols, `|` as the word boundary at id 40, blank 0), the same space and layout as `card_2024`'s `phonemes`, so `PhonemeEncoder` and a CTC head apply to both datasets unchanged. For sentences and words it is the prompt's dictionary pronunciation, generated once with the authors' `g2p_en` recipe (`mat_to_tfrecord.py`: punctuation stripped, stress removed, a boundary at every space and at the end, their `SIL` written as `|`) and shipped in `assets/cue_phonemes.json` (3,141 cues, `g2p_en` 2.1.0). It is not a measurement of what T17 produced, and the source has no per-phoneme timing, so it is a sequence target for CTC-style losses only. `utils.CUE_PHONEMES` (a `text.PhonemeTable`) reads the table and raises on a cue it lacks, `utils.build_cue_phonemes()` regenerates it through `text.PhonemeTable.build` and needs the `g2p` extra (`uv pip install -e ".[g2p]"`), which nothing else in the package uses. The recipe and the table format are shared package code (`neuralshed.text`), so any dataset whose source gives only the prompt text can ship the same kind of table. For the isolated phoneme blocks `phonemes` equals `text`, the authors' own cue map.

## Notes / Scope

- **Download:** `Study.download()` (`_download(overwrite=False)`) fetches `Data.zip` and `README.md` from Dryad version 8 (`utils.DRYAD_FILES`, ~19 GB, `NEURALFETCH_DRYAD_TOKEN` required for the transfer) through the shared backend `datasets/dryad.py` and writes `dryad_jude_2026_speech_success_download.txt` next to `t17/`. Without `overwrite`, a tree that passes `utils.verify_tree` (the README digest against the public Dryad listing, the 140 block files of `SESSION_BLOCKS` and `EXCLUDED_BLOCKS` present in their 14 sessions, every file but the catalogued unreadable one opening with `whosmat`) is accepted as downloaded and the marker records the check and its date. `utils.py` reads `.mat` blocks with `scipy.io.loadmat` (use the `fields=` argument to avoid loading the large neural/audio arrays).
- **Per-session channel count (256 vs 384)** is the main gotcha, see Data Modalities. It is 384 on 11 sessions and 256 (6v + 55b only) on 2024.04.15/16/17. Read it from each file, never hardcode.
- **Task type comes from the Dryad block tables, not `block_description`** (a generic, misleading string in every file). Encoded in `utils.SESSION_BLOCKS`.
- **Known corrupt block:** `t17.2024.05.23` block 7 (`20240523_184141_(7).mat`, 480 MB) is truncated/unreadable, listed in `utils.EXCLUDED_BLOCKS`.
- **Excluded trials.** The per-trial flags `trial_paused_by_CNRA` and `trial_timed_out` mark the trials the companion pipeline drops.
- **Microphone audio is not speech.** Audio is present, but the participant is anarthric with no intelligible vocalization, so the channel captures effort and breathing rather than speech. It is not a usable acoustic target.
- The 5-gram language model of the WER analysis is not part of Dryad version 8, the deposit README links it separately. Out of scope here.

### Online decoder configuration

Three different things are called "the decoder's channels" in the sources, and only the first is in the files:

- **What ran online.** Each closed-loop block stores the normalization vectors the online decoder used (`norm_channel_means`, one row per update, one column per feature). Their width, matched against the per-column means of the block's own neural data (correlations 0.97-1.00), gives the electrodes the decoder ran on that day. Threshold crossings and spike-band power are stacked, so the width is twice the electrode count.

| period | normalization width | electrodes | file columns (Study columns) |
|---|---:|---:|---|
| `03.26` block 27 | 128 | 64 | early 192-255 (`ventral_6v`, Study 0-63) |
| `04.02` | 256 | 128 | early 128-255 (area 6v, Study 0-127) |
| `04.17` through `05.14` | 256 | 128 | 0-127 (area 6v) |
| `05.28` and August | 294 | 147 | 0-127 plus 224-242 (area 6v plus a 19-electrode subset of `dorsal_55b`) |

- **The paper's offline analyses** use other sets: the 6v pair, 6v plus 55b (256 electrodes, the "512 features" of the deposit README, which no released closed-loop block used) and all 384, see the figure notebooks in the reference code.
- **The raw columns** are 384 or 256 per session (Data Modalities), independent of both.

The logits themselves (`decoder_logit_output`, 41 per row) are written every few milliseconds with a non-zero row every 80 ms (the paper's cadence, the deposit README says 20 ms). They are not emitted: the reference code carries two incompatible class orderings (`mat_to_tfrecord.py` uses blank, AA ... ZH, SIL, `Online-Speech-Performance.ipynb` reads the stored logits as blank, SIL, AA ... ZH), so the map would have to be verified against the data first.

### Downstream preprocessing in the authors' code

The Study emits the features as the files hold them. The reference pipeline (`utils/mat_to_tfrecord.py`, `utils/brainToText_trainDecoder.py`) then, per trial: slices from 4.5 s before the go cue (`start_offset` -450 bins for T17, -250 or 0 in the figure notebooks) to the trial end, caps spike-band power at 10,000 (`spikePowerMax`, below the 50,000 cap already in the files), optionally compresses bins by a factor (2 in the offline analyses, summing crossings and averaging power), subtracts each block's per-feature mean and divides by one global standard deviation (`globalStd=True`), and at training time adds white noise, a constant offset and Gaussian temporal smoothing (`gaussSmooth`). None of this is in the rows. `processing` on the neural rows describes the acquisition chain only.

### Channel order

Two column orders appear in this dataset:

- **File order** is the order of the columns in the source `.mat` matrices. It changed between 2024-04-02 and 2024-04-15 and stayed fixed afterwards. The three early sessions (`03.25`, `03.26`, `04.02`) store the arrays as 6d, 6v, 55b. Every later session stores them as 6v, 55b, 6d, and the 256-channel sessions store the first four of those groups (6d dropped at acquisition). Inside each area the two arrays also swap between the two orders.
- **Study order** is the order of the columns in the neural rows the Study emits: one contiguous block of 64 per array, `ventral_6v, dorsal_6v, ventral_55b, dorsal_55b, ventral_6d, dorsal_6d`, which is the late file order and the order of the authors' phased-task notebooks. `_load_features` converts file order to Study order with `utils.to_study_order`, so a column is the same electrode in every timeline.

| array | Study columns | early file columns (`03.25`, `03.26`, `04.02`) | late file columns (`05.06` on) | 256-channel file columns |
| --- | --- | --- | --- | --- |
| `ventral_6v` | 0-63 | 192-255 | 0-63 | 0-63 |
| `dorsal_6v` | 64-127 | 128-191 | 64-127 | 64-127 |
| `ventral_55b` | 128-191 | 320-383 | 128-191 | 128-191 |
| `dorsal_55b` | 192-255 | 256-319 | 192-255 | 192-255 |
| `ventral_6d` | 256-319 | 0-63 | 256-319 | absent |
| `dorsal_6d` | 320-383 | 64-127 | 320-383 | absent |

No source document states the column order, so it is established from the authors' code and from the data:

- The authors' Figure 1 notebooks (sessions `03.25`, `03.26`) slice `6d = 0-127, 6v = 128-255, 55b = 256-383`, with the early FVAF notebook titling columns 128-191 `v6v` and 192-255 `d6v`. Their phased-task notebooks (session `05.23`) slice `v6v = 0-63, d6v = 64-127, v55b = 128-191, d55b = 192-255` and title the six column groups `v6v, d6v, v55b, d55b, v6d, d6d`. This fixes the area of every 64-column group in each era.
- The online decoder's normalization vectors stored in the closed-loop blocks (`norm_channel_means`) match file columns 128-255 on `04.02` and file columns 0-127 on every later closed-loop day, the 128 electrodes of area 6v in each era.
- Channel-by-channel correlation matrices show the six arrays as 64-aligned blocks in every era (86-93% of channels correlate most with their own 64-column group). Matching the within-group correlation structure and the per-electrode median power of each early group to each late group, one to one, gives the permutation `early groups [0, 1, 2, 3, 4, 5] -> late groups [4, 5, 1, 0, 3, 2]` on 30 of 30 pairs between the six early blocks and the five May sessions tested, with identity within each era and for the 256-channel sessions against May. The August sessions, where spike-band power halves on five of the six arrays, agree on 6d and 55b and are ambiguous on the 6v pair. Inside a matched group, 23-59% of electrodes correlate best with themselves across eras (chance 1.6%), so the order of the 64 electrodes inside a group is preserved.

The area of each group is therefore established from the authors' code and confirmed by the data. The ventral/dorsal assignment of the two arrays inside 6v and inside 55b for the early sessions comes from the measured correspondence to the late files, where the authors' notebooks name them, and reverses the titles the authors' early FVAF notebook gives those groups. The Study follows the measurement.

### Source discrepancies

The files are the authority for the Study. Where the deposit README or the paper says otherwise:

| fact | source says | files hold | Study uses |
|---|---|---|---|
| trials per block | Dryad tables, 5,090 over 120 blocks | 5,064 in those blocks. Eight rows differ: `04.02` block 10 (40 vs 12) and 13 (10 vs 11), `04.17` blocks 7, 8, 10 (14, 11, 5 vs 15, 12, 6), `05.06` block 2 (13 vs 14) and 9 (40 vs 36), `05.14` block 8 (17 vs 18) | the files |
| phased short sentences | Dryad tables, block 3 only | blocks 3 and 6 | both |
| sentence sessions | paper Table S4, 3,640 sentences including Trial Days 94 and 95 | no session for days 94 or 95, 3,524 non-phased sentence trials | the files |
| August trial days | Dryad tables and Table S4, 164 and 165 | dates 2024-08-12 and 08-13, which are days 165 and 166 after the 2024-02-29 implant | 165 and 166 |
| `delay_duration_ms` | the delay before the go cue | 200 in the 15 August 50-word sentence blocks (`08.12`, `08.13`) while go cue minus trial start measures 5.2-6.3 s. The field and the timestamps agree in every other block, 4,500 in the March and April 50-word blocks | the NSP timestamps |
| RNN output cadence | Dryad README, every 20 ms | non-zero logit rows every 80 ms | not emitted |
| `norm_channel_means` width | Dryad README, 512 features | 128, 256 or 294 per block, the 64, 128 or 147 electrodes the online decoder used on that day | not emitted |

## Files

- `__init__.py`: package marker.
- `metadata.py`: `DATASET_METADATA` (catalog, source-only streams with no event class) + `sessions()` (trial counts and task condition per session) + `get_metadata()` (DB-validated).
- `utils.py`: `.mat` readers + accessors (`list_sessions`, `list_blocks`, `task_of`, `exclusion_reason`, `get_neural`, `get_trials`, `get_audio`, `phase_windows`, `to_study_order`, `array_channels`, `area_channels`) and constants (rates, `ARRAY_ORDER`, `ARRAY_CHANNELS`, `AREA_CHANNELS`, `FILE_GROUPS`, `SESSION_BLOCKS`, `EXCLUDED_BLOCKS`, `PHONEME_CUES`, `OROFACIAL_CUES`, `SAY_CUES`, `WORD_CUES`, `CUE_CODES`, `PHONE_DEF_SIL`, `CUE_PHONEMES`) plus `post_implant_day`, `cue_code` and `build_cue_phonemes`.
- `assets/`: the experiment overview used by notebook 01, and `cue_phonemes.json`, the cue-to-phonemes table of the sentence and word cues (see Phoneme targets).
- `study.py`: `Jude2026IntendedSpeech` neuralset Study (one timeline per block).
- `01_explore_source_data.ipynb`: local source-data demo.
- `02_explore_study.ipynb`: Study + extractor read-back demo.
- `03_model_example.ipynb`: sentence-level brain-to-text on the 125k task (`RaggedSegmenter` + `NeuralFeatureExtractor` + `PhonemeEncoder`, CTC decoder over the 6v and 55b columns, session 2024.05.07 held out whole, PER before a language model).
