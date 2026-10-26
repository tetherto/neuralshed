# Wairagkar et al. 2025: An instantaneous voice-synthesis neuroprosthesis

Intracortical brain-to-voice speech neuroprosthesis in one BrainGate2 participant (**T15**, a 45-year-old left-handed man with ALS and severe dysarthria). Four 64-channel Utah arrays (256 electrodes) in the left precentral gyrus record 10 ms-binned spike-band power and threshold crossings while he attempts to speak cued sentences at his own pace. A causal decoder maps the neural features to 20-dimensional LPCNet acoustic features that a vocoder turns into a synthesized voice with about 8.5 ms of latency. Because T15 cannot produce intelligible speech, the training target is a text-to-speech rendering of the cue, time-aligned to his attempt.

**Working dataset = `t15_neural_data`** (37 sessions, 8,489 trials, open vocabulary, 10 ms-binned neural features plus the time-aligned LPCNet target and word timings). Same participant **T15** and the same four arrays as [`card_2024`](../card_2024/README.md) (brain-to-text) and the T15 part of [`kunz_2025`](../kunz_2025/README.md). 4,925 trials are also in `card_2024` (see [Trials shared with `card_2024`](#trials-shared-with-card_2024)). The 50-word `t15_control_experiment` release is secondary, see [Notes / Scope](#notes--scope).

## References

- **Paper:** Wairagkar, M., Card, N. S., Singer-Clark, T., Hou, X., Iacobacci, C., Miller, L. M., Hochberg, L. R., Brandman, D. M., Stavisky, S. D. *An instantaneous voice-synthesis neuroprosthesis.* **Nature** 644, 145-153 (2025). https://doi.org/10.1038/s41586-025-09127-3
- **Source data:** [10.5061/dryad.2280gb64f](https://datadryad.org/dataset/doi:10.5061/dryad.2280gb64f)
- **Code repository:** https://github.com/Neuroprosthetics-Lab/brain-to-voice-2025 (clone under `references/`)

## Species & Subjects

- Human. Single BrainGate2 clinical-trial participant **T15**, registered as `braingate2/T15` in `datasets/participants.py`: 45-year-old left-handed man with amyotrophic lateral sclerosis (ALS), no functional use of the limbs, mixed upper- and lower-motor-neuron dysarthria, ALSFRS-R 23. He could vocalize but not produce intelligible speech, and spoke slowly, one word at a time. Data are de-identified.
- **Implant:** four 64-electrode Utah arrays (**256 electrodes total**) in the **left precentral gyrus**, implanted `2023-07-17` (implant date in the participant registry, data collection starts 25 days later). Placement was targeted from pre-operative scans with the Human Connectome Project parcellation: ventral 6v, area 4 (primary motor cortex), area 55b and dorsal 6v. The arrays are named `ventral_6v`, `4`, `55b`, `dorsal_6v`, the same names, labels, ontology ids and MNI centers as `card_2024`, which documents the same implant. The source files and the paper call them `v6v`, `M1`, `55b`, `d6v`.
- **Channel order.** The 256-wide channel axis is ordered by array: `ventral_6v, 4, 55b, dorsal_6v` (64 each). Block `i` of 64 channels is the same array in `card_2024` and `kunz_2025`, but no source states an electrode-level mapping, and the streams are binned differently (100 Hz here, 50 Hz in `card_2024`). The source's 512-wide decoder input is `[spike-band power | threshold crossings]`, the reverse of `card_2024`'s, and `aggregation="cat"` follows event order, so pooled reads of the two datasets do not line up channel for channel.

## Behavioral Task

Cued-speech task. Each trial starts with a **delay** period (1.5-4 s, red square and the text cue, prepare to speak) followed by a **go** period (green square) in which T15 attempts to speak the cue at his own pace, then ends the trial by looking at a "Done" icon with an eye tracker. In the evaluation sessions the decoder synthesized his voice in real time from the neural features and played it back through a speaker. **The release holds only the go period**, one pre-cropped clip per trial, with no delay, cue or inter-trial timing.

37 sessions between 2023-08-11 and 2024-01-28 (post-implant days 25-195), 10-432 trials per session, 8,235 unique sentences over about 3,500 words. The source labels every trial with one of seven `trial_description` values. The Study collapses them into the two-way `original_split`: `eval` when the label contains "Evaluation", `train` otherwise (7,273 train, 1,216 eval). Both ride the timeline.

| `trial_description` | trials | `original_split` | meaning |
|---|---|---|---|
| `Training data` | 6,502 | `train` | open-vocabulary cues used to train the decoders |
| `Training data (50-word vocab)` | 771 | `train` | closed-vocabulary cues, mostly the first sessions |
| `Evaluation trial` | 964 | `eval` | closed-loop trials in the 9 evaluation sessions |
| `Evaluation trial (preliminary)` | 110 | `eval` | the first evaluation session, day 110 |
| `Evaluation trial (mimed speech)` | 59 | `eval` | silent articulation without vocalizing |
| `Evaluation trial (free response)` | 52 | `eval` | T15 answered a question in his own words. The cue is a transcript of what he said, not a prompt |
| `Evaluation trial (spelling)` | 31 | `eval` | cues are letter sequences such as `q u i c k`, so their `Word` events are letters |

The 9 evaluation sessions (days 110, 172, 174, 179, 181, 186, 188, 193, 195) each ran a causal decoder trained on all earlier sessions. 44 cue strings occur in both the training and the evaluation trials. The paper reports 956 vocalized and 58 mimed evaluation sentences and 57 free-response trials against the release's 964, 59 and 52. The release is the authoritative object.

## Data Modalities

| Modality | Source | Rate | Per-trial shape |
|---|---|---|---|
| Spike-band power (250-5000 Hz) | `spikepow_trials` | 100 Hz (10 ms bins) | `[N x 256]` float32, normalized |
| Threshold crossings (-4.5 x RMS) | `threshcross_trials` | 100 Hz (10 ms bins) | `[N x 256]` float32, normalized |
| LPCNet acoustic target | `target_speech_feats_{word,syl}` | 100 Hz | `[N x 20]` float32, both variants |
| Speech timings | `word_indices`, `syl_indices` | 100 Hz bin indices | `[n x 2]` onset, offset |
| Cue text and label | `cues`, `trial_description` | per trial | string |
| Synthesized voice | `brain2voice_pred_audio_16kHz` | 16 kHz | `[N * 160]` float32 samples, evaluation trials only |

- **Neural features** (paper Methods, source readme). The 30 kHz signal is band-passed 250-5000 Hz and denoised by linear regression referencing, then per electrode and 1 ms step: threshold crossings at -4.5 x RMS and spike-band power clipped at 50,000 uV². Both are binned to 10 ms, log-transformed, normalized by a rolling 10 s mean and SD, and causally smoothed with a 1.5 s sigmoid kernel. The shipped values are dimensionless and slightly under unit variance. The Study declares `unit` as what fed the chain (`uV^2`, `counts/bin`), `normalization="z_scored"` and the rest in `processing`.
- **LPCNet target.** 20 features per 10 ms bin (18 Bark-frequency cepstral coefficients, pitch period, pitch correlation) in the original LPCNet range, extracted from a text-to-speech rendering of the cue and time-stretched onto T15's attempt. Two variants ship for every trial. **Word-aligned** stretches each TTS word as one block onto his word interval. **Syllable-aligned** stretches each syllable separately, so the rhythm inside a word follows his attempt. The paper's decoders trained on the syllable-aligned target on every session but the first (day 25 used word alignment because no prior neural data existed to derive syllable boundaries), so `lpcnet_syllable_aligned` reproduces the published target except on day 25. The two variants coincide exactly when every word in the cue is monosyllabic, and on all 216 trials of day 69, where the syllable variant is a copy of the word variant.
- **Timing provenance.** Word and syllable onsets were measured from a microphone on day 25 only. On every later session they were inferred by segmenting the envelope of a previous decoder's synthesized output, with manual review. The `text` of every `Word` event is the cue word and is exact. Its `start` and `duration` are measured on day 25 and are decoder-derived estimates, reviewed by hand, on the other 36 sessions. 12 syllable rows in the source have equal onset and offset. Syllable timings are not emitted.
- **Synthesized voice.** Only the evaluation trials carry the closed-loop audio, sample-aligned to the neural bins (160 samples per bin). It is the online system's output, not a target, and is not emitted as an event.

## Source Data Format & Layout

Per-session MATLAB 5 `.mat` files (`scipy.io.loadmat`).

```
<NEURALSHED_DATA_ROOT>/wairagkar_2025/
├── data/Wairagkar2025Brain2Voice/                # <- the Study path
│   ├── README.md                                 Dryad README
│   ├── t15_neural_data/                          # <- utils.DEFAULT_DATA_DIR, 30 GB
│   │   ├── t15_neural_data_readme.txt
│   │   ├── test_idx.mat                          listener-test trial indices
│   │   └── t15_dayNNN/                           37 session folders
│   │       ├── t15_dayNNN_neural_data.mat        neural features, cues, trial_description
│   │       ├── t15_dayNNN_speech_indices.mat     word_indices, syl_indices
│   │       ├── t15_dayNNN_target_speech_feats.mat  LPCNet target, both variants
│   │       └── brain2voice_t15_dayMMM.h5         causal decoder, 9 evaluation sessions
│   └── t15_control_experiment/                   secondary, 1.1 GB
├── references/                                   # paper, code repo
└── cache/                                        # extractor disk cache, created on demand
```

- **Per-session variables.** `spikepow_trials` and `threshcross_trials` are cell arrays of `[N x 256]` matrices, one per trial, with `N` the number of 10 ms bins. `cues` and `trial_description` are string arrays. `neural_bin_size_ms` is 10 in every file. The evaluation sessions add `brain2voice_pred_audio_16kHz` and `brain2voice_decoder`, the name of the `.h5` decoder used in that session (trained through the previous session).
- **Trial identity** is the position in the session file, unique within a session. `test_idx.mat` indexes trials globally, 0-based over all sessions in day order, which the Study resolves into the `benchmark` column.
- **File sizes.** The neural file of a session is 31 MB to 1.36 GB, the target file 50-90 MB, the indices file about 100 kB.

## Study, timelines and Events

`study.py` defines **`Wairagkar2025Brain2Voice`**: **one timeline per trial** (8,489 across 37 sessions), each an independent go-period clip. The timeline dict broadcasts to every event row:

| Column | Value |
|---|---|
| `subject` | `T15` |
| `session` | `t15_dayNNN` |
| `date`, `post_implant_day` | from the day number and the registry implant date |
| `original_split` | `train` or `eval`, collapsed from `trial_description` (table above) |
| `trial_description` | the source's verbatim label, one of the seven values above |
| `task_condition` | `attempted`, or `mimed` on the 59 mimed trials |
| `benchmark` | `random`, `video` or empty, see below |
| `trial_id` | position in the session file, unique within a session |

Per timeline:

| Event type | Source | Shape / rate | Notes |
|---|---|---|---|
| `BandPower` | `spikepow_trials` | `(256, n_bins)` @ 100 Hz | `feature="spike_band_power"`, `lower_freq=250`, `upper_freq=5000`, unit `uV^2`, listed first |
| `SpikeCounts` | `threshcross_trials` | `(256, n_bins)` @ 100 Hz | `feature="threshold_crossings"`, `method="threshold_crossing"`, `threshold="-4.5 x RMS"`, unit `counts/bin` |
| `Sentence` | `cues` | text, whole trial | the cue |
| `Word` | `word_indices` | onset and duration | one per cue word, `text` and `sentence` |
| `AcousticFeatures` | `target_speech_feats_word` | `(20, n_bins)` @ 100 Hz | `feature="lpcnet_word_aligned"`, `normalization="raw"` |
| `AcousticFeatures` | `target_speech_feats_syl` | `(20, n_bins)` @ 100 Hz | `feature="lpcnet_syllable_aligned"`, `normalization="raw"` |

Both neural rows carry `normalization="z_scored"` and a `processing` string with the full chain. `NeuralFeatureExtractor(target_unit="native", aggregation="cat")` reproduces the source's 512-row decoder input. Because the values are already normalized, `target_unit="Hz"` leaves them untouched. The two acoustic rows cover the same span, so an extractor must pick one: `BehavioralSeriesExtractor(event_types="AcousticFeatures", query="feature == 'lpcnet_syllable_aligned'")`. Without a query the extractor raises a collision error. All event and extractor types come from the package, so there is no dataset-specific Event or Extractor.

**What `benchmark` is.** The two lists in `test_idx.mat` mark the closed-loop evaluation trials whose synthesized voice the authors put in front of human listeners: `random` (90 trials drawn from the evaluation trials for the open-transcription listener test) and `video` (38 clips from the paper's videos, scored additionally). It is a benchmark for the synthesized audio, not a neural test split. Those trials' neural features and targets are ordinary `Evaluation trial` rows, and `original_split == "eval"` is the held-out set to compare against the paper. `benchmark == "random"` matters only for repeating the listener test on the same sentences. One trial is in both lists and is tagged `random`, so `benchmark == "video"` selects 37.

### Trials shared with `card_2024`

[`card_2024`](../card_2024/README.md) releases many of the same T15 trials, binned at 20 ms and z-scored over the preceding 20 trials. The two releases share 25 session dates (days 25-165, 2023-08-11 to 2023-12-29). On those days 4,925 trials here are the same recording as a `card_2024` trial: 4,551 `Training data`, 279 `Training data (50-word vocab)` and 95 `Evaluation trial (preliminary)`. Each pair has the same cue, and its threshold crossings, rebinned to 20 ms, correlate at 0.81-0.90 over time and channels with the `card_2024` clip, against at most 0.2 for every other trial of the day. The clip here starts 0.96-0.98 s after the `card_2024` clip and ends 0.05-0.15 s before it (5th-95th percentile). On the shared days, 873 trials here and 1,650 there have no counterpart, and the other 12 sessions here and 20 there are not shared.

- A collection that pools the two datasets holds these trials twice. Keep one copy of each pair, or assign both copies to the same split.
- All 95 trials of day 110, the first evaluation session and part of `original_split == "eval"`, are `train` (80) and `val` (15) trials of `card_2024`. The other 8 evaluation sessions are not in `card_2024`.
- 68 trials here are `test` trials of `card_2024`, whose sentences that release withholds.

## Notes / Scope

- **`t15_neural_data` only.** The control experiment and the synthesized audio are not emitted, see below.
- **Download** via `Study.download()` (the shared `datasets/dryad.py` `Dryad`): fetches the Dryad README, `t15_neural_data.zip` and `t15_control_experiment.zip`, 31 GB extracted. Needs a Dryad API token in `NEURALFETCH_DRYAD_TOKEN` (the notebooks read it from a `.env` file).
- **Source path** is `dataset_dir("wairagkar_2025") / "data"`, which the Study resolves to `data/Wairagkar2025Brain2Voice/`. Artifacts read the raw source directly with `scipy.io.loadmat`.
- **Reads go by session file.** MATLAB 5 files have no random access, so every neural read loads the whole session file (up to 1.36 GB, two kept in memory). Reading timelines in order costs one load per session, and a full `run()` enumerates all 8,489 timelines in about a minute. Shuffled access across sessions is slow until the extractor disk cache is warm: run the segmenter's `prepare()` once in timeline order, then shuffle.
- **Features arrive already normalized.** The rolling 10 s statistics cannot be undone, and the values are not on the raw count or uV² scale that most sibling datasets use.
- **The target is synthesized, not recorded.** It is a well-defined regression target and is what the paper's decoders trained on, but it is not the participant's voice, and its timing after day 25 comes from a previous decoder.
- **Chronology.** The paper's decoders trained on every session before each evaluation day. `AssignSourceSplit(mapping={"train": "train", "eval": "val"})` honours the source partition.
- **Not emitted.** The closed-loop synthesized audio and the per-session `.h5` decoders are the online system's output, reachable through `utils.get_pred_audio` and `SessionRef.decoder_path`. The `t15_control_experiment` release is a single session (`t15_day482`, 10 blocks of 24 trials, 50-word vocabulary) that adds a stethoscopic microphone (30 kHz) and a 6-axis IMU (300 Hz) but ships no LPCNet target.

## Files in this folder

- `__init__.py`: package marker.
- `metadata.py`: catalog `DATASET_METADATA` (participant, arrays with MNI coordinates, brain-area ontology ids, events) + `sessions()` + `get_metadata()`.
- `utils.py`: `SessionRef` + source readers (`list_sessions`, `get_cues`, `get_trial_descriptions`, `get_neural_features`, `get_target_feats`, `get_word_indices`, `get_syllable_indices`, `get_pred_audio`, `load_test_idx`) and constants.
- `study.py`: the neuralset `Study` (`Wairagkar2025Brain2Voice`), one timeline per trial.
- `assets/`: the experiment illustration used by notebook 01.
- `01_explore_source_data.ipynb`: executed walkthrough of reading and visualizing the source data.
- `02_explore_study.ipynb`: build the Study, browse the events, read each event type through its extractor.
- `03_model_example.ipynb`: train a small neural-to-LPCNet decoder on one evaluation session, using the source's partition through `AssignSourceSplit` (its training trials to train, its closed-loop evaluation trials to validate), and report per-feature correlation.
