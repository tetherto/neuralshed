# Card et al. 2024: An accurate and rapidly calibrating speech neuroprosthesis

Intracortical brain-to-text speech neuroprosthesis in one BrainGate2 participant (**T15**, a 45-year-old left-handed man with ALS, tetraparesis and severe dysarthria). Four 64-channel Utah arrays (256 electrodes) in the left precentral gyrus record 20 ms-binned threshold crossings and spike-band power while he attempts to speak cued sentences (the "copy task"). The aim is to decode attempted speech into phonemes and text. This is the source dataset for the **Brain-to-Text '25** challenge.

**Working dataset = the `t15_copyTask` neural data** (45 sessions, 10,948 trials, 20 ms-binned neural features plus phoneme and sentence labels). Same participant **T15** and the same four arrays as [`wairagkar_2025`](../wairagkar_2025/README.md) (brain-to-voice) and the T15 part of [`kunz_2025`](../kunz_2025/README.md), and 4,925 trials are also in `wairagkar_2025` and 200 in `kunz_2025` (see [Trials shared with other datasets](#trials-shared-with-other-datasets)).

## References

- **Paper:** Card, N. S., Wairagkar, M., Iacobacci, C., Hou, X., Singer-Clark, T., Willett, F. R., Kunz, E. M., et al. *An accurate and rapidly calibrating speech neuroprosthesis.* **N Engl J Med** 391(7), 609-618 (2024). https://doi.org/10.1056/NEJMoa2314132
- **Source data:** [10.5061/dryad.dncjsxm85](https://datadryad.org/dataset/doi:10.5061/dryad.dncjsxm85)
- **Code repository:** https://github.com/Neuroprosthetics-Lab/nejm-brain-to-text

## Species & Subjects

- Human. Single BrainGate2 clinical-trial participant **T15**, registered as `braingate2/T15` in `datasets/participants.py`: 45-year-old left-handed man with amyotrophic lateral sclerosis (ALS), tetraparesis (no functional use of the limbs at enrollment) and severe dysarthria (unintelligible to non-expert listeners). Data are de-identified.
- **Implant:** four 64-electrode Utah arrays (**256 electrodes total**) placed in the **left precentral gyrus** on `2023-07-17` (implant date in the participant registry, data collection starts 25 days later). Two percutaneous pedestals, each wired to two arrays. Array centers in MNI space are in `metadata.py` (appendix S1.12).
- **Channel order.** The 256-wide channel axis is ordered by array: `ventral 6v, area 4, 55b, dorsal 6v` (64 each). "area 4" is primary motor cortex. Block `i` of 64 channels is the same array in `card_2024`, `wairagkar_2025` and `kunz_2025`, but no source states an electrode-level mapping, and the streams are binned differently (50 Hz here, 100 Hz in `wairagkar_2025`).

## Behavioral Task

**Instructed-delay copy task** (appendix S1.07): the prompted sentence appears as text on a monitor, a square turns from red to green as the go cue, and T15 attempts to speak it (vocalized in most blocks, silent in some). He ends the trial with an on-screen eye-tracker button (in early sessions by eye contact with a researcher, who pressed a button). Most blocks are 50 trials and last 15-25 minutes. Sentences come from a 50-word vocabulary set (the first two sessions), the Switchboard corpus, OpenWebText2 and the Harvard Sentences (appendix S1.10).

45 sessions between 2023-08-11 and 2025-04-13 (post-implant days 25-636). The release is a separate curation from the paper: the paper reports 84 sessions on days 25-288 and 10,481 prompted sentences over 50 of them (Table S2, S1.10), of which 28 days are also in the release, with per-session counts that differ on 26 of those days by a few trials to a few dozen. The remaining 17 released sessions are later than the paper. The release is the authoritative object, and Table S2 cannot be reconciled with it trial for trial.

Not in the release: the per-session "diagnostic block" of 8 single-word cues used to fit thresholds and referencing, the speaking strategy of each block (intentionally withheld by the authors), and all Conversation Mode data (`t15_personalUse.pkl` holds only per-sentence ratings, no neural data, and is excluded here).

## Data Modalities

| Modality | Source | Rate | Per-trial shape |
|---|---|---|---|
| Threshold crossings (-4.5 x RMS) | `input_features` cols 0-255 | 50 Hz (20 ms bins) | `(n_time_steps, 256)` float32, z-scored |
| Spike-band power (250-5000 Hz) | `input_features` cols 256-511 | 50 Hz (20 ms bins) | `(n_time_steps, 256)` float32, z-scored |
| Phoneme labels | `seq_class_ids` | per trial | `(500,)` int32, 0-padded (train/val only) |
| Sentence text | `transcription` + `sentence_label` attr | per trial | ASCII codes / string (train/val only) |

- **Neural features** (appendix S1.04, S2.01). 30 kHz signal band-passed 250-5000 Hz, linear regression referencing per 64-electrode array, threshold crossings at -4.5 x RMS and spike-band power clipped at 12,500 µV² at the 1 ms stage, then binned to 20 ms (counts summed, power averaged). Thresholds and referencing weights are refit after every block. Values are **z-scored within session** with statistics from the speech epochs of the preceding 20 trials, then clipped at +10 SD (the max is exactly 10.0 in every session). The 40 ms Gaussian smoothing in the paper is applied at training time in the authors' code, so the released features are unsmoothed.
- **Labels.** `seq_class_ids`: `0` = blank/pad, `1-39` = 39 ARPAbet phonemes, `40` = word boundary `|` (see `PHONEME_LABELS` in `utils.py`, verified against the repo). `transcription` is the cue as 0-padded ASCII. **`test`-split trials carry only `input_features`** (labels reserved for the Brain-to-Text '25 challenge).
- **No within-trial timing.** Each trial is a pre-cropped clip that starts about 1 s before the go cue (see [Trials shared with other datasets](#trials-shared-with-other-datasets)). The release has no cue, delay, go or inter-trial timestamps and no per-phoneme or per-word alignment.

## Source Data Format & Layout

Per-session HDF5 (`h5py`).

```
<NEURALSHED_DATA_ROOT>/card_2024/
├── data/Card2024Brain2Text/                  # <- the Study path (utils.DATA_DIR)
│   ├── README.md                             deposit README
│   ├── t15_copyTask.pkl                      online-decoding metrics (Fig. 2), not used here
│   └── hdf5_data_final/                      # <- utils.sessions_dir() (the zip's wrapper dir)
│       └── t15.YYYY.MM.DD/                   45 session folders
│           ├── data_train.hdf5               always present
│           ├── data_val.hdf5                 41/45 sessions
│           └── data_test.hdf5                41/45 sessions (no labels)
└── references/                               # paper, appendix, protocol, nejm-brain-to-text repo
```

- **45 sessions**, **10,948 trials**: 8,072 train / 1,426 val / 1,450 test. Four sessions are train-only: `2023.08.11`, `2024.03.03`, `2024.04.25`, `2024.04.28`. Where a block has val and test trials they are both cut from the same block by `trial_num`.
- **Per-trial HDF5 group** (`trial_0000`, `trial_0001`, ...). Group position inside a split file is not a trial identity: `block_num` + `trial_num` is, unique within a session.

| dataset / attr | type / shape | notes |
|---|---|---|
| `input_features` | float32 `(n_time_steps, 512)` | neural features, 20 ms bins |
| `seq_class_ids` | int32 `(500,)` | phoneme ids 1-40, 0-padded (train/val) |
| `transcription` | int32 `(500,)` | ASCII codes of the sentence, 0-padded (train/val) |
| attr `sentence_label` | str | cue sentence text (train/val) |
| attr `session`, `block_num`, `trial_num`, `n_time_steps` | int/str | always present |
| attr `seq_len` | int | valid length of `seq_class_ids` (train/val) |

- **`t15_copyTask.pkl`** holds the authors' online-decoding results for 1,718 closed-loop evaluation trials (the paper's Figure 2): cue, decoded sentence, RNN logits, speech duration. No neural data. It is downloaded with the deposit but nothing here reads it.

## Study, timelines and Events

`study.py` defines **`Card2024Brain2Text`**: **one timeline per trial** (10,948 across 45 sessions), each an independent copy-task clip. The timeline dict broadcasts to every event row:

| Column | Value |
|---|---|
| `subject` | `T15` |
| `session` | `t15.YYYY.MM.DD` |
| `date`, `post_implant_day` | from the session name and the registry implant date |
| `original_split` | `train`, `val` or `test` (the authors' split) |
| `block` | source `block_num` |
| `trial_id` | source `trial_num`, unique within `(session, block)` |
| `task_condition` | `attempted` |

Per timeline:

| Event type | Source | Shape / rate | Notes |
|---|---|---|---|
| `SpikeCounts` | `input_features` cols 0-255 | `(256, n_bins)` @ 50 Hz | `method="threshold_crossing"`, `threshold="-4.5 x RMS"`, listed first |
| `BandPower` | `input_features` cols 256-511 | `(256, n_bins)` @ 50 Hz | `lower_freq=250`, `upper_freq=5000`, unit `uV^2` before normalization |
| `Sentence` | `sentence_label` + `seq_class_ids` | text, whole trial | orthographic `text` + ARPAbet `phonemes` target |

Both neural rows carry `normalization="z_scored"` and a `processing` string with the full chain (referencing, clipping, z-scoring window, +10 SD clip). To reproduce the source's 512-row layout (threshold crossings on top), `NeuralFeatureExtractor(target_unit="native", aggregation="cat")` stacks the two streams, which have different native units. Because the values are already z-scored, `target_unit="Hz"` does not apply.

The phoneme target rides the `Sentence` because the source ships no per-phoneme or per-word timing. `extractors.PhonemeEncoder` turns it into CTC ids with `text.PhonemeVocab` (41 classes, ids matching `seq_class_ids`). **`test`-split timelines carry neural events only.** All event and extractor types come from the package, so there is no card-specific Event or Extractor.

### Trials shared with other datasets

**`wairagkar_2025`.** [`wairagkar_2025`](../wairagkar_2025/README.md) releases many of the same T15 trials, binned at 10 ms, log-transformed, normalized over a rolling 10 s window and smoothed. The two releases share 25 session dates (2023-08-11 to 2023-12-29, post-implant days 25-165). On those days 4,925 trials of this dataset are the same recording as a `wairagkar_2025` trial: 4,100 `train`, 757 `val` and 68 `test`. Each pair has the same cue, and its threshold crossings, with the `wairagkar_2025` clip rebinned to 20 ms, correlate at 0.81-0.90 over time and channels, against at most 0.2 for every other trial of the day. The `wairagkar_2025` clip starts 0.96-0.98 s after the clip here and ends 0.05-0.15 s before it (5th-95th percentile). On the shared days, 1,650 trials here and 873 there have no counterpart, and the other 20 sessions here and 12 there are not shared.

- 68 `test` trials, whose sentence this release withholds, carry their cue in `wairagkar_2025`.
- The 95 trials of the first `wairagkar_2025` evaluation session (day 110, `Evaluation trial (preliminary)`) are `train` (80) and `val` (15) trials here.

**`kunz_2025`.** The 200 trials of the [`kunz_2025`](../kunz_2025/README.md) file `t15.2024.03.03_attempted_50words` are 200 of the 219 `train` trials of session `t15.2024.03.03` here. Each pair has the same cue, and its threshold crossings correlate at 0.97 or more with the `kunz_2025` recording rebinned to 20 ms, against at most 0.12 for other trials. The imagined-speech blocks of that day are not in this release.

**Clip start.** `kunz_2025` releases the go epochs, and each of the 200 clips above starts 0.98-1.02 s before the go cue (5th-95th percentile), which agrees with the `wairagkar_2025` go-period clips starting about 0.97 s after the clips here. A clip here therefore holds about 1 s of the delay period before the attempt.

A collection that pools these datasets holds the shared trials twice. Keep one copy of each pair, or assign both copies to the same split.

## Notes / Scope

- **Copy task only.** Conversation Mode and the pretrained RNN baseline are excluded, the baseline being optional for onboarding and needed by no artifact here.
- **Download** via `Study.download()` (the shared `datasets/dryad.py` `Dryad`): fetches the deposit README, `t15_copyTask_neuralData.zip` and `t15_copyTask.pkl`, skipping the personal-use pickle and the pretrained baseline. Needs a Dryad API token in `NEURALFETCH_DRYAD_TOKEN` (the notebooks read it from a `.env` file).
- **Source path** is `dataset_dir("card_2024") / "data"`, which the Study resolves to `data/Card2024Brain2Text/`. Artifacts read the raw source directly (`h5py`).
- **Features arrive already normalized.** `input_features` are z-scored within session, so they are not on the raw count or µV² scale that most sibling datasets use, and the rolling 20-trial window crosses any train/val boundary and cannot be undone. Anything pooling this dataset with others has to account for that.
- **Test-split labels are held out.** Test trials carry only `input_features`, reserved for the Brain-to-Text '25 challenge. They remain usable as label-free input.

## Files in this folder

- `__init__.py`: package marker.
- `metadata.py`: catalog `DATASET_METADATA` (participant, arrays with MNI coordinates, brain-area ontology ids, events) + `sessions()` + `get_metadata()`.
- `utils.py`: `SessionRef` / `TrialRef` + source readers (`list_sessions`, `list_trials`, `read_trial`, `read_input_features`, `read_labels`, `session_summary`), feature layout and label decoding. No NWB dependencies.
- `study.py`: the neuralset `Study` (`Card2024Brain2Text`), one timeline per trial.
- `assets/`: the experiment illustration used by notebook 01.
- `01_explore_source_data.ipynb`: executed walkthrough of reading and visualizing the source data.
- `02_explore_study.ipynb`: build the Study, browse the events, read each event type through its extractor.
- `03_model_example.ipynb`: train a small CTC brain-to-text decoder (neural to phonemes) on the authors' train split and report phoneme error rate on their val split.
