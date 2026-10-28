# Jude 2026: bimanual QWERTY-typing neuroprosthesis (T17 + T18)

Intracortical BCI study of bimanual typing in two BrainGate2 participants with tetraplegia, T17 (ALS, anarthric, locked-in) and T18 (C4 spinal-cord injury). Each has six 64-channel Utah arrays (384 electrodes) in precentral cortex. The files hold 10 ms-binned threshold crossings and spike-band power recorded while the participant attempts individual finger movements mapped to the keys of a QWERTY layout. Online decoding of those movements into text reached about 110 characters per minute (22 words per minute, 1.6% word error rate).

## References

- **Paper (Nature Neuroscience, 2026):** Jude, Levi-Aharoni, Acosta, Allcroft, Nicolas, Lacayo, Card, Wairagkar, Brandman, Stavisky, Willett, Williams, Simeral, Hochberg, Rubin. "Restoring rapid natural bimanual typing with a neuroprosthesis after paralysis." doi:[10.1038/s41593-026-02218-y](https://www.nature.com/articles/s41593-026-02218-y)
- **Source data (Dryad):** doi:[10.5061/dryad.cz8w9gjjk](https://datadryad.org/dataset/doi:10.5061/dryad.cz8w9gjjk)
- **Code repository:** <https://github.com/justin-jude/typingBCI>
- **Clinical trial:** BrainGate2, NCT00912041.

## Species & Subjects

- **Species:** human (*Homo sapiens*), both participants male.
- **T17** (`braingate2/T17` in `datasets/participants.py`, implant 2024-02-29): 33-year-old right-handed man with ALS, diagnosed about 3 years before. Tetraplegia, anarthria and ventilator dependence, with volitional control left only over the extraocular muscles (locked-in). Six 64-channel Utah arrays in the left precentral gyrus, two per area: ventral (6v), middle (55b) and dorsal (6d). A microphone is recorded in every block. The same implant is recorded in `jude_2026_speech` (days 25-166) and `kunz_2025` (day 284), on different dates.
- **T18** (`braingate2/T18`, implant 2024-10-21): 48-year-old right-handed man with tetraplegia from a C4 ASIA A cervical spinal-cord injury about 18 months before enrollment, with no residual volitional movement below C4. Six 64-channel Utah arrays, all in area 6d (the cortical hand knob): four in the left hemisphere and two in the right. No microphone.

## Behavioral Task

Every trial is built on one primitive, an attempted finger movement with an instructed delay:

1. Prepare: the target is shown as text in a red cue rectangle.
2. Go: the rectangle turns green (`go_cue_nsp_neural_time`).
3. The participant attempts the cued movement and holds the posture through the go period.
4. Return to rest before the next trial.

There are 30 movements, 10 fingers by 3 directions from rest (extend upwards, straight downwards, curl in to the palm), plus a "DO NOTHING" rest condition. In the typing tasks each finger owns a column of a 3-row by 10-column keyboard and the direction picks the row (paper Fig. 3a):

| finger | up | down | palm |
|---|---|---|---|
| left pinky | Q | A | Z |
| left ring | W | S | X |
| left middle | E | D | C |
| left index | R | F | V |
| left thumb | T | G | B |
| right thumb | Y | H | space |
| right index | U | J | N |
| right middle | I | K | M |
| right ring | O | L | , |
| right pinky | P | ? | . |

Task families (`utils.SESSION_BLOCKS`, from the Dryad block tables):

- **Isolated finger movements** (`finger_movement`): each trial cues one movement as text ("RIGHT INDEX - Extend Upwards", "DO NOTHING"), 31 conditions with 2 repetitions each per 62-trial block (`t18.2024.12.03` block 1 stops after 17 trials). The two T18 blocks `t18.2024.12.04` 2 and 3 cue the key instead ("M", "? (QUESTION MARK)", "SPACE", "DO_NOTHING"). `utils.CUE_CODES` maps both kinds of cue to one class id (0 rest, 1-30 the keys in keyboard order), and on T18 the neural responses agree: 28 of the 31 key cues best match the movement of their table key, the other 3 a movement of an adjacent finger.
- **Calibration sentences** (`calibration`): open-loop sentence copy. The participant attempts to type the cued sentence with no decoded output on screen. These blocks fit the decoder.
- **Sentence copy** (`sentence_copy`): closed-loop sentence copy. The decoder's most likely key is highlighted on the on-screen keyboard in real time.
- **Free typing** (`free_typing`): closed loop and uncued. The participant types any sentence, a trial ends after a run of blank predictions, and the `cue` holds the intended sentence as confirmed with the participant afterwards.
- **Handwriting** (`handwriting`, T18 only, session `t18.2025.04.01`): calibration sentences the participant attempts to handwrite. Cues encode a space as `>` and the end as `~`, which `utils.get_trials` removes. The blocks are open loop except block 13, two closed-loop trials.

The loop type comes from each file's `block_description` ("Runs a open loop ...", "Runs a closed loop ...", "Runs a personal use ..."), which `graph_name` and the stored decoder output agree with (see [Source discrepancies](#source-discrepancies)).

### Sessions

Trial counts per task block are from the files (`utils.get_trials`), the same numbers `metadata.sessions()` reports. Run order lists the task blocks in block-number order, which is also the order of the file timestamps.

| session | date | post-implant day | task blocks | trials by task family | run order |
|---|---|---:|---:|---|---|
| `t17.2024.04.24` | 2024-04-24 | 55 | 8 | finger_movement 496 | finger x8 |
| `t17.2024.04.25` | 2024-04-25 | 56 | 4 | calibration 80 | cal x4 |
| `t17.2024.06.03` | 2024-06-03 | 95 | 7 | calibration 120, sentence_copy 20 | cal x6, copy |
| `t17.2024.06.04` | 2024-06-04 | 96 | 9 | calibration 65, sentence_copy 20, free_typing 35 | cal x4, copy, free x4 |
| `t17.2024.06.05` | 2024-06-05 | 97 | 5 | calibration 48, sentence_copy 20, free_typing 11 | cal, copy, free, cal x2 |
| `t17.2024.06.13` | 2024-06-13 | 105 | 2 | calibration 40 | cal x2 |
| `t17.2024.07.09` | 2024-07-09 | 131 | 10 | calibration 65, free_typing 28 | cal x3, free, cal, free x5 |
| `t17.2024.07.10` | 2024-07-10 | 132 | 4 | calibration 25, free_typing 15 | cal x2, free x2 |
| `t17.2024.07.22` | 2024-07-22 | 144 | 5 | calibration 60, free_typing 16 | cal x3, free x2 |
| `t18.2024.12.03` | 2024-12-03 | 43 | 6 | finger_movement 327 | finger x6 |
| `t18.2024.12.04` | 2024-12-04 | 44 | 5 | finger_movement 124, calibration 60 | finger x2, cal x3 |
| `t18.2024.12.05` | 2024-12-05 | 45 | 6 | calibration 30, sentence_copy 43, free_typing 5 | cal x2, copy x2, free, copy |
| `t18.2025.01.14` | 2025-01-14 | 85 | 11 | calibration 20, sentence_copy 82 | cal x2, copy x9 |
| `t18.2025.01.15` | 2025-01-15 | 86 | 13 | calibration 40, sentence_copy 90 | cal, copy, cal x3, copy x8 |
| `t18.2025.01.21` | 2025-01-21 | 92 | 11 | calibration 30, sentence_copy 80 | cal x3, copy x8 |
| `t18.2025.01.22` | 2025-01-22 | 93 | 13 | calibration 30, sentence_copy 100 | cal x3, copy x10 |
| `t18.2025.02.04` | 2025-02-04 | 106 | 14 | calibration 30, sentence_copy 100, free_typing 12 | cal x3, copy, free, copy x9 |
| `t18.2025.02.05` | 2025-02-05 | 107 | 16 | calibration 30, sentence_copy 130 | cal x3, copy x13 |
| `t18.2025.04.01` | 2025-04-01 | 162 | 11 | handwriting 122 | hw x11 |

In total 2,649 trials in 160 task blocks: 947 finger movements (496 T17, 451 T18), 773 calibration sentences, 685 sentence-copy sentences, 122 free-typing trials (121 with a recorded sentence) and 122 handwritten sentences. Five sessions return to a task family after another one (`06.05`, `07.09`, `12.05`, `01.15`, `02.04`), so a block-level split within a day can put one family on both sides of another.

Trial timing per participant and family, measured from the NSP markers on the bin grid (median, range in brackets):

| participant | family | trials | delay (trial start to go) | go window (go to trial end) | gap to the next trial |
|---|---|---:|---|---|---|
| T17 | finger_movement | 496 | 3.96 s (1.5-4.5) | 2.5 s | 2.5 s |
| T17 | calibration | 503 | 4.0 s | 47.8 s | 0 |
| T17 | sentence_copy | 60 | 4.0 s | 49.1 s | 0.8 s |
| T17 | free_typing | 105 | 3.9 s (0-65.6) | 53.8 s | 5.5 s |
| T18 | finger_movement | 451 | 2.02 s (1.5-2.5) | 2.5 s | 2.5 s |
| T18 | calibration | 270 | 4.0 s | 36.0 s | 1.0 s |
| T18 | sentence_copy | 625 | 7.0 s (4-9) | 25.8 s | 3.2 s |
| T18 | free_typing | 17 | 10.7 s (3.6-23.8) | 41.6 s | 3.5 s |
| T18 | handwriting | 122 | 5.0 s | 36.0 s | 1.0 s |

The finger-movement delays match the paper (3.5-4.5 s for T17, 1.5-2.5 s for T18), except T17 block 1 of `04.24`, which uses 1.5-2.5 s. The key-cue blocks `t18.2024.12.04` 2 and 3 have a 2.0 s go window. The authors' Fig. 2 analyses leave out block 1 of both finger-movement sessions. In free typing the "delay" is an uncued wait before typing.

For its character-error analysis (Fig. 4a-c) the paper trains on all calibration and sentence-copy trials of T17 days 56, 96, 97 and 105 plus 40 trials of day 95, and of T18 days 86, 92, 106 and 107 plus 30 trials of day 93, and evaluates on 100 held-out trials of day 95 (T17) and day 93 (T18). Day 93 holds 30 calibration and 100 sentence-copy trials.

## Data Modalities

| Modality | Source location | Type / shape |
|---|---|---|
| Threshold crossings | `.mat` `binned_neural_threshold_crossings` | int16 `(n_bins, 384)`, 10 ms bins (100 Hz), -3.5 x SD counts |
| Spike-band power | `.mat` `binned_neural_spike_band_power` | float32 `(n_bins, 384)`, 10 ms bins, sum of squared 250 Hz-5 kHz voltage, capped at 50,000 |
| Bin timestamps | `.mat` `binned_neural_nsp_timestamp`, `binned_neural_redis_clock` | float64 `(n_bins,)` ns (NSP), uint64 ms (BRAND) |
| Cue (label) | `.mat` `cue` | char `(n_trials,)`, whitespace-padded sentence, movement or key |
| Trial timing | `.mat` `*_nsp_neural_time` (plus `_redis_time`, and `_nsp_analog_time` for T17) | float64 `(n_trials,)`, trial start, go cue and trial end |
| Online decoder (closed loop) | `.mat` `decoder_logit_output`, `ngram_decoder_partial_output`, `ngram_decoder_final_output` | float32 `(n_rows, 41)` character logits, 5-gram language-model strings. Not emitted, see [Online decoder configuration](#online-decoder-configuration) |
| Decoder normalization (closed loop) | `.mat` `norm_channel_means`, `norm_channel_stds`, `norm_redis_times` | float32 `(n_updates, 512 or 768)`, 256 on `t17.2024.06.03`. Not emitted |
| Microphone audio (T17 only) | `.mat` `microphone_data`, `microphone_nsp_time` | int16 `(n_samples, 1)` at 30 kHz |

- `n_ch` is 384 in every file, six 64-channel arrays per participant (`utils.ARRAYS`, column order in [Channel order](#channel-order)). `utils.group_channels(subject, by="area" | "hemisphere")` gives the column groups.
- Features are raw (not z-scored). The authors' training pipeline z-scores per block and compresses the bins to 20 ms.

## Source Data Format & Layout

`Data.zip` (Dryad version 2, 19.5 GB) expands to a `Data/{t17,t18}/` tree, one `.mat` per block, with a single-trial feature preview `.png` beside each:

```
<NEURALSHED_DATA_ROOT>/jude_2026_typing/data/Jude2026Typing/        <- study folder (self.path, utils.DATA_DIR)
    README.md                                       Dryad description + block tables
    dryad_jude_2026_typing_success_download.txt     download marker (what was verified, and when)
    Data/                                           utils.BLOCKS_ROOT
        t17/t17.YYYY.MM.DD/<Typing|brainToText>/RedisMat/
            20240425_135520_(1).mat                 block 1 (one block per file)
            20240425_135520_(1).png
        t18/t18.YYYY.MM.DD/<Typing|brainToText|Handwriting>/RedisMat/...
        t18/t18.2024.12.04/Typing/tfdata_20ms/      the authors' preprocessing output for one session
```

19 sessions and 165 `.mat` files, 14.5-765.8 MB each and 34.9 GB in total. Each session has one task folder: `Typing`, `brainToText` (the two finger-movement sessions) or `Handwriting`. Per-block `.mat` fields are listed in `utils.py`'s module docstring and in the Dryad `README.md`.

- **The 165 files** hold the 160 task blocks of `utils.SESSION_BLOCKS` and five others: three reference blocks with a single `DO_NOTHING` trial (`t18.2024.12.03` 0, 7 and 8, the blocks the referencing coefficients and thresholds were fit on), a free-typing block whose four cues are empty (`t17.2024.06.04` 7), both in `utils.EXCLUDED_BLOCKS`, and a second export of `t18.2024.12.04` block 7 with identical arrays (`20241204_164727_(7).mat`, `utils.DUPLICATE_FILES`).
- **`tfdata_20ms/`** holds block means and standard deviations for blocks 2-8 of `t18.2024.12.04` (block 8 is not in the deposit), one train and one test TFRecord and the `mat_to_tfrecord_params.json` that made them (`block_nums [5, 6, 7]`, 20 ms bins, spike-band power capped at 50,000). It is out of scope.
- `Data.zip` also carries six `.ipynb_checkpoints` notebooks of the authors, which the extracted tree does not need.
- `LanguageModels.zip` (18.5 GB, the 5-gram character language model of the word error rate analysis) is out of scope.

## Study, timelines and Events

`study.py` (`Jude2026Typing`). Each `.mat` block is a recording with many cued trials inside, so one block is one timeline (160 task blocks). The files in `utils.EXCLUDED_BLOCKS` and `utils.DUPLICATE_FILES` are skipped, and the Study raises on a file in none of the tables. `subject`, `session`, `date`, `post_implant_day` (from the participant registry), `block`, `task_family` and `loop_type` (`open` or `closed`) ride the timeline and broadcast to every row. Select neural or cue rows by event type and slice conditions by attribute at query time. Event windows use neural-bin indices (the nearest bin to each NSP marker), so they align exactly to the 100 Hz feature grid.

Every trial emits, with the block-local 0-based `trial_id`:

1. The on-screen cue as a `Stimulus` over the delay (trial start to go), except in free typing, which shows no cue.
2. A one-bin `Stimulus` go marker (`movement_go_cue`).
3. The attempted unit over the go window (go to trial end), with `period="go"` and `success` (the source's `trial_paused_by_CNRA` and `trial_timed_out` flags, zero on every trial).
4. An `Intertrial` over the gap to the next trial when it is 0.1 s or longer and the recording does not pause inside it.

| Event type | Source | Shape / rate | Notes |
|---|---|---|---|
| `SpikeCounts` (threshold crossings) | `binned_neural_threshold_crossings` | `(384, n_bins)` at 100 Hz, whole block | first in each timeline, so `aggregation="cat"` gives the source's `[crossings, power]` order |
| `BandPower` (spike-band power) | `binned_neural_spike_band_power` | `(384, n_bins)` at 100 Hz, whole block | 250 Hz-5 kHz, `unit="uV^2"` |
| `Stimulus` (cue) | `cue`, trial start and go markers | per trial, over the delay | `modality="visual"`, `period="delay"`, `stimulus_kind` `movement_cue`, `rest_cue` or `sentence_cue`, the cue as `description`, and for finger cues the class id as `code` |
| `Stimulus` (go marker) | go marker | per trial, one bin | `stimulus_kind="movement_go_cue"` |
| `Action` (finger movement, rest) | `cue` | per trial, go to trial end | `action_kind` `finger_movement` (`task_condition="attempted"`) or `rest` (`task_condition="na"`, code 0), `code` from `utils.CUE_CODES`, the movement as `description` (also for key cues) |
| `Sentence` | `cue` | per trial, go to trial end | `text` as the cue gives it, `language="en"`, `modality` `typed` or `written` (handwriting), `task_condition="attempted"` |
| `Intertrial` | trial end and next trial start | gaps of 0.1 s or more | `period="intertrial"` |

The package `SpikeCounts`, `BandPower` and `Intertrial` events and the built-in neuralset `Stimulus`, `Action` and `Sentence` cover every row. The neural rows declare the acquisition chain in `processing`: a non-causal 4th-order 250 Hz-5 kHz Butterworth (`filtfilt`), linear regression referencing per array fit on the session's reference block, thresholds at -3.5 x SD per electrode, and spike-band power as the sum of squared voltages per bin, capped at 50,000 uV^2. The microphone audio (T17) is anarthric and read with `utils.get_audio`, not emitted. One free-typing trial has no recorded sentence (`t18.2025.02.04` block 5 trial 6) and emits its go marker and gap but no unit row.

**Target alphabet.** The authors' decoder (`utils/mat_to_tfrecord.py`, `CHAR_DEF`) predicts 30 tokens plus the CTC blank: space, comma, question mark, period and `A`-`Z`, which are the 30 keys. Their recipe removes `--`, upper-cases the text and drops every other character, apostrophes and hyphens included. `Sentence.text` keeps the cue as given: calibration and sentence-copy cues are sentence case, free-typing and handwriting cues mostly lower case, and 341 sentences hold an apostrophe and 4 an exclamation mark, which the keyboard cannot type. A character target that matches what the participant typed applies the authors' mapping.

## Notes / Scope

- **Download.** `Study.download()` (`_download(overwrite=False)`) fetches `Data.zip` and `README.md` from Dryad (`utils.DRYAD_FILES`, `NEURALFETCH_DRYAD_TOKEN` required for the transfer) through the shared backend `datasets/dryad.py` and writes `dryad_jude_2026_typing_success_download.txt` next to `Data/`. Without `overwrite`, a tree that passes `utils.verify_tree` (the README digest against the public Dryad listing, every block of `SESSION_BLOCKS` and `EXCLUDED_BLOCKS` and the duplicate file present, every file opening with `whosmat`) is accepted and the marker records the check and its date. Once the marker exists `download()` returns at once. `utils.py` reads blocks with `scipy.io.loadmat`, and its `fields=` argument avoids loading the neural and audio arrays.
- **Recording pauses.** 11 blocks are not one continuous recording: 10 free-typing blocks and handwriting block `t18.2025.04.01` 3 hold 49 pauses where the NSP clock jumps between consecutive bins with no bins in between, 45 of 5.9-420 s and 4 of 22 ms-0.9 s. `t18.2025.04.01` block 3 is three 10-sentence segments separated by pauses of 420 s and 410 s. The Study positions events by bin index, no trial window crosses a pause, and no `Intertrial` is emitted across one (`utils.clock_breaks`). A window built by hand across a pause joins recordings up to minutes apart.
- **Sentence reuse.** The cues come from the Switchboard corpus, as in the lab's speech datasets, and repeat within and across datasets (see [Sentences shared with other datasets](#sentences-shared-with-other-datasets)). A split at the trial level shares prompts between train and test. Split on the prompt text, or hold out whole sessions as the paper does.
- **Microphone audio is not speech.** T17 is anarthric, so the channel holds effort and breathing.

### Online decoder configuration

- **Features.** The normalization vectors of each closed-loop block (`norm_channel_means`) give the features the online decoder ran on, crossings and power stacked. T17 used 256 features on `t17.2024.06.03` (the two 6d arrays, file columns 256-383) and 512 from `06.04` on (6v and 6d, file columns 0-127 and 256-383, 55b left out). T18 used all 768. The vectors match the per-column means of the block's own data in those columns.
- **Logits.** `decoder_logit_output` has 41 columns, of which the first 31 are used (blank, space, comma, question mark, period and `A`-`Z` in the authors' notebooks), and the rest are zero. About half the rows are all-zero padding, and the non-zero rows arrive every 80 ms for T17 and about 100-160 ms for T18 (320 ms in handwriting block 13), against the decoder's patch strides of 180 ms (T17) and 120 ms (T18). The deposit README says every 20 ms.
- The logits are in all 65 sentence-copy blocks, all 17 free-typing blocks, handwriting block 13 and the excluded free-typing block `t17.2024.06.04` 7. Every other file stores a `(1, 41)` placeholder. None of this is emitted.

### Channel order

Both participants store 384 columns, six arrays of 64 columns each, and the Study emits them in file order.

| columns | T17 | T18 |
|---|---|---|
| 0-63 | `ventral_6v` | `mid_ventral_6d` (left) |
| 64-127 | `dorsal_6v` | `ventral_6d` (left) |
| 128-191 | `ventral_55b` | `dorsal_6d` (left) |
| 192-255 | `dorsal_55b` | `mid_dorsal_6d` (left) |
| 256-319 | `ventral_6d` | `lateral_6d` (right) |
| 320-383 | `dorsal_6d` | `medial_6d` (right) |

No source document states the column order, so it is established from the authors' code and from the data:

- **T17.** The authors' FVAF notebook titles the six column groups `v6v, d6v, v55b, d55b, v6d, d6d`, which is the Study order of `jude_2026_speech` for the same implant. Matching the within-group correlation structure and the per-electrode median spike-band power of each 64-column group to a late `jude_2026_speech` block (`05.28` block 6, Study order) gives the identity on `04.25`, `06.03` and `06.13`, and against the early speech file `04.02` it gives that package's early permutation. `07.22` (day 144) matches on five of the six groups. Independently, the online normalization vectors match file columns 256-383 (6d) on day 95 and 0-127 plus 256-383 (6v and 6d) from day 96, the paper's "128 6v electrodes and 128 6d electrodes".
- **T18.** The authors' FVAF notebook and `PlotArrayCERs.ipynb` agree on `mv6d` 0-63, `v6d` 64-127, `d6d` 128-191, `md6d` 192-255 in the left hemisphere and `Rl6d` 256-319, `Rm6d` 320-383 in the right, and the data confirm the hemisphere split (stronger tuning to the contralateral hand). Reading the left "m" as mid and the right "l" and "m" as lateral and medial is ours, the sources do not expand the abbreviations.

### Source discrepancies

The files are the authority for the Study. Where the deposit README or the paper says otherwise:

| fact | source says | the files show | Study uses |
|---|---|---|---|
| sentence-copy loop type | deposit README: open loop, no decoded output on screen | closed loop: every sentence-copy file's `block_description` reads "Runs a closed loop ..." and the file stores the online decoder's output. The paper (p. 6) also describes the decoded key highlighted in real time | `closed` |
| post-implant day of `t18.2025.04.01` | deposit README: trial day 163 | the session date is 162 days after the 2024-10-21 implant | 162 |
| trials per block | deposit README tables: 3,668 trials in 159 blocks | the 156 of those blocks that were deposited hold 2,609 trials. 83 blocks differ: 73 T18 blocks are listed with 20 trials and hold 10, `t18.2024.12.03` block 1 is listed with 62 and holds 17 (a sweep stopped early), and 9 others differ by 1-18 | the counts in the files |
| listed blocks that are missing | deposit README lists `t17.2024.06.03` blocks 1-3 (finger movements, 62 trials each) | no such files in `Data.zip`. The paper calls these extra repetitions "not used" | skipped (`utils.NOT_DEPOSITED`) |
| deposited blocks that are not listed | deposit README tables leave them out | 8 blocks: four 10-trial sentence-copy blocks (`t18.2025.01.21` 7, `01.22` 4 and 7, `02.05` 7) that the authors' training code uses, one free-typing block whose four cues are empty (`t17.2024.06.04` 7), and three reference blocks (`t18.2024.12.03` 0, 7 and 8) | the four sentence-copy blocks are emitted, the other four are excluded |
| `t18.2024.12.04` block 7 | listed once | two files with identical data, `20241204_163713_(7).mat` and `20241204_164727_(7).mat` | the first file |
| handwriting block numbers (`t18.2025.04.01`) | deposit README: block 3 has 30 trials, block 4 has 10 | the file named `(3)` holds 30 trials but stores `block_number` 0, the file named `(4)` holds 10 trials and stores 3 | the number in the filename |
| spike-band power | deposit README: "sum of squared voltages" in its description, "mean averaged sum" in its variable list | values capped at 50,000, which do not settle sum or mean | sum, as the paper states |
| `binned_neural_redis_clock` | deposit README: one timestamp per bin from the acquisition software (BRAND) | 7 files have a different number of timestamps than bins: 5 are off by one, and `t17.2024.07.22` blocks 4 and 5 have 551 and 836 fewer. All 7 are free-typing blocks with recording pauses | not used, everything is timed with the NSP timestamps |
| decoder output timing | deposit README: a character prediction every 20 ms | non-zero `decoder_logit_output` rows every 80-320 ms, with all-zero rows in between | not emitted |

### Trials removed in the authors' code

The authors' session files (`T17_SessionArgs.py` and `T18_SessionArgs.py` in folder `25)` of the reference code, `T18_SessionArgs_comp.py` in folder `15)`) remove 27 trials from training without a stated reason, and the paper's reporting summary says no data was excluded. The Study emits them as ordinary trials:

| session | block: trials |
|---|---|
| `t17.2024.06.13` | 7: 0, 1, 2 |
| `t17.2024.07.22` | 1: 18 |
| `t18.2024.12.04` | 5: 9, 12, 16 |
| `t18.2025.01.14` | 1: 0. 2: 2. 3: 4. 4: 1, 8. 10: 0, 3 |
| `t18.2025.01.15` | 5: 8. 7: 9 |
| `t18.2025.01.22` | 3: 1, 8 |
| `t18.2025.02.04` | 1: 0, 1, 2, 3 |
| `t18.2025.02.05` | 1: 0, 1 |
| `t18.2025.04.01` | 3: 20, 21, 22 |

Most are the first trials of a session or of a recording segment (handwriting trials 20-22 follow the 7-minute pause), and their typing speed is unremarkable (3 of the 27 are above the 95th percentile of seconds per character).

### Sentences shared with other datasets

The 1,701 `Sentence` rows hold 1,511 distinct texts after `text.normalize_text` (lower case, letters, apostrophes and spaces), and 168 of them occur more than once: 73 calibration texts repeat within calibration, 71 texts are cued in both calibration and sentence copy, and 44 are cued to both participants. The cued sentences come from the Switchboard corpus, as in the lab's speech datasets, so most of them also occur there. Distinct normalized texts shared with each dataset:

| dataset | distinct texts | shared with this dataset |
|---|---:|---:|
| `card_2024` (train and val) | 8,565 | 513 |
| `wairagkar_2025` | 8,231 | 613 |
| `willet_2023` | 9,683 | 597 |
| `jude_2026_speech` (sentence blocks) | 3,081 | 198 |
| `kunz_2025` (sentence blocks, keyword prefix removed) | 1,472 | 64 |

1,147 of the 1,511 texts occur in at least one of the five. No recording is shared: the T17 dates here differ from those of `jude_2026_speech` and `kunz_2025`, and T18 is in no other dataset. A collection that pools these datasets shares prompts across them, so a prompt-disjoint split has to be made over all members at once. Free-typing texts were composed by the participant, and six of them write the apostrophe as U+2019, which `normalize_text` turns into a space.

## Files

- `__init__.py`: package marker.
- `metadata.py`: `DATASET_METADATA` (catalog, with the source-only streams that have no event class) + `sessions()` (trial counts and task condition per session) + `get_metadata()` (DB-validated).
- `utils.py`: `.mat` readers and accessors (`list_sessions`, `list_blocks`, `block_path`, `task_of`, `loop_type`, `described_loop_type`, `exclusion_reason`, `verify_tree`, `get_neural`, `get_trials`, `get_audio`, `clock_breaks`, `array_channels`, `group_channels`, `post_implant_day`, `cue_code`) and constants (rates, `ARRAYS`, `SESSION_BLOCKS`, `EXCLUDED_BLOCKS`, `DUPLICATE_FILES`, `NOT_DEPOSITED`, `TASK_FAMILY_INFO`, `LOOP_TYPE_OVERRIDES`, `CUE_CODES`, `MOVEMENTS`).
- `study.py`: `Jude2026Typing` neuralset Study (one timeline per block).
- `01_explore_source_data.ipynb`: source-data demo.
- `02_explore_study.ipynb`: Study and extractor read-back demo.
- `03_model_example.ipynb`: character CTC decoder on T18 sentences, trained on the calibration and sentence-copy blocks of the other sessions, tested on held-out day-93 sentences.
