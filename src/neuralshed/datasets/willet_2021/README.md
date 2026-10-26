# Willett 2021: High-Performance Brain-to-Text Communication via Handwriting

Intracortical BCI study of attempted handwriting in one BrainGate2 participant (T5, C4 spinal cord injury). Two 96-channel Utah arrays (192 electrodes) in the hand knob of the left precentral gyrus record 10 ms-binned threshold-crossing counts while T5 attempts to write single characters, straight lines and full sentences under an instructed-delay paradigm. Real-time decoding of the attempted handwriting reached 90 characters per minute.

## References

- Willett, F.R., Avansino, D.T., Hochberg, L.R., Henderson, J.M., Shenoy, K.V. (2021). High-performance brain-to-text communication via handwriting. *Nature* 593, 249-254. [doi:10.1038/s41586-021-03506-2](https://doi.org/10.1038/s41586-021-03506-2)
- Supplementary Methods: [41586_2021_3506_MOESM1_ESM.pdf](https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-021-03506-2/MediaObjects/41586_2021_3506_MOESM1_ESM.pdf), archived as `references/willet_2021_supplementary_methods.pdf` (sha-256 `8ddc48e62937baa1f48ace21c28326ea96eecf25bab42f7643845f082babc0ad`).
- Dryad data repository: [doi:10.5061/dryad.wh70rxwmv](https://doi.org/10.5061/dryad.wh70rxwmv)
- Code repository: [github.com/fwillett/handwritingBCI](https://github.com/fwillett/handwritingBCI)
- Clinical trial: BrainGate2, NCT00912041. Recorded at Stanford University.

## Species & Subjects

- **Species:** human (*Homo sapiens*).
- **T5** (`braingate2/T5` in `datasets/participants.py`, implant 2016-08-17): right-handed man, 65 years old at the time of data collection, with a C4 AIS C (motor incomplete) spinal cord injury about 9 years before enrollment (Supplementary Methods 1.1). Head, face and shoulder movement are intact. Voluntary movement below the injury is limited, mostly to the left elbow, and small micromotions of the right hand were visible during attempted handwriting.
- **Implant:** two 96-electrode Blackrock NeuroPort arrays (1.5 mm electrode length, 0.4 mm pitch) in the hand knob of the left (dominant) precentral gyrus, implanted August 2016. Channels 0-95 are array 1 (more lateral), 96-191 are array 2 (more medial), per the deposit readme. `utils.ARRAY_GEOMETRY_MAP` holds the 10 x 10 electrode layout the files give.
- **Sessions:** 11, post-implant days 994-1246 (2019-05-08 to 2020-01-15).
- The same participant is recorded in `fan_2023`, `deo_2024`, `wilson_2025`, `willsey_2025` and `gusman_2025`. No source states that channel `i` here is the same electrode as channel `i` there, so channel order is treated as dataset-specific.

## Behavioral Task

Every task is an instructed-delay trial. A cue appears on screen next to a red square during the delay, then the text changes to "Go" and the square turns green, and T5 attempts the movement. Trials follow each other with no gap: the next delay starts when the previous go period ends, or within 0.02 s of a sentence's end.

### Single characters (`singleLetters.mat`, 10 sessions)

T5 writes one cued character during a 1 s go period. 31 classes: a-z plus `greaterThan` (space), `comma`, `apostrophe`, `tilde` (period) and `questionMark`. Most sessions have 10 repetitions per character (310 trials). The pilot session 2019.05.08 has 27 repetitions per character plus 27 trials of a `doNothing` rest condition (864 trials).

### Straight lines (`straightLines.mat`, 2019.06.03)

Single straight pen strokes, 48 conditions `straightLine_{short,medium,long}_{1..16}` (3 distances x 16 angles 22.5 degrees apart), 24 repetitions each (1,152 trials). Condition 1 is a rightward stroke, 5 upward and 9 leftward. Only the long strokes are analyzed in the paper. The go period lasts 1.5 s.

### Sentences (`sentences.mat`, 10 sessions)

The sentence is shown during the delay and stays on screen while T5 writes it. T5 ends a trial by turning his head to the right, which a head tracker detects. Conditions, spelled as in `sentenceCondition`:

- `OL Copy`: copy the prompt, no real-time decoder.
- `OL Copy With Pauses`: copy with pauses at the `#` marks of the prompt.
- `OL Copy from Memory`: write a short phrase from memory.
- `CL Pandarinath`: closed-loop copy of 7 benchmark sentences (Pandarinath et al. 2017).
- `CL Corpus`: closed-loop copy of sentences from the British National Corpus.
- `CL Free Write`: closed-loop free answers to question prompts.

1,000 sentences, 43,501 intended characters. The condition is constant within each of the 95 sentence blocks. The prompt differs from the intended text on 90 trials (the 40 `OL Copy With Pauses` prompts and the 50 free-write questions).

### Sessions

| session | post-implant day | files | content |
|---|---|---|---|
| t5.2019.05.08 | 994 | sentences, singleLetters | pilot: 102 open-loop sentences, 27 repetitions per character plus rest |
| t5.2019.06.03 | 1020 | straightLines | 48 line conditions x 24 |
| t5.2019.11.25, t5.2019.12.09 | 1195, 1209 | sentences, singleLetters | real-time decoding pilots |
| t5.2019.12.11 to t5.2020.01.08 (5 sessions) | 1211-1239 | sentences, singleLetters | copy-typing evaluation |
| t5.2020.01.13, t5.2020.01.15 | 1244, 1246 | sentences, singleLetters | free-answer evaluation, sentences with pauses, phrases from memory |

### Delay durations in the data

| family / condition | delay (s) |
|---|---|
| singleLetters, pilot | 1.50-3.00, median 2.40 |
| singleLetters, other sessions | 1.50-2.50, median about 1.98 |
| straightLines | 1.00-2.00, median 1.40 |
| `OL Copy` | 4.02-9.00, median 5.00 |
| `OL Copy With Pauses`, `CL Corpus`, `CL Pandarinath` | 5.00 |
| `OL Copy from Memory` | 6.00 |
| `CL Free Write` | 1.00 |

## Data Modalities

- **Neural signal:** threshold-crossing counts, `uint8`, values 0-10, 192 channels, mean rates 18.06-24.14 Hz per file. The chain: 0.3 Hz-7.5 kHz analog filter, 30 kHz sampling, common average reference per array, non-causal 250-3000 Hz bandpass, -3.5 x RMS threshold set once per session (Supplementary Methods 1.2 and the deposit readme).
- **Bin size / rate:** 10 ms bins, 100 Hz. 6,297,018 bins (17.49 h) in 130 blocks.
- **Behavior:** the task structure: cues, sentence prompts and intended text, and delay, go and end bins. The deposit holds no pen or hand kinematics.

## Source Data Format & Layout

The Dryad deposit holds `handwritingBCIData.tar.gz` (1,408,201,020 bytes) and `readme.pdf` (288,173 bytes). The top-level `readme.pdf` is the current documentation. The older copy inside the tarball (286,305 bytes) lacks the Straight Lines section.

```
willet_2021/data/Willett2021Handwriting/
  readme.pdf
  handwritingBCIData/
    readme.pdf               # older copy
    Datasets/                # 443 MB
      t5.2019.05.08/{singleLetters,sentences}.mat
      t5.2019.06.03/straightLines.mat
      ... (11 sessions, task files 4.3-47.6 MB)
      computerMouseTemplates.mat
    RNNTrainingSteps/        # 14 GB
    BigramLM/                # 268 MB
```

All files are MATLAB `.mat`, read with `scipy.io.loadmat(..., simplify_cells=True)`.

### Key variables

Common to all task files:

| variable | shape | notes |
| --- | --- | --- |
| `neuralActivityTimeSeries` | `(T, 192)` uint8 | the file's blocks spliced end to end. Activity between blocks was not saved |
| `blockNumsTimeSeries` | `(T,)` | block number per bin |
| `clockTimeSeries` | `(T,)` | system clock (s), re-zeroed at the start of every block |
| `blockList` | `(n_blocks,)` | block numbers in recording order |
| `blockStartDates` | `(n_blocks,)` | block start date and time |
| `arrayGeometryMap` | `(10, 10)` | electrode number at each array position, rows toward the central sulcus, columns medial |

`singleLetters.mat` and `straightLines.mat`:

| variable | shape | notes |
| --- | --- | --- |
| `neuralActivityCube_{cond}` | `(reps, 201, 192)` uint8 | trials of one condition in recording order, 0-indexed step 50 is the go cue |
| `goPeriodOnsetTimeBin`, `delayPeriodOnsetTimeBin` | `(n_trials,)` | onset bins per trial |
| `characterCues` / `movementCues` | `(n_trials,)` | cued character or line condition |
| `meansPerBlock`, `stdAcrossAllData` | `(n_blocks, 192)`, `(192,)` | per-block means and per-channel std (singleLetters only) |

`sentences.mat`:

| variable | shape | notes |
| --- | --- | --- |
| `neuralActivityCube` | `(S, T_max, 192)` uint8 | one row per sentence from its go cue. Bins past `numTimeBinsPerSentence` hold the recording that follows, not padding |
| `goCueOnsetTimeBin`, `delayCueOnsetTimeBin`, `sentenceEndTimeBin` | `(S,)` | onset and end bins per sentence |
| `numTimeBinsPerSentence` | `(S,)` | go-to-end length |
| `sentencePrompt`, `intendedText` | `(S,)` | cued and intended text, `>` for space and `~` for period |
| `sentenceCondition`, `sentenceBlockNums` | `(S,)` | condition and block per sentence |
| `excludedSentences` | `(S,)` | 1 = excluded by the authors |
| `rnn_*` | various | real-time decoder outputs, zero in open-loop blocks |

The onset bins are 1-based MATLAB indices into `neuralActivityTimeSeries`. The single-trial files name them `*PeriodOnsetTimeBin`, the sentence files `*CueOnsetTimeBin`.

## Study, timelines and Events

[`study.py`](study.py) (`Willett2021Handwriting`) emits one timeline per recording block, 130 in all: 27 singleLetters, 8 straightLines and 95 sentences blocks. Each block is the largest span on one continuous clock. The timeline carries `session`, `task` (`singleLetters`, `straightLines` or `sentences`), `block`, `date`, `post_implant_day`, `loop_type` (`closed` for the `CL` sentence conditions, `open` otherwise) and, on sentence blocks, `sentence_condition`. Event times are seconds from the block's first bin.

| event type | rows | span | notes |
| --- | --- | --- | --- |
| `SpikeCounts` | 130 | whole block | `(192, n_bins)` at 100 Hz, `threshold_crossings`, `counts/bin`, `normalization="raw"` |
| `Stimulus`, `period="delay"` | 5,806 | delay onset to go | `stimulus_kind` `character_cue`, `movement_cue`, `rest_cue` (with the unit's `code`) or `sentence_cue` (the prompt as `description`) |
| `Stimulus`, `period="go"` | 5,806 | one bin at the go cue | `stimulus_kind="movement_go_cue"` |
| `Character` | 3,627 | go to the next delay (1.0 s) | `text` the glyph (space and period as ` ` and `.`), `char_class` the source label, `code` 1-31 |
| `Action`, `action_kind="drawing"` | 1,152 | go to the next delay (1.5 s) | `description` the condition, `distance`, `angle_index`, `code` 32-79 |
| `Action`, `action_kind="rest"` | 27 | go to the next delay (1.0 s) | `doNothing`, `code` 0, `task_condition="na"` |
| `Sentence` | 999 | go to the head turn | `text` from `intendedText`, `excluded`, `original_split` |

Every trial row carries `trial_id`, the 0-based position in the file's trial arrays, unique within a `(session, task)` pair. The published partitions and the authors' code index sentences the same way. `code` is dataset-wide: rest 0, characters 1-31 in `utils.CHAR_LIST` order, lines `32 + 16 * distance + angle - 1` with distance 0, 1, 2 for short, medium, long. The last character or line trial of a block ends 1.0 s or 1.5 s after its go cue.

The 33 sentences in `excludedSentences` keep their rows. 32 carry `valid=False` on the `Sentence` and the neural rows stay valid. The 33rd, `t5.2020.01.15` trial 142, has an empty `intendedText` (the readme's unresolved restaurant spelling), so it has a delay cue and go marker and no `Sentence`. The readme's reasons are premature head-turn detection, a skipped free-write question, that spelling, forgotten memory phrases, and on 2020.01.06 the first trial of each evaluation block after a decoder initialization bug, whose recordings are intact. Notebook 02 leaves the 32 out of the targets with `ValidateTarget(condition="type == 'Sentence' and canonical_text != '' and valid")` in the willet_2021 chain.

`original_split` on each `Sentence` is the `HeldOutBlocks` partition from `RNNTrainingSteps/trainTestPartitions_HeldOutBlocks.mat` (`study._PARTITION`), with `""` for sentences in neither set: 502 `train`, 228 `test`, 269 `""`. Every non-excluded `CL Corpus` and `CL Pandarinath` sentence is `test`, every non-excluded `OL Copy` and `OL Copy With Pauses` sentence is `train`, and `OL Copy from Memory` and `CL Free Write` are in neither. Train and test share no block and no target text. The other partition, `HeldOutTrials` (574 train, 63 test, 363 in neither), holds out single trials inside blocks and is the default of the authors' pipeline. `utils.load_partition` reads either.

## Notes / Scope

- Threshold crossings are the deposit's only neural feature.
- The Study reads each block's neural data through a `SpecialLoader`. `utils.read_neural` reads a file's whole matrix and keeps the last two in memory, so the blocks of one file share a read. Point the [`NeuralFeatureExtractor`](../../extractors.py) at `SpikeCounts`.
- Out of scope, each with its reason:
  - `RNNTrainingSteps/` other than the two partition files: the authors' time-warped data, synthetic sentences, trained decoders and their outputs.
  - `RNNTrainingSteps/Step2_HMMLabels`: per-character start times and durations inside sentences, inferred by HMM forced alignment.
  - `BigramLM/`: Kaldi language model files.
  - `rnn_*` variables in `sentences.mat`: the real-time decoder's outputs.
  - `Datasets/computerMouseTemplates.mat`: 31 two-dimensional velocity templates at 100 Hz in arbitrary units, drawn with a computer mouse by another person following T5's descriptions.
- Every block opens with 10-107 s of neural data before its first delay (1.00 h in total) and closes with 2.6-32.5 s after its last trial (0.24 h). These spans carry only the neural stream, and no source describes them.
- In the pilot `singleLetters` file, blocks 11 and 19 start with 14 zero values of `clockTimeSeries` followed by 5.10 and 5.09 s. The neural rows there are distinct, and the first delays start 24 and 54 s later.
- A sentence trial ends when the head tracker detects a head turn, and the excluded sentences include trials where the detection fired early.

### Source discrepancies

Where the documentation and the data disagree, the Study follows the data.

- Delays: the supplement gives 2.0-3.0 s for single characters and says sentence delays "always lasted 5 seconds". The data give the durations in the table above.
- Lines: the supplement calls the line paradigm identical to the character paradigm. In the data the line go period is 1.5 s and the delays are 1.0-2.0 s.
- Line angles: the readme says the conditions "proceed clockwise", but the sequence it gives (1 rightward, 5 upward, 9 leftward) runs counterclockwise. `angle_index` keeps the source's condition number.
- The readme's overview speaks of ten sessions, and its session table and the deposit hold eleven.

## Files

- `metadata.py`: `DATASET_METADATA`, `sessions()` and `get_metadata()` for the metadata DB.
- `utils.py`: source readers, each taking the `handwritingBCIData` folder as `root`: `list_sessions`, `list_task_files`, `n_time_bins`, `load_variables`, `load_trial_variables`, `load_single_letters`, `load_sentences`, `read_neural`, `block_bins`, `extract_single_letter_trials`, `extract_sentence_trials`, `load_partition`, `session_summary`, `list_cube_keys`, `get_condition_cube`, plus `condition_code`, `decode_text`, `session_date`, `post_implant_day` and constants (`ARRAY_GROUPS`, `ARRAY_GEOMETRY_MAP`, `CHAR_LIST`, `CHAR_CLASS_TO_GLYPH`, `LINE_CONDITIONS`, `SESSION_DESCRIPTIONS`, `PARTITIONS`, `BIN_SIZE_S`, `RATE_HZ`).
- `study.py`: the neuralset `Study` (`Willett2021Handwriting`).
- `__init__.py`: package marker.
- `01_explore_source_data.ipynb`: reading and plotting the source files.
- `02_explore_study.ipynb`: the Study, its events table and its extractors.
- `03_model_example.ipynb`: classifying the handwritten character (31 classes) from go-period features.
