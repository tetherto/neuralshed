# Kunz 2025: Inner Speech in Motor Cortex

Four BrainGate2 participants with severe dysarthria or anarthria took part. Each had multiple 64-channel Utah arrays implanted in the left precentral gyrus, and T12 also had two arrays in the left inferior frontal gyrus. The dataset contains binned threshold crossings and spike-band power from eleven instructed-delay tasks spanning attempted, imagined and perceived speech. Together, these tasks probe how human motor cortex represents inner speech and how those signals might support speech neuroprostheses.

## References

- **Publication:** Kunz, Krasa, Kamdar, Avansino, Hahn, Yoon, Singh, Card, Jude, et al. (2025). "Inner speech in motor cortex and implications for speech neuroprostheses." *Cell* 188, 4658-4673. [doi:10.1016/j.cell.2025.06.015](https://doi.org/10.1016/j.cell.2025.06.015)
- **Supplement:** Document S1 (Tables S1, S2, S4, S5), Tables S3, S6, S7 and Figures S1-S6, from the open-access record [PMC12360486](https://pmc.ncbi.nlm.nih.gov/articles/PMC12360486/). Table S4 holds the task timings per participant.
- **Source data (Dryad):** [doi:10.5061/dryad.gf1vhhn1j](https://datadryad.org/dataset/doi:10.5061/dryad.gf1vhhn1j)
- **Code repository:** [github.com/nptl-stanford/inner_speech](https://github.com/nptl-stanford/inner_speech)
- **Array placement and thresholds:** Deo et al. (2026), "A mosaic of whole-body representations on the human precentral gyrus", *Nature*, [doi:10.1038/s41586-026-10653-x](https://doi.org/10.1038/s41586-026-10653-x) (a 2024 preprint of the same work). It describes the signal chain of all four participants and states the threshold of each.

This dataset is part of BrainGate2, ClinicalTrials.gov [NCT00912041](https://clinicaltrials.gov/study/NCT00912041).

## Species & Subjects

*Homo sapiens.* Each of the four participants had chronically implanted **64-channel silicon Utah arrays** (1.5 mm, sputtered iridium oxide, Blackrock). Placement was guided by individualized HCP cortical parcellation. All arrays are in the **left precentral gyrus**, except T12's two area 44 arrays in the **left inferior frontal gyrus**. Each participant is registered in `datasets/participants.py`, which holds the implant date (the paper gives only the month) and derives `post_implant_day`. The array names are those of [`metadata.py`](metadata.py) and `utils.ARRAY_LAYOUTS`, in stored column order, and follow the corpus rule of [standardization.md](../../../../docs/standardization.md#arrays).

| ID | Registry id | Sex, hand, age | Clinical condition (ALS-FRS at enrollment) | Implant | Post-implant days | Arrays in stored column order |
|---|---|---|---|---|---|---|
| T12 | `braingate2/T12` | F, left, 68 | Slowly progressive bulbar-onset ALS diagnosed at 59, severely dysarthric for nearly 8 years, partial use of the limbs (26) | 2022-03-30, 4 arrays | 412-995 | `ventral_6v`, `dorsal_6v` (area 6v), `ventral_44`, `dorsal_44` (area 44). 46 of 127 timelines store the 6v pair only |
| T15 | `braingate2/T15` | M, left, 45 | ALS diagnosed at 40, no functional use of the limbs, severe dysarthria (23) | 2023-07-17, 4 arrays | 230-538 | `ventral_6v`, `4` (area 4), `55b`, `dorsal_6v` |
| T16 | `braingate2/T16` | F, right, 52 | Pontine stroke about 19 years before enrollment, tetraplegia and dysarthria. Speaks slowly and quietly, limited voluntary control of the arms, sensation fully intact | 2023-12-07, 4 arrays | 88-377 | `pef_55b` (border of the premotor eye field and 55b), `6v`, `ventral_6d`, `dorsal_6d` (hand knob) |
| T17 | `braingate2/T17` | M, right, 33 | ALS, incomplete locked-in syndrome: anarthric, quadriplegic and ventilator-dependent, communicates with eye movements (0) | 2024-02-29, 6 arrays | 284 (paper 284-287, see Source discrepancies) | `ventral_6v`, `dorsal_6v`, `ventral_55b`, `dorsal_55b`. The two hand-knob 6d arrays are not in this release |

**Arrays without speech tuning.** The paper found no significant decoding of any speech behavior on T16's two 6d arrays, T16's PEF array and T17's two 55b arrays, and excluded them from its analyses (Figure 1C and 1E). T16's PEF and 6d arrays record spiking that is not tuned to speech, while T17's 55b arrays record very little spiking activity. The Study emits them unchanged, because the recordings are genuine.

T15's arrays carry the MNI centers published with `card_2024` (NEJM 2024 appendix S1.12). No coordinates are published for the other three participants. The `jude_2026_speech` and `jude_2026_typing` datasets state T17's age as 34, the paper behind this dataset says 33.

## Behavioral Task

The active tasks use an **instructed-delay** paradigm: a cue appears during a *delay* period (red square), and a *go* cue (green square) then signals execution. The isolated listening and reading blocks are passive: a word is presented and followed by an intertrial interval, with no delay or go cue. The verbal-behavior tasks use a fixed set of **7 single-syllable words** (ban, choice, day, feel, kite, though, were) with non-overlapping phonemes, plus a DO_NOTHING rest condition.

The eleven task types span attempted, inner and perceived speech, along with cognitive tasks that may elicit inner speech incidentally:

| Task type | Participants | Cue and target |
|---|---|---|
| `isolatedVerbalBehaviors` | T12, T15, T16, T17 | 7 behaviors in separate blocks: attempted vocalized, mimed, motoric inner speech, auditory inner speech, imagined listening, passive listening, silent reading |
| `interleavedVerbalBehaviors` | T12, T15, T16, T17 | the same 7 words, three behaviors intermixed within a block: attempted (mimed for T16), one inner-speech behavior and passive listening |
| `sentenceData` | T12, T15, T16 | attempt or imagine speaking a displayed sentence (50-word or 125,000-word vocabulary, `_eval` = closed-loop) |
| `chittychittybangbang` | T12 | inner-speech sentences, 344 of the 680 randomly opened by the imagined unlock keyword "chittychittybangbang" |
| `conjunctiveCounting` | T15, T16 | silently count a cued shape-color combination in a grid (10-20 targets), then say the count |
| `seqRecallUninstructed3ElementArrow` | T12, T16 | recall a sequence of three arrows (64 sequences for T12), then move a joystick (T12) or attempt to draw it (T16) |
| `seqRecallUninstructedSingleArrow` | T12 | one arrow (4 directions), joystick |
| `seqRecallUninstructed3ElementLines` | T12, T16 | draw the cued line sequence with a stylus (T12) or attempt to draw it (T16). T12's file holds 49 cues: 32 three-segment sequences including diagonals, 16 two-segment sequences and rest. The paper analyses the 32 three-segment cues |
| `seqRecallVerbalMemory` | T12, T16 | hold the sequence in inner speech over the delay, then draw it |
| `seqRecallVisualMemory` | T12, T16 | hold the sequence as a visual image over the delay, then draw it |
| `seqRecallSpeech` | T12, T16 | say the direction sequence aloud, the cue played as audio to T12 and shown to T16 |

**Sentence strategies.** For the inner-speech sentences each participant chose the form of inner speech they could perform most consistently. T12 used motoric inner speech (sentence and keyword sessions), T16 first-person auditory inner speech, and T15 motoric inner speech on 2024-03-03 and a hybrid of motoric imagery and imagined listening in the two 125k sessions (2024-12-15 and 2025-01-05). The Study records the strategy as `behavior` on each `Sentence` row.

Period durations represented by the Study, in seconds, median with the range in parentheses. Every value comes from the released epochs, except T16's sequence-recall periods (Table S4, see Notes) and the T16 passive intertrials (the next trial's onset, see Passive trials).

| Task | Participant | Delay | Go | Intertrial |
|---|---|---|---|---|
| isolated, active blocks | T12 | 2.0 (2.0-11.9) | 3.0 (3.0-6.8) | none, the trials abut |
| | T15, T17 | 1.5 | 3.0 | 0.5 |
| | T16 | 2.0 | 3.0 (3.0-4.0) | none |
| isolated, passive blocks | T12 | presentation 2.0 (1.5-11.8) | none | 1.5 (1.5-5.2) |
| | T15, T17 | presentation 1.5 | none | 1.0, 0.5 after a block's last trial |
| | T16 | presentation 1.5 (1.5-2.0) | none | 1.5, none after a block's last trial |
| interleaved | T12 | 1.8 (1.8-8.1), listening trials presentation 1.8 (1.8-8.0) | 3.0 (3.0-6.4) | after listening trials only, 3.0 (3.0-6.4) |
| | T15, T17 | 1.5 (0.8-1.7) | 3.0 (3.0-3.3) | 0.5 |
| | T16 | 2.0 | 2.0 | 1.5 |
| sentenceData | T12 | 3.5 (3.0-113) | 5.0 (1.6-17.3), self-paced | none in the open-loop files, 2.1 (1.6-4.0) in the `_eval` file |
| | T15 | 4.0 | 11.6 (4.3-79.5), self-paced | 2.8 (0.5-195) |
| | T16, 20 ms files | 4.0 | 5.8 (1.3-24.2), self-paced | 2.0 (2.0-42.0) |
| | T16, 33 ms `_eval` file | 0.17 (0.13-0.20), a stub (see Notes) | 7.1 (5.0-13.0), self-paced | 7.2 (5.4-10.4) |
| chitty | T12, open-loop | 4.0 | 12.3 (3.7-33.3), self-paced | none |
| | T12, `_eval` | 5.0 (4.0-5.1) | 20.8 (4.7-36.3), self-paced | 1.5 (0.7-2.5) |
| counting | T15 | grid 48.9 (27.4-95.2) | report 5.9 (4.4-47.5) | 1.0 |
| | T16 | grid 32.1 (26.2-50.1) | report 8.6 (6.7-23.2) | 1.0 |
| three-element arrows | T12 | 3.0 (3.0-14.1) | 4.0 (4.0-8.1) | none |
| single arrow | T12 | 2.5 (2.5-13.1) | 2.5 (2.5-6.5) | none |
| three-element lines | T12 | 2.5 (2.5-12.7) | 4.0 | return period 1.5 (1.5-1.54) |
| verbal and visual memory | T12 | 2.5 (2.4-2.5) | 4.0 | return period 1.5 (1.5-1.52) |
| speech recall | T12 | 3.5 | 4.5 (4.1-4.5) | none |
| every sequence-recall task | T16 | 2.0 | 4.0 | 1.0 (1.0-1.04) |

The counting "delay" is the silent count over the grid, which the participant ends at will. After the go period of the lines and memory tasks, T12 had a 1.5 s **return period** to bring the pen tip back to the start location (STAR Methods). The Study emits it as an `Intertrial` with `description="return period"`. Table S4 gives T12's three-element arrow task a 2.5 s delay and a 3 s go period, whereas the file holds 3.0 s and 4.0 s. The Study follows the file. The long maxima are covered in Notes.

## Data Modalities

| Modality | Source variable | Type, shape, rate |
|---|---|---|
| Threshold crossings | `binnedTX` `(T, F)` | LRR-denoised crossing counts. Thresholds are -4.5 x SD for T12 and T15 and -3.5 x SD for T16 and T17. Stored as integer counts in the word, sentence and counting families, and as `int8` (T12) or `float64` (T16) in the mean-subtracted sequence-recall families |
| Spike-band power | `spikePow` `(T, F)` | Mean squared >250 Hz voltage per bin (µV², LRR-denoised). The paper describes a sum per bin, whereas the deposit READMEs describe a mean, and the released values follow the deposit. Stored as `float64` in the raw families and as `float32` or `float64` in the sequence-recall families |
| Trial epochs | `delayTrialEpochs`, `goTrialEpochs` `(N, 2)` | 1-based **inclusive** [start, stop] bin of the delay and go periods |
| Cue metadata | `trialCues` + `cueList` / `sentenceText` / `numTargets` | word and behavior / sentence / counted-target labels per trial |
| Channels | `chanSets`, `chanSetNames` | per-array channel indices (1-based) and the lab's labels, kept as source labels only (see Channel layout) |
| Blocks | `blockNum` `(T,)` | block ID per bin (absent from the sequence-recall files) |

Threshold terminology differs across the references: the paper describes -3.5 and -4.5 times the standard deviation as site defaults without naming the sites, the deposit READMEs and Deo et al. (2026) use "RMS" for the same quantity, and Deo et al. (2026) states the value per participant. The emitted rows use SD. Although the source metadata labels the raw-family integer arrays as MATLAB `double`, `scipy.io.loadmat` returns them as `uint8`.

Bin size by file: 20 ms (21 files), 10 ms (13), 40 ms (the five T16 sequence-recall files, inferred, see Notes), 25 ms (1) and 33 ms (1). Timelines therefore run at 50 Hz (149), 100 Hz (74), 25 Hz (17), 40 Hz (2) or 30.3 Hz (1). `F` is the stored width, 128 or 256, and every stored column is an electrode.

The files use two normalization regimes. The **word, sentence and counting families** contain un-normalized counts and µV² power. The **sequence-recall families** contain block-wise **mean-subtracted** features (both streams include negative values and have means near zero), so their scale is not directly comparable with the other families.

**Not in the deposit.** The paper records joystick and stylus trajectories in the T12 sequence tasks and confirms the final count in the counting task, but the release holds neither the kinematics nor the correctness. Audio cue waveforms, microphone audio and the contents of the counting grids (shapes, colors, positions) are also absent. Three sets of recordings are missing entirely: the prompted verbal and autobiographical thought data of Figure S5 (`t15.2024.12.15` and `t16.2024.12.18` in Table S3), the sentence sessions that trained the seven real-time decoders (Table S7, 11-28 sessions per decoder, reaching back to 2022-07-05 for T12, 2023-08-11 for T15 and 2024-01-08 for T16, most of them absent here), and the diagnostic or rest block that opened every session. The Study can only place the cue text and the task periods of the eleven released task types.

## Source Data Format & Layout

```
data/Kunz2025InnerSpeech/                    (dataset directory expected by the Study)
├── README.md                               (top-level Dryad description)
├── isolatedVerbalBehaviors/     32 .mat     (per participant: 7 per-behavior + 1 *_allBehaviors)
├── interleavedVerbalBehaviors/   4 .mat
├── sentenceData/                15 .mat
├── chittychittybangbang/         4 .mat
├── conjunctiveCounting/          2 .mat
├── seqRecallSpeech/              2 .mat
├── seqRecallUninstructed3ElementArrow/  2 .mat
├── seqRecallUninstructed3ElementLines/  2 .mat
├── seqRecallUninstructedSingleArrow/    1 .mat
├── seqRecallVerbalMemory/        3 .mat
└── seqRecallVisualMemory/        2 .mat
```

The deposit contains **69 `.mat` files** (MATLAB v5/v7, read with `scipy.io.loadmat`), named `t<NN>.<YYYY.MM.DD>_<task>[_<vocab>][_eval].mat` and ranging from 5 MB to 820 MB. Each file contains one task recorded on one day and may concatenate several recording blocks. Samples are uniformly spaced within a block, but time between blocks was omitted. Adjacent matrix rows on opposite sides of a block boundary are therefore not continuous in real time.

Every folder includes its own field reference (`<folder>_readme.txt` in the word, sentence, chitty and counting folders, and `README.txt` in the six sequence-recall folders). These files describe the source variables and state that the sequence-recall features were mean-subtracted block by block. Where they disagree with the files or the paper, see Source discrepancies.

**Indexing and quirks.**

- Epochs and `chanSets` are 1-based MATLAB indices, and `utils` converts them to 0-based. `utils.get_trials` clamps every epoch to the recording, cuts a go span at the next trial's delay start (the inclusive go stop equals the next delay start in the chitty and T16 isolated files), and raises if a trial does not fit or overlaps its predecessor.
- The epoch stop is read as inclusive, which makes a nominal period one bin longer than its round value: 3.01 s go periods and 0.49 s intertrials at 100 Hz, and a `sentenceDurations` one bin shorter than the go span on every sentence trial. The half-open reading would move each stop by one bin (10-20 ms).
- Three **T12 sequence-recall** files (`joystickSeq`, `lineSequenes`, `joystickSingle`) store block-mean-subtracted `binnedTX` under a `uint8` class tag. `utils._load_stream` reads those bytes as `int8`: the companion `spikePow` is signed and centred, and the reinterpreted values match the sibling `int8` files. The neural rows note this in `processing`.
- The five **T16 sequence-recall** files store `binSize=20` but are read at 40 ms, and their stored epochs are analysis windows rather than task periods. See Notes.
- One T12 interleaved file (`t12.2024.04.11`) stores a byte-identical copy of `binnedTX` as `spikePow`. Its `BandPower` rows are emitted with `valid=False`.
- Every stream is checked against its declared normalization on load (raw counts are non-negative integers, mean-subtracted streams are centred), and a mismatch raises.

## Study, timelines and events

`study.py` defines `Kunz2025InnerSpeech`. **The Study emits one timeline per recorded block, for 243 timelines in total.** Because neural activity between blocks was not recorded, each timeline represents the largest span known to be continuous.

For the 29 word, sentence and counting files, `utils.block_segments` reads 162 source block numbers from `blockNum`. The 12 sequence-recall files have no `blockNum`, and their 81 blocks are defined by the checked-in `utils.RECOVERED_SEQ_BLOCK_STOPS` table described in Notes.

Source block numbers are preserved where available. Sequence-recall block IDs instead give each block's position within its file, and the chronological order across task files recorded on the same day is unknown. `session` is the file stem, so several timelines may share a session. The seven per-behavior `isolatedVerbalBehaviors` subsets are omitted in favour of their combined `*_allBehaviors` file, which is their exact concatenation. All times are relative to the start of the block. Each timeline emits:

| Event type | Source | Span | Notes |
|---|---|---|---|
| `SpikeCounts` | `binnedTX` | whole block, `(n_ch, n_bins)` at 25-100 Hz | `feature="threshold_crossings"`, listed first so `aggregation="cat"` gives the native `[binnedTX \| spikePow]` order |
| `BandPower` | `spikePow` | whole block | `feature="spike_band_power"`, `lower_freq=250`, `upper_freq=5000` |
| `Stimulus` | every active trial | delay period, then one bin at go | The cue over the delay (`period="delay"`), with `stimulus_kind` `word_cue`, `sentence_cue`, `speech_delay_cue`, `movement_cue`, `counting_grid` or `rest_cue`, `description` = cue, and `modality` `visual`, or `audio` for T12's `seqRecallSpeech`. Then a one-bin go marker (`period="go"`, `speech_go_cue` or `movement_go_cue`, `audio` in the three-element lines task). The self-paced counting task has no go marker. `task_condition` is `imagined` on the delay of the verbal-memory strategy and `na` otherwise |
| `Word` | word-cue trials | go period, or the delay epoch for passive trials | `text` = word, `task_condition` (attempted, mimed, imagined, listening, read), canonical `behavior` (`attempted`, `mimed`, `imagined_motor`, `imagined_auditory`, `imagined_listening`, `listening`, `reading`), `modality` (`heard` for listening, `imagined` for inner speech, `read` otherwise) |
| `Sentence` | sentence trials | go period | `text` (see Keyword trials), `task_condition`, `behavior` (the strategy, see Behavioral Task), `keyword`, `modality` (`imagined` for inner speech, `read` for attempted) |
| `Action` | counting, sequence-recall and rest trials | go period | Sequence recall: `action_kind` `joystick` (T12 arrow tasks), `drawing` (T12 lines and memory tasks, every T16 movement task, attempted) or `speech`, `description` = cue, `strategy` (`verbal`, `visual`, `uninstructed`, `speech`), `task_condition="attempted"`. Counting: `action_kind="speech"`, `num_targets`, the spoken report of the silent count. Rest trials of every family: `action_kind="rest"`, `code=0`, `description="DO_NOTHING"`, `task_condition="na"` |
| `Intertrial` | the gap between two trials of a block | from the end of one trial to the next delay onset | `period="intertrial"`, `task_condition="na"`, `description` `"return period"` in T12's lines and memory files, `"intertrial interval"` elsewhere. After a passive trial it starts at the source's go epoch |

Every trial row carries `trial_id` (row of the file's epoch tables, unique within a session), `period` (`delay`, `go`, `presentation` or `intertrial`) and `cue_text` (the verbatim source cue). The delay, unit and intertrial rows of a block tile it from the first delay onset to the last unit's end. The per-period durations are in the Behavioral Task table.

**`code`.** On word cues, `code` is the 1-based index into `utils.WORDS` (1-7) and 0 on rest cues. On sequence-recall rows it is the index into that file's own `cueList`, so it is file-local: the same number names different sequences in different files, and `description` is the key for pooling. On counting rows, `code` equals `num_targets` and is comparable across files. `code=0` identifies a rest trial everywhere. Sentence cues and go markers have no class and carry neuralset's `-100`.

**Passive trials.** The isolated listening and reading blocks and T12's interleaved listening trials have no delay or go cue. The word is presented during the source's delay epoch, so the emitted `Word` spans that epoch with `period="presentation"` and no cue `Stimulus`. The `Intertrial` runs from the source's go epoch to the next trial's delay onset. This follows Table S4 and the authors' `Fig1D`, `Fig1EF` and `Fig5_ABCD.ipynb` scripts, which analyse these trials over the delay window (the MATLAB `Fig5E` and `Fig5FGH` scripts go-align the same T12 interleaved trials). T12's unlabelled `$0` interleaved cues represent listening, following `Fig5_ABCD.ipynb`. The interleaved listening trials of T15, T16 and T17 are go-aligned by the same notebook and are emitted as active trials. In the interleaved files, a bare `imagined` cue maps to `imagined_motor` for T12 and `imagined_auditory` for T16. T16's isolated passive blocks store a one-bin go stub instead of Table S4's 1.5 s intertrial, so their 275 `Intertrial` rows run from that stub to the next trial's onset (1.50-1.51 s, which agrees with Table S4), and a block's last passive trial has none.

**Keyword trials.** The 344 chitty trials whose source cue opens with `*` asked T12 to imagine the unlock keyword before the sentence. Their `text` is the full intended utterance, `utils.KEYWORD` as one token followed by the sentence (`"chittychittybangbang <sentence>"`, as in the authors' decoder corpus), `keyword=True` selects them, and `cue_text` keeps the verbatim `* ...` cue. A prompt comparison against another dataset strips the prefix first.

**Timeline attributes**, broadcast to every row:

| Column | Value |
|---|---|
| `subject` | `T12`, `T15`, `T16` or `T17` |
| `session` | the file stem, shared by the blocks of one file |
| `block` | the source block number, or the block's position in its file for sequence recall |
| `date`, `post_implant_day` | the file's date, and days since the registry implant date |
| `experiment` | the source folder, one of the eleven task types |
| `vocab` | `50w` or `125k` for `sentenceData` and chitty, `na` elsewhere |
| `loop_type` | `closed` for the `_eval` files, `open` elsewhere |
| `original_split` | `train` or `eval` for `sentenceData` and chitty, empty elsewhere |
| `mean_subtracted` | `True` for the sequence-recall families |

Trials can also be filtered by `task_condition`, `behavior`, `period` and `keyword`. `NeuralFeatureExtractor` selects the neural streams by event type, one at a time or both together with `aggregation="cat"`. The Study reuses the package's `SpikeCounts`, `BandPower` and `Intertrial` classes, together with the built-in `Word`, `Sentence`, `Stimulus` and `Action` classes.

**`original_split`.** `train` or `eval` for the two sentence families, following the deposit's own naming, and empty elsewhere because no other experiment publishes a split. `_eval` reliably means closed-loop decoding, but three `_eval` files hold more trials than the paper's formal evaluation sets: T12's 50-word file has 100 trials in two blocks against 50 in the paper (its first block repeats a prompt from the open-loop set), T15's 2025-01-05 125k file has 40 against 25, and T16's 2024-12-18 125k file has 35 against 29. The extra trials are the preliminary closed-loop blocks that let the rolling z-scoring settle after decoder training (STAR Methods). For T15 those blocks also retrained the decoder online, T16 retrained online only in the 125k evaluation, and T12 never did. The release does not mark which trials formed the reported sets, so `original_split` is the source grouping, not a leakage-checked partition. The 50-word `_eval` prompts also appear in `willet_2023` (see Correspondence with other datasets).

### Channel layout

The Study emits every stored column in `utils.ARRAY_LAYOUTS` order, truncated to the width of the file. It does not derive the layout from the file because the in-file labels are inconsistent:

| Files | Stored width | Study arrays (columns) | In-file `chanSetNames` |
|---|---|---|---|
| T12, twelve word and sentence files (2023 and 2024.04.11) | 128 | `ventral_6v` 0-63, `dorsal_6v` 64-127 | `i6v`, `s6v` |
| T12, seven sequence-recall files | 256 | `ventral_6v`, `dorsal_6v`, `ventral_44` 128-191, `dorsal_44` 192-255 | `i6v`, `s6v` (`chanSets` lists the first 128 columns only) |
| T12, four December 2024 chitty files | 256 | `ventral_6v`, `dorsal_6v`, `ventral_44`, `dorsal_44` | `i6v`, `M1`, `55b`, `s6v` |
| T15, all ten files | 256 | `ventral_6v` 0-63, `4` 64-127, `55b` 128-191, `dorsal_6v` 192-255 | `i6v`, `M1`, `55b`, `s6v` |
| T16, eight word, sentence and counting files | 256 | `pef_55b` 0-63, `6v` 64-127, `ventral_6d` 128-191, `dorsal_6d` 192-255 | `PEF`, `i6v` (or `ventral6v`), one 128-column `6d` group |
| T16, five sequence-recall files | 256 | same | `PEF/55b`, `i6v`, `i6d`, `s6d` |
| T17, both files | 256 | `ventral_6v` 0-63, `dorsal_6v` 64-127, `ventral_55b` 128-191, `dorsal_55b` 192-255 | `s6v-v`, `s6v-d`, one 128-column `55b` group |

The in-file `i` (inferior) and `s` (superior) prefixes map to `ventral_` and `dorsal_`. T12's area 44 pair follows the paper, Deo et al. (2026), the authors' Figure 1 script and `willet_2023`. Columns 128-255 of the sequence-recall files contain genuine recordings and are uncorrelated with columns 0-127.

The December 2024 T12 files come from a different rig. T12's sessions before December 2024 ran on Simulink Real-Time with task software in the Psychophysics Toolbox, and T12 from December 2024, T15, T16 and T17 ran on BRAND (STAR Methods, Data collection rig). The December T12 files therefore carry the BRAND configuration's T15 labels (`M1`, `55b`), and the chitty `_eval` file uses BRAND's 25 ms bin. Per-electrode statistics indicate that they contain the same four arrays in the same column positions as T12's earlier sequence-recall files, so the Study treats the labels as a relabel.

The ventral-before-dorsal order of T12's area 44 pair is inferred from the `willet_2023` ordering convention and a docstring in the authors' Python code (`inner_speech/analysis/subfunctions/utils.py`). No source states it directly, and the same docstring lists the 6v pair in the reverse of the in-file labels, the authors' MATLAB channel sets and the Study layout, so it is weak support.

T17's files label columns 128-255 as a single `55b` group. Channel correlations of spike-band power split it at column 192 (62 of 64 and 46 of 64 channels correlate best with their own half), so the Study emits two arrays. The assignment of `ventral_55b` to columns 128-191 and `dorsal_55b` to 192-255 is assumed from `jude_2026_speech`'s order for the same participant, the same ventral-then-dorsal order as the 6v pair in both datasets. No source publishes it.

### Correspondence with other datasets

The four participants appear in six other datasets of this corpus. Pool by array or area, never by column index, unless stated below.

| Participant | Other datasets | What is established |
|---|---|---|
| T12 | [`willet_2023`](../willet_2023/README.md) | Same implant, array names, areas and ontology IDs. `willet_2023` emits `dorsal_6v, ventral_6v, dorsal_44, ventral_44`, and this Study emits the ventral array first in each pair. No electrode-level mapping is established |
| T15 | [`card_2024`](../card_2024/README.md), [`wairagkar_2025`](../wairagkar_2025/README.md), [`singer_2025`](../singer_2025/README.md) | Same implant, array names and the same array per block of 64 columns (`ventral_6v`, `4`, `55b`, `dorsal_6v`). No source states an electrode-level mapping. The 200 trials of `t15.2024.03.03_attempted_50words` are also `train` trials of `card_2024`, see below |
| T16 | none | |
| T17 | [`jude_2026_speech`](../jude_2026_speech/README.md), [`jude_2026_typing`](../jude_2026_typing/README.md) | Same implant. Those Studies emit six arrays (6v, 55b, 6d pairs), this release holds the 6v and 55b pairs only. No published correspondence between the column orders |

**Trials shared with `card_2024`.** The 200 attempted-speech trials of `t15.2024.03.03_attempted_50words` are 200 of the 219 `train` trials of `card_2024` session `t15.2024.03.03`, released there as z-scored 20 ms clips that start about 1 s before the go cue. Each pair has the same cue, and its threshold crossings, rebinned to 20 ms, correlate at 0.97 or more, against at most 0.12 for other trials. The imagined-speech files of that day are not in `card_2024`. A collection that pools the two datasets holds these trials twice, so keep one copy or assign both to the same split.

**Shared sentence prompts with `willet_2023`.** The three 50-word `_eval` files (T12, T15, T16) use the fixed 50-sentence evaluation set of Moses et al. (2021), which `willet_2023` also uses. 152 kunz `Sentence` rows carry one of those 50 prompts (50 per participant, and 2 more in the first block of T12's file), and 1,076 `willet_2023` rows over 21 sessions carry them too. Across all sentences, 303 distinct texts (431 kunz rows) are shared between the two datasets. The neural data are distinct, but a collection that trains on one dataset and evaluates on the other has seen the target sentences, so a pooled split must group prompts across datasets (strip the keyword prefix first).

## Notes / Scope

- **Download.** `Kunz2025InnerSpeech.download()` fetches and extracts the ~10.8 GB Dryad deposit (12 assets) into `data/Kunz2025InnerSpeech/` through the shared `datasets/dryad.py` backend (needs `NEURALFETCH_DRYAD_TOKEN`). Feature matrices are read one stream at a time, and an 820 MB file costs about 0.9 GB of peak RAM per stream.
- **Out of scope:** `decodedSentenceText` (real-time RNN and language-model output) and `sentenceFeatures` (a go-aligned `[binnedTX | spikePow]` convenience copy) are present in some files but never emitted as events.
- **Sequence-recall features are mean-subtracted** and the other families are raw, so analyses that pool across families must normalize them first. `mean_subtracted` on the timeline and `normalization` on the neural rows identify the regime.
- **Filter dense rows on `valid`.** The `BandPower` rows of `interleavedVerbalBehaviors/t12.2024.04.11` carry `valid=False` with the reason in `valid_description`.
- **Read the rate per timeline.** Five rates coexist, so set the extractor `frequency` per subset or resample.
- **Removed trials.** The authors removed trials interrupted by coughing, care needs, a person entering the room or loud noise, and in the 50-word sentence sessions they also removed the matching sentence of the other behavior (STAR Methods). This explains trial counts below the planned ones, such as the 170 trials (19-23 per cue) of T12's `seqRecallSpeech` file and the T15 and T16 sentence counts that do not reconcile with Table S7.
- **LRR schedule.** Thresholds and linear regression referencing were fit on a diagnostic or rest block at the start of each session. For T15's sentence sessions they were recalculated after every block. The `processing` string of the neural rows states the schedule per file.
- **T16 33 ms `_eval` file.** In `sentenceData/t16.2024.04.29_imagined_50words_eval.mat` every delay epoch is 4-6 bins (0.13-0.20 s) and the gap before the next trial is 5.4-10.4 s, so the stored delay is a transition stub and the sentence cue was most likely shown during the preceding gap. No released timestamp identifies the cue onset, so the Study emits the stub as stored (`sentence_cue` rows of 0.13-0.20 s) and the gap as an `Intertrial`. Do not use these 50 delay rows as cue windows. The go periods are source-backed.
- **Long pauses between sentence trials.** 31 sentence intertrials last over 30 s: 25 in T15's files (up to 195 s) and 6 in T16's 125k files (39-42 s). The release gives no reason, and the Study emits them as they are.
- **Recovered sequence-recall blocks.** The deposit explains that these tasks were recorded in separate blocks, interleaved with other sequence-recall tasks, and that data between blocks were omitted. Because the files do not include `blockNum`, the boundaries cannot be read directly. However, both neural streams were mean-subtracted separately within each block. Consequently, the cumulative sum of `spikePow` returns close to zero at every block boundary, to about 0.1% of its typical magnitude in T12 and within floating-point precision in T16. `utils.recover_block_stops` performs this analysis offline, and the recovered boundaries are stored in `utils.RECOVERED_SEQ_BLOCK_STOPS`. Normal timeline enumeration therefore remains lightweight and never loads a neural feature matrix.

  The recovery identifies 81 blocks:

  - **T12, 64 blocks:** 28 joystick-sequence blocks (14 with 32 trials and 14 with 33), 17 lines blocks with 49 trials each, 3 single-arrow blocks with 50 trials each, 4 blocks with 45 trials in each of the three memory files, and 4 speech blocks containing 50, 40, 40, and 40 trials.
  - **T16, 17 blocks:** 4 blocks with 40 trials in each of the four 160-trial files, plus one 60-trial speech block.

  Trial metadata is not used to recover the boundaries, but independently confirms them: every T12 block begins at a source delay onset, every T16 block begins 50 bins before a go cue, and no trial crosses a boundary. The recovered spans are therefore the spans the authors centred, which the deposit describes as the recording blocks.

- **Long first and last trials.** In T12's joystick-sequence, single-arrow and lines files, the first trial of every block has a source delay epoch of 12-14 s rather than the nominal 2.5-3.0 s, and so does the first trial of every T12 sentence block (12.5-13.9 s, 113 s in the second block of the `_eval` file). In the two joystick files, the final trial has a 6-8 s source go epoch rather than the nominal 2.5-4 s. In the lines file, each block contains another 3-5 s of recording after the final go epoch. These durations are consistent with recording lead-in or tail being included around the boundary trials, but the release provides no separate cue-onset or movement-end timestamps. The Study therefore preserves the source epochs instead of applying an inferred trim. For uniform windows, take the nominal delay before the go cue and the nominal go period after it.

- **T16 sequence-recall files use a 40 ms clock.** The five T16 `seqRecall` files store `binSize=20` and include `20ms` in their names, but the Study reads them at 40 ms (`utils.T16_SEQ_BIN_MS`, also recorded in the neural rows' `processing`). This is a curation inference rather than an author-confirmed correction.

  Three independent observations support the 40 ms interpretation:

  - **Task geometry.** Every block begins 50 bins before its first go cue, ends 99-100 bins after its last go cue and advances by 175-176 bins per trial. At 40 ms, these values reproduce Table S4's 2 s delay, 4 s go period and approximately 1 s intertrial interval. At 20 ms, the resulting 3.5 s trial cycle could not contain the documented 6 s task period.
  - **Published analysis.** The authors' notebook extracts `TRIAL_WIN = [-50, 100]` bins. STAR Methods describes the T16 decoding window as 0.75 s under the 20 ms label, whereas the Figure S3 caption describes the same window as 1.5 s, a factor-of-two discrepancy consistent with a stale 20 ms time base.
  - **Count spread.** The files are mean-subtracted, so their mean counts cannot be compared, but their spread can. In the three files recorded on 2024-07-17, the per-channel variance of `binnedTX` is 1.8-1.9 times the mean count of T16's same-day 20 ms interleaved file (median over channels). Within that 20 ms file the ratio is 1.05, and 2.2 when its bins are summed in pairs, so the sequence-recall counts behave like a bin twice as long.

  The stored epochs are analysis windows extending 100 bins before and 150 bins after each go cue, so they overlap neighboring trials and do not represent the task periods. The Study instead emits the Table S4 periods: 50 bins before and 100 bins after each go cue, clipped to the block, with an `Intertrial` over the 25-26 bins between trials. To follow the stored 20 ms label instead, halve all times and double all rates on the 17 affected timelines, which `events[(events.subject == "T16") & events.mean_subtracted]` selects.

### Source discrepancies

The files are the authority for the Study. Where the references say otherwise:

| Fact | References say | Files hold | Study uses |
|---|---|---|---|
| Figure numbers | Table S3 and the interleaved folder readme cite pre-publication numbers (the interleaved behaviors as Figure 5, a "Supp Fig X") | | the published numbers (the interleaved behaviors are Figure 6) |
| Bin sizes | the folder readmes list 10 or 20 ms | also 25, 33 and (T16 sequence recall) 40 ms | the files, with the T16 inference above |
| `delayTrialEpochs` | the sentence, chitty and counting readmes define it as the "go period", a copy of the `goTrialEpochs` line | a delay epoch that ends where the go epoch starts | the delay |
| T12's arrays | the main text counts 2 (the analysed 6v pair) | 4, as in STAR Methods | 4 |
| T17's 6v arrays | Table S5 names them `s6v-A` and `s6v-B` | `s6v-v` and `s6v-d` | `ventral_6v`, `dorsal_6v` |
| T17's interleaved session | Table S3 dates it 2024-12-10 | 2024-12-09 in the file name and the authors' scripts, the same day as the isolated file. The paper's day range 284-287 implies data up to 2024-12-12 | 2024-12-09, day 284 |
| T12 three-element arrow periods | Table S4, 2.5 s delay and 3 s go | 3.0 s and 4.0 s | the files |
| Spike-band power | the paper, a sum of squared voltages per bin | a mean per bin, as the deposit readmes say | the files |
| `upper_freq=5000` | Kunz et al. give only the 250 Hz high-pass. Deo et al. (2026) band-pass T12 at 250-4,900 Hz and T15 and T17 at 250-5,000 Hz, and high-pass T16 at 250 Hz | | 5000, the corpus convention for these rigs |

## Files in this folder

- `__init__.py`: package marker.
- `metadata.py`: `DATASET_METADATA` (subjects, arrays, events) + `sessions()` + `get_metadata()`, validates against the DB schema.
- `utils.py`: source readers, accessors and constants (`list_timeline_files`, `block_segments`, `get_neural_stream`, `get_trials`, `array_layout`, `summary_table`).
- `study.py`: the neuralset `Study` (`Kunz2025InnerSpeech`).
- `assets/`: `experiment.png`, the paper's graphical abstract.
- `01_explore_source_data.ipynb`: tour of the raw `.mat` modalities, trial structure and source quirks.
- `02_explore_study.ipynb`: builds the `Study`, shows the standardized events and reads each event type through its extractor.
- `03_model_example.ipynb`: decodes the 7 cued words per verbal behavior with a plain PyTorch linear classifier and a block-grouped split, with train-only and block-adapted normalization.
