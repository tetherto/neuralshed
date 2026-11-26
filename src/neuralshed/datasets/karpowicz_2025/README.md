# Karpowicz 2025: cursor control across two sessions 274 days apart

Intracortical recordings from BrainGate2 participant **T11**, two 96-channel Utah arrays in the **left precentral gyrus**, on **two sessions 153 and 427 days after implant**. The deposit supports the paper's work on decoder **stability**: NoMAD aligns latent dynamics so a decoder keeps working without supervised recalibration. The paper tests it across days on monkeys, and on these two human sessions within each session, block by block (see How the paper used these sessions).

The task is not the usual center-out. After one radial-8 calibration block, T11 plays a **"Simon" memory-matching game**: four coloured targets illuminate in sequence and he imagines right-hand movements to drive the cursor through that sequence, one target at a time. A released trial is therefore **one movement**, not one sequence, and every movement of a sequence carries the sequence's success or failure.

## References

- **Paper:** Karpowicz et al. 2025, "Stabilizing brain-computer interfaces through alignment of latent dynamics," *Nature Communications* 16:4662, [10.1038/s41467-025-59652-y](https://doi.org/10.1038/s41467-025-59652-y).
- **Source data:** Dryad [10.5061/dryad.q83bk3jtp](https://doi.org/10.5061/dryad.q83bk3jtp), two NWB files, 247.2 MB, CC0-1.0. The monkey datasets in the same paper are a **separate** deposit ([10.5061/dryad.cvdncjt7n](https://doi.org/10.5061/dryad.cvdncjt7n)) and are out of scope here.
- **Code repository:** https://github.com/snel-repo/nomad
- The same recordings were first published in Rubin et al., *J Neurosci* 2022.
- T11's array locations come from Gusman et al., *PNAS* 2026, [10.1073/pnas.2537457123](https://doi.org/10.1073/pnas.2537457123).

## Species & Subjects

- **Species:** *Homo sapiens* (human).
- **Subject:** BrainGate2 participant **T11** (NCT00912041, FDA IDE G090003), a 36-year-old right-handed man with tetraplegia from a C4 spinal cord injury (AIS-B, the grade Gusman et al. 2025 give). Sessions took place at his residence.
- **Implant:** two 96-channel Blackrock Utah arrays in the left precentral gyrus, area 6d (Gusman et al. 2026), 192 electrodes. The deposit names no arrays. Channels 0-95 are taken as the lateral array and 96-191 as the medial one, the names `gusman_2025` uses for the same participant, by inference from two statements of Rubin et al. 2022 about these sessions: activity on the lateral array is sparser, and 39 of the online decoder's 40 features came from the medial array. Here channels 0-95 fire at 12.1 Hz against 30.1 Hz on day 153 and 8.9 against 36.5 Hz on day 427, and the released decoder output can be predicted from 96-191 (held-out R² 0.58 and 0.37) but not from 0-95 (0.005 and 0.03). The online decoder also used spike-band power, which the release does not include.
- **Sessions:** 153 and 427 days after array placement. The same T11 as `vargas_2025` and `gusman_2025`, where he is 36 and 39.

## Behavioral Task

Each session opens with a **radial-8 center-out calibration block** used to fit the session's Kalman filter, cueing eight targets at radius 0.5 (0.35 on the diagonals). After each outer target the centre target is shown (`cond_id` 0, at (0.0012, 0)) and T11 drives the cursor back to it: nothing recentres the cursor automatically in this task. Cursor paths in this block are less direct than in the Simon blocks: median straightness 0.69 against 0.94, the straight-line distance over the path length within each emitted movement window. The calibration is partly computer-assisted: Rubin et al. describe an error attenuation that falls from 0.9 to 0 in steps over the first 90 s of the 5 min routine, after which the participant has full control. The release does not include the attenuation level.

Every later block is the **Simon memory-matching task**. Four coloured targets illuminate one at a time for 400 ms each, with a distinct tone, and T11 imagines moving his right hand to drive the cursor out to each in the order presented. Dwelling 300 ms on the correct target acquires it and the cursor **recentres automatically**. The sequence fails if the cursor dwells on a wrong target or reaches no target within 5 s. A block holds 16 sequences, 12 of them a fixed "target" sequence that repeats across the session's blocks and 4 randomly chosen distractors. T11 was not told a fixed sequence existed.

The released trial is **one of the four movements**, and its window is the trial table's own: from the movement's start to the row where its target is replaced, which is acquisition and the automatic return to the centre in Simon blocks and the end of the dwell on the target in calibration. The paper caps a trial at 600 ms after its start in its analysis, and defines intended velocity over 500 ms, but the release does not apply the cap: 938 of the 1,116 movements last longer than 0.60 s. Simon movements last 0.52 to 3.24 s on day 153 (median 0.80 s) and 0.46 to 4.88 s on day 427 (median 0.64 s), and calibration movements have a median of 2.94 s and 3.08 s. `result` is the **sequence's** outcome, so every movement of a sequence carries the same flag, and a failed sequence keeps only the movements made up to the failure: 26 of the 58 failed sequences keep one movement, 10 keep two, 18 keep three and 4 keep four. The calibration block has no sequences, and the release does not say what `result` means there: one calibration movement has `result` 0 (day 153, trial 4, 4.22 s). 93.2% of day 153's movements and 84.5% of day 427's carry `result` 1.

### How the paper used these sessions

Each session is analysed on its own, and nothing is carried between the two days (Supplementary Fig. 10). A decoder is trained on the session's first Simon block, the second Simon block is the first evaluation, and every later Simon block is evaluated after aligning (NoMAD) or recalibrating (retrospective target inference) on all the blocks before it. The release labels none of these roles. The paper's session 1 (Fig. 5b) is day 427 and its session 2 (Fig. 5c) is day 153. Matching the blocks' start times to the alignment points of the paper's Fig. 5 gives:

| Session | Trains the decoder | First evaluation | Evaluated after alignment |
|---|---|---|---|
| day 153 | block 5 | block 6 | blocks 7 to 14 |
| day 427 | block 7 | block 8 | blocks 9, 12, 14, 15, 16 and 18 |

The "Calib." block of Fig. 5 is this first Simon block, not the centre-out calibration block (block 3), which is not among the blocks of Fig. 5. The Study emits no `original_split`: the release defines no partition, and every block after the first is evaluated once and then trains the next, which a fixed train or test label would misstate. A query on `block` selects these blocks.

## Data Modalities

| Modality | Source location | Type / shape |
|---|---|---|
| Threshold crossings | `acquisition/binned_spikes` | `float64 (n_bins, 192)`, integral counts, up to 17 per bin |
| Cursor position | `acquisition/cursor_pos` | `float64 (n_bins, 2)` |
| Cursor velocity | `acquisition/cursor_vel` | `float64 (n_bins, 2)` |
| Target position | `acquisition/target_pos` | `float64 (n_bins, 2)` |
| Block identity | `acquisition/block_num` | `float64 (n_bins, 1)` |
| Condition identity | `acquisition/cond_id` | `float64 (n_bins, 1)` |
| Trial table | `intervals/trials` | one row per movement: `cond_id`, `tgt_x`, `tgt_y`, `result`, `is_calibration`, `block_num` |

- **Bin size / rate:** 20 ms bins at **50 Hz**, one clock shared by every stream. Mean firing rate is 21.1 Hz per channel on day 153 and 22.7 Hz on day 427, and no channel is silent.
- **Workspace units are arbitrary.** The source calls the unit "position" and publishes no scale. Simon targets sit at radius 0.375, calibration targets at 0.5, and the cursor stays within 0.63 of centre in every block but day 427 block 15, where it reaches 1.225 during movement 347.

### The three source quirks worth knowing

1. **`pynwb` before 4.2.0 cannot open these files.** Their de-identified `session_start_time` has a seconds-resolution UTC offset that `dateutil` refuses ([pynwb #2230](https://github.com/NeurodataWithoutBorders/pynwb/pull/2230)), so `utils.py` reads them with `h5py`.
2. **Timestamps are absolute Unix nanoseconds**, not the seconds NWB nominally stores. Dividing by `1e9` recovers a normal clock.
3. **`cursor_vel` is a displacement per 20 ms bin**, not the `position/s` its unit string claims. Fitting `diff(cursor_pos)` against `cursor_vel` per trial with no intercept gives a slope of 0.99 on day 427 and 1.01 on day 153, so `utils.get_stream` multiplies it by the bin rate and the emitted event declares `a.u./s` truthfully.

### Reading the condition labels

`cond_id` is **block-local**. A calibration block spends 1 to 8 on the eight radial targets, while a Simon block uses only 1 to 4, on the four cardinal targets at a different radius, so `cond_id = 2` means `(0.35, 0.35)` in one block and `(0, 0.375)` in the next. Compare targets through `tgt_x`/`tgt_y`, which every emitted `Action` carries. Note the trial table rounds them to two decimals (0.38) while the continuous `target_pos` stream holds the exact 0.375.

## Source Data Format & Layout

```
karpowicz_2025/data/Karpowicz2025CursorStability/
├── session1.nwb    104 MB   day 427, 63,157 bins, 9 blocks, 472 movements
├── session2.nwb    143 MB   day 153, 86,531 bins, 11 blocks, 644 movements
└── README.md       the deposit's own field list
```

The file names follow the Karpowicz paper's session numbers, so `session1.nwb` is day 427 and `session2.nwb` is day 153. Rubin et al. 2022 number the sessions by date. This package keys sessions by implant day (`T11_day153`, `T11_day427`) instead, and enumerates them in chronological order.

Only the task blocks were released. Block numbers therefore skip (day 427 runs 3, 7, 8, 9, 12, 14, 15, 16, 18), consecutive blocks are separated by gaps of up to 70 minutes, and the 30 minute baseline rest the paper describes is absent. Total released data is 49.9 minutes across 20 blocks and 1,116 movements. The release is also a subset of the Simon blocks. Rubin et al. report 10 Simon blocks of 16 sequences per session. Day 153's 10 blocks match their first session (142 of 160 sequences completed, against their 109 of 120 target and 33 of 40 distractor sequences), but day 427 holds 8 blocks, 128 sequences with 88 completed, against their 160 and 102. Day 153's calibration block lasts 242.44 s by the release clock, against the 5 min routine Rubin et al. describe.

## Study, timelines and events

`study.py` (`Karpowicz2025CursorStability`) treats **each block as one timeline**, 20 in total, following `wilson_2025`. A session is not a continuous recording, but a block is: within one, the 20 ms grid runs unbroken apart from the odd dropped bin, and the task entities (which targets are cued, whether the block calibrates) hold still.

| Event type | Count | Source | Shape / rate | Key attributes |
|---|---|---|---|---|
| `SpikeCounts` | 1 | `binned_spikes` | `(192, n_bins)` at 50 Hz | `unit="counts/bin"`, `normalization="raw"` |
| `CursorPositionSeries` | 1 | `cursor_pos` | `(2, n_bins)` at 50 Hz | `dims=("x", "y")`, `unit="a.u."` |
| `TargetPositionSeries` | 1 | `target_pos` | `(2, n_bins)` at 50 Hz | `dims=("x", "y")`, `unit="a.u."` |
| `DecodedCursorVelocitySeries` | 1 | `cursor_vel` | `(2, n_bins)` at 50 Hz | `dims=("x", "y")`, `unit="a.u./s"` |
| `Action` | per movement | trial table, and `cond_id` for the calibration returns | time span | `period="go"`, `code`, `target_x`, `target_y`, `target_angle_deg`, `success`, `trial_id` (returns: `code`, `target_x`, `target_y` only) |
| `Stimulus` | per Simon sequence | `target_pos` | time span, about 3.1 s | `stimulus_kind="movement_cue"`, `period="delay"`, `code`, `modality="visual"` |
| `Intertrial` | per gap between trials | trial table | time span | `description`, `period="intertrial"` |

**The emitted clock counts bins, not wall time.** The source drops individual 20 ms bins (251 across the release, 246 of them in day 427, and 243 of *those* in its calibration block), so a block's wall span runs up to 4.9 s ahead of `n_bins / 50`. Trial times are resolved against the source timestamps in `utils.trials` and then expressed in bin indices, which keeps every `Action` aligned with the dense stream it indexes into. Each dropped bin is one missing row, 40 ms between stamps instead of 20, and every stream misses the same rows, so neural and behavioural rows stay matched. 51 outward movements and 45 calibration returns contain at least one missing row (up to 7), so they are slightly shorter by rows than by the release clock: day 427 calibration movement 5, rows [2181, 2379), lasts 3.96 s by rows and 4.02 s by the clock. The Study keeps one timeline per block and does not cut at missing rows, which would split those movements for 4.86 s of drift at most.

**Four movements have an estimated start.** In day 427, four down movements (trials 131, 269, 283 and 347, in blocks 8, 14 and 15) carry the `start_time` of the up movement before them, while their own `stop_time` is correct. Both movements of each pair are real: on the row where the up movement ends, the target switches to down and the cursor is recentred. `utils.trials` starts each down movement 4 rows (80 ms) after the up movement ends, the median gap between movements of a day 427 sequence, and its `description` ends in "(start estimated)".

**Some movement windows are one row off the target stream.** The trial table and the target stream sometimes disagree by one row (20 ms) about where a movement starts or ends, and the release does not say which is right. In 47 of the 1,036 Simon windows the last row already shows the next target (for example day 153 block 8, trial 240, rows [3437, 3477), where row 3476 shows the down target). None of them is next to a missing row, and they cluster in day 427 blocks 8, 14 and 16 and day 153 blocks 9 and 11. All 80 calibration windows sit one row early: each starts on the last row of the centre target and stops one row before its outer target is replaced (day 153 block 3, trial 0, rows [475, 617), while its target shows over [476, 618)). The Study keeps the release's windows as they are, with no flag, since one row is about 0.2% of the rows inside movements. With the decoder output as the target nothing is mislabelled, and with the intention label 127 edge rows may point to the wrong target. The calibration offset is also why one row stays an `Intertrial` between each calibration movement and its return.

**Calibration returns are movements.** The trial table lists only the outward movements of the calibration blocks, but each is followed by a cued movement back to the centre target, under decoder control. The Study emits each return as an `Action` with `action_kind="cursor_control"`, spanning the rows where `cond_id` is 0 and ending where the next outward movement starts, with `code` 11 to 18 (10 plus the `cond_id` of the outer target it leaves, so returns from eight directions keep eight labels) and `description` naming the direction ("return to the centre from up-left"). The source gives no trial id or outcome for a return, so those stay empty. That is 81 returns, 33 on day 153 and 48 on day 427, 254 s in all. Rubin et al. count the centre targets as presented targets too. The first return of each block follows an outer target the trial table does not list, and day 427's last return is cut by the end of the block.

**Each Simon sequence is shown, then reproduced.** Before each sequence the task holds the cursor at the centre and the target stream steps through the sequence's four targets, one every 0.78 s (39 rows), then switches back to the first target as the response opens. The whole presentation is one `Stimulus` with `stimulus_kind="movement_cue"`, because what instructs the four movements is the sequence, as in `kunz_2025`'s sequence-recall tasks. `description` gives the order ("sequence shown: left, right, down, up") and `code` the four targets' `cond_id` as digits in that order (3142), the same in every Simon block. All 18 Simon blocks hold 16 presentations, and the first target shown is always the target of the movement that follows. 12 of each block's 16 are the session's fixed sequence (3142 on day 153, 2413 on day 427) and 4 are distractors, as Rubin et al. describe. Presentations last 3.08 to 3.16 s, except one of 3.94 s in day 427 block 14 whose steps are longer in the stream. The paper lights each target for 400 ms with a tone, somewhere inside its 0.78 s step, and the release does not say where, so `modality` is `visual` and the tone is not represented. When the first target is the one already showing, its step shows no change in the stream and is placed 39 rows before the second. When the presentation ends the cursor appears at the centre (Rubin et al. 2022, Fig. 1C), and the time from then until it leaves the centre is an `Intertrial`.

`trial_id` is the source trial table's id, numbered from 0 in each session file (0 to 643 on day 153, 0 to 471 on day 427), so it is unique within a session and not across the dataset. The calibration returns carry no `trial_id`.

Every task row carries `period`: `go` on movements and returns, `delay` on the sequence presentation (as `kunz_2025` marks its sequence cues) and `intertrial` on the gaps between trials. The stretch before a block's first trial and after its last is a lead-in and a tail, not a gap between two trials, so it carries no task row, as in `singer_2025`. In the calibration blocks the lead-in holds an outward movement the trial table does not list (6.88 s on day 153, 3.42 s on day 427).

Timelines carry `subject`, `session`, `block`, `task`, `loop_type`, `post_implant_day` and `path` (the session file), so a query can hold out a whole session or keep only the Simon blocks (`task == 'simonsequence'`). `loop_type` is `assisted` in the two calibration blocks and `closed` in the Simon blocks. Missing rows per block are listed above, not in the timeline labels.

## Notes / Scope

- **Local files only.** A single Dryad deposit downloaded to `<NEURALSHED_DATA_ROOT>/karpowicz_2025/data/`. Dryad requires an API token, so export `NEURALFETCH_DRYAD_TOKEN`. The Study resolves into the class-named subfolder, which is where `_download` lands the files.
- **The bulk arrays are read per block.** `utils` keeps the `h5py` handle open and slices each block's threshold crossings and behavioural streams out of it, so asking for one block never reads the other 19 blocks' arrays. The small arrays are read whole, once per file: the block numbers that give the block boundaries and the trial table.
- **No calendar dates.** `post_implant_day` (153 and 427) is what the paper publishes and what the sessions are keyed on.
- **The decode target is a kinematic**, so `decode_target` is `movement`.
- **Channel order matches `gusman_2025` as a candidate.** No recording is in both releases, and the block numbers suggest no shared day. Across days, columns 0-95 fall on the array `gusman_2025` labels `Lateral` and 96-191 on `Medial` in all eight comparisons, which supports the array names under Species & Subjects. Electrode order within each array is a candidate, and `gusman_2025`'s Gesture Hero sessions disagree with these sessions in crossing rates on columns 0-95. `vargas_2025` is not covered. See [t11.md](../../../../docs/channel_mapping/t11.md).
- **Out of scope:** the monkey isometric-force and reaching datasets analyzed in the same paper, which live in a separate Dryad deposit.

## Files

- `metadata.py`: Metadata for database insertion.
- `utils.py`: Dataset constants and accessors (`list_blocks`, `get_stream`, `trials`, `presentations`, `returns`, `bin_times`, `dropped_bins`, `target_layout`), including the h5py reads and the two unit corrections.
- `study.py`: the neuralset `Study` (`Karpowicz2025CursorStability`).
- `01_explore_source_data.ipynb`: source-data exploration demo.
- `02_explore_study.ipynb`: Study demo.
- `03_model_example.ipynb`: cursor-velocity decoding example.
