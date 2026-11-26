# Wilson 2025: five years of closed-loop cursor BCI control by a single participant

Closed-loop two-dimensional cursor control by BrainGate2 participant T5, recorded across 103 days between 2016 and 2021. The dataset was assembled to study how intracortical signals drift over long timescales and how a decoder can be recalibrated without supervision, so its defining feature is not any single session but the five-year span of the same two arrays.

## References

- Publication: [Wilson et al. 2025, *Nature Biomedical Engineering*](https://doi.org/10.1038/s41551-025-01536-z), "Long-term unsupervised recalibration of cursor-based intracortical brain-computer interfaces using a hidden Markov model"
- Source archive: [Dryad doi:10.5061/dryad.1jwstqk6g](https://doi.org/10.5061/dryad.1jwstqk6g) (CC0-1.0), downloaded through `datasets/dryad.py`
- Code repository: [github.com/guyhwilson/nonstationarities](https://github.com/guyhwilson/nonstationarities)

## Species & Subjects

| field | value |
|---|---|
| species | human |
| participant | `braingate2/T5` (shared with `willet_2021`, `fan_2023`, `deo_2024`, `willsey_2025` and `gusman_2025`) |
| sex / handedness | male, right-handed |
| age | 69 years at the time of the study |
| implant date | 2016-08-17 |
| time since implant | 40 days at the first released session, 1804 days at the last |
| clinical conditions | C4 AIS-C spinal cord injury (11 years before the study), tetraplegia |
| implant | 2 × 96-channel Utah arrays (Blackrock Neurotech), hand knob of left dorsal motor cortex (precentral gyrus), 192 channels total |
| stereotaxic coordinates | not published |
| lab | Stanford NPTL / BrainGate2 (ClinicalTrials.gov NCT00912041) |

Channels 0-95 are taken as the lateral array and 96-191 as the medial array. The source does not say which is which, so the labels are a candidate inferred across the T5 datasets ([docs/channel_mapping/t5.md](../../../../docs/channel_mapping/t5.md)). The source publishes no per-array anatomical subdivision beyond the shared hand-knob location, so both arrays carry the same `brain_area`. Seven of the days are also in `deo_2024` and two in `willet_2021`, with different blocks, so a split that holds out days should keep each of them on one side across datasets.

## Behavioral Task

T5 attempted to move his hand as though controlling a joystick, and that neural activity drove a linear velocity decoder moving a cursor on screen. Every block is a two-dimensional cursor-to-target acquisition task: move the cursor onto the target that just appeared. The three tasks differ in where the targets are.

| task | blocks | hours | layout |
|---|---|---|---|
| `cursor` | 397 | 30.1 | 8 or 6 targets around a centre target, or on one day 6 on a horizontal line through it |
| `keyboard` | 103 | 3.2 | a grid of equally sized cells, up to 57 distinct targets per block |
| `fitts` | 60 | 4.9 | one randomly placed target at a time |

Trials are back-to-back: a new target appears as soon as the current one is acquired, so there is no meaningful rest period between them. Targets were selected either by dwelling on them or, on 195 of the 560 blocks, by a separate neural click decoder. Overall trial success across the release is high.

**cursor**: Outer targets around one in the middle. The cursor goes centre, out to a target, back to centre, out to the next target, and so on. This is the standard motor-BCI benchmark and it is 80% of the release. Most blocks use eight outer targets like the spokes of a wheel: on a circle of 409 px on 284 blocks, and on an ellipse 484 px to the sides and 409 px up and down on 61 blocks from September 2016 to May 2017. Four days of 2019 (`t5.2019.01.14`, `.01.16`, `.01.30` and `.03.13`) use six on a circle of 409 px, 39 blocks, and `t5.2019.04.01` blocks 6 to 16 put six on a horizontal line at ±286, ±573 and ±859 px. Six short blocks of 4 to 24 trials cue only part of a layout, and `t5.2016.09.26` block 19 has no centre target.

**keyboard**: The screen is divided into a lattice of equally sized cells, up to 57 of them in a block, and one lights up as the target. The paper says the data include free-response settings from a typing study (its ref. 3, Pandarinath et al. 2017), so some grid blocks may be typing, but the release does not say which. This is the one whose coordinates behave differently in the raw files. The other two tasks put the origin at the centre of the screen, but the grid task's coordinates run 0 to 1920 across and 0 to 1080 down, measured from the top-left corner. `utils` re-centres them (see [Notes / Scope](#notes--scope)).

**fitts**: One target at a time, dropped at a random position. There is no fixed set of positions to memorize, so almost every trial has a target the participant has never seen in that exact spot. The paper calls these targets "often variably sized", but `target_size` (field 1) is 50 px on all 10,043 `fitts` trials. The on-target flag treats it as a radius on `cursor` and `fitts` blocks (see [Notes / Scope](#notes--scope)).

## Data Modalities

| modality | source location | type / shape |
|---|---|---|
| threshold crossings | field 12 | `(n_bins, 192)` uint16, **firing rate in Hz** at 50 Hz |
| cursor position | field 6 | `(n_bins, 2)` float, screen pixels |
| target position | field 7 | `(n_bins, 2)` int16 or float64, screen pixels |
| online decoded velocity | field 10 | `(n_bins, 2)` float32 |
| online decoded click | field 11 | `(n_bins, 1)` uint8 or float64, present on 195 blocks |
| on-target indicator | field 8 | `(n_bins, 1)` uint8 |
| trial epochs / success | fields 15, 19 | `(n_trials, 2)` and `(n_trials,)` |
| block index per bin / names | fields 3, 18 | `(n_bins,)` and `(n_blocks,)` |

**The neural stream is a rate, not a count.** Every released value is an exact multiple of 50, that is the 20 ms bin count already scaled by 1 / bin size. Mean is about 20 Hz per channel and the peak is 1000 Hz (20 counts in a bin). `utils.SPIKE_RATE_SCALE` documents the conversion, and the emitted event carries `unit="Hz"`, where other pre-binned datasets in this corpus use `counts/bin`.

**The spike threshold is −3.5 × RMS, not the −4.5 the paper's Methods state.** All 103 released sessions carry −3.5 in their own header. The data is taken as authoritative.

## Source Data Format & Layout

A single 1.7 GB tarball unpacks to roughly 30 GB, most of which is method output (recalibration algorithms), not recordings.

```
Wilson2025CursorRecalibration/
└─ PRITpaper2025/
   ├─ T5/
   │  ├─ raw_data/
   │  │  ├─ historical/      83 × t5.YYYY.MM.DD.mat   (2016-09-26 to 2020-02-26, 431 MB)
   │  │  ├─ new/             20 × t5.YYYY.MM.DD.mat   (2021-04-26 to 2021-07-26,  46 MB)
   │  │  └─ one_month_recal/ 11 day folders of .pkl blocks (2022, 7.7 GB)  [out of scope]
   │  └─ methods/            ADAN / HMM / Stabilizer sweep outputs (20 GB)  [out of scope]
   └─ simulator/             performance and HP sweeps (1.2 GB)             [out of scope]
```

Each `.mat` holds one recording day as a MATLAB struct reached as `loadmat(f)["dataset"][0][0]`. Its 20 fields have names (`blockList`, `targetSize`, `cursorSize`, `blockNums`, `sysClock`, `nspClocks`, `cursorPos`, `targetPos`, `onTarget`, `spikePow`, `decVel`, `decClick`, `TX`, `TX_thresh`, `decodingClick`, `trialEpochs`, `instructedDelays`, `intertrialPeriods`, `gameNames`, `isSuccessful`), and `utils.load_session` reads them by position, the field numbers this README uses. Files range from 0.7 to 16.3 MB. Two fields are dead and are not exposed: field 9 (`spikePow`) is always empty and field 16 (`instructedDelays`) is always all-NaN.

## Study, timelines and Events

**One timeline is one section of a task block**, the bins between two recording pauses, or the whole block when it has none. A day contains several back-to-back blocks whose task label and online decoder differ, and 310 of the 560 blocks contain recording pauses (see Notes). This yields **1,014 timelines** from 560 blocks over 103 days, 38.2 hours of stored bins and 68,327 trials.

Timeline attributes, broadcast to every row of the timeline, chosen so the slices a model might want are all reachable:

| attribute | use |
|---|---|
| `session`, `run`, `date`, `post_implant_day` | hold out whole days. The day gap is the axis this dataset exists to study |
| `block`, `section` | the source's block number, and the section within it, numbered from 0 in recording order. Group on `session` and `block` to get a whole block |
| `task` | `cursor` / `keyboard` / `fitts` |
| `original_split` | `""`: the release ships no train/test partition. The paper's offline comparisons split each day's analysis blocks 50:50 by block number, which `session`, `block` and `analysis_block` rebuild |
| `analysis_block` | whether the authors' analyses kept this block |
| `click_enabled` | whether a click decoder ran, and so whether a click stream is emitted |
| `loop_type` | `open` on the two blocks where the computer moved the cursor, `closed` on every other block (see Notes) |

| event type | source | shape / rate | notes |
|---|---|---|---|
| `SpikeCounts` | threshold crossings | `(192, T)` at 50 Hz | unit **Hz**, threshold −3.5 × RMS |
| `CursorPositionSeries` | cursor position | `(2, T)` at 50 Hz | pixels, screen-centred |
| `TargetPositionSeries` | target position | `(2, T)` at 50 Hz | pixels, screen-centred |
| `DecodedCursorVelocitySeries` | decoded velocity | `(2, T)` at 50 Hz | the online decoder's output, which drove the cursor (on `fitts` only in part, see Notes). Unit `a.u.`, with a per-block scale (see Notes). On the two open-loop blocks it holds the computer's cursor command and carries `valid=False` |
| `ClickStateSeries` | decoded click | `(1, T)` at 50 Hz | **only on the 195 click-enabled blocks**. Unit `binary` on the 138 that hold 0 and 1, and the other 57 carry `valid=False` (see Notes) |
| `Stimulus` | trial epochs | one per trial | `stimulus_kind="target_cue"`, carries `code`, `target_x`, `target_y`, `target_size`, and `trial_kind` on `cursor` blocks with a centre target. On the 18 held-cursor blocks it spans only the hold when the cursor moves, and on 2021 outward trials with a lockout only the lockout, with `period="delay"` and `description` "cued cursor target, delay period" (see Notes) |
| `Action` | trial epochs | one per trial, except the 2021 returns with a lockout | `action_kind="cursor_control"`, `task_condition="attempted"`, carries `success` and the same trial fields as `Stimulus`. On the held-cursor blocks it starts where the cursor first moves, and on 2021 outward trials with a lockout at the first bin of nonzero decoder output, with `period="go"` and ", go period" at the end of its `description`. Its `description` is "attempted hand movement driving the cursor", or "attempted hand movement while the computer moved the cursor" on the two open-loop blocks |
| `AutoMovement` | trial epochs | one per 2021 return with a lockout | `description="return to centre"`, carries `trial_id`. The task moves the cursor back to the centre, so it has no `code` and no `success` (see Notes) |

`code` on the trial rows is the **block-local** index into that block's distinct target centres. A fixed spoke code would be wrong, because target layouts vary from 8 spokes plus a centre to a 57-cell grid to randomly placed targets. Use `target_x` / `target_y` to compare geometry across blocks. `trial_id` is the trial's 0-based row in its session file's trial list, unique within a session.

`code` names the cued target, not the movement. On the 396 `cursor` blocks with a centre target, 22,973 trials return to the centre, from a median of 8 different targets per block, and all share the centre's code. These blocks carry `trial_kind`, `center_out` or `return`, and the target a return leaves from is the previous trial's. The one `cursor` block without a centre target (`t5.2016.09.26` block 19) goes from target to target and has no `trial_kind`.

No new Events or Extractors were needed. The dataset is a pure consumer of `neuralshed/events.py` and `neuralshed/extractors.py`.

## Notes / Scope

Verified against the released files, since several things differ from the documentation:

- **Neural values are rates in Hz**, not counts per bin (see above).
- **Threshold is −3.5 × RMS**, contradicting the paper's Methods.
- **Field 14 (`decodingClick`) is a per-block click-enabled flag** that the release README does not document. It agrees with the click stream on all 560 blocks, and `utils.click_enabled` exposes it.
- **`t5.2017.08.04` has its pedestal cables swapped**, so channels 0-95 and 96-191 are exchanged relative to every other day. `utils` undoes it, as the reference code does.
- **`t5.2019.01.14` ships duplicated data.** Blocks 24, 27, 28 and 29 each appear twice and the second copy is byte-identical to the first (same counts, same cursor positions, same system clock), about 20 minutes of verbatim duplicate. Only the first copy is exposed, which keeps it out of training data and out of any train/test split. The authors also dropped this day from their analyses.
- **The intertrial field (17, `intertrialPeriods`) is not a rest period.** It never exceeds 2 bins (40 ms) anywhere in the release, so it marks trial boundaries. No `Intertrial` events are emitted, since 68,327 rows of 20 ms would carry no information.
- **The `keyboard` task stores positions from the top-left corner of the screen.** Its cursor and target positions have a mean of about (964, 528) px, against about (0, 0) for `cursor` and `fitts`, and the 103 `keyboard` blocks are all from 2016 and 2017. `utils.get_series` subtracts each task's origin (`utils.SCREEN_ORIGINS`: (960, 540) for `keyboard`, (-0.5, -0.5) for `fitts`), as the reference code does, so every task is screen-centred and the positions pool across tasks. The y axis already points the same way in every task.
- **The on-target flag (field 8) and the cursor size (field 2) are not emitted.** The cursor size is 25 px on every block. On `cursor` and `fitts` blocks the flag is 1 exactly when the cursor centre is less than `target_size` + 25 px from the target centre, so it can be recomputed from the emitted positions and `target_size`. On `keyboard` blocks no simple rule reproduces it.
- **The decoded velocity has no fixed unit.** Regressing the cursor's per-bin displacement on it, leaving out the jumps above 100 px where the cursor resets, gives two regimes. On 163 blocks one unit moves the cursor about 19.8 px per bin (about 990 px/s): every `keyboard` block, the 2016 blocks, two days in early 2017 and the two open-loop blocks of June 2021. On the other 389 blocks it moves it 65,000 to 139,000 px per bin (5th-95th percentile) and the values are about 1e-4. The direction is right in both. Two days (`t5.2021.06.04`, `t5.2021.06.07`) mix the regimes. Eight analysis blocks are all zeros: `t5.2018.01.08` blocks 0, 1 and 2, `t5.2018.01.17` blocks 2 and 3, `t5.2018.01.19` block 4, `t5.2018.01.22` block 7 and `t5.2019.03.27` block 11. The cursor still moves there and 89 to 100% of trials succeed, so the zeros are not what moved it, and their `DecodedCursorVelocitySeries` rows carry `valid=False`. Whether the decoder ran with its output unsaved or the computer moved the cursor is not known: on the four January 2018 days the zero blocks open the day, and six of the seven hold 216 to 292 trials at 100% success, against 10 to 36 trials at 60 to 72% on the first decoded blocks. Their paths and trial durations vary from trial to trial as on other closed-loop blocks, and the authors' analyses keep all eight, so they carry `loop_type="closed"`. The stream is declared `a.u.` and is comparable across blocks only after a per-block rescale.
- **On `fitts` the decoded velocity explains only part of the cursor's movement.** It accounts for a median 63% of the variance of the cursor's steps, against 99.9% on the other tasks, and the cause is not known.
- **The clocks restart at each block.** Of the 457 boundaries between consecutive exposed blocks of a day, the system clock runs backwards at 455 and the NSP clocks at 454, so they are monotonic only within a block.
- **Recording pauses cut 310 of the 560 blocks.** Inside a block the system clock (field 4) advances 20 or 21 ms per bin, but 454 times it advances by more (median 9.5 s, up to 193 s, 1.43 h in total), and the NSP clocks (field 5) advance by the same amount: time passed with no bins stored. 296 of them come right after the 0.52 s first trial that opens most blocks. Each block is cut at its pauses (`utils.section_windows`), so the bins of every timeline are evenly spaced. On `t5.2019.03.13` blocks 31 and 32 a trial's last bin comes after a pause. Its `Stimulus` and `Action` end at the pause and carry `valid=False`.
- **About one bin in 80 of every `fitts` block lasts 21 ms.** In all 60 `fitts` blocks, and in no other task, 1.2% of bins (0.15% of the release) last 21 ms instead of 20, and the NSP clocks agree (630 samples instead of 600). The source scales every count by 50 as if each bin lasted 20 ms, so those bins read 5% high. A timeline places bin i at i / 50 s, so by the end of a 5-minute `fitts` block its times run behind the clock by 180 ms (median) to 230 ms. The streams share their bins and stay aligned with each other, and the bins are emitted as released.
- **Channels 0-95 were not recorded at the start of `t5.2018.08.08` blocks 12, 16 and 18.** The NSP clock for those channels (field 5, first column) reads 0 for the first 0.54, 6.12 and 9.48 s of stored bins, while the second advances normally, and channels 0-95 are all zero on 96 to 100% of those bins against 18 to 20 Hz afterwards. Channels 96-191 record normally throughout. The zeros are emitted as released. Recording pauses put these bins in five timelines, section 0 of each block and section 1 of blocks 16 and 18, each holding one trial. Those timelines carry `valid=False`, with the span in `valid_description` ("channels 0-95 not recorded for 5.58 of 10.02 s"). The other timelines of these blocks record both arrays throughout. The authors kept no analysis block on this day.
- **Short all-zero runs on one array, cause unknown.** In seven blocks of 2016 and 2017 (`t5.2016.10.05` blocks 17 and 21, `t5.2016.12.06` blocks 3 and 5, `t5.2016.12.15` block 12, `t5.2017.02.15` block 22, `t5.2017.08.07` block 8), one array reads all zero for 5 to 17 bins (0.10 to 0.34 s) inside a trial, twelve runs in all, while its NSP clock advances normally and the other array reads 18 to 36 Hz. They are emitted as released and not flagged.
- **The click field changes encoding across years.** Targets are selected by a neural click decoder on 195 blocks and by dwelling on the rest, and field 11, which the release README does not document, holds the click decoder's output in three encodings. On the 138 click blocks of 2016 and 2017 it is 0 or 1, and a 1 sits on the last bins of 91% of successful trials in runs of about 3 bins, so it is the click that selects the target. On `t5.2016.10.31` blocks 7 and 9 (`fitts`) it takes continuous values from 0 to 0.989, nonzero on 98% of bins, and targets are selected without it reaching 0.5. The reference code calls them "click probabilities (?)" and rounds them. On the 55 click blocks of 2021 it is 0 or 2, in runs of about 13 bins that carry into the next trial, and 40% of successful trials end with no 2 at all, so 2 does not line up with selections the way 1 does. The values are emitted as released, with unit `binary` on the 0/1 blocks, and the other 57 carry `valid=False` with unit `proportion` (continuous) or `a.u.` (0 and 2).
- **The authors' click analysis drops the end of `t5.2016.10.24`.** Their click decoder code keeps only the day's first 50,000 bins, "day with weird nonstationarity (??)" (`click_utils.py`), which cuts the end of block 38 and all of blocks 41 and 45. Those two blocks have somewhat lower success (98% and 87%) and slower trials, while the neural rates stay steady, so they are emitted as released, and the authors' main analyses keep all ten blocks of the day.
- **Trial windows end at the last bin of their own target.** On 53.9% of trials the source's last bin already shows the next trial's target, so `utils.trial_windows` drops it.
- **On 18 blocks the task held the cursor after each target appeared.** On `t5.2019.04.01` blocks 6 to 16 and 22 to 31, `t5.2021.06.04` block 15 and `t5.2021.06.07` block 14 (`utils.HELD_CURSOR_BLOCKS`), the cursor stays at its start position after the target appears and then moves: for a median of 54 to 106 bins (1.1 to 2.1 s) on outward trials and 13 to 26 bins on returns, while 314 outward trials of `t5.2019.04.01` have no hold. On every other block the cursor of the median trial moves within its first few bins. This looks like an instructed delay, but the source records no cue times (field 16 is all NaN) and the paper does not describe these blocks. On `t5.2019.04.01` the decoder keeps putting out a velocity during the hold, and on the two 2021 blocks its output is zero. On these blocks the `Stimulus` spans the hold with `period="delay"` and the `Action` starts at the first bin the cursor has left its start position, with `period="go"`, so the go is measured from the cursor and not a recorded cue. On the 33 trials of these blocks whose cursor never moves there is no go, so the `Stimulus` and `Action` both span the whole trial, without `period`.
- **Most 2021 trials open with a 0.5 s decoder lockout.** On 50 of the 55 closed-loop 2021 blocks the decoder output is exactly zero for 25 bins from the first or second bin of each trial (`utils.lockout_end`). On the 3,574 return trials with a lockout the cursor travels back to the centre during it and ends within 0.1 px of the centre, and the trial ends after 25 or 26 bins as a success. The task moves the cursor there, so each is an `AutoMovement` (`description="return to centre"`) in place of an `Action`, with no `success`. The 55 first trials of a block also open with 25 zero bins, but their cursor starts on the centre and does not move, so they get a `Stimulus` and an `Action` like the first trial of any other block. On the 3,562 outward trials with a lockout the cursor creeps by under 5 px during it, apart from a one-bin jump to the centre on the 50 trials right after a recording pause, and then T5 drives it. Their `Stimulus` spans the lockout with `period="delay"`, and the `Action` starts at the first bin of nonzero decoder output with `period="go"`. On the other five blocks (`t5.2021.04.26` blocks 5 to 7, `t5.2021.04.28` block 2 and `t5.2021.05.03` block 6) only the first trial has a lockout, and T5 drives the 255 returns as well, which take about 2 s and end 40 to 67 px from the centre. From 2016 to 2020 only first trials of a block open with such a zero run, outside the eight all-zero blocks, and the release does not describe the lockout.
- **Two blocks are open-loop.** On `t5.2021.06.04` block 15 and `t5.2021.06.07` block 14 (`utils.OPEN_LOOP_BLOCKS`) the computer moved the cursor. After the hold every outward path is straight and every trial has the same speed profile, and field 10 equals the cursor's own step at about 40 px per unit, against about 9e4 on the other blocks of these days. No source labels loop type per block. The paper and the release README call the data closed-loop, and the authors' block list leaves both blocks out without giving a reason, so the label is inferred from the cursor data. The deo_2024 readme, from the same lab and year, describes its open-loop blocks the same way: the cursors "moved autonomously to their designated targets". These blocks carry `loop_type="open"`, their `DecodedCursorVelocitySeries` rows carry `valid=False`, and their `Action` rows describe an attempted movement while the computer moved the cursor. The held blocks of `t5.2019.04.01` also have straight paths, but there the decoder output accounts for the cursor with no extra pull toward the target, so they carry `loop_type="closed"`.
- **The first trial of most blocks looks like the task starting the block.** In 308 of the 560 blocks the first trial puts the cursor on the centre target, it never moves, and the decoder output is exactly zero on 301 of them. After 26 bins (0.52 s) the dwell timer counts the target as acquired and the source marks the trial a success, and 310 first trials end at a recording pause. No source describes these trials, so this reading is inferred from the data. On 149 more trials outside the held-cursor blocks the cursor never moves either, although the decoder output is not zero: 128 of them are the trial right after that pause, and 76 run the full 10 s and fail. The release does not say why. With those of the held-cursor blocks and the two cut at a pause, the cursor never leaves its start position on 474 trials, and the source marks 333 of them successful and 141 failed. Each has an `Action` carrying that `success`, like every other trial.
- **The last trial of a block can overhang it by one bin**, an artifact of the source's MATLAB-inclusive end convention: in 311 of the 560 blocks it ends one bin into the next block, and in the other 249 at the block's edge. Trials are assigned to the block containing their first bin and clipped, which loses nothing.
- **21 `cursor` trials last under 6 bins (0.12 s).** The authors' preprocessing drops trials this short as "weird short trials" (`preprocess.py`), and the Study keeps them.
- **The authors' block curation is reproduced.** Transcribing `getBlockConstraints` from the reference repo and keeping days with at least two such blocks yields exactly the paper's 73-session cohort, so it is carried as `ANALYSIS_BLOCKS` and stamped on every timeline.
- **`t5.2019.09.18` is an "array reference issue day" in the authors' code.** `getBlockConstraints` keeps none of its blocks, with that comment. Its two `fitts` blocks (31 and 32) hold 65 and 79 trials at 91% and 95% success, against 138 to 221 trials at 99 to 100% on the `fitts` blocks of neighbouring days, although its mean rates, silent channels and correlation of each channel with its array's mean look like those days'. Every timeline of the day carries `valid=False` (`utils.INVALID_SESSIONS`).

Out of scope, and why:

- `one_month_recal` (2022, 7.7 GB) is a separate closed-loop experiment with a different pickle schema and normalized screen coordinates in place of pixels. Bringing it in means a second loader path for marginal additional decoding value on top of 38 hours of clean cursor data.
- `T5/methods/` and `simulator/` (21 GB) are analysis outputs, not recordings.
- The T11 personal-use data behind Fig. 6 was never released, because it may contain PHI.

Downloads use the shared `datasets/dryad.py` backend and need `NEURALFETCH_DRYAD_TOKEN`. Data is read from local files under `self.path`, and nothing is streamed.

## Files

| file | contents |
|---|---|
| `metadata.py` | `DATASET_METADATA`, `sessions()` (one row per recording day), `get_metadata()` |
| `utils.py` | positional-field mapping, `BlockRef`, block and trial windows, `ANALYSIS_BLOCKS`, the cable-swap and duplicate-block fixes |
| `study.py` | `Wilson2025CursorRecalibration`, one timeline per block section |
| `01_explore_source_data.ipynb` | the raw release: inventory, units, task geometry, five-year drift, release quirks |
| `02_explore_study.ipynb` | the Study: standardized events, timeline attributes, every event family read through its extractor |
| `03_model_example.ipynb` | displacement decoding, and how it decays as the gap to the training day grows |
