# Wilson 2025 — five years of closed-loop cursor BCI control by a single participant

Closed-loop two-dimensional cursor control by BrainGate2 participant T5, recorded across 103 days between 2016 and 2021. The dataset was assembled to study how intracortical signals drift over long timescales and how a decoder can be recalibrated without supervision, so its defining feature is not any single session but the five-year span of the same two arrays.

## References

- Publication: [Wilson et al. 2025, *Nature Biomedical Engineering*](https://doi.org/10.1038/s41551-025-01536-z), "Long-term unsupervised recalibration of cursor-based intracortical brain-computer interfaces using a hidden Markov model"
- Source archive: [Dryad doi:10.5061/dryad.1jwstqk6g](https://doi.org/10.5061/dryad.1jwstqk6g) (CC0-1.0), downloaded through `datasets/dryad.py`
- Code repository: [github.com/guyhwilson/nonstationarities](https://github.com/guyhwilson/nonstationarities)

## Species & Subjects

| field | value |
|---|---|
| species | human |
| participant | `braingate2/T5` (shared with `willet_2021` and `fan_2023`) |
| sex / handedness | male, right-handed |
| age | 69 years at the time of the study |
| implant date | 2016-08-17 |
| time since implant | 40 days at the first released session, 1804 days at the last |
| clinical conditions | C4 AIS-C spinal cord injury (11 years before the study), tetraplegia |
| implant | 2 × 96-channel Utah arrays (Blackrock Neurotech), hand knob of left dorsal motor cortex (precentral gyrus), 192 channels total |
| stereotaxic coordinates | not published |
| lab | Stanford NPTL / BrainGate2 (ClinicalTrials.gov NCT00912041) |

Channels 0-95 are the lateral array and 96-191 the medial array. The source publishes no per-array anatomical subdivision beyond the shared hand-knob location, so both arrays carry the same `brain_area`.

## Behavioral Task

T5 attempted to move his hand as though controlling a joystick, and that neural activity drove a linear velocity decoder moving a cursor on screen. Every block is a two-dimensional cursor-to-target acquisition task. All three tasks are the same underlying activity: T5 sat in front of a screen, attempted to move his hand as though it were on a joystick, and that neural activity drove a cursor. Every block is "move the cursor onto the target that just appeared". What differs between the three is only where the targets are and how big they are.

| task | blocks | hours | layout |
|---|---|---|---|
| `cursor` | 397 | 30.1 | radial-8 points plus a centre target, and some blocks with randomly placed targets |
| `keyboard` | 103 | 3.2 | a grid of equally sized cells, up to 55 distinct targets per block |
| `fitts` | 60 | 4.9 | Fitts-style randomly placed, randomly sized target at a time |

Trials are back-to-back: a new target appears as soon as the current one is acquired, so there is no meaningful rest period between them. Targets were selected either by dwelling on them or, on 195 of the 560 blocks, by a separate neural click decoder. Overall trial success across the release is high.

**cursor**: Eight targets arranged on a ring like the spokes of a wheel, plus one in the middle. The cursor goes centre, out to a spoke, back to centre, out to the next spoke, and so on. This is the standard motor-BCI benchmark and it is 80% of the release. The ring radius is about 484 px in the 2016 sessions and about 409 px from 2017 onward, so the workspace shrank slightly at some point but is otherwise stable.

**keyboard**: The activity is not typing. The screen is divided into a lattice of equally sized cells, up to 55 of them in a block, and one lights up as the target. This is the one whose coordinates behave differently in the raw files. The other two tasks put the origin at the centre of the screen, but the grid task's coordinates run 0 to 1920 across and 0 to 1080 down, measured from the top-left corner. `utils` re-centres them (see [Notes / Scope](#notes--scope)).

**fitts**: One target at a time, dropped at a random position and at a random size. There is no fixed set of positions to memorize, so almost every trial has a target the participant has never seen in that exact spot.

## Data Modalities

| modality | source location | type / shape |
|---|---|---|
| threshold crossings | field 12 | `(n_bins, 192)` uint16, **firing rate in Hz** at 50 Hz |
| cursor position | field 6 | `(n_bins, 2)` float, screen pixels |
| target position | field 7 | `(n_bins, 2)` int, screen pixels |
| online decoded velocity | field 10 | `(n_bins, 2)` float32 |
| online decoded click | field 11 | `(n_bins, 1)` uint8, present on 195 blocks |
| on-target indicator | field 8 | `(n_bins, 1)` uint8 |
| trial epochs / success | fields 15, 19 | `(n_trials, 2)` and `(n_trials,)` |
| block index per bin / names | fields 3, 18 | `(n_bins,)` and `(n_blocks,)` |

**The neural stream is a rate, not a count.** Every released value is an exact multiple of 50, that is the 20 ms bin count already scaled by 1 / bin size. Mean is about 20 Hz per channel and the peak is 800 Hz (16 counts in a bin). `utils.SPIKE_RATE_SCALE` documents the conversion, and the emitted event carries `unit="Hz"` rather than the `counts/bin` other pre-binned datasets in this corpus use.

**The spike threshold is −3.5 × RMS, not the −4.5 the paper's Methods state.** All 103 released sessions carry −3.5 in their own header. The data is taken as authoritative.

## Source Data Format & Layout

A single 1.7 GB tarball unpacks to roughly 30 GB, most of which is method output (recalibration algorithms) rather than recordings.

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

Each `.mat` holds one recording day as a **positional** MATLAB struct reached as `loadmat(f)["dataset"][0][0]`; fields have no names, only indices. `utils.load_session` is the single place that mapping is written down. Files range from 4 to 130 MB. Two of the 20 positional fields are dead (one always empty, one always all-NaN) and are not exposed.

## Study, timelines and Events

**One timeline is one task block**, not one recording day. A day contains several back-to-back blocks whose task label and online decoder differ. This yields **560 timelines** over 103 days, 38.2 hours and 68,327 trials.

Timeline attributes, broadcast to every row of the block, chosen so the slices a model might want are all reachable:

| attribute | use |
|---|---|
| `session`, `run`, `date`, `post_implant_day` | hold out whole days; the day gap is the axis this dataset exists to study |
| `task` | `cursor` / `keyboard` / `fitts` |
| `original_split` | the release folder, `historical` (2016-2020) or `new` (2021) |
| `analysis_block` | whether the authors' analyses kept this block |
| `click_enabled` | whether a click decoder ran, and so whether a click stream is emitted |

| event type | source | shape / rate | notes |
|---|---|---|---|
| `SpikeCounts` | threshold crossings | `(192, T)` at 50 Hz | unit **Hz**, threshold −3.5 × RMS |
| `CursorPositionSeries` | cursor position | `(2, T)` at 50 Hz | pixels, screen-centred |
| `TargetPositionSeries` | target position | `(2, T)` at 50 Hz | pixels, screen-centred |
| `DecodedCursorVelocitySeries` | decoded velocity | `(2, T)` at 50 Hz | the online decoder's output, which drove the cursor. Unit `a.u.`, with a per-block scale (see Notes) |
| `ClickStateSeries` | decoded click | `(1, T)` at 50 Hz | **only on the 195 click-enabled blocks** |
| `Stimulus` | trial epochs | one per trial | `stimulus_kind="target_cue"`, carries `code`, `target_x`, `target_y`, `target_size` |
| `Action` | trial epochs | one per trial | `action_kind="cursor_control"`, `task_condition="attempted"`, carries `success` |

`code` on the trial rows is the **block-local** index into that block's distinct target centres. A fixed spoke code would be wrong, because target layouts vary from 8 spokes plus a centre to a 55-cell grid to randomly placed targets. Use `target_x` / `target_y` to compare geometry across blocks.

No new Events or Extractors were needed; the dataset is a pure consumer of `neuralshed/events.py` and `neuralshed/extractors.py`.

## Notes / Scope

Verified against the released files rather than the documentation, since several things differ:

- **Neural values are rates in Hz**, not counts per bin (see above).
- **Threshold is −3.5 × RMS**, contradicting the paper's Methods.
- **Field 14 is an undocumented per-block click-enabled flag.** It agrees with the click stream on all 560 blocks, and `utils.click_enabled` exposes it.
- **`t5.2017.08.04` has its pedestal cables swapped**, so channels 0-95 and 96-191 are exchanged relative to every other day. `utils` undoes it, as the reference code does.
- **`t5.2019.01.14` ships duplicated data.** Blocks 24, 27, 28 and 29 each appear twice and the second copy is byte-identical to the first (same counts, same cursor positions, same system clock), about 20 minutes of verbatim duplicate. Only the first copy is exposed, which keeps it out of training data and out of any train/test split. The authors also dropped this day from their analyses.
- **The "intertrial" field is not a rest period.** It never exceeds 2 bins (40 ms) anywhere in the release; it marks trial boundaries. No `Intertrial` events are emitted, since 68,327 rows of 20 ms would carry no information.
- **The `keyboard` task stores positions from the top-left corner of the screen.** Its cursor and target positions have a mean of about (964, 528) px, against about (0, 0) for `cursor` and `fitts`, and the 103 `keyboard` blocks are all from 2016 and 2017. `utils.get_series` subtracts each task's origin (`utils.SCREEN_ORIGINS`: (960, 540) for `keyboard`, (-0.5, -0.5) for `fitts`), as the reference code does, so every task is screen-centred and the positions pool across tasks. The y axis already points the same way in every task.
- **The decoded velocity has no fixed unit.** Regressing the cursor's per-bin displacement on it, leaving out the jumps above 100 px where the cursor resets, gives two regimes. On 163 blocks one unit moves the cursor about 19.8 px per bin (about 990 px/s): every `keyboard` block, the 2016 blocks, two days in early 2017 and two blocks in June 2021. On the other 389 blocks it moves it 65,000 to 139,000 px per bin (5th-95th percentile) and the values are about 1e-4. The direction is right in both, with a median correlation of 0.99. Two days (`t5.2021.06.04`, `t5.2021.06.07`) mix the regimes. Eight analysis blocks are all zeros: `t5.2018.01.08` blocks 0, 1 and 2, `t5.2018.01.17` blocks 2 and 3, `t5.2018.01.19` block 4, `t5.2018.01.22` block 7 and `t5.2019.03.27` block 11. The stream is declared `a.u.` and is comparable across blocks only after a per-block rescale.
- **The system clock restarts at each block**, so it is monotonic only within a block.
- **Trials at block boundaries overhang by exactly one bin** in every case, an artifact of the source's MATLAB-inclusive end convention. They are assigned to the block containing their first bin and clipped, which loses nothing.
- **The authors' block curation is reproduced.** Transcribing `getBlockConstraints` from the reference repo and keeping days with at least two such blocks yields exactly the paper's 73-session cohort, so it is carried as `ANALYSIS_BLOCKS` and stamped on every timeline.

Out of scope, and why:

- `one_month_recal` (2022, 7.7 GB) is a separate closed-loop experiment with a different pickle schema and normalized rather than pixel screen coordinates. Bringing it in means a second loader path for marginal additional decoding value on top of 38 hours of clean cursor data.
- `T5/methods/` and `simulator/` (21 GB) are analysis outputs, not recordings.
- The T11 personal-use data behind Fig. 6 was never released, because it may contain PHI.

Downloads use the shared `datasets/dryad.py` backend and need `NEURALFETCH_DRYAD_TOKEN`. Data is read from local files under `self.path`; nothing is streamed.

## Files

| file | contents |
|---|---|
| `metadata.py` | `DATASET_METADATA`, `sessions()` (one row per recording day), `get_metadata()` |
| `utils.py` | positional-field mapping, `BlockRef`, block and trial windows, `ANALYSIS_BLOCKS`, the cable-swap and duplicate-block fixes |
| `study.py` | `Wilson2025CursorRecalibration`, one timeline per block |
| `01_explore_source_data.ipynb` | the raw release: inventory, units, task geometry, five-year drift, release quirks |
| `02_explore_study.ipynb` | the Study: standardized events, timeline attributes, every event family read through its extractor |
| `03_model_example.ipynb` | displacement decoding, and how it decays as the gap to the training day grows |
