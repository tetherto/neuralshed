# Singer-Clark et al. 2025: Speech motor cortex enables BCI cursor control and click

Intracortical 2-D cursor control and click in one BrainGate2 participant (**T15**, a 45-year-old left-handed man with ALS, tetraplegia, and severe dysarthria). Four 64-channel Utah arrays (256 electrodes) in the left ventral precentral gyrus record 10 ms-binned threshold crossings and spike-band power while he uses motor imagery to drive a cursor to targets and select them. The finding is that **speech** motor cortex carries enough movement-intention signal to run a cursor BCI, so the decode target here is movement, not speech.

Same BrainGate2 participant **T15** and the same four Utah arrays as [`card_2024`](../card_2024/README.md), [`wairagkar_2025`](../wairagkar_2025/README.md), and [`kunz_2025`](../kunz_2025/README.md), which are speech deposits from the same lab. The array names, channel ranges, HOMBA and UBERON mappings and MNI array centers in `metadata.py` are shared with those datasets.

## References

- **Paper:** Singer-Clark, T., Hou, X., Card, N. S., Wairagkar, M., Iacobacci, C., Peracha, H., Hochberg, L. R., Stavisky, S. D., Brandman, D. M. *Speech motor cortex enables BCI cursor control and click.* **J. Neural Eng.** 22(3), 036015 (2025). https://doi.org/10.1088/1741-2552/add0e5
- **Source data:** [10.5061/dryad.prr4xgxzq](https://datadryad.org/dataset/doi:10.5061/dryad.prr4xgxzq)
- **Code repository:** https://github.com/Neuroprosthetics-Lab/BCI-cursor-control-from-speech-motor-cortex
- **Supplementary material:** supp. figures 1-4 and supp. movies 1-6, on the article page at the paper DOI (CC BY 4.0).

## Species & Subjects

- **Species:** *Homo sapiens* (human).
- **Subject:** BrainGate2 participant **T15**, male, 45 years old at these recordings, left-handed.
- **Clinical condition:** ALS, tetraplegia, severe dysarthria. He retained eye movement, impaired neck movement, and limited orofacial muscle control.
- **Implant:** four 64-electrode Blackrock NeuroPort Utah arrays (256 electrodes) in the **left ventral precentral gyrus**, implanted 17 July 2023 (from the shared participant registry, as the paper reports only post-implant days 39-468). Source labels `v6v`, `4`, `55b`, `d6v`, mapping to electrode columns 0-63, 64-127, 128-191, 192-255 respectively (verified against `array_label_by_electrode`, which is identical in all 51 blocks). The paper localizes them as ventral premotor cortex (`v6v`, `d6v`, area 6v), primary motor cortex (`4`, area 4), and middle precentral gyrus (`55b`).
- **Channel order.** The 256-wide channel axis is ordered by array: `v6v`, `4`, `55b`, `d6v` (64 each), the same order as `card_2024`, `wairagkar_2025` and `kunz_2025`. No source states an electrode-level mapping, and no recording is shared with those datasets, so column `i` being the same electrode across them is a candidate, see [T15 channel correspondence](../../../../docs/channel_mapping/t15.md#singer_2025).
- **Released sessions:** post-implant days 39, 202, and 468.
- **Lab / institution:** Neuroprosthetics Lab, UC Davis (recordings performed at the participant's home).

## Behavioral Task

Three task families, all cursor control by motor imagery on a screen whose coordinates are centered on the screen with a distance of 1.0 equal to the screen height, `+x` right and `+y` up.

### Radial8 Calibration Task

Eight outer circular targets and one center target were presented in alternating center-out-and-back order. A target turning green was both the target presentation and the cursor go cue. T15 used instructed motor imagery to move toward it. Selection was by dwell on days 39 and 202 (`dwell_requirement_sec` is 0.8 s on day 39 and 1.0 s on day 202 for these radial8 blocks, and 1.5 s in the simultaneous speech blocks) and by attempted click on day 468, where the radial8 blocks carry the click streams too. Calibration blocks could transition from full computer assistance to closed-loop control as the online decoder improved. With click assist on, the computer registered a click once the cursor had been on the target for 1.5 s. Click assist is on at the start of day-468 blocks 00, 06 and 12.

### Grid Evaluation Task

The cursor selected square targets in a densely packed grid using closed-loop velocity and click decoders. The paper describes 6 by 6 and 14 by 14 grids. The 9 released blocks, all from day 468, are the paper's "optimized" grid blocks 9-17: 14 by 14 (`grid_num_rows` is 14, `grid_total_height` is 0.92) with the speed parameter `r` lowered to 0.05. The paper reports 2.90 ± 0.16 bps for them. Per the paper, any click ended a trial, correct or not, and a trial with no click ended at a 10 second timeout. The target the cursor started on could not be selected for 2.0 s or until the cursor left it.

### Simultaneous Speech and Cursor Task

A red target and a word appeared together, and the word was also played aloud by text-to-speech. After a randomized 1.3 to 1.7 second delay the target turned green and the word disappeared, beginning cursor movement. Some trials later played a beep instructing T15 to attempt the prompted word, from memory, while continuing cursor control. In verbal blocks he attempted to speak after the beep, in control blocks he did not. Trials were separated by a one-second intertrial period. The Dryad release contains day 202 blocks (14 verbal, 7 control).

The 21 blocks were recorded in three sets, verbal (A) and control (B) in the order AAB AAB, AAB AAB AAB and AAB AAB. The paper's figure 4(c) uses the 12 blocks 03-05, 11-16 and 22-24, one ABA run per set (the authors' `example_figure4_simultaneous_speech_and_cursor.py`).

### Online cursor decoder

The cursor decoder was linear with temporal smoothing, `v_t = α v_(t-1) + (1 - α) β D f_t`, where `f_t` is the 512-feature neural vector and `D` the decoding matrix, plus a nonlinear speed adjustment set by `r` and `p` on day 468. Table 1 of the paper gives the parameters per released session:

| Day | Task | α | β | r | p |
|---|---|---|---|---|---|
| 39 | Radial8 | 0.9 | | | |
| 202 | Radial8 | 0.9 | 0.2 | | |
| 202 | Simultaneous speech and cursor | 0.9 | 0.15 | | |
| 468 | Radial8, Grid | 0.92 | 0.13 | 0.05 | 1.5 |

`β` and `r` are in screen heights per second.

## Data Modalities

| Modality | Source field | Shape / rate | Notes |
|---|---|---|---|
| Threshold crossings | `threshold_crossings` | `(n_bins, 256)` @ 100 Hz | Counts below an electrode-specific -4.5 x RMS threshold, summed per 10 ms bin |
| Spike-band power | `spike_band_power` | `(n_bins, 256)` @ 100 Hz | Mean squared 250-5000 Hz signal per 10 ms bin |
| Cursor position | `cursor_position` | `(n_bins, 2)` @ 100 Hz | Screen-centered x and y, in screen-height units |
| Target position | `target_position` | `(n_bins, 2)` @ 100 Hz | Cued target center, constant within a trial |
| Online cursor output | `cursor_decoder_output` | `(n_bins, 2)` @ 100 Hz | Online decoder velocity, screen-height units per second |
| Velocity assistance | `assist_amount` | `(n_bins,)` @ 100 Hz | 1.0 is full computer control, 0.0 is closed-loop neural control |
| Click output | `click_decoder_output` | `(n_bins,)`, day 468 only | Binary output of the online click decoder |
| Click assistance | `click_assist` | `(n_bins,)`, day 468 only | Whether target selection was computer-assisted |
| Trial timing | `trial_start_bin` plus task-specific fields | per trial | Trial start, and (speech task only) target-presentation, cursor-go, speech-go, trial-end indices |
| Task geometry | `cursor_radius`, `target_radius`, `dwell_requirement_sec`, `grid_num_rows`, `grid_total_height` | per block | Selection criteria and target layout, surfaced as timeline attributes |

Only the Simultaneous Speech and Cursor Task has `target_presentation_bin`, `cursor_go_cue_bin`, `speech_go_cue_bin`, and `trial_end_bin`. In the other two tasks a trial's target presentation and go cue coincide with its start, and trials run back to back, so trial windows are bounded by the next trial's start.

- **Bin size / rate:** all streams share one 10 ms grid (100 Hz). `timestamp_sec` starts at 0.0 and steps by exactly 0.010 s in every block, so bin index `i` is time `i / 100` relative to block start.
- **Raw 30 kHz voltage is not released**, only these processed, aligned features.
- **Online decoder outputs are provenance, not measurement.** `cursor_decoder_output` and `click_decoder_output` are what the deployed system emitted, not independent measurements of intent. See [Notes / Scope](#notes--scope).

## Source Data Format & Layout

Dryad version 5 contains **51 cursor-task MATLAB files** (about 1.47 GB) plus one comparison file and the source README. The cursor blocks hold **2,521 trials and 2.52 hours** of aligned data (1,288 radial8, 554 speech, 679 grid). Each cursor file is one block of 179-181 s, except day 468 block 01 (70 s), loadable with `scipy.io.loadmat`.

Filenames are `t15_day<DDDDD>_block<BB>_<task>_task.mat`, which is what `utils.block_ref()` parses into the day, block, and task family.

```text
<NEURALSHED_DATA_ROOT>/singer_2025/
  references/
    Singer-Clark_2025_J._Neural_Eng._22_036015.pdf
    BCI-cursor-control-from-speech-motor-cortex/     # example-code repo
  data/
    Singer2025CursorControl/                         # = the Study class name
      README.md                                      # the source's own field documentation
      t15_day00039_block00_radial8_calibration_task.mat
      t15_day00202_block02_simultaneous_speech_and_cursor_task.mat
      t15_day00468_block03_grid_evaluation_task.mat
      ...
      t15_day00202_words.mat                         # comparison file, outside the cursor Study
```

Blocks per session: day 39 has 6 (all radial8), day 202 has 27 (6 radial8, 21 speech), day 468 has 18 (9 radial8, 9 grid).

The paper describes six sessions (days 39, 171, 202, 233, 339, 468). Dryad currently releases cursor-task files only for days 39, 202, and 468. The comparison file `t15_day00202_words.mat` is downloaded for reference but deliberately excluded from `list_blocks()` and the Study. It holds 54 trials, six each of seven words and of the cues `[DO NOTHING]` and `[RIGHT HAND - CLOSE]`, with its own field names (`prompt`, `presentation_bin`, `go_bin`, `end_bin`). Its timestamps step by 9, 10 or 11 ms (933, 25,429 and 944 intervals), unlike the exact 10 ms of the cursor blocks.

## Study, timelines and Events

`study.py` defines **`Singer2025CursorControl`**: **one timeline per block** (51), the smallest independently stored source recording. Each timeline carries `subject="T15"`, `session=<t15_dayDDDDD>`, `block=<BB>` (the source block number, an int), `task=<source task family>`, and the filterable attributes below, broadcast to every event row.

Per timeline it emits:

| Event type | Count | Source → shape / rate | Key attributes |
|---|---|---|---|
| `SpikeCounts` | 1 | `threshold_crossings` → `(256, n_bins)` @ 100 Hz | `method="threshold_crossing"`, `threshold="-4.5 x RMS"`, `unit="counts/bin"`, `normalization="raw"`, `processing` (paper section 2.2) |
| `BandPower` | 1 | `spike_band_power` → `(256, n_bins)` @ 100 Hz | `lower_freq=250`, `upper_freq=5000`, `unit="uV^2"`, `normalization="raw"`, `processing` (paper section 2.2) |
| `CursorPositionSeries` | 1 | `cursor_position` → `(2, n_bins)` | `dims=(x,y)`, `unit="screen_height"`, `reference_frame="screen_center"` |
| `TargetPositionSeries` | 1 | `target_position` → `(2, n_bins)` | `dims=(x,y)`, `kind="target_position"` |
| `DecodedCursorVelocitySeries` | 1 | `cursor_decoder_output` → `(2, n_bins)` | `kind="online_decoder_output"`, `unit="screen_height/s"` |
| `CursorAssistSeries` | 1 | `assist_amount` → `(1, n_bins)` | `kind="computer_assist"`, `unit="proportion"` |
| `ClickAssistSeries` | 1, day 468 only | `click_assist` → `(1, n_bins)` | `kind="computer_assist"`, `unit="binary"` |
| `ClickStateSeries` | 1, day 468 only | `click_decoder_output` → `(1, n_bins)` | `kind="online_decoder_output"`, `unit="binary"` |
| `Stimulus` (target cue) | per trial | trial timing + `target_position` | `stimulus_kind="target_cue"`, `modality="visual"`, `code` (see below), `target_x`, `target_y`, `trial_id` |
| `Action` (cursor) | per trial | trial timing + `assist_amount` | `action_kind="cursor_control"`, `task_condition="imagined"`, same `code` as its cue, the trial's `loop_type` |
| `Stimulus` (beep) | per beeped trial, speech only | `speech_go_cue_bin` | `stimulus_kind="speech_go_cue"`, `modality="audio"`, `code=1`, one bin long |
| `Action` (speech) | per beeped trial, **verbal** blocks only | `speech_go_cue_bin` → trial end | `action_kind="speech"`, `task_condition="attempted"`, `text` (the prompted word) |
| `Word` | per trial, speech only | `speech_prompt` | `text`, `language="en"`, `modality="read"`, spans the delay period |
| `Intertrial` | per trial, speech only | `trial_start_bin` → `target_presentation_bin` | The paper's 1.0 s intertrial period |

The eight whole-block streams load lazily and dims-first via `SpecialLoader` (`_load_series` transposes the source's time-first arrays to `(n_dims, n_bins)` float32) and are read by `NeuralFeatureExtractor` / `BehavioralSeriesExtractor`. The trial rows are pure time-span metadata computed from the per-trial bin indices. `trial_id` numbers the trials of a block from 0, so it is unique only together with `session` and `block`.

**Trial periods.** Trial rows carry the standard `period`. In speech blocks the `Word` is `delay`, the cursor `Action`, the beep and the speech `Action` are `go`, and `Intertrial` is `intertrial`. In radial8 and grid the target appears as the go cue, so the target cue and the cursor `Action` are `go`. The speech target cue has no `period`, because the target stays on screen from the delay through go, red and then green.

**Trial ends.** Speech trials end at the source `trial_end_bin`. That bin is inclusive and event stops are half-open, so a speech trial's rows stop one bin after it. Radial8 and grid carry no `trial_end_bin`, so each trial runs to the next trial's start. The last trial of each of these 30 blocks is cut by the end of the recording (0.09-6.64 s long, with no click, no full dwell and no timeout), and its target-cue `Stimulus` and cursor `Action` carry `valid=False`, `valid_description="trial truncated by block end"`. Each speech block also ends with a trial that starts after the last `trial_end_bin`, with a new `trial_idx` and target but no timing fields. These 21 tails (71.25 s in total) are in the whole-block streams and have no trial rows.

**Trial outcomes.** The release has no outcome field, and the Study emits no `success`. Per the paper, radial8 and speech trials ended on a selection (a dwell on days 39 and 202, a click on day 468) or at a 10 second timeout, and a grid trial ended on any click and was correct when the cursor was inside the cued cell. The authors' `example_figure2_cursor_BCI_grid_evaluation.py` scores grid trials from the cursor position at the bin before the next trial's start.

**`Intertrial` is emitted only for the speech task**, the one task whose trials separate their start from their target presentation. In the other two tasks trials run back to back, so there is no intertrial period to label (the small gap before a block's first trial is a recording lead-in, not an intertrial, and is deliberately left unlabelled).

**Code scheme.** The `code` on a target-cue `Stimulus` and its paired cursor `Action` is the target identity:

| Task | `code` | Meaning |
|---|---|---|
| Radial8, Speech | 0-7 | Outer target, by 45 degree spoke (`0` = +x, counter-clockwise) |
| Radial8, Speech | 8 | Center target |
| Grid | 0-195 | Row-major grid cell index (`row * 14 + col` on the released 14 by 14 grids, rows counted from the bottom, so `0` is the bottom-left cell) |

A beep `Stimulus` also carries a `code` (always `1`), so **filter on `stimulus_kind`, not on `code` alone**, when selecting target cues.

**Filterable timeline attributes:** `post_implant_day`, `date` (ISO, the T15 implant date plus `post_implant_day`), `cursor_imagery`, `click_enabled`, `speech_condition` (`none` / `control` / `verbal`), `loop_type` (`closed` when no trial's movement period has assistance, `assisted` in the 15 radial8 calibration blocks, whose assistance steps down from full to none), and the per-block geometry (`cursor_radius`, `target_radius`, `dwell_requirement_sec`, `grid_num_rows`, `grid_total_height`, each `None` where its task does not define it). Each cursor `Action` carries the `loop_type` of its own movement period, `open` under full assistance throughout, `closed` with none and `assisted` otherwise, and the timeline value fills the other rows. In the calibration blocks that gives 130 open, 102 assisted and 529 closed trials, so `type == 'Action' and loop_type == 'closed'` selects closed-loop trials in every block. The source publishes no train/validation/test split, so the Study emits no `original_split`. Reusable types come from the package (`neuralshed.events` and `neuralshed.extractors`), so no singer-specific Event or Extractor is needed.

## Notes / Scope

- **Download** via the Study's `_download()` (shared `datasets/dryad.py` `Dryad`), which needs a Dryad API token (`NEURALFETCH_DRYAD_TOKEN`). Source path is `singer_2025/data/`, and the Study base appends the class-name folder `Singer2025CursorControl/`.
- **`cursor_decoder_output` is a poor decoding target.** It is the online decoder's output, and that decoder was a linear map of the very `threshold_crossings` and `spike_band_power` features a model would take as input, so fitting it measures recovery of the experimenters' own decoding matrix, not neural decoding skill. Prefer paper-style **intended velocity**: the unit vector from cursor to target, excluding bins where the cursor is already on the target (paper section 2.4.3). Day 39 used position-error estimation instead (intended speed scaling with distance to target), so a model spanning days should switch rule per session. `03_model_example.ipynb` decodes the intended direction with a causal decoder and scores the online decoder output against the same target as a baseline.
- **Assist state matters.** Calibration blocks mix open-loop, transition, and closed-loop control, so select trials on the `Action` rows' `loop_type`, or use `CursorAssistSeries` for windows that are not trials. The day-468 grid blocks are fully closed-loop (`assist_amount` is 0.0 throughout).
- **Partial release.** The paper reports six sessions (days 39, 171, 202, 233, 339, 468), Dryad releases only days 39, 202, and 468.
- **Binned features only.** Raw 30 kHz voltage is not released, so timing finer than 10 ms and features with another threshold or band cannot be derived. Coarser bins are possible, and the extractors re-bin on request.
- **Electrode 8 on day 202.** Its threshold crossings are zero throughout blocks 00, 02 and 06, while its spike-band power is present. The Study does not mask it.
- **`cursor_imagery` is the instruction, not the strategy.** The first-ever-use session varied instructed imagery across blocks and the release attaches no per-bin imagery label, so the Study keeps only a coarse day-level description. The paper also cautions (section 4.2) that T15 often described his method as intuition even on blocks where a specific body part was instructed. Day 39 had six instructed imageries (right and left hand mouse, right and left hand paintbrush, tongue sticking out, tongue inside mouth) over six blocks, and no source gives which block used which.

## Files

- `metadata.py`: catalog `DATASET_METADATA` + `sessions()` + `get_metadata()`.
- `utils.py`: `BlockRef` + source readers (block discovery, MATLAB loading, trial windows, per-block geometry, grid cell index) and constants.
- `study.py`: the neuralset `Study` (`Singer2025CursorControl`), one timeline per block.
- `01_explore_source_data.ipynb`: session table, per-task field schemas, channel layout, directional tuning, click and assistance, speech after the beep, and the truncated last trials.
- `02_explore_study.ipynb`: build the Study, read every event family through its extractor.
- `03_model_example.ipynb`: decode the intended cursor direction with a causal decoder, with the online decoder output as a baseline.
