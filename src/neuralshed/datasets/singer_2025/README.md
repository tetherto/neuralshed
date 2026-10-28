# Singer-Clark et al. 2025: Speech motor cortex enables BCI cursor control and click

Intracortical 2-D cursor control and click in one BrainGate2 participant (**T15**, a 45-year-old left-handed man with ALS, tetraplegia, and severe dysarthria). Four 64-channel Utah arrays (256 electrodes) in the left ventral precentral gyrus record 10 ms-binned threshold crossings and spike-band power while he uses motor imagery to drive a cursor to targets and select them. The finding is that **speech** motor cortex carries enough movement-intention signal to run a cursor BCI, so the decode target here is movement, not speech.

Same BrainGate2 participant **T15** and the same four Utah arrays as [`card_2024`](../card_2024/README.md), [`wairagkar_2025`](../wairagkar_2025/README.md), and [`kunz_2025`](../kunz_2025/README.md), which are speech deposits from the same lab. The per-array HOMBA and UBERON mappings in `metadata.py` are shared with those datasets.

## References

- **Paper:** Singer-Clark, T., Hou, X., Card, N. S., Wairagkar, M., Iacobacci, C., Peracha, H., Hochberg, L. R., Stavisky, S. D., Brandman, D. M. *Speech motor cortex enables BCI cursor control and click.* **J. Neural Eng.** 22(3), 036015 (2025). https://doi.org/10.1088/1741-2552/add0e5
- **Source data:** [10.5061/dryad.prr4xgxzq](https://datadryad.org/dataset/doi:10.5061/dryad.prr4xgxzq)
- **Code repository:** https://github.com/Neuroprosthetics-Lab/BCI-cursor-control-from-speech-motor-cortex

## Species & Subjects

- **Species:** *Homo sapiens* (human).
- **Subject:** BrainGate2 participant **T15**, male, 45 years old at these recordings, left-handed.
- **Clinical condition:** ALS, tetraplegia, severe dysarthria. He retained eye movement, impaired neck movement, and limited orofacial muscle control.
- **Implant:** four 64-electrode Blackrock NeuroPort Utah arrays (256 electrodes) in the **left ventral precentral gyrus**, implanted 17 July 2023 (from the shared participant registry; the paper reports only post-implant days 39-468). Source labels `v6v`, `4`, `55b`, `d6v`, mapping to electrode columns 0-63, 64-127, 128-191, 192-255 respectively (verified against `array_label_by_electrode`, which is identical in all 51 blocks). The paper localizes them as ventral premotor cortex (`v6v`, `d6v`, area 6v), primary motor cortex (`4`, area 4), and middle precentral gyrus (`55b`).
- **Released sessions:** post-implant days 39, 202, and 468.
- **Lab / institution:** Neuroprosthetics Lab, UC Davis (recordings performed at the participant's home).

## Behavioral Task

Three task families, all cursor control by motor imagery on a screen whose coordinates are centered on the screen with a distance of 1.0 equal to the screen height.

### Radial8 Calibration Task

Eight outer circular targets and one center target were presented in alternating center-out-and-back order. A target turning green was both the target presentation and the cursor go cue. T15 used instructed motor imagery to move toward it. Selection was by dwell on days 39 and 202 (`dwell_requirement_sec` is 0.8 s and 1.0 s respectively for these radial8 blocks; the simultaneous speech blocks use 1.5 s) and by attempted click on day 468, where the radial8 blocks carry the click streams too. Calibration blocks could transition from full computer assistance to closed-loop control as the online decoder improved.

### Grid Evaluation Task

The cursor selected square targets in a densely packed grid using closed-loop velocity and click decoders. The paper describes 6 by 6 and 14 by 14 grids, but every one of the 9 released blocks is 14 by 14 (`grid_num_rows` is 14, `grid_total_height` is 0.92). Per the paper, trials ended on a correct or incorrect selection or at a 10 second timeout, but the release carries no `trial_end_bin` for this task, so the Study cannot label a trial's outcome and simply runs each trial window to the next trial's start. The Dryad release contains day 468 evaluation blocks.

### Simultaneous Speech and Cursor Task

A red target and a word appeared together. After a randomized 1.3 to 1.7 second delay the target turned green, beginning cursor movement. Some trials later played a beep instructing T15 to attempt the prompted word while continuing cursor control. In verbal blocks he spoke after the beep, in control blocks he did not. Trials were separated by a one-second intertrial period. The Dryad release contains day 202 blocks (14 verbal, 7 control).

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

Dryad version 5 contains **51 cursor-task MATLAB files** (about 1.47 GB) plus one speech-only comparison file and the source README. The cursor blocks hold **2,521 trials and 2.52 hours** of aligned data (1,288 radial8, 554 speech, 679 grid). Each cursor file is one 3 to 5 minute block, loadable with `scipy.io.loadmat`.

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
      t15_day00202_words.mat                         # speech-only, outside the cursor Study
```

Blocks per session: day 39 has 6 (all radial8), day 202 has 27 (6 radial8, 21 speech), day 468 has 18 (9 radial8, 9 grid).

The paper describes six sessions (days 39, 171, 202, 233, 339, 468). Dryad currently releases cursor-task files only for days 39, 202, and 468. The speech-only `t15_day00202_words.mat` is downloaded for reference but deliberately excluded from `list_blocks()` and the Study.

## Study, timelines and Events

`study.py` defines **`Singer2025CursorControl`**: **one timeline per block** (51), the smallest independently stored source recording. Each timeline carries `subject="T15"`, `session=<t15_dayDDDDD>`, `run=<blockBB>`, `task=<source task family>`, and the filterable attributes below, broadcast to every event row.

Per timeline it emits:

| Event type | Count | Source → shape / rate | Key attributes |
|---|---|---|---|
| `SpikeCounts` | 1 | `threshold_crossings` → `(256, n_bins)` @ 100 Hz | `method="threshold_crossing"`, `threshold="-4.5 x RMS"`, `unit="counts/bin"` |
| `BandPower` | 1 | `spike_band_power` → `(256, n_bins)` @ 100 Hz | `lower_freq=250`, `upper_freq=5000`, `unit="uV^2"` |
| `CursorPositionSeries` | 1 | `cursor_position` → `(2, n_bins)` | `dims=(x,y)`, `unit="screen_height"`, `reference_frame="screen_center"` |
| `TargetPositionSeries` | 1 | `target_position` → `(2, n_bins)` | `dims=(x,y)`, `kind="target_position"` |
| `DecodedCursorVelocitySeries` | 1 | `cursor_decoder_output` → `(2, n_bins)` | `kind="online_decoder_output"`, `unit="screen_height/s"` |
| `CursorAssistSeries` | 1 | `assist_amount` → `(1, n_bins)` | `kind="computer_assist"`, `unit="proportion"` |
| `ClickAssistSeries` | 1, day 468 only | `click_assist` → `(1, n_bins)` | `kind="computer_assist"`, `unit="binary"` |
| `ClickStateSeries` | 1, day 468 only | `click_decoder_output` → `(1, n_bins)` | `kind="online_decoder_output"`, `unit="binary"` |
| `Stimulus` (target cue) | per trial | trial timing + `target_position` | `stimulus_kind="target_cue"`, `modality="visual"`, `code` (see below), `target_x`, `target_y`, `trial_id` |
| `Action` (cursor) | per trial | trial timing | `action_kind="cursor_control"`, `task_condition="imagined"`, same `code` as its cue |
| `Stimulus` (beep) | per beeped trial, speech only | `speech_go_cue_bin` | `stimulus_kind="speech_go_cue"`, `modality="audio"`, `code=1`, one bin long |
| `Action` (speech) | per beeped trial, **verbal** blocks only | `speech_go_cue_bin` → trial end | `action_kind="speech"`, `task_condition="attempted"` |
| `Word` | per trial, speech only | `speech_prompt` | `text`, `language="en"`, `modality="read"`, spans the delay period |
| `Intertrial` | per trial, speech only | `trial_start_bin` → `target_presentation_bin` | The paper's 1.0 s intertrial period |

The eight whole-block streams load lazily and dims-first via `SpecialLoader` (`_load_series` transposes the source's time-first arrays to `(n_dims, n_bins)` float32) and are read by `NeuralFeatureExtractor` / `BehavioralSeriesExtractor`. The trial rows are pure time-span metadata computed from the per-trial bin indices.

**`Intertrial` is emitted only for the speech task**, the one task whose trials separate their start from their target presentation. In the other two tasks trials run back to back, so there is no intertrial period to label (the small gap before a block's first trial is a recording lead-in, not an intertrial, and is deliberately left unlabelled).

**Code scheme.** The `code` on a target-cue `Stimulus` and its paired cursor `Action` is the target identity:

| Task | `code` | Meaning |
|---|---|---|
| Radial8, Speech | 0-7 | Outer target, by 45 degree spoke (`0` = +x, counter-clockwise) |
| Radial8, Speech | 8 | Center target |
| Grid | 0-195 | Row-major grid cell index (`row * 14 + col` on the released 14 by 14 grids) |

A beep `Stimulus` also carries a `code` (always `1`), so **filter on `stimulus_kind`, not on `code` alone**, when selecting target cues.

**Filterable timeline attributes:** `original_split` (`calibration` / `evaluation` / `interference`), `post_implant_day`, `cursor_imagery`, `click_enabled`, `speech_condition` (`none` / `control` / `verbal`), and the per-block geometry (`cursor_radius`, `target_radius`, `dwell_requirement_sec`, `grid_num_rows`, `grid_total_height`, each `None` where its task does not define it). Reusable types come from the package (`neuralshed.events` and `neuralshed.extractors`), so no singer-specific Event or Extractor is needed.

## Notes / Scope

- **Download** via the Study's `_download()` (shared `datasets/dryad.py` `Dryad`), which needs a Dryad API token (`NEURALFETCH_DRYAD_TOKEN`). Source path is `singer_2025/data/`, and the Study base appends the class-name folder `Singer2025CursorControl/`.
- **`cursor_decoder_output` is a poor decoding target.** It is the online decoder's output, and that decoder was a linear map of the very `threshold_crossings` and `spike_band_power` features a model would take as input, so fitting it measures recovery of the experimenters' own decoding matrix rather than neural decoding skill. Prefer paper-style **intended velocity**: the unit vector from cursor to target, excluding bins where the cursor is already on the target (paper section 2.4.3). Day 39 used position-error estimation instead (intended speed scaling with distance to target), so a model spanning days should switch rule per session. `03_model_example.ipynb` decodes intended velocity and keeps the decoder-output fit only as a labelled sanity check.
- **Assist state matters.** Calibration blocks mix open-loop, transition, and closed-loop control, so use `CursorAssistSeries` when selecting training windows. The day-468 grid blocks are fully closed-loop (`assist_amount` is 0.0 throughout).
- **Partial release.** The paper reports six sessions (days 39, 171, 202, 233, 339, 468), Dryad releases only days 39, 202, and 468. Grid trial outcomes (correct, incorrect, timeout) are not recoverable from the release.
- **Binned features only.** Raw 30 kHz voltage is not released, so re-binning at another width or re-deriving band power from the raw signal is not possible.
- **`cursor_imagery` is the instruction, not the strategy.** The first-ever-use session varied instructed imagery across blocks and the release attaches no per-bin imagery label, so the Study keeps only a coarse day-level description. The paper also cautions (section 4.2) that T15 often described his method as intuition even on blocks where a specific body part was instructed.

## Files

- `metadata.py`: catalog `DATASET_METADATA` + `sessions()` + `get_metadata()`.
- `utils.py`: `BlockRef` + source readers (block discovery, MATLAB loading, trial windows, per-block geometry, grid cell index) and constants.
- `study.py`: the neuralset `Study` (`Singer2025CursorControl`), one timeline per block.
- `01_explore_source_data.ipynb`: source structure, per-task field schemas, and alignment checks.
- `02_explore_study.ipynb`: build the Study, read every event family through its extractor.
- `03_model_example.ipynb`: decode intended cursor velocity from neural features, scored against the deployed online decoder.
