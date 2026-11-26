# Deo et al. 2024: Brain control of bimanual movement enabled by recurrent neural networks

Intracortical control of **two** on-screen cursors (bimanual and unimanual) by one BrainGate2 participant (**T5**, C4 spinal cord injury, tetraplegia). Two 96-channel Utah arrays (192 electrodes) in the hand-knob of left motor cortex record 20 ms-binned threshold crossings while T5 attempts delayed reaching movements. Each trial opens by revealing one of three cued movement types (unimanual right, unimanual left, or bimanual), holds an instructed delay (0.96-2.96 s in the release), then releases a GO cue. The dataset spans open-loop instructed-delay blocks (cursors move autonomously to targets) and real-time closed-loop blocks where RNN or linear decoders drive the two cursors from neural activity. The decode target is movement (cursor kinematics).

Same participant (**T5**) and arrays as the other T5 datasets listed in the [T5 channel correspondence report](../../../../docs/channel_mapping/t5.md). The T5 subject block and hand-knob mapping in `metadata.py` are shared with [`willet_2021`](../willet_2021/README.md) and [`fan_2023`](../fan_2023/README.md).

## References

- **Paper:** Deo, D. R., Willett, F. R., Avansino, D. T., Hochberg, L. R., Henderson, J. M., Shenoy, K. V. *Brain control of bimanual movement enabled by recurrent neural networks.* **Scientific Reports** 14, 1598 (2024). https://doi.org/10.1038/s41598-024-51617-3
- **Supplementary Information:** https://static-content.springer.com/esm/art%3A10.1038%2Fs41598-024-51617-3/MediaObjects/41598_2024_51617_MOESM1_ESM.pdf
- **Source data:** [10.5061/dryad.sn02v6xbb](https://datadryad.org/dataset/doi:10.5061/dryad.sn02v6xbb)
- **Code repository:** https://github.com/d-r-deo/bimanualBCI

## Species & Subjects

- **Species:** *Homo sapiens* (human).
- **Subject:** BrainGate2 participant **T5**, male, 69 years old at these recordings, right-handed.
- **Clinical condition:** cervical spinal cord injury classified **C4 AIS-C**, tetraplegia. The injury occurred about 9 years before trial enrollment. T5 retains full head and face movement and can shrug his shoulders, with very limited voluntary motion of the arms and legs, so all task movement is **attempted** (small-amplitude movements were intermittently observed).
- **Implant:** two 96-electrode Blackrock NeuroPort Utah arrays (192 electrodes, 1.5 mm electrode length) in the **hand-knob area of the left precentral gyrus**. The source does not label which 96 channels belong to which array, so the 0-95 / 96-191 split **and** the `lateral_array` / `medial_array` names both follow the `willet_2021` convention for the same participant. The Study preserves source channel order, and these array labels and the cross-dataset channel correspondence remain inferred candidates, as detailed in the [T5 channel correspondence report](../../../../docs/channel_mapping/t5.md). No stereotaxic coordinates are published.
- **Implant date:** the paper gives only "**August 2016**" (month precision). The exact day, **17 August 2016**, comes from the shared participant registry (`datasets/participants.py`, derived from `willet_2021`).
- **Released sessions:** 13 experiment days from 2 June to 13 October 2021, **trial days 1750-1883** in the paper's post-implant numbering.

## Behavioral Task

A monitor showed two large circular targets (one purple, one white) and two smaller cursors of matching colors, labelled **L** (left, purple) and **R** (right, white). T5's attempted-movement strategy was to imagine gripping and pushing joysticks, one per cursor. The released day notes describe unimanual and bimanual wrist movements. On each trial one of three movement types was cued at random: **unimanual right** (UniR, right cursor only), **unimanual left** (UniL, left cursor only), or **bimanual** (Bi, both cursors simultaneously).

The trial structure is **preparation, movement, and the interval between trials**:

1. **Prepare (0.96-2.96 s in the release).** The trial opens with the targets in place and lines connecting each cursor to its intended target, which is what reveals the movement type. The paper describes a random 1-2 s preparation delay. The released `durationBinnedDelay` values span 48-148 bins (0.96-2.96 s).
2. **Move (released open-loop window: 0.48-6.22 s).** A beep marks the GO cue and the connecting lines disappear. The paper describes autonomous cursor travel lasting 1-2 s, with simultaneous target arrival on bimanual trials. The released open-loop `durationBinnedMove` values span 24-311 bins (0.48-6.22 s). These are movement windows, which can include a delay before cursor motion and a hold at the target, rather than measurements of cursor travel alone.
3. **Between trials.** A second beep ends the trial and instructs T5 to stop attempting movement. The released intervals from movement end to the next preparation start are 560 or 580 ms on June 2, and 60 or 80 ms on the other days. They describe the gaps between the released trial windows, without implying that the cursor or participant is at rest.

Three control conditions:

- **Open-loop (OL)** instructed-delay blocks: the cursors move autonomously to their targets along a trajectory. T5 attempts the movement but does not drive the cursors.
- **Closed-loop (CL)** blocks: the cursors are driven in real time by a neural decoder (RNN or linear). T5 had a maximum of **10 s** to acquire both targets, where acquisition means both cursors dwelling simultaneously inside their targets for an uninterrupted **500 ms**. A trial that does not acquire within 10 s is **failed**. See [Notes / Scope](#notes--scope) for how trial outcome is inferred, since the release has no outcome field.
- **Closed-loop recalibration (CLR)**, an "assisted" variant used to recalibrate a decoder before true closed-loop evaluation.

Early days are open-loop only. Later days add closed-loop control, sometimes comparing decoders (RNN vs linear), gains, augmentation, or movement strategies (sequential-unimanual vs true-bimanual). Supplementary Table 2 of the paper lists each day's session number, trial day and the figures it supports.

## Data Modalities

Each experiment day is one `.mat` holding a concatenated stream of its released blocks, with **recording gaps between blocks** according to the source documentation. `blockNum` tags each bin.

| Modality | Source field | Shape / rate | Notes |
|---|---|---|---|
| Threshold crossings | `tx` | `(192, T)` @ 50 Hz | Counts below a -3.5 x RMS threshold per 20 ms bin, `uint8`, channels-first |
| Cursor position | `Cp` | `(4, T)` @ 50 Hz | `[right_x, right_y, left_x, left_y]`, screen coordinates |
| Target position | `Tp` | `(4, T)` @ 50 Hz | Cued target center for each cursor, same dim order |
| Block index | `blockNum` | `(T,)` | Active block per bin. Value changes mark (unrecorded) recording gaps |
| Trial timing | `goCue`, `durationBinnedDelay`, `durationBinnedMove` | per trial (`R`) | GO index (**MATLAB 1-based**), delay / move lengths in bins |
| Movement type | `moveType` | per trial (`R`) | 1=UniR, 2=UniL, 3=Bi (`moveTypeName` holds the strings) |
| Day notes | `Notes` | per day | Free-text description |

- **Storage types:** `tx` has declared MATLAB class `double`, reported by `scipy.io.whosmat`, but the saved count array is `uint8`, as returned by the default `scipy.io.loadmat`. The catalog keeps `dtype="uint8"` for that saved representation. The Study emits `float32` streams with the same count values (0 to 15 across the release). `durationBinnedMove` is saved as `uint8` on June 2 and June 4, and `uint16` on the other days.
- **Position units:** The paper, release documentation and published analysis code do not explicitly define the coordinate unit. Cursor coordinates reach exactly -960 to 960 horizontally and -540 to 540 vertically, consistent with a centred 1920 by 1080 screen. These bounds suggest screen-centred pixel coordinates, but that interpretation is inferred. The Study retains `unit="a.u."` (arbitrary units) for both cursor and target positions and preserves the released values without rescaling. Closed-loop cursors stop at these edges.
- **Bin size / rate:** all streams share one 20 ms grid (50 Hz).
- **Only threshold crossings are released** for the neural feature, and no spike-band power.
- **Neural signal chain (from the paper).** Analog band 0.3-7.5 kHz, digitized at 30 kHz with 250 nV resolution, common-average referenced across each array, then a 250 Hz digital high-pass per electrode before spike detection at -3.5 x RMS. The released counts are raw. The paper's decoders used counts with each block's mean removed, then z-scored.
- **Control loop and experimental conditions are per block and live in `readme.txt`**, not in the `.mat`. The source control labels remain in `utils.SESSION_BLOCKS`, with decoder settings and task instructions in the adjacent condition tables. The Study maps the control labels to `loop_type` values `open`, `assisted` and `closed`.
- **In closed-loop blocks `Cp` is decoder output** (neural-driven), not the autonomous instructed trajectory of open-loop blocks. The cursor stream means different things across loop types.
- **`durationBinnedMove` also means different things across loop types.** In open loop it is the released movement-window duration (0.48-6.22 s), distinct from the paper's nominal 1-2 s cursor travel. In closed loop it is the time to target acquisition, which is the paper's primary performance metric. The nominal 10 s timeout is stored as 499 bins (9.98 s).

## Source Data Format & Layout

The Dryad deposit contains **13 MATLAB files** (about 139 MB), one per experiment day, plus a source `readme.txt` documenting the fields and per-block conditions. Each file is loadable with `scipy.io.loadmat`. Filenames are `t5_MM_DD_YYYY.mat`, parsed by `utils.session_date()`.

```text
<NEURALSHED_DATA_ROOT>/deo_2024/
  references/
    Deo_2024_Sci_Rep_14_1598.pdf                # publication (add manually)
    Deo_2024_Sci_Rep_14_1598_supplementary.pdf  # supplement (add manually)
    bimanualBCI/                                # code repo clone (add manually)
  data/
    Deo2024Bimanual/                            # neuralset Study folder
      README.md                                 # Dryad landing-page README
      readme.txt                                # source field + block documentation
      t5_06_02_2021.mat
      t5_06_04_2021.mat
      ...
      t5_10_13_2021.mat
```

Trials per day range from 276 to 1439 (11,044 total). Blocks per day range from 4 to 28, though block *numbers* run as high as 33 because the numbering is not contiguous within a day. The two pilot days hold 771 released trials, and the paper reports 780.

Each block occupies one contiguous run of stored bins, so slicing by `blockNum` recovers its samples. Blocks are stored out of numeric order on September 27, September 29, October 11 and October 13.

## Study, timelines and Events

`study.py` defines **`Deo2024Bimanual`**: **one timeline per block** (204 across the 13 days). This follows the source documentation that recording gaps occur at block changes. The release provides bin indices on a 20 ms grid, but no device timestamps, wall-clock timestamps or per-bin time vector. A pause during which both the task and recording stopped inside a block could neither be measured nor ruled out. `iter_timelines` slices each day by `blockNum`. Each timeline carries `subject="T5"`, `session=<t5_MM_DD_YYYY>`, `block`, `date` and the block's control settings (`loop_type`, `decoder_type`, `decoder_gain`, `decoder_training_data`), broadcast to every event row.

Per timeline it emits:

| Event type | Count | Source → shape / rate | Key attributes |
|---|---|---|---|
| `SpikeCounts` | 1 | `tx` block slice → `(192, n_bins)` @ 50 Hz | `method="threshold_crossing"`, `threshold="-3.5 x RMS"`, `unit="counts/bin"` |
| `CursorPositionSeries` | 2 (right, left) | `Cp` rows → `(2, n_bins)` | `dims=(x,y)`, `effector=right|left`, `unit="a.u."`, `reference_frame="screen_center"` |
| `TargetPositionSeries` | 2 (right, left) | `Tp` rows → `(2, n_bins)` | `dims=(x,y)`, `effector=right|left` |
| `Stimulus` (movement cue) | per trial | trial timing + `moveType` + `Tp` at `goCue` | `stimulus_kind="movement_cue"`, `modality="visual"`, `period="delay"`, `task_condition="na"`, `code` (moveType), `description` (UniR/UniL/Bi), `trial_id`, `target_*` coordinates |
| `Stimulus` (go cue) | per trial | `goCue`, one 20 ms bin | `stimulus_kind="movement_go_cue"`, `modality="audio"`, `period="go"`, `task_condition="na"`, same `code`, `trial_id` |
| `Action` (attempted move) | per trial | trial timing | `action_kind="cursor_control"`, `task_condition="attempted"`, `period="go"`, same `code`, `trial_id`, `movement_strategy`, `declared_trial_mix`, `success` in closed and assisted blocks |
| `Intertrial` | per pair of consecutive trials | movement end to next preparation start | `period="intertrial"`, `task_condition="na"`, preceding `trial_id` |

`trial_id` is generated by this adapter, since the release has no trial identifier. It is a zero-based counter in source trial order that restarts in each block, so `(session, block, trial_id)` uniquely identifies a trial within this dataset. The movement cue, go cue, action and following intertrial interval (when present) share that identifier.

`code` is copied from the source `moveType` field and has dataset-wide meanings: 1 = right-hand-only movement, 2 = left-hand-only movement, 3 = bimanual movement.

The visual movement cue carries `target_right_x`, `target_right_y`, `target_left_x` and `target_left_y`, sampled from the source `Tp` at `goCue`. These coordinates use the same arbitrary units and screen-centred reference frame as `TargetPositionSeries`. They describe the targets presented for that trial and remain constant from preparation through movement in every released trial. Associate the cue with its `Action` using `(session, block, trial_id)`. On unimanual trials the inactive hand's target sits on its cursor, so that hand's `target_*` values are its cursor position.

The two hands are split into separate homogeneous `CursorPositionSeries` / `TargetPositionSeries` (right = `Cp` rows 0-1, left = rows 2-3), each tagged with an `effector` attribute, so a model can select one hand. The whole-block streams load lazily and dims-first via `SpecialLoader` (`_load_series` slices the day array to the block's bin range, transposes neural to `(192, n_bins)`, and picks the two position rows), and are read by `NeuralFeatureExtractor` / `BehavioralSeriesExtractor`. The trial rows use the per-trial bin indices: the visual `Stimulus` spans preparation up to the go cue, the audio `Stimulus` marks go-cue onset, and the `Action` spans the released go-to-move-end window.

The preparation cue starts at `goCue - durationBinnedDelay`, preserving the released delay boundary. Target positions change two bins (40 ms) earlier, but that does not establish when the connecting lines appeared. The go marker occupies one 20 ms bin by convention and does not claim a measured beep duration.

Each `Intertrial` spans the interval from a released movement end to the next preparation start within the same block. The portions before the first trial and after the last trial are left unlabelled. On June 2, cursor motion can continue into an `Intertrial`, as described below. These rows preserve the released movement boundaries.

Each block runs under one control setting for its whole length. `loop_type` is `open` on 92 blocks, `assisted` (recalibration) on 20 and `closed` on 92. Decoder type is specified on 112 blocks (`linear` or `recurrent_neural_network`), gain on 27, and decoder training data on 24 (`raw` or `augmented`). These are settings of the online experiment. The training-data label does not describe preprocessing applied to the released neural arrays. Missing settings remain unspecified.

Only `Action` rows carry `movement_strategy` and `declared_trial_mix`. The strategy is specified for 18 blocks: `sequential_unimanual` means moving one cursor at a time, and `true_bimanual` means simultaneous movements on bimanual trials. The declared mix is available for every block as `unimanual_only` or `unimanual_and_bimanual`. It preserves the source protocol, while `code` preserves the actual trial type. 40 trials have bimanual codes in blocks declared unimanual-only. 39 are the first trial of an open-loop block, in which both cursors move. The 40th (September 29, block 20, `trial_id` 13) is a closed-loop timeout in which only the left cursor moves, so its code looks wrong. All 40 keep their released codes.

**Filterable timeline attributes:** `session`, `block`, `date`, `loop_type`, `decoder_type`, `decoder_gain`, `decoder_training_data`, for example `query="loop_type == 'closed'"`. Reusable types come from `neuralshed.events` and `neuralshed.extractors`.

The paper reports five-fold cross-validation on trial day 1883, but the release provides no partition labels or fold assignments, so the Study does not emit `original_split`. Deo and `wilson_2025` contain different blocks from seven shared recording days, documented in the [T5 channel correspondence report](../../../../docs/channel_mapping/t5.md). An evaluation on unseen days should assign all recordings from the same participant and date to one partition across both datasets.

## Notes / Scope

- **Download** via the Study's public `download()` (shared `datasets/dryad.py` `Dryad`), which needs a Dryad API token (`NEURALFETCH_DRYAD_TOKEN`). The Study resolves its source path to `deo_2024/data/Deo2024Bimanual/`.
- **Open-loop vs closed-loop matters for the target.** In open-loop blocks `Cp` is the autonomous instructed trajectory, in closed-loop blocks it is the online decoder's output. Select blocks by `loop_type` when choosing a decoding target, and prefer open-loop cursor kinematics for a clean intended-movement target.
- **Open-loop motion timing.** Cursor motion starts about 200 ms after `goCue`, except on June 2, where it starts one bin after. On June 2 the cursors also keep moving a median 440 ms past the released movement window. The Study keeps the released `Action` windows, and the cursor streams keep the full motion.
- **Cursor resets between trials.** In closed and assisted blocks, the cursors jump to new start positions 1-2 bins after each movement ends, inside the `Intertrial` rows. A velocity derived over whole blocks spikes there. Open-loop blocks have no such jumps.
- **Closed-loop recalibration is not clean decoder output either.** `assisted` blocks carry experimenter-set push and error assistance plus a locked non-active cursor, and the assistance level is not included in the release. Treat CLR kinematics as partly computer-driven.
- **Trial outcome is derived from trial duration.** The release has no outcome field. The authors state that a closed-loop trial not acquired within 10 s is failed, and `durationBinnedMove` stops at 499 bins. In closed and assisted blocks, `success` is False when `durationBinnedMove` is 499 and True otherwise. Open-loop trials leave it unset, since the computer moves the cursors.
- **`goCue` is a MATLAB 1-based index** into the time axis. Accessors convert it to a 0-based bin index.
- **Inactive-cursor locking beyond recalibration is not an emitted condition label.** A stationary inactive cursor is observable in additional closed-loop blocks, but the source does not consistently label that setting. `loop_type` follows the source block classification.

## Files

- `metadata.py`: catalog `DATASET_METADATA` + `sessions()` + `get_metadata()`.
- `utils.py`: source readers (session discovery, MATLAB loading, per-trial table, block bin ranges) and constants, plus the `SESSION_BLOCKS` source control labels and the decoder-setting and task-instruction maps.
- `study.py`: the neuralset `Study` (`Deo2024Bimanual`), one timeline per block.
- `01_explore_source_data.ipynb`: source structure, the 13 days, neural and cursor/target signals, trial timing, open- vs closed-loop.
- `02_explore_study.ipynb`: build the Study, read every event family through its extractor.
- `03_model_example.ipynb`: decode both cursors' velocity from neural features, with whole blocks held out for validation.
