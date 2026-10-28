# Deo et al. 2024: Brain control of bimanual movement enabled by recurrent neural networks

Intracortical control of **two** on-screen cursors (bimanual and unimanual) by one BrainGate2 participant (**T5**, C4 spinal cord injury, tetraplegia). Two 96-channel Utah arrays (192 electrodes) in the hand-knob of left motor cortex record 20 ms-binned threshold crossings while T5 attempts delayed reaching movements. Each trial opens by revealing one of three cued movement types (unimanual right, unimanual left, or bimanual), holds a 1-2 s instructed delay, then releases a GO cue. The dataset spans open-loop instructed-delay blocks (cursors move autonomously to targets) and real-time closed-loop blocks where RNN or linear decoders drive the two cursors from neural activity. The decode target is movement (cursor kinematics).

Same BrainGate2 participant **T5** and the same two hand-knob Utah arrays as [`willet_2021`](../willet_2021/README.md) (handwriting) and [`fan_2023`](../fan_2023/README.md) (FALCON H2 handwriting). The T5 subject block and the hand-knob HOMBA/UBERON mapping in `metadata.py` are shared with those datasets.

## References

- **Paper:** Deo, D. R., Willett, F. R., Avansino, D. T., Hochberg, L. R., Henderson, J. M., Shenoy, K. V. *Brain control of bimanual movement enabled by recurrent neural networks.* **Scientific Reports** 14, 1598 (2024). https://doi.org/10.1038/s41598-024-51617-3
- **Source data:** [10.5061/dryad.sn02v6xbb](https://datadryad.org/dataset/doi:10.5061/dryad.sn02v6xbb)
- **Code repository:** https://github.com/d-r-deo/bimanualBCI

## Species & Subjects

- **Species:** *Homo sapiens* (human).
- **Subject:** BrainGate2 participant **T5**, male, 69 years old at these recordings, right-handed.
- **Clinical condition:** cervical spinal cord injury classified **C4 AIS-C**, tetraplegia. The injury occurred about 9 years before trial enrollment. T5 retains full head and face movement and can shrug his shoulders, with very limited voluntary motion of the arms and legs, so all task movement is **attempted** (small-amplitude movements were intermittently observed).
- **Implant:** two 96-electrode Blackrock NeuroPort Utah arrays (192 electrodes, 1.5 mm electrode length) in the **hand-knob area of the left precentral gyrus**. The source does not label which 96 channels belong to which array, so the 0-95 / 96-191 split **and** the `lateral_array` / `medial_array` names both follow the `willet_2021` convention for the same participant. No stereotaxic coordinates are published.
- **Implant date:** the paper gives only "**August 2016**" (month precision). The exact day, **17 August 2016**, comes from the shared participant registry (`datasets/participants.py`, derived from `willet_2021`).
- **Released sessions:** 13 experiment days from 2 June to 13 October 2021, **trial days 1750-1883** in the paper's post-implant numbering.

## Behavioral Task

A monitor showed two large circular targets (one purple, one white) and two smaller cursors of matching colors, labelled **L** (left, purple) and **R** (right, white). T5's attempted-movement strategy was to imagine gripping and pushing joysticks, one per cursor. On each trial one of three movement types was cued at random: **unimanual right** (UniR, right cursor only), **unimanual left** (UniL, left cursor only), or **bimanual** (Bi, both cursors simultaneously).

The trial structure is **prepare, move, idle**:

1. **Prepare (1-2 s).** The trial opens with the targets in place and lines connecting each cursor to its intended target, which is what reveals the movement type. T5 prepares the movement over a random 1-2 s delay.
2. **Move (1-2 s in open loop).** A beep marks the GO cue and the connecting lines disappear. Both cursors reach their targets simultaneously on bimanual trials.
3. **Idle.** A second beep ends the trial, T5 is instructed to stop all attempted movement and start preparing for the next one. This inter-trial period is whatever `durationBinnedDelay` / `durationBinnedMove` leave unaccounted for, about 560 ms (28-29 bins) in practice, and roughly 15% of a block's bins.

Two control regimes:

- **Open-loop (OL)** instructed-delay blocks: the cursors move autonomously to their targets along a trajectory. T5 attempts the movement but does not drive the cursors.
- **Closed-loop (CL)** blocks: the cursors are driven in real time by a neural decoder (RNN or linear). T5 had a maximum of **10 s** to acquire both targets, where acquisition means both cursors dwelling simultaneously inside their targets for an uninterrupted **500 ms**. A trial that does not acquire within 10 s is **failed**. See [Notes / Scope](#notes--scope) for how to recover trial outcome, which the release does not label directly.
- **Closed-loop recalibration (CLR)**, an "assisted" variant used to recalibrate a decoder before true closed-loop evaluation.

Early days are open-loop only. Later days add closed-loop control, sometimes comparing decoders (RNN vs linear), gains, augmentation, or movement strategies (sequential-unimanual vs true-bimanual).

## Data Modalities

Each experiment day is one `.mat` holding a whole-day concatenated stream (blocks joined in order, with **recording gaps between blocks**). `blockNum` tags each bin.

| Modality | Source field | Shape / rate | Notes |
|---|---|---|---|
| Threshold crossings | `tx` | `(192, T)` @ 50 Hz | Counts below a -3.5 x RMS threshold per 20 ms bin, `uint8`, channels-first |
| Cursor position | `Cp` | `(4, T)` @ 50 Hz | `[right_x, right_y, left_x, left_y]`, screen coordinates |
| Target position | `Tp` | `(4, T)` @ 50 Hz | Cued target center for each cursor, same dim order |
| Block index | `blockNum` | `(T,)` | Active block per bin. Value changes mark (unrecorded) recording gaps |
| Trial timing | `goCue`, `durationBinnedDelay`, `durationBinnedMove` | per trial (`R`) | GO index (**MATLAB 1-based**), delay / move lengths in bins |
| Movement type | `moveType` | per trial (`R`) | 1=UniR, 2=UniL, 3=Bi (`moveTypeName` holds the strings) |
| Day notes | `Notes` | per day | Free-text description |

- **Bin size / rate:** all streams share one 20 ms grid (50 Hz).
- **Only threshold crossings are released** for the neural feature, and no spike-band power.
- **Neural signal chain (from the paper).** Analog band 0.3-7.5 kHz, digitized at 30 kHz with 250 nV resolution, common-average referenced across each array, then a 250 Hz digital high-pass per electrode before spike detection at -3.5 x RMS.
- **Control loop (OL / CL / CLR) is per block and lives only in `readme.txt`**, not in the `.mat`. It is transcribed to `utils.SESSION_BLOCKS`.
- **In closed-loop blocks `Cp` is decoder output** (neural-driven), not the autonomous instructed trajectory of open-loop blocks. The cursor stream means different things across loop types.
- **`durationBinnedMove` also means different things across loop types.** In open loop it is the prescribed movement duration (1-2 s). In closed loop it is the time to target acquisition, which is the paper's primary performance metric, and it saturates at the 10 s timeout on failed trials.

## Source Data Format & Layout

The Dryad deposit contains **13 MATLAB files** (about 135 MB), one per experiment day, plus a source `readme.txt` documenting the fields and per-block conditions. Each file is loadable with `scipy.io.loadmat`. Filenames are `t5_MM_DD_YYYY.mat`, parsed by `utils.session_date()`.

```text
<NEURALSHED_DATA_ROOT>/deo_2024/
  references/
    Deo_2024_Sci_Rep_14_1598.pdf                    # publication (add manually)
    bimanualBCI/                              # code repo clone (add manually)
  data/
    Deo2024Bimanual/                          # neuralset Study folder
      README.md                               # Dryad landing-page README
      readme.txt                              # source field + block documentation
      t5_06_02_2021.mat
      t5_06_04_2021.mat
      ...
      t5_10_13_2021.mat
```

Trials per day range from 276 to 1439 (11,044 total). Blocks per day range from 4 to 28, though block *numbers* run as high as 33 because the numbering is not contiguous within a day. The day's array is blocks concatenated in order, and a block's bins are a contiguous run within it, so slicing by `blockNum` recovers each block's continuous stream.

## Study, timelines and Events

`study.py` defines **`Deo2024Bimanual`**: **one timeline per block** (204 across the 13 days). A block is the natural timeline unit because it is both the continuous recording unit (recording pauses between blocks) and the single-condition unit (`loop_type` is fixed within a block). A whole day would span gaps and mix control loops. It also makes the paper's block-wise mean removal a per-timeline operation. `iter_timelines` slices each day by `blockNum`. Each timeline carries `subject="T5"`, `session=<t5_MM_DD_YYYY>`, `block`, `loop_type`, and `date`, broadcast to every event row.

Per timeline it emits:

| Event type | Count | Source → shape / rate | Key attributes |
|---|---|---|---|
| `SpikeCounts` | 1 | `tx` block slice → `(192, n_bins)` @ 50 Hz | `method="threshold_crossing"`, `threshold="-3.5 x RMS"`, `unit="counts/bin"` |
| `CursorPositionSeries` | 2 (right, left) | `Cp` rows → `(2, n_bins)` | `dims=(x,y)`, `effector=right|left`, `unit="a.u."`, `reference_frame="screen_center"` |
| `TargetPositionSeries` | 2 (right, left) | `Tp` rows → `(2, n_bins)` | `dims=(x,y)`, `effector=right|left` |
| `Stimulus` (movement cue) | per trial | trial timing + `moveType` | `stimulus_kind="movement_cue"`, `modality="visual"`, `code` (moveType), `description` (UniR/UniL/Bi), `trial_id` |
| `Action` (attempted move) | per trial | trial timing | `action_kind="cursor_control"`, `task_condition="attempted"`, same `code`, `trial_id` |

The two hands are split into separate homogeneous `CursorPositionSeries` / `TargetPositionSeries` (right = `Cp` rows 0-1, left = rows 2-3), each tagged with an `effector` attribute, so a model can select one hand. The whole-block streams load lazily and dims-first via `SpecialLoader` (`_load_series` slices the day array to the block's bin range, transposes neural to `(192, n_bins)`, and picks the two position rows), and are read by `NeuralFeatureExtractor` / `BehavioralSeriesExtractor`. The trial rows are pure time-span metadata computed from the per-trial bin indices, with the `Stimulus` spanning the delay-to-move window and the `Action` spanning the GO-to-move-end window.

**Filterable timeline attributes:** `loop_type` (`open_loop` / `closed_loop` / `closed_loop_recal`), `session`, `block`, `date`. Reusable types come from the package (`neuralshed.events` and `neuralshed.extractors`), so no deo-specific Event or Extractor is needed.

## Notes / Scope

- **Download** via the Study's `_download()` (shared `datasets/dryad.py` `Dryad`), which needs a Dryad API token (`NEURALFETCH_DRYAD_TOKEN`). The Study resolves its source path to `deo_2024/data/Deo2024Bimanual/`.
- **Open-loop vs closed-loop matters for the target.** In open-loop blocks `Cp` is the autonomous instructed trajectory, in closed-loop blocks it is the online decoder's output. Filter on `loop_type` when choosing a decoding target, and prefer open-loop cursor kinematics for a clean intended-movement target.
- **Closed-loop recalibration is not clean decoder output either.** `closed_loop_recal` blocks carry experimenter-set push and error assistance plus a locked non-active cursor, and the assistance level is never logged. Treat CLR kinematics as partly experimenter-injected.
- **Failed closed-loop trials are present and unlabelled, but recoverable.** A closed-loop trial that did not acquire both targets within 10 s was a failure, and `durationBinnedMove` saturates at that timeout. **974 of 3196** `closed_loop` trials (30.5%) and **105 of 976** `closed_loop_recal` trials (10.8%) sit at exactly 499 bins (9.98 s). For now, we need to filter them with `durationBinnedMove < 499` until the Study emits a success attribute, otherwise about 30% of closed-loop training targets are failures. Consider adding a `trial_outcome` attribute.
- **`goCue` is a MATLAB 1-based index** into the time axis. Accessors convert it to a 0-based bin index.
- **Later days vary the closed-loop condition** (RNN vs linear decoder, gains, data augmentation, sequential-unimanual vs true-bimanual movement strategy). These finer per-block labels are documented in `readme.txt` but are not currently carried as timeline attributes beyond the coarse `loop_type`.

## Files

- `metadata.py`: catalog `DATASET_METADATA` + `sessions()` + `get_metadata()`.
- `utils.py`: source readers (session discovery, MATLAB loading, per-trial table, block bin ranges) and constants, plus the `SESSION_BLOCKS` open/closed-loop map.
- `study.py`: the neuralset `Study` (`Deo2024Bimanual`), one timeline per block.
- `01_explore_source_data.ipynb`: source structure, the 13 days, neural and cursor/target signals, trial timing, open- vs closed-loop.
- `02_explore_study.ipynb`: build the Study, read every event family through its extractor.
- `03_model_example.ipynb`: (Phase 3, pending) decode cursor kinematics from neural features.
