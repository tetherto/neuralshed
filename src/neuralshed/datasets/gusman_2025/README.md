# Gusman 2025 — Sustained gestures during 2D cursor control

Intracortical recordings from two BrainGate2 participants, **T11** and **T5**, each with two 96-channel Utah arrays in the hand knob of the **left** precentral gyrus. A cursor BCI can point and it can click, but it cannot **hold**, and dragging requires keeping one gesture active for seconds while the cursor moves. The release holds **five sessions** across two tasks: an open-loop **Gesture Hero** task measuring how a gesture representation decays over 1, 2 and 4 second holds, and a closed-loop **drag-and-drop** task driven by a **latch decoder** that holds a gesture once it has been stable rather than re-deciding every 20 ms.

Two things make it unusual in the catalog. It ships **five local field potential bands** alongside threshold crossings and spike-band power, the first genuine `lfp` features here. And its closed-loop status is recorded **per stage and per decoder**, since the kinematic and gesture decoders close independently.

## References

- **Paper:** Gusman et al. 2025, "Multi-gesture drag-and-drop decoding in a 2D iBCI control task," *Journal of Neural Engineering* 22:026054 - [10.1088/1741-2552/adb180](https://doi.org/10.1088/1741-2552/adb180). Preprint at [10.1101/2024.09.17.24313659](https://doi.org/10.1101/2024.09.17.24313659).
- **Source data:** Dryad [10.5061/dryad.98sf7m0v1](https://doi.org/10.5061/dryad.98sf7m0v1), three files, 7.7 GB, CC0-1.0.
- **Code repository:** https://github.com/jgusman/drag-and-drop

Same Brown group as `vargas_2025` with the authorship inverted: Vargas-Irwin is last author here and first there, and Gusman is a co-author there.

## Species & Subjects

- **Species:** *Homo sapiens* (human).
- **Subjects:** BrainGate2 participants **T11** and **T5** (NCT00912041, FDA IDE G090003). Sessions took place at the participants' residences.
- **T11:** 39 years old, C4 AIS-B spinal cord injury 9 to 11 years before enrollment. Both tasks. He is 36 in `vargas_2025`, recorded earlier.
- **T5:** 70 years old, C4 AIS-C. Gesture Hero only. The same T5 as `willsey_2025`, `wilson_2025`, `willet_2021`, `fan_2023` and `vargas_2025`, and 68 in the last of those.
- **Implant:** two 96-channel Utah arrays per participant, 192 electrodes, in the hand knob of the **left** precentral gyrus. The source names them `Lateral` and `Medial`, so unlike most deposits the array identity of each channel block is stated.

Worth noting that this group's reported ages advance consistently across its two papers, 36 to 39 and 68 to 70, unlike the copy-forward 69 that the Stanford T5 deposits repeat.

## Behavioral Task

**Gesture Hero** is the open-loop task and exists to measure how a gesture representation behaves over time. Instruction boxes fall on screen and the participant attempts and **maintains** one gesture for 1, 2 or 4 seconds. Nothing on screen depends on the neural signal. T11 ran two sessions of 10 blocks, T5 one of 3, in both cases 20 trials per gesture-duration condition.

T11 was cued **seven** gestures: index finger down, OK, power grasp, key grasp, open, right thumb down, and **left thumb down**. That last one differs from a right-hand class only by effector, which makes it the interesting pair. T5 used **three** of them: power grasp, OK, index finger down.

**Multi-Gesture Drag-and-Drop** is the closed-loop task, a 2D centre-out-and-return with four cardinal targets. Every block holds Move Only, Click and Drag trials, and each trial runs through named stages. The first four blocks of a session calibrate the two decoders, a steady-state Kalman filter for the cursor and the latch decoder for gesture, and the rest are assessment. Only three gestures are cued here.

The eleven stage names in the release are:

```
Prepare - Direction + Gesture        Prepare - Direction
Prepare - Direction + Gesture click  Wait Direction
Center Out                           Wait Gesture hold
Wait click                           Return & Drag
Return                               Hold
Intertrial
```

**Closed loop is per stage and per decoder here**, not a property of a block. `isKinDecoderCL` and `isGestureDecoderCL` are separate flags on every stage, and `kinErrorAttenuation` records how much the cursor was being helped, taking values 0, 0.3 and 0.5. A stage can have the cursor under closed-loop control while the gesture decoder is still open.

## Data Modalities

| Modality | Source location | Type / shape |
|---|---|---|
| Neural features | `feat` | `float64 (1344, n_bins)` |
| Feature layout | `featInfo.featNames` / `featOffset` / `arrayNames` | 7 names, 7 offsets, 2 array names |
| Task table | `taskInfo` | one row per trial (Gesture Hero) or per stage (drag-and-drop) |
| Online decode | `outCL.decodedState` | one gesture label per 20 ms bin, drag-and-drop only |
| Online likelihoods | `outCL.stateLL` | `float32 (10, n_bins)` |
| Decode threshold | `outCL.stateThresh` | `float32 (1, n_bins)`, per bin rather than a constant |
| On-target flag | `outCL.isOnTarget` | `float32 (1, n_bins)` |

- **Gesture selectivity falls with hold duration**, which is the observation the latch decoder exists to work around. Measured in a fixed 500 ms window at the end of each hold, so data volume is held constant and only elapsed time varies, the fraction of spike-band-power features that separate the gestures goes 42.7% to 35.4% to 32.8% for T11 and 55.7% to 27.6% to 26.0% for T5 across 1, 2 and 4 second holds. Averaging over the whole hold instead hides it, since a longer hold is simply less noisy.
- **Bin size / rate:** **20 ms bins at 50 Hz**, one continuous grid per session with blocks concatenated onto it. Task tables index into that grid with half-open bin ranges.
- **Normalization:** every feature is **z-scored against a rolling 3 minute window** before release. That makes the values dimensionless, so they cannot be converted through a bin rate, and it is a `processing` fact as much as a `normalization` one. See [issue #14](https://github.com/catalystneuro/neuralshed/issues/14).

### The 1,344 columns are attributable

Seven feature types for each of 192 electrodes. `featInfo.featOffset` gives the boundaries and `featInfo.featNames` names them, so every column maps to a feature type and an electrode. That is the direct contrast with `vargas_2025`, where the same group shipped an anonymous feature axis.

| columns | feature | kind |
|---|---|---|
| 0-191 | `ncTX` | non-causal threshold crossings |
| 192-383 | `spikePower` | spike-band power |
| 384-575 | `STFT 0-8` | local field potential |
| 576-767 | `STFT 12-16` | local field potential |
| 768-959 | `STFT 20-31` | local field potential |
| 960-1151 | `STFT 39-125` | local field potential |
| 1152-1343 | `STFT 129-250` | local field potential |

The five band-power rows are the first features in the catalog to use the `lfp` signal type, which `vocab.py` has always had and nothing has needed.

### Reading the categorical labels

`labels`, `cuedGesture`, `cuedDirection`, `trialAttemptType` and `decodedState` are MATLAB `categorical` objects, stored as `(1, 6)` uint32 MCOS references rather than as data.

They decode with `h5py` alone. The reference's fifth element is a 1-based object index, and `#subsystem#/MCOS` holds a header followed by one pair per object: the category names at `2 * index` and the per-element codes at `2 * index + 1`, with code 0 meaning undefined. `utils.read_categorical` does exactly that.

This is a much softer version of the problem `vargas_2025` posed, where MATLAB `string` arrays needed the character buffer reconstructed byte by byte. Verified against all eight categoricals in the release, including `decodedState`, which is one code per bin rather than one per trial.

## Source Data Format & Layout

Three files, MATLAB **v7.3**, so HDF5 and therefore `h5py` rather than `scipy.io.loadmat`.

```
sesData_GH_T5.mat    712 MB   1 session   Gesture Hero, T5
sesData_GH_T11.mat   3.4 GB   2 sessions  Gesture Hero, T11
sesData_DD_T11.mat   4.1 GB   2 sessions  Drag-and-drop, T11
```

A file holding one session stores its fields directly; a file holding several stores an object reference per session, so `utils._field` handles both.

| session | task | bins | duration | blocks | rows | gestures |
|---|---|---|---|---|---|---|
| `T5_gesture_hero_s1` | Gesture Hero | 70,596 | 23.5 min | 3 | 180 trials | 3 |
| `T11_gesture_hero_s1` | Gesture Hero | 166,543 | 55.5 min | 10 | 420 trials | 7 |
| `T11_gesture_hero_s2` | Gesture Hero | 166,605 | 55.5 min | 10 | 421 trials | 7 |
| `T11_drag_and_drop_s1` | Drag-and-drop | 219,677 | 73.2 min | 11 | 1,557 stages | 3 |
| `T11_drag_and_drop_s2` | Drag-and-drop | 191,957 | 64.0 min | 11 | 1,672 stages | 3 |

Total is 4.5 hours. Drag-and-drop rows are **stages**, not trials: session 1 holds 288 trials across its 1,557 stages.

## Notes / Scope

- **Local files only.** A single Dryad deposit downloaded to `<NEURALSHED_DATA_ROOT>/gusman_2025/data/`. Dryad requires an API token, so export `NEURALFETCH_DRYAD_TOKEN`.
- **Nothing is read eagerly.** A session's `feat` is up to 2.3 GB, so `utils` keeps the `h5py` handle open and slices it. Ask for one feature type or one bin range rather than the whole matrix.
- **No dates and no trial days.** The source publishes neither, so `post_implant_day` is null for every session and neither subject carries a `time_since_implant`.
- **The decode target is a gesture class**, so `decode_target` is `other` rather than `movement`, the same call as `vargas_2025`.
- **Channel order is unverified** against the other T5 and T11 datasets, though the `Lateral` and `Medial` names give more to go on than most deposits. See the channel-mapping table in [standardization.md](../../../../docs/standardization.md).

## Study, timelines and events

`study.py` (`Gusman2025DragAndDrop`) treats **each session as one timeline**, five in total, following `kunz_2025`. The source concatenates a session's blocks onto a continuous grid with inter-block gaps removed, so a session reads as one recording.

| Event type | Count | Source | Shape / rate | Key attributes |
|---|---|---|---|---|
| `SpikeCounts` | 1 | `ncTX` | `(192, n_bins)` at 50 Hz | `unit="counts/bin"`, `normalization="z_scored"` |
| `BandPower` | 6 | `spikePower` and the five STFT bands | `(192, n_bins)` at 50 Hz each | `unit="uV^2"`, `normalization="z_scored"`, `lower_freq`/`upper_freq` |
| `BehavioralSeries` | 1, drag-and-drop only | `outCL.decodedState` | `(1, n_bins)` at 50 Hz | the latch decoder's class index |
| `Stimulus` | per trial or cue stage | task table | time span | `gesture`, and for drag-and-drop `direction` |
| `Action` | per trial or stage | task table | time span | `hold_s`, or the stage plus both decoder flags |
| `Intertrial` | per gap, drag-and-drop only | task table | time span | |

**Every band is its own `BandPower` event, carrying its own edges.** `spikePower` is 250 Hz to 5 kHz and the five LFP bands are 0-11, 12-19, 20-38, 39-128 and 129-250 Hz, so a band is reached by `query="feature == 'lfp_12_19'"` instead of by slicing a row block out of a stacked array. Those numbers are the paper's Methods, not the file's `featNames`, which label the same bands 0-8, 12-16, 20-31, 39-125 and 129-250. The labels sit on the 3.90625 Hz grid of the 256-sample FFT at 1 kHz, so they read as the realized bin edges while the paper gives the definitions. `lower_freq`/`upper_freq` carry the definition and each event's description records the label. The six span the same session and therefore overlap, so an extractor selecting `"BandPower"` has to pick one with a `query` or join them with `aggregation="cat"`. The default `aggregation="single"` raises on the overlap, which is deliberate: it means asking for "the band power" without saying which one is an error rather than a silent choice.

**Closed-loop status rides the `Action` rows, not the timeline.** The kinematic and gesture decoders close independently and it varies stage by stage, which nothing else in the catalog needs. In drag-and-drop session 1 the kinematic decoder is closed for 1,147 of 1,271 action stages and the gesture decoder for 775.

## Files

- `metadata.py`: Metadata for database insertion.
- `utils.py`: Dataset constants + accessors (`list_sessions`, `get_features`, `feature_layout`, `epochs`, `decoded_state`, `blocks`, `gestures`) and the MATLAB categorical decoder.
- `study.py`: the neuralset `Study` (`Gusman2025DragAndDrop`).
- `01_explore_source_data.ipynb`: source-data exploration demo.
- `02_explore_study.ipynb`: Study demo.
- `03_model_example.ipynb`: gesture decoding example.
