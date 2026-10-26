# Karpowicz 2025 — Cursor control across two sessions 274 days apart

Intracortical recordings from BrainGate2 participant **T11**, two 96-channel Utah arrays in the **left precentral gyrus**, on **two sessions 153 and 427 days after implant**. The deposit exists to support work on decoder **stability**: the paper's NoMAD method aligns latent dynamics across recording days so a decoder keeps working without supervised recalibration, and these two sessions are the human half of that evaluation.

The task is not the usual center-out. After one radial-8 calibration block, T11 plays a **"Simon" memory-matching game**: four coloured targets illuminate in sequence and he imagines right-hand movements to drive the cursor through that sequence, one target at a time. A released trial is therefore **one movement**, not one sequence, and the four movements of a sequence share a single success or failure.

## References

- **Paper:** Karpowicz et al. 2025, "Stabilizing brain-computer interfaces through alignment of latent dynamics," *Nature Communications* 16:4662 - [10.1038/s41467-025-59652-y](https://doi.org/10.1038/s41467-025-59652-y).
- **Source data:** Dryad [10.5061/dryad.q83bk3jtp](https://doi.org/10.5061/dryad.q83bk3jtp), two NWB files, 262 MB, CC0-1.0. The monkey datasets in the same paper are a **separate** deposit ([10.5061/dryad.cvdncjt7n](https://doi.org/10.5061/dryad.cvdncjt7n)) and are out of scope here.
- **Code repository:** https://github.com/snel-repo/nomad
- The same recordings were first published in Rubin et al., *J Neurosci* 2022.

## Species & Subjects

- **Species:** *Homo sapiens* (human).
- **Subject:** BrainGate2 participant **T11** (NCT00912041, FDA IDE G090003), a 36-year-old right-handed man with tetraplegia from a cervical spinal cord injury. Sessions took place at his residence.
- **Implant:** two 96-channel Blackrock Utah arrays in the left precentral gyrus, 192 electrodes. The deposit gives no per-array channel block, so `metadata.py` splits them 0-95 and 96-191 by convention rather than by a published mapping.
- **Sessions:** 153 and 427 days after array placement. The same T11 as `vargas_2025` and `gusman_2025`, where he is 36 and 39.

## Behavioral Task

Each session opens with a **radial-8 center-out calibration block** used to fit the session's Kalman filter, cueing eight targets at radius 0.5 (0.35 on the diagonals). Cursor paths in this block are noticeably less direct than in the blocks that follow (median straightness 0.70 against 0.90), which fits a partly assisted calibration rather than a computer-driven replay.

Every later block is the **Simon memory-matching task**. Four coloured targets illuminate one at a time for 400 ms each, with a distinct tone, and T11 imagines moving his right hand to drive the cursor out to each in the order presented. Dwelling 300 ms on the correct target acquires it and the cursor **recentres automatically**; the trial fails if the cursor dwells on a wrong target or takes longer than 5 s. A block holds 16 sequences, 12 of them a fixed "target" sequence that repeats across the session's blocks and 4 randomly chosen distractors. T11 was not told a fixed sequence existed.

The released trial is **one of the four movements**, cut from the cursor's outward zero-crossing to 600 ms later or to target acquisition, whichever comes first. Median trial length is 0.76 s. `result` is the **sequence's** outcome, so all four movements of a sequence carry the same flag: 93.2% of day 153's movements and 84.8% of day 427's belong to successfully completed sequences.

## Data Modalities

| Modality | Source location | Type / shape |
|---|---|---|
| Threshold crossings | `acquisition/binned_spikes` | `float64 (n_bins, 192)`, integral counts, up to 15 per bin |
| Cursor position | `acquisition/cursor_pos` | `float64 (n_bins, 2)` |
| Cursor velocity | `acquisition/cursor_vel` | `float64 (n_bins, 2)` |
| Target position | `acquisition/target_pos` | `float64 (n_bins, 2)` |
| Block identity | `acquisition/block_num` | `float64 (n_bins, 1)` |
| Condition identity | `acquisition/cond_id` | `float64 (n_bins, 1)` |
| Trial table | `intervals/trials` | one row per movement: `cond_id`, `tgt_x`, `tgt_y`, `result`, `is_calibration`, `block_num` |

- **Bin size / rate:** 20 ms bins at **50 Hz**, one clock shared by every stream. Mean firing rate is roughly 25 Hz per channel and no channel is silent.
- **Workspace units are arbitrary.** The source calls the unit "position" and publishes no scale. Simon targets sit at radius 0.375, calibration targets at 0.5, and the cursor stays within 0.63 of centre.

### The three source quirks worth knowing

1. **`pynwb` cannot open these files.** Both carry a de-identified `session_start_time` of `1900-10-01T00:00:00-05:50:36`, and `dateutil` refuses that seconds-resolution UTC offset, so `NWBHDF5IO.read` raises before any data is reached. `utils.py` reads the HDF5 tree directly with `h5py`. Upstream this is fixed by [pynwb #2230](https://github.com/NeurodataWithoutBorders/pynwb/pull/2230), merged for a release after 4.1.0, so the h5py path can be revisited once that ships.
2. **Timestamps are absolute Unix nanoseconds**, not the seconds NWB nominally stores. Dividing by `1e9` recovers a normal clock. As a side effect the real recording dates survive in the timestamps even though the file's own date field was scrubbed, so this dataset records **days since implant only** and no calendar dates.
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

The file names carry no ordering: `session2.nwb` is the **earlier** session. This package keys sessions by implant day (`T11_day153`, `T11_day427`) instead, and enumerates them in chronological order.

Only the task blocks were released. Block numbers therefore skip (day 427 runs 3, 7, 8, 9, 12, 14, 15, 16, 18), consecutive blocks are separated by gaps of up to 70 minutes, and the 30 minute baseline rest the paper describes is absent. Total released data is 49.9 minutes across 20 blocks and 1,116 movements, of which 1,112 are emitted (see below).

## Study, timelines and events

`study.py` (`Karpowicz2025CursorStability`) treats **each block as one timeline**, 20 in total, following `wilson_2025`. A session is not a continuous recording, but a block is: within one, the 20 ms grid runs unbroken apart from the odd dropped bin, and the task entities (which targets are cued, whether the block calibrates) hold still.

| Event type | Count | Source | Shape / rate | Key attributes |
|---|---|---|---|---|
| `SpikeCounts` | 1 | `binned_spikes` | `(192, n_bins)` at 50 Hz | `unit="counts/bin"`, `normalization="raw"` |
| `CursorPositionSeries` | 1 | `cursor_pos` | `(2, n_bins)` at 50 Hz | `dims=("x", "y")`, `unit="a.u."` |
| `TargetPositionSeries` | 1 | `target_pos` | `(2, n_bins)` at 50 Hz | `dims=("x", "y")`, `unit="a.u."` |
| `DecodedCursorVelocitySeries` | 1 | `cursor_vel` | `(2, n_bins)` at 50 Hz | `dims=("x", "y")`, `unit="a.u./s"` |
| `Action` | per movement | trial table | time span | `code`, `target_x`, `target_y`, `target_angle_deg`, `success`, `trial_id` |
| `Intertrial` | per gap | trial table | time span | |

**The emitted clock counts bins, not wall time.** The source drops individual 20 ms bins (251 across the release, 246 of them in day 427, and 243 of *those* in its calibration block), so a block's wall span runs up to 4.9 s ahead of `n_bins / 50`. Trial times are resolved against the source timestamps in `utils.trials` and then expressed in bin indices, which keeps every `Action` aligned with the dense stream it indexes into.

**Four movements are dropped.** In day 427, four pairs of trials share a start time exactly, an artifact of the authors' onset detection: for an up-then-down pair the backward search for the centre crossing landed on the same instant twice. The cursor follows the first of each pair, and re-anchoring the second would be inventing timing, so `utils.trials` keeps the first and drops the second. That leaves 1,112 movements of 1,116, and keeps trigger times unique, which the segmenter requires.

**No `Stimulus` row is emitted.** The deposit publishes no cue timing, and a trial already starts after the target has been illuminated, so inventing a cue window would be fiction. The cued target instead rides every `Action` and the continuous `TargetPositionSeries`.

Timelines carry `session`, `block`, `task`, `is_calibration`, `post_implant_day`, `n_targets` and `dropped_bins`, so a query can hold out a whole session, keep only the Simon blocks, or avoid the calibration block's dropped bins.

## Notes / Scope

- **Local files only.** A single Dryad deposit downloaded to `<NEURALSHED_DATA_ROOT>/karpowicz_2025/data/`. Dryad requires an API token, so export `NEURALFETCH_DRYAD_TOKEN`. The Study resolves into the class-named subfolder, which is where `_download` lands the files.
- **Nothing is read eagerly.** `utils` keeps the `h5py` handle open and slices each block out of it, so asking for one block never reads the other 19.
- **No calendar dates**, by choice: see quirk 2 above. `post_implant_day` (153 and 427) is what the paper publishes and what the sessions are keyed on.
- **The decode target is a kinematic**, so `decode_target` is `movement`.
- **Channel order is unverified** against the other T11 datasets (`vargas_2025`, `gusman_2025`), and the deposit names no arrays, so the 0-95 / 96-191 split in `metadata.py` is a convention. See the channel-mapping table in [standardization.md](../../../../docs/standardization.md).
- **Out of scope:** the monkey isometric-force and reaching datasets analyzed in the same paper, which live in a separate Dryad deposit.

## Files

- `metadata.py`: Metadata for database insertion.
- `utils.py`: Dataset constants and accessors (`list_blocks`, `get_stream`, `trials`, `bin_times`, `dropped_bins`, `target_layout`), including the h5py reads and the two unit corrections.
- `study.py`: the neuralset `Study` (`Karpowicz2025CursorStability`).
- `01_explore_source_data.ipynb`: source-data exploration demo.
- `02_explore_study.ipynb`: Study demo.
- `03_model_example.ipynb`: cursor-velocity decoding example.
