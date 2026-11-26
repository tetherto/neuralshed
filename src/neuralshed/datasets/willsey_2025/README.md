# Willsey 2025: Closed-loop finger control and quadcopter game

Intracortical BCI recordings from BrainGate2 participant T5, implanted with two 96-channel Utah arrays in the hand knob of left precentral gyrus. During **closed-loop finger target acquisition**, 58 blocks over seven recording days in March and April 2023 record **spike-band power** on 192 electrodes in 50 ms bins alongside the online decoder's **finger posture** and the **cued target** posture. Three finger groups are controlled (thumb, index-middle, ring-little), with the thumb contributing both flexion and abduction, for four degrees of freedom in total. Two additional open-loop blocks cue single-degree-of-freedom movements against a rest baseline, and one saved quadcopter flight path is released without any neural data.

## References

- **Paper:** Willsey et al. 2025, "A high-performance brain-computer interface for finger decoding and quadcopter game control in an individual with paralysis," *Nature Medicine* 31(1):96-104, [10.1038/s41591-024-03341-8](https://doi.org/10.1038/s41591-024-03341-8). Open access via [PMC11750708](https://pmc.ncbi.nlm.nih.gov/articles/PMC11750708/).
- **Source data:** Dryad [10.5061/dryad.1jwstqk4f](https://doi.org/10.5061/dryad.1jwstqk4f), 181.5 MB, CC0-1.0.
- **Code repository:** https://github.com/WillseyBCILab/BCI_Finger_Decoding_Quadcopter

The reference code is the only source for several facts the released files do not carry. `t5_parse.m` defines the open-loop trial boundaries and the notebook's `CalculateTrialStats` the closed-loop ones, `RunAnalysis.m` names the four degrees of freedom and the nine open-loop conditions, `ManuscriptFigsFinal.ipynb` gives the column selection and the per-block decoder labels, and `PlotFlightPath.m` fixes the quadcopter coordinate frame.

## Species & Subjects

- **Species:** *Homo sapiens* (human).
- **Subject:** BrainGate2 participant **T5** (NCT00912041, FDA IDE G090003), male, right-handed, 69 years old at the time of the study.
- **Clinical condition:** C4 AIS-C spinal cord injury with tetraplegia, leaving only non-functional twitches and micromotion of the extremities.
- **Implant:** two 96-channel Neuroport (Blackrock) silicon microelectrode arrays with 1.5 mm electrodes, 192 channels total, in the **hand knob area of the left precentral gyrus**. The paper shows the MRI reconstruction in Extended Data Fig. 1a and publishes no stereotaxic coordinates.
- **Time since implant:** the first released session is 2,395 days (6.6 years) after the 2016-08-17 implant. T5 also appears in `willet_2021`, `fan_2023`, `wilson_2025`, `deo_2024` and `gusman_2025`.
- **Lab / institution:** Neural Prosthetics Translational Laboratory, Stanford University, with Brown University and the Providence VA (BrainGate consortium).

## Behavioral Task

**Closed-loop finger target acquisition.** A virtual hand on screen is driven by the online decoder. Each trial shows a target posture and the trial ends when the participant holds every controlled degree of freedom inside its target for **500 ms**, or when 10 s elapse. The target spans **plus or minus 0.1** of the range of motion around its centre, which is the 20% target width quoted in the paper. Positions are a fraction of the range of motion, where 1 is full flexion (full abduction for the thumb's sideways dimension), 0 is full extension (full adduction) and 0.5 is the neutral posture. The authors' materials disagree on the direction. The paper's Methods give flexion and abduction the positive sign, the authors' notebook takes that sign from the change in target, and its plots label positions "% Flex", but `RunAnalysis.m` and the Extended Data Fig. 1 it draws name 0 flexion. The Study follows the Methods, and its open-loop conditions are named accordingly (`thumb_flexion-flex` is a target of 1).

The release does not record whether a trial was acquired, and the flash that closes a trial fires at the timeout too, so `utils.trial_success` derives it with the test of the authors' `CalculateTrialStats`: every cued degree of freedom within 0.1 of the target for the 500 ms before the flash. 87 of the 3,172 trials fail it (97.3% success), all of them 10.0 to 10.43 s long, so every failure is a timeout. The test runs on the full-rate task stream, not on neural bins as the authors' does, because the neural recording of `t5.2023.03.09_block18` stops 155 ms before its last trial's flash, and on bins that acquired trial would count as a failure.

Trials strictly **alternate a randomly placed target with a return to the neutral posture**, so 49% of the 3,172 trials are return-to-rest. Each outward target moves two of the three fingers away from neutral (the thumb, whose two degrees of freedom count as one finger, the index-middle group and the ring-little group), except in blocks 44, 46, 47 and 50 of 2023-04-13, where the authors cued one finger at a time. `utils.new_targets` reads that count off the target stream. Median trial duration is 1.90 s (5th to 95th percentile 0.95 s to 5.87 s).

Two task variants are interleaved across blocks. In the **four-degree-of-freedom task** all four dimensions receive targets, and in the **two-degree-of-freedom task** the thumb's abduction and the ring-little group are pinned at neutral and only thumb flexion and the index-middle group are cued. `utils.target_dof` recovers which variant a block ran, and the split is 36 blocks against 22.

The **decoder** the block ran is a separate fact. On 2023-04-06 and 2023-04-13 the authors interleaved a four-degree-of-freedom decoder driving the two-degree-of-freedom task with a genuine two-degree-of-freedom decoder, and the task stream looks identical either way because the game pins the unused dimensions in both cases. `utils.decoder_dof` gives four on the four-degree-of-freedom task, which a two-output decoder cannot drive. For the 22 two-degree-of-freedom task blocks it reads `utils.DECODER_DOF`, collected from the authors' `ManuscriptFigsFinal.ipynb`: cell 11 is their two-degree-of-freedom decoder population (2023-03-21 blocks 18 and 19, 2023-03-23 blocks 7 to 9, 2023-04-06 blocks 15 and 16, the paper's "529 trials (3 days) for the 2D decoder"), and cell 18 labels twelve blocks of 2023-04-06 and 2023-04-13. Ten of those twelve have a trained decoder in the release's `Decoders/` folders, whose normalization vectors (`myMean`, `mySDev`) hold one entry per decoder output, and all ten agree with cell 18. The last three blocks, 2023-04-06 blocks 41 and 48 and 2023-04-13 block 32, carry no label, and cell 31 groups them with the two-degree-of-freedom decoder blocks. For block 32 that makes four such blocks on 2023-04-13, where the paper's Table 2 lists three (decoders 10, 10b and 10c, one block each).

**Open-loop cued movement** (two blocks on 2023-04-06). The participant attempts a cued single-joint movement while the hand on screen is driven by the task rather than by a decoder. The cued posture changes every 3 s, back to back: each step is 1 s with the virtual fingers still, then a 2 s go phase in which they move to the new posture. Whether the target is drawn during the still second is not documented. Each block has 100 go phases: 44 drive one degree of freedom out to full flexion or full extension, 44 bring it back to neutral, and 12 start and end at neutral. The Study names an outward phase by its end posture (eight conditions), a return by the outward movement it undoes (`thumb_flexion-flex_return` and so on, eight more), and keeps `rest` for the neutral-to-neutral phases only. The authors' `RunAnalysis.m` names every phase by its end posture, so its nine-class analysis counts the returns as rest. The authors' tuning analysis averages the 2 s (40 bin) go window.

**Quadcopter game.** Finger positions were mapped to velocity control of a virtual quadcopter in a Unity AirSim simulator, the thumb giving translation, the index-middle group elevation, and the ring-little group rotation. The release contains no neural data for this task, only one saved flight path.

## Data Modalities

| Modality | Source location | Type / shape |
|---|---|---|
| Neural | `binnedNeural_hlfp` | Spike-band power, `float32 (n_bins, 256)`, of which columns 0-191 are the electrodes and 192-255 are exactly zero |
| Neural bin clock | `binnedNeural_redisClock` | `int64 (n_bins,)`, unix milliseconds |
| Neural bin clock (real-time machine) | `binnedNeural_xpcClock` | `uint32 (1, n_bins)`, milliseconds on the real-time machine's counter, not zeroed at block start (first values 6.8 to 107.2 s), closed-loop blocks only |
| Decoded finger position | `binned_taskOutput_stream_estimated` | `float32 (n_samples, 6)`, closed-loop blocks only |
| Cued finger target | `(binned_)taskOutput_stream_target` | `float32 (n_samples, 6)` |
| Trial phase marker | `(binned_)taskOutput_stream_hand_color` | `uint8 (n_samples, 4)` closed loop, `float32 (1, n_samples)` open loop |
| Task stream clock | `(binned_)taskOutput_stream_redis_clock` | `int64 (n_samples,)`, unix milliseconds |
| Task parameters | `holdTime`, `fn_hold_threshold` | strings, `"0.5"` and `"0.1"` in every closed-loop block |
| Quadcopter flight path | `x`, `y`, `z` | `float32 (81601, 1)` each, no clock |

- **Neural signal type:** spike-band power, which the source names `hlfp` (high-frequency LFP). The Supplementary Methods give the full chain: the amplifier's **0.3 Hz to 7.5 kHz analog filter**, sampling at 30 kHz with 250 nV resolution, then **common average referencing**, a **250 Hz digital high-pass**, and the **sum of the squared magnitude** over each 50 ms window. The band therefore runs from 250 Hz to 7.5 kHz (`lower_freq=250`, `upper_freq=7500`), and the unit is `uV^2`.
- **The 50 ms bins sit on the scale of a mean, not a sum.** The typical raw value per channel is about 1,400 `uV^2` (median over channels, four closed-loop blocks spread across the release), against 370 to 725 in `singer_2025`, `willet_2023` and `kunz_2025`, which average. A plain sum of the 1,500 samples in a 50 ms bin would be about 1,500 times larger than a mean. The sum the Supplementary Methods describe was therefore divided by a fixed count, and [The 20 ms prefix](#the-20-ms-prefix) shows that the count is the same for every bin. The stream behaves as a mean square, so rebinning averages.
- **One front-end configuration covers all the neural data.** The rig ran in three configurations across the study, but the two later ones cover the quadcopter sessions, so all seven recording days in this release share the analog configuration described above.
- **Bin size / rate:** **50 ms bins at 20 Hz**, confirmed from the clocks rather than assumed. The **first 76 to 88 bins of every block are spaced 20 ms, not 50 ms**. The authors drop the first 80 before normalizing (`ManuscriptFigsFinal.ipynb`: "First 80 samples are noisy"). The earliest trial starts at bin 121 in open loop and bin 192 in closed loop, so the affected bins fall outside the task anyway.
- **Task stream rate:** roughly 1.3 kHz in the closed-loop blocks and 500 Hz in the open-loop blocks, with duplicate millisecond timestamps. The decoder itself updates every 50 ms, matching the neural bins.
- **Alignment:** the two streams share only the unix-millisecond `redisClock`. `utils.on_bins` interpolates the task stream onto the neural bin times, as the authors do. `utils.trial_windows` places trial edges on bins by timestamp rather than by nearest match, which is a deliberate one-bin divergence from the reference (see [The hand colour marker](#the-hand-colour-marker)).

### The 20 ms prefix

Every block opens with 76 to 88 bins spaced 20 ms instead of 50 ms, always contiguous from bin 0, and the switch is a step rather than a ramp. These are **not transients**. Every bin is divided by the same count, sized for a 50 ms window, so a 20 ms bin carries 40% of a 50 ms one, and the numbers say exactly that: across all 60 neural blocks the raw prefix sits at 0.386 of the block mean (20/50 = 0.4), and rescaling it by 50/20 recovers the bins that follow, with a median per-channel correlation of 0.97 and a level ratio of 0.956.

The practical consequence is that the prefix biases any per-channel mean and standard deviation computed over the whole block, which is why the authors drop it before z-scoring and the Study starts every stream after it. The only genuine artefact is the single short bin at the changeover. This measurement was made before the Supplementary Methods were read, and it fits their sum once every bin is divided by the same count.

### The six finger columns

The position and target streams carry six columns but only **four independent values**. Per the source README the columns are the two thumb degrees of freedom followed by index, middle, ring and little, and index duplicates middle while ring duplicates little because those fingers move as a group. The authors select columns `[0, 1, 2, 5]`, exposed as `utils.DECODED_COLUMNS` with names `thumb_flexion`, `thumb_abduction`, `index_middle` and `ring_little`.

### The hand colour marker

`hand_color` means different things in the two block kinds, and both are needed to segment trials.

| Block kind | Code 1 | Code 2 | Code 3 |
|---|---|---|---|
| Closed loop | a roughly 10 ms flash that closes each trial, at acquisition or at the 10 s timeout | every other sample | never |
| Open loop | the 1 s with the fingers still | the 2 s go phase | "stationary" per the release README: a single sample between a go phase and the next still second, 2 and 4 times per block |

In the closed-loop blocks a trial therefore runs from the end of one flash to the start of the next. The authors' notebook delimits the same trials by target changes on the binned stream (`TranLog` in `CalculateTrialStats`).

Placing those edges on bins is where `utils.trial_windows` diverges by one bin from `t5_parse.m`, the authors' open-loop parser, which matches each edge to its *nearest* bin, but a bin's timestamp marks the end of its 50 ms window and the flash is shorter than one bin, so nearest-matching opens a trial on the window that mostly covered the *previous* target. Measured on the released blocks, the cued target changes at `trial start + 1` in 95 of 101 trials, and consecutive trials share a bin, which an extractor rejects as overlapping coverage. Taking the first bin timestamped after the flash instead moves the target change onto the trial start in 100 of 101 trials and makes the windows half-open and contiguous. The authors only used nearest-matching to average 2 s open-loop windows, where one bin is immaterial.

## Source Data Format & Layout

One MATLAB v5 file per task block, 1.2 MB to 5.4 MB each, named `t5_t5.<YYYY.MM.DD>_Data_RedisMat_<YYYYMMDD>_<HHMMSS>_(<block>).mat`. The trailing number is the source's own block number and is not contiguous.

```
Data/
├─ 20230309/RedisMat/   1 closed-loop block
├─ 20230314/RedisMat/   1
├─ 20230316/RedisMat/   7
├─ 20230321/RedisMat/   2
├─ 20230323/RedisMat/  14
├─ 20230406/RedisMat/  13   + Decoders/  (4 .mat + 4 .pt)
├─ 20230413/RedisMat/  20   + Decoders/  (6 .mat + 6 .pt)
├─ 20230712/RedisMat/   1   the quadcopter flight path
└─ OpenLoopData/        2 open-loop blocks (recorded on 2023-04-06)
```

Session labels follow the source's own day naming, `t5.2023.03.09`, which is also what `wilson_2025` uses for the same participant. Block labels add the block number, `t5.2023.03.09_block18`.

| session | blocks | duration | trials | 4-DoF blocks | 2-DoF blocks |
|---|---|---|---|---|---|
| t5.2023.03.09 | 1 | 4.4 min | 101 | 1 | 0 |
| t5.2023.03.14 | 1 | 5.7 min | 101 | 1 | 0 |
| t5.2023.03.16 | 7 | 21.8 min | 346 | 7 | 0 |
| t5.2023.03.21 | 2 | 9.7 min | 318 | 0 | 2 |
| t5.2023.03.23 | 14 | 40.9 min | 688 | 11 | 3 |
| t5.2023.04.06 | 15 | 35.1 min | 638 | 3 | 10 |
| t5.2023.04.13 | 20 | 37.4 min | 980 | 13 | 7 |
| t5.2023.07.12 | 1 | n/a | 0 | 0 | 0 |

Totals are 188,936 neural bins (157 minutes), of which 176,142 bins (144 minutes) are closed loop and 12,794 bins (11 minutes) are open loop. Blocks run 1,543 to 6,898 bins.

The `Decoders` folders hold the trained network weights (`.pt`) and the per-decoder normalization constants (`mask`, `myMean`, `mySDev` in `.mat`) used online. They are decoder artefacts rather than recordings and are not read.

## Study, timelines and events

`study.py` (`Willsey2025FingerDecoding`) treats **each block as one timeline**, 60 in total, rather than each day, because each block is a separate file on its own stretch of the shared clock, 39 s to 78 min from the next with no overlaps, and a timeline is the largest span on one continuous clock. A day holds up to twenty blocks, whose task dimensionality and online decoder differ. The quadcopter block is not a timeline, since it holds no neural data.

Every stream starts at `utils.steady_state_start` rather than at bin 0, so a uniform 20 Hz time base holds for every array a segment returns. The earliest trial starts at bin 121 (open loop) or 192 (closed loop), after the narrow bins end (bin 89 at the latest), so nothing is lost.

The finger streams cover only the bins the task stream reaches (`utils.task_span`). The neural recording starts about 7.3 s before the task stream in the closed-loop blocks and 2.4 s before it in the open-loop ones (medians), and runs up to 29 s past its end (`t5.2023.04.06_block48`). Interpolating the task stream onto those bins would repeat its first or last sample, 17,873 of 184,025 bins (894 s), so `TargetGraspSeries` and `GraspSeries` start and end inside the `BandPower` row, and the neural data before and after them stays available. The authors' `LoadData` drops those neural bins.

| Event type | Count | Source | Shape / rate | Key attributes |
|---|---|---|---|---|
| `BandPower` | 1 | `binnedNeural_hlfp` | `(192, n_bins)` at 20 Hz | `unit="uV^2"`, `normalization="raw"`, `lower_freq=250`, `upper_freq=7500`, `processing` records the analog filter, the referencing, the 250 Hz high-pass and the per-bin sum over a fixed count |
| `TargetGraspSeries` | 1 | `taskOutput_stream_target` | `(4, n_task_bins)` at 20 Hz, over `utils.task_span` | `dims` = the four decoded degrees of freedom, `unit="proportion"` |
| `GraspSeries` | 1, closed loop only | `binned_taskOutput_stream_estimated` | `(4, n_task_bins)` at 20 Hz, over `utils.task_span` | same dims and unit, `kind="online_decoder_output"` |
| `Stimulus` | per go phase, open loop only | hand-colour edges | time span | `modality="visual"`, `task_condition="na"`, `period="go"`, `code`, `condition`, `target_<dim>` |
| `Action` | per trial | hand-colour edges | time span | `task_condition="attempted"` (`"na"` with `action_kind="rest"` on rest phases), `period="go"`, same `code` and `condition`, `success` on closed-loop trials |

`TargetGraspSeries` is a new reusable event type in `neuralshed/events.py`, a **sibling** of `GraspSeries` rather than a subclass: subclassing would make an extractor selecting `"GraspSeries"` sweep the target stream too and fail on overlapping coverage.

No velocity stream is emitted. The source ships positions only, and the online decoder's own velocity output was never released, so deriving one inside the Study would be inventing data. `03_model_example.ipynb` differences the posture stream where that step is visible, which is also what the authors' `LoadData` does.

Timeline keys broadcast to every row of their block: `session`, `run`, `task` (`fingertargets` or `cuedmovement`), `block`, `loop_type` (`closed` or `open`, the standard column), `task_dof`, `decoder_dof`, `new_targets`, `date` and `post_implant_day`. `task_dof`, `decoder_dof` and `new_targets` are integers, and `None` on the open-loop blocks where they do not apply, so filter with `task_dof == 4`. Closed-loop trials emit no `Stimulus`: they have no instructed delay, so the target appears when the trial starts and the `Action` already carries its onset and position. Per-trial `condition` is one of the two closed-loop conditions or the seventeen open-loop ones, and `code` is its index in `utils.CONDITION_CODES`, which spans both block kinds so a code means the same thing throughout. `trial_id` is the 0-based index of the trial or go phase within its block, so it is unique within a timeline, not across the dataset.

## Notes / Scope

- **Local files only.** The deposit is a single 181.5 MB zip on Dryad, downloaded to `<NEURALSHED_DATA_ROOT>/willsey_2025/data/Willsey2025FingerDecoding/` and read from disk. Dryad requires an API token for downloads, so export `NEURALFETCH_DRYAD_TOKEN`.
- **The quadcopter block carries no time base.** `x`, `y` and `z` are 81,601 samples with no clock and no documented sample rate, and the authors plot them against sample index. `z` points down, so elevation is `-z`, which `PlotFlightPath.m` confirms by plotting `x, y, -z` against course rings centred at `z = +14.91`. Since it has no neural data, it is out of scope for the Study.
- **No dead channels.** All 192 electrode columns are non-constant in all 60 neural blocks.
- **The neural unit is known, the behavioral one is not.** Spike-band power is `uV^2` per the Supplementary Methods. Finger position has no physical unit, being a fraction of the range of motion, and is declared `proportion`. The 50 ms power bins are on the scale of a mean square, which is what `docs/standardization.md` assumes for band power, so the extractor's treatment of `uV^2` as non-convertible, with no bin-width factor, fits this stream. Whether every dataset documented as summing is on the same scale is an open corpus-wide check.
- **Channel order.** On `t5.2023.04.13` columns 0-95 and 96-191 are exchanged relative to every other day, in all 20 blocks, and `utils` swaps them back. No source states it, so the swap is inferred from the data. With it, the 192 columns match the order of the other T5 datasets as a candidate, `array_1` and `array_2` being their `lateral_array` and `medial_array`. Evidence in [t5.md](../../../../docs/channel_mapping/t5.md).

## Files

- `metadata.py`: Metadata for database insertion.
- `utils.py`: Dataset constants + accessors (`list_blocks`, `load_block`, `get_neural`, `bin_times`, `steady_state_start`, `on_bins`, `get_finger_position`, `get_finger_target`, `trial_windows`, `trial_targets`, `target_dof`, `decoder_dof`, `new_targets`, `cue_windows`, `quadcopter_path`).
- `study.py`: the neuralset `Study` (`Willsey2025FingerDecoding`).
- `01_explore_source_data.ipynb`: source-data exploration demo.
- `02_explore_study.ipynb`: Study demo.
- `03_model_example.ipynb`: decoding example.
