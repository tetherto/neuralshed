# Athalye 2023: two macaques moving a 2D cursor under brain control

Two rhesus macaques, **monkey G** and **monkey J**, move a cursor on a screen with a brain-machine interface (BMI) decoder that stays fixed for the whole session, on a center-out task and an obstacle-avoidance task. The paper uses the decoder as a known map from neural activity to command and shows that the same command is issued by different activity patterns in different movements, and that invariant low-dimensional dynamics predict those differences.

This is the monkey dataset in the corpus that matches the closed-loop human intracortical brain-computer interface (iBCI) data: the cursor is driven by the decoder, not by the hand. The two monkeys were recorded with different rigs and different decoders, and the release reflects it, so they differ in rate, units and quirks.

## References

- **Paper:** Athalye, Khanna, Gowda, Orsborn, Costa and Carmena 2023, "Invariant neural dynamics drive commands to control different movements," *Current Biology* 33(14):2962-2976.e15, [10.1016/j.cub.2023.06.027](https://doi.org/10.1016/j.cub.2023.06.027). Author manuscript at [PMC10527529](https://pmc.ncbi.nlm.nih.gov/articles/PMC10527529/), preprint [10.1101/2021.08.27.457931](https://www.biorxiv.org/content/10.1101/2021.08.27.457931v2).
- **Source data:** DANDI [000404](https://dandiarchive.org/dandiset/000404) "Monkey 2D cursor BMI", 13 NWB files, 1.0 GB, CC-BY-4.0.
- **Code repository:** [pkhanna104/bmi_dynamics_code](https://github.com/pkhanna104/bmi_dynamics_code), archived at [Zenodo 8006653](https://zenodo.org/record/8006653). The NWB writer is `online_analysis/nwb_utils.py`.

## Species & Subjects

- **Species:** *Macaca mulatta* (rhesus macaque), two adult males.
- **monk_g** (monkey G, colony id mmu 38380): born 2007-05-13, recorded 2 to 19 March 2016 (8 years 9 to 10 months, `P8Y`). The paper states 7 years.
- **monk_j** (monkey J, colony id mmu 35195): born 2003-05-25, recorded 24 to 31 August 2013 (10 years, `P10Y`). The NWB says `P12Y`, copied from monkey G's comment in the writer.
- **Implant:** in both monkeys, one chronic 16 by 8 array of Teflon-coated tungsten microwires (Innovative Neurophysiology) per hemisphere, targeting the upper-arm area of primary motor cortex (M1) and posterior dorsal premotor cortex (PMd), 2.5 to 3 mm deep. Implant dates and time since implant are not published. `metadata.py` names both arrays `array_1` (one per hemisphere) and maps the region to the frontal motor cortex HOMBA term, the deepest one covering both M1 and PMd.
- **Recording:** Plexon, 128-channel MAP for J, 256-channel Omniplex D for G. Units were sorted online and the decoder units were hand-picked each session. Only those units are released.
- **Lab:** Carmena lab, UC Berkeley.

## Behavioral Task

Both tasks were performed with the same decoder in a session. The decoder was initialized from passive observation, calibrated in closed loop with closed-loop decoder adaptation, then fixed, and only the fixed-decoder blocks are released.

- **Center-out:** hold on the center target, go cue, move to one of 8 radial targets and hold. G: 10 cm reach, 1.7 cm target radius, cursor reset to center after each trial. J: 6.5 cm reach, 1.2 cm radius, returns to center himself.
- **Obstacle avoidance:** G reaches the same 8 targets around a 2 or 3 cm square obstacle placed 5 cm out on the target direction. Touching it is an error. J makes point-to-point movements around an ellipse obstacle, from start targets that are not always the center.

**Decoders.** G: velocity Kalman filter on 100 ms bins, updated at 10 Hz on a 60 Hz task loop, state in cm and cm/s. J: velocity point-process filter on binary spiking in 5 ms bins, updated every 200 Hz sample, state in m and m/s.

A released trial is a **rewarded** trial from go cue (peripheral target onset) to reward. Failed trials, holds and cursor return stay in the continuous streams but are unlabeled. There is no target, direction or condition column. Whether the arm was free or restrained is not stated, and there is no arm kinematics or electromyography.

## Data Modalities

| Modality | Source location | Type / shape |
|---|---|---|
| Decoder-unit spike counts | `processing/behavior/spike_counts` | G `uint32 (T, 35..151)` per 60 Hz frame, J `uint8 (T, 20 or 21)`, 0 or 1 per 5 ms |
| Cursor position | `processing/behavior/cursor` | `float64 (T, 2)`, G cm, J m |
| Decoder state | `processing/behavior/decoder_state` | `float64 (T, 5)`, `[pos_x, pos_y, vel_x, vel_y, 1]` |
| Target state | `processing/behavior/target_state` | `float64 (T, 5)`, all zeros for G, misaligned for J |
| Obstacle position | `processing/behavior/obstacle_position` | `float64 (T, 2)` cm, NaN in center-out blocks |
| Obstacle details | `processing/behavior/obs_details` | G strings `2_cm_sq`, `3_cm_sq`, `na`, J `float64 (T, 5)` ellipse |
| Block id | `processing/behavior/te_num` | G float task entry (4377), J string (`jeev082413d`) |
| Decoder update flag | `processing/behavior/update_bmi` | `float64 (T,)`, 1 on every sixth frame for G, always 1 for J |
| Trials | `intervals/trials` | `start_time`, `stop_time` only |

Every stream of a file shares one grid: 60 Hz for G, 200 Hz for J. There is no units table, no electrodes table, no spike times and no raw or local field potential data. `spike_counts` columns are the day's decoder units, not channels, so they are not matched across sessions and carry no area or hemisphere label.

## Source Data Format & Layout

```
athalye_2023/data/Athalye2023CursorBMI/000404/
├── dandiset.yaml
├── sub-monk-g/sub-monk-g_ses-session{0..8}.nwb   17 to 173 MB, 12.9 to 66.0 min, 315 min in all
└── sub-monk-j/sub-monk-j_ses-session{0..3}.nwb   69 to 186 MB, 26.0 to 71.0 min, 178 min nominal, 121.5 min recorded
```

One file per session, NWB 2.5. Monkey G's sessions 7 and 8 were both recorded on 19 March 2016, and J's sessions 0 and 1 both on 24 August 2013. `session_start_time` holds the real dates at midnight.

Each file concatenates several task entries (blocks) back to back with no gap. The center-out block comes first whatever the recording order (session 3 of G holds block 4499 before 4497, 4498 and 4504), so time in a file is a sample count, not wall clock, and `te_num` is the only record of the seams. G's `te_num` values increase with recording time, and J's letter suffixes do too.

The Study looks for the files under its own folder (`athalye_2023/data/Athalye2023CursorBMI/000404/`, where `_download` puts them) and streams from DANDI when they are not there.

## Study, timelines and Events

`study.py` (`Athalye2023CursorBMI`) treats **each block as one timeline**, 34 in total (26 for G, 8 for J), in file order. A file is not one continuous recording, since its blocks were recorded at different times and reordered, but a block is. Timelines carry `session` (the file stem), `task_entry` (the source's `te_num` as text), `task` (`center_out` or `obstacle_avoidance`) and `loop_type="closed"`.

| Event type | Count | Source | Shape / rate | Key attributes |
|---|---|---|---|---|
| `SpikeCounts` | 1 | `spike_counts` | `(n_units, n)` at 60 or 200 Hz | `feature="sorted_spike_counts"`, `method="sorted"`, `unit="counts/bin"`, `normalization="raw"` |
| `CursorPositionSeries` | 1 | `cursor` | `(2, n)` at 60 or 200 Hz | `feature="cursor_position"`, `unit="cm"`, `dims=("x", "y")` |
| `DecodedCursorVelocitySeries` | 1 | `decoder_state[:, 2:4]` | `(2, n)` at 60 or 200 Hz | `feature="cursor_velocity"`, `unit="cm/s"` |
| `Action` | per rewarded trial | `intervals/trials` | go cue to reward | `action_kind="cursor_control"`, `trial_id`, `success=True`, `obstacle_x`, `obstacle_y`, `obstacle_size_cm` |
| `Intertrial` | per gap | `intervals/trials` | between trials | |

- **Native rates.** Streams keep the source grid and the extractor resamples. G's counts per 16.7 ms frame convert cleanly to Hz, but its cursor and velocity only change on decoder updates (10 Hz), so they are step functions at 60 Hz.
- **Units.** J's positions and velocities are scaled from m to cm on read, so every row declares `cm` or `cm/s`. G's center target is at the origin, J's at about (4.1, 14.0) cm.
- **The velocity is the decoder's.** `decoder_state` velocity is the decoder's output, and the cursor equals its position state (to float precision for J, on over 99.2% of frames for G). The cursor step at each decoder update correlates with the velocity state at 0.94 to 0.995 per G session and 0.83 to 0.99 per J session, not exactly 1, because both filters also update the position directly.
- **Obstacle attributes** are set on G's obstacle trials only, read at the go cue (constant within every trial, obstacle 5 cm out on the target direction, within 12 degrees of the reward-time cursor). They are NaN on center-out trials and on all J trials.
- **Padding is cut.** Four J blocks end in zero-filled padding with no spikes and a frozen cursor (session 0 block `jeev082413e`, 1846.0 s, session 1 blocks `jeev082413h`, 755.7 s, and `jeev082413g`, 544.0 s, session 3 block `jeev083113f`, 266.9 s, 56.9 min in all). No trial falls in it. Every row of those timelines ends where the recorded data end, which `utils.DEAD_TAIL_START` records and `utils.find_dead_tail` recomputes.
- **No target stream.** G's `target_state` is all zeros. J's `target_state` and `obs_details` are written from `go_ix`, a row index into the event-code array rather than a sample index, so their values switch in the first seconds of each block and mix m and cm. Neither is emitted.

`task_condition` is `other` on every `Action`, since no existing value describes a monkey under brain control. `trial_id` is unique within a session.

## Notes / Scope

- **Local files or streaming.** Reads go through `nwb.load_h5`, so the same code serves a local file or a DANDI streaming URL. Nothing is read eagerly: `_load_timeline_events` reads `te_num`, `obstacle_position` and the trials table, and the dense streams are sliced out of the open file only when an extractor asks.
- **Unit counts differ per session** (44, 35, 148, 53, 151, 42, 50, 48, 48 for G, 20 or 21 for J), so a batch spanning sessions cannot be stacked without padding or a per-session readin.
- **Resampling J's velocity to 50 Hz by interpolation aliases it.** The point-process velocity is noisy at 200 Hz, and point samples every 20 ms track the 1 s cursor displacement with correlation 0.65 to 0.94, against 1.0 at the native rate.
- **Not released:** decoder matrices and task parameters (commented out in the writer), so the paper's command `K x_t` cannot be recomputed, and the passive-observation and calibration blocks. The authors hold the remaining raw files.
- **Residual assist in J.** The writer drops the first 55,003 samples of session 1 block `jeev082413h` because assist was on until then. J's source file names all contain `assist_ofc_cont`, and the release cannot confirm that no residual assist remains.
- **Metadata errors in the NWB.** J's `experiment_description` says Kalman filter (the keywords correctly say point-process filter) and J's age is wrong, as above.

## Files

- `metadata.py`: metadata for database insertion.
- `utils.py`: constants and accessors (`list_assets`, `resolve_source`, `blocks`, `trials`, `get_spike_counts`, `get_cursor`, `get_decoder_velocity`, `find_dead_tail`), including the m to cm scaling and the padding cut.
- `study.py`: the neuralset `Study` (`Athalye2023CursorBMI`).
- `todo.md`: shared-file changes, open decisions and data problems for this dataset.
