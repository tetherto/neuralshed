# Even-Chen 2019: instructed-delay center-out-and-back reaching in two macaques

Two rhesus macaques, **JenkinsC** and **Reggie**, reach with the arm to move a cursor from the center of a screen to one of 24 to 48 targets and back, with an instructed delay between target appearance and the go cue. Each monkey has a 96-channel Utah array in **M1** and one in **PMd** of the left hemisphere. The paper studies the structure of PMd delay activity across many reach endpoints. Earlier center-out studies varied only the reach direction, so the target layouts here vary both direction and distance, which shows how the delay-period neural states are organized by the x, y location of the reach endpoint and by maximum speed. The release is almost entirely hand control, but one session contains an undocumented **brain-control block** where a decoder drives the cursor.

## References

- **Paper:** Even-Chen, Sheffer, Vyas, Ryu and Shenoy 2019, "Structure and variability of delay activity in premotor cortex," *PLoS Computational Biology* 15(2):e1006808, [10.1371/journal.pcbi.1006808](https://doi.org/10.1371/journal.pcbi.1006808).
- **Source data:** DANDI [000121](https://dandiarchive.org/dandiset/000121/0.220124.2156), version 0.220124.2156, 12 NWB files, about 38 GB, CC-BY-4.0. A 2021 conversion of the lab's own release, which the paper's data statement points to.
- **Code repository:** none published.

## Species & Subjects

- **Species:** *Macaca mulatta*, both male.
- **JenkinsC:** arrays implanted 74 months before the study (`time_since_implant="P74M"`). Probably the monkey called Jenkins in `churchland_2012` (same lab, matching implant age), which the files do not state.
- **Reggie:** arrays implanted 64 months before the study (`"P64M"`). The paper calls his arrays worse than Jenkins's and his M1 array too poor to use, and analyzes only PMd for both monkeys.
- **Implant:** two 96-channel Blackrock Utah arrays in the left hemisphere, one in M1 and one in caudal PMd, placed by visual landmarks. No stereotaxic coordinates, impedances or electrode positions are published (the electrodes table holds NaN and -1).
- **Channel order:** 0 to 95 are M1 and 96 to 191 are PMd. The electrodes table, the LFP series and the units' `electrodes` links agree on this, while the units table's own `electrode_group` column says the opposite. The electrodes table is followed because Reggie has 29 silent channels in 0 to 95, which would leave 67 live PMd channels, fewer than the 80 tuned PMd channels the paper reports.
- **Age, implant date, handedness:** not published. The contralateral (right) arm reached.
- **Lab:** Shenoy lab, Stanford.

## Behavioral Task

Each outward trial starts with the cursor held at center. About 20 ms after the trial starts the target appears, and after an instructed delay the go cue tells the monkey to reach. The trial succeeds when the cursor stays inside the acceptance box for 500 ms. The paper gives a 2 cm box, but the cursor in the successful trials stays within about 15 mm of the target on each axis for JenkinsC, 17 mm for Reggie on 2017-01-15 and 01-16, and 20 mm for Reggie from 2017-01-17, and reaches those limits (measured over the last 0.5 s before `target_held_time`), so the box is about 3 cm wide for JenkinsC and 3.4 to 4 cm for Reggie. These are measured cursor limits, a lower bound on the window edges, which the files do not give. `target_size` is 8 in every trial and is not the window. A **return trial** to center follows every outward trial, with no delay. Delays run 300 to 700 ms for JenkinsC, with zero delay in 19 to 22% of outward trials (about 9 to 11% of all trials), and 400 to 1000 ms for Reggie.

Four target layouts, one per session, derived from `target_pos` because `task_id` (always 2) and `version_id` (21 for JenkinsC, 23 for Reggie) never change:

| session | subject | layout (from the file) | outward targets | trials |
|---|---|---|---|---|
| 2015-10-01 | JenkinsC | vertical line, 10 mm steps | 24 | 3426 |
| 2015-10-02 | JenkinsC | vertical line | 24 | 4612 |
| 2015-10-15 | JenkinsC | ring at 80 mm | 24 | 2654 |
| 2016-01-27 | JenkinsC | three rings at 40, 80, 120 mm | 48 | 4667 |
| 2016-01-28 | JenkinsC | three rings | 48 | 7020 |
| 2017-01-15 | Reggie | horizontal line, about 10.4 mm steps | 24 | 5333 |
| 2017-01-16 | Reggie | horizontal line | 24 | 5040 |
| 2017-01-17 | Reggie | vertical line, plus six 80 mm targets early on | 30 | 5398 |
| 2017-01-18 | Reggie | vertical line | 24 | 5188 |
| 2017-01-19 | Reggie | three rings | 48 | 3701 |
| 2017-01-24 | Reggie | ring at 80 mm | 36 | 4303 |
| 2017-01-25 | Reggie | ring at 80 mm | 36 | 5274 |

The paper labels Reggie's 2017-01-15 and 01-16 sessions "vertical" and 01-17 and 01-18 "horizontal", the opposite of the target positions in the files. The Study names layouts by the files. JenkinsC's horizontal-line sessions are not in the release.

**Brain-control block.** In Reggie 2017-01-19, trials 974 to 1107 (134 trials, about 6.5 minutes) are decoder-driven: `DecodePos` is filled on every sample, `Cursor` equals `DecodePos` x, y minus (70, 200) exactly, and `Hand` is in its tracking-loss state most of the time. Nothing in the trial table flags it, apart from reach limits of 4 or 8 s instead of 2 s. `DecodePos` is NaN in every other trial of every session. The paper does not describe this block, so the decoder and whether the arm was free are unknown. The nine trials after it (1108 to 1116) have the cursor far off the screen, near (-1832, 1263) mm.

**Trials whose target never appeared.** Between 12 and 722 trials per session have no `target_shown_time` and no `go_cue_time`. They are all outward trials and all failures, lasting 0.3 to 1.2 s for Reggie and about 2 s for JenkinsC.

## Data Modalities

| Modality | Source location | Type / shape |
|---|---|---|
| Threshold crossings | `units` (`spike_times`, ragged) | 192 rows, one per electrode, spike times in s |
| Hand position | `processing/behavior/Position/Hand` | `float64 (n, 3)`, x, y, z in mm |
| Cursor position | `processing/behavior/Position/Cursor` | `float64 (n, 2)`, mm |
| Eye position | `processing/behavior/Position/Eye` | `float64 (n, 2)`, raw units 0 to about 5700 |
| Decoded position | `processing/behavior/Position/DecodePos` | `float64 (n, 3)`, NaN outside the brain-control block |
| LFP | `processing/ecephys/LFP/{M1,PMd}_{1..4}` | `int16 (n, 96)` at 2 kHz, `conversion=1e-6` |
| Reward | `stimulus/presentation/juice_reward` | `uint64 (n,)`, 0 to 400 |
| Trial table | `intervals/trials` | times, `is_successful`, `target_pos` (3 per trial), limits, ragged `target_acquire_time` |

- **Threshold crossings, not sorted units.** The dandiset description says "sorted spikes", but there is exactly one unit per electrode. Thresholds are -4.5 x RMS (JenkinsC) and -4.0 x RMS (Reggie). Reggie has the same 30 silent channels in every session (29 on M1, 1 on PMd). JenkinsC has none.
- **Behavior is 60 Hz on a 1 kHz grid.** All four position series share one timestamp vector at 1 ms steps, but `Hand` and `Cursor` change every 16 to 17 ms (the 60 frames/s tracker) and `Eye` every 8 to 9 ms. Under hand control `Cursor` is `Hand` x, y one sample later.
- **Positions are millimeters** from screen center, although the files say meters with `conversion=NaN`. Targets sit at 40, 80 and 120 mm, matching the paper's 4, 8 and 12 cm.
- **Tracking loss is a sentinel row.** `Hand` jumps to (70, 1278.5, -219) or (70, 1258.5, -219) for JenkinsC and (70, 1258.5, -149) for Reggie, and `Cursor` copies its x, y. It covers 0.02% to 12% of samples per session. `Eye` dropouts are all-zero rows.

## Source Data Format & Layout

```
even_chen_2019/data/EvenChen2019DelayedReach/000121/
├── sub-JenkinsC/sub-JenkinsC_ses-<YYYYMMDDTHHMMSS>_behavior+ecephys.nwb   5 files, 2.2 to 5.0 GB
└── sub-Reggie/sub-Reggie_ses-<YYYYMMDDTHHMMSS>_behavior+ecephys.nwb       7 files, 2.7 to 3.3 GB
```

NWB 2.2.5, uncompressed and unchunked, read with `h5py`. Most of each file is LFP, which the Study never reads. The session clock starts at the first trial (0 s), and **no data exist between trials**: behavior samples and spike times lie only inside `[start_time, stop_time]` of each trial. The gaps are always 220 ms for JenkinsC and 40 ms for Reggie, with pauses of up to 11 minutes. `reach_time`, `target_hold_time` and `fail_time` are limits with `start_time` added (2 s, 0.5 s, and 1.5 s or 0.02 s), not events. `is_successful` is 1 exactly when `target_held_time` is present, despite its description.

## Study, timelines and Events

`EvenChen2019DelayedReach` treats **one NWB session as one timeline** (12 timelines). Trials, spikes and behavior all sit on the session clock, so the session is the largest continuous span. Timeline labels: `subject`, `session` (the file stem, as in `metadata.sessions()`), `date`, `task="center_out_and_back"`, `target_layout` (`vertical_line`, `horizontal_line`, `ring`, `three_rings`) and `n_targets`.

All dense streams share one **50 Hz grid** (20 ms bins starting at 0). The spike times are binned into it, and the 1 kHz behavior is averaged into the same bins after the tracking-loss and dropout samples are set to NaN. A bin with no valid sample is NaN, which includes every bin inside a gap between trials. Spike counts in those gaps are zero because the source holds nothing there, not because the channels were silent.

| event type | feature | source | shape, rate | notes |
|---|---|---|---|---|
| `SpikeCounts` | `threshold_crossings` | `units` | `(192, n_bins)`, 50 Hz | `counts/bin`, raw, M1 then PMd |
| `PositionSeries` | `hand_position` | `Hand` | `(3, n_bins)`, 50 Hz | mm, x, y, z |
| `CursorPositionSeries` | `cursor_position` | `Cursor` | `(2, n_bins)`, 50 Hz | mm, decoder output in the brain-control block |
| `PositionSeries` | `eye_position` | `Eye` | `(2, n_bins)`, 50 Hz | `a.u.`, uncalibrated |
| `TargetPositionSeries` | `target_position` | trial table | `(2, n_bins)`, 50 Hz | mm, from target appearance to trial end, NaN elsewhere |
| `Stimulus` | | trial table | target appearance to trial end | `stimulus_kind="target_cue"`, `modality="visual"` |
| `Action` | | trial table | go cue to trial end | `action_kind="cursor_control"`, whole trial when the target never appeared |
| `Intertrial` | | trial table | gap between trials | no data released there |

Attributes on every `Stimulus` and `Action` row:

- `trial_id`, the trial table's `id`, unique within a session.
- `code`, a **session-local** index into the session's distinct targets (center included), sorted by x then y. `target_x` and `target_y` in mm carry the geometry across sessions.
- `trial_kind`, `center_out` or `return`.
- `delay_duration`, go cue minus target appearance in s, NaN when the target never appeared.
- `success`, the source's `is_successful`.
- `loop_type`, `hand_control` for arm-driven trials and `closed` for the brain-control block. It rides on trial rows only, because one session mixes both, so select with a trigger query or an event filter that keeps the dense rows.
- `task_condition`, `executed` under hand control and `other` under brain control, where the arm's state is unknown.
- `valid=False` with `valid_description` on the nine off-screen cursor trials of 2017-01-19.

The `StudyInfo` query (`timeline_index in [2, 9]`) covers JenkinsC 2015-10-15 and Reggie 2017-01-19, which together exercise every emission path.

## Notes / Scope

- **Data folder.** The Study expects the dandiset at `even_chen_2019/data/EvenChen2019DelayedReach/000121/`. When no local file is found it streams from DANDI, which is slow for files this size.
- **Not emitted:** LFP (segments with NaN start times that the descriptions say cannot be synced), `DecodePos` (its x, y are the cursor plus a fixed offset in the one block where it exists), and `juice_reward` (described as a 0/1 flag but holding values up to 400, all zeros in three JenkinsC sessions).
- **Velocity.** No velocity stream is published. Differencing the 50 Hz positions is reasonable, differencing the source's 1 kHz grid is not (mostly zeros with spikes every 16 ms).
- **Reggie's arrays.** The 30 silent channels are emitted as zeros, and the paper's judgment that his M1 array is unusable is recorded here, not as a `valid` flag.
- **Trial counts** exceed the paper's because the release keeps return trials, failures and no-delay trials, which the paper dropped. Only 5 of JenkinsC's 16 paper sessions are released.

## Files

- `__init__.py`: package marker.
- `utils.py`: h5py readers for the trial table, the brain-control flag, the off-screen cursor check, target layout, and the binned neural, behavioral and target streams.
- `metadata.py`: `DATASET_METADATA`, `sessions()` and `get_metadata()`.
- `study.py`: the `EvenChen2019DelayedReach` Study.
- `todo.md`: open decisions and shared-file changes for this dataset.
