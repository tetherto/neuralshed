# Perich 2018: monkey center-out and random-target reaching with M1 and PMd arrays

Four macaques (C, J, M, T) move a cursor with the handle of a planar manipulandum, either in a center-out (CO) task with an instructed delay or in a random-target (RT) task of four sequential reaches. The release pools 111 sessions recorded between 2013 and 2016 (43.0 hours, 35,410 trials, 28,866 rewarded) with sorted units from 96-channel Utah arrays in M1 (primary motor cortex) and, in 52 sessions, PMd (dorsal premotor cortex), and the cursor's position, velocity and acceleration at 100 Hz. The release holds unperturbed reaching only: the force-field and visuomotor-rotation epochs that Perich 2018 is about are not in it.

## References

- **Paper:** Perich, Gallego, Miller 2018, "A Neural Population Mechanism for Rapid Learning", *Neuron* 100(4):964-976, [10.1016/j.neuron.2018.09.030](https://doi.org/10.1016/j.neuron.2018.09.030).
- **Paper:** Glaser, Perich, Ramkumar, Miller, Kording 2018, "Population coding of conditional probability distributions in dorsal premotor cortex", *Nat. Commun.* 9:1788, [10.1038/s41467-018-04062-6](https://doi.org/10.1038/s41467-018-04062-6).
- **Cited by the dandiset:** Perich and Miller 2017 (*Exp. Brain Res.*, [10.1007/s00221-017-4997-1](https://doi.org/10.1007/s00221-017-4997-1)), Gallego et al. 2020 (*Nat. Neurosci.*, [10.1038/s41593-019-0555-4](https://doi.org/10.1038/s41593-019-0555-4)) and Azabou et al. 2023 (POYO, [arXiv 2310.16046](https://arxiv.org/abs/2310.16046)), whose pretraining set this release was assembled for.
- **Source data:** DANDI [000688](https://dandiarchive.org/dandiset/000688), "Long-term recordings of motor and premotor cortical spiking activity during reaching in monkeys", CC-BY-4.0. The dandiset has no published version and its citation DOI is a placeholder.

## Species & Subjects

All four are male macaques from the Miller lab (Northwestern University). No source publishes implant dates, handedness or time since implant.

| subject | species | age at first session | arrays (hemisphere) | sessions |
|---|---|---|---|---|
| C | *Macaca mulatta* | P5Y7M (born 2008-03-01) | M1 (right) in 2013 to 2015, M1 and PMd (left) in 2016 | 53 CO, 15 RT |
| J | *Macaca mulatta* | not published (files give P8Y/P15Y) | M1 (left) | 3 CO |
| M | *Macaca mulatta* | P5Y10M (born 2008-03-01) | M1 and PMd (right, inferred) | 22 CO, 6 RT |
| T | *Macaca fascicularis* | P8Y8M (born 2004-12-01) | M1 and PMd (hemisphere not stated) | 6 CO, 6 RT |

- **Arrays:** 96-electrode Blackrock Utah arrays, recorded on a Cerebus system. The files report only "Primary Motor Cortex" and "Dorsal Premotor Cortex", mapped to HOMBA_10163 (area 4) and HOMBA_10169 (laterodorsal area 6). No somatotopic subregion is asserted.
- **Hemispheres:** from Gallego 2020 for C and J. Monkey C had a right M1 array first (96-electrode files, 2013 to 2015), then left M1 and PMd arrays (192-electrode files, 2016). M's hemisphere is inferred from the `ma_2023` release, which says M reached with the left hand and arrays were contralateral to the working arm. The files themselves record no hemisphere or hand.
- **Monkey T:** every T file declares an M1 array with 3 to 12 units per session, while Glaser 2018 and Gallego 2020 describe a PMd array only. `metadata.py` records the arrays as the files declare them.
- **Participant ids** in `metadata.py` (`miller/C` and so on) use the letters the dandiset publishes. C and M are very likely Chewie and Mihili of `ma_2023` (same lab, same task, overlapping session dates).

## Behavioral Task

The monkey grasps the handle of a two-link manipulandum moving in a 20 by 20 cm horizontal workspace, and the cursor follows the handle. The cursor is therefore a measurement of hand position, not a decoder output.

- **CO (center-out):** hold at the center, one of eight 2 cm targets appears on an 8 cm circle, instructed delay, auditory go cue, reach within 1 s and hold 0.5 s for a liquid reward. The delay is about 0.5 to 1.5 s in most sessions, 1 ms (effectively none) in nine Monkey C sessions (December 2013 and March 2015) and a fixed 0.3 s in five October to November 2013 sessions.
- **RT (random target):** four sequential 2 cm targets per trial, each placed at random 5 to 15 cm from the previous one. A new target appears 100 ms after the cursor lands on the previous one, so there is no instructed delay. Reward after the fourth target.
- **Outcomes:** `R` reward, `A` abort, `F` fail, `I` incomplete, as the source codes them.

## Data Modalities

| Modality | NWB location | Type / shape |
|---|---|---|
| Neural | `units` | Sorted units, ragged `spike_times` (s), `electrodes` index, 48-sample waveforms. 18 to 353 units per session |
| Electrodes | `general/extracellular_ephys/electrodes` | 96 rows (M1) or 192 (M1 then PMd), `group_name`, `location`, "High pass 750 Hz" |
| Cursor position | `processing/behavior/Position/cursor_pos` | `(n, 2)` cm, x right and y up, origin at the screen center |
| Cursor velocity | `processing/behavior/Velocity/cursor_vel` | `(n, 2)` cm/s, the numerical derivative of position |
| Cursor acceleration | `processing/behavior/Acceleration/cursor_acc` | `(n, 2)` cm/s^2 |
| Trials | `intervals/trials` | CO: `target_on_time`, `go_cue_time`, `target_dir`, `target_id`, `target_corners`, `result`. RT: `go_cue_time_array (n, 4)`, `target_{1..4}_{x,y}_position`, `num_attempted`, `result` |

The behavior streams share explicit timestamps on a 10 ms grid starting 0 to 0.31 s after `session_start_time`. There is no hand, force or EMG (electromyography) stream and nothing in `acquisition`.

## Source Data Format & Layout

One NWB file per session, 111 in total and about 13 GB: `sub-<X>/sub-<X>_ses-<CO|RT>-<YYYYMMDD>_behavior+ecephys.nwb`. Files range from 10 to 106 minutes, most are 13 to 22 minutes. Locally they sit under `<NEURALSHED_DATA_ROOT>/perich_2018/000688/` (see `todo.md` for the move the Study expects).

## Study, timelines and Events

`study.py` (`Perich2018Reaching`) treats **each NWB file as one timeline**, because a file is one session on one continuous clock. The timeline carries `subject`, `session` (the file stem, also the label in `metadata.sessions()`), `task` (`center_out` or `random_target`) and `date`. The seven half-scale sessions (see Notes) carry `valid=False` and a `valid_description`.

Every dense row sits on the 100 Hz cursor grid, starting at the first cursor timestamp. Trial rows keep the source's times. Per timeline:

| Event type | Count | Source, shape and rate | Key attributes |
|---|---|---|---|
| `SpikeCounts` | 1 | `units` binned by the Study, `(n_units, n_bins)` at 100 Hz | `feature="sorted_spikes"`, `method="sorted"`, `unit="counts/bin"`, `normalization="raw"`, `n_units_m1`, `n_units_pmd` (M1 rows first) |
| `CursorPositionSeries` | 1 | `cursor_pos`, `(2, n_bins)` | `unit="cm"`, `dims=(x, y)`, `reference_frame="screen_center"` |
| `CursorVelocitySeries` | 1 | `cursor_vel`, `(2, n_bins)` | `unit="cm/s"` |
| `AccelerationSeries` | 1 | `cursor_acc`, `(2, n_bins)` | `unit="cm/s^2"` |
| `Stimulus` | per CO trial with a target | target onset to go cue | `stimulus_kind="target_cue"`, `period="delay"`, `modality="visual"` |
| `Stimulus` | per CO go cue | one 10 ms bin | `stimulus_kind="movement_go_cue"`, `period="go"`, `modality="audio"` |
| `Stimulus` | per RT target | one 10 ms bin at the target's go cue | `stimulus_kind="target_cue"`, `period="go"` |
| `Action` | per CO trial with a go cue, per RT reach | go cue to trial end (CO), go cue to the next go cue (RT) | `action_kind="cursor_control"`, `task_condition="executed"`, `success`, `result`, `reach_index`, `delay` (CO) |
| `Intertrial` | gaps | between trials, center holds, trials ending before any target | `period="intertrial"`, `description` |

Trial rows carry `trial_id` (unique within a timeline), `result`, `target_x` and `target_y` (cm, screen frame) and `code`. `code` is the CO target index in the source's convention (0 up, clockwise by 45 degrees), derived from `target_dir`, and -100 for RT reaches, whose targets have no class. An RT reach's `success` is true when the monkey moved on to the next target or the trial was rewarded. The `Intertrial`, delay `Stimulus` and `Action` rows tile each timeline without gaps or overlap. Failed and aborted trials are kept with their `result` and `success=False`.

The cursor stream is how the hand moved, so its velocity is a `CursorVelocitySeries` and not a `DecodedCursorVelocitySeries`.

## Notes / Scope

- **Half-scale cursor.** In the seven Monkey C CO sessions from 2016-09-09 to 2016-09-23, `session_description` ends in a truncated "Warning: This session with Monkey" and rewarded reaches end at a median radius of about 4.2 cm against 8 cm targets (position and velocity both at about 0.53). The Study marks them `valid=False` and leaves the values as released. `study.py` raises if the warning text and `utils.HALF_SCALE_SESSIONS` ever disagree.
- **Target positions.** CO targets come from `target_dir` at the 8 cm radius, not from `target_corners`, which is shifted by one row in eight sessions (C 2013-12-03, 2013-12-04, 2013-12-19, M 2014-02-03, 2014-02-18, 2014-06-26, 2014-06-27, T 2013-08-19). A few trials have a `target_id` of 8 or 12 with no `target_dir`, and get `code=-100`. RT coordinates that are NaN, float garbage (about 1e37) or exactly (0, 0) become NaN. The (0, 0) targets are placeholders: successful reaches to them end a median 5.5 cm away, against 0.5 cm for the others.
- **Trials with nothing logged.** 15 to 96 rewarded RT trials per session have no targets and no go cues. Each becomes one `Action` over the whole trial with `reach_index=-1` and NaN targets. 24 CO trials are rewarded with no target, 22 of them shorter than 0.1 s, and become `Intertrial` rows.
- **Binning.** Units are binned into `[t0 + i * 0.01, t0 + (i + 1) * 0.01)`. Spikes outside the cursor span (up to about 2 s at the end of some sessions) are dropped. In two sessions (C 2016-09-23 and 2016-10-11) cursor timestamps swap a 0 ms step for a 20 ms one, but no sample is more than one bin off the uniform grid.
- **Sorted or threshold crossings.** The dandiset says one of the four monkeys is threshold crossings without naming it. Gallego 2020 says C, M and T were sorted, which points to J, but J's files have up to two units per electrode on a subset of electrodes, like the sorted monkeys. The Study declares `method="sorted"` for all four.
- **Reading.** Everything is read with `h5py` by key through `datasets/nwb.py`, so a local file and a streamed DANDI asset take the same path. Streaming needs the `dandi` package, which the current environment lacks.
- **Out of scope.** Spike waveforms, the `Spikes` (spike-time) event, a synthesized `TargetPositionSeries`, and the adaptation epochs, which are not released.

## Files

- `metadata.py`: catalog metadata and `sessions()` for the database.
- `utils.py`: constants and `h5py` accessors, spike binning, trial tables and target geometry.
- `study.py`: the neuralset `Study` (`Perich2018Reaching`).
- `todo.md`: open decisions and shared-file changes.
