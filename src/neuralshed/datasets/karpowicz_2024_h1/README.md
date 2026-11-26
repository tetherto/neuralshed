# Karpowicz et al. 2024 (FALCON H1): Human 7-DoF Reach and Grasp Motor BCI

Intracortical BCI benchmark from a single human participant (Univ. of Pittsburgh) implanted with two 96-channel Utah arrays in hand-knob motor cortex. During **open-loop 7-DoF reach and grasp** (3 translation, 1 rotation, 3 grasp), 40 NWB files (27 recordings and 13 copies of their first minutes) record threshold-crossing **spike times** on 176 channels alongside 20 ms / 50 Hz **7-DoF kinematics** (position and velocity) plus trial and eval-mask structure. The goal is to decode the reach-and-grasp kinematics and benchmark **cross-day decoder stability** across held-in and held-out days under few-shot recalibration.

## References

- **Paper:** *Karpowicz et al. 2024, "Few-shot Algorithms for Consistent Neural Decoding (FALCON) Benchmark," NeurIPS 2024 Datasets & Benchmarks* - [10.52202/079017-2439](https://proceedings.neurips.cc/paper_files/paper/2024/hash/8c2e6bb15be1894b8fb4e0f9bcad1739-Abstract-Datasets_and_Benchmarks_Track.html). Preprint: [10.1101/2024.09.15.613126](https://www.biorxiv.org/content/10.1101/2024.09.15.613126v1).
- **Source data:** https://dandiarchive.org/dandiset/000954, a draft with no published version or DOI, so its files can change. CC-BY-4.0 with a Data Use Agreement, see [Terms of use](#terms-of-use).
- **Code repository:** https://github.com/snel-repo/falcon-challenge

## Terms of use

DANDI 000954 is released under CC-BY-4.0 together with a Data Use Agreement, which downloading the data accepts. Its four terms:

- "I will receive access to de-identified data and will not attempt to establish the identity of or attempt to contact any of the subjects."
- "I will not attempt to make direct contact with PIs, affiliated data partner PIs, or staff at sites concerning the specific results of individual subjects. Data partner PIs may be contacted about potential collaboration opportunities."
- "I will not further disclose these data beyond the uses outlined in this agreement and my data use application and understand that redistribution of data in any manner is prohibited."
- "I will require anyone on my team who utilizes these data or anyone with whom I share these data to comply with this Data Use Agreement."

The agreement forbids redistribution, which CC-BY-4.0 alone would allow. Read it as covering files derived from the data too, such as exported spike counts or kinematics. Because of its first term, the catalog keeps `HumanPitt` as its own participant and does not link it to a participant of a published study.

## Species & Subjects

- **Species:** *Homo sapiens* (human).
- **Subject:** single iBCI participant, asset subject `sub-HumanPitt` (`subject_id = HumanPitt-...`). Details redacted by the data owners, with `sex: U` and `age: P22Y/P70Y` (range).
- **Implant:** two 96-channel Utah arrays (192 electrodes, 176 released as channels) in **hand-knob motor cortex** (precentral gyrus).
- **Lab / institution:** Rehab Neural Engineering Labs, University of Pittsburgh.

## Behavioral Task

Open-loop **7-DoF reach and grasp**. The 7 degrees of freedom are 3 translation (`tx, ty, tz`), 1 rotation (`rx`, wrist roll), and 3 grasp dimensions (`g1, g2, g3`), see the `OpenLoopKinematics.description` field. FALCON (appendix A.3.5 and the H1 demo) names the grasp dimensions, in order, pinch (flexion and extension of thumb, index and ring finger), scoop (flexion and extension of ring and pinky finger) and thumb abduction.

**Open-loop calibration.** A virtual arm performs each movement while the participant attempts it with their right hand, with their native limbs at rest (FALCON H1 demo). The arm is simulated in the Virtual Integration Environment of the Johns Hopkins University Applied Physics Laboratory, and a proportional autocontroller moves it to position, orientation and hand-posture targets picked in random order from a fixed set. The participant does not control the arm and cannot correct it (FALCON). The kinematics are therefore the arm's programmed trajectory, not decoded output, and every movement is attempted (the Study's `task_condition="attempted"` and `loop_type="open"`). The demo notes that the simulator's physics add some jitter to the reported kinematics.

Each trial is a reach, grasp and transport sequence of movement phases (Reach → Orient → Shape → Grasp → Carry → Orient2 → Release), most of them preceded by a cue period in which the arm holds still while the next target is shown. These phases are exactly the `intervals/epochs` rows, see [Task phases & events](#task-phases--events).

FALCON framing: the dataset is split into **held-in** and **held-out** days to test cross-day decoding consistency. A decoder is trained on held-in calibration data, then **few-shot recalibrated** on a small held-out calibration slice and evaluated on held-out data. Only bins flagged by the per-bin `eval_mask` are scored.

## Task phases & events

Each `intervals/epochs` **row is one task *phase*, not a whole trial**, it carries exactly **one** `tags` entry (verified across all 40 sessions / 3 436 rows). The phases tile each trial contiguously. A typical trial reads:

```
Presentation → Reach → [Presentation2] → Orient → SnapTo → Presentation3 →
Shape → Presentation4 → Grasp → Presentation5 → Carry → [Presentation6] →
Orient2 → Presentation7 → Release
```

There are **16 distinct tags**. The Study (`study.py`) splits them four ways:

| Group | Tags | Event type | Notes |
|---|---|---|---|
| **Cues** | `Presentation`, `Presentation2`…`Presentation7` | `Stimulus` | Target-presentation periods. The arm finishes the previous movement in the first 0.3 s, then translation and roll hold still while the next target is shown, and grasp only jitters. The number is **positional** (which movement it cues), *not* a different stimulus kind. `Presentation2` (pre-`Orient`) and `Presentation6` (pre-`Orient2`) are **optional**, so they appear in fewer trials. |
| **Movements** | `Reach`, `Orient`, `Shape`, `Grasp`, `Carry`, `Orient2`, `Release` | `Action` | Attempted-movement phases, one `code` each. `Orient` turns the hand before the grasp, `Orient2` turns the held object after `Carry`. `Shape` = hand pre-shaping. |
| **Automatic** | `SnapTo` | `AutoMovement` | An uncued move the task software makes on its own, see [SnapTo](#snapto). |
| **Rest** | `Intertrial` | `Intertrial` | The rest before the first trial, a custom event type (a pure time-span marker, no code). Trials run back to back, so there is no rest between them. The source `Intertrial` tag that closes each recording is in reality a `Release`, see [The last Release](#the-last-release). |

**Code scheme.** `Action.code` is one index per movement phase in task order:

| code | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| Action | Reach | Orient | Shape | Grasp | Carry | Orient2 | Release |

Each `Presentation*` cue is emitted as a `Stimulus` whose `code` **equals the `code` of the `Action` it precedes** (so a cue and its target movement share a code), with `description="cue: <Action>"`. The cue→action mapping is `Presentation→Reach`, `Presentation2→Orient`, `Presentation3→Shape`, `Presentation4→Grasp`, `Presentation5→Carry`, `Presentation6→Orient2`, `Presentation7→Release`.

### SnapTo

`SnapTo` is not described by DANDI, the FALCON paper and demo, or the Pittsburgh papers. In the data it behaves as an automatic correction by the task software:

- Every trial has exactly one, always between `Orient` and `Presentation3` (the `Shape` cue), with no cue of its own. It never occurs before `Release`.
- It lasts 0.12-1.04 s (median 0.39 s) and continues the trial's motion: its translation points along the `Reach` (median cosine 1.00) and its roll turns the same way as the `Orient`, covering about 4% of the `Reach` distance and 9% of the `Orient` roll. Grasp barely changes.
- It moves at a near-constant speed and stops abruptly, where the cued phases slow down toward their target as the demo's "proportional autocontroller" would.
- FALCON's `eval_mask` leaves `SnapTo` out. Epoch times fall halfway between samples, so the mask can turn True one sample before a `SnapTo` row ends.

This is consistent with the software moving the hand from where the autocontroller stopped, within tolerance, onto the object's exact grasp pose. The Study emits it as an `AutoMovement` (`description="SnapTo"`, with `trial_id`, no `period`), so it is not selected with the `Action` rows. The timeline's `task_condition="attempted"` still appears on it, describing the block.

**Cue modality is left unset** (`Stimulus.modality=None`) because the sources describe it differently. DANDI says "audiovisual cues indicated to the participant to attempt a particular reach or grasp". FALCON (appendix A.3.5 and the H1 demo) says "each phase begins with a presentation of a combined visual and word cue for a particular movement", and the demo adds "translation and rotation were cued by a virtual object on a screen, while grasping was cued by an audio command". None says what each `Presentation*` phase showed, and `modality` holds a single value (`audio`, `visual` or `tactile`). To follow the demo's split, use `visual` for the cues of `arm_movement` and `wrist_movement` actions and `audio` for those of `hand_movement`, matching cues to actions by `code`.

### The last Release

The source tags the last phase of every recording `Intertrial`, and its content is the last trial's `Release`:

- It always follows `Presentation7`, the `Release` cue, and the last trial of every recording has no `Release` row otherwise.
- The hand opens during it (pinch and scoop at 4.9x and 5.1x their mean speed, against 5.7x and 6.4x in full `Release` rows), and FALCON's `eval_mask` scores all of it, as it does `Release`.
- It lasts 0.74-0.80 s and ends at the file end. A full `Release` lasts 0.80-1.00 s and `TrialNum` steps to the next trial 0.74-0.86 s into it, so each file ends where the next trial would begin and the last `Release` is cut short.
- The 13 held-in-minival files show the same cut. Each one copies the start of a held-in-calib file and ends with an `Intertrial` row, which the calib file tags `Release`, with the same start time, and runs on past the cut.

The Study emits it as an `Action` `Release` (`code=6`, `action_kind="hand_movement"`, `period="go"`) with the file's tag kept in `source_tag="Intertrial"`. Only the rest before the first trial stays an `Intertrial`.

### Phases without movement

191 of the 1,270 movement rows last one or two bins (at most 0.04 s), and in each of them the arm is already at that phase's target:

| Phase | Rows ≤ 0.04 s | Change of its DoF in them | Median change in the other rows | Cue before it |
|---|---|---|---|---|
| `Reach` | 14 of 212 | 0 | 0.26 | always, about 1 s |
| `Carry` | 25 of 211 | at most 0.011 | 0.27 | always, about 1 s |
| `Orient` | 81 of 211 | at most 0.005 | 0.51 | never |
| `Orient2` | 71 of 212 | at most 0.008 | 0.51 | never |

The DoF is translation (distance in x, y, z) for `Reach` and `Carry` and roll for `Orient` and `Orient2`. `Shape`, `Grasp` and `Release` are never this short, and velocity stays near zero in these rows.

`Presentation2` and `Presentation6`, the two optional cues, appear exactly when a rotation is needed: all 130 longer `Orient` rows and 141 longer `Orient2` rows follow their cue, and none of the short ones do. The short rows keep the roll the arm already had (mostly about 0 or ±0.74), where the longer ones end near ±0.25 or ±0.5. Short `Reach` and `Carry` rows follow a normal cue whose target is where the hand already is, and 2 of the 14 `Reach` rows open a recording. `eval_mask` scores these rows, except 12 of the 95 bins in short `Orient` rows.

The Study keeps them as `Action` rows, so every trial has its full code sequence. To keep only phases with movement, select `Action` rows with `duration > 0.04`.

## Data Modalities

| Modality | NWB location | Type / shape |
|---|---|---|
| Neural | `nwb.units` (Units table) | Spike times of 176 channels (one table row each), ragged `spike_times`, not pre-binned. |
| Kinematics (position) | `acquisition/OpenLoopKinematics` | `TimeSeries (n_bins, 7)`, 7 DoF |
| Kinematics (velocity) | `acquisition/OpenLoopKinematicsVelocity` | `TimeSeries (n_bins, 7)` |
| Trial id per bin | `acquisition/TrialNum` | `TimeSeries (n_bins,)` |
| FALCON eval mask | `acquisition/eval_mask` | bool `TimeSeries (n_bins,)` |
| Task phases / intervals | `intervals/epochs` | one row **per task phase** (single `tags` entry + `start_time`, `stop_time`), ~29-31 phase-rows in minival files, up to ~218 in held-in-calib. See [Task phases & events](#task-phases--events). |

- **Neural signal type:** unsorted threshold crossings at -4.5 x RMS, detected per electrode (FALCON H1 demo), released as spike times. The files store them in an NWB `Units` table, which is meant for sorted units. Each of its 176 rows is one channel, the table has no `electrodes` column and the files have no electrode table, so the electrode and array behind each channel are not released. The Study reads the rows as channels in table order. 176 channels against 192 electrodes, while FALCON's Figure 2 gives 172. No source says which electrodes were left out. Channel 66 has no crossing in any file.
- **Bin size / rate:** kinematics and masks are sampled at **20 ms bins → 50 Hz**, sample `k` at `k·0.02` s. Spike times start at -0.02 s because sample `k` goes with the spikes in the 20 ms ending at it, `[k·0.02 - 0.02, k·0.02)`. FALCON bins them this way (`bin_units` in `falcon_challenge.dataloaders`), and so do the Study and `utils.bin_spikes`. ⚠️ The `TimeSeries.rate` field stores `0.02`, which is the bin **period in seconds**, *not* a rate in Hz. Use the 20 ms / 50 Hz constants in `utils.py` (`BIN_SIZE_S`), not the raw `.rate`.
- **eval_mask:** True on the bins FALCON trains on and scores, 96.2-98.3% of bins per recording. False covers every `SnapTo` bin, the leading `Intertrial` rows and a few `Orient` bins. It cannot be rebuilt exactly from the phases, since a few `Orient` bins are unscored and some leading `Intertrial` rows are partly scored. The NWB description, "Timesteps to ignore covariates", states the opposite polarity, while FALCON's loader treats True as scored. The DANDI description also lists a `Blacklist` acquisition, which no released file has. FALCON's loader reads `Blacklist` only for an older file format.
- **Kinematics units:** both kinematics series declare `unit="arbitrary"`, and the Study emits `a.u.` for translation and roll. The values look physical: at the end of `Reach` and `Carry`, `tx` clusters at 0.35-0.40 and 0.49-0.52, and roll stays between -0.97 and 0.85, about -55° to 49° if read as radians. No source states the units, so they stay `a.u.`. Grasp is "an arbitrary scaling between a neutral pose and a predefined grasp pose" from 0 to 1 (FALCON demo), emitted as `unit="proportion"` with `reference_frame="range_of_motion"`.
- **Velocity:** `OpenLoopKinematicsVelocity` is a smoothed temporal derivative of position per 20 ms bin, not per second, and is FALCON's decoding target. It is close to the demo's `create_targets` (490 ms Gaussian smoothing, then `np.gradient`) but not equal, with a maximum deviation of 25-36% of peak, and the exact kernel is unpublished. The Study keeps it per bin as shipped, with `unit="a.u."` and the recipe in `processing`.

## Source Data Format & Layout

**DANDI asset naming** (40 assets total):

```
sub-HumanPitt-held-in-calib/   sub-HumanPitt-held-in-calib_ses-<YYYYMMDDThhmmss>.nwb   (13 files, ~3–7 MB)
sub-HumanPitt-held-in-minival/ sub-HumanPitt-held-in-minival_ses-<...>.nwb             (13 files, ~0.9–1.1 MB)
sub-HumanPitt-held-out-calib/  sub-HumanPitt-held-out-calib_ses-<...>.nwb              (14 files, ~1.2–1.5 MB)
```

The FALCON **split** is encoded in the `sub-*` folder / path: `held-in-calib` (training), `held-in-minival`, `held-out-calib` (few-shot recalibration). Session timestamps are anonymized to 1925, keeping the spacing between days.

The 27 calib files are 27 recordings from 13 days, 2 or 3 recordings a day a few minutes apart: 6 held-in days (1925-01-01 to 01-20) and 7 held-out days (01-26 to 02-09). FALCON kept the last 20% of every recording for its hidden evaluation. `held-in-calib` holds the rest of each held-in recording and `held-out-calib` the first 20% of each held-out recording. Each `held-in-minival` file is an exact copy of the first 1,503-1,831 bins of the `held-in-calib` file with the same timestamp (spikes, kinematics, `eval_mask`, `TrialNum` and epochs), which FALCON provides to check that a submission runs. It is not separate data, and a decoder trained on `held-in-calib` and scored on `held-in-minival` is scored on its own training data.

**NWB internal layout** (identical across splits, only `n_bins` differs):

```
NWBFile
├─ session_description : "FALCON H1. Open loop 7DoF calibration."
├─ subject             : sub-HumanPitt (Homo sapiens)
├─ acquisition/
│   ├─ OpenLoopKinematics          TimeSeries (n_bins, 7)   # 7-DoF position
│   ├─ OpenLoopKinematicsVelocity  TimeSeries (n_bins, 7)   # 7-DoF velocity
│   ├─ TrialNum                    TimeSeries (n_bins,)     # per-bin trial id
│   └─ eval_mask                   TimeSeries (n_bins,) bool# FALCON scored bins
├─ units (Units table)            176 rows = channels, spike_times (ragged)
└─ intervals/epochs               ~29–218 rows, one per task phase
                                   (start_time/stop_time + single `tags` label)
```

`n_bins` ranges ~1500-1830 (held-in-minival), ~2300-2710 (held-out-calib), and ~6500-12,300 (held-in-calib). Electrodes / electrode_groups / devices tables are empty in these files, array identity is documented only in metadata text.

## Study, timelines and Events

`study.py` (`Karpowicz2024FalconH1`) treats **each recording as one timeline**, 27 from the `held-in-calib` and `held-out-calib` files. The `held-in-minival` copies are left out. Each timeline carries `subject="HumanPitt"`, `session`, `block`, `task="reach_grasp"`, `task_condition="attempted"`, `loop_type="open"`, and its FALCON `original_split` (`held-in-calib` / `held-out-calib`) broadcast to every event row. `block` numbers the recordings of a day by start time from 1, and `session` is `<day>_block<block>` (e.g. `19250101_block1`), both from `utils.recordings`. FALCON's code and evaluator name the same recordings `S<day>_set_<block>`, with the 13 days numbered 0-12 in date order (`H1_NEW_TO_OLD` in `falcon_challenge/config.py`), so `19250113_block1` is FALCON's `S2_set_1`, and the held-in days are S0-S5 and the held-out days S6-S12. The relative asset `path` is the local, or streamed, source key.

Per timeline it emits:

| Event type | Count | Source → shape / rate | Key attributes |
|---|---|---|---|
| `SpikeCounts` | 1 | `units` table rows → `(176, n_bins)` channel counts, bin `k` ending at kinematic sample `k` | `frequency=50`, `start=0`, `unit="counts/bin"`, `feature="threshold_crossings"`, `threshold="-4.5 x RMS"` |
| `BinaryMask` | 1 | `acquisition/eval_mask` → `(1, n_bins)` bool | `feature="falcon_eval_mask"`, `true_means="scored"`, `unit="binary"`, `frequency=50` |
| `PositionSeries` | 1 | `OpenLoopKinematics[:, 0:3]` → `(3, n_bins)` | `feature="hand_position"`, `dims=(x,y,z)`, `unit="a.u."` |
| `OrientationSeries` | 1 | `OpenLoopKinematics[:, 3]` → `(1, n_bins)` | `feature="hand_orientation"`, `dims=(roll,)`, `unit="a.u."` |
| `GraspSeries` | 1 | `OpenLoopKinematics[:, 4:7]` → `(3, n_bins)` | `feature="grasp"`, `dims=(pinch,scoop,thumb_abduction)`, `unit="proportion"`, `reference_frame="range_of_motion"` |
| `LinearVelocitySeries` | 1 | `OpenLoopKinematicsVelocity[:, 0:3]` → `(3, n_bins)` | `feature="hand_linear_velocity"`, `dims=(x,y,z)`, `unit="a.u."`, per bin |
| `AngularVelocitySeries` | 1 | `OpenLoopKinematicsVelocity[:, 3]` → `(1, n_bins)` | `feature="hand_angular_velocity"`, `dims=(roll,)`, `unit="a.u."`, per bin |
| `GraspVelocitySeries` | 1 | `OpenLoopKinematicsVelocity[:, 4:7]` → `(3, n_bins)` | `feature="grasp_velocity"`, `dims=(pinch,scoop,thumb_abduction)`, `unit="a.u."`, `reference_frame="range_of_motion"`, per bin |
| `Stimulus` | per `Presentation*` row | one `intervals/epochs` cue phase | `code` = cued action's code, `description="cue: <Action>"`, `cue_tag`, `trial_id`, `period="delay"`, `stimulus_kind="movement_cue"` |
| `Action` | per movement row | one `intervals/epochs` movement phase | `code` (0-6), `description=<tag>`, `trial_id`, `period="go"`, `action_kind` (`arm_movement` for Reach and Carry, `wrist_movement` for Orient and Orient2, `hand_movement` for Shape, Grasp and Release), `source_tag="Intertrial"` on each recording's last `Release` |
| `AutoMovement` | per `SnapTo` row | one `intervals/epochs` `SnapTo` phase | `description="SnapTo"`, `trial_id` |
| `Intertrial` | per leading `Intertrial` row | rest before the first trial | `description="Intertrial"`, `trial_id`, `period="intertrial"` |

All six kinematics rows have `frequency=50`, `start=0` and `normalization="raw"`. Position rows have `processing=""`, velocity rows the recipe described under [Data Modalities](#data-modalities). `reference_frame` is `""` for translation and roll, since no source states the frame.

The `SpikeCounts` row, the `BinaryMask` row and the six kinematics rows load lazily via `SpecialLoader` (`_load_spike_counts` → a `(176, n_bins)` array read by `NeuralFeatureExtractor`, `_load_eval_mask` → a `(1, n_bins)` bool array read with `event.read()`, `_load_kinematics` → a dims-first `(n_dims, n_bins)` array read by `BehavioralSeriesExtractor`). The kinematics decompose the 7 mixed DoF into homogeneous, separately-selectable streams (position + velocity). The phase rows are pure time-span metadata read from `intervals/epochs`, see [Task phases & events](#task-phases--events) for the cue/movement taxonomy and `code` scheme.

`trial_id` on each phase event is the source's `TrialNum` value over most of the row's bins (`utils.phase_trial_ids`). `TrialNum` steps to the next trial a few samples before a `Release` row ends (2-4 in 180 of these rows, up to 11 in the rest), so 185 of the 3,052 phase rows span two values and take the one covering most of them. The leading `Intertrial` takes the first trial's id. Ids restart in every recording (at 1, or 3 in two recordings), so a trial is identified by `timeline` and `trial_id` together. The first and last trial of a recording can be partial. In `19250101_block1` and `19250115_block1` the recording opens with a 0.06 s `Release` fragment of trial 2, shorter than the early step, so it carries `trial_id=3`. Split by trial with `AssignHashSplit(by=["timeline", "trial_id"], event_type=...)` to keep a trial's phases together.

## Notes / Scope

- **Local or streamed:** the full dataset (**~102 MB**, 40 files) downloads to `<NEURALSHED_DATA_ROOT>/karpowicz_2024_h1/data/Karpowicz2024FalconH1/000954/` with the Study's `download()` (`overwrite=True` to fetch again). Reads work either way: a local file is read directly, otherwise it is streamed in place with `remfile` + `h5py` + `pynwb` (lazy reads).
- **Streaming + DANDI resolution are shared in `datasets/nwb.py`**: local or remote NWB opening (`stream="auto"`) plus DANDI asset listing/URL helpers, so `study.py` reads each session locally when present and otherwise streams it.
- **Per-bin eval mask.** The Study emits `acquisition/eval_mask` as a `BinaryMask` row (`true_means="scored"`). Benchmark-comparable training and scoring use only its True bins.

## Files

- `metadata.py`: Metadata for database insertion.
- `utils.py`: Dataset constants + accessors (`get_spike_times`, `bin_edges`, `bin_spikes`, `get_kinematics`, `get_eval_mask`, `get_epochs`).
- `study.py`: the neuralset `Study` (`Karpowicz2024FalconH1`).
- `01_explore_source_data.ipynb`: source-data exploration demo.
- `02_explore_study.ipynb`: Study demo.
- `03_model_example.ipynb`: decoding example.
