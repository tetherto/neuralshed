# FALCON Benchmark H1: Human 7-DoF Reach and Grasp Motor BCI

Intracortical BCI benchmark from a single tetraplegic human participant (Univ. of Pittsburgh) implanted with two 96-channel Utah arrays in hand-knob motor cortex. During **open-loop 7-DoF reach and grasp** (3 translation, 1 rotation, 3 grasp), 40 NWB sessions record multi-unit **spike times** (176 units) alongside 20 ms / 50 Hz **7-DoF kinematics** (position and velocity) plus trial and eval-mask structure. The goal is to decode the reach-and-grasp kinematics and benchmark **cross-day decoder stability** across held-in and held-out days under few-shot recalibration.

## References

- **Paper:** *Karpowicz et al. 2024, "Few-shot Algorithms for Consistent Neural Decoding (FALCON) Benchmark," NeurIPS 2024 Datasets & Benchmarks* - [10.1101/2024.09.15.613126](https://www.biorxiv.org/content/10.1101/2024.09.15.613126v1).
- **Source data:** https://dandiarchive.org/dandiset/000954
- **Code repository:** https://github.com/snel-repo/falcon-challenge

**Task-paradigm references:**
- Collinger et al. 2013, *Lancet* 381:557: "High-performance neuroprosthetic control by an individual with tetraplegia."
- Wodlinger et al. 2015, *J. Neural Eng.* 12:016011: "Ten-dimensional anthropomorphic arm control…"
- Flesher et al. 2021, *Science* 372:831: "A brain-computer interface that evokes tactile sensations…"

## Species & Subjects

- **Species:** *Homo sapiens* (human).
- **Subject:** single iBCI participant, asset subject `sub-HumanPitt` (`subject_id = HumanPitt-...`). Details redacted by the data owners, with `sex: U` and `age: P22Y/P70Y` (range).
- **Implant:** two 96-channel Utah arrays (192 channels total) in **hand-knob motor cortex** (precentral gyrus).
- **Lab / institution:** Rehab Neural Engineering Labs, University of Pittsburgh.

## Behavioral Task

Open-loop **7-DoF reach and grasp**. The 7 degrees of freedom are 3 translation (`tx, ty, tz`), 1 rotation (`rx`), and 3 grasp dimensions (`g1, g2, g3`), see the `OpenLoopKinematics.description` field.

**"Open-loop" = observation-based calibration.** The prosthetic limb moves *automatically* along computer-generated target trajectories while the (tetraplegic) participant watches and **attempts/imagines** the movement. The neural data is recorded during that observation. So the kinematics are the *observed target* trajectory, not closed-loop brain-control output, and every movement is **attempted** (reflected in the Study's `task_condition="attempted"`).

Each trial is a structured **reach-grasp-transport sequence** decomposed into phases (Reach → Orient → Grasp → Carry → Reorient → Release), each preceded by a cue period in which the limb is held still while the next target is presented. These phases are exactly the `intervals/epochs` rows, see [Task phases & events](#task-phases--events).

FALCON framing: the dataset is split into **held-in** and **held-out** days to test cross-day decoding consistency. A decoder is trained on held-in calibration data, then **few-shot recalibrated** on a small held-out calibration slice and evaluated on held-out data. Only bins flagged by the per-bin `eval_mask` are scored.

## Task phases & events

Each `intervals/epochs` **row is one task *phase*, not a whole trial**, it carries exactly **one** `tags` entry (verified across all 40 sessions / 3 436 rows). The phases tile each trial contiguously. A typical trial reads:

```
Presentation → Reach → [Presentation2] → Orient → SnapTo → Presentation3 →
Shape → Presentation4 → Grasp → Presentation5 → Carry → [Presentation6] →
Orient2 → Presentation7 → Release
```

There are **16 distinct tags**. The Study (`study.py`) splits them three ways:

| Group | Tags | Event type | Notes |
|---|---|---|---|
| **Cues** | `Presentation`, `Presentation2`…`Presentation7` | `Stimulus` | Target-presentation periods, the limb is **paused** while the next target is shown. The number is **positional** (which movement it cues), *not* a different stimulus kind. `Presentation2` (pre-`Orient`) and `Presentation6` (pre-`Orient2`) are **optional**, so they appear in fewer trials. |
| **Movements** | `Reach`, `Orient`, `SnapTo`, `Shape`, `Grasp`, `Carry`, `Orient2`, `Release` | `Action` | Attempted-movement phases, one `code` each. `Orient` (pre-grasp) vs `Orient2` (= the references' **Reorient** phase) are genuinely different movements. `Shape` = hand pre-shaping. `SnapTo` is not named in the papers (likely a virtual auto-snap/assist state) but is treated as a normal movement phase. |
| **Rest** | `Intertrial` | `Intertrial` | Gap between trials, a custom event type (a pure time-span marker, no code). |

**Code scheme.** `Action.code` is one index per movement phase in task order:

| code | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| Action | Reach | Orient | SnapTo | Shape | Grasp | Carry | Orient2 | Release |

Each `Presentation*` cue is emitted as a `Stimulus` whose `code` **equals the `code` of the `Action` it precedes** (so a cue and its target movement share a code), with `description="cue: <Action>"`. The cue→action mapping is `Presentation→Reach`, `Presentation2→Orient`, `Presentation3→Shape`, `Presentation4→Grasp`, `Presentation5→Carry`, `Presentation6→Orient2`, `Presentation7→Release`. (`SnapTo`, code 2, has no preceding cue, so code 2 never appears on a `Stimulus`.)

**Cue modality is left unset** (`Stimulus.modality=None`): across the source paradigms the cues were *mixed* (translation/carry targets shown visually via LED / VR object / green plane, but orientation and grasp targets given as **auditory** verbal prompts), and the per-phase modality can't be recovered from the NWB alone.

The whole-session **`Spikes`** event (the binned Units table) is emitted unchanged alongside these phase events.

## Data Modalities

| Modality | NWB location | Type / shape |
|---|---|---|
| Neural | `nwb.units` (Units table) | **Spike times** (multi-unit), 176 units, ragged `spike_times`. NOT pre-binned. Bin at 20 ms yourself. |
| Kinematics (position) | `acquisition/OpenLoopKinematics` | `TimeSeries (n_bins, 7)`, 7 DoF |
| Kinematics (velocity) | `acquisition/OpenLoopKinematicsVelocity` | `TimeSeries (n_bins, 7)` |
| Trial id per bin | `acquisition/TrialNum` | `TimeSeries (n_bins,)` |
| FALCON eval mask | `acquisition/eval_mask` | bool `TimeSeries (n_bins,)` |
| Task phases / intervals | `intervals/epochs` | one row **per task phase** (single `tags` entry + `start_time`, `stop_time`), ~29-31 phase-rows in minival files, up to ~218 in held-in-calib. See [Task phases & events](#task-phases--events). |

- **Neural signal type:** sorted/threshold multi-unit **spike times** (not LFP, not pre-binned counts). 176 units in the released files (2 × 96-ch arrays).
- **Bin size / rate:** kinematics and masks are sampled at **20 ms bins → 50 Hz**. ⚠️ The `TimeSeries.rate` field stores `0.02`, which is the bin **period in seconds**, *not* a rate in Hz. Use the 20 ms / 50 Hz constants in `utils.py` (`BIN_SIZE_S`, `bin_centers()`), not the raw `.rate`.
- **eval_mask:** True ≈ 96-99% of bins, False marks transients / inter-trial bins excluded from FALCON scoring.

## Source Data Format & Layout

**DANDI asset naming** (40 assets total):

```
sub-HumanPitt-held-in-calib/   sub-HumanPitt-held-in-calib_ses-<YYYYMMDDThhmmss>.nwb   (13 files, ~3–7 MB)
sub-HumanPitt-held-in-minival/ sub-HumanPitt-held-in-minival_ses-<...>.nwb             (13 files, ~0.9–1.1 MB)
sub-HumanPitt-held-out-calib/  sub-HumanPitt-held-out-calib_ses-<...>.nwb              (14 files, ~1.2–1.5 MB)
```

The FALCON **split** is encoded in the `sub-*` folder / path: `held-in-calib` (training), `held-in-minival` (validation), `held-out-calib` (few-shot recalibration). Session timestamps are anonymized to 1925.

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
├─ units (Units table)            176 rows, spike_times (ragged)
└─ intervals/epochs               ~29–218 rows, one per task phase
                                   (start_time/stop_time + single `tags` label)
```

`n_bins` ranges ~1500-1830 (held-in-minival), ~2300-2710 (held-out-calib), and ~6500-12,300 (held-in-calib). Electrodes / electrode_groups / devices tables are empty in these files, array identity is documented only in metadata text.

## Study, timelines and Events

`study.py` (`FanFalconH1`) treats **each NWB session as one timeline** (40 across the FALCON splits). Each timeline carries `subject="HumanPitt"`, `session=<file stem>`, `task="reach_grasp"`, and its FALCON `original_split` (`held-in-calib` / `held-in-minival` / `held-out-calib`) broadcast to every event row. The relative asset `path` is the local, or streamed, source key.

Per timeline it emits:

| Event type | Count | Source → shape / rate | Key attributes |
|---|---|---|---|
| `Spikes` | 1 | `units` table (ragged spike times) → binned by `SpikesExtractor` | `frequency=50`, whole-session span |
| `PositionSeries` | 1 | `OpenLoopKinematics[:, 0:3]` → `(3, n_bins)` | `dims=(x,y,z)`, `frequency=50`, `unit="arbitrary"` |
| `OrientationSeries` | 1 | `OpenLoopKinematics[:, 3]` → `(1, n_bins)` | `dims=(x,)`, wrist rotation |
| `GraspSeries` | 1 | `OpenLoopKinematics[:, 4:7]` → `(3, n_bins)` | `dims=(g1,g2,g3)` |
| `LinearVelocitySeries` | 1 | `OpenLoopKinematicsVelocity[:, 0:3]` → `(3, n_bins)` | `dims=(x,y,z)` |
| `AngularVelocitySeries` | 1 | `OpenLoopKinematicsVelocity[:, 3]` → `(1, n_bins)` | `dims=(x,)` |
| `GraspVelocitySeries` | 1 | `OpenLoopKinematicsVelocity[:, 4:7]` → `(3, n_bins)` | `dims=(g1,g2,g3)` |
| `Stimulus` | per `Presentation*` row | one `intervals/epochs` cue phase | `code` = cued action's code, `description="cue: <Action>"`, `cue_tag` |
| `Action` | per movement row | one `intervals/epochs` movement phase | `code` (0-7), `description=<tag>`, `task_condition="attempted"` |
| `Intertrial` | per `Intertrial` row | inter-trial gap | `description="Intertrial"` |

The `Spikes` row and the six kinematics rows load lazily via `SpecialLoader` (`_load_spikes` → a live `h5py.File` the extractor closes after binning, `_load_kinematics` → a dims-first `(n_dims, n_bins)` array read by `BehavioralSeriesExtractor`). The kinematics decompose the 7 mixed DoF into homogeneous, separately-selectable streams (position + velocity). The phase rows are pure time-span metadata read from `intervals/epochs`, see [Task phases & events](#task-phases--events) for the cue/movement taxonomy and `code` scheme.

## Notes / Scope

- **Local or streamed:** the full dataset (**~102 MB**, 40 files) can be downloaded to `<NEURALSHED_DATA_ROOT>/fan_h1/` via the Study's `_download()`. Reads work either way: a local file is read directly, otherwise it is streamed in place with `remfile` + `h5py` + `pynwb` (lazy reads).
- **Streaming + DANDI resolution are shared in `datasets/nwb.py`**: local or remote NWB opening (`stream="auto"`) plus DANDI asset listing/URL helpers, so `study.py` reads each session locally when present and otherwise streams it.
- **Per-bin eval mask.** `acquisition/eval_mask` is a boolean series where `True` marks the bins the FALCON benchmark scores. Transients and inter-trial bins are masked out, so benchmark-comparable evaluation has to respect it.

## Files

- `metadata.py`: Metadata for database insertion.
- `utils.py`: Dataset constants + accessors (`get_spike_times`, `bin_spikes`, `get_kinematics`, `get_eval_mask`, `get_trials`).
- `study.py`: the neuralset `Study` (`FanFalconH1`).
- `01_explore_source_data.ipynb`: source-data exploration demo.
- `02_explore_study.ipynb`: Study demo.
- `03_model_example.ipynb`: decoding example.
