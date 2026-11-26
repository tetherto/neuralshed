# even_chen_2019: things to sort out

## Shared-file changes needed

Applied: the `studies.py` import, the two monkeys in `participants.py` (`shenoy/Jenkins` and `shenoy/Reggie`, under a `shenoy` lab namespace like `miller` in perich_2018), and `mm` and `mm/s` in `units.py`. `executed` in `vocab.TASK_CONDITIONS` came with perich_2018. Still open:

- **`src/neuralshed/database/vocab.py`** and **`docs/standardization.md`**: `loop_type` is documented as `open` or `closed` and nothing checks it. This Study adds `hand_control` (see the decision below). A controlled set belongs with #34 (Add an `assisted` value to `loop_type`).

- **`src/neuralshed/datasets/nwb.py`**: move `utils.binned_spike_counts` there as a generic `bin_units(h5file, bin_size_s, n_bins)` over any NWB `units` table (`fan_h1/utils.py` has a pynwb copy of the same loop).

## Decisions for Heberto

- **`loop_type` for hand control.** Hand control is not `open_loop`: no decoder runs and the subject's real arm closes a feedback loop with the cursor. I emit `hand_control`, and `closed` for the brain-control block as `standardization.md` documents it. Alternative: keep `loop_type` for decoder datasets only and add a separate `control` attribute (`hand`, `brain`). Recommendation: `hand_control` in `loop_type`, since one attribute answers "what drove the cursor" across the corpus.
- **`loop_type` sits on trial rows only.** One session mixes both kinds, so the dense rows have no value and `QueryEvents("loop_type == 'hand_control'")` would drop them. The working filter is `"not (type in ['Action', 'Stimulus'] and loop_type != 'hand_control')"`. Recommendation: keep it, and document the pattern once in `standardization.md` for per-trial attributes.
- **Behavior at 50 Hz, not the source's 1 kHz.** The 1 kHz grid is 60 Hz sample-and-hold, so a velocity differenced on it is mostly zeros and spikes. The Study averages it into the spike-count bins. Recommendation: keep, the alternative (emit 1 kHz and fix `DeriveStream`) touches shared code and multiplies memory by 20.
- **Tracking-loss sentinel and eye dropouts become NaN in the loader.** This decodes a missing-data marker rather than repairing data, but the standardization text says "never repaired, masked or dropped in the loader". Recommendation: keep, and state the distinction in `standardization.md`.
- **Motor cursor collection membership.** Inputs (`SpikeCounts`) and the `action_kind == 'cursor_control'` trigger already fit. The declared target `DecodedCursorVelocitySeries` does not exist here (no decoder), but `derive=("velocity",)` from `CursorPositionSeries` gives `CursorVelocitySeries` in mm/s. A member query should keep `loop_type == 'hand_control'` and `valid`. Recommendation: add it as a hand-control member only if the collection target moves to `CursorVelocitySeries`.
- **Reggie's M1 array.** The paper calls it unusable and 29 of its channels are silent. Options: leave as is (current), or emit one `SpikeCounts` row per array and set `valid=False` on Reggie's M1 row. Recommendation: per-array rows, if the collection code handles two neural rows per timeline.
- **Array names `M1` and `PMd`.** The naming rule's examples are human area labels (`4`, `6d`). I used the monkey literature's names. Recommendation: keep for macaques, and add the rule to `standardization.md`.
- **Attribute names I coined:** `trial_kind` (`center_out`, `return`), `delay_duration`, `target_layout`, `n_targets` (the last as in `wilson_2025`). Rename if the corpus has better ones.
- **`action_kind="cursor_control"`** for an arm reach. `area2_bump_2020` uses `reaching` for the same kind of movement. Recommendation: `cursor_control` whenever a cursor is the controlled object, so the motor cursor trigger works across members, and settle it across the monkey datasets.

## Data problems

- **Brain-control block unflagged** (Reggie 2017-01-19 trials 974 to 1107). Detected from `DecodePos` at each trial's middle sample, verified filled on all samples of those 134 trials and NaN in every other trial of all 12 files. The paper-note rule "reach limit above 2 s" is not safe: one Reggie 2017-01-15 trial has a 2.5 s limit under hand control. Emitted as `loop_type="closed"`, `task_condition="other"`.
- **Off-screen cursor after the block** (trials 1108 to 1116). Detected by the cursor lying beyond 500 mm, sentinel excluded, on most samples. Emitted with `valid=False` on the trial rows, data untouched.
- **Tracking-loss sentinel** in `Hand` and `Cursor` (three distinct rows, 0.02% to 12% of samples). Set to NaN before binning.
- **Eye dropouts** as all-zero rows (1% to 7% of samples). Set to NaN. Eye stays in raw units (`a.u.`).
- **No data between trials.** Spike counts are zero and behavior NaN there. Emitted as `Intertrial` rows saying so. Windows that reach before a trial start see these zeros.
- **Units table contradicts the electrodes table** on which array holds channels 0 to 95. Followed the electrodes table (M1 first), which also contradicts the paper note (it follows `electrode_group`). The note's "29 PMd channels silent in Reggie" is really 29 M1 channels.
- **Trials whose target never appeared** (12 to 722 per session, all failed outward trials). Emitted as a whole-trial `Action` with `success=False` and `delay_duration=NaN`, no `Stimulus`.
- **Positions labelled meters** are millimeters. Emitted as `mm`.
- **`reach_time`, `target_hold_time`, `fail_time`** are limits plus `start_time`. Not emitted. `utils.get_trials` exposes them as `reach_limit`, `hold_limit`, `fail_limit`.
- **`task_id` and `version_id`** constant per monkey. Layout derived from `target_pos`, Reggie's line labels follow the files, not the paper.
- **LFP** segments cannot be aligned, `juice_reward` values undocumented. Neither is emitted.

## Not done

- Notebooks `01_explore_source_data.ipynb`, `02_explore_study.ipynb`, `03_model_example.ipynb` (out of scope for this pass). Status is `study_wip` in `metadata.py`.
- `python -m neuralshed.database.validate --datasets even_chen_2019` and `--check-sessions` pass (12 timelines, 12 labels) with the data at `even_chen_2019/data/EvenChen2019DelayedReach/000121/`.
- `_info` was computed by running the Study on its query (18,735 events), not with `update_source_info`.
- Streaming mode (no local files) is wired through `utils.resolve_source` but was not exercised.
