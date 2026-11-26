# athalye_2023: things to sort out

## Shared-file changes needed

Applied: the `studies.py` import and the two monkeys in `participants.py` (`carmena/G` and `carmena/J`, under a `carmena` lab namespace like `miller` in perich_2018). `cm` and `cm/s` came with perich_2018. Still open:

- **`src/neuralshed/datasets/participants.py`**: the module docstring and the `PARTICIPANTS` comment say "real people", which would need a word changed to cover animals.

## Decisions for Heberto

- **Rate.** The Study emits the source grid, 60 Hz for G and 200 Hz for J, and leaves resampling to the extractor. For the 20 ms collections: G's counts convert to Hz fine at any rate, but its cursor and velocity only change at 10 Hz, so a 50 Hz target is a step function. J sums cleanly into 20 ms bins, but `BehavioralSeriesExtractor` interpolates, and point-sampling J's noisy 200 Hz velocity every 20 ms drops its agreement with the 1 s cursor displacement from correlation 1.0 to 0.65 to 0.94. I recommend keeping native rates in the Study and making behavioral downsampling average over the destination bin (a shared extractor change), rather than rebinning inside this Study. The alternative is a per-subject emission rate (G at its 10 Hz decoder bins, J summed to 50 Hz), which makes G incompatible with every 50 Hz collection anyway.
- **`task_condition` for brain control.** No `vocab.TASK_CONDITIONS` value fits a monkey driving a cursor with a decoder (arm state unknown), so `Action` rows and `sessions()` use `other`. I recommend adding `"brain_control"` to `vocab.TASK_CONDITIONS` and switching the two literals here. The other monkey datasets being onboarded will hit the same question.
- **Position units.** J is scaled from m to cm on read so the dataset declares one unit per quantity. Keep that, or emit J in `m`/`m/s` and register those too. I recommend cm, since behavior is not converted by the extractor and G and J would otherwise differ by 100 in one pool.
- **Joining the motor cursor collection** (`models/data.py`, `MOTOR_CURSOR_MEMBERS`). Possible after registration, with `("Athalye2023CursorBMI", "athalye_2023", "athalye_2023/data")`, but the unit count changes every session (35 to 151 for G, 20 or 21 for J), so a batch spanning sessions does not stack (`Segmenter.load_all` raises on 44 versus 35 channels). This needs a per-session readin or padding, which the current members do not need.
- **Obstacle as a stream.** Monkey G's obstacle is emitted as `obstacle_x`, `obstacle_y` and `obstacle_size_cm` on the trial `Action` rows, not as a stream. If a continuous stream is wanted, add a sibling type in `events.py` and `vocab.EVENT_CLASSES`: `class ObstaclePositionSeries(PositionSeries): """Center of a task obstacle over time."""`. I recommend the attributes, since the obstacle is fixed within a trial.
- **Array naming and region.** Each hemisphere's array spans M1 and PMd, so `metadata.py` names both `array_1` and maps the single region to frontal motor cortex (`HOMBA_AA30009`, `UBERON_0001870`). The paper does not say whether the 16 x 8 array is one device or two per hemisphere.
- **`block` attribute.** The source block id is `te_num`, a float for G and a string for J, so it rides the timeline as `task_entry` (text) rather than the standard integer `block`. Say if you want `block` too, for G only or as a within-file index.

## Data problems

- **Dead padding in 4 monkey J blocks** (1846.0, 755.7, 544.0 and 266.9 s, 56.9 min in all): no spikes, frozen cursor, no trials. Every row of those timelines is cut at `utils.DEAD_TAIL_START`, recomputed and verified with `utils.find_dead_tail`. No `valid=False` is set, since the recorded part is fine.
- **G `target_state` all zeros**, J `target_state` and `obs_details` written against `go_ix`, a row index into the event-code array (verified in `online_analysis/nwb_utils.py`), and mixing m and cm. None is emitted and no target stream exists.
- **J `obstacle_position`** comes from the same misaligned `obs_state`, so it is not read. Only the block's task label is used for J.
- **Blocks reordered and concatenated**, center-out first. One timeline per block, identified by `task_entry`.
- **Rewarded trials only**, go cue to reward, no target or condition. `success=True` on every `Action`, gaps are `Intertrial`.
- **Unit identity changes per session** and there is no area or hemisphere per unit. `SpikeCounts.n_features` follows the file, and `metadata.py` declares no channel range.
- **Occasional large G counts** (up to 128 in one 16.7 ms frame, under 0.01% of entries above 5). Emitted as released.
- **G cursor differs from the decoder position** on up to 0.8% of frames per session (cause not checked). Emitted as released.
- **NWB metadata errors for J**: says Kalman filter and `P12Y`. `metadata.py` uses 10 years from the birth date, matching the paper. G's paper age (7 years) disagrees with its birth date (8 years 9 months), `metadata.py` follows the birth date.
- **Residual assist in J not verifiable.** The writer drops the first 55,003 samples of `jeev082413h` (assist on), and all J file names contain `assist_ofc_cont`. Every timeline says `loop_type="closed"`.

## Not done

- Notebooks (`01_explore_source_data.ipynb`, `02_explore_study.ipynb`, `03_model_example.ipynb`), so `metadata.py` says `status="study_wip"`.
- Target positions and condition labels. G's targets look recoverable from the reward-time cursor (on one of 8 directions, 10 cm out, and on the obstacle direction in obstacle blocks). J's center-out targets likewise, J's obstacle targets not.
- `update_bmi` is not emitted (`utils.get_update_bmi` reads it).
- `update_source_info` was not run. `_info` was computed by hand from a full `run()` (34 timelines, 242 events in the query) and has no `frequency`, because the query mixes 60 and 200 Hz.
- `python -m neuralshed.database.validate --datasets athalye_2023` and `--check-sessions` pass (34 timelines, 13 labels) with the data at `athalye_2023/data/Athalye2023CursorBMI/000404/`.
