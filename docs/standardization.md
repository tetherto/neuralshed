# Standardization: shared conventions

The meanings and values every dataset must agree on, so that datasets can be indexed together in the metadata database and pooled together in training. It covers two layers that use overlapping words for different things:

- **Catalog metadata**, declared in `datasets/<name>/metadata.py` and ingested into the database. Allowed values live in [`vocab.py`](../src/neuralshed/database/vocab.py) (soft-checked) and [`schema.py`](../src/neuralshed/database/schema.py) (hard-checked).
- **Event metadata**, emitted by a Study as columns on its event rows and read at training time. Field definitions live in [`events.py`](../src/neuralshed/events.py), registered units in [`units.py`](../src/neuralshed/units.py).

Sections: [participant](#participant), [subject](#subject), [session id](#session-id), [arrays](#arrays), [modality](#modality), [event attributes](#event-attributes), [dense streams](#dense-streams), [units and normalization](#units-and-normalization), [brain area ontologies](#brain-area-ontologies), [channel correspondence](#channel-correspondence).

## participant

Person-invariant facts, declared once per real person in [`participants.py`](../src/neuralshed/datasets/participants.py), never on the subject. A dataset's subject reaches them through its `participant_id`.

- `species`: `human`, `mouse`, `rat`, `macaque` or `marmoset`. Hard-validated.
- `sex`: `M`, `F` or `U`, where `U` is unknown, redacted or not reported. Hard-validated.
- `implant_date`: full ISO date, e.g. `"2023-07-17"`, or `None` when no source publishes it. Day precision only, a month-precise `"2022-03"` is rejected. Hard-validated.
- `handedness`: one of `vocab.HANDEDNESS`.

A `sessions()` that derives `post_implant_day` or `date` reads the implant date from the registry, so it lives in exactly one place.

## subject

One dataset's view of a participant at the time of that dataset's recordings.

- `age`: ISO-8601 duration at the recordings, e.g. `"P30Y"` for 30 years, `"P6M"` for 6 months.
- `time_since_implant`: ISO-8601 duration from implant to the dataset's first session, e.g. `"P25D"`.
- `clinical_condition`: `list[str]`, one entry per condition, diagnosis first and any grading in parentheses, e.g. `"Spinal cord injury (C4 AIS-C)"`. `[]` when healthy or unknown.

Both durations are `None` when the source publishes no timing. Bare integers are rejected rather than coerced.

## session id

The session id joins the two representations of a session: `sessions.label` in the database (from `metadata.sessions()`) and the `session` key on each Study timeline. They must agree. Use the source's own per-recording identifier when it exists, typically the file stem, and define it once in the dataset's `utils.py` so both sides call the same function.

Enforce with `python -m neuralshed.database.validate --check-sessions`, which asserts that the timeline sessions are a subset of the session labels and skips datasets whose data is not reachable.

## arrays

An array is identified within a subject by its `hemisphere` and `name`.

- `hemisphere`: `left` or `right` (`vocab.HEMISPHERES`), `None` when the source does not state it.
- `name`: the implant area without a prefix (`6v`, `55b`, `6d`, `4`, `44`) when the area holds one array in that hemisphere, and `ventral_<area>` / `dorsal_<area>` when it holds two. The source's *inferior* and *superior* are *ventral* and *dorsal* on the lateral surface of the gyrus. With more than two along that axis, the inner arrays are `mid_ventral_<area>` / `mid_dorsal_<area>`. When the source names the axis lateral and medial, use `lateral_<area>` / `medial_<area>` (jude_2026_typing's T18 has four left 6d arrays and a lateral/medial right pair). The hemisphere never goes in the name: a bilateral implant has two `6d` arrays, one per `hemisphere`.
- Two arrays in the usual hand-knob implant, where the source gives no area label, are `lateral_array` and `medial_array`. When the source gives no position, use stable neutral names such as `array_1`.
- Give the same array the same name in every dataset of a participant. A position assumed rather than published is stated in the dataset's `metadata.py` comment and README.

`device_type` is count-agnostic (`utah_array`, never `utah_64`), the count is `n_channels`.

## modality

Three fields share the word:

- Catalog `events[].modality` (`metadata.py`) is the recording modality of a data stream, one of `vocab.RECORDING_MODALITIES` (`intracortical_microelectrode`, `lfp`, `behavior`, `derived`, ...).
- `BaseText.modality` on text events: `heard`, `read`, `imagined`, `typed`, `written` or `caption`.
- `Stimulus.modality` on stimulus events: `audio`, `visual` or `tactile`.

## event attributes

Beyond `type`, `start` and `duration`, a Study emits the columns a model will filter on. These are conventions rather than closed vocabularies, except where a `vocab.py` set already exists. Reuse an existing name before inventing one, and say in the README whether an id is unique within the dataset or within a timeline.

| attribute | where | meaning | values |
|---|---|---|---|
| `task_condition` | timeline or event | behavioral mode of the task | `vocab.TASK_CONDITIONS` (`attempted`, `imagined`, `mimed`, `typed`, `na`, ...) |
| `original_split` | timeline or event | the source's own partition label, verbatim | as published (`train`, `test`, `held-in-calib`, ...). Downstream transforms map it to train, validation and test. |
| `code` | `Action`, `Stimulus` | integer class id of the condition | int, state whether block-local or dataset-wide |
| `description` | any | human-readable label of the event | free text |
| `action_kind`, `stimulus_kind` | `Action`, `Stimulus` | coarse category of the action or cue | snake_case, reuse existing ones: `cursor_control`, `finger_control`, `finger_movement`, `gesture_hold`, `speech`, `joystick`, `drawing`, `rest`, `target_cue`, `gesture_cue`, `movement_cue`, `character_cue`, `word_cue`, `sentence_cue`, `speech_delay_cue`, `counting_grid`, `rest_cue`, `speech_go_cue`, `movement_go_cue` |
| `period` | trial event | the part of an instructed-delay trial a row spans | `delay` (cue shown, before the go cue), `go` (execution, and the one-bin go marker), `intertrial` (the gap before the next trial), `presentation` (a passive trial's stimulus, with no delay or go cue) |
| `trial_id` | event | the source's trial identifier | int |
| `block` | timeline | the source's block number | int |
| `success` | `Action` | trial outcome as the source reports it | bool |
| `loop_type` | timeline | whether the decoder drove the task | `open`, `closed` |
| `vocab` | timeline | vocabulary the cued sentences were drawn from | `50w`, `125k`, `na` when the timeline cues no sentences |
| `valid`, `valid_description` | timeline or dense event | whether the released data are usable as recorded, and why not: on the timeline when the whole block is affected, on a dense neural row or a trial event when only that row is (the row keeps its value, the timeline value fills the rest) | bool, `True` unless the source or the audit found the data bad; a short free-text reason when `False` (`"area 44 recordings invalid"`), `""` otherwise |
| `post_implant_day`, `date` | timeline | session clock | int, ISO date |
| `feature` | dense events | logical stream name | see [dense streams](#dense-streams) |

A gap between two trials of a timeline is an `Intertrial`, including a movement the source records no kinematics for, such as returning a pen to its start (`description="return period"`). `Action` is kept for the cued behavior of a trial.

Preserve source labels rather than replacing them. A trial the source marks as failed or timed out carries `success=False`, it is not dropped. A block whose recording is known to be bad (e.g. a collapsed array) is emitted as released with `valid=False` and the reason in `valid_description`, never repaired, masked or dropped in the loader. When the Study emits one dense row per array, a bad array carries the flag on its own rows and the timeline stays valid. `NeuralshedStudy` sets `valid=True` and `valid_description=""` on every timeline that does not declare them and keeps both keys out of the timeline id, so `query="valid"` selects the usable timelines of any dataset and the two columns are complete and boolean in a collection frame. A Study declares them only where the data are bad.

Timeline-dict values keep their types on the events frame: `valid` is a bool, `block` and `post_implant_day` are ints, and the same holds after concatenation across timelines and datasets. Filter with `valid == True` or `block == 3`, never with the string forms.

## dense streams

Every `DenseArrayEvent` row (`SpikeCounts`, `BandPower`, the `BehavioralSeries` family) declares `feature`, `n_features`, `frequency`, `unit`, `normalization` and `processing`. Type-specific fields are documented in `events.py`.

- `feature` names the logical stream. It is a query key and a cache key, so equivalent streams should share it across datasets: `threshold_crossings` and `spike_band_power` for the two standard neural features, `lfp_<lo>_<hi>` for a local-field-potential band, and for behavior the quantity rather than the source key (`cursor_position`, `target_position`, `cursor_velocity`). It need not equal the catalog's coarser `signal_type`.
- `threshold` (`SpikeCounts`) keeps the corpus shape `"<signed multiplier> x <RMS|SD>"`, e.g. `"-4.5 x RMS"`, a range as `"-3.5/-4.5 x RMS"`, `""` when unpublished.
- `processing` is free text holding, in order, everything the source's chain did beyond what `normalization` names, as semicolon-separated clauses, e.g. `"log-transformed before normalization; rolling 10 s statistics"`. `""` when the source did nothing else.

## units and normalization

Three fields describe what the numbers in a neural stream are. `NeuralFeatureExtractor` reads them with `frequency` to put pre-binned features from different datasets on one scale, so a wrong or missing value produces wrong numbers rather than a cosmetic blemish. Every `unit` and `normalization` string a Study emits must be registered in `units.py`, and `tests/test_units.py` reads the study sources to enforce it. [`02_neural_harmonization.ipynb`](tutorials/02_neural_harmonization.ipynb) is the executable version of this section.

| field | question it answers |
|---|---|
| `unit` | what physical quantity went into the source's chain, and so how to convert |
| `normalization` | which statistic the source removed, and so whether conversion is still valid and which datasets can pool |
| `processing` | everything else the chain did, for people and for pooling code deciding whether two streams really match |

### unit

`unit` records the quantity the source's chain took as input, whatever it did afterwards. A z-scored threshold-crossing stream declares `counts/bin`, not `a.u.`. `a.u.` is for a quantity that is unknown or was never a measurement of anything physical. Three classes:

| class | meaning | strings |
|---|---|---|
| `count` | per bin, scales with bin width | `counts/bin` (alias `count`) |
| `rate` | per second, invariant to bin width | `Hz` |
| `non_convertible` | neither | `uV^2`, `a.u.` (aliases `au`, `arbitrary`), `a.u./s`, and the behavioral `px`, `px/s`, `screen_height`, `screen_height/s`, `proportion`, `binary`, `LPCNet` |

`counts/bin` is not self-contained: a channel firing at 20 Hz reads `0.2` in a 10 ms dataset and `0.4` in a 20 ms one. `Hz` is, which is why it is the harmonization target. Crossings are counts and are summed when bins merge, band power is an average and is `uV^2`.

### normalization

One of `raw`, `mean_subtracted`, `z_scored`, in that order of processing. It names only the statistic that was removed. A pool's target must be at least as processed as every member, so a pool containing one z-scored dataset is a z-scored pool.

### the interaction rule

Unit conversion is valid only when `normalization` is `raw` or `mean_subtracted`. Z-scoring divides by a per-channel standard deviation and leaves dimensionless values, so a frequency factor would be meaningless. This is what lets `unit` record the pre-normalization quantity: `card_2024` declares `counts/bin` and looks convertible, and `normalization="z_scored"` is what stops it.

### asking for a unit

```python
NeuralFeatureExtractor(target_unit="Hz", frequency=50.0)       # harmonized across datasets
NeuralFeatureExtractor(target_unit="native", frequency=50.0)   # as the source stored it
```

`native` keeps the source's unit and inspects nothing when the rate is unchanged. Conversion runs before resampling and routes through the rate at each count end, so `counts/bin` at 100 Hz to `counts/bin` at 50 Hz is a factor of 2, not a no-op. The extractor's header always states the `(unit, frequency)` its values are in.

Neural resampling averages the source over each destination bin, behavioral resampling interpolates. Two cautions: a rebinned z-scored stream is no longer unit-variance and wants re-standardizing downstream, and upsampling invents no structure below the source bin width.

### adding a dataset

Trace the source's chain back to its input, verify against the data rather than the paper, and declare that quantity as `unit`. Declare `normalization` explicitly for every neural stream, never rely on the default. Record the rest of the chain in `processing`. Register any new string in `units.py`. Studies stay faithful to their source units, the extractor is the only layer that converts.

## brain area ontologies

Each subject in `metadata.py` declares its implant regions in a `brain_areas` list, one dict per region keyed by `raw_label`, and its arrays reference a region by `brain_area == raw_label`. Each region carries two ontology ids, stored compact (`HOMBA_10165`, `UBERON_0001384`):

- `homba_id`, the finest HOMBA term the source evidence supports. HOMBA (Harmonized Ontology of Mammalian Brain Anatomy) ships as `homba-full.json` at the repo root, and `neuralshed/regions/` holds the induced subtree of the terms in use.
- `uberon_id`, the closest UBERON term. UBERON is coarser than HOMBA for somatotopic subdivisions, so it is often the parent region.

Procedure:

1. Translate the source's implant description into canonical anatomy, e.g. "hand knob of the precentral gyrus" is the hand area of primary motor cortex.
2. Pick the deepest HOMBA term the evidence justifies and no deeper. Do not assert a somatotopic subregion when the source says only "motor cortex", and do not pick a laminar term for a surface recording that spans layers.
3. Take the UBERON term linked from that node or, when the leaf has none, from its nearest ancestor that does.

Give the same region the same ids across subjects and datasets. A region with no mapping yet needs no entry.

## channel correspondence

The same participant appears across several datasets, but a shared participant and an equal channel count do not establish that channel `i` is the same electrode in both. Treat channel order as dataset-specific until a source states the mapping. Evidence for a mapping belongs in the datasets' READMEs, not here. Within a dataset, the columns of a dense neural row follow the catalog's array order, one contiguous block per array as `channel_start` / `channel_stop` describe. When that differs from the source column order, the Study applies the permutation and the README states it (see willet_2023 for example).
