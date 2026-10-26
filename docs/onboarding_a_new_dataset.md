# Onboarding a new dataset: the phase guide

The end-to-end workflow for taking a dataset from the catalog ([datasets.md](datasets.md)) to ML-ready. Three phases, each ending in a status from the `datasets.md` legend. The `Study` class itself is covered in [writing_a_neuralfetch_study.md](writing_a_neuralfetch_study.md), this guide places it in context.

| Phase | Deliverables | End status |
|---|---|---|
| 1, Comprehension | `utils.py`, `metadata.py`, `01_explore_source_data.ipynb`, `README.md` | `onboard_ready` |
| 2, Study composition | `study.py`, any new package-level Events/Extractors, `02_explore_study.ipynb` | `study_ready` |
| 3, Utilization | `03_model_example.ipynb` | `ml_ready` |

Everything for one dataset lives in `src/neuralshed/datasets/<name>/`, except reusable Events and Extractors (see Conventions). `<name>` is the first author's surname and the journal year of the published paper (`card_2024`), with a topic suffix when that author has more than one paper in the year (e.g.: `jude_2026_speech` and `jude_2026_typing`).

---

## Conventions (apply throughout)

- **Environment.** Run everything in the project environment described in the root [README](../README.md).
- **Source-data path, never hardcode.** `dataset_dir("<name>")` from `neuralshed.datasets.utils` resolves to `<NEURALSHED_DATA_ROOT>/<name>/`. Source files downloaded by the Study land under `<NEURALSHED_DATA_ROOT>/<name>/data/<StudyClass>/`. Pass `must_exist=True` for a clear error when the data drive is not accessible.
- **Reads default to downloaded local files.** DANDI/NWB datasets can instead be streamed in place through `datasets/nwb.py`, which might be preferable to bulk-downloading a very large dandiset. See [dandi_nwb.md](dandi_nwb.md).
- **Downloads go through shared backends.** `datasets/dryad.py` (`Dryad`) for Dryad deposits, `neuralfetch.download.Dandi` for DANDI. Dryad needs an API token in `NEURALFETCH_DRYAD_TOKEN`.
- **Reference materials** live at `<NEURALSHED_DATA_ROOT>/<name>/references/`, typically these are the publication PDF and a clone of the authors' code repository.
- **Generalizable code goes in package modules, not the dataset folder.** Reusable Events in `src/neuralshed/events.py`, Extractors in `src/neuralshed/extractors.py`, shared NWB access in `src/neuralshed/datasets/nwb.py`. Only dataset-specific helpers go in `datasets/<name>/utils.py`.
- **Notebooks** are didactic walkthroughs, not code references. One statement per line, a short comment per code block.
- **Every notebook starts by building the Study and calling `download()`.** The setup cell is `s = StudyClass(path=dataset_dir("<name>") / "data")` followed by `s.download()`, and later cells read source files through `s.path`. The public `download()` returns immediately once the success file exists, so the call costs nothing after the first run, and it is the same path the CLI uses. Never call the private `_download()`. Credentials come from the environment or a local `.env` file loaded with `python-dotenv`.
- **README and notebook markdown.** No soft line breaks, no em-dashes and no semicolons inside sentences. Use commas or parentheses.
- **Verify against real data files**, not only the paper. Units, shapes and timebase quirks are routinely undocumented or wrong.
- **Concise comments and docstrings.**
- **Standardize field values.** Controlled metadata values and shared event attributes follow [standardization.md](standardization.md) and `vocab.py`. Do not invent per-dataset variants.

---

## Phase 1, Dataset comprehension, ends in `onboard_ready`

**1.1 Read all references.** Publications, the source archive page, and any benchmark or code repository. Extract subjects and implants, brain regions, task structure, the recording modalities and their sampling, time alignment, published splits and known quirks.

Subject and implant metadata is a first-class deliverable. Record everything the sources publish: age, time since implant, handedness, clinical conditions, implant locations, stereotaxic coordinates when available, array and device details. It lives in `metadata.py` (`DATASET_METADATA`) and must validate against the metadata-DB schema, see [the schema section](#dataset-metadata-schema-metadatapy). Each subject references a person in [`participants.py`](../src/neuralshed/datasets/participants.py) through `participant_id`. Add a new entry there only for a genuinely new person. Use `None` or `[]` for unavailable fields rather than dropping them.

**1.2 Parse the source data.** Write `utils.py`, the dataset's data-access layer, with typed accessors (`get_*`, `list_*`) and constants (rates, channel counts, dimension labels). Define the session label once here so `metadata.sessions()` and the Study's timelines agree (see the session id section of [standardization.md](standardization.md)).

**1.3 Write `01_explore_source_data.ipynb`.** A didactic, end-to-end demo that exposes the main modalities and the experiment's organization to a human reader. Execute it clean.

**1.4 Write `README.md`** following the skeleton below.

---

## Phase 2, Study composition, ends in `study_ready`

**2.1 Map event types per modality.** For each modality pick the closest event type, from neuralset (`Spikes`, `Stimulus`, `Action`, `Sentence`, ...) or from `neuralshed/events.py` (`SpikeCounts`, `BandPower`, the `BehavioralSeries` family, `Intertrial`, ...). Where none fits, design a new one. Three things to be attentive to:

- **Decompose heterogeneous signals into homogeneous events.** A 7-DoF kinematic array becomes a `PositionSeries` (x, y, z), an `OrientationSeries` and a `GraspSeries`, not one mixed `Kinematics` event. Dense arrays are dims-first, `(n_features, n_samples)`.
- **Type selects the kind, attributes select the stream and the condition.** An extractor's `event_types` picks the kind of event, its `query` picks one stream among several of that kind (for example one band among overlapping `BandPower` events, by `feature`), and `Study.query` or a `trigger_query` picks timelines and trials by attribute. Give parallel streams of one type a distinct `feature`, and use a thin subclass when two streams must be selectable by type alone.
- **Emit the attributes relevant for filtering.** Beyond `type`, `start` and `duration` an event carries any column you attach (`code`, `description`, `task_condition`, `original_split`, `success`, ...). Decide up front which conditions a model might train on or hold out, and attach a column for each. Preserve the source's own labels, such as its published split and its trial outcomes. An attribute you do not emit cannot be selected later. Shared names are listed in [standardization.md](standardization.md).

**2.2 Write `study.py`.** Subclass `NeuralshedStudy` and follow [writing_a_neuralfetch_study.md](writing_a_neuralfetch_study.md) for the contract. Decide what one timeline is (session, block or trial, whichever is the largest span the source delivers on a single continuous timebase) and which whole-timeline labels ride on the timeline dict.

**2.3 Write or extend Events and Extractors in package modules.** New reusable Events go in `events.py`, new Extractors in `extractors.py`. Import the event types you emit in `study.py` (`# noqa: F401`) so they register. Prefer the existing `NeuralFeatureExtractor` and `BehavioralSeriesExtractor` over a new one.

**2.4 Register the dataset.** One line each: `dataset_id` on the Study, `participant_id` on every subject in `metadata.py`, the Study import in `neuralshed/studies.py`, new event classes in `vocab.EVENT_CLASSES`, new `unit` strings in `units.py`. Then run:

```bash
python -m neuralshed.database.validate --datasets <name>
python -m neuralshed.database.validate --check-sessions --datasets <name>
```

The second command asserts that the Study's timeline sessions are a subset of `metadata.sessions()`. It reports skipped, not passed, when the data is not reachable.

**2.5 Write `02_explore_study.ipynb`.** Build the Study on a small `query`, show the standardized events table, then read each event type through its extractor to confirm shapes, units and timing against the source. Exercise every event family the Study emits, and sample distinct recording conditions where the processing differs. After the events table, open it in `viz.explore_events(events)` so a reader can click through the timelines and see each event's payload (see [10_visualization.ipynb](tutorials/10_visualization.ipynb)). Execute clean.

**2.6 Lock the `StudyInfo` guard.** Every Study ships a filled `_info` computed from the real data, with a `query` that exercises every emission path. The recipe is in section 11 of the study guide. A missing or empty `StudyInfo()` does not count as `study_ready`.

**2.7 Review before declaring `study_ready`.** Check that processing and referencing are stated per block kind rather than assumed constant, that sign and axis conventions were verified against the data rather than the paper, that timed-out or failed trials carry an attribute rather than being dropped or passed off as successes, that blocks with known-bad recordings carry `valid=False` and a `valid_description` rather than being dropped or silently repaired, and that the timeline dict holds no plumbing keys.

---

## Phase 3, Dataset utilization, ends in `ml_ready`

**3.1 Write `03_model_example.ipynb`.** Simple yet informative: a `Segmenter` with a neural input extractor and a target extractor from `neuralshed/extractors.py`, a split that keeps related trials together, normalization fitted on the training split only, a small model in plain PyTorch, and a task-appropriate metric reported in the target's own units. State the tutorial's scope and limits rather than tuning past them.

**Next steps.** Pooling several datasets, collection layouts and batching are covered by the tutorials under [docs/tutorials/](tutorials/), starting with `00_collections.ipynb`.

---

## README.md organization

Section skeleton (see `datasets/fan_h1/README.md` for a worked example):

1. **Title**, `<Dataset>: <one-line task description>`.
2. **References**, publication(s), source archive and API, benchmark or code repository.
3. **Species & Subjects**, species, sex and implant date (participant), subject id, age, time since implant, clinical conditions, implant (arrays, brain area, stereotaxic coordinates if published), lab. Mirror `metadata.py`.
4. **Behavioral Task**, what the subject did and the trial structure.
5. **Data Modalities**, a table of modality, source location and type or shape.
6. **Source Data Format & Layout**, asset naming, formats, internal file layout, size range.
7. **Study, timelines and Events**, what `study.py` treats as one timeline and why, the event rows it emits per timeline and their filterable attributes. A table of event type, source, shape or rate and notes works well.
8. **Notes / Scope**, streaming versus download, reusable read pattern, gotchas and known limitations.
9. **Files**, one line per `.py` and notebook in the dataset folder.

---

## Dataset metadata schema (`metadata.py`)

`metadata.py` is standalone (it does not import the DB package) and exposes:

- `DATASET_METADATA: dict`, static, catalog-level, no data drive needed.
- `sessions() -> list[dict]` (optional), per-session rows wrapping `utils.py`, needs the data drive.
- `get_metadata() -> dict`, the accessor the DB layer consumes: `DATASET_METADATA` with `sessions()` folded in under `"sessions"`, `[]` when the drive is absent.

The fields and allowed values live in the code: the ORM [`schema.py`](../src/neuralshed/database/schema.py) and the soft-enforced vocabularies [`vocab.py`](../src/neuralshed/database/vocab.py). An unknown value warns but does not fail, and most sets carry an `"other"` value. [standardization.md](standardization.md) is the human-readable companion for the controlled values. Read those plus a worked example such as [`card_2024/metadata.py`](../src/neuralshed/datasets/card_2024/metadata.py), and validate after any change with the command in 2.4.

Guidance the schema cannot tell you:

- **One `events` list for everything.** Data streams and text or categorical markers share the single `events` list, each tagged with `category` (`data`, `text`, `categorical`) and, where it maps cleanly, an `event_class` (the concrete event subclass).
- **Standardize, don't invent per-dataset.** Take `category`, `event_class`, `signal_type`, `modality` and `device_type` from the vocab, and keep the source's own label in the event's `name`. When nothing fits use `"other"` where the vocabulary allows it and add the value to `vocab.py`. `device_type` is count-agnostic: `utah_array`, never `utah_64`.
- **`clinical_condition` is a `list[str]`**, one entry per condition, `[]` when healthy or unknown.
- **`brain_area` and `stereotaxic_coordinates` are independent.** `brain_area` always names the implant location, coordinates are added only when the source publishes them.
- **Region ontology lives in the subject's `brain_areas` list, not on the array.** Each array names its region through `brain_area`, and the ontology mapping (`homba_id`, `uberon_id`, optional `brodmann` and `description`) goes in the subject's `brain_areas` list, one entry per region keyed by `raw_label`. See [standardization.md](standardization.md) for choosing the ids.
- **Every array declares `hemisphere`** (`left`, `right` or `None`) and follows the naming rule in [standardization.md](standardization.md#arrays). `hemisphere` and `name` together identify an array within its subject.
- **Don't drop a field you can't fill.** Set `None` or `[]` so the absence is explicit and the shape stays uniform across datasets.

---

## Per-dataset file checklist

```
src/neuralshed/datasets/<name>/
├─ __init__.py
├─ metadata.py                     # DATASET_METADATA + sessions() + get_metadata()
├─ utils.py                        # accessors and constants
├─ study.py                        # the NeuralshedStudy subclass
├─ README.md
├─ 01_explore_source_data.ipynb
├─ 02_explore_study.ipynb
└─ 03_model_example.ipynb
```

Reusable, cross-dataset code lives outside this folder:

- `src/neuralshed/events.py`
- `src/neuralshed/extractors.py`
- `src/neuralshed/text.py`: the frozen character and phoneme vocabularies, and `PhonemeTable` for datasets whose source gives only the prompt text (build a cue-to-phonemes table once with the shared `g2p_phonemes` recipe, ship it under `assets/`, read it at load time)
- `src/neuralshed/datasets/nwb.py`
