# Writing a neuralshed `Study`

How to write the `Study` class for a dataset in `src/neuralshed/datasets/<name>/study.py`. A neuralshed Study is a neuroai `Study` (from `neuralset`) with a thin project base on top. This guide covers the contract, the neuralshed conventions, and a skeleton to copy. For a complete worked examples, read current implementations at `neuralshed/datasets/<dataset_id>/study.py`.

Framework references, when you need them: `neuralset/events/study.py` (base `Study`, `StudyInfo`, `SpecialLoader`) and `neuralset/events/etypes.py` (built-in event types).

---

## 1. What a Study is

A `Study` is an adapter from one external dataset to a single, canonical `Events` DataFrame. It does three jobs:

1. **Fetch** the raw files (`_download`).
2. **Enumerate** the recording units of the dataset (`iter_timelines`).
3. **Describe** each unit as rows of events (`_load_timeline_events`).

A Study produces metadata about events: rows with a `type`, a time span, and a pointer to where the signal lives. The heavy signal arrays are read lazily through the `filepath` column, later, by extractors.

A **timeline** is the largest span the source delivers on a single continuous timebase: a session when its blocks share one grid, a block when blocks are separated by gaps or re-zeroed clocks, a trial when the release ships per-trial clips. Do not split a continuous recording further. `iter_timelines` is the dataset's table of contents, it enumerates the dataset as one dict per timeline. `.run()` loads every timeline (or those selected by `query`), concatenates them, and returns the standardized DataFrame.

Timelines load inline and uncached in neuralshed, so `_load_timeline_events` should stay light. Opening the raw file to read timing and labels is expected. Reading the bulk signal is not.

---

## 2. The neuralshed base: `NeuralshedStudy`

Every study subclasses `NeuralshedStudy` from [`datasets/study.py`](../src/neuralshed/datasets/study.py) and declares a `dataset_id` equal to its folder name. The base:

- Maps each timeline `subject` to a `participant_id` through `datasets/<name>/metadata.py`. A subject label missing from `metadata.py`, or a subject without a `participant_id` known to [`participants.py`](../src/neuralshed/datasets/participants.py), raises.
- Adds `dataset_id` and `participant_id` columns to every event row.
- Validates every dense-array event (`SpikeCounts`, `BandPower`, `BehavioralSeries` subclasses): positive `duration` and `frequency`, and a positive integer `n_features`.
- Configures timeline loading, so no `model_post_init` override is needed.

One optional hook: `_postprocess_timeline(events, timeline)` runs after provenance injection and before validation, for any customization that might need to be done on top ofthe standardized frame.

---

## 3. The methods you implement

| Method | Required | Returns | Job |
|---|---|---|---|
| `iter_timelines(self)` | yes | `Iterator[dict]` | Yield one dict per timeline. Must contain `"subject"`. |
| `_load_timeline_events(self, timeline)` | yes | `pd.DataFrame` | Build the event rows for one timeline. |
| `_download(self, overwrite=False)` | yes | `None` | Fetch the raw files under `self.path`, forwarding `overwrite` to the backend. |
| `_load_<signal>(self, timeline, **kw)` | usually | array or handle | The lazy loader a `SpecialLoader` points to (section 6). A convention, not a base method. |

---

## 4. The Events DataFrame

Every row is one event. neuralset validates each row against its typed `Event` model, fills defaults, sorts by `(timeline, start)` and adds `stop`.

**Required per row:** `type` (an `Event` subclass name), `start` and `duration` in seconds.

**Injected by the framework, never set them yourself:** `subject` (rewritten to `<StudyName>/<subject>`), `timeline`, `study`, `session`/`task`/`run` (from the timeline dict), `stop`, and the neuralshed columns `dataset_id` and `participant_id`.

**Any other column rides through.** `filepath`, `frequency`, `feature`, `code`, `description`, `task_condition`, `original_split` and domain fields are all ordinary columns. Emit the attributes a model might later filter on. An attribute you do not emit cannot be selected later. Names and values for the shared ones are in [standardization.md](standardization.md).

**Timeline dict keys broadcast to every row.** Every key except `subject`, `path` and `timeline` becomes a column on all of that timeline's events. Put whole-timeline labels there (`original_split`, `task_condition`, `block`). Keep event-level facts on the rows. Emitting the same column at both levels raises when the values differ and warns when they match.

**Timeline dict keys form the timeline id.** The id is the class name plus every key and value in the dict, `path` included, and it feeds every downstream cache key. Keep the dict stable and concise: identity, whole-timeline labels, and one file reference under `path`. Do not add plumbing such as a file index or a start time.

---

## 5. Event types

neuralset provides the general taxonomy: `Spikes` (spike times, from NWB units), stimuli and media (`Image`, `Audio`, `Video`), language (`Text`, `Sentence`, `Word`, `Phoneme`, `Keystroke`), and categorical markers (`Stimulus`, `Action`, `Background`, ...). A new type is a subclass of the closest existing one and registers itself by class name.

neuralshed adds the types iBCI datasets need in [`events.py`](../src/neuralshed/events.py). The dense ones share `DenseArrayEvent`, a lazily read `(n_features, n_samples)` array at `frequency` Hz:

- `SpikeCounts` for pre-binned spike or threshold-crossing counts, `BandPower` for binned power in a band.
- `BehavioralSeries` and its concrete subclasses (`PositionSeries`, `CursorPositionSeries`, `TargetPositionSeries`, `GraspSeries`, velocity and acceleration series, `AcousticFeatures`, ...). Always emit a concrete subclass so it can be selected by type.
- `Character` (a glyph) and `Intertrial` (dead time between trials) as non-dense additions.

Every dense row declares `feature` (the stream name), `n_features` (rows of the array), `frequency`, `unit`, `normalization` and `processing`. `unit` and `normalization` are read by `NeuralFeatureExtractor` to put datasets on one scale, and every value must be registered in [`units.py`](../src/neuralshed/units.py). The meaning of the three fields is in [standardization.md](standardization.md). Type-specific fields (`method`, `threshold`, `lower_freq`, `dims`, ...) are documented in `events.py`.

Decompose heterogeneous signals into homogeneous events: a 7-DoF kinematic array becomes a `PositionSeries`, an `OrientationSeries` and a `GraspSeries`, not one mixed stream.

---

## 6. Loading the signal

`filepath` holds a pointer. The signal is loaded when an extractor asks for it. Three forms:

**(a) A path string**, when the Event type's built-in `.read()` can open the file directly. `DenseArrayEvent` reads `.npy` and `.npz`.

**(b) A `SpecialLoader`**, when the read needs study-specific assembly (slicing one stream out of a stacked matrix, transposing to dims-first, unit scaling, picking one trial out of a `.mat`). Serialize a reference to one of the Study's own methods, bound to the timeline plus any kwargs:

```python
filepath = study.SpecialLoader(method=self._load_stream, timeline=timeline, key="binned_spikes").to_json()

def _load_stream(self, timeline, key):
    data = utils.get_stream(self._ref(timeline), key)   # (n_samples, n_features) in the source
    return np.ascontiguousarray(data.T, dtype=np.float32)  # dims-first, as the event expects
```

**(c) A custom Event subclass overriding `_read()`**, when the same custom read is reusable across studies and should be typed.

Dense arrays are dims-first, `(n_features, n_samples)`. Transpose in the loader if the source stores time first.

---

## 7. Downloading and the data folder

Data lives under `<NEURALSHED_DATA_ROOT>/<name>/data/<StudyClass>/`. Construct the Study with the `data` folder and neuralset appends the class-name subfolder:

```python
from neuralshed.datasets.utils import dataset_dir

study = Author2025Slug(path=dataset_dir("<name>") / "data")
events = study.run()
```

`study.download()` calls your `_download(overwrite=...)`. Most sources are Dryad deposits, fetched through the shared [`datasets/dryad.py`](../src/neuralshed/datasets/dryad.py) backend, which needs `NEURALFETCH_DRYAD_TOKEN` in the environment:

```python
def _download(self, overwrite: bool = False) -> None:
    from neuralshed.datasets.dryad import Dryad

    Dryad(
        study="<name>",
        dset_dir=self.path.parent,
        folder=self.path.name,
        doi="10.5061/dryad.xxxxx",
    ).download(overwrite=overwrite)
```

The backend writes a success marker so later calls skip the transfer. Its `include`, `exclude`, `max_size` and `preview()` options select a subset of a deposit and are described in the module docstring. For DANDI sources use `neuralfetch.download.Dandi` the same way and see [dandi_nwb.md](dandi_nwb.md) for streaming. For a source with no backend, raise `NotImplementedError` with instructions to place the files under `self.path`.

---

## 8. Class-level metadata

```python
class Author2025Slug(NeuralshedStudy):
    """Summary, experimental design, data format."""

    dataset_id: tp.ClassVar[str] = "<name>"
    aliases: tp.ClassVar[tuple[str, ...]] = ("<name>", "short-alias")
    url: tp.ClassVar[str] = "https://doi.org/..."
    bibtex: tp.ClassVar[str] = """@article{...}"""
    licence: tp.ClassVar[str] = "CC0-1.0"
    description: tp.ClassVar[str] = "One paragraph."
    requirements: tp.ClassVar[tuple[str, ...]] = ("h5py",)
    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(...)
```

`dataset_id` and `_info` are required. `requirements` lists extra packages needed to read the files. No initialization override is needed.

---

## 9. Selecting data

Three levels, each with its own query:

- `Study.query` selects timelines. Virtual columns `timeline_index`, `subject_index` and `subject_timeline_index` are available, plus any timeline-dict key.
- An extractor's `event_types` selects the kind of event, and a dense extractor's `query` selects one stream among events of that kind, for example `query="feature == 'lfp_12_19'"` when several `BandPower` events overlap.
- A `Segmenter`'s `trigger_query` selects the events that become examples, for example `type == 'Action'`.

So a stream is identified by its type plus its attributes, and a condition is identified by event attributes. Thin homogeneous subclasses are cheap when two streams must be selectable by type alone.

---

## 10. Registration

Importing `study.py` registers the class with neuralset under its class name, which is the public id used in `Study(name=...)`, in `filepath` serialization and in the `subject`/`study` columns. Convention: `Author2025Slug`. Names starting with `_` are not registered, which is how `_NeuralshedStudy` stays out of the catalog.

Three project registrations are manual:

1. Import the class in [`neuralshed/studies.py`](../src/neuralshed/studies.py), so collection configs naming it resolve.
2. Add any new event class to `vocab.EVENT_CLASSES` in [`database/vocab.py`](../src/neuralshed/database/vocab.py).
3. Register any new `unit` string in [`units.py`](../src/neuralshed/units.py). `tests/test_units.py` fails on an unregistered one.

---

## 11. `StudyInfo`

`_info` records the expected ground truth: `num_timelines`, `num_subjects`, a `query`, and the `num_events_in_query`, `event_types_in_query` and `frequency` of that subset. At runtime `.run()` aborts when the timeline count differs from `num_timelines`, which catches a partial download. There is no neuralshed test that recomputes `_info` automatically today, so verify it yourself after any change to `iter_timelines` or `_load_timeline_events`.

Pick a `query` whose timelines exercise every emission path, for example one timeline per task kind. Compute the values from the data:

```python
from neuralfetch.utils import update_source_info
from neuralshed.datasets.utils import dataset_dir
import neuralshed.studies  # noqa: F401, registers every Study listed there

update_source_info("Author2025Slug", folder=dataset_dir("<name>") / "data")
```

The class must be registered before the call, so do the `studies.py` import of section 10 first. The helper rewrites the `_info` block in `study.py` and formats the file. It writes neither `query` nor `frequency`, so re-add both by hand after it runs. This is a known issue we flagged [here](https://github.com/facebookresearch/neuroai/issues/198).

---

## 12. Skeleton

```python
import typing as tp

import numpy as np
import pandas as pd
from neuralset.events import study

from neuralshed.datasets.study import NeuralshedStudy
from neuralshed.events import SpikeCounts  # noqa: F401, registers the type

from . import utils

_RATE_HZ = utils.FS


class Author2025Slug(NeuralshedStudy):
    """Summary, experimental design, data format."""

    dataset_id: tp.ClassVar[str] = "<name>"
    aliases: tp.ClassVar[tuple[str, ...]] = ("<name>",)
    url: tp.ClassVar[str] = "https://doi.org/..."
    bibtex: tp.ClassVar[str] = """@article{...}"""
    licence: tp.ClassVar[str] = "CC0-1.0"
    description: tp.ClassVar[str] = "..."
    requirements: tp.ClassVar[tuple[str, ...]] = ("h5py",)
    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=...,
        num_subjects=...,
        query="timeline_index < 1",
        num_events_in_query=...,
        event_types_in_query={"SpikeCounts", "Action"},
        frequency=_RATE_HZ,
    )

    def _download(self, overwrite: bool = False) -> None:
        from neuralshed.datasets.dryad import Dryad

        Dryad(
            study="<name>",
            dset_dir=self.path.parent,
            folder=self.path.name,
            doi="10.5061/dryad.xxxxx",
        ).download(overwrite=overwrite)

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        # One dict per timeline: identity, whole-timeline labels, one file reference.
        for ref in utils.list_sessions(self.path):
            yield {
                "subject": ref.subject,       # a label declared in metadata.py
                "session": ref.label,         # equals metadata.sessions()'s label
                "path": ref.path.name,
                "task_condition": "attempted",
            }

    def _load_neural(self, timeline: dict[str, tp.Any]) -> np.ndarray:
        # SpecialLoader target: (n_channels, n_bins) float32
        data = utils.get_counts(self.path / timeline["path"])
        return np.ascontiguousarray(data.T, dtype=np.float32)

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        source = self.path / timeline["path"]
        duration = utils.n_bins(source) / _RATE_HZ
        rows: list[dict[str, tp.Any]] = [
            {
                "type": "SpikeCounts",
                "feature": "threshold_crossings",
                "start": 0.0,
                "duration": duration,
                "frequency": _RATE_HZ,
                "n_features": utils.N_CHANNELS,
                "unit": "counts/bin",
                "normalization": "raw",
                "processing": "",
                "method": "threshold_crossing",
                "threshold": "-4.5 x RMS",
                "filepath": study.SpecialLoader(method=self._load_neural, timeline=timeline).to_json(),
            }
        ]
        for trial in utils.trials(source).itertuples():
            rows.append(
                {
                    "type": "Action",
                    "start": trial.start_bin / _RATE_HZ,
                    "duration": (trial.stop_bin - trial.start_bin) / _RATE_HZ,
                    "description": "...",
                    "code": int(trial.condition),
                    "trial_id": int(trial.trial_id),
                }
            )
        return pd.DataFrame(rows)
```

Everything named `utils.*` is a dataset-specific accessor you write in `utils.py`. The rest is the contract.

---

## 13. Gotchas

- `subject` is mandatory in the timeline dict, and every value must be a subject label in `metadata.py` with a `participant_id`.
- Do not write `subject`, `timeline`, `study`, `stop`, `dataset_id` or `participant_id` as event columns.
- BIDS entities (`session`, `task`, `run`) must be invariant within one timeline.
- Dense rows without `n_features`, `frequency`, `duration`, `unit` or `normalization` fail validation or fail later in the extractor. Neural streams must set `normalization` explicitly.
- A dense array is `(n_features, n_samples)`. Transpose time-first sources in the loader.
- Several overlapping events of one dense type are legal (one per band, one per array). Give each a distinct `feature` so extractors can select one.
- Assert dataset integrity (shapes, counts, alignment of cues to the timebase) and raise informative errors rather than emitting wrong events.

---

## 14. Authoring sequence

1. Fill class metadata: `dataset_id`, `url`, `bibtex`, `licence`, `description`, `requirements`.
2. Implement `_download` and get the source data on disk to develop against.
3. Implement `iter_timelines`. Decide what a timeline is and which whole-timeline labels ride on the dict.
4. Implement the `_load_<signal>` loaders and the rows in `_load_timeline_events`, neural streams first, then task events with the attributes you will select on.
5. Do the three registrations of section 10.
6. Run `update_source_info` (section 11), re-add `query` and `frequency`, and check `study.run()` on the query subset reads through the extractors with the expected shapes and units.
