# neuralshed Metadata Database

A searchable index over the curated dataset collection, the queryable counterpart to the per-dataset packages. Filter datasets / subjects / sessions by modality, species, brain area, macro task, device type, source platform, and subject/session attributes, from Python or SQL.

**First version:** 
- SQLAlchemy ORM models for the schema
- SQLite by default, PostgreSQL-ready via a `DATABASE_URL` env var (psycopg driver)
- Alembic for schema versioning. It is populated from each dataset's `metadata.get_metadata()` accessor

## Layout

```
src/neuralshed/database/
  schema.py           # SQLAlchemy ORM models
  db.py               # engine/session
  vocab.py            # controlled vocabularies
  populate.py         # build catalog from each dataset's metadata.get_metadata()
  validate.py         # check a dataset still fits the schema (no running DB needed)
  query.py            # Python query API
  schema_diagram.py   # regenerate the visual diagram (schema.mmd + schema.html)
  alembic/            # migrations
  neuralshed_metadata.db  # generated SQLite (git-ignored)
```

## Schema diagram

Generated straight from the ORM metadata (so it never drifts from `schema.py`):

```bash
python -m neuralshed.database.schema_diagram   # writes schema.mmd + schema.html, syncs the block below
```

The command rewrites all three outputs, the block below included, and its output is byte-stable, so a non-empty `git diff` after running it means the diagram was out of date. `schema.mmd` is the raw source and `schema.html` is a standalone rendering (Mermaid via CDN). Every table and column, PK = primary key, FK = foreign key:

```mermaid
erDiagram
    BRAIN_AREAS |o--o{ ARRAYS : brain_area
    COLLECTIONS ||--o{ COLLECTION_DATASETS : collection
    DATASETS ||--o{ COLLECTION_DATASETS : dataset
    DATASETS ||--o{ EVENT_TYPES : dataset
    DATASETS ||--o{ SESSIONS : dataset
    DATASETS ||--o{ SUBJECTS : dataset
    PARTICIPANTS |o--o{ SUBJECTS : participant
    SUBJECTS |o--o{ SESSIONS : subject
    SUBJECTS ||--o{ ARRAYS : subject
    SUBJECTS ||--o{ BRAIN_AREAS : subject

    COLLECTIONS {
        int id PK
        string name
        text description
        text notes
        json chain
        json input_event_types
        json target_event_types
        json manifest
        string manifest_built_at
        string manifest_recipe_digest
        int manifest_builder_version
    }
    DATASETS {
        string id PK
        string title
        text description
        string source_data_platform
        string source_id
        string source_data_url
        string status
        string macro_task
        string decode_target
        text bibtex
        string publication_doi
        string code_repo
        string licence
        text notes
        string study_class
    }
    PARTICIPANTS {
        string id PK
        string label
        string namespace
        string species
        string sex
        string implant_date
        string handedness
        text description
    }
    COLLECTION_DATASETS {
        int id PK
        int collection_id FK
        string dataset_id FK
    }
    EVENT_TYPES {
        int id PK
        string dataset_id FK
        string name
        string category
        string event_class
        string signal_type
        string modality
        float bin_size_ms
        float rate_hz
        string dtype
        string task_condition
        text description
    }
    SUBJECTS {
        int id PK
        string dataset_id FK
        string participant_id FK
        string label
        string age
        string time_since_implant
        json clinical_condition
    }
    BRAIN_AREAS {
        int id PK
        int subject_id FK
        string raw_label
        string homba_term
        string homba_id
        string uberon_id
        string brodmann
        text description
    }
    SESSIONS {
        int id PK
        string dataset_id FK
        int subject_id FK
        string label
        string task
        string task_condition
        string date
        int post_implant_day
        int n_trials
        string original_split
    }
    ARRAYS {
        int id PK
        int subject_id FK
        string name
        string hemisphere
        string device_type
        int n_channels
        int channel_start
        int channel_stop
        int brain_area_id FK
        json stereotaxic_coordinates
    }
```

## Tables

Relationships:
- `Dataset 1─* Subject 1─* {Array, BrainArea}`; `Array *─1 BrainArea` (both owned by the subject);
- `Participant 1─* Subject` — the same person recorded in several datasets is one `Participant`, several `Subject` rows;
- `Dataset 1─* {EventType, Session}`;
- `Session *─1 Subject`;
- `Collection *─* Dataset` through `CollectionDataset` — a saved group of datasets to train one decoder together.

### `participants` — one row per real person (cross-dataset identity)
`id` (PK, `<namespace>/<label>`, e.g. `braingate2/T15`) · `label` (T5/T12/HumanPitt) · `namespace` (the clinical trial or lab the participant belongs to, `braingate2` / `pitt`) · `species` (common name — `human`/`mouse`/`rat`/`macaque`/`marmoset`; hard-validated) · `sex` (`M`/`F`/`U`, `U` = unknown/redacted; hard-validated) · `implant_date` (full ISO date `YYYY-MM-DD`, day precision; hard-validated; assumes one implant per person) · `description`. Only person-invariant facts live here — `species`, `sex` and `implant_date` included, since a person cannot have one value in one dataset and another elsewhere. What varies per dataset (age, time since implant) stays on `subjects`. The participant records are the source of truth in [`../datasets/participants.py`](../datasets/participants.py); the dataset-to-person mapping is co-located with each subject as a `participant_id` reference in that dataset's `metadata.py`. `populate` upserts one participant row per referenced id and warns on a subject that declares no reference or references an id not defined in the registry. Query via `query.participants()`.

The BrainGate2 (NPTL, NCT00912041) `T*` labels are a shared identity across datasets, so `T15` in `card_2024`, `wairagkar_2025`, and `kunz_2025` is one person. The Pitt/FALCON-H1 participant uses a separate numbering (`HumanPitt`) and is unrelated.

### `datasets` — one row per dataset (provenance / scope)
| column | type | notes |
|---|---|---|
| `id` (PK) | str | snake_case slug, e.g. `card_2024` |
| `title`, `description` | str | |
| `source_data_platform` | str | `dandi` / `dryad` / `zenodo` / `osf` / `gin` / `other` |
| `source_id` | str | accession / DOI within the platform (e.g. `000954`, `doi:10.5061/dryad.…`) |
| `source_data_url` | str | resolved full URL to the source data |
| `status` | str | `not_started` / `onboard_wip` / `onboard_ready` / `study_wip` / `study_ready` / `ml_wip` / `ml_ready` |
| `macro_task` | str | `human_communication` / `human_motor` |
| `decode_target` | str | what a decoder trained on this dataset outputs: `text` / `voice` / `movement` / `other`. Names the *output* (attempted-speech-to-text is `text`, not `voice`); the axis that decides which datasets can pool into one decoder. |
| `bibtex` | str | raw BibTeX entry (the canonical `Study.bibtex`) |
| `publication_doi`, `code_repo` | str | references (`publication_doi` denormalized from the bibtex `doi=`) |
| `licence`, `notes` | str | |
| `study_class` | str | the dataset's neuralset `Study` subclass, e.g. `Card2024Brain2Text` — the `name` discriminator a collection config uses to name this member |

### `subjects` — one row per subject within a dataset
One dataset's view of a participant, at the time of that dataset's recordings.

`id` (PK) · `dataset_id` (FK) · `participant_id` (FK → `participants`; the cross-dataset person, `None` if unmapped) · `label` (T5/T12/…) · `age` (ISO-8601 duration, e.g. `P45Y`) · `time_since_implant` (ISO-8601 duration from implant to this dataset's first session, e.g. `P25D`) · `handedness` · `clinical_condition` (**JSON** `list[str]`). Unique on `(dataset_id, label)`.

Both durations are `None` when the source publishes no timing, and non-ISO values are rejected rather than coerced. `species`, `sex` and `implant_date` are not here: they are person-invariant and live on `participants`. `query.subjects()` joins them back in, so a subject with no `participant_id` reports them as null.

The two durations are the only per-dataset temporal facts, so `participant.implant_date + subject.time_since_implant` is the dataset's first session date.

### `arrays` — recording arrays / implants
`id` (PK) · `subject_id` (FK) · `name` · `device_type` (`utah_array` / `silicon_probe` / `ecog_grid` / `microwire_array` / `seeg_electrode` / `other` — count-agnostic) · `n_channels` · `channel_start` · `channel_stop` · `brain_area_id` (FK) · `stereotaxic_coordinates` (**JSON**, optional). Unique on `(subject_id, name)`.

### `brain_areas` — a subject's implant region(s), owned by `Subject`
`id` (PK) · `subject_id` (FK) · `raw_label` (as named by the source) · `homba_term` · `homba_id` (BICAN HOMBA id, e.g. `HOMBA_10163`) · `uberon_id` (UBERON id, e.g. `UBERON_0001384`) · `brodmann` · `description`. The ontology fields are nullable (filled during HOMBA/UBERON harmonization); `homba_id`/`uberon_id` are prefix-validated. Unique on `(subject_id, raw_label)`.

### `event_types` — every kind of event a dataset emits
neuroai unifies signals and task events under one `Event` hierarchy, so this one table holds both. `id` (PK) · `dataset_id` (FK) · `name` (the source's own label) · `category` (`data` / `text` / `categorical` — the neuroai base family) · `event_class` (nullable; the concrete neuroai subclass, e.g. `BandPower` / `Sentence` / `Stimulus` — the bridge to a future `populate_from_study()`). Data-stream attributes apply when `category == "data"`: `signal_type` (canonical: `threshold_crossings` / `spike_band_power` / `sorted_spikes` / `lfp` / `calcium_fluorescence` / `emg` / `ecog` / `kinematics` / `audio` / `acoustic_features` / `synthesized_audio` / `other`) · `modality` (`intracortical_microelectrode` / `ecog` / `eeg` / `lfp` / `calcium_imaging` / `emg` / `audio` / `behavior` / `derived` / `other`) · `bin_size_ms` · `rate_hz` · `dtype`. The behavioral `task_condition` (`attempted` / `mimed` / `imagined` / `listening` / `read` / `typed` / `na` / `other`) applies to `text`/`categorical` events. Plus `description`. Unique on `(dataset_id, name)`.

### `sessions` — per-session rows (populated only when the data drive is mounted)
`id` (PK) · `dataset_id` (FK) · `subject_id` (FK) · `label` · `task` · `task_condition` · `date` · `post_implant_day` · `n_trials` · `original_split` (the paper's own split label, e.g. FALCON held-in/held-out). Unique on `(dataset_id, label)`.

### `collections` / `collection_datasets` — saved poolable groups (many-to-many)
The persisted form of a decoder's data spec: a group of datasets assembled to train one decoder together, built in the collection builder app.
- `collections`: `id` (PK) · `name` (unique) · `description` · `notes` · `chain` (the composed `CombineStudies` step chain, carrying no paths). Then two groups that differ in kind. *Declared*: `input_event_types` / `target_event_types`, neuroai event class names saying what the collection reads and what it decodes. Naming the target class is what says which decoder head this is, more precisely than the dataset-level `decode_target` family would. *Resolved*: `manifest` (the layout table, a rebuildable cache) with `manifest_built_at` / `manifest_recipe_digest` / `manifest_builder_version`.
- `collection_datasets` (association object, `Collection` ↔ `Dataset`): `id` (PK) · `collection_id` (FK) · `dataset_id` (FK). Unique on `(collection_id, dataset_id)`. Per-study filters live on the study step of the collection chain (`Study.query`), not here.

Read/write via [`collections.py`](collections.py) (`save_collection` / `list_collections` / `get_collection` / `load_chain` / `delete_collection`), **not** `query.py` (which stays read-only over the catalog). `load_chain` resolves each member's path against `NEURALSHED_DATA_ROOT` and returns a runnable `Chain`.

Saving is cheap and opens no data file. The manifest is a separate call because it runs the chain: `manifest_status` (`not_built` / `stale` / `built`, derived from a digest rather than stored) · `build_manifest` · `get_manifest`.

## Controlled vocabularies (`vocab.py`)

Allowed values for `source_data_platform`, `status`, `macro_task`, `decode_target`, `device_type`, `hemisphere`, `category`, `event_class`, `signal_type`, `modality`, and `task_condition` live in one place and are **soft-enforced**: `populate`/`validate` print a warning for an unknown value but do not fail, so a new dataset with an unforeseen value ingests without a migration. Promote to hard Postgres `ENUM`/`CHECK` once the vocabularies stabilize.

Human-readable conventions for these and the **hard-validated** fields not in `vocab.py` — `species` (common names), `sex` (`M`/`F`/`U`), `implant_date` (ISO date), `age` / `time_since_implant` (ISO-8601 durations), and the brain-area HOMBA/UBERON harmonization — live in [`standardization.md`](../../../docs/standardization.md).

## Validating a dataset (no database needed)

Check that a dataset's declarations still fit the schema — useful whenever a `metadata.py` changes, before the real database is even running. Each dataset is populated into a throwaway **in-memory SQLite** built from the current ORM; it *validates* if it ingests without error (vocab mismatches are reported as soft warnings).

```bash
python -m neuralshed.database.validate                       # all datasets (exit 1 if any fail)
python -m neuralshed.database.validate --datasets card_2024  # a subset
python -m neuralshed.database.validate --strict              # vocab warnings count as failures
```

```python
from neuralshed.database.validate import validate_dataset
res = validate_dataset("wairagkar_2025")
res.ok, res.warnings, res.counts
```

## Running it

All commands use the project env interpreter and run from the **repo root**.

```bash
# create the schema (versioned, via Alembic)
alembic -c src/neuralshed/database/alembic.ini upgrade head

# populate the catalog (+ sessions when the seagate data drive is mounted)
python -m neuralshed.database.populate
#   --catalog-only        skip session-level population (no drive needed)
#   --datasets a,b        only these datasets
#   --db <sqlalchemy-url> target a specific database
```

`populate` is idempotent (re-running leaves row counts unchanged). It also calls `create_all()` as a convenience, so step 1 is optional for a quick start — but Alembic is the source of truth for schema changes.

### Choosing the backend
- **SQLite (default):** `src/neuralshed/database/neuralshed_metadata.db`.
- **PostgreSQL:** `export DATABASE_URL="postgresql+psycopg://user:pass@host:5432/neuralshed"` then run the same `alembic upgrade head` + `populate`. The schema and queries are backend-agnostic.

## Querying

```python
from neuralshed.database import query

query.catalog_summary()                                  # one row per dataset, rolled-up counts
query.datasets(signal_type="spike_band_power")           # datasets with SBP (source label or canonical)
query.datasets(macro_task="human_communication", species="human")
query.datasets(decode_target="text")                     # datasets a text decoder can pool
query.datasets(source_data_platform="dandi")             # all DANDI datasets
query.datasets(brain_area="motor cortex")                # by brain-area label (or HOMBA term)
query.subjects(species="human")
query.sessions(dataset="card_2024")                      # per-session rows
query.sessions(original_split="held-out-calib")          # FALCON held-out sessions
query.event_types(dataset="wairagkar_2025")              # everything the dataset emits
query.event_types(dataset="wairagkar_2025", category="data")  # just the recorded signal streams
query.brain_areas()                                      # areas + how many datasets/arrays use each

# Direct SQL (ad-hoc analytics) — returns a DataFrame:
query.read_sql("SELECT macro_task, COUNT(*) n FROM datasets GROUP BY macro_task")
```

All query helpers return pandas DataFrames.

### Collections

Saved poolable groups, the persisted form of a `build_pooled_dataset` spec. `target_event_types` says what the group's decoder outputs, and so which head this is. Writes live in `collections.py` (not `query.py`).

```python
from neuralshed.database import collections

collections.save_collection(
    collection_name="speech-text",
    members=["card_2024", "willet_2023", "fan_2023"],
    input_event_types=("SpikeCounts", "BandPower"),   # wanted if present
    target_event_types=("Sentence",),                 # required of every member
)
collections.list_collections()   # one row per collection, members rolled up
collections.get_collection(collection_name="speech-text")     # fields + member ids + chain
collections.load_chain(collection_name="speech-text")         # runnable Chain, paths resolved
collections.delete_collection(collection_name="speech-text")
```

The declared classes are checked against the `event_types` catalog before the row is written, which needs no data drive, so a member that cannot supply the target is refused at save time rather than showing up as an empty extraction later. Resolving what each recording actually contains is the separate, slower `build_manifest(collection_name="speech-text")`.

## Adding / updating a dataset

Each dataset declares its metadata in `src/neuralshed/datasets/<name>/metadata.py`:
- `DATASET_METADATA: dict` — catalog-level, **static, no data drive needed** (dataset/provenance, `subjects`→`arrays`, and `events` — one list covering data streams *and* task events, each with a `category`).
- `def sessions() -> list[dict]` *(optional)* — wraps that dataset's `utils.py` enumeration (`list_sessions()` / `list_assets()` / `list_blocks()` / …); needs the data drive.
- `def get_metadata() -> dict` — the single accessor the DB layer consumes: returns `DATASET_METADATA` with the per-session rows folded in under `"sessions"` (empty when the drive isn't mounted).

Then run `python -m neuralshed.database.validate --datasets <name>` to confirm it still fits the schema, and `populate` to (re)build the catalog. `get_metadata()` is intentionally the same shape a neuroai `Study` will emit, so a future `populate_from_study()` is a drop-in (no schema change).

## Schema migrations (versioning)

```bash
# after editing schema.py:
alembic -c src/neuralshed/database/alembic.ini revision --autogenerate -m "describe change"
alembic -c src/neuralshed/database/alembic.ini upgrade head      # apply
alembic -c src/neuralshed/database/alembic.ini current   # inspect
```

`alembic/env.py` is wired to `schema.Base.metadata` and the app's `DATABASE_URL`, with `render_as_batch=True` for SQLite-safe `ALTER`s.


## Future work
- Populating from real neuroai `Study`/`Event` objects;
- per-trial rows and electrode-level quality metrics;
- HOMBA/UBERON harmonization for the remaining datasets;
- hardening the soft vocabularies into Postgres `ENUM`/`CHECK`;
- promoting the JSON columns (`clinical_condition`, `stereotaxic_coordinates`) to `JSONB` for in-field querying.
