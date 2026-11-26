# neuralshed

Exploration and scaffolding for the **neuralshed** project, built to be used with [neuroai](https://github.com/facebookresearch/neuroai). 

This is an effort to assemble a large corpus of public intracortical brain–computer-interface (iBCI) and neural datasets for foundation-model training. This repo does two things:

1. **Standardize datasets**: convert each public dataset into neuroai `Study`/`Event` objects (uniform event taxonomy, streaming readers, cached extractors) so they are directly trainable and benchmarkable. ~37 datasets are catalogued.
2. **Index them**: a queryable **metadata database** that standardizes names/attributes across datasets and makes the collection searchable by modality, species, brain area, task, device, etc.

Each dataset lives self-contained in `src/neuralshed/datasets/<name>/`; 

Reusable Events/Extractors and the metadata DB live at the package level, at:
- `src/neuralshed/events.py`
- `src/neuralshed/extractors.py`
- `src/neuralshed/database/`

## Installation

Python **3.12+**. The project runs in a conda env named `tether`:

```bash
conda env create -f environment.yaml   # creates `tether`, pip-installs this repo as -e .[all]
conda activate tether
```

Into an existing environment instead:

```bash
pip install -e ".[all]"                 # drop [all] to skip jupyterlab / ipykernel
```

## Source data

**Source data is not bundled.** Readers stream where possible. Local data lives under one root folder, set by `NEURALSHED_DATA_ROOT`.

| Variable | Needed for |
|---|---|
| `NEURALSHED_DATA_ROOT` | Reading source data. Each dataset lives in `<root>/<name>/`. |
| `NEURALFETCH_DRYAD_TOKEN` | Downloading Dryad datasets, which require a personal API token. |
| `NEURALSHED_DATABASE_URL` | Pointing the metadata database at PostgreSQL. Defaults to the local SQLite file. |

Export the variables, or keep them in a `.env` file (`cp .env.example .env`, gitignored) and load it explicitly:

```python
import neuralshed

neuralshed.configure(env_file="path/to/.env")
neuralshed.configure(data_root="/path/to/source/data")  # arguments override the file
neuralshed.show_config()  # each value and where it came from, secrets hidden
```

`configure(env_file=...)` reads only the variables above, and its values override the environment. `import neuralshed` reads no files. It prints the data root when one is set, warns when that folder is missing, and says how to set it otherwise. The repository notebooks call `configure()` in their first cell with a relative path to the repo's `.env`.

## Repository layout

```
src/neuralshed/
  datasets/<name>/        # one self-contained package per dataset (see each folder's README)
    metadata.py             # catalog metadata + get_metadata() accessor
    study.py                # the neuroai Study (timelines + events); _download() fetches source data
    utils.py                # streaming readers / accessors / constants
    0{1,2,3}_*.ipynb        # explore source · explore study · model example
  database/               # metadata database (SQLAlchemy ORM + query / validate / alembic)
  events.py               # reusable custom Events
  extractors.py           # reusable custom Extractors
docs/                     # cross-cutting guides
```

## References

- **Dataset catalog**: [docs/datasets.md](docs/datasets.md): the ~37 candidate datasets by macro-task group, with source and pipeline status.
- **Metadata database**: [src/neuralshed/database/README.md](src/neuralshed/database/README.md): schema, controlled vocabularies, querying, and per-dataset validation.
- **Onboarding a dataset**: [docs/onboarding_a_new_dataset.md](docs/onboarding_a_new_dataset.md): the 3-phase workflow (comprehension → study → utilization) and the `metadata.py` contract.
- **Standardization**: [docs/standardization.md](docs/standardization.md): the shared conventions that `metadata.py` and Study events follow (participants and subjects, arrays, event attributes, units and normalization, brain-area ontologies).
- **Writing a Study**: [docs/writing_a_study.md](docs/writing_a_study.md): the deep mechanics of the `Study` class.
- **DANDI / NWB access**: [docs/dandi_nwb.md](docs/dandi_nwb.md): shared helpers for Studies that read NWB from DANDI, from local files or streamed.
- **Channel correspondence**: [docs/channel_mapping/](docs/channel_mapping/channel_correspondence_mapping.md): per-participant evidence on whether a column is the same electrode across datasets, and the T15 trials released in more than one dataset.
- **Tutorials**: [docs/tutorials/](docs/tutorials/): notebooks on working with the standardized datasets.
- **Per-dataset details**: each `datasets/<name>/README.md` (subjects, modalities, source layout, the Study's timelines/events).
