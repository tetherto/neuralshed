# neuralshed

Exploration and scaffolding for the **neuralshed** project, built to be used with [neuroai](https://github.com/facebookresearch/neuroai). 

This is an effort to assemble a large corpus of public intracortical brain–computer-interface (iBCI) and neural datasets for foundation-model training. This repo does two things:

1. **Standardize datasets** — convert each public dataset into neuroai `Study`/`Event` objects (uniform event taxonomy, streaming readers, cached extractors) so they are directly trainable and benchmarkable. ~37 datasets are catalogued.
2. **Index them** — a queryable **metadata database** that standardizes names/attributes across datasets and makes the collection searchable by modality, species, brain area, task, device, etc.

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

**Source data is not bundled.** Readers stream where possible. For local data, set the root via `NEURALSHED_DATA_ROOT` env var. 

```python
import os
from neuralshed.datasets.utils import dataset_dir

# point to your local source data root
os.environ["NEURALSHED_DATA_ROOT"] = "/path/to/source/data"

# now it can be accessed by readers via utils.dataset_dir()
dataset_path = dataset_dir("example_dataset")  # e.g. "/path/to/source/data/example_dataset"
```

**Dryad datasets need an API token.** Dryad gates all file downloads behind a personal token. Export it before running a Dryad `Study`'s `_download()`:

```bash
export NEURALFETCH_DRYAD_TOKEN="<your token>"
```

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

- **Dataset catalog** — [docs/datasets.md](docs/datasets.md): the ~37 candidate datasets by macro-task group, with source and pipeline status.
- **Metadata database** — [src/neuralshed/database/README.md](src/neuralshed/database/README.md): schema, controlled vocabularies, querying, and per-dataset validation.
- **Onboarding a dataset** — [docs/onboarding_a_new_dataset.md](docs/onboarding_a_new_dataset.md): the 3-phase workflow (comprehension → study → utilization) and the `metadata.py` contract.
- **Metadata standardization** — [docs/standardization.md](docs/standardization.md): the controlled field values and conventions (species, age, modality, brain-area HOMBA/UBERON ontologies) that `metadata.py` must follow.
- **Writing a Study** — [docs/writing_a_neuralfetch_study.md](docs/writing_a_neuralfetch_study.md): the deep mechanics of the `Study` class.
- **Per-dataset details** — each `datasets/<name>/README.md` (subjects, modalities, source layout, the Study's timelines/events).
