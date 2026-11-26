import os
from pathlib import Path

from neuralshed.config import DATA_ROOT_HELP, DATA_ROOT_VAR


def data_root() -> Path:
    """Return the root directory holding all per-dataset data folders."""
    root = os.environ.get(DATA_ROOT_VAR)
    if not root:
        raise RuntimeError(f"{DATA_ROOT_VAR} is not set. {DATA_ROOT_HELP}")
    return Path(root)


def data_root_is_set() -> bool:
    """Whether ``NEURALSHED_DATA_ROOT`` is set, for datasets that can stream without it."""
    return bool(os.environ.get(DATA_ROOT_VAR))


def dataset_dir(name: str, must_exist: bool = False) -> Path:
    """Return a dataset's source-data directory: ``<DATA_ROOT>/<name>``.

    The dataset's source files live directly under this folder (there is no
    extra ``data_source/`` level).

    Parameters
    ----------
    name : str
        Dataset folder name, e.g. ``"willet_2021"``.
    must_exist : bool, optional
        If True, raise a clear error when the directory is missing (typically
        the external drive is not mounted, or ``NEURALSHED_DATA_ROOT`` is wrong).
    """
    d = data_root() / name
    if must_exist and not d.exists():
        raise FileNotFoundError(
            f"Data directory not found: {d}\n"
            f"Is the data drive mounted? {DATA_ROOT_VAR} is '{data_root()}'."
        )
    return d
