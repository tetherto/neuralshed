import os
from pathlib import Path

# Default (overridable via ``NEURALSHED_DATA_ROOT``)
DEFAULT_DATA_ROOT = "/Volumes/seagate_0/tether/datasets"


def data_root() -> Path:
    """Return the root directory holding all per-dataset data folders."""
    return Path(os.environ.get("NEURALSHED_DATA_ROOT", DEFAULT_DATA_ROOT))


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
        the external drive is not mounted, or ``NEURALSHED_DATA_ROOT`` is unset/wrong).
    """
    d = data_root() / name
    if must_exist and not d.exists():
        raise FileNotFoundError(
            f"Data directory not found: {d}\n"
            f"Is the data drive mounted? Current data root is '{data_root()}'. "
            f"Set the NEURALSHED_DATA_ROOT environment variable to override "
            f"(default: {DEFAULT_DATA_ROOT})."
        )
    return d
