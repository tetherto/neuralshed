"""Runtime configuration: the data root, the Dryad token and the database URL.

Values come from the environment, or from :func:`configure`, which reads a ``.env``
file and explicit arguments. Importing ``neuralshed`` reads no files.
"""

import os
import warnings
from pathlib import Path

from dotenv import dotenv_values

DATA_ROOT_VAR = "NEURALSHED_DATA_ROOT"
DRYAD_TOKEN_VAR = "NEURALFETCH_DRYAD_TOKEN"  # name read by neuralfetch
DATABASE_URL_VAR = "NEURALSHED_DATABASE_URL"
VARS = (DATA_ROOT_VAR, DRYAD_TOKEN_VAR, DATABASE_URL_VAR)

DATA_ROOT_HELP = (
    "Export it, or call neuralshed.configure(env_file=...) "
    "or neuralshed.configure(data_root=...)."
)

#: Where each value came from: "configure()" or a ``.env`` path. Others are "environment".
_sources: dict[str, str] = {}


def configure(
    env_file: str | Path | None = None,
    data_root: str | Path | None = None,
    dryad_token: str | None = None,
    database_url: str | None = None,
) -> None:
    """Set values for this process, then report the data root.

    ``env_file`` contributes only the variables above and overrides the environment.
    Arguments override both. Set ``database_url`` before the first database call,
    since the engine is created once.
    """
    if env_file is not None:
        path = Path(env_file).resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"No .env file at {path}. Point env_file at your .env "
                "(see .env.example), or drop it if the variables are exported."
            )
        values = dotenv_values(path)
        for var in VARS:
            value = values.get(var)
            if value:
                os.environ[var] = value
                _sources[var] = str(path)
    arguments = {
        DATA_ROOT_VAR: data_root,
        DRYAD_TOKEN_VAR: dryad_token,
        DATABASE_URL_VAR: database_url,
    }
    for var, value in arguments.items():
        if value is not None:
            os.environ[var] = str(value)
            _sources[var] = "configure()"
    report_data_root()


def report_data_root() -> None:
    """Print the data root in use, or say that it is unset or missing."""
    root = os.environ.get(DATA_ROOT_VAR)
    if not root:
        print(f"neuralshed: {DATA_ROOT_VAR} is not set. {DATA_ROOT_HELP}")
    elif not Path(root).is_dir():
        warnings.warn(
            f"{DATA_ROOT_VAR} is {root}, which does not exist. "
            "Is the data drive mounted?",
            stacklevel=2,
        )
    else:
        print(f"neuralshed data root: {root} (from {source(DATA_ROOT_VAR)})")


def source(var: str) -> str:
    """Where ``var``'s current value came from."""
    return _sources.get(var, "environment")


def show_config() -> None:
    """Print each value and where it came from, with secrets hidden."""
    from sqlalchemy.engine import make_url

    for var in VARS:
        value = os.environ.get(var)
        if not value:
            shown = "(not set)"
        elif var == DRYAD_TOKEN_VAR:
            shown = f"(hidden)  [{source(var)}]"
        elif var == DATABASE_URL_VAR:
            url = make_url(value).render_as_string(hide_password=True)
            shown = f"{url}  [{source(var)}]"
        else:
            shown = f"{value}  [{source(var)}]"
        print(f"{var:<24} {shown}")
