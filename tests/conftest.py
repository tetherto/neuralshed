import os
from pathlib import Path

from neuralshed import config

REPO_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"

if REPO_ENV_FILE.is_file():
    config.configure(env_file=REPO_ENV_FILE)

# Without a data root, point at a missing folder so data-dependent tests skip.
if not os.environ.get(config.DATA_ROOT_VAR):
    os.environ[config.DATA_ROOT_VAR] = "/nonexistent/neuralshed-data"
