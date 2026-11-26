"""Data root, token and database URL resolution."""

import os

import pytest

from neuralshed import config
from neuralshed.datasets import nwb
from neuralshed.datasets.karpowicz_2024_h1 import utils as karpowicz_h1
from neuralshed.datasets.utils import data_root

OTHER_VAR = "NEURALSHED_TEST_UNRELATED"


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    for var in (*config.VARS, OTHER_VAR):
        monkeypatch.setenv(var, "")  # records the original value, or its absence
        monkeypatch.delenv(var)
    monkeypatch.setattr(config, "_sources", {})
    return tmp_path


def test_unset_data_root_raises_on_use(clean_env):
    with pytest.raises(RuntimeError, match=config.DATA_ROOT_VAR):
        data_root()


def test_dandi_datasets_stream_without_data_root(clean_env, monkeypatch):
    path = "sub-HumanPitt-held-in-calib/sub-HumanPitt-held-in-calib_ses-19250106T120811.nwb"

    def list_assets(dandiset, version, glob=None):
        assert dandiset == karpowicz_h1.DANDISET_ID
        return [{"path": path, "asset_id": "a1", "size": 1}]

    monkeypatch.setattr(nwb, "dandi_list_assets", list_assets)
    assert karpowicz_h1.default_data_dir() is None
    assert karpowicz_h1.list_assets() == [
        {"path": path, "asset_id": "a1", "size": 1, "split": "held-in-calib"}
    ]


def test_env_file_sets_only_our_variables(clean_env, monkeypatch):
    env_file = clean_env / "my.env"
    env_file.write_text(
        f"{config.DATA_ROOT_VAR}={clean_env}\n"
        f"{config.DRYAD_TOKEN_VAR}=from-file\n"
        f"{OTHER_VAR}=unrelated\n"
    )
    monkeypatch.setenv(config.DATA_ROOT_VAR, "/from/environment")
    config.configure(env_file=env_file)
    assert data_root() == clean_env
    assert config.source(config.DATA_ROOT_VAR) == str(env_file)
    assert os.environ[config.DRYAD_TOKEN_VAR] == "from-file"
    assert OTHER_VAR not in os.environ


def test_arguments_override_env_file(clean_env):
    env_file = clean_env / ".env"
    env_file.write_text(f"{config.DATA_ROOT_VAR}=/from/file\n")
    config.configure(env_file=env_file, data_root=clean_env)
    assert data_root() == clean_env
    assert config.source(config.DATA_ROOT_VAR) == "configure()"


def test_missing_env_file_raises(clean_env):
    with pytest.raises(FileNotFoundError, match="No .env file"):
        config.configure(env_file=clean_env / "absent.env")


def test_report_data_root(clean_env, capsys):
    config.report_data_root()
    assert "is not set" in capsys.readouterr().out
    with pytest.warns(UserWarning, match="does not exist"):
        config.configure(data_root=clean_env / "missing")
    config.configure(data_root=clean_env)
    assert str(clean_env) in capsys.readouterr().out


def test_show_config_hides_secrets(clean_env, capsys):
    config.configure(
        dryad_token="secret-token",
        database_url="postgresql+psycopg://user:hunter2@host/db",
    )
    config.show_config()
    out = capsys.readouterr().out
    assert "secret-token" not in out
    assert "hunter2" not in out
