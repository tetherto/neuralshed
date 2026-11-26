"""singer_2025: download and truncated last trials against the drive."""

import pytest

from neuralshed.datasets.singer_2025 import utils
from neuralshed.datasets.singer_2025.study import Singer2025CursorControl
from neuralshed.datasets.utils import dataset_dir

ROOT = dataset_dir("singer_2025") / "data"


def _drive():
    if not (ROOT / utils.DATA_SUBDIR).exists():
        pytest.skip(f"singer_2025 data not mounted at {dataset_dir('singer_2025')}")


# --- Download --------------------------------------------------------------------
def test_download_forwards_overwrite(tmp_path, monkeypatch):
    from neuralshed.datasets import dryad

    calls = []
    study = Singer2025CursorControl(path=tmp_path)

    def fake_download(self, **kwargs):
        # the base download() refuses an empty study folder
        calls.append(kwargs)
        (study.path / "README.md").write_text("")

    monkeypatch.setattr(dryad.Dryad, "download", fake_download)
    study.download(overwrite=False)
    study.download(overwrite=True)
    assert [c.get("overwrite") for c in calls] == [False, True]


# --- Against the drive ----------------------------------------------------------
def test_truncated_last_trials_are_invalid():
    _drive()
    events = Singer2025CursorControl(path=ROOT).run()
    truncated = events[events.valid == False]  # noqa: E712
    assert truncated.type.value_counts().to_dict() == {"Stimulus": 30, "Action": 30}
    assert set(truncated.stimulus_kind.dropna()) == {"target_cue"}
    assert set(truncated.action_kind.dropna()) == {"cursor_control"}
    assert set(truncated.valid_description) == {"trial truncated by block end"}

    cursor = events[events.action_kind == "cursor_control"]
    last = cursor.groupby("timeline").trial_id.transform("max") == cursor.trial_id
    no_end = cursor.task != "simultaneous_speech_and_cursor"
    assert (cursor.valid == False).eq(last & no_end).all()  # noqa: E712
    assert "success" not in events.columns
