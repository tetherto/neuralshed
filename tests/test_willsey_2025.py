"""willsey_2025: open-loop go phases are named by where they start and end."""

from collections import Counter

import pytest

from neuralshed.datasets.willsey_2025 import utils
from neuralshed.datasets.willsey_2025.study import Willsey2025FingerDecoding


def _data_dir():
    data_dir = utils.default_data_dir()
    if not data_dir.is_dir():
        pytest.skip(f"willsey_2025 data not mounted at {data_dir}")
    return data_dir


def _open_loop_conditions():
    refs = utils.list_blocks(_data_dir())
    return {
        ref.label: [condition for _, _, condition in utils.cue_windows(ref)]
        for ref in refs
        if ref.kind == utils.KIND_OPEN_LOOP
    }


def test_open_loop_returns_are_not_rest():
    for label, conditions in _open_loop_conditions().items():
        counts = Counter(conditions)
        assert counts["rest"] == 12, label
        returns = sum(n for name, n in counts.items() if name.endswith("_return"))
        assert returns == 44, label


def test_each_return_undoes_the_movement_before_it():
    for label, conditions in _open_loop_conditions().items():
        for previous, condition in zip(conditions, conditions[1:]):
            if condition.endswith("_return"):
                assert condition == f"{previous}_return", label


def test_rest_actions_are_not_finger_movements(tmp_path):
    (tmp_path / Willsey2025FingerDecoding.__name__).symlink_to(_data_dir().parent)
    study = Willsey2025FingerDecoding(path=str(tmp_path))
    timeline = next(t for t in study.iter_timelines() if t["loop_type"] == "open")
    events = study._load_timeline_events(timeline)
    actions = events[events["type"] == "Action"]
    rest = actions[actions["condition"] == "rest"]
    assert len(rest) == 12
    assert set(rest["action_kind"]) == {"rest"}
    assert set(rest["task_condition"]) == {"na"}
    moving = actions[actions["condition"] != "rest"]
    assert set(moving["action_kind"]) == {"finger_control"}


def test_finger_streams_stay_inside_the_task_stream(tmp_path):
    (tmp_path / Willsey2025FingerDecoding.__name__).symlink_to(_data_dir().parent)
    study = Willsey2025FingerDecoding(path=str(tmp_path))
    fingers = ("GraspSeries", "TargetGraspSeries")
    for timeline in study.iter_timelines():
        ref = study._ref(timeline)
        data = utils.load_block(ref)
        clock = data["neural_clock"][utils.steady_state_start(ref) :]
        events = study._load_timeline_events(timeline)
        for row in events[events["type"].isin(fingers)].itertuples():
            first = round(row.start * utils.FS)
            stop = first + round(row.duration * utils.FS)
            assert clock[first] >= data["task_clock"][0], ref.label
            assert clock[stop - 1] <= data["task_clock"][-1], ref.label


def test_a_target_of_one_is_flexion_or_abduction():
    # The paper's Methods give flexion and abduction the positive sign.
    conditions = utils.OPEN_LOOP_CONDITIONS
    assert conditions[(1.0, 0.5, 0.5, 0.5)] == "thumb_flexion-flex"
    assert conditions[(0.5, 1.0, 0.5, 0.5)] == "thumb_abduction-abduct"
    assert conditions[(0.5, 0.0, 0.5, 0.5)] == "thumb_abduction-adduct"
    assert conditions[(0.5, 0.5, 1.0, 0.5)] == "index_middle-flex"
    assert conditions[(0.5, 0.5, 0.5, 1.0)] == "ring_little-flex"


def test_download_forwards_overwrite(tmp_path, monkeypatch):
    from neuralshed.datasets import dryad

    calls = []
    study = Willsey2025FingerDecoding(path=tmp_path)

    def fake_download(self, **kwargs):
        calls.append(kwargs)
        (study.path / "README.md").write_text("")

    monkeypatch.setattr(dryad.Dryad, "download", fake_download)
    study.download(overwrite=False)
    study.download(overwrite=True)
    assert [call.get("overwrite") for call in calls] == [False, True]


def test_arrays_of_the_swapped_day_are_swapped_back():
    # Mean power is higher on columns 96-191 than on 0-95 in every block of every other
    # day, and in the source it is lower in every block of t5.2023.04.13.
    for ref in utils.list_blocks(_data_dir()):
        if ref.kind == utils.KIND_QUADCOPTER:
            continue
        power = utils.get_neural(ref)
        assert power[:, 96:].mean() > power[:, :96].mean(), ref.label
