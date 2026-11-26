"""karpowicz_2025: trial edges and sequence presentations follow the source."""

import pytest

from neuralshed.datasets.karpowicz_2025 import utils
from neuralshed.datasets.karpowicz_2025.study import Karpowicz2025CursorStability


def _blocks():
    data_dir = utils.default_data_dir()
    if not data_dir.is_dir():
        pytest.skip(f"karpowicz_2025 data not mounted at {data_dir}")
    return utils.list_blocks(data_dir)


def test_trial_edges_match_source_stamps():
    for ref in _blocks():
        path = str(ref.path)
        start, stop = utils._block_bounds(path)[ref.block]
        stamps = utils._file(path)["acquisition/binned_spikes/timestamps"][start:stop]
        table = utils.trials(ref)
        assert (stamps[table["start_bin"]] == table["start_time"]).all(), ref.label
        assert (stamps[table["stop_bin"]] == table["stop_time"]).all(), ref.label


def test_movement_ends_before_the_next_target():
    ref = next(r for r in _blocks() if r.session == "T11_day153" and r.block == 5)
    trial = utils.trials(ref).set_index("trial_id").loc[39]
    assert (trial.start_bin, trial.stop_bin) == (892, 924)


def test_repeated_start_is_estimated_from_the_previous_stop():
    estimated = []
    for ref in _blocks():
        table = utils.trials(ref)
        assert table["start_bin"].is_unique, ref.label
        for index in table.index[table["start_estimated"]]:
            previous, trial = table.loc[index - 1], table.loc[index]
            assert trial.start_bin == previous.stop_bin + utils.ESTIMATED_START_LAG
            estimated.append(int(trial.trial_id))
    assert sorted(estimated) == [131, 269, 283, 347]


def test_each_simon_sequence_is_shown_before_its_movements():
    for ref in _blocks():
        shown = utils.presentations(ref)
        if ref.is_calibration:
            assert shown == [], ref.label
            continue
        assert len(shown) == 16, ref.label
        table = utils.trials(ref)
        for start, stop, codes in shown:
            assert sorted(codes) == [1, 2, 3, 4], (ref.label, start)
            first = table.loc[table["start_bin"] > stop].iloc[0]
            assert first.cond_id == codes[0], (ref.label, start)


def test_sequence_presentation_is_one_stimulus():
    _blocks()
    study = Karpowicz2025CursorStability(
        path=str(utils.default_data_dir().parent), query="timeline_index == 1"
    )
    events = study.run()
    assert set(events.block) == {5} and set(events.session) == {"T11_day153"}
    cues = events.loc[events["type"] == "Stimulus"]
    assert len(cues) == 16
    cue = cues.loc[(cues["start"] * 50).round() == 1048].iloc[0]
    assert round(cue.duration * 50) == 156
    assert cue.code == 3142 and cue.stimulus_kind == "movement_cue"


def test_calibration_returns_are_cued_movements():
    for ref in _blocks():
        back = utils.returns(ref)
        if not ref.is_calibration:
            assert back == [], ref.label
            continue
        assert len(back) == {"T11_day153": 33, "T11_day427": 48}[ref.session]
        cond = utils.get_stream(ref, "cond_id")[:, 0]
        for start, stop, origin, _ in back:
            assert (cond[start:stop] == utils.CENTRE_CODE).all(), (ref.label, start)
            assert cond[start - 1] == origin and 1 <= origin <= 8, (ref.label, start)


def test_calibration_return_is_an_action():
    _blocks()
    study = Karpowicz2025CursorStability(
        path=str(utils.default_data_dir().parent), query="timeline_index == 0"
    )
    events = study.run()
    assert set(events.block) == {3} and set(events.session) == {"T11_day153"}
    actions = events.loc[events["type"] == "Action"]
    back = actions.loc[(actions["start"] * 50).round() == 1749].iloc[0]
    assert round(back.duration * 50) == 203 and back.code == 18
    assert back.action_kind == "cursor_control"
    gaps = events.loc[events["type"] == "Intertrial", "description"]
    assert not gaps.str.contains("recentred").any()


def test_download_forwards_overwrite(tmp_path, monkeypatch):
    from neuralshed.datasets import dryad

    calls = []
    study = Karpowicz2025CursorStability(path=tmp_path)

    def fake_download(self, **kwargs):
        calls.append(kwargs)
        (study.path / "README.md").write_text("")

    monkeypatch.setattr(dryad.Dryad, "download", fake_download)
    study.download(overwrite=False)
    study.download(overwrite=True)
    assert [call.get("overwrite") for call in calls] == [False, True]
