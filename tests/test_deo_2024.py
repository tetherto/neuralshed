import numpy as np
import pandas as pd
import pytest

from neuralshed.datasets.deo_2024 import utils
from neuralshed.datasets.deo_2024.study import Deo2024Bimanual
from neuralshed.models.data import (
    MOTOR_CURSOR_MEMBER_QUERIES_CLOSED_LOOP,
    motor_cursor_collection,
)


def test_closed_loop_collection_keeps_trials_that_end_before_timeout():
    if not utils.default_data_dir().is_dir():
        pytest.skip(f"deo_2024 data not mounted at {utils.default_data_dir()}")

    events = motor_cursor_collection(
        members=("deo_2024",),
        member_queries=MOTOR_CURSOR_MEMBER_QUERIES_CLOSED_LOOP,
    ).run()
    actions = events[events.type == "Action"]
    assert len(actions) == 1351
    move_bins = (actions.duration * 50).round().astype(int)
    assert (move_bins < 499).all()
    late = actions[move_bins.between(496, 498)]
    assert set(
        late[["session", "block", "trial_id"]].itertuples(index=False, name=None)
    ) == {
        ("t5_06_28_2021", 16, 36),
        ("t5_07_12_2021", 19, 38),
        ("t5_07_12_2021", 20, 40),
        ("t5_07_14_2021", 18, 44),
        ("t5_07_14_2021", 21, 8),
        ("t5_09_13_2021", 15, 25),
    }
    assert set(actions.success) == {True}


def test_success_is_false_on_closed_loop_timeouts():
    data_dir = utils.default_data_dir()
    if not data_dir.is_dir():
        pytest.skip(f"deo_2024 data not mounted at {data_dir}")

    events = Deo2024Bimanual(path=data_dir.parent).run()
    actions = events[events.type == "Action"]
    assert events.loc[events.type != "Action", "success"].isna().all()
    assert actions.loc[actions.loop_type == "open", "success"].isna().all()
    closed = actions[actions.loop_type != "open"]
    assert closed.success.notna().all()
    move_bins = (closed.duration * 50).round().astype(int)
    np.testing.assert_array_equal(closed.success.astype(bool), move_bins < 499)
    failed = closed[~closed.success.astype(bool)]
    assert failed.loop_type.value_counts().to_dict() == {"closed": 974, "assisted": 104}


def test_trial_cues_follow_released_delay_and_go_boundaries():
    data_dir = utils.default_data_dir()
    if not data_dir.is_dir():
        pytest.skip(f"deo_2024 data not mounted at {data_dir}")

    study = Deo2024Bimanual(path=data_dir.parent)
    trial_count = 0
    for timeline in study.iter_timelines():
        path = study.path / timeline["path"]
        start_bin = np.flatnonzero(utils.get_block_numbers(path) == timeline["block"])[
            0
        ]
        source = utils.get_trials(path)
        source = source[source.block == timeline["block"]]
        events = pd.DataFrame(study._trial_rows(timeline))
        cue = events[events.stimulus_kind == "movement_cue"]
        go = events[events.stimulus_kind == "movement_go_cue"]
        action = events[events.type == "Action"]
        expected_go = (source.go_cue_bin - start_bin) / 50
        np.testing.assert_allclose(cue.start + cue.duration, expected_go)
        assert len(cue) == len(go) == len(action) == len(source)
        np.testing.assert_array_equal(cue.trial_id, action.trial_id)
        np.testing.assert_array_equal(go.trial_id, action.trial_id)
        np.testing.assert_allclose(cue.start, (source.delay_start_bin - start_bin) / 50)
        np.testing.assert_allclose(cue.duration, source.delay_bins / 50)
        np.testing.assert_allclose(go.start, expected_go)
        np.testing.assert_allclose(go.duration, 0.02)
        np.testing.assert_allclose(action.start, expected_go)
        np.testing.assert_allclose(action.duration, source.move_bins / 50)
        assert set(cue.period) == {"delay"}
        assert set(go.period) == set(action.period) == {"go"}
        assert set(cue.modality) == {"visual"}
        assert set(go.modality) == {"audio"}
        assert set(cue.task_condition) == set(go.task_condition) == {"na"}
        assert set(action.task_condition) == {"attempted"}
        if trial_count == 0:
            assert len(study._load_timeline_events(timeline)) == 316
            assert cue.iloc[0].start == pytest.approx(0.04)
            assert cue.iloc[0].duration == pytest.approx(1.46)
            assert go.iloc[0].start == pytest.approx(1.50)
            assert action.iloc[0].duration == pytest.approx(1.24)
        trial_count += len(source)
    assert trial_count == 11044


def test_intertrials_cover_only_gaps_between_released_trials():
    data_dir = utils.default_data_dir()
    if not data_dir.is_dir():
        pytest.skip(f"deo_2024 data not mounted at {data_dir}")

    study = Deo2024Bimanual(path=data_dir.parent)
    gap_count = 0
    gap_bins = 0
    for timeline in study.iter_timelines():
        path = study.path / timeline["path"]
        start_bin = np.flatnonzero(utils.get_block_numbers(path) == timeline["block"])[
            0
        ]
        source = utils.get_trials(path)
        source = source[source.block == timeline["block"]]
        events = pd.DataFrame(study._trial_rows(timeline))
        gaps = events[events.type == "Intertrial"]
        assert len(gaps) == len(source) - 1
        expected_start = source.move_end_bin.to_numpy()[:-1]
        expected_stop = source.delay_start_bin.to_numpy()[1:]
        np.testing.assert_allclose(gaps.start, (expected_start - start_bin) / 50)
        np.testing.assert_allclose(
            gaps.start + gaps.duration,
            (expected_stop - start_bin) / 50,
        )
        assert (gaps.duration > 0).all()
        assert set(gaps.period) == {"intertrial"}
        assert set(gaps.task_condition) == {"na"}
        assert set(gaps.description) == {"between released trials"}
        np.testing.assert_array_equal(gaps.trial_id, np.arange(len(source) - 1))
        periods = events[events.stimulus_kind != "movement_go_cue"].sort_values("start")
        np.testing.assert_allclose(
            (periods.start + periods.duration).to_numpy()[:-1],
            periods.start.to_numpy()[1:],
        )
        if gap_count == 0:
            assert gaps.iloc[0].start == pytest.approx(2.74)
            assert gaps.iloc[0].duration == pytest.approx(0.56)
            block_events = study._load_timeline_events(timeline)
            assert set(block_events.type) == study._info.event_types_in_query
        gap_count += len(gaps)
        gap_bins += int((gaps.duration * 50).round().sum())
    assert gap_count == 10840
    assert gap_bins == 70003


def test_block_conditions_are_timeline_attributes():
    data_dir = utils.default_data_dir()
    if not data_dir.is_dir():
        pytest.skip(f"deo_2024 data not mounted at {data_dir}")

    study = Deo2024Bimanual(path=data_dir.parent)
    timelines = pd.DataFrame(study.iter_timelines())
    assert timelines.loop_type.value_counts().to_dict() == {
        "open": 92,
        "closed": 92,
        "assisted": 20,
    }
    settings = ["loop_type", "decoder_type", "decoder_gain", "decoder_training_data"]
    assert timelines.decoder_type.notna().sum() == 112
    assert timelines.decoder_gain.notna().sum() == 27
    assert timelines.decoder_training_data.notna().sum() == 24
    assert timelines.loc[timelines.loop_type == "open", settings[1:]].isna().all().all()
    indexed = timelines.set_index(["session", "block"])
    for session, block, decoder_type, gain in [
        ("t5_09_13_2021", 12, "linear", 0.5),
        ("t5_09_13_2021", 13, "recurrent_neural_network", 0.3),
        ("t5_09_15_2021", 31, "linear", 1.2),
        ("t5_09_15_2021", 32, "recurrent_neural_network", 0.8),
    ]:
        row = indexed.loc[(session, block)]
        assert row.decoder_type == decoder_type
        assert row.decoder_gain == pytest.approx(gain)
    assert indexed.loc[("t5_09_13_2021", 6)].loop_type == "assisted"
    assert indexed.loc[("t5_09_13_2021", 6)].decoder_type == "linear"
    assert indexed.loc[("t5_09_13_2021", 7)].decoder_type == "recurrent_neural_network"
    assert indexed.loc[("t5_09_27_2021", 7)].decoder_training_data == "augmented"
    assert indexed.loc[("t5_09_27_2021", 8)].decoder_training_data == "raw"
    assert indexed.loc[("t5_09_29_2021", 9)].decoder_training_data == "augmented"
    assert indexed.loc[("t5_09_29_2021", 8)].decoder_training_data == "raw"

    selected = Deo2024Bimanual(
        path=data_dir.parent,
        query="loop_type == 'assisted' and decoder_type == 'linear'",
    ).study_summary()
    assert set(zip(selected.session, selected.block)) == {
        ("t5_09_13_2021", 6),
        ("t5_09_13_2021", 8),
        ("t5_09_15_2021", 6),
        ("t5_09_15_2021", 7),
    }

    events = study.run()
    assert "ControlMode" not in set(events.type)
    per_block = events.groupby("timeline")[settings].nunique(dropna=False)
    assert (per_block == 1).all().all()
    block_rows = events.drop_duplicates("timeline").set_index(["session", "block"])
    for column in settings:
        pd.testing.assert_series_equal(
            block_rows[column].sort_index(),
            indexed[column].sort_index(),
            check_dtype=False,
        )

    actions = events[events.type == "Action"]
    attributes = ["movement_strategy", "declared_trial_mix"]
    assert events.loc[events.type != "Action", attributes].isna().all().all()
    assert actions.declared_trial_mix.notna().all()
    assert actions.loc[actions.movement_strategy.notna(), "timeline"].nunique() == 18
    for session, block, strategy in [
        ("t5_10_11_2021", 22, "sequential_unimanual"),
        ("t5_10_11_2021", 23, "true_bimanual"),
        ("t5_10_13_2021", 22, "sequential_unimanual"),
        ("t5_10_13_2021", 20, "true_bimanual"),
    ]:
        block_actions = actions[(actions.session == session) & (actions.block == block)]
        assert set(block_actions.movement_strategy) == {strategy}
        assert set(block_actions.declared_trial_mix) == {"unimanual_and_bimanual"}
    assert (
        len(
            actions[
                (actions.declared_trial_mix == "unimanual_only") & (actions.code == 3)
            ]
        )
        == 40
    )


def test_open_loop_collection_keeps_trial_windows_and_continuous_streams():
    if not utils.default_data_dir().is_dir():
        pytest.skip(f"deo_2024 data not mounted at {utils.default_data_dir()}")

    events = motor_cursor_collection(members=("deo_2024",)).run()
    assert len(events[events.type == "Action"]) == 4344
    assert len(events[events.type == "SpikeCounts"]) == 92
    assert len(events[events.type == "CursorPositionSeries"]) == 92
    assert len(events[events.type == "TargetPositionSeries"]) == 92
    assert set(events.loop_type) == {"open"}


def test_block_ranges_are_derived_without_timeline_offsets():
    from scipy.io import loadmat

    data_dir = utils.default_data_dir()
    if not data_dir.is_dir():
        pytest.skip(f"deo_2024 data not mounted at {data_dir}")

    study = Deo2024Bimanual(path=data_dir.parent)
    timelines = list(study.iter_timelines())
    assert len(timelines) == 204
    source_session = None
    for timeline in timelines:
        assert set(timeline) == {
            "subject",
            "session",
            "block",
            "date",
            "path",
            "loop_type",
            "decoder_type",
            "decoder_gain",
            "decoder_training_data",
        }
        if timeline["session"] != source_session:
            source = loadmat(
                study.path / timeline["path"],
                variable_names=["blockNum", "tx", "Cp", "Tp"],
            )
            source_session = timeline["session"]
        bins = np.flatnonzero(source["blockNum"].ravel() == timeline["block"])
        assert np.all(np.diff(bins) == 1)
        rows = study._load_timeline_events(timeline)
        assert {"start_bin", "stop_bin"}.isdisjoint(rows.columns)
        streams = rows[rows.filepath.notna()]
        assert len(streams) == 5
        np.testing.assert_allclose(streams.start, 0)
        np.testing.assert_allclose(streams.duration, len(bins) / 50)
        for key, field, channels in [
            ("threshold_crossings", "tx", slice(None)),
            ("right_cursor_position", "Cp", slice(0, 2)),
            ("left_cursor_position", "Cp", slice(2, 4)),
            ("right_target_position", "Tp", slice(0, 2)),
            ("left_target_position", "Tp", slice(2, 4)),
        ]:
            expected = source[field][channels, bins[0] : bins[-1] + 1]
            actual = study._load_series(timeline, key)
            assert actual.dtype == np.float32
            np.testing.assert_array_equal(actual, expected.astype(np.float32))


def test_only_visual_cues_carry_source_target_coordinates():
    from scipy.io import loadmat

    data_dir = utils.default_data_dir()
    if not data_dir.is_dir():
        pytest.skip(f"deo_2024 data not mounted at {data_dir}")

    study = Deo2024Bimanual(path=data_dir.parent)
    attributes = [
        "target_right_x",
        "target_right_y",
        "target_left_x",
        "target_left_y",
    ]
    source_session = None
    trial_count = 0
    for timeline in study.iter_timelines():
        if timeline["session"] != source_session:
            source = loadmat(
                study.path / timeline["path"],
                variable_names=["Tp", "goCue", "blockNum"],
            )
            source_session = timeline["session"]
        go_bins = source["goCue"].ravel().astype(np.int64) - 1
        in_block = source["blockNum"].ravel()[go_bins] == timeline["block"]
        expected = source["Tp"][:, go_bins[in_block]].T
        events = pd.DataFrame(study._trial_rows(timeline))
        is_cue = events.stimulus_kind == "movement_cue"
        cues = events[is_cue]
        np.testing.assert_array_equal(cues[attributes].to_numpy(), expected)
        assert events.loc[~is_cue, attributes].isna().all().all()
        assert set(cues.type) == {"Stimulus"}
        assert set(cues.modality) == {"visual"}
        actions = events[events.type == "Action"]
        np.testing.assert_array_equal(cues.trial_id, actions.trial_id)
        trial_count += len(cues)
    assert trial_count == 11044
