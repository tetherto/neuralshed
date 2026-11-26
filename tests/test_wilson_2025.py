"""wilson_2025: sections between recording pauses and trial windows against the drive."""

import numpy as np
import pandas as pd
import pytest

from neuralshed.datasets.utils import dataset_dir
from neuralshed.datasets.wilson_2025 import utils
from neuralshed.datasets.wilson_2025.study import Wilson2025CursorRecalibration

ROOT = dataset_dir("wilson_2025") / "data"


def test_download_forwards_overwrite(tmp_path, monkeypatch):
    from neuralshed.datasets import dryad

    calls = []
    study = Wilson2025CursorRecalibration(path=tmp_path)

    def fake_dryad(self, overwrite=False):
        calls.append((self.doi, overwrite))
        (study.path / "placeholder").write_text("")  # the base refuses an empty folder

    monkeypatch.setattr(dryad.Dryad, "download", fake_dryad)
    study.download(overwrite=True)
    assert calls == [("10.5061/dryad.1jwstqk6g", True)]


def _drive():
    if not utils.default_data_dir().exists():
        pytest.skip(f"wilson_2025 data not mounted at {dataset_dir('wilson_2025')}")


@pytest.fixture(scope="module")
def study():
    _drive()
    return Wilson2025CursorRecalibration(path=ROOT)


@pytest.fixture(scope="module")
def events(study):
    return study.run()


@pytest.fixture(scope="module")
def flagged_timelines(events):
    """Reason for each timeline flagged as a whole, by (session, block, section).

    SpikeCounts carries no flag of its own, so it holds its timeline's."""
    neural = events[events.type == "SpikeCounts"]
    neural = neural[~neural.valid.astype(bool)]
    keys = zip(neural.session, neural.block, neural.section)
    return dict(zip(keys, neural.valid_description))


@pytest.fixture(scope="module")
def valid_timelines(events, flagged_timelines):
    """Events of the timelines not flagged as a whole."""
    keys = zip(events.session, events.block, events.section)
    return events[[key not in flagged_timelines for key in keys]]


@pytest.fixture(scope="module")
def blocks(study):
    return {(r.session, r.block): r for r in utils.list_blocks(study._raw_dir)}


def test_no_timeline_holds_a_recording_pause(events, blocks):
    neural = events[events.type == "SpikeCounts"]
    sections = neural.groupby(["session", "block"]).section.agg(sorted)
    assert len(sections) == 560 and len(neural) == 1014
    assert all(s == list(range(len(s))) for s in sections)
    for (session, block), rows in neural.groupby(["session", "block"]):
        clock = utils.get_series(blocks[session, block], "system_clock").ravel()
        bins = np.rint(rows.sort_values("section").duration * utils.FS).astype(int)
        assert bins.sum() == len(clock)
        bounds = np.concatenate([[0], np.cumsum(bins)])
        for a, b in zip(bounds[:-1], bounds[1:]):
            assert set(np.diff(clock[a:b])) <= {20.0, 21.0}


def test_trial_rows_hold_one_target(events, blocks):
    trials = events[events.type == "Action"]
    assert len(trials) == 64_808
    for (session, block), rows in trials.groupby(["session", "block"]):
        ref = blocks[session, block]
        targets = utils.get_series(ref, "target_position")
        offset = dict(enumerate(a for a, _ in utils.section_windows(ref)))
        first = rows.section.map(offset) + np.rint(rows.start * utils.FS).astype(int)
        n_bins = np.rint(rows.duration * utils.FS).astype(int)
        for a, n, x, y in zip(first, n_bins, rows.target_x, rows.target_y):
            assert (targets[a : a + n] == (x, y)).all()


def test_unrecorded_array_flags_its_timelines(events, flagged_timelines):
    """A timeline with bins whose channels 0-95 were not recorded (the first NSP clock
    reads 0) carries valid=False on every row, with the unrecorded span."""
    unrecorded = {
        key: reason
        for key, reason in flagged_timelines.items()
        if key[0] not in utils.INVALID_SESSIONS
    }
    assert unrecorded == {
        ("t5.2018.08.08", 12, 0): "channels 0-95 not recorded for 0.54 of 0.54 s",
        ("t5.2018.08.08", 16, 0): "channels 0-95 not recorded for 0.54 of 0.54 s",
        ("t5.2018.08.08", 16, 1): "channels 0-95 not recorded for 5.58 of 10.02 s",
        ("t5.2018.08.08", 18, 0): "channels 0-95 not recorded for 0.54 of 0.54 s",
        ("t5.2018.08.08", 18, 1): "channels 0-95 not recorded for 8.94 of 10.02 s",
    }
    keys = zip(events.session, events.block, events.section)
    rows = events[[key in unrecorded for key in keys]]
    assert {"SpikeCounts", "Stimulus", "Action"} <= set(rows.type)
    assert not rows.valid.astype(bool).any()


def test_invalid_trials(valid_timelines):
    """Outside the timelines flagged as a whole, trials that span a recording pause carry
    valid=False on their trial rows."""
    events = valid_timelines
    trials = events[events.type.isin(["Stimulus", "Action"])]
    invalid = trials[~trials.valid.astype(bool)]
    assert len(invalid) == 4 and set(invalid.type) == {"Stimulus", "Action"}
    assert set(zip(invalid.session, invalid.block, invalid.trial_id)) == {
        ("t5.2019.03.13", 31, 1087),
        ("t5.2019.03.13", 32, 1270),
    }
    assert set(invalid.valid_description) == {"trial spans a recording pause"}
    flagged_elsewhere = ["DecodedCursorVelocitySeries", "ClickStateSeries"]
    other = events[~events.type.isin(["Stimulus", "Action", *flagged_elsewhere])]
    assert other.valid.astype(bool).all()


def test_all_zero_decoder_output_is_invalid(valid_timelines):
    rows = valid_timelines[valid_timelines.type == "DecodedCursorVelocitySeries"]
    rows = rows[rows.loop_type == "closed"]
    invalid = rows[~rows.valid.astype(bool)]
    zero_blocks = {
        ("t5.2018.01.08", 0),
        ("t5.2018.01.08", 1),
        ("t5.2018.01.08", 2),
        ("t5.2018.01.17", 2),
        ("t5.2018.01.17", 3),
        ("t5.2018.01.19", 4),
        ("t5.2018.01.22", 7),
        ("t5.2019.03.27", 11),
    }
    assert set(zip(invalid.session, invalid.block)) == zero_blocks
    valid = rows[rows.valid.astype(bool)]
    assert not set(zip(valid.session, valid.block)) & zero_blocks
    assert set(invalid.valid_description) == {
        "decoder output all zeros while the cursor moves"
    }


def test_click_rows_declare_what_they_hold(events):
    """Only the 0/1 blocks are a binary click; the 0/2 and continuous ones are flagged."""
    click = events[events.type == "ClickStateSeries"]
    blocks = click.drop_duplicates(["session", "block"])
    by_unit = blocks.groupby(["unit", blocks.valid.astype(bool)]).size().to_dict()
    assert by_unit == {
        ("binary", True): 138,
        ("a.u.", False): 55,
        ("proportion", False): 2,
    }
    years = blocks.groupby("unit").session.agg(lambda s: set(s.str[3:7])).to_dict()
    assert years == {
        "binary": {"2016", "2017"},
        "a.u.": {"2021"},
        "proportion": {"2016"},
    }


def test_cursor_trials_say_whether_they_return_to_the_centre(events):
    """Returns to the centre share the centre's code, so trial_kind tells them apart."""
    # The 2021 returns the task makes (AutoMovement) have no Action.
    for kind, n_returns in (("Stimulus", 22_973), ("Action", 19_454)):
        trials = events[events.type == kind]
        returns = (trials.target_x == 0) & (trials.target_y == 0)
        no_centre = (trials.session == "t5.2016.09.26") & (trials.block == 19)
        labelled = (trials.task == "cursor") & ~no_centre
        expected = np.where(returns[labelled], "return", "center_out")
        assert (trials.trial_kind[labelled] == expected).all()
        assert trials.trial_kind[~labelled].isna().all()
        assert (trials.trial_kind == "return").sum() == n_returns


def test_held_cursor_trials_split_at_the_go(events, blocks):
    """On the blocks that hold the cursor after the target appears, the Stimulus spans
    the hold (period="delay") and the Action starts where the cursor first moves. When
    the cursor never moves, both span the whole trial without a period."""
    cursors = {
        key: utils.get_series(ref, "cursor_position") for key, ref in blocks.items()
    }
    medians = {}
    for key, ref in blocks.items():
        holds = []
        for start, stop in utils.trial_windows(ref):
            trial = cursors[key][start:stop]
            moved = np.flatnonzero((trial != trial[0]).any(axis=1))
            holds.append(moved[0] if len(moved) else stop - start)
        if holds:
            medians[key] = np.median(holds)
    held = {
        (s, b) for s, block_ids in utils.HELD_CURSOR_BLOCKS.items() for b in block_ids
    }
    assert {key for key, median in medians.items() if median > 20} == held

    trials = events[events.type.isin(["Stimulus", "Action"])]
    inside = pd.MultiIndex.from_arrays([trials.session, trials.block]).isin(list(held))
    lockout_year = trials.session.str.startswith("t5.2021")
    assert trials[~inside & ~lockout_year].period.isna().all()
    cues = trials[inside & (trials.type == "Stimulus")]
    moves = trials[inside & (trials.type == "Action")]
    assert len(cues) == len(moves) == 1_646
    assert set(cues.period.dropna()) == {"delay"}
    assert set(moves.period.dropna()) == {"go"}
    pairs = moves.merge(
        cues, on=["session", "block", "trial_id"], suffixes=("", "_cue")
    )
    assert len(pairs) == len(moves)
    whole = pairs[pairs.period.isna()]
    assert len(whole) == 33 and whole.period_cue.isna().all()
    assert np.allclose(whole.start, whole.start_cue)
    assert np.allclose(whole.duration, whole.duration_cue)
    pairs = pairs[pairs.period.notna()]
    assert np.allclose(pairs.start, pairs.start_cue + pairs.duration_cue)
    for row in pairs.itertuples():
        offset = utils.section_windows(blocks[row.session, row.block])[row.section][0]
        cue = offset + round(row.start_cue * utils.FS)
        go = offset + round(row.start * utils.FS)
        cursor = cursors[row.session, row.block]
        assert (cursor[cue:go] == cursor[cue]).all() and (
            cursor[go] != cursor[cue]
        ).any()


def test_every_trial_outcome_is_on_an_action(events, blocks):
    """Every trial has an Action carrying the source's success, whether or not its cursor
    moves, except the 2021 returns the task makes (AutoMovement)."""
    keys = ["session", "block", "trial_id"]
    source = {}
    for (session, block), ref in blocks.items():
        for trial_id, success in zip(
            utils.trial_indices(ref), utils.trial_success(ref)
        ):
            source[session, block, trial_id] = bool(success)
    autos = set(events[events.type == "AutoMovement"].set_index(keys).index)
    actions = events[events.type == "Action"]
    emitted = dict(zip(actions.set_index(keys).index, actions.success.astype(bool)))
    assert len(emitted) == len(actions)
    assert emitted == {k: v for k, v in source.items() if k not in autos}
    assert sum(not success for success in emitted.values()) == 1_788


def test_2021_lockout(events, blocks):
    """On 2021 closed-loop trials whose decoder output opens with 25 zero bins, the task
    moves the cursor back to the centre on returns (an AutoMovement with no Action), and
    outward trials split into the lockout (delay) and T5's movement (go)."""
    keys = ["session", "block", "trial_id"]
    autos = events[events.type == "AutoMovement"]
    assert len(autos) == 3_519
    assert autos.code.isna().all() and autos.success.isna().all()
    actions = events[events.type == "Action"].set_index(keys).index
    assert not autos.set_index(keys).index.isin(actions).any()

    trials = events[
        events.type.isin(["Stimulus", "Action"])
        & events.session.str.startswith("t5.2021")
        & (events.loop_type == "closed")
    ]
    split = trials[trials.period.notna()]
    cues = split[split.type == "Stimulus"]
    moves = split[split.type == "Action"]
    assert len(cues) == len(moves) == 3_562
    assert set(cues.period) == {"delay"} and set(moves.period) == {"go"}
    assert ((cues.target_x != 0) | (cues.target_y != 0)).all()
    pairs = moves.merge(cues, on=keys, suffixes=("", "_cue"))
    assert np.allclose(pairs.start, pairs.start_cue + pairs.duration_cue)

    for (session, block), rows in pd.concat([autos, pairs]).groupby(
        ["session", "block"]
    ):
        ref = blocks[session, block]
        cursor = utils.get_series(ref, "cursor_position")
        zero = (utils.get_series(ref, "cursor_decoder_output") == 0).all(axis=1)
        offsets = dict(enumerate(a for a, _ in utils.section_windows(ref)))
        for row in rows.itertuples():
            offset = offsets[row.section]
            if row.type == "AutoMovement":
                a = offset + round(row.start * utils.FS)
                b = a + round(row.duration * utils.FS)
                assert zero[a : a + 25].all() or zero[a + 1 : a + 26].all()
                assert (cursor[a:b] != cursor[a]).any()
                assert np.linalg.norm(cursor[b - 1]) < 0.1
            else:
                cue = offset + round(row.start_cue * utils.FS)
                go = offset + round(row.start * utils.FS)
                assert go - cue in (25, 26)
                assert zero[cue + 1 : go].all() and not zero[go]


def test_original_split_is_empty_and_files_come_from_path(study, events):
    """The release folder is not a partition, so it sits in `path`, not `original_split`."""
    assert set(events.original_split) == {""}
    folders = {t["path"].split("/")[0] for t in study.iter_timelines()}
    assert folders == {"historical", "new"}


def test_array_reference_issue_day_is_invalid(events):
    """The authors' code keeps no block of t5.2019.09.18, an "array reference issue
    day", so every row of its two blocks carries valid=False."""
    day = events[events.session == "t5.2019.09.18"]
    assert set(day.block) == {31, 32}
    assert (day.type == "Action").sum() == 144
    assert not day.valid.astype(bool).any()
    assert set(day.valid_description) == {
        "array reference issue (authors' analysis code)"
    }
