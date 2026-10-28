"""jude_2026_typing: block tables, keyboard codes, arrays and emitted rows against the drive."""

import hashlib

import numpy as np
import pandas as pd
import pytest
import scipy.io

from neuralshed.datasets.jude_2026_typing import utils
from neuralshed.datasets.jude_2026_typing.metadata import DATASET_METADATA
from neuralshed.datasets.jude_2026_typing.study import Jude2026Typing
from neuralshed.datasets.utils import dataset_dir

MOVEMENT_SWEEP = ("t18.2024.12.03", 2)
KEY_SWEEP = ("t18.2024.12.04", 2)
EMPTY_CUE = ("t18.2025.02.04", 5, 6)  # free typing, sentence not recorded
PAUSED_BLOCKS = {
    ("t17.2024.06.04", 9),
    ("t17.2024.06.04", 10),
    ("t17.2024.06.04", 11),
    ("t17.2024.06.05", 3),
    ("t17.2024.07.09", 4),
    ("t17.2024.07.09", 10),
    ("t17.2024.07.10", 3),
    ("t17.2024.07.10", 4),
    ("t17.2024.07.22", 4),
    ("t17.2024.07.22", 5),
    ("t18.2025.04.01", 3),
}


def _drive():
    if not utils.BLOCKS_ROOT.exists():
        pytest.skip(
            f"jude_2026_typing data not mounted at {dataset_dir('jude_2026_typing')}"
        )


def _study(path=None):
    return Jude2026Typing(path=path or dataset_dir("jude_2026_typing") / "data")


# --- Offline: tables and codes -----------------------------------------------
def test_block_tables_are_disjoint_and_complete():
    for session, blocks in utils.EXCLUDED_BLOCKS.items():
        assert not set(blocks) & set(utils.SESSION_BLOCKS.get(session, {}))
    for session, blocks in utils.NOT_DEPOSITED.items():
        assert not set(blocks) & set(utils.expected_blocks().get(session, set()))
    assert sum(map(len, utils.SESSION_BLOCKS.values())) == 160
    assert sum(map(len, utils.expected_blocks().values())) == 164
    assert sum(map(len, utils.DUPLICATE_FILES.values())) == 1
    assert len(utils.SESSION_BLOCKS) == 19


def test_loop_type_table():
    assert utils.loop_type("t18.2025.01.14", 3) == "closed"  # sentence copy
    assert utils.loop_type("t17.2024.06.04", 8) == "closed"  # free typing
    assert utils.loop_type("t17.2024.04.25", 1) == "open"  # calibration
    assert utils.loop_type(*MOVEMENT_SWEEP) == "open"
    assert utils.loop_type("t18.2025.04.01", 3) == "open"
    assert utils.loop_type("t18.2025.04.01", 13) == "closed"


def test_cue_codes_follow_the_keyboard():
    assert sorted(set(utils.CUE_CODES.values())) == list(range(31))
    for code in range(1, 31):
        cues = [c for c, v in utils.CUE_CODES.items() if v == code]
        assert len(cues) == 2 and utils.MOVEMENTS[code] in cues
    assert utils.cue_code("Q") == utils.cue_code("LEFT PINKY - Extend Upwards") == 1
    assert utils.cue_code("SPACE") == utils.cue_code("RIGHT THUMB - In to Palm") == 26
    assert (
        utils.cue_code("? (QUESTION MARK)")
        == utils.cue_code("RIGHT PINKY - Straight Downwards")
        == 20
    )
    assert utils.cue_code("DO NOTHING") == utils.cue_code("DO_NOTHING") == 0
    with pytest.raises(KeyError):
        utils.cue_code("LEFT WRIST - Extend Upwards")


def test_post_implant_day_from_the_registry():
    assert utils.post_implant_day("t17.2024.04.24") == 55
    assert utils.post_implant_day("t18.2024.12.03") == 43
    assert utils.post_implant_day("t18.2025.04.01") == 162  # the deposit says 163


def test_arrays_match_the_catalog():
    for subject in DATASET_METADATA["subjects"]:
        declared = [
            (a["name"], a["hemisphere"], a["channel_start"], a["channel_stop"])
            for a in subject["arrays"]
        ]
        expected = [
            (name, hemisphere, start, stop - 1)
            for (name, hemisphere, _), (start, stop) in zip(
                utils.ARRAYS[subject["label"]],
                utils.array_channels(subject["label"]).values(),
            )
        ]
        assert declared == expected
        for array, (_, _, area) in zip(
            subject["arrays"], utils.ARRAYS[subject["label"]]
        ):
            assert f"area {area})" in array["brain_area"]


def test_t17_arrays_are_in_the_jude_2026_speech_study_order():
    from neuralshed.datasets.jude_2026_speech import utils as speech

    assert tuple(name for name, _, _ in utils.ARRAYS["T17"]) == speech.ARRAY_ORDER


def test_channel_groups():
    t17 = utils.group_channels("T17")
    assert {k: (v.min(), v.max() + 1) for k, v in t17.items()} == {
        "6v": (0, 128),
        "55b": (128, 256),
        "6d": (256, 384),
    }
    t18 = utils.group_channels("T18", by="hemisphere")
    assert t18["left"].tolist() == list(range(256))
    assert t18["right"].tolist() == list(range(256, 384))
    assert list(utils.group_channels("T18")) == ["6d"]


def test_clock_breaks():
    step = 1e9 / utils.FS_NEURAL
    times = np.concatenate([np.arange(5) * step, 1e11 + np.arange(5) * step])
    assert utils.clock_breaks(times).tolist() == [4]
    jitter = np.arange(10) * step + np.r_[0, 0.3 * step, np.zeros(8)]
    assert utils.clock_breaks(jitter).tolist() == []


# --- Offline: synthetic deposit tree ------------------------------------------
SESSION = "t18.2024.12.04"
FILES = {
    "20241204_120000_(2).mat": 2,
    "20241204_120100_(3).mat": 3,
    "20241204_120150_(5).mat": 5,
    "20241204_120200_(5).mat": 5,
}


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A one-session deposit: blocks 2 and 5 catalogued, 3 excluded, 5 deposited twice."""
    monkeypatch.setattr(
        utils, "SESSION_BLOCKS", {SESSION: {2: "finger_movement", 5: "calibration"}}
    )
    monkeypatch.setattr(utils, "EXCLUDED_BLOCKS", {SESSION: {3: "reference block"}})
    monkeypatch.setattr(
        utils, "DUPLICATE_FILES", {SESSION: {"20241204_120200_(5).mat": "copy"}}
    )
    data_dir = tmp_path / "Jude2026Typing"
    rm = data_dir / "Data" / "t18" / SESSION / "Typing" / "RedisMat"
    rm.mkdir(parents=True)
    for name in FILES:
        _write_block(rm / name)
    (data_dir / "README.md").write_text("deposit readme\n")
    return data_dir, rm


def _write_block(path):
    scipy.io.savemat(
        str(path),
        {"binned_neural_threshold_crossings": np.zeros((2, 384), dtype=np.int16)},
    )


def test_verify_tree_accepts_a_complete_tree(tree):
    data_dir, _ = tree
    digest = hashlib.sha256((data_dir / "README.md").read_bytes()).hexdigest()
    assert utils.verify_tree(data_dir, readme_digest=digest) == []
    assert utils.verify_tree(data_dir, readme_digest="0" * 64) == [
        "README.md differs from the Dryad copy"
    ]


def test_verify_tree_flags_each_fault(tree):
    data_dir, rm = tree
    (data_dir / "README.md").unlink()
    (rm / "20241204_120000_(2).mat").unlink()
    (rm / "20241204_120200_(5).mat").rename(rm / "20241204_120300_(5).mat")
    _write_block(rm / "20241204_120400_(9).mat")
    (rm / "20241204_120100_(3).mat").write_bytes(b"not a mat file")
    problems = utils.verify_tree(data_dir)
    assert problems[:5] == [
        "README.md missing",
        f"{SESSION} missing blocks [2]",
        f"{SESSION} has uncatalogued blocks [9]",
        f"{SESSION} missing duplicate file 20241204_120200_(5).mat",
        f"{SESSION} has an unlisted second file for a block",
    ]
    assert len(problems) == 6
    assert problems[5].startswith(f"{SESSION} 20241204_120100_(3).mat unreadable")


def test_verify_tree_flags_a_missing_session(tree, monkeypatch):
    data_dir, _ = tree
    monkeypatch.setattr(
        utils,
        "SESSION_BLOCKS",
        {**utils.SESSION_BLOCKS, "t18.2024.12.05": {1: "calibration"}},
    )
    assert utils.verify_tree(data_dir) == ["t18.2024.12.05 missing"]


def test_duplicates_are_skipped_and_unlisted_ones_rejected(tree):
    data_dir, rm = tree
    blocks_root = data_dir / "Data"
    path = utils.block_path(SESSION, 5, blocks_root)
    assert path.name == "20241204_120150_(5).mat"
    _write_block(rm / "20241204_120500_(2).mat")
    with pytest.raises(ValueError, match="share a block number"):
        utils.list_blocks(SESSION, blocks_root)


def test_iter_timelines_rejects_an_unlisted_block(tree):
    data_dir, rm = tree
    study = _study(data_dir.parent)
    assert [(t["block"], t["task_family"]) for t in study.iter_timelines()] == [
        (2, "finger_movement"),
        (5, "calibration"),
    ]
    _write_block(rm / "20241204_120400_(9).mat")
    with pytest.raises(ValueError, match="neither in SESSION_BLOCKS"):
        list(study.iter_timelines())


# --- Offline: trial rows -------------------------------------------------------
def _trials(cues, starts, gos, ends):
    return pd.DataFrame(
        dict(
            cue=cues,
            trial_start_ns=starts,
            go_cue_ns=gos,
            trial_end_ns=ends,
            paused=False,
            timed_out=False,
        )
    )


def test_trial_rows_skip_pauses_and_refuse_to_span_them(tmp_path):
    study = _study(tmp_path)
    step = 1e9 / utils.FS_NEURAL
    # bins 0-99, then a 60 s pause, then bins 100-199
    times = np.concatenate([np.arange(100), 6000 + np.arange(100, 200)]) * step
    trials = _trials(
        ["the cat.", "a dog."],
        [times[10], times[120]],
        [times[20], times[130]],
        [times[60], times[180]],
    )
    rows = pd.DataFrame(study._trial_rows("sentence_copy", trials, times))
    assert "Intertrial" not in set(rows.type)  # the gap 60-120 crosses the pause
    assert rows[rows.type == "Sentence"].text.tolist() == ["the cat.", "a dog."]

    spanning = _trials(["the cat."], [times[10]], [times[20]], [times[150]])
    with pytest.raises(ValueError, match="spans a recording pause"):
        study._trial_rows("sentence_copy", spanning, times)


def test_an_empty_cue_has_no_unit_outside_free_typing(tmp_path):
    study = _study(tmp_path)
    times = np.arange(100) * 1e9 / utils.FS_NEURAL
    trials = _trials([""], [times[10]], [times[20]], [times[60]])
    rows = pd.DataFrame(study._trial_rows("free_typing", trials, times))
    assert rows.stimulus_kind.tolist() == ["movement_go_cue"]
    with pytest.raises(ValueError, match="empty cue"):
        study._trial_rows("calibration", trials, times)


# --- Against the drive ---------------------------------------------------------
def test_deposit_tree_is_complete():
    _drive()
    assert utils.verify_tree() == []


def test_loop_type_matches_block_description():
    _drive()
    silent = []
    for session, blocks in utils.SESSION_BLOCKS.items():
        for block in blocks:
            described = utils.described_loop_type(utils.block_path(session, block))
            if described is None:
                silent.append((session, block))
            else:
                assert described == utils.loop_type(session, block), (session, block)
    assert len(silent) == 14
    assert {utils.task_of(*key) for key in silent} == {"finger_movement"}


@pytest.fixture(scope="module")
def all_events():
    """Every timeline's rows, with the block's pause indices beside them."""
    _drive()
    study = _study()
    frames, breaks = [], {}
    for timeline in study.iter_timelines():
        key = (timeline["session"], timeline["block"])
        rows = study._load_timeline_events(timeline)
        rows["session"], rows["block"] = key
        rows["task_family"] = timeline["task_family"]
        frames.append(rows)
        times = utils.get_bin_times_ns(study._block_path(timeline))
        breaks[key] = utils.clock_breaks(times)
    return pd.concat(frames, ignore_index=True), breaks


def test_row_counts(all_events):
    events, _ = all_events
    assert events.type.value_counts().to_dict() == {
        "Stimulus": 5176,
        "Intertrial": 1996,
        "Sentence": 1701,
        "Action": 947,
        "SpikeCounts": 160,
        "BandPower": 160,
    }
    kinds = events[events.type == "Stimulus"].stimulus_kind.value_counts().to_dict()
    assert kinds == {
        "movement_go_cue": 2649,
        "sentence_cue": 1580,
        "movement_cue": 917,
        "rest_cue": 30,
    }


def test_no_window_spans_a_recording_pause(all_events):
    events, breaks = all_events
    assert {key for key, b in breaks.items() if len(b)} == PAUSED_BLOCKS
    assert sum(len(breaks[key]) for key in PAUSED_BLOCKS) == 49
    trial_rows = events[events.trial_id.notna()]
    for key in PAUSED_BLOCKS:
        b = breaks[key]
        rows = trial_rows[(trial_rows.session == key[0]) & (trial_rows.block == key[1])]
        start = np.round(rows.start.to_numpy() * utils.FS_NEURAL).astype(int)
        stop = start + np.round(rows.duration.to_numpy() * utils.FS_NEURAL).astype(int)
        inside = (b[None, :] >= start[:, None]) & (b[None, :] < stop[:, None] - 1)
        assert not inside.any(), key


def test_finger_rows_carry_the_keyboard_code(all_events):
    events, _ = all_events
    finger = events[events.task_family == "finger_movement"]
    units = finger[finger.type == "Action"]
    cues = finger[(finger.type == "Stimulus") & (finger.period == "delay")]
    assert len(units) == len(cues) == 947
    assert set(units.code) == set(range(31))
    assert (units.description == units.code.astype(int).map(utils.MOVEMENTS)).all()
    keyed = ["session", "block", "trial_id"]
    pairs = units.merge(cues, on=keyed, suffixes=("", "_cue"))
    assert len(pairs) == 947 and (pairs.code == pairs.code_cue).all()
    rest = units[units.code == 0]
    assert set(rest.action_kind) == {"rest"} and set(rest.task_condition) == {"na"}
    assert set(cues[cues.code == 0].stimulus_kind) == {"rest_cue"}
    for key in (MOVEMENT_SWEEP, KEY_SWEEP):
        sweep = units[(units.session == key[0]) & (units.block == key[1])]
        assert sweep.code.nunique() == 31


def test_sentence_rows(all_events):
    events, _ = all_events
    sentences = events[events.type == "Sentence"]
    assert set(sentences.task_condition) == {"attempted"}
    assert sentences.success.all() and (sentences.text.str.len() > 0).all()
    written = sentences.task_family == "handwriting"
    assert set(sentences[written].modality) == {"written"} and written.sum() == 122
    assert set(sentences[~written].modality) == {"typed"}
    free = events[events.task_family == "free_typing"]
    assert not (free.period == "delay").any()
    session, block, trial = EMPTY_CUE
    empty = events[
        (events.session == session)
        & (events.block == block)
        & (events.trial_id == trial)
    ]
    assert empty.type.tolist() == ["Stimulus", "Intertrial"]


def test_trial_ids_are_contiguous_per_block(all_events):
    events, _ = all_events
    go = events[events.stimulus_kind == "movement_go_cue"]
    for _, ids in go.groupby(["session", "block"]).trial_id:
        assert ids.tolist() == list(range(len(ids)))
    assert np.allclose(go.duration, 1 / utils.FS_NEURAL)
    assert (events[events.type == "Intertrial"].duration >= 0.1).all()


def _medians_and_correlations(power):
    x = np.log1p(np.asarray(power, dtype=np.float32))
    n = (x.shape[0] // 10) * 10
    x = x[:n].reshape(-1, 10, x.shape[1]).mean(1)
    corr = np.corrcoef(x.T)
    med = np.median(x, axis=0)
    out = []
    for g in range(x.shape[1] // utils.CHANNELS_PER_ARRAY):
        sl = slice(g * utils.CHANNELS_PER_ARRAY, (g + 1) * utils.CHANNELS_PER_ARRAY)
        sub = corr[sl, sl]
        out.append((sub[~np.eye(len(sub), dtype=bool)], med[sl]))
    return out


def test_t17_column_groups_match_jude_2026_speech_on_the_data():
    """Each 64-column group of a day-95 block best matches the same group of a speech block."""
    _drive()
    from neuralshed.datasets.jude_2026_speech import utils as speech

    if not speech.T17_DIR.exists():
        pytest.skip("jude_2026_speech data not mounted")
    field = "binned_neural_spike_band_power"
    late = ("t17.2024.05.28", 6)
    ref = speech.read_block(speech.block_path(*late), fields=[field])[field]
    ref = _medians_and_correlations(speech.to_study_order(np.asarray(ref), late[0]))
    typing = utils.read_block(utils.block_path("t17.2024.06.03", 4), fields=[field])
    groups = _medians_and_correlations(typing[field])
    sim = np.array(
        [
            [
                0.5 * np.corrcoef(sa, sb)[0, 1] + 0.5 * np.corrcoef(ma, mb)[0, 1]
                for sb, mb in ref
            ]
            for sa, ma in groups
        ]
    )
    assert sim.argmax(1).tolist() == [0, 1, 2, 3, 4, 5]


def test_info_reproduces_from_a_fresh_run():
    _drive()
    info = Jude2026Typing._info
    root = dataset_dir("jude_2026_typing") / "data"
    events = Jude2026Typing(path=root, query=info.query).run()
    assert len(events) == info.num_events_in_query
    assert set(events.type) == set(info.event_types_in_query)
    assert events.timeline.nunique() == 11
    assert events.subject.nunique() <= info.num_subjects
    neural = events[events.type.isin(["SpikeCounts", "BandPower"])]
    assert set(neural.frequency) == {info.frequency}
    assert len(list(_study().iter_timelines())) == info.num_timelines
