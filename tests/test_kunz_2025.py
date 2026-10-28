"""kunz_2025 source invariants: channel layout, stream normalization, trial epochs."""

import os

import numpy as np
import pytest
import scipy.io as sio

from neuralshed.datasets.kunz_2025 import utils
from neuralshed.datasets.utils import dataset_dir

UINT8_TAGGED = (
    "seqRecallUninstructed3ElementArrow/t12.2023.05.16_binnedTX_spikePow_20ms_joystickSeq.mat",
    "seqRecallUninstructed3ElementLines/t12.2023.09.19_binnedTX_spikePow_20ms_lineSequenes.mat",
    "seqRecallUninstructedSingleArrow/t12.2023.05.16_binnedTX_spikePow_20ms_joystickSingle.mat",
)
COPIED_BAND_POWER = (
    "interleavedVerbalBehaviors/t12.2024.04.11_interleavedVerbalBehaviors_raw.mat"
)


def _drive():
    if not utils.DATA_DIR.exists():
        pytest.skip(f"kunz_2025 data not mounted at {dataset_dir('kunz_2025')}")


def test_an_unknown_experiment_folder_is_rejected():
    with pytest.raises(ValueError, match="unknown experiment folder"):
        utils.family_of_path("newTask/t12.2023.08.15_newTask.mat")


def test_stored_width_matches_the_declared_layout():
    _drive()
    widths = {}
    for p in utils.list_timeline_files():
        layout = utils.array_layout(p)
        assert sum(sl.stop - sl.start for _, sl in layout) == utils.n_channels(p)
        widths[p.name] = utils.n_channels(p)
    assert sum(w == 128 for w in widths.values()) == 5
    assert sum(w == 256 for w in widths.values()) == 36
    seq = utils.DATA_DIR / UINT8_TAGGED[2]
    assert utils.array_names(seq) == [
        "ventral_6v",
        "dorsal_6v",
        "ventral_44",
        "dorsal_44",
    ]
    assert utils.array_names(utils.DATA_DIR / COPIED_BAND_POWER) == [
        "ventral_6v",
        "dorsal_6v",
    ]
    t17 = (
        utils.DATA_DIR
        / "interleavedVerbalBehaviors/t17.2024.12.09_interleavedVerbalBehaviors_raw.mat"
    )
    assert [(n, sl.start) for n, sl in utils.array_layout(t17)] == [
        ("ventral_6v", 0),
        ("dorsal_6v", 64),
        ("ventral_55b", 128),
        ("dorsal_55b", 192),
    ]


@pytest.mark.parametrize(
    "rel",
    [
        UINT8_TAGGED[2],
        COPIED_BAND_POWER,
        "seqRecallSpeech/t16.2024.07.17_binnedTX_spikePow_20ms.mat",
    ],
)
def test_read_shape_equals_stored_width(rel):
    _drive()
    p = utils.DATA_DIR / rel
    tx = utils.get_neural_stream(p, "threshold_crossings")
    assert tx.shape == (utils.n_channels(p), utils.n_bins(p))


@pytest.mark.parametrize("rel", UINT8_TAGGED)
def test_uint8_tagged_counts_are_read_as_signed(rel):
    _drive()
    p = utils.DATA_DIR / rel
    assert utils.stored_dtype(p, "binnedTX") == "uint8"
    assert utils.reinterpreted_uint8(p)
    raw = sio.loadmat(str(p), variable_names=["binnedTX"])["binnedTX"]
    assert raw.dtype == np.uint8 and raw.max() == 255
    tx = utils.get_neural_stream(p, "threshold_crossings")
    assert tx.min() < 0 and abs(float(tx.mean())) < 0.5 and tx.max() <= 20


def test_raw_files_are_not_reinterpreted():
    _drive()
    p = utils.DATA_DIR / COPIED_BAND_POWER
    assert utils.stored_dtype(p, "binnedTX") == "uint8"
    assert not utils.reinterpreted_uint8(p)
    tx = utils.get_neural_stream(p, "threshold_crossings")
    assert tx.min() >= 0


def test_copied_band_power_is_flagged():
    _drive()
    p = utils.DATA_DIR / COPIED_BAND_POWER
    assert utils.band_power_invalid_reason(p)
    tx = sio.loadmat(str(p), variable_names=["binnedTX"])["binnedTX"]
    sp = sio.loadmat(str(p), variable_names=["spikePow"])["spikePow"]
    assert np.array_equal(tx, sp)
    assert all(
        utils.band_power_invalid_reason(q) == ""
        for q in utils.list_timeline_files()
        if q.name != p.name
    )


def test_trial_epochs_fit_the_recording_and_do_not_overlap():
    _drive()
    inferred = 0
    for p in utils.list_timeline_files():
        tr = utils.get_trials(p)
        total = utils.n_bins(p)
        assert (tr.delay_start_bin < tr.delay_stop_bin).all()
        assert (tr.delay_stop_bin <= tr.go_start_bin).all()
        assert (tr.go_start_bin < tr.go_stop_bin).all()
        assert (tr.go_stop_bin <= total).all()
        assert (tr.go_stop_bin.values[:-1] <= tr.delay_start_bin.values[1:]).all()
        if utils.bin_size_is_inferred(p):  # 40 ms clock, 2 s delay + 4 s go
            inferred += 1
            meta = utils.load_meta(str(p))
            assert (meta["binSizeStored"], utils.bin_size_ms(meta)) == (20, 40)
            assert set(tr.delay_duration.round(6)) == {2.0}
            assert set(tr.go_duration.round(6)) <= {3.96, 4.0}
            # one trial every 7 s: a 1 s gap after each go period
            gaps = tr.delay_start_bin.values[1:] - tr.go_stop_bin.values[:-1]
            same = tr.block.values[1:] == tr.block.values[:-1]
            assert set(gaps[same]) <= {25, 26}
    assert inferred == 5


def test_blocks_tile_every_file_and_hold_their_trials():
    _drive()
    n_segments = 0
    for p in utils.list_timeline_files():
        segs = utils.block_segments(p)
        n_segments += len(segs)
        assert segs[0].start_bin == 0 and segs[-1].stop_bin == utils.n_bins(p)
        assert all(a.stop_bin == b.start_bin for a, b in zip(segs[:-1], segs[1:]))
        tr = utils.get_trials(p)
        bounds = {s.block: (s.start_bin, s.stop_bin) for s in segs}
        assert set(tr.block) == set(bounds)
        lo = tr.block.map(lambda b: bounds[b][0])
        hi = tr.block.map(lambda b: bounds[b][1])
        assert ((tr.delay_start_bin >= lo) & (tr.go_stop_bin <= hi)).all()
    assert n_segments == 243


@pytest.mark.skipif(
    not os.environ.get("KUNZ_SLOW_TESTS"),
    reason="loads every isolated file's streams, set KUNZ_SLOW_TESTS=1",
)
def test_per_behavior_files_duplicate_the_combined_blocks():
    """Each of the 28 per-behavior files holds some blocks of its ``*_allBehaviors`` file,
    with the same trials (behavior from the file name) and the same feature matrices."""
    _drive()
    folder = utils.DATA_DIR / "isolatedVerbalBehaviors"
    subsets = [p for p in utils.list_mat_files(folder) if "allBehaviors" not in p.name]
    assert len(subsets) == 28
    columns = ["word", "behavior", "passive", "delay", "go", "go_stop"]

    def blockwise(p):
        segs = {s.block: s for s in utils.block_segments(p)}
        tr = utils.get_trials(p)
        start = tr.block.map(lambda b: segs[b].start_bin)
        tr = tr.assign(
            delay=tr.delay_start_bin - start,
            go=tr.go_start_bin - start,
            go_stop=tr.go_stop_bin - start,
        )
        return segs, tr

    for p in subsets:
        combined = p.with_name(p.name.split("_")[0] + "_allBehaviors_raw.mat")
        segs, tr = blockwise(p)
        all_segs, all_tr = blockwise(combined)
        assert set(tr.behavior) - {utils.REST} == {utils._file_behavior(p)}
        mine = all_tr[all_tr.block.isin(segs)].reset_index(drop=True)
        assert tr[["block", *columns]].equals(mine[["block", *columns]])
        for feature in utils.STREAM_KEYS:
            for b, seg in segs.items():
                whole = all_segs[b]
                np.testing.assert_array_equal(
                    utils.get_neural_stream(p, feature, seg.start_bin, seg.stop_bin),
                    utils.get_neural_stream(
                        combined, feature, whole.start_bin, whole.stop_bin
                    ),
                )


def test_recovered_t16_blocks_are_separately_centred():
    _drive()
    p = (
        utils.DATA_DIR
        / "seqRecallVerbalMemory/t16.2024.07.17_binnedTX_spikePow_20ms_verbal.mat"
    )
    segs = utils.block_segments(p)
    assert [(s.source, s.stop_bin - s.start_bin) for s in segs] == [
        ("recovered", 6977)
    ] * 4
    for seg in segs:
        tx = utils.get_neural_stream(
            p, "threshold_crossings", seg.start_bin, seg.stop_bin
        )
        assert np.abs(tx.mean(axis=1)).max() < 1e-4


def test_labels_follow_the_source_conditions():
    _drive()
    t12 = utils.get_trials(utils.DATA_DIR / COPIED_BAND_POWER)
    assert t12.behavior.value_counts().to_dict() == {
        "attempted": 140,
        "imagined_motor": 140,
        "listening": 140,
        "rest": 20,
    }
    assert set(t12[t12.behavior == "listening"].text_modality) == {"heard"}
    verbal = utils.get_trials(
        utils.DATA_DIR
        / "seqRecallVerbalMemory/t12.2023.09.28_binnedTX_spikePow_20ms_arrow.mat"
    )
    assert int(verbal.rest.sum()) == 20
    assert set(verbal[verbal.rest].task_condition) == {"na"}
    assert set(verbal[verbal.rest].cue_id) == {0}
    assert set(verbal[~verbal.rest].task_condition) == {"attempted"}
    t16 = utils.get_trials(
        utils.DATA_DIR / "isolatedVerbalBehaviors/t16.2024.03.04_allBehaviors_raw.mat"
    )
    stub = t16[t16.passive].go_stop_bin - t16[t16.passive].go_start_bin
    assert stub.value_counts().to_dict() == {1: 252, 2: 28}
    passive = t16[t16.passive & (t16.next_start_bin > t16.go_stop_bin)]
    gaps = passive.next_start_bin - passive.go_start_bin
    assert len(passive) == 275 and set(gaps) == {150, 151}
    assert t16.groupby("block").tail(1).eval("next_start_bin == go_stop_bin").all()


def test_keyword_trials_carry_the_imagined_keyword():
    _drive()
    counts = {True: 0, False: 0}
    for p in utils.list_timeline_files():
        if utils.experiment_of_path(p) != "chittychittybangbang":
            continue
        trials = utils.get_trials(p)
        prefixed = trials.sentence_text.str.startswith(utils.KEYWORD + " ")
        assert (prefixed == trials.keyword).all()
        assert (trials.keyword == trials.cue_text.str.strip().str.startswith("*")).all()
        assert not trials.sentence_text.str.contains("*", regex=False).any()
        for key, n in trials.keyword.value_counts().items():
            counts[bool(key)] += int(n)
    assert counts == {True: 344, False: 336}


def test_sentence_strategy_follows_the_paper():
    _drive()
    found = {}
    for p in utils.list_timeline_files():
        if utils.family_of_path(p) != utils.FAMILY_SENTENCE:
            continue
        found[p.stem] = set(utils.get_trials(p).behavior)
    assert found["t12.2024.12.06_chittychittybangbang_50words"] == {"imagined_motor"}
    assert found["t15.2024.03.03_imagined_50words"] == {"imagined_motor"}
    assert found["t15.2024.12.15_imagined_125kwords"] == {"imagined_hybrid"}
    assert found["t15.2025.01.05_imagined_125kwords_eval"] == {"imagined_hybrid"}
    assert found["t16.2024.12.18_imagined_125kwords"] == {"imagined_auditory"}
    assert found["t16.2024.04.29_attempted_50words"] == {"attempted"}


def test_passive_blocks_and_sequence_actions():
    _drive()
    for name, n_passive in (("t12.2023.08.15", 336), ("t16.2024.03.04", 280)):
        trials = utils.get_trials(
            utils.DATA_DIR / f"isolatedVerbalBehaviors/{name}_allBehaviors_raw.mat"
        )
        passive = trials[trials.passive]
        assert len(passive) == n_passive
        assert set(passive.behavior) == {"listening", "reading", "rest"}
        assert not set(trials[~trials.passive].behavior) & set(utils.PASSIVE_BEHAVIORS)
    kinds = {
        "seqRecallUninstructed3ElementArrow/t12.2023.05.16_binnedTX_spikePow_20ms_joystickSeq.mat": "joystick",
        "seqRecallUninstructed3ElementArrow/t16.2024.04.08_binnedTX_spikePow_20ms.mat": "drawing",
        "seqRecallSpeech/t16.2024.07.17_binnedTX_spikePow_20ms.mat": "speech",
    }
    for rel, kind in kinds.items():
        assert set(utils.get_trials(utils.DATA_DIR / rel).action_kind) == {kind}


def test_trial_rows_follow_the_task_periods():
    _drive()
    from neuralshed.datasets.kunz_2025.study import Kunz2025InnerSpeech

    instance = Kunz2025InnerSpeech(path=utils.DATA_DIR.parent)
    timelines = list(instance.iter_timelines())

    def load(index):
        events = instance._load_timeline_events(timelines[index])
        return events[~events.type.isin(["SpikeCounts", "BandPower"])]

    active = load(0)  # chitty block: delay cue, go marker, sentence
    assert active.groupby("trial_id").size().eq(3).all()
    for _, trial in active.groupby("trial_id"):
        delay, marker, unit = trial.sort_values(["start", "duration"]).itertuples()
        assert (delay.period, marker.period, unit.period) == ("delay", "go", "go")
        assert delay.start + delay.duration == pytest.approx(unit.start)
        assert marker.start == pytest.approx(unit.start)
        assert unit.type == "Sentence"
        # the pairing anchor rides on every row and names the delay start
        assert unit.pair_start == pytest.approx(delay.start)
    passive = load(66)  # T12 listening or reading block: word over the delay epoch
    assert set(passive.type) == {"Word", "Action", "Intertrial"}
    units = passive[passive.type != "Intertrial"]
    assert set(units.period) == {"presentation"}
    assert units.duration.min() >= 1.4
    counting = load(17)
    assert set(counting.type) == {"Stimulus", "Action", "Intertrial"}
    assert set(counting.stimulus_kind.dropna()) == {"counting_grid"}


@pytest.fixture(scope="module")
def all_trial_events():
    """Every timeline's events, tagged with its timeline attributes."""
    _drive()
    import pandas as pd

    from neuralshed.datasets.kunz_2025.study import Kunz2025InnerSpeech

    instance = Kunz2025InnerSpeech(path=utils.DATA_DIR.parent)
    frames = []
    for index, timeline in enumerate(instance.iter_timelines()):
        events = instance._load_timeline_events(timeline)
        events["timeline_index"] = index
        events["session"] = timeline["session"]
        events["experiment"] = timeline["experiment"]
        frames.append(events)
    return pd.concat(frames, ignore_index=True)


def test_every_trial_row_lies_inside_its_block(all_trial_events):
    dense = all_trial_events[all_trial_events.type.isin(["SpikeCounts", "BandPower"])]
    assert len(dense) == 2 * 243
    assert (dense.start == 0).all()
    block_end = dense.groupby("timeline_index").duration.first()
    assert (dense.groupby("timeline_index").duration.nunique() == 1).all()
    rows = all_trial_events[~all_trial_events.index.isin(dense.index)]
    assert len(rows) == 43189 - len(dense)
    stop = rows.start + rows.duration
    assert (rows.start >= 0).all() and (rows.duration > 0).all()
    assert (stop <= rows.timeline_index.map(block_end) + 1e-6).all()
    assert rows.trial_id.notna().all() and rows.period.notna().all()


def test_units_of_a_block_do_not_overlap(all_trial_events):
    units = all_trial_events[all_trial_events.type.isin(["Word", "Sentence", "Action"])]
    for _, block in units.groupby("timeline_index"):
        block = block.sort_values("start")
        stop = (block.start + block.duration).to_numpy()[:-1]
        assert (stop <= block.start.to_numpy()[1:] + 1e-6).all()


def test_trials_tile_their_block(all_trial_events):
    """Delay, unit and intertrial rows cover each block from the first delay onwards."""
    rows = all_trial_events
    spans = rows[
        ~rows.type.isin(["SpikeCounts", "BandPower"])
        & ~((rows.type == "Stimulus") & (rows.period == "go"))
    ]
    for _, block in spans.groupby("timeline_index"):
        block = block.sort_values("start")
        stop = (block.start + block.duration).to_numpy()[:-1]
        assert abs(block.start.to_numpy()[1:] - stop).max() < 1e-6


def test_intertrials_and_presentations(all_trial_events):
    rows = all_trial_events
    gaps = rows[rows.type == "Intertrial"]
    assert len(gaps) == 8217 and set(gaps.task_condition) == {"na"}
    returns = gaps[gaps.description == "return period"]
    assert returns.experiment.value_counts().to_dict() == {
        "seqRecallUninstructed3ElementLines": 816,
        "seqRecallVerbalMemory": 352,
        "seqRecallVisualMemory": 176,
    }
    assert returns.duration.between(1.48, 1.55).all()
    shown = rows[rows.period == "presentation"]
    assert set(shown.type) == {"Word", "Action"} and len(shown) == 1396
    t12 = shown[shown.session == "t12.2024.04.11_interleavedVerbalBehaviors_raw"]
    assert len(t12) == 140 and set(t12.behavior) == {"listening"}
    cues = rows[rows.session == "t12.2024.04.11_interleavedVerbalBehaviors_raw"]
    listened = set(t12.trial_id)
    assert not cues[(cues.type == "Stimulus") & cues.trial_id.isin(listened)].size


def test_sibling_conventions(all_trial_events):
    """Cue kinds, text modality and vocab follow jude_2026_speech and willet_2023."""
    rows = all_trial_events
    delays = rows[(rows.type == "Stimulus") & (rows.period == "delay")]
    kinds = delays.groupby("experiment").stimulus_kind.agg(set)
    assert kinds["isolatedVerbalBehaviors"] == {"word_cue", "rest_cue"}
    assert kinds["sentenceData"] == {"sentence_cue"}
    assert kinds["seqRecallSpeech"] == {"speech_delay_cue", "rest_cue"}
    text = rows[rows.type.isin(["Word", "Sentence"])]
    expected = text.task_condition.map(
        {"listening": "heard", "imagined": "imagined"}
    ).fillna("read")
    assert (text.modality == expected).all()
    assert "family" not in rows.columns and "phase" not in rows.columns
    for p in utils.list_timeline_files():
        family = utils.family_of_path(p)
        sentence = family == utils.FAMILY_SENTENCE
        assert (utils.sentence_vocab(p) != "na") == sentence
        condition = utils.file_task_condition(p)
        assert (condition is None) == (family == utils.FAMILY_WORD_CUE)


def test_emitted_labels_follow_the_conventions(all_trial_events):
    rows = all_trial_events
    words = rows[rows.type == "Word"]
    assert words.task_condition.value_counts().to_dict() == {
        "imagined": 2786,
        "listening": 1141,
        "attempted": 952,
        "mimed": 665,
        "read": 525,
    }
    assert set(words[words.task_condition == "listening"].modality) == {"heard"}
    word_cues = rows[
        (rows.type == "Stimulus")
        & (rows.period == "delay")
        & rows.experiment.isin(
            ["isolatedVerbalBehaviors", "interleavedVerbalBehaviors"]
        )
    ]
    assert set(word_cues.code) == set(range(len(utils.WORDS) + 1))
    sentences = rows[rows.type == "Sentence"]
    assert sentences.behavior.notna().all()
    rest = rows[(rows.type == "Action") & (rows.action_kind == "rest")]
    assert len(rest) == 805
    assert set(rest.code) == {0} and set(rest.task_condition) == {"na"}
    delays = rows[(rows.type == "Stimulus") & (rows.period == "delay")]
    imagined = delays[delays.task_condition == "imagined"]
    assert len(imagined) == 480
    assert set(imagined.experiment) == {"seqRecallVerbalMemory"}
    counting = rows[rows.experiment == "conjunctiveCounting"]
    assert not counting.stimulus_kind.isin(["speech_go_cue", "movement_go_cue"]).any()


def test_only_the_copied_band_power_is_invalid(all_trial_events):
    rows = all_trial_events
    invalid = rows[rows.get("valid", True) == False]  # noqa: E712
    assert set(invalid.type) == {"BandPower"}
    assert set(invalid.session) == {"t12.2024.04.11_interleavedVerbalBehaviors_raw"}
    assert len(invalid) == 10


def test_pair_start_anchors_each_trial_at_its_delay_start(all_trial_events):
    # the published builder paired spikes/power from the trial's delay start;
    # every trial row must carry exactly one such anchor per trial
    rows = all_trial_events[all_trial_events.trial_id.notna()]
    assert rows.pair_start.notna().all()
    grouped = rows.groupby(["timeline_index", "trial_id"]).pair_start.nunique()
    assert (grouped == 1).all()
    delays = rows[rows.period == "delay"]
    assert np.allclose(delays.pair_start, delays.start)
    units_go = rows[
        (rows.period == "go") & rows.type.isin(["Word", "Sentence", "Action"])
    ]
    merged = units_go.merge(
        delays[["timeline_index", "trial_id", "start"]].rename(
            columns={"start": "delay_start"}
        ),
        on=["timeline_index", "trial_id"],
    )
    assert len(merged) == len(units_go)  # every active unit has a delay cue
    assert np.allclose(merged.pair_start, merged.delay_start)
    assert (merged.pair_start <= merged.start + 1e-9).all()


def test_inferred_bin_size_reaches_the_neural_rows(all_trial_events):
    dense = all_trial_events[all_trial_events.type == "SpikeCounts"]
    assert dense.frequency.round(2).value_counts().to_dict() == {
        50.0: 149,
        100.0: 74,
        25.0: 17,
        40.0: 2,
        30.3: 1,
    }
    inferred = dense[dense.frequency == 25.0]
    assert inferred.processing.str.contains("40 ms bins inferred").all()
    assert inferred.session.str.startswith("t16").all()


def test_info_reproduces_from_a_fresh_run():
    _drive()
    from neuralshed.datasets.kunz_2025.study import Kunz2025InnerSpeech

    info = Kunz2025InnerSpeech._info
    events = Kunz2025InnerSpeech(path=utils.DATA_DIR.parent, query=info.query).run()
    assert len(events) == info.num_events_in_query
    assert set(events.type) == set(info.event_types_in_query)
    assert events.subject.nunique() <= info.num_subjects
    instance = Kunz2025InnerSpeech(path=utils.DATA_DIR.parent)
    assert len(list(instance.iter_timelines())) == info.num_timelines


SEQ_TRIALS_PER_BLOCK = {
    "t12.2023.09.28_binnedTX_spikePow_20ms_speak": [50, 40, 40, 40],
    "t16.2024.07.17_binnedTX_spikePow_20ms": [60],
    "t12.2023.05.16_binnedTX_spikePow_20ms_joystickSeq": [32] * 14 + [33] * 14,
    "t12.2023.09.19_binnedTX_spikePow_20ms_lineSequenes": [49] * 17,
    "t12.2023.05.16_binnedTX_spikePow_20ms_joystickSingle": [50] * 3,
    "t12.2023.09.28_binnedTX_spikePow_20ms_arrow": [45] * 4,
    "t12.2023.10.12_binnedTX_spikePow_20ms_verbalMemory": [45] * 4,
    "t12.2023.10.12_binnedTX_spikePow_20ms_visualMemory": [45] * 4,
}


def _sequence_files():
    return [
        p
        for p in utils.list_timeline_files()
        if utils.family_of_path(p) == utils.FAMILY_SEQUENCE
    ]


def test_every_sequence_file_has_recovered_blocks():
    _drive()
    files = _sequence_files()
    keys = {"/".join(p.parts[-2:]) for p in files}
    assert keys == set(utils.RECOVERED_SEQ_BLOCK_STOPS)
    assert sum(len(v) for v in utils.RECOVERED_SEQ_BLOCK_STOPS.values()) == 81
    for p in files:
        segs = utils.block_segments(p)
        assert {s.source for s in segs} == {"recovered"}
        assert [s.block for s in segs] == list(range(1, len(segs) + 1))
        trials = utils.get_trials(p)
        per_block = trials.groupby("block").size().tolist()
        expected = SEQ_TRIALS_PER_BLOCK.get(p.stem, [40] * 4)
        assert per_block == expected
        # a block starts where a trial's delay starts (T16: 50 bins before the go cue)
        starts = {s.start_bin for s in segs[1:]}
        if utils.bin_size_is_inferred(p):
            anchors = set(trials.go_start_bin - utils.T16_SEQ_DELAY_BINS)
        else:
            anchors = set(trials.delay_start_bin)
        assert starts <= anchors


def test_an_unlisted_sequence_file_is_rejected(monkeypatch):
    _drive()
    p = _sequence_files()[0]
    table = dict(utils.RECOVERED_SEQ_BLOCK_STOPS)
    table.pop("/".join(p.parts[-2:]))
    monkeypatch.setattr(utils, "RECOVERED_SEQ_BLOCK_STOPS", table)
    with pytest.raises(ValueError, match="RECOVERED_SEQ_BLOCK_STOPS"):
        utils.block_segments(p)


def test_enumerating_timelines_loads_no_feature_matrix(monkeypatch):
    _drive()
    from neuralshed.datasets.kunz_2025.study import Kunz2025InnerSpeech

    def forbidden(*args, **kwargs):
        raise AssertionError("a neural stream was loaded")

    monkeypatch.setattr(utils, "_load_stream", forbidden)
    monkeypatch.setattr(utils, "recover_block_stops", forbidden)
    instance = Kunz2025InnerSpeech(path=utils.DATA_DIR.parent)
    assert len(list(instance.iter_timelines())) == 243


def test_recovered_blocks_are_separately_centred():
    """Band power sums to about zero over each block, and not over a shifted block."""
    _drive()
    for p in _sequence_files():
        power = sio.loadmat(str(p), variable_names=["spikePow"])["spikePow"]
        scale = power.std(axis=0, dtype=np.float64)
        trials = utils.get_trials(p)
        shift = int(np.diff(trials.go_start_bin.to_numpy()[:2])[0])  # one trial
        for seg in utils.block_segments(p):
            block = power[seg.start_bin : seg.stop_bin].astype(np.float64)
            residual = np.abs(block.sum(axis=0) / scale).mean()
            moved = power[seg.start_bin : seg.stop_bin - shift].astype(np.float64)
            assert residual < 1.0
            assert np.abs(moved.sum(axis=0) / scale).mean() > 20 * max(residual, 0.05)


@pytest.mark.skipif(
    not os.environ.get("KUNZ_SLOW_TESTS"),
    reason="re-runs the block recovery on every seqRecall file, set KUNZ_SLOW_TESTS=1",
)
def test_offline_recovery_reproduces_the_committed_stops():
    _drive()
    for p in _sequence_files():
        key = "/".join(p.parts[-2:])
        assert utils.recover_block_stops(p) == utils.RECOVERED_SEQ_BLOCK_STOPS[key]
