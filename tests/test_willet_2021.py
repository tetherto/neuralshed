"""willet_2021: block timelines, trial rows, class codes and targets against the drive."""

import numpy as np
import pandas as pd
import pytest
import scipy.io as sio

from neuralset.events import etypes
from neuralshed.datasets.utils import dataset_dir
from neuralshed.datasets.willet_2021 import utils
from neuralshed.datasets.willet_2021.metadata import DATASET_METADATA
from neuralshed.datasets.willet_2021.study import Willett2021Handwriting
from neuralshed.extractors import NeuralFeatureExtractor

ROOT = dataset_dir("willet_2021") / "data"
DATASETS = ROOT / "Willett2021Handwriting" / "handwritingBCIData" / "Datasets"
BIN_S = 0.01
EMPTY_TARGET = ("t5.2020.01.15", 142)  # intendedText is an empty array
DISTANCES = ("short", "medium", "long")


def _drive():
    if not DATASETS.exists():
        pytest.skip(f"willet_2021 data not mounted at {dataset_dir('willet_2021')}")


def _source(session, task, names):
    path = DATASETS / session / f"{task}.mat"
    return sio.loadmat(str(path), variable_names=names, simplify_cells=True)


def _line_code(cue):
    _, distance, angle = cue.split("_")
    return 32 + 16 * DISTANCES.index(distance) + int(angle) - 1


# --- Offline: catalog ----------------------------------------------------------
def test_subject_age_from_the_supplement():
    (subject,) = DATASET_METADATA["subjects"]
    assert subject["age"] == "P65Y"


# --- Download --------------------------------------------------------------------
def test_download_forwards_overwrite(tmp_path, monkeypatch):
    from neuralshed.datasets import dryad

    calls = []
    study = Willett2021Handwriting(path=tmp_path)

    def fake_download(self, **kwargs):
        # the base download() refuses an empty study folder
        calls.append(kwargs)
        (study.path / "readme.pdf").write_text("")

    monkeypatch.setattr(dryad.Dryad, "download", fake_download)
    study.download(overwrite=False)
    study.download(overwrite=True)
    assert [c.get("overwrite") for c in calls] == [False, True]


# --- Against the drive ----------------------------------------------------------
@pytest.fixture(scope="module")
def events():
    _drive()
    return Willett2021Handwriting(path=ROOT).run()


def test_one_timeline_per_block(events):
    assert events.timeline.nunique() == 130
    assert not {"block_num", "trial_day"} & set(events.columns)
    assert pd.api.types.is_integer_dtype(events.block)
    assert pd.api.types.is_integer_dtype(events.post_implant_day)
    neural = events[events.type == "SpikeCounts"]
    assert len(neural) == 130
    assert round(neural.duration.sum() / BIN_S) == 6_297_018
    assert set(neural.frequency) == {100.0}
    assert (neural.processing.str.len() > 0).all()


def test_no_row_leaves_its_block(events):
    span = events[events.type == "SpikeCounts"].set_index("timeline").duration
    stop = events.start + events.duration
    assert (events.start >= 0).all()
    assert (stop <= events.timeline.map(span) + 1e-6).all()


def test_loop_type(events):
    units = events[events.period.eq("go") & (events.type != "Stimulus")]
    sentences = units[units.type == "Sentence"]
    closed = sentences.sentence_condition.str.startswith("CL")
    assert set(sentences[closed].loop_type) == {"closed"}
    assert set(sentences[~closed].loop_type) == {"open"}
    assert set(units[units.type != "Sentence"].loop_type) == {"open"}


def test_every_trial_has_a_delay_cue_and_a_go_marker(events):
    delay = events[events.period.eq("delay")]
    go = events[events.stimulus_kind.eq("movement_go_cue")]
    assert set(delay.type) == {"Stimulus"}
    assert delay.stimulus_kind.value_counts().to_dict() == {
        "character_cue": 3627,
        "movement_cue": 1152,
        "sentence_cue": 1000,
        "rest_cue": 27,
    }
    assert len(go) == 5806 and np.allclose(go.duration, BIN_S)
    assert (go.period == "go").all()


def test_trial_ids_are_contiguous_per_file(events):
    go = events[events.stimulus_kind.eq("movement_go_cue")]
    for _, rows in go.groupby(["session", "task"]):
        ordered = rows.sort_values(["block", "start"]).trial_id
        assert ordered.tolist() == list(range(len(rows)))


def test_line_codes_are_condition_classes(events):
    lines = events[(events.type == "Action") & events.action_kind.eq("drawing")]
    assert len(lines) == 1152
    assert (lines.code == lines.description.map(_line_code)).all()
    assert lines.code.value_counts().to_dict() == {c: 24 for c in range(32, 80)}


def test_go_rows_end_at_the_next_delay(events):
    units = events[events.period.eq("go") & (events.type != "Stimulus")]
    lines = units[units.action_kind.eq("drawing")]
    letters = units[units.type == "Character"]
    assert lines.duration.between(1.50, 1.53).all()
    assert letters.duration.between(1.00, 1.03).all()


def test_rest_is_an_action(events):
    rest = events[events.action_kind.eq("rest")]
    assert len(rest) == 27
    assert set(rest.type) == {"Action"} and set(rest.code) == {0}
    assert set(rest.task_condition) == {"na"} and set(rest.session) == {"t5.2019.05.08"}
    stimuli = events[events.type == "Stimulus"]
    assert set(stimuli.period) == {"delay", "go"}
    assert np.allclose(stimuli[stimuli.period == "go"].duration, BIN_S)


def test_character_codes_and_text(events):
    letters = events[events.type == "Character"]
    assert len(letters) == 3627
    expected = letters.char_class.map(lambda c: utils.CHAR_LIST.index(c) + 1)
    assert (letters.code == expected).all()
    assert letters.code.value_counts().to_dict() == {c: 117 for c in range(1, 32)}
    glyph = dict(zip(letters.char_class, letters.text))
    assert glyph["greaterThan"] == " " and glyph["tilde"] == "."
    assert not {">", "~"} & set(letters.text)


def test_sentence_targets(events):
    sentences = events[events.type == "Sentence"]
    assert len(sentences) == 999
    assert "[]" not in set(sentences.text)
    assert sentences.text.str.len().sum() == 43_501
    session, trial = EMPTY_TARGET
    cue = events[
        (events.session == session)
        & (events.task == "sentences")
        & (events.trial_id == trial)
        & events.period.eq("delay")
    ]
    assert len(cue) == 1
    assert sentences.original_split.value_counts().to_dict() == {
        "train": 502,
        "test": 228,
        "": 269,
    }


def test_excluded_sentences_are_invalid_units(events):
    sentences = events[events.type == "Sentence"]
    flagged = set()
    for session in sorted(sentences.session.unique()):
        d = _source(session, "sentences", ["excludedSentences"])
        flagged |= {(session, int(i)) for i in np.flatnonzero(d["excludedSentences"])}
    assert len(flagged) == 33
    invalid = sentences[~sentences.valid.astype(bool)]
    assert set(zip(invalid.session, invalid.trial_id.astype(int))) == flagged - {
        EMPTY_TARGET
    }
    assert (invalid.valid_description.str.len() > 0).all()
    assert events[events.type == "SpikeCounts"].valid.astype(bool).all()


def test_sentence_cue_carries_the_prompt(events):
    cues = events[events.stimulus_kind.eq("sentence_cue")]
    for session, rows in cues.groupby("session"):
        prompts = _source(session, "sentences", ["sentencePrompt"])["sentencePrompt"]
        for trial, description in zip(rows.trial_id.astype(int), rows.description):
            assert description == utils.decode_text(str(prompts[trial]))


@pytest.mark.parametrize(
    "session,task,condition",
    [
        ("t5.2019.12.11", "singleLetters", "m"),
        ("t5.2019.06.03", "straightLines", "straightLine_long_5"),
    ],
)
def test_extractor_matches_the_source_cube(events, session, task, condition):
    rows = events[(events.session == session) & (events.task == task)]
    col = "char_class" if task == "singleLetters" else "description"
    units = rows[rows[col].eq(condition) & rows.type.isin(["Character", "Action"])]
    cube = _source(session, task, [f"neuralActivityCube_{condition}"])
    cube = cube[f"neuralActivityCube_{condition}"]
    assert len(units) == cube.shape[0]
    extractor = NeuralFeatureExtractor(
        target_unit="native", event_types="SpikeCounts", frequency=100.0
    )
    # cubes hold trials in recording order
    order = [c for c in ("block", "start") if c in units.columns]
    for k, (_, unit) in enumerate(units.sort_values(order).iterrows()):
        neural = rows[(rows.timeline == unit.timeline) & (rows.type == "SpikeCounts")]
        stream = etypes.Event.from_dict(neural.iloc[0].to_dict())
        x = extractor([stream], start=unit.start - 0.5, duration=2.0).numpy()
        assert np.array_equal(x, cube[k, 0:200].T.astype(x.dtype))


def test_a_relocated_root_does_not_read_the_default_drive(tmp_path, monkeypatch):
    _drive()
    folder = tmp_path / "Willett2021Handwriting"
    folder.mkdir()
    (folder / "handwritingBCIData").symlink_to(DATASETS.parent)
    missing = tmp_path / "absent"
    monkeypatch.setattr(utils, "DATA_DIR", missing, raising=False)
    monkeypatch.setattr(utils, "SOURCE_DIR", missing / "Datasets", raising=False)
    study = Willett2021Handwriting(path=tmp_path, query="timeline_index < 1")
    events = study.run()
    assert (events.type == "SpikeCounts").sum() == 1


def test_info_reproduces_from_a_fresh_run():
    _drive()
    info = Willett2021Handwriting._info
    events = Willett2021Handwriting(path=ROOT, query=info.query).run()
    assert len(events) == info.num_events_in_query
    assert set(events.type) == set(info.event_types_in_query)
    assert info.frequency == 100.0
    assert len(list(Willett2021Handwriting(path=ROOT).iter_timelines())) == (
        info.num_timelines
    )
