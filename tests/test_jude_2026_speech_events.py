"""Event semantics of jude_2026_speech: cue tables offline, emitted rows against the drive."""

import numpy as np
import pytest

from neuralshed.datasets.jude_2026_speech import utils
from neuralshed.text import WORD_BOUNDARY
from neuralshed.datasets.jude_2026_speech.study import _task_attrs, _unit_type
from neuralshed.datasets.utils import dataset_dir
from neuralshed.text import PhonemeVocab

PHONEME = ("t17.2024.03.25", 5)
TRUNCATED = ("t17.2024.03.25", 8)
OROFACIAL = ("t17.2024.03.25", 14)
WORD = ("t17.2024.03.26", 10)
CLOSED_LOOP = ("t17.2024.03.26", 27)
PHASED = ("t17.2024.05.23", 1)


def _drive():
    if not utils.T17_DIR.exists():
        pytest.skip(
            f"jude_2026_speech data not mounted at {dataset_dir('jude_2026_speech')}"
        )


def test_phoneme_cues_map_into_the_canonical_vocabulary():
    vocab = PhonemeVocab.canonical()
    assert set(utils.PHONEME_CUES.values()) <= set(vocab.stoi)
    assert len(set(utils.PHONEME_CUES.values())) == 39
    for cue, symbol in utils.PHONEME_CUES.items():
        assert utils.cue_code(cue) == vocab.stoi[symbol]
    for cue in utils.REST_CUES:
        assert cue not in utils.PHONEME_CUES
        assert utils.cue_code(cue) == 40


def test_cue_codes_are_unique_per_class():
    classes = {}
    for cue, code in utils.CUE_CODES.items():
        label = utils.PHONEME_CUES.get(
            cue, cue if cue not in utils.REST_CUES else "rest"
        )
        assert classes.setdefault(code, label) == label
    assert utils.cue_code("Any sentence at all.") == utils.NO_CODE
    assert utils.cue_code("Comfortable") == utils.cue_code("COMFORTABLE")
    assert len(utils.WORD_CUES) == 50 and len(utils.OROFACIAL_CUES) == 12


def test_task_family_helpers():
    assert _unit_type("phased_short_sentence") == "Sentence"
    assert _unit_type("orofacial") == "Action"
    with pytest.raises(ValueError):
        _unit_type("diagnostic")
    assert _task_attrs("word") == {"vocab": "50w", "loop_type": "open"}
    assert _task_attrs("sentence_125k_cl") == {"vocab": "125k", "loop_type": "closed"}
    assert _task_attrs("phased_word") == {"vocab": "na", "loop_type": "open"}


def test_post_implant_day_from_the_registry():
    assert utils.post_implant_day("t17.2024.03.25") == 25
    assert utils.post_implant_day("t17.2024.08.13") == 166


@pytest.fixture(scope="module")
def study():
    _drive()
    from neuralshed.datasets.jude_2026_speech.study import Jude2026IntendedSpeech

    return Jude2026IntendedSpeech(path=dataset_dir("jude_2026_speech") / "data")


def _events(study, key):
    timeline = next(
        t for t in study.iter_timelines() if (t["session"], t["block"]) == key
    )
    return study._load_timeline_events(timeline)


def test_phoneme_block_rows(study):
    df = _events(study, PHONEME)
    units = df[df.type.isin(["Phoneme", "Action"])]
    assert len(units) == 80 and units.trial_id.tolist() == list(range(80))
    assert set(df[df.type == "Phoneme"].text) <= set(PhonemeVocab.canonical().stoi)
    rest = df[df.type == "Action"]
    assert set(rest.action_kind) == {"rest"} and set(rest.code) == {40}
    cues = df[df.stimulus_kind.isin(["phoneme_cue", "rest_cue"])]
    assert len(cues) == 80 and np.allclose(cues.duration, 2.5, atol=0.02)
    assert (cues.code[cues.stimulus_kind == "phoneme_cue"] > 0).all()
    go = df[df.stimulus_kind == "speech_go_cue"]
    assert len(go) == 80 and np.allclose(go.start, cues.start + cues.duration)
    assert units.success.all()
    assert "Intertrial" not in set(df.type)


def test_truncated_trial_is_flagged(study):
    df = _events(study, TRUNCATED)
    flagged = df[df.valid == False]  # noqa: E712
    assert len(flagged) == 1 and flagged.iloc[0].trial_id == 79
    assert "before the trial end marker" in flagged.iloc[0].valid_description


def test_orofacial_block_rows(study):
    df = _events(study, OROFACIAL)
    acts = df[df.type == "Action"]
    assert len(acts) == 34
    assert set(acts.action_kind) == {"orofacial_movement", "speech", "rest"}
    assert set(acts[acts.action_kind == "speech"].description) <= set(utils.SAY_CUES)
    assert (acts.code != utils.NO_CODE).all()
    assert "Phoneme" not in set(df.type)


def test_closed_loop_block_has_intertrial_gaps(study):
    df = _events(study, CLOSED_LOOP)
    gaps = df[df.type == "Intertrial"]
    assert len(gaps) == 39 and (gaps.duration > 1).all()
    sentences = df[df.type == "Sentence"]
    assert (sentences.modality == "read").all()
    assert (sentences.task_condition == "attempted").all()


def test_phased_block_rows(study):
    df = _events(study, PHASED)
    units = df[df.type == "Phoneme"]
    assert len(units) == 150
    assert units.groupby("trial_id").phase.apply(list).map(tuple).unique().tolist() == [
        ("read", "imagined", "attempted")
    ]
    assert set(units.task_condition) == {"read", "imagined", "attempted"}
    assert set(units[units.phase == "imagined"].modality) == {"imagined"}
    assert (df[df.type == "Intertrial"].duration.round(1) == 2.0).all()
    assert (df[df.stimulus_kind == "speech_delay_cue"].duration.round(1) == 2.0).all()


def test_cue_phoneme_table_is_canonical_and_complete():
    from neuralshed.text import PhonemeVocab

    table = utils.CUE_PHONEMES.cues
    assert not PhonemeVocab.canonical().coverage(table.values())
    assert all(v.endswith(WORD_BOUNDARY) for v in table.values())
    for word in utils.WORD_CUES:
        assert utils.CUE_PHONEMES[word].count(WORD_BOUNDARY) == 1
    with pytest.raises(KeyError):
        utils.CUE_PHONEMES["not a cue"]


def test_text_rows_carry_phonemes(study):
    for key in (WORD, CLOSED_LOOP, PHASED):
        events = _events(study, key)
        units = events[events.type.isin(["Sentence", "Word", "Phoneme"])]
        assert units.phonemes.notna().all()
        words = units[units.type != "Phoneme"]
        assert words.phonemes.str.endswith(WORD_BOUNDARY).all()
        phones = units[units.type == "Phoneme"]
        assert (phones.phonemes == phones.text).all()
