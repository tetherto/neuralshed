"""The shared phoneme recipe and table in ``neuralshed.text``."""

import json
import re

import pytest

from neuralshed.text import WORD_BOUNDARY, PhonemeTable, PhonemeVocab, g2p_phonemes


def _fake_g2p(text: str) -> list[str]:
    """Stands in for ``g2p_en.G2p``: one stressed vowel per word, spaces kept."""
    out = []
    for i, word in enumerate(text.split(" ")):
        if i:
            out.append(" ")
        out += ["K", "AH0"] if word else []
    out.append(".")
    return out


def test_g2p_phonemes_follows_the_speechbci_recipe():
    assert g2p_phonemes("Need that?", _fake_g2p) == "K AH | K AH |"
    assert g2p_phonemes("", _fake_g2p) == WORD_BOUNDARY
    assert not PhonemeVocab.canonical().coverage([g2p_phonemes("a b", _fake_g2p)])


def _fake_tokenizing_g2p(text: str) -> list[str]:
    """Like ``g2p_en.G2p``, reads ``?`` / ``.`` as tokens of their own, spaces between."""
    out = []
    for i, token in enumerate(re.findall(r"[a-z']+|[?.]", text.lower())):
        if i:
            out.append(" ")
        out += [token] if token in "?." else ["K", "AH0"]
    return out


def test_g2p_phonemes_apostrophe_join():
    # "I 'll go": joined it is two words, split the detached "'ll" is a third
    assert g2p_phonemes("I 'll go.", _fake_tokenizing_g2p) == "K AH | K AH |"
    split = g2p_phonemes("I 'll go.", _fake_tokenizing_g2p, join_split_apostrophes=False)
    assert split == "K AH | K AH | K AH |"


def test_g2p_phonemes_per_word():
    # a boundary after every word, plus one for the punctuation token g2p reads
    got = g2p_phonemes("Are you good?", _fake_tokenizing_g2p, per_word=True)
    assert got == "K AH | K AH | K AH | |"
    assert g2p_phonemes("Are you good?", _fake_tokenizing_g2p) == "K AH | K AH | K AH |"
    assert g2p_phonemes("  ", _fake_tokenizing_g2p, per_word=True) == WORD_BOUNDARY


def test_g2p_phonemes_options_on_real_g2p():
    g2p_en = pytest.importorskip("g2p_en")
    g2p = g2p_en.G2p()
    cue = "I 'll take it out."
    assert g2p_phonemes(cue, g2p) == "AY L | T EY K | IH T | AW T |"
    assert (
        g2p_phonemes(cue, g2p, join_split_apostrophes=False)
        == "AY | | L EH L | T EY K | IH T | AW T |"
    )
    assert (
        g2p_phonemes("Are they right?", g2p, per_word=True)
        == "AA R | DH EY | R AY T | |"
    )


def test_phoneme_table_lookup(tmp_path):
    path = tmp_path / "t.json"
    path.write_text(
        json.dumps(
            {"description": "d", "g2p_en": "0", "n_cues": 1, "cues": {"Hi.": "HH AY |"}}
        )
    )
    table = PhonemeTable(path)
    assert len(table) == 1 and "Hi." in table
    assert table["Hi."] == "HH AY |"
    with pytest.raises(KeyError, match="t.json"):
        table["Bye."]


def test_phoneme_table_build(tmp_path):
    pytest.importorskip("g2p_en")
    table = PhonemeTable.build(
        ["Need that?", "Need that?", "Bring it closer."], tmp_path / "t.json", "d"
    )
    assert table["Bring it closer."] == "B R IH NG | IH T | K L OW S ER |"
    assert table.payload["n_cues"] == 2
    assert table.payload["g2p_options"] == dict(
        join_split_apostrophes=True, per_word=False
    )


def test_phoneme_table_build_records_its_options(tmp_path):
    pytest.importorskip("g2p_en")
    table = PhonemeTable.build(["Need that?"], tmp_path / "t.json", "d", per_word=True)
    assert table["Need that?"] == "N IY D | DH AE T | |"
    assert table.payload["g2p_options"]["per_word"] is True
