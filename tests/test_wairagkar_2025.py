"""wairagkar_2025 control-experiment reader invariants"""

import datetime as dt
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio

from neuralshed.datasets.participants import implant_date
from neuralshed.datasets.wairagkar_2025 import utils
from neuralshed.datasets.wairagkar_2025.control_study import (
    Wairagkar2025ControlExperiment,
)

#: the two known homes of the release: the utils default (honours
#: ``NEURALSHED_DATA_ROOT``) and this machine's raw download folder
_CANDIDATES = (
    utils.DEFAULT_CONTROL_DIR,
    Path("/data/data/datasets/wairagkar/t15_control_experiment"),
)

_SESSION = "t15_day482"


def _release():
    for candidate in _CANDIDATES:
        if candidate.is_dir() and list(candidate.glob("t15_day*_block*.mat")):
            return candidate
    pytest.skip(f"t15_control_experiment not mounted at {_CANDIDATES}")


def _farm(tmp_path):
    """A farm folder whose only entry is the class-named symlink into the
    release's parent (same trick as ``parity/step8_jude_membership.py``)."""
    release = _release()
    link = tmp_path / Wairagkar2025ControlExperiment.__name__
    link.symlink_to(release.parent)
    return Wairagkar2025ControlExperiment(path=str(tmp_path))


# ---------------------------------------------------------------------------
# Catalogue: exactly the published release, nothing invented or dropped
# ---------------------------------------------------------------------------


def test_catalogue_is_the_published_release():
    release = _release()
    refs = utils.list_control_blocks(release)
    assert [(r.session, r.block) for r in refs] == [(_SESSION, b) for b in range(10)], (
        "step-8 D4: the reader must catalogue exactly these 10 blocks"
    )
    for ref in refs:
        assert len(utils.get_control_sentences(ref)) == 24, ref.file_stem
        assert utils.get_block_description(ref), ref.file_stem
        # each block ships the online decoder trained on the other nine
        assert ref.decoder_path is not None, ref.file_stem
        assert ref.decoder_path.exists(), ref.decoder_path


# ---------------------------------------------------------------------------
# Identity: one timeline per trial, dated from the implant record
# ---------------------------------------------------------------------------


def test_one_timeline_per_trial_dated_from_the_implant(tmp_path):
    study = _farm(tmp_path)
    tls = study.study_summary(apply_query=False)
    assert len(tls) == 240
    assert set(tls["session"]) == {_SESSION}
    assert sorted(tls["block"].unique().tolist()) == list(range(10))
    for block, group in tls.groupby("block"):
        assert sorted(group["trial_id"].tolist()) == list(range(24))
    assert set(tls["post_implant_day"]) == {482}
    expected = (implant_date("braingate2/T15") + dt.timedelta(days=482)).isoformat()
    assert set(tls["date"]) == {expected}


# ---------------------------------------------------------------------------
# Verbatim neural: reader output == stored float, bit-for-bit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mat_key", ["spikepow_trials", "threshcross_trials"])
def test_reader_emits_the_stored_arrays_verbatim(tmp_path, mat_key):
    study = _farm(tmp_path)
    ref = utils.list_control_blocks(_release())[3]
    # read the stored cell array the way scipy hands it to the reader:
    # squeeze_me drops the .mat cell-array wrapper dimension
    raw = sio.loadmat(str(ref.path), squeeze_me=True)[mat_key]
    for trial in (0, 23):
        timeline = {"session": ref.session, "block": ref.block, "trial_id": trial}
        got = study._load_binned(timeline, trial, mat_key)
        want = np.ascontiguousarray(np.asarray(raw[trial], dtype=np.float32).T)
        assert got.dtype == np.float32
        np.testing.assert_array_equal(got, want)


# ---------------------------------------------------------------------------
# The shipped phoneme target on every cue
# ---------------------------------------------------------------------------


def test_every_sentence_carries_its_phoneme_target(tmp_path):
    study = _farm(tmp_path)
    ev = study.__class__(path=str(tmp_path), query="block == 3").run()
    sent = ev[ev["type"] == "Sentence"]
    assert len(sent) == 24
    table = utils.CUE_PHONEMES.cues
    for text, phonemes in zip(sent["text"], sent["phonemes"]):
        assert phonemes is not None and phonemes == table[text], text
