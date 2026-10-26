"""fan_2023: Dryad plus DANDI held-out days, per-trial timelines, and the sources' identity."""

import hashlib
import struct

import numpy as np
import pandas as pd
import pytest

from neuralset.events import etypes
from neuralshed.datasets.fan_2023 import metadata, utils
from neuralshed.datasets.fan_2023.study import Fan2023Handwriting
from neuralshed.datasets.utils import dataset_dir
from neuralshed.datasets.willet_2021.metadata import (
    DATASET_METADATA as WILLET_2021,
)
from neuralshed.extractors import NeuralFeatureExtractor

ROOT = dataset_dir("fan_2023") / "data"
SOURCE = ROOT / "Fan2023Handwriting"
FALCON = SOURCE / utils.DANDISET_ID
FLAGGED = {"2023-02-28", "2023-08-16", "2023-10-09"}


def _drive():
    if not (SOURCE / utils.DRYAD_ROOT).exists() or not FALCON.exists():
        pytest.skip(f"fan_2023 data not downloaded under {SOURCE}")


def _falcon(split, date):
    (path,) = [
        p
        for p in (FALCON / f"sub-T5-{split}").glob("*.nwb")
        if not p.name.startswith("._") and utils.date_of(p) == date
    ]
    return utils.read_nwb(path)


def _digest(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


# --- Offline -------------------------------------------------------------------
def test_download_fetches_both_sources_and_forwards_overwrite(tmp_path, monkeypatch):
    from neuralfetch import download

    from neuralshed.datasets import dryad

    calls = []
    study = Fan2023Handwriting(path=tmp_path)

    def fake_dryad(self, overwrite=False):
        calls.append(("dryad", self.doi, overwrite))
        (study.path / "placeholder").write_text("")  # the base refuses an empty folder

    def fake_dandi(self, overwrite=False):
        calls.append(("dandi", self.version, overwrite))

    monkeypatch.setattr(dryad.Dryad, "download", fake_dryad)
    monkeypatch.setattr(download.Dandi, "download", fake_dandi)
    study.download(overwrite=True)
    assert calls == [
        ("dryad", "10.5061/dryad.hqbzkh1p6", True),
        ("dandi", "0.241029.1403", True),
    ]


def test_brain_area_label_matches_the_other_t5_datasets():
    (fan,) = metadata.DATASET_METADATA["subjects"]
    (willet,) = WILLET_2021["subjects"]
    assert [a["raw_label"] for a in fan["brain_areas"]] == [
        a["raw_label"] for a in willet["brain_areas"]
    ]


def test_falcon_split_keeps_the_first_sixty_percent():
    labels = [utils.falcon_split(i, 50) for i in range(50)]
    assert labels.count("held-in-calib") == 30 and labels[29:31] == [
        "held-in-calib",
        "held-in-eval",
    ]


def test_trial_bins_reject_a_clock_that_is_not_one_clip_per_trial():
    t = np.r_[np.arange(5) * 0.02, 10.1 + np.arange(5) * 0.02]
    trials = pd.DataFrame({"start_time": [0.0, 10.1], "stop_time": [0.08, 10.18]})
    assert utils.trial_bins(t, trials).values.tolist() == [[0, 5], [5, 10]]
    with pytest.raises(ValueError):
        utils.trial_bins(t, trials.assign(stop_time=[0.06, 10.18]))


# --- The two deposits ------------------------------------------------------------
def test_falcon_held_in_trials_are_byte_copies_of_the_dryad_prefix():
    _drive()
    sources = [s for s in utils.list_sources(SOURCE) if s["path"].endswith(".mat")]
    held_in = [s for s in sources if s["block_kind"] != "no_recalibration"]
    assert len(held_in) == 21
    for src in held_in:
        dryad = utils.read_mat(SOURCE / src["path"])
        calib = _falcon("held-in-calib", src["date"])
        minival = _falcon("held-in-minival", src["date"])
        n = len(dryad["tx_feats"])
        assert len(calib["tx_feats"]) == n - int(0.4 * n), src["date"]
        for i, x in enumerate(calib["tx_feats"]):
            assert x.tobytes() == dryad["tx_feats"][i].tobytes(), (src["date"], i)
        assert calib["sentences"] == dryad["sentences"][: len(calib["sentences"])]
        assert calib["blocks"] == dryad["blocks"][: len(calib["blocks"])]
        assert [x.tobytes() for x in minival["tx_feats"]] == [
            x.tobytes() for x in dryad["tx_feats"][:2]
        ]


def test_no_recalibration_trials_are_absent_from_falcon():
    _drive()
    falcon = {
        _digest(x)
        for p in FALCON.rglob("*.nwb")
        if not p.name.startswith("._")
        for x in utils.read_nwb(p)["tx_feats"]
    }
    for src in utils.list_sources(SOURCE):
        if src["block_kind"] == "no_recalibration":
            trials = utils.read_mat(SOURCE / src["path"])["tx_feats"]
            assert not any(_digest(x) in falcon for x in trials), src["date"]


def test_held_out_days_are_only_on_dandi():
    _drive()
    sources = utils.list_sources(SOURCE)
    dryad_days = {s["date"] for s in sources if s["path"].endswith(".mat")}
    held_out = {s["date"] for s in sources if s["path"].endswith(".nwb")}
    assert len(held_out) == 5 and not held_out & dryad_days
    assert min(held_out) > max(dryad_days)


def _n_records(folder):
    """Number of records in the tfrecord files of a folder."""
    n = 0
    for f in folder.glob("*.tfrecord"):
        if f.name.startswith("._"):
            continue
        data = f.read_bytes()
        i = 0
        while i < len(data):
            (size,) = struct.unpack("<Q", data[i : i + 8])
            i += 12 + size + 4
            n += 1
    return n


def test_corp_split_matches_the_dryad_tfrecords():
    _drive()
    for src in utils.list_sources(SOURCE):
        if not src["path"].endswith(".mat"):
            continue
        n = utils.count_trials(SOURCE / src["path"])
        labels = [utils.corp_split(src["block_kind"], i, n) for i in range(n)]
        mat = SOURCE / src["path"]
        records = mat.parent.parent / "tfrecords" / mat.stem
        for part in ("train", "test"):
            assert labels.count(part) == _n_records(records / part), (src["path"], part)


# --- The Study -----------------------------------------------------------------
@pytest.fixture(scope="module")
def events():
    _drive()
    return Fan2023Handwriting(path=ROOT).run()


def test_one_timeline_per_trial_of_every_source(events):
    expected = sum(
        utils.count_trials(SOURCE / s["path"]) for s in utils.list_sources(SOURCE)
    )
    assert events.timeline.nunique() == expected
    per_timeline = events.groupby("timeline").type.apply(sorted)
    assert (per_timeline.map(tuple) == ("Sentence", "SpikeCounts")).all()
    for column in ("block", "trial_id", "post_implant_day"):
        assert pd.api.types.is_integer_dtype(events[column]), column
    spans = events.groupby("timeline")[["start", "duration"]].nunique()
    assert (spans == 1).all().all() and (events.start == 0.0).all()


def test_block_kind_loop_type_and_falcon_labels(events):
    rows = events.drop_duplicates("timeline")
    loop = rows.groupby("block_kind").loop_type.unique().map(list).to_dict()
    assert loop == {
        "no_recalibration": ["closed"],
        "recalibration": ["closed"],
        "seed": ["open"],
    }
    assert set(rows.loc[rows.block_kind == "seed", "date"]) <= {
        d for d in rows.date if d <= "2022-09-01"
    }
    splits = rows.groupby("block_kind").original_split.unique().map(set).to_dict()
    assert splits["no_recalibration"] == {""}
    assert splits["seed"] == {"held-in-calib", "held-in-eval"}
    assert splits["recalibration"] == {
        "held-in-calib",
        "held-in-eval",
        "held-out-calib",
    }
    assert rows.original_split.value_counts().to_dict() == {
        "held-in-calib": 671,
        "held-in-eval": 445,
        "": 199,
        "held-out-calib": 15,
    }
    assert rows.corp_split.value_counts().to_dict() == {
        "train": 1205,
        "test": 110,
        "": 15,
    }
    assert set(rows.loc[rows.corp_split == "test", "block_kind"]) == {"seed"}


def _source_trial(sentence):
    """The trial's (192, n_bins) array in its source file, found by day and block kind."""
    paths = {
        (s["date"], s["block_kind"]): s["path"] for s in utils.list_sources(SOURCE)
    }
    trials = utils.read_source(SOURCE, paths[(sentence.date, sentence.block_kind)])
    return trials["tx_feats"][sentence.trial_id].T


def test_every_window_equals_its_source_trial(events):
    extractor = NeuralFeatureExtractor(
        target_unit="native", event_types="SpikeCounts", frequency=50.0
    )
    neural = events[events.type == "SpikeCounts"].set_index("timeline")
    for _, sentence in events[events.type == "Sentence"].iterrows():
        row = neural.loc[sentence.timeline].to_dict() | {"timeline": sentence.timeline}
        window = extractor(
            [etypes.Event.from_dict(row)], start=0.0, duration=sentence.duration
        )
        assert np.array_equal(window.numpy(), _source_trial(sentence)), (
            sentence.timeline
        )


def test_a_resampled_window_averages_source_bin_pairs(events):
    sentence = events[events.type == "Sentence"].iloc[-1]
    neural = events[
        (events.timeline == sentence.timeline) & (events.type == "SpikeCounts")
    ]
    extractor = NeuralFeatureExtractor(
        target_unit="Hz", event_types="SpikeCounts", frequency=25.0
    )
    window = extractor(
        [etypes.Event.from_dict(neural.iloc[0].to_dict())],
        start=0.0,
        duration=sentence.duration,
    ).numpy()
    clip = _source_trial(sentence)
    n = clip.shape[1] // 2
    pairs = clip[:, : 2 * n].reshape(clip.shape[0], n, 2).mean(axis=2) * 50.0
    np.testing.assert_allclose(window[:, :n], pairs, rtol=1e-5, atol=1e-4)


def test_rows_carry_the_published_acquisition_and_targets(events):
    neural = events[events.type == "SpikeCounts"]
    assert set(neural.threshold) == {"-4.5 x SD"}
    assert set(neural.unit) == {"counts/bin"} and set(neural.normalization) == {"raw"}
    assert set(neural.frequency) == {50.0} and set(neural.n_features) == {192}
    assert (neural.processing.str.len() > 0).all()
    sentences = events[events.type == "Sentence"]
    assert set(sentences.modality) == {"written"}
    assert not sentences.text.str.contains("[>~]").any()
    online_dryad = sentences.block_kind.ne("seed") & sentences.original_split.ne(
        "held-out-calib"
    )
    assert sentences.loc[online_dryad, "lm_decoded"].notna().all()
    assert sentences.loc[~online_dryad, "lm_decoded"].isna().all()


def test_flagged_days_are_invalid_with_a_reason(events):
    flagged = events[~events.valid]
    assert set(flagged.date) == FLAGGED
    assert (flagged.valid_description.str.len() > 0).all()
    assert (events[events.valid].valid_description == "").all()


def test_study_info_reproduces():
    _drive()
    info = Fan2023Handwriting._info
    assert info.frequency == 50.0
    events = Fan2023Handwriting(path=ROOT, query=info.query).run()
    assert len(events) == info.num_events_in_query
    assert set(events.type) == info.event_types_in_query
    assert set(events.block_kind) == {"seed", "recalibration", "no_recalibration"}
    assert not events.valid.all()


def test_sessions_are_the_26_days(events):
    rows = metadata.sessions()
    assert len(rows) == 26
    assert sum(r["n_trials"] for r in rows) == events.timeline.nunique()
    assert {r["label"] for r in rows} == set(events.session)
