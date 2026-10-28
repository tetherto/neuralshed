"""Channel order of jude_2026_speech: the per-era permutation and its source-backed check."""

import numpy as np
import pytest

from neuralshed.datasets.jude_2026_speech import utils
from neuralshed.datasets.utils import dataset_dir

EARLY = ("t17.2024.03.25", 5)
MIDDLE = ("t17.2024.04.17", 4)
LATE = ("t17.2024.05.06", 1)


def _drive():
    if not utils.T17_DIR.exists():
        pytest.skip(
            f"jude_2026_speech data not mounted at {dataset_dir('jude_2026_speech')}"
        )


def test_channel_order_is_a_permutation_per_era():
    for session, n in [
        ("t17.2024.03.25", 384),
        ("t17.2024.05.06", 384),
        ("t17.2024.04.15", 256),
    ]:
        order = utils.channel_order(session)
        assert sorted(order.tolist()) == list(range(n))
    assert np.array_equal(utils.channel_order("t17.2024.05.06"), np.arange(384))
    assert np.array_equal(utils.channel_order("t17.2024.04.15"), np.arange(256))
    early = utils.channel_order("t17.2024.03.25")
    # Study block k takes early file group FILE_GROUPS["early"][ARRAY_ORDER[k]].
    for k, name in enumerate(utils.ARRAY_ORDER):
        g = utils.FILE_GROUPS["early"][name]
        assert early[k * 64 : (k + 1) * 64].tolist() == list(
            range(g * 64, (g + 1) * 64)
        )


def test_study_columns_cover_the_areas():
    assert utils.area_channels("t17.2024.05.06") == {
        "6v": (0, 128),
        "55b": (128, 256),
        "6d": (256, 384),
    }
    assert utils.area_channels("t17.2024.04.15") == {"6v": (0, 128), "55b": (128, 256)}
    assert list(utils.array_channels("t17.2024.04.15")) == list(utils.ARRAY_ORDER[:4])


def _group_fingerprints(session, block):
    d = utils.read_block(
        utils.block_path(session, block), fields=["binned_neural_spike_band_power"]
    )
    x = np.log1p(np.asarray(d["binned_neural_spike_band_power"], dtype=np.float32))
    x = utils.to_study_order(x, session)
    n = (x.shape[0] // 10) * 10
    x = x[:n].reshape(-1, 10, x.shape[1]).mean(1)
    corr = np.corrcoef(x.T)
    med = np.median(x, axis=0)
    out = []
    for g in range(x.shape[1] // 64):
        sub = corr[g * 64 : (g + 1) * 64, g * 64 : (g + 1) * 64]
        out.append((sub[~np.eye(64, dtype=bool)], med[g * 64 : (g + 1) * 64]))
    return out


def _best_matches(a, b):
    sim = np.array(
        [
            [
                0.5 * np.corrcoef(sa, sb)[0, 1] + 0.5 * np.corrcoef(ma, mb)[0, 1]
                for sb, mb in b
            ]
            for sa, ma in a
        ]
    )
    return sim.argmax(1).tolist()


def test_study_order_aligns_the_arrays_across_eras():
    """After the permutation, group k of an early or 256-channel block matches group k late."""
    _drive()
    late = _group_fingerprints(*LATE)
    assert _best_matches(_group_fingerprints(*EARLY), late) == [0, 1, 2, 3, 4, 5]
    assert _best_matches(_group_fingerprints(*MIDDLE), late) == [0, 1, 2, 3]


def test_loader_applies_the_permutation():
    _drive()
    from neuralshed.datasets.jude_2026_speech.study import Jude2026IntendedSpeech

    study = Jude2026IntendedSpeech(path=dataset_dir("jude_2026_speech") / "data")
    timeline = next(
        t for t in study.iter_timelines() if (t["session"], t["block"]) == EARLY
    )
    loaded = study._load_features(timeline, "threshold_crossings")
    d = utils.read_block(
        utils.block_path(*EARLY), fields=["binned_neural_threshold_crossings"]
    )
    raw = np.asarray(d["binned_neural_threshold_crossings"], dtype=np.float32)
    g = utils.FILE_GROUPS["early"]["ventral_6v"]
    assert loaded.shape == (384, raw.shape[0])
    assert np.array_equal(loaded[:64], raw[:, g * 64 : (g + 1) * 64].T)
