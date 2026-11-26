"""karpowicz_2024_h1: FALCON H1 (DANDI 000954) on the kinematics grid."""

import numpy as np
import pytest

from neuralset.events import etypes
from neuralshed.datasets import nwb
from neuralshed.datasets.karpowicz_2024_h1 import utils
from neuralshed.datasets.karpowicz_2024_h1.study import Karpowicz2024FalconH1
from neuralshed.datasets.utils import dataset_dir
from neuralshed.extractors import NeuralFeatureExtractor

ROOT = dataset_dir("karpowicz_2024_h1") / "data"


def test_a_spike_bin_ends_at_its_kinematic_sample():
    """Sample k at k * 0.02 s goes with the crossings of the 20 ms before it, so one
    before 0 belongs to sample 0 and one at 0 to sample 1."""
    spike_times = [np.array([-0.015, 0.0, 0.019, 0.021, 0.039]), np.array([])]
    counts = utils.bin_spike_times(spike_times, n_bins=3)
    assert counts.tolist() == [[1, 2, 2], [0, 0, 0]]


def test_minival_copies_are_left_out_and_blocks_count_within_a_day():
    """A held-in-minival file repeats the start of the held-in-calib file with the same
    timestamp, so only the calib file becomes a recording."""
    paths = [
        "sub-HumanPitt-held-in-calib/sub-HumanPitt-held-in-calib_ses-19250101T112404.nwb",
        "sub-HumanPitt-held-in-minival/sub-HumanPitt-held-in-minival_ses-19250101T111740.nwb",
        "sub-HumanPitt-held-in-calib/sub-HumanPitt-held-in-calib_ses-19250101T111740.nwb",
        "sub-HumanPitt-held-out-calib/sub-HumanPitt-held-out-calib_ses-19250126T113454.nwb",
    ]
    rows = utils.recordings(paths)
    assert [(r["session"], r["block"], r["split"]) for r in rows] == [
        ("19250101_block1", 1, "held-in-calib"),
        ("19250101_block2", 2, "held-in-calib"),
        ("19250126_block1", 1, "held-out-calib"),
    ]
    assert rows[0]["path"] == paths[2]


@pytest.fixture(scope="module")
def first_timeline():
    if not (ROOT / "Karpowicz2024FalconH1").exists():
        pytest.skip(f"karpowicz_2024_h1 data not downloaded under {ROOT}")
    study = Karpowicz2024FalconH1(path=ROOT, query="timeline_index == 0")
    return study, study.run()


def _falcon_counts(source):
    """``falcon_challenge.dataloaders.load_nwb`` for H1, as ``(n_channels, n_bins)``."""
    with nwb.open_nwb(source) as nwbfile:
        kinematics = nwbfile.acquisition["OpenLoopKinematics"]
        n = kinematics.data.shape[0]
        timestamps = kinematics.offset + np.arange(n) * kinematics.rate
        spike_times = utils.get_spike_times(nwbfile)
    edges = np.concatenate([[timestamps[0] - 0.02], timestamps])
    counts = np.stack([np.histogram(t, bins=edges)[0] for t in spike_times])
    return counts, sum(len(t) for t in spike_times)


def test_spike_counts_pair_with_kinematics_as_falcon_bins_them(first_timeline):
    study, events = first_timeline
    neural = events[events.type == "SpikeCounts"].iloc[0]
    position = events[events.type == "PositionSeries"].iloc[0]
    assert (neural.start, neural.duration, neural.frequency) == (
        position.start,
        position.duration,
        position.frequency,
    )

    window = NeuralFeatureExtractor(event_types="SpikeCounts", frequency=50.0)(
        [etypes.Event.from_dict(neural.to_dict())], start=0.0, duration=neural.duration
    ).numpy()
    timeline = next(study.iter_timelines())
    expected, n_spikes = _falcon_counts(study._resolve_source(timeline["path"]))
    assert expected[:, 0].sum() > 0 and expected[:, -1].sum() > 0  # both edges in use
    assert np.array_equal(window, expected)
    assert window.sum() == n_spikes


def test_the_eval_mask_row_is_the_source_mask(first_timeline):
    study, events = first_timeline
    row = events[events.type == "BinaryMask"].iloc[0]
    assert row.true_means == "scored"
    mask = etypes.Event.from_dict(row.to_dict()).read()
    timeline = next(study.iter_timelines())
    with nwb.open_h5(study._resolve_source(timeline["path"])) as f:
        source = f["acquisition/eval_mask/data"][:]
    assert mask.dtype == bool and mask.shape == (1, len(source))
    assert np.array_equal(mask[0], source) and not mask.all()


def test_a_phase_takes_the_trial_covering_most_of_it():
    """TrialNum steps to the next trial a few bins before a Release ends, and the
    Release still belongs to the trial it started in."""
    trial_num = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
    starts = [
        0.0,
        0.06,
        0.16,
    ]  # a cue, a Release ending 2 bins into trial 2, the next cue
    stops = [0.06, 0.14, 0.2]
    assert utils.phase_trial_ids(trial_num, starts, stops).tolist() == [1, 1, 2]
