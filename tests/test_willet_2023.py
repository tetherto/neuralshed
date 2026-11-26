"""willet_2023: the Study's channel order against the authors' unscrambleChans."""

import numpy as np

from neuralshed.datasets.willet_2023 import utils
from neuralshed.datasets.willet_2023.metadata import DATASET_METADATA

# chanToElec of unscrambleChans, fwillett/speechBCI AnalysisExamples/analysis.py at
# commit 522a960: the 1-based electrode of each file column of one 128-column NSP half,
# electrodes 1-64 the inferior array and 65-128 the superior array.
CHAN_TO_ELEC = np.array([
    63, 64, 62, 61, 59, 58, 60, 54, 57, 50, 53, 49, 52, 45, 55, 44, 56, 39, 51, 43,
    46, 38, 48, 37, 47, 36, 42, 35, 41, 34, 40, 33, 96, 90, 95, 89, 94, 88, 93, 87,
    92, 82, 86, 81, 91, 77, 85, 83, 84, 78, 80, 73, 79, 74, 75, 76, 71, 72, 68, 69,
    66, 70, 65, 67, 128, 120, 127, 119, 126, 118, 125, 117, 124, 116, 123, 115, 122, 114, 121, 113,
    112, 111, 109, 110, 107, 108, 106, 105, 104, 103, 102, 101, 100, 99, 97, 98, 32, 30, 31, 29,
    28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 16, 17, 7, 15, 6, 14, 5, 13, 4, 12, 3, 11, 2, 10, 1, 9, 8,
]) - 1  # fmt: skip


def test_channel_order_is_unscramble_chans():
    expected = np.empty(256, int)
    for half in (0, 1):
        expected[128 * half + CHAN_TO_ELEC] = np.arange(128) + 128 * half
    assert np.array_equal(utils.CHANNEL_ORDER, expected)


def test_array_blocks_hold_their_grid_columns():
    for name in utils.ARRAY_ORDER:
        block = utils.CHANNEL_ORDER[utils.STUDY_ARRAY_SLICES[name]]
        assert set(block) == set(utils.ARRAY_GRIDS[name].ravel())


def test_catalog_ranges_follow_the_array_order():
    (subject,) = DATASET_METADATA["subjects"]
    for array in subject["arrays"]:
        block = utils.STUDY_ARRAY_SLICES[array["name"]]
        assert (array["channel_start"], array["channel_stop"]) == (
            block.start,
            block.stop - 1,
        )
