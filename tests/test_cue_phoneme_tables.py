"""Each shipped ``assets/cue_phonemes.json`` is what its ``build_cue_phonemes`` makes
from the release on disk: same header, same cues, same targets. Skips per dataset when
its release is absent under ``NEURALSHED_DATA_ROOT``."""

import json

import pytest

from neuralshed.datasets.kunz_2025 import utils as kunz_utils
from neuralshed.datasets.wairagkar_2025 import utils as wairagkar_utils
from neuralshed.datasets.willet_2023 import utils as willet_utils

_DATASETS = {
    "willet_2023": (willet_utils, willet_utils.source_root()),
    "kunz_2025": (kunz_utils, kunz_utils.DATA_DIR),
    "wairagkar_2025": (wairagkar_utils, wairagkar_utils.DEFAULT_CONTROL_DIR),
}


@pytest.mark.parametrize("name", sorted(_DATASETS))
def test_shipped_table_rebuilds_from_the_release(name, tmp_path):
    pytest.importorskip("g2p_en")
    utils, data_dir = _DATASETS[name]
    if not data_dir.is_dir():
        pytest.skip(f"{name} release not present at {data_dir}")
    rebuilt = utils.build_cue_phonemes(data_dir, path=tmp_path / "cue_phonemes.json")
    shipped = json.loads(utils.CUE_PHONEMES.path.read_text())
    assert rebuilt.payload == shipped
