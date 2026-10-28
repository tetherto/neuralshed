"""``RescaleNeural``: fit-and-divide moved into the row's loader, bit-for-bit.

The fixture Study emits SpecialLoader-backed dense rows (the shape Willet,
Kunz and Wairagkar studies have) with synthetic arrays, so every expectation
is hand-computable from the audit formulas in
``docs/migration/audit``: Willet/Kunz ``(x - mu) / (sd + 1e-8)``, Wairagkar
``sd == 0 -> 1.0``. No dataset data is touched.
"""

import json
import typing as tp

import numpy as np
import pandas as pd
import pytest
from neuralset.events import study as ns_study
from neuralset.events.utils import SpecialLoader

from neuralshed import scaling
from neuralshed.datasets.study import NeuralshedStudy
from neuralshed.events import BandPower, SpikeCounts  # noqa: F401, registers types
from neuralshed.transforms import RescaleNeural

RATE_HZ = 50.0
BIN_S = 1.0 / RATE_HZ
N_BINS = 40
# Block 0: two disjoint stat windows -> bins [5,15) and [25,35).
# Block 1: two overlapping windows -> union bins [0,30) (the Kunz go-cue
# overlaps the go window; the shared bins must count exactly once).
WINDOWS = {0: [(0.10, 0.30), (0.50, 0.70)], 1: [(0.00, 0.40), (0.30, 0.60)]}


def _arrays(timeline: dict[str, tp.Any]) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(timeline["block"])
    sc = rng.standard_normal((4, N_BINS)).astype(np.float32)
    bp = rng.standard_normal((4, N_BINS)).astype(np.float32)
    if timeline["block"] == 1:
        # constant over block 1's pooled stat bins, live after them: the two
        # zero-guards agree inside the stat window and diverge outside it.
        sc[0, :30] = 2.0
    return sc, bp


class RescaleFixtureStudy(NeuralshedStudy):
    dataset_id: tp.ClassVar[str] = "jude_2026_typing"

    def iter_timelines(self):
        yield {"subject": "T17", "session": "b0", "block": 0}
        yield {"subject": "T17", "session": "b1", "block": 1}

    def _load_array(self, timeline: dict[str, tp.Any], *, which: str) -> np.ndarray:
        sc, bp = _arrays(timeline)
        return sc if which == "sc" else bp

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        sc, bp = _arrays(timeline)
        rows = []
        for etype, which in (("SpikeCounts", "sc"), ("BandPower", "bp")):
            rows.append(
                {
                    "type": etype,
                    "start": 0.0,
                    "duration": N_BINS * BIN_S,
                    "frequency": RATE_HZ,
                    "n_features": 4,
                    "filepath": ns_study.SpecialLoader(
                        method=self._load_array, timeline=timeline, which=which
                    ).to_json(),
                }
            )
        for i, (a, b) in enumerate(WINDOWS[timeline["block"]]):
            rows.append(
                {
                    "type": "Sentence",
                    "start": a,
                    "duration": b - a,
                    "text": f"window {i}",
                }
            )
        rows.append(
            {
                "type": "Stimulus",
                "start": 0.0,
                "duration": 0.08,
                "modality": "visual",
                "description": "cue",
            }
        )
        return pd.DataFrame(rows)

    def _download(self) -> None:
        pass


def _events(tmp_path) -> pd.DataFrame:
    study_obj = RescaleFixtureStudy(path=tmp_path)
    return pd.concat(
        [study_obj._load_one(dict(t)) for t in study_obj.iter_timelines()],
        ignore_index=True,
    )


def _hand_scaled(raw_json: str, columns: list[int], *, guard: str) -> np.ndarray:
    """The audit formula spelled out by hand on the raw payload.

    The statistics are fitted over the builders' column-major stat array
    (``asfortranarray(... .T)`` then a columnwise reduce), not along the
    row-major payload's last axis: float32 reductions are layout-dependent
    and ``fit_rescale`` matches the old pickles only through the F-ordered
    The formula (rows, guard) is what this pins; the summation tree mirrors
    ``fit_rescale`` because float32 reductions are layout-dependent, and the
    tree's real anchor is the Willet pickle parity test in the brainwhisperer
    suite, not a second hand computation here.
    """
    x = np.asarray(SpecialLoader.from_json(raw_json).load())
    cols = np.asarray(columns)
    pooled = np.asfortranarray(x[:, cols].T)
    mu = pooled.mean(axis=0, keepdims=True).T
    sd = pooled.std(axis=0, keepdims=True).T
    div = sd + 1e-8 if guard == "eps" else np.where(sd == 0, 1.0, sd)
    return (x - mu) / div


def _rescaled_rows(events: pd.DataFrame, **knobs) -> dict[tuple[int, str], dict]:
    out = RescaleNeural(**knobs)(events)  # pyrefly: ignore [bad-argument-type]
    rows = {}
    for _, row in out.iterrows():
        if row["type"] not in RescaleNeural.model_fields["event_types"].default:
            continue
        payload = json.loads(row["filepath"])
        block = int(str(row["timeline"]).split("block=")[1].split(",")[0])
        rows[(block, str(row["type"]))] = payload
    return rows


def test_rescale_fits_windows_and_divides_at_read_time(tmp_path):
    events = _events(tmp_path)
    payloads = _rescaled_rows(events, stat_row_query="type == 'Sentence'", guard="eps")
    assert set(payloads) == {
        (0, "SpikeCounts"),
        (0, "BandPower"),
        (1, "SpikeCounts"),
        (1, "BandPower"),
    }
    for (block, etype), payload in payloads.items():
        assert payload["method"] == scaling.METHOD
        assert payload["kwargs"]["guard"] == "eps"
        expected_bins = sorted(
            [int(round(a * RATE_HZ)), int(round(b * RATE_HZ))]
            for a, b in WINDOWS[block]
        )
        assert sorted(payload["kwargs"]["stat_rows"]) == expected_bins
        raw = payload["kwargs"]["inner"]
        assert json.loads(raw)["method"] == "_load_array"  # the raw loader, wrapped
        assert json.loads(raw)["timeline"] == payload["timeline"]
        hand = _hand_scaled(
            raw,
            sorted({c for a, b in expected_bins for c in range(a, b)}),
            guard="eps",
        )
        np.testing.assert_array_equal(
            SpecialLoader.from_json(  # pyrefly: ignore [bad-argument-type]
                json.dumps(payload)
            ).load(),
            hand,
        )


def test_overlapping_stat_windows_count_shared_bins_once(tmp_path):
    events = _events(tmp_path)
    payloads = _rescaled_rows(events, stat_row_query="type == 'Sentence'", guard="eps")
    payload = payloads[(1, "SpikeCounts")]
    raw = payload["kwargs"]["inner"]
    union = _hand_scaled(raw, list(range(30)), guard="eps")
    double_counted = _hand_scaled(
        raw, list(range(20)) + list(range(15, 30)), guard="eps"
    )
    got = SpecialLoader.from_json(json.dumps(payload)).load()
    np.testing.assert_array_equal(got, union)
    assert not np.array_equal(got, double_counted)


def test_trim_leading_bins_drops_the_first_bin_of_each_window(tmp_path):
    events = _events(tmp_path)
    payloads = _rescaled_rows(
        events, stat_row_query="type == 'Sentence'", trim_leading_bins=1, guard="eps"
    )
    payload = payloads[(1, "BandPower")]
    # windows [0,20) and [15,30) trim to [1,20) and [16,30): merged [1,30).
    assert sorted(payload["kwargs"]["stat_rows"]) == [[1, 20], [16, 30]]
    hand = _hand_scaled(payload["kwargs"]["inner"], list(range(1, 30)), guard="eps")
    np.testing.assert_array_equal(
        SpecialLoader.from_json(json.dumps(payload)).load(), hand
    )


def test_zero_to_one_guard_divides_constant_channels_by_one(tmp_path):
    events = _events(tmp_path)
    payloads = _rescaled_rows(
        events, stat_row_query="type == 'Sentence'", guard="zero_to_one"
    )
    payload = payloads[(1, "SpikeCounts")]
    hand = _hand_scaled(
        payload["kwargs"]["inner"], list(range(30)), guard="zero_to_one"
    )
    got = SpecialLoader.from_json(json.dumps(payload)).load()
    np.testing.assert_array_equal(got, hand)
    # Channel 0 is constant over the stat bins: the Wairagkar guard leaves the
    # post-window bins raw-scaled by 1.0, where Willet's eps guard explodes.
    raw = np.asarray(SpecialLoader.from_json(payload["kwargs"]["inner"]).load())
    assert got[0, 30] == (raw[0, 30] - 2.0) / 1.0
    eps_hand = _hand_scaled(payload["kwargs"]["inner"], list(range(30)), guard="eps")
    assert abs(eps_hand[0, 30]) > 100 * abs(got[0, 30])


def test_empty_query_fits_the_whole_stored_array(tmp_path):
    events = _events(tmp_path)
    payloads = _rescaled_rows(events, guard="eps")
    payload = payloads[(0, "SpikeCounts")]
    assert payload["kwargs"]["stat_rows"] == []
    hand = _hand_scaled(payload["kwargs"]["inner"], list(range(N_BINS)), guard="eps")
    np.testing.assert_array_equal(
        SpecialLoader.from_json(json.dumps(payload)).load(), hand
    )


def test_rewritten_loader_is_a_distinct_cache_key(tmp_path):
    events = _events(tmp_path)

    def block_of(timeline: str) -> int:
        return int(str(timeline).split("block=")[1].split(",")[0])

    original = {
        (block_of(r["timeline"]), str(r["type"])): str(r["filepath"])
        for _, r in events.iterrows()
        if r["type"] in ("SpikeCounts", "BandPower")
    }
    payloads = _rescaled_rows(events, stat_row_query="type == 'Sentence'")
    for (block, etype), payload in payloads.items():
        new = json.dumps(payload, sort_keys=True)
        old = original[(block, etype)]
        assert new != old
        assert ":" in new  # -> its own extractor cache path (etypes.py:375)


def test_non_neural_rows_and_columns_pass_through(tmp_path):
    events = _events(tmp_path)
    out = RescaleNeural(stat_row_query="type == 'Sentence'")(events)
    changed = {
        i
        for i in events.index
        if pd.notna(events.at[i, "filepath"])
        and events.at[i, "filepath"] != out.at[i, "filepath"]
    }
    assert set(out.loc[sorted(changed), "type"]) == {"SpikeCounts", "BandPower"}
    for column in ("start", "duration", "frequency", "text", "type", "timeline"):
        assert events[column].equals(out[column])


def test_query_matching_nothing_raises(tmp_path):
    events = _events(tmp_path)
    with pytest.raises(ValueError, match="matched \\s*no stat rows"):
        RescaleNeural(stat_row_query="type == 'Word'")(events)


def test_plain_filepath_stream_is_refused(tmp_path):
    events = _events(tmp_path)
    events = events.copy()
    events.loc[0, "filepath"] = "some/raw/path.npy"
    with pytest.raises(ValueError, match="SpecialLoader-backed"):
        RescaleNeural(stat_row_query="type == 'Sentence'")(events)


def test_trim_beyond_the_array_raises(tmp_path):
    events = _events(tmp_path)
    with pytest.raises(ValueError, match="no stat bins"):
        RescaleNeural(stat_row_query="type == 'Sentence'", trim_leading_bins=N_BINS)(
            events
        )


def test_baked_mean_skips_the_fit_entirely(tmp_path):
    """``mean`` + ``divisor`` = ``(x - mean) / divisor``, no fit, no stat rows.

    The stat rows are deliberately ones that clamp away from the array: a
    fitting path would raise on them, so loading proves the fit is skipped
    (the Kunz T15 / Jude / Wairagkar transforms bake statistics fitted on
    another array than the payload and must never re-fit here).
    """
    events = _events(tmp_path)
    row = events[events["type"] == "SpikeCounts"].iloc[0]
    raw = str(row["filepath"])
    mean = [0.5, -1.25, 2.0, 11.0 / 7]
    divisor = [1.5, 0.7, 3.0, 1e-8]
    payload = json.loads(
        scaling.rescaled_loader_json(
            raw,
            stat_rows=[[100, 200]],  # clamps away: only a skipped fit survives
            guard="eps",
            divisor=divisor,
            mean=mean,
        )
    )
    assert payload["kwargs"]["mean"] == mean  # float64 reprs survive verbatim
    assert payload["kwargs"]["divisor"] == divisor
    got = SpecialLoader.from_json(json.dumps(payload)).load()
    x = np.asarray(SpecialLoader.from_json(raw).load())
    mu = np.asarray(mean, dtype=x.dtype).reshape(-1, 1)
    div = np.asarray(divisor, dtype=x.dtype).reshape(-1, 1)
    np.testing.assert_array_equal(got, (x - mu) / div)


def test_baked_mean_without_divisor_is_refused():
    with pytest.raises(ValueError, match="baked mean requires a baked divisor"):
        scaling.rescaled_loader_json(
            json.dumps({"cls": "X", "timeline": "t", "method": "m"}),
            stat_rows=[],
            mean=[1.0],
        )


def test_apply_rescale_without_sd_needs_a_divisor():
    x = np.ones((2, 4), dtype=np.float32)
    mu = np.zeros((2, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="fitted sd or an explicit divisor"):
        scaling.apply_rescale(x, mu)
    # the divisor path ignores sd entirely
    np.testing.assert_array_equal(
        scaling.apply_rescale(x, mu, None, "eps", divisor=[2.0, 4.0]),
        np.array([[0.5, 0.5, 0.5, 0.5], [0.25, 0.25, 0.25, 0.25]], dtype=np.float32),
    )
