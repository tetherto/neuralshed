"""Unit harmonization: `NeuralFeatureExtractor.target_unit` and the `units` registry."""

import ast
import pathlib

import numpy as np
import pytest

from neuralshed import units
from neuralshed.events import BandPower, PositionSeries, SpikeCounts
from neuralshed.events import DenseArrayEvent
from neuralshed.extractors import BehavioralSeriesExtractor, NeuralFeatureExtractor

TARGETS = ("native", "Hz", "counts/bin")

# every unit string a Study emits today, plus the two that must raise
CORPUS_UNITS = sorted(units.NON_CONVERTIBLE | {units.COUNT, units.RATE})
UNREGISTERED_UNITS = ("", "bananas/fortnight")


@pytest.fixture
def make_event(tmp_path):
    """Build a `DenseArrayEvent` backed by a real `.npy`, so `read()` is the normal path."""
    counter = {"n": 0}

    def _make(data, unit, frequency, cls=SpikeCounts, normalization="raw", **extra):
        data = np.asarray(data, dtype=np.float32)
        counter["n"] += 1
        path = tmp_path / f"feat{counter['n']}.npy"
        np.save(path, data)
        return cls(
            timeline="t1",
            subject="s1",
            feature=extra.pop("feature", "threshold_crossings"),
            start=0.0,
            duration=data.shape[-1] / frequency,
            frequency=frequency,
            n_features=data.shape[0],
            unit=unit,
            normalization=normalization,
            filepath=str(path),
            **extra,
        )

    return _make


def extract(event, target_unit, frequency="native"):
    """The `(n_rows, T)` array one extractor call yields, plus its header."""
    ex = NeuralFeatureExtractor(frequency=frequency, target_unit=target_unit)
    timed = ex._preprocess(event)
    return timed.data, timed.header


# --------------------------------------------------------------------------- #
# conversion
# --------------------------------------------------------------------------- #


def test_round_trip_is_exact(make_event):
    """Rate -> counts -> rate reproduces the source exactly.

    Mirrors `wilson_2025`, whose released values are all exact multiples of its 50 Hz bin
    rate with an implied count of at most 20 (the 20 samples in a 20 ms bin at 1 kHz).
    """
    rate_hz = 50.0
    counts = np.arange(21, dtype=np.float32).reshape(1, 21)  # 0..20 counts per bin
    event = make_event(counts * rate_hz, unit="Hz", frequency=rate_hz)

    as_counts, header = extract(event, target_unit="counts/bin")
    assert np.array_equal(as_counts, counts)
    assert as_counts.min() >= 0 and as_counts.max() <= 20
    assert np.array_equal(as_counts, np.round(as_counts))
    assert header["unit"] == "counts/bin"

    back, _ = extract(make_event(as_counts, unit="counts/bin", frequency=rate_hz), "Hz")
    assert np.array_equal(back, counts * rate_hz)


def test_cross_dataset_agreement(make_event):
    """The same physiology, binned at 10 ms as counts and at 20 ms as a rate, converges."""
    fast = make_event(np.full((1, 10), 0.2), unit="counts/bin", frequency=100.0)
    slow = make_event(np.full((1, 5), 20.0), unit="Hz", frequency=50.0)

    fast_hz, _ = extract(fast, target_unit="Hz")
    slow_hz, _ = extract(slow, target_unit="Hz")

    assert np.allclose(fast_hz, 20.0)
    assert np.allclose(slow_hz, 20.0)
    # and they disagree by 100x when nobody harmonizes them
    assert np.allclose(extract(fast, "native")[0], 0.2)


def test_native_is_a_strict_no_op(make_event):
    """Every unit, registered or not, passes through untouched under `native`."""
    values = np.array([[0.0, 1.5, -2.0, 300.0]], dtype=np.float32)
    for unit in [*CORPUS_UNITS, *UNREGISTERED_UNITS]:
        event = make_event(values, unit=unit, frequency=50.0)
        out, header = extract(event, target_unit="native")
        assert np.array_equal(out, values), unit
        assert header["unit"] == unit


# --------------------------------------------------------------------------- #
# what must never be scaled
# --------------------------------------------------------------------------- #


def test_band_power_is_untouched(make_event):
    """`uV^2` is neither a count nor a rate, under every target."""
    values = np.array([[1.0, 250.0, 31400.0]], dtype=np.float32)
    event = make_event(values, unit="uV^2", frequency=100.0, cls=BandPower)
    for target in TARGETS:
        out, header = extract(event, target_unit=target)
        assert np.array_equal(out, values), target
        assert header["unit"] == "uV^2"


@pytest.mark.parametrize("unit", ["a.u.", "uV^2"])
def test_non_convertible_is_untouched(make_event, unit):
    """A unit that is neither a count nor a rate is never scaled.

    `uV^2` is the spike-band-power case. `a.u.` covers a source that never states what
    quantity went in.
    """
    values = np.array([[-1.5, 0.0, 2.25]], dtype=np.float32)
    event = make_event(values, unit=unit, frequency=100.0)
    for target in TARGETS:
        out, header = extract(event, target_unit=target)
        assert np.array_equal(out, values), target
        assert header["unit"] == unit


def test_mean_subtracted_counts_stay_convertible(make_event):
    """An additive offset commutes with rebinning, so the base unit's class survives.

    `(c1 - u) + (c2 - u) = (c1 + c2) - 2u`, and `2u` is the wider bin's own mean.
    """
    values = np.array([[-1.0, 0.0, 3.0]], dtype=np.float32)
    event = make_event(
        values, unit="counts/bin", frequency=50.0, normalization="mean_subtracted"
    )
    out, header = extract(event, target_unit="Hz")
    assert np.array_equal(out, values * 50.0)
    assert header["unit"] == "Hz"
    assert header["normalization"] == "mean_subtracted"


@pytest.mark.parametrize("unit", UNREGISTERED_UNITS)
def test_unregistered_unit_raises(make_event, unit):
    """An absent or unknown unit is loud, never a silent passthrough."""
    event = make_event(np.ones((1, 4), dtype=np.float32), unit=unit, frequency=100.0)
    with pytest.raises(ValueError, match="unregistered unit"):
        extract(event, target_unit="Hz")


# --------------------------------------------------------------------------- #
# the regression this change exists for
# --------------------------------------------------------------------------- #


def test_resampling_a_count_stream_keeps_its_scale(make_event):
    """A 20 ms bin holds what its two 10 ms bins held, whichever unit is asked for.

    The signal must vary over time: a constant one would pass against a resampler that is
    still wrong. Rebinning is area-preserving, so 100 Hz -> 50 Hz sums each pair.
    """
    counts = np.array([[0.0, 0.4, 0.1, 0.9, 0.2, 0.6, 0.3, 0.5]], dtype=np.float32)
    event = make_event(counts, unit="counts/bin", frequency=100.0)
    merged = counts.reshape(1, 4, 2).sum(axis=2)  # what a 20 ms bin holds

    as_hz, header = extract(event, target_unit="Hz", frequency=50.0)
    assert np.allclose(as_hz, merged * 50.0)
    assert header["unit"] == "Hz"

    as_counts, header = extract(event, target_unit="counts/bin", frequency=50.0)
    assert np.allclose(as_counts, merged)
    assert header["unit"] == "counts/bin"

    # `native` keeps the source's unit string, so it has to keep the values true to it
    native, header = extract(event, target_unit="native", frequency=50.0)
    assert np.allclose(native, merged)
    assert header["unit"] == "counts/bin"


# --------------------------------------------------------------------------- #
# rebinning
# --------------------------------------------------------------------------- #


def _total_spikes(data, frequency, unit):
    """Spikes across the whole window, however the stream happens to express them."""
    return (data / frequency if unit == "Hz" else data).sum()


@pytest.mark.parametrize("dst", [50.0, 100.0, 25.0, 30.0, 200.0])
@pytest.mark.parametrize("target", ["Hz", "counts/bin", "native"])
def test_rebinning_conserves_the_spike_total(make_event, dst, target):
    """No change of bin width, in either direction or at any ratio, gains or loses spikes."""
    rng = np.random.default_rng(0)
    counts = rng.poisson(1.2, size=(3, 60)).astype(np.float32)
    event = make_event(counts, unit="counts/bin", frequency=100.0)

    out, header = extract(event, target_unit=target, frequency=dst)
    assert out.shape[-1] == round(dst * 60 / 100.0)
    assert _total_spikes(out, dst, header["unit"]) == pytest.approx(
        counts.sum(), rel=1e-5
    )


def test_downsampling_merges_bins_rather_than_sampling_one(make_event):
    """The distinction interpolation gets wrong: reading one bin in N discards the rest."""
    counts = np.array([[3.0, 0.0, 0.0, 3.0]], dtype=np.float32)
    event = make_event(counts, unit="counts/bin", frequency=100.0)
    out, _ = extract(event, target_unit="counts/bin", frequency=50.0)
    # point-sampling the even bins and rescaling would give [[6.0, 0.0]]
    assert np.allclose(out, [[3.0, 3.0]])


def test_upsampling_splits_a_bin_evenly(make_event):
    """Nothing below the source bin is recoverable, so the count is divided, not repeated."""
    counts = np.array([[2.0, 6.0]], dtype=np.float32)
    event = make_event(counts, unit="counts/bin", frequency=50.0)
    out, _ = extract(event, target_unit="counts/bin", frequency=100.0)
    assert np.allclose(out, [[1.0, 1.0, 3.0, 3.0]])


def test_a_trailing_partial_bin_is_not_diluted(make_event):
    """3 source bins into 2 destination bins: the last one covers only half a bin's window.

    It averages what it actually covers rather than counting the uncovered half as silence.
    """
    counts = np.ones((1, 3), dtype=np.float32)
    event = make_event(counts, unit="counts/bin", frequency=100.0)
    out, _ = extract(event, target_unit="Hz", frequency=50.0)
    assert out.shape[-1] == 2
    assert np.allclose(out, 100.0)  # a steady 1 count per 10 ms is 100 Hz throughout


def test_a_rate_stream_is_invariant_to_the_target_rate(make_event):
    """`wilson_2025`'s case: an intensive quantity reads the same at every bin width."""
    event = make_event(np.full((1, 40), 20.0), unit="Hz", frequency=50.0)
    for dst in (50.0, 25.0, 100.0, 10.0):
        out, header = extract(event, target_unit="Hz", frequency=dst)
        assert np.allclose(out, 20.0), dst
        assert header["unit"] == "Hz"


def test_band_power_is_averaged_across_the_merge(make_event):
    """`uV^2` is non-convertible but still intensive, so the wider bin holds the mean."""
    values = np.array([[10.0, 20.0, 4.0, 8.0]], dtype=np.float32)
    event = make_event(values, unit="uV^2", frequency=100.0, cls=BandPower)
    out, header = extract(event, target_unit="Hz", frequency=50.0)
    assert np.allclose(out, [[15.0, 6.0]])
    assert header["unit"] == "uV^2"


def test_z_scored_values_are_averaged_and_never_rescaled(make_event):
    """Dimensionless values have no bin width to correct for, so only the merge applies.

    The wider bin is the mean of its parts, not their sum. That is the honest fallback: the
    true z-score of a wider bin needs statistics the source never released, so a rebinned
    z-scored stream is no longer unit-variance.
    """
    values = np.array([[1.0, 3.0, -2.0, 0.0]], dtype=np.float32)
    event = make_event(
        values, unit="counts/bin", frequency=100.0, normalization="z_scored"
    )
    out, header = extract(event, target_unit="Hz", frequency=50.0)
    assert np.allclose(out, [[2.0, -1.0]])
    assert header["unit"] == "counts/bin"


# --------------------------------------------------------------------------- #
# isolation and registry
# --------------------------------------------------------------------------- #


def test_behavioral_extractor_has_no_target_unit(tmp_path):
    """Behavioral streams share the base class but must never be swept into a rate."""
    assert "target_unit" not in BehavioralSeriesExtractor.model_fields
    assert "target_unit" in NeuralFeatureExtractor.model_fields

    values = np.array([[0.0, 12.5], [3.0, -4.0]], dtype=np.float32)
    path = tmp_path / "pos.npy"
    np.save(path, values)
    event = PositionSeries(
        timeline="t1",
        subject="s1",
        feature="cursor_position",
        dims=("x", "y"),
        n_features=2,
        start=0.0,
        duration=2 / 50.0,
        frequency=50.0,
        unit="px",
        filepath=str(path),
    )
    out = BehavioralSeriesExtractor(event_types="PositionSeries")._preprocess(event)
    assert np.array_equal(out.data, values)
    assert out.header["unit"] == "px"


def _declared_units(path: pathlib.Path, arg: str = "unit") -> set[str]:
    """Every literal reaching an `arg` argument in a study module.

    Covers the `unit="..."` keyword and the positional form used by the per-dataset `row`
    helpers, expanding a conditional into both branches. A call that omits the argument
    entirely is invisible here, and falls back to the event's default: `""` for `unit`,
    which is unregistered by design, and `"raw"` for `normalization`, which the neural
    streams are separately required to override.
    """
    tree = ast.parse(path.read_text())
    unit_index = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            names = [a.arg for a in node.args.args]
            if arg in names:
                unit_index[node.name] = names.index(arg)

    def literals(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return {node.value}
        if isinstance(node, ast.IfExp):
            return literals(node.body) | literals(node.orelse)
        return set()

    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg == arg:
                found |= literals(kw.value)
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        idx = unit_index.get(name)
        if idx is not None and idx < len(node.args):
            found |= literals(node.args[idx])
    return found


def _studies() -> list[pathlib.Path]:
    datasets = pathlib.Path(units.__file__).parent / "datasets"
    studies = sorted(datasets.glob("*/study.py"))
    assert studies, "no study modules found"
    return studies


def test_every_declared_unit_is_registered():
    """A new dataset introducing a novel unit string fails here, not mid-training."""
    unregistered = {
        f"{path.parent.name}: {unit!r}"
        for path in _studies()
        for unit in _declared_units(path, "unit")
        if not units.is_registered(unit)
    }
    assert not unregistered, "unit strings not in neuralshed.units: " + ", ".join(
        sorted(unregistered)
    )


def test_every_declared_normalization_is_registered():
    """Same enforcement for the second field, which shares the same failure mode."""
    unknown = {
        f"{path.parent.name}: {value!r}"
        for path in _studies()
        for value in _declared_units(path, "normalization")
        if value not in units.NORMALIZATIONS
    }
    assert (
        not unknown
    ), "normalization values not in neuralshed.units: " + ", ".join(sorted(unknown))


def test_every_neural_stream_declares_a_normalization():
    """Neural events must be explicit, never riding on the `raw` default.

    The default exists for the behavioral streams that share `DenseArrayEvent`. Letting a
    neural stream inherit it would make "nobody checked" indistinguishable from "verified
    to be raw", which is the ambiguity this schema exists to remove.

    Scoped per enclosing function rather than per call, since several datasets build their
    rows through a local helper that sets the field once for every stream it emits.
    """
    neural = {"SpikeCounts", "BandPower"}
    missing = []
    for path in _studies():
        tree = ast.parse(path.read_text())
        for func in ast.walk(tree):
            if not isinstance(func, ast.FunctionDef):
                continue
            body = list(ast.walk(func))
            emitted = {
                n.value
                for n in body
                if isinstance(n, ast.Constant) and n.value in neural
            }
            if not emitted:
                continue
            declares = any(
                kw.arg == "normalization"
                for n in body
                if isinstance(n, ast.Call)
                for kw in n.keywords
            ) or any(
                isinstance(n, ast.Constant) and n.value == "normalization" for n in body
            )
            if not declares:
                missing.append(
                    f"{path.parent.name}:{func.lineno} {func.name} "
                    f"({', '.join(sorted(emitted))})"
                )
    assert not missing, "neural rows without an explicit normalization: " + ", ".join(
        missing
    )


# --------------------------------------------------------------------------- #
# registry semantics
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "unit,expected",
    [
        ("counts/bin", "count"),
        ("count", "count"),  # the singular alias of counts/bin
        ("Hz", "rate"),
        ("uV^2", "non_convertible"),
        ("a.u.", "non_convertible"),
        ("au", "non_convertible"),
        ("arbitrary", "non_convertible"),
    ],
)
def test_classify(unit, expected):
    assert units.classify(unit) == expected


@pytest.mark.parametrize(
    "unit,target,expected",
    [
        ("counts/bin", "Hz", 100.0),  # in via the source rate
        (
            "counts/bin",
            "counts/bin",
            2.0,
        ),  # a 20 ms bin holds twice what a 10 ms one does
        ("Hz", "Hz", 1.0),
        ("Hz", "counts/bin", 1 / 50.0),  # out via the *destination* rate
        ("uV^2", "Hz", 1.0),
    ],
)
def test_scale_factor_uses_the_rate_at_each_count_end(unit, target, expected):
    assert units.scale_factor(unit, target, 100.0, 50.0) == pytest.approx(expected)


@pytest.mark.parametrize(
    "unit,target",
    [("counts/bin", "Hz"), ("Hz", "counts/bin"), ("counts/bin", "counts/bin")],
)
def test_scale_factor_defaults_to_an_unchanged_bin_width(unit, target):
    assert units.scale_factor(unit, target, 50.0) == units.scale_factor(
        unit, target, 50.0, 50.0
    )


def test_scale_factor_needs_a_positive_bin_rate():
    with pytest.raises(ValueError, match="positive bin rate"):
        units.scale_factor("counts/bin", "Hz", 0.0)


def test_scale_factor_rejects_an_unknown_target():
    with pytest.raises(ValueError, match="target must be one of"):
        units.scale_factor("counts/bin", "spikes/fortnight", 50.0)


# --------------------------------------------------------------------------- #
# normalization: the second field
# --------------------------------------------------------------------------- #


def test_the_interaction_rule(make_event):
    """Conversion is valid iff the regime left the unit's class intact.

    `card_2024` is the case this protects: its unit is now `counts/bin`, the
    pre-normalization quantity, so the string alone no longer says "do not scale me".
    """
    counts = np.array([[0.0, 1.0, 2.0]], dtype=np.float32)

    scaled, header = extract(
        make_event(counts, unit="counts/bin", frequency=50.0), "Hz"
    )
    assert np.array_equal(scaled, counts * 50.0)
    assert header["unit"] == "Hz"

    event = make_event(
        counts, unit="counts/bin", frequency=50.0, normalization="z_scored"
    )
    out, header = extract(event, target_unit="Hz")
    assert np.array_equal(out, counts)  # no-op, not an error
    assert header["unit"] == "counts/bin"  # and the header does not lie
    assert header["normalization"] == "z_scored"


def test_unregistered_normalization_raises(make_event):
    """The second field gets the same enforcement as the first."""
    event = make_event(
        np.ones((1, 3), dtype=np.float32),
        unit="counts/bin",
        frequency=50.0,
        normalization="vibes",
    )
    with pytest.raises(ValueError, match="unregistered normalization"):
        extract(event, target_unit="Hz")

    # ...but `native` inspects nothing, so it stays a strict no-op
    out, _ = extract(event, target_unit="native")
    assert np.array_equal(out, np.ones((1, 3), dtype=np.float32))


def test_normalization_rides_in_the_header(make_event):
    """Observable downstream rather than invisible, like `unit`."""
    event = make_event(
        np.ones((1, 3), dtype=np.float32),
        unit="counts/bin",
        frequency=50.0,
        normalization="mean_subtracted",
    )
    _, header = extract(event, target_unit="native")
    assert header["normalization"] == "mean_subtracted"


@pytest.mark.parametrize(
    "regime,preserves",
    [
        ("raw", True),
        ("mean_subtracted", True),
        ("z_scored", False),
    ],
)
def test_preserves_unit(regime, preserves):
    assert units.preserves_unit(regime) is preserves


def test_the_lattice_is_one_way():
    """`raw -> mean_subtracted -> z_scored` only; the statistics were never released."""
    assert units.is_reachable("raw", "z_scored")
    assert units.is_reachable("raw", "mean_subtracted")
    assert units.is_reachable("mean_subtracted", "z_scored")
    assert units.is_reachable("z_scored", "z_scored")  # identity

    assert not units.is_reachable("z_scored", "raw")
    assert not units.is_reachable("z_scored", "mean_subtracted")
    assert not units.is_reachable("mean_subtracted", "raw")


def test_rank_equality_does_not_mean_interchangeable(make_event):
    """Two z-scored streams compare reachable even when their chains differ.

    `card_2024` z-scores over trailing trials, `wairagkar_2025` logs first and z-scores over
    a rolling window. `processing` is what separates them, and `is_reachable` cannot see it.
    """
    values = np.ones((1, 3), dtype=np.float32)
    kwargs = dict(unit="counts/bin", frequency=50.0, normalization="z_scored")
    card = make_event(values, processing="statistics from trailing trials", **kwargs)
    wairagkar = make_event(values, processing="log-transformed before...", **kwargs)

    assert units.is_reachable(card.normalization, wairagkar.normalization)
    assert card.processing != wairagkar.processing


def test_lattice_helpers_reject_unknown_values():
    for fn in (units.check_normalization, units.preserves_unit):
        with pytest.raises(ValueError, match="unregistered normalization"):
            fn("almost_raw")
    with pytest.raises(ValueError, match="unregistered normalization"):
        units.is_reachable("raw", "almost_raw")


# --------------------------------------------------------------------------- #
# cache identity
# --------------------------------------------------------------------------- #


@pytest.fixture
def baseline(make_event):
    """One event, so variants below differ only in the field under test."""
    return make_event(
        np.ones((1, 4), dtype=np.float32),
        unit="counts/bin",
        frequency=100.0,
        normalization="raw",
    )


def _uid(event, **overrides):
    from neuralshed.extractors import _item_uid

    return _item_uid(event.model_copy(update=overrides) if overrides else event)


#: a value differing from the baseline, for each cache-relevant field
_OTHER_VALUE = {
    "unit": "Hz",
    "normalization": "z_scored",
    "frequency": 50.0,
    "feature": "spike_band_power",
}


@pytest.mark.parametrize("field", sorted(_OTHER_VALUE))
def test_item_uid_covers_everything_preprocess_reads(baseline, field):
    """A corrected declaration must not read back a conversion made under the old one.

    Parametrized over `_OTHER_VALUE` rather than `CACHE_FIELDS`, so dropping a field from
    `CACHE_FIELDS` fails here instead of quietly deleting its own test.
    """
    assert _uid(baseline) != _uid(baseline, **{field: _OTHER_VALUE[field]})


def test_every_cache_field_is_probed():
    """The other direction: a field added to `CACHE_FIELDS` needs a probe value above."""
    assert set(DenseArrayEvent.CACHE_FIELDS) == set(_OTHER_VALUE)


def test_cache_fields_are_real_fields():
    """A typo in `CACHE_FIELDS` would silently drop a field from the key."""
    fields = set(DenseArrayEvent.model_fields) | {"frequency", "duration", "start"}
    assert set(DenseArrayEvent.CACHE_FIELDS) <= fields


def test_processing_is_not_in_the_cache_key(baseline):
    """`processing` is documentation, so it must not cost a corpus-wide recompute."""
    assert _uid(baseline) == _uid(baseline, processing="log-transformed")


def test_item_uid_carries_the_implementation_fingerprint(baseline):
    """Changing the conversion or resampling code re-keys the cache, so a stale entry
    becomes a miss rather than a wrong array."""
    from neuralshed.extractors import _IMPL_HASH

    assert _uid(baseline).startswith(f"{_IMPL_HASH}|")
    assert len(_IMPL_HASH) == 8
