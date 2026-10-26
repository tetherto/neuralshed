"""`event_types` declarations: keys must map to columns, dtypes must be real dtypes."""

import copy
import importlib

import pytest

from neuralshed.database import vocab
from neuralshed.database.populate import (
    _KNOWN_EVENT_KEYS,
    _event_rows,
    discover_datasets,
)
from neuralshed.database.validate import validate_metadata


@pytest.fixture(scope="module")
def catalogs():
    out = {}
    for name in discover_datasets():
        mod = importlib.import_module(f"neuralshed.datasets.{name}.metadata")
        out[name] = mod.DATASET_METADATA
    assert out, "no dataset metadata modules found"
    return out


def test_no_event_key_without_a_column(catalogs):
    """A key with no `event_types` column is dropped on the floor, so forbid it."""
    stray = {
        f"{name}.{ev.get('name')}: {sorted(set(ev) - _KNOWN_EVENT_KEYS)}"
        for name, meta in catalogs.items()
        for ev in _event_rows(meta)
        if set(ev) - _KNOWN_EVENT_KEYS
    }
    assert not stray, "event keys with no column in event_types: " + ", ".join(
        sorted(stray)
    )


def test_every_declared_dtype_is_a_real_dtype(catalogs):
    """`dtype` is the source's own dtype, so `float`/`int` are not answers."""
    bad = {
        f"{name}.{ev.get('name')}: {part!r}"
        for name, meta in catalogs.items()
        for ev in _event_rows(meta)
        if ev.get("dtype")
        for part in str(ev["dtype"]).split("/")
        if part.strip() not in vocab.DTYPES
    }
    assert not bad, "dtype values not in vocab.DTYPES: " + ", ".join(sorted(bad))


# --------------------------------------------------------------------------- #
# the guards themselves
# --------------------------------------------------------------------------- #


@pytest.fixture
def meta():
    mod = importlib.import_module("neuralshed.datasets.wilson_2025.metadata")
    return mod.get_metadata()


def _warnings(meta, **event_updates):
    m = copy.deepcopy(meta)
    m["events"][0].update(event_updates)
    result = validate_metadata(m, name="probe")
    assert result.ok, result.error
    return result.warnings


def test_baseline_is_warning_free(meta):
    assert _warnings(meta) == []


def test_unknown_event_key_warns(meta):
    (warning,) = _warnings(meta, unit="Hz")
    assert "no column in event_types" in warning and "unit" in warning


def test_unknown_dtype_warns(meta):
    (warning,) = _warnings(meta, dtype="float")
    assert "dtype='float'" in warning


def test_slash_separated_dtype_checks_each_part(meta):
    assert _warnings(meta, dtype="uint8/int8") == []
    (warning,) = _warnings(meta, dtype="uint8/int")
    assert "dtype='int'" in warning


def test_every_array_declares_its_hemisphere(catalogs):
    missing = [
        f"{name}/{subject['label']}/{array['name']}"
        for name, meta in catalogs.items()
        for subject in meta["subjects"]
        for array in subject.get("arrays", [])
        if "hemisphere" not in array
    ]
    assert not missing, "arrays without a hemisphere key: " + ", ".join(missing)


def _two_arrays(first: str | None, second: str | None) -> dict:
    arrays = [
        {"name": "6d", "hemisphere": h, "device_type": "utah_array"}
        for h in (first, second)
    ]
    subject = {"label": "S1", "arrays": arrays, "brain_areas": []}
    return {"id": "fake", "title": "fake", "subjects": [subject], "events": []}


def test_a_bilateral_pair_shares_a_name():
    assert validate_metadata(_two_arrays("left", "right")).ok


@pytest.mark.parametrize("hemisphere", ["left", None])
def test_a_repeated_array_is_rejected(hemisphere):
    result = validate_metadata(_two_arrays(hemisphere, hemisphere))
    assert not result.ok and "two arrays named '6d'" in result.error
