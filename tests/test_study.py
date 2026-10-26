"""Shared neuralshed Study provenance, identity, hooks, and validation."""

import importlib
import typing as tp

import pandas as pd
import pytest
from neuralset.events import study as ns_study
from pydantic import ValidationError

from neuralshed.database.populate import discover_datasets
from neuralshed.datasets.study import NeuralshedStudy
from neuralshed.events import PositionSeries, SpikeCounts


class _SyntheticStudy(NeuralshedStudy):
    dataset_id: tp.ClassVar[str] = "jude_2026_typing"

    def iter_timelines(self):
        yield {"subject": "T17", "session": "one"}

    def _load_timeline_events(self, timeline):
        return pd.DataFrame(
            [{"type": "Sentence", "start": 0.0, "duration": 1.0, "text": "hello"}]
        )

    def _download(self) -> None:
        pass


class _HookedSyntheticStudy(_SyntheticStudy):
    def _postprocess_timeline(self, events, timeline):
        events = events.copy()
        events.loc[:, "hook_subject"] = timeline["subject"]
        return events


class _InvalidHookStudy(_SyntheticStudy):
    def _postprocess_timeline(self, events, timeline):
        events = events.copy()
        events.loc[:, "participant_id"] = "unknown/person"
        return events


def test_shared_base_stays_out_of_neuroai_concrete_registry():
    assert NeuralshedStudy.__name__ == "_NeuralshedStudy"
    assert "_NeuralshedStudy" not in ns_study.STUDIES


def test_willet_threshold_variant_swaps_the_emitted_stream():
    from neuralshed.datasets.willet_2023.study import (
        Willett2023Sentences,
        _feature_specs,
    )

    def spike_row(variant):
        return next(spec for key, spec in _feature_specs(variant) if key == variant)

    # default stays tx2 with the metadata every existing catalog was built with
    assert Willett2023Sentences.model_fields["threshold_variant"].default == "tx2"
    tx2 = spike_row("tx2")
    assert tx2["threshold"] == "-4.5 x RMS"
    assert "thresholds at -4.5 x RMS, also refit per session" in tx2["processing"]
    # tx1 (the authors' released baseline feature) carries its own honest threshold
    tx1 = spike_row("tx1")
    assert tx1["threshold"] == "-3.5 x RMS"
    assert "thresholds at -3.5 x RMS" in tx1["processing"]
    # the spike-band-power stream is identical for both variants
    pow2 = next(spec for key, spec in _feature_specs("tx2") if key == "spikePow")
    pow1 = next(spec for key, spec in _feature_specs("tx1") if key == "spikePow")
    assert pow1 == pow2


def test_every_dataset_study_uses_its_metadata_participant_mapping():
    for dataset_id in discover_datasets():
        module = importlib.import_module(f"neuralshed.datasets.{dataset_id}.study")
        metadata = importlib.import_module(
            f"neuralshed.datasets.{dataset_id}.metadata"
        ).DATASET_METADATA
        classes = [
            value
            for value in vars(module).values()
            if isinstance(value, type)
            and issubclass(value, NeuralshedStudy)
            and value is not NeuralshedStudy
            and value.__module__ == module.__name__
        ]
        assert len(classes) == 1, dataset_id
        study_class = classes[0]
        assert study_class.dataset_id == dataset_id
        assert study_class._participant_map() == {
            str(subject["label"]): str(subject["participant_id"])
            for subject in metadata["subjects"]
        }


def test_neuralshed_study_defaults_to_inline_timeline_loading(tmp_path):
    instance = _SyntheticStudy(path=tmp_path)
    assert instance.timelines.infra is None


def test_timeline_events_gain_dataset_and_multi_participant_provenance(tmp_path):
    instance = _SyntheticStudy(path=tmp_path)

    t17 = instance._load_one({"subject": "T17", "session": "one"})
    t18 = instance._load_one({"subject": "T18", "session": "two"})

    assert t17["dataset_id"].unique().tolist() == ["jude_2026_typing"]
    assert t18["dataset_id"].unique().tolist() == ["jude_2026_typing"]
    assert t17["participant_id"].unique().tolist() == ["braingate2/T17"]
    assert t18["participant_id"].unique().tolist() == ["braingate2/T18"]
    assert t17["subject"].unique().tolist() == ["_SyntheticStudy/T17"]
    assert t18["subject"].unique().tolist() == ["_SyntheticStudy/T18"]

    timeline_id = t17["timeline"].iat[0]
    assert "dataset_id" not in timeline_id
    assert "participant_id" not in timeline_id


def test_unknown_timeline_subject_is_rejected(tmp_path):
    instance = _SyntheticStudy(path=tmp_path)
    with pytest.raises(ValueError, match="not declared.*T17.*T18"):
        instance._load_one({"subject": "T19", "session": "one"})


def test_postprocess_hook_receives_enriched_events(tmp_path):
    events = _HookedSyntheticStudy(path=tmp_path)._load_one(
        {"subject": "T18", "session": "one"}
    )
    assert events["hook_subject"].unique().tolist() == ["T18"]
    assert events["participant_id"].unique().tolist() == ["braingate2/T18"]


def test_validation_runs_after_postprocessing(tmp_path):
    instance = _InvalidHookStudy(path=tmp_path)
    with pytest.raises(ValueError, match="participant_id"):
        instance._load_one({"subject": "T17", "session": "one"})


def test_dense_array_events_need_positive_duration_and_frequency(tmp_path):
    instance = _SyntheticStudy(path=tmp_path)
    events = pd.DataFrame(
        [
            {
                "type": "SpikeCounts",
                "start": 0.0,
                "duration": 1.0,
                "frequency": 0.0,
                "timeline": "timeline",
                "study": "_SyntheticStudy",
                "subject": "_SyntheticStudy/T17",
                "dataset_id": "jude_2026_typing",
                "participant_id": "braingate2/T17",
            }
        ]
    )
    with pytest.raises(ValueError, match="positive duration and frequency"):
        instance._validate_timeline_events(
            events,
            {"subject": "T17", "session": "one"},
            "braingate2/T17",
        )


def test_dense_array_event_requires_a_positive_feature_count(tmp_path):
    common = dict(
        timeline="timeline",
        subject="subject",
        start=0.0,
        duration=1.0,
        frequency=1.0,
        filepath=str(tmp_path / "spikes.npy"),
    )
    with pytest.raises(ValidationError, match="n_features"):
        SpikeCounts(**common)
    with pytest.raises(ValidationError, match="n_features"):
        SpikeCounts(**common, n_features=0)


def test_behavioral_series_requires_feature_count_matching_dims(tmp_path):
    path = tmp_path / "position.npy"
    path.touch()
    event = PositionSeries(
        timeline="timeline",
        subject="subject",
        start=0.0,
        duration=1.0,
        frequency=1.0,
        filepath=str(path),
        dims=("x", "y"),
        n_features=2,
    )
    assert event.n_features == 2

    with pytest.raises(ValidationError, match="n_features"):
        PositionSeries(
            timeline="timeline",
            subject="subject",
            start=0.0,
            duration=1.0,
            frequency=1.0,
            filepath=str(path),
            dims=("x", "y"),
        )


class _TypedTimelineStudy(_SyntheticStudy):
    def iter_timelines(self):
        yield {"subject": "T17", "session": "one", "block": 3}
        yield {
            "subject": "T17",
            "session": "two",
            "block": 4,
            "valid": False,
            "valid_description": "array collapsed",
        }


def test_timelines_default_valid_and_keep_it_out_of_the_timeline_id(tmp_path):
    instance = _TypedTimelineStudy(path=tmp_path)
    summary = instance.study_summary(apply_query=False)
    assert summary["valid"].tolist() == [True, False]
    assert summary["valid_description"].tolist() == ["", "array collapsed"]
    identity = {"subject": "T17", "session": "one", "block": 3}
    flagged = {**identity, "valid": False, "valid_description": "array collapsed"}
    assert instance._to_timeline_string(flagged) == instance._to_timeline_string(
        identity
    )
    assert "valid" not in instance._to_timeline_string(flagged)


def test_query_on_valid_selects_usable_timelines(tmp_path):
    instance = _TypedTimelineStudy(path=tmp_path, query="valid")
    events = instance.run()
    assert events["session"].unique().tolist() == ["one"]
    assert events["valid"].dtype == bool


def test_timeline_dict_values_keep_their_types_on_the_events_frame(tmp_path):
    events = _TypedTimelineStudy(path=tmp_path).run()
    assert events["block"].dtype.kind == "i"
    assert events["block"].tolist() == [3, 4]
    assert events["valid"].dtype == bool
    assert events["valid"].tolist() == [True, False]
    assert events["valid_description"].tolist() == ["", "array collapsed"]


class _RowFlagStudy(_SyntheticStudy):
    def _load_timeline_events(self, timeline):
        return pd.DataFrame(
            [
                {"type": "Sentence", "start": 0.0, "duration": 1.0, "text": "a"},
                {
                    "type": "Sentence",
                    "start": 1.0,
                    "duration": 1.0,
                    "text": "b",
                    "valid": False,
                    "valid_description": "cut short",
                },
            ]
        )


def test_row_level_valid_overrides_the_timeline_default(tmp_path):
    events = _RowFlagStudy(path=tmp_path).run()
    assert events["valid"].dtype == bool
    assert events["valid"].tolist() == [True, False]
    assert events["valid_description"].tolist() == ["", "cut short"]
