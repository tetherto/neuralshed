"""Collection structure and automatic dense-data layout assignment."""

import pandas as pd
import pytest

import neuralset as ns
from neuralshed.collection import Collection
from neuralshed.transforms import (
    DATA_LAYOUT_KEY,
    DATA_ROLE,
    N_FEATURES,
    AssignDataLayouts,
    CombineStudies,
    DummyTransform,
)


class _LayoutFrame(ns.Step):
    two_sessions: bool = False
    reverse_second: bool = False
    second_has_input: bool = True
    second_rate: float = 100.0
    second_n_features: int = 3
    include_channel_names: bool = True

    def _run(self):
        sessions = ("s1", "s2") if self.two_sessions else ("s1",)
        rows = []
        for index, session in enumerate(sessions):
            common = {
                "start": 0.0,
                "duration": 1.0,
                "timeline": f"timeline-{session}",
                "study": "LayoutStudy",
                "dataset_id": "layout_data",
                "participant_id": "consortium/P1",
                "subject": "LayoutStudy/P1",
                "session": session,
                "frequency": self.second_rate if index else 50.0,
            }
            n_features = self.second_n_features if index else 3
            spike_names = [f"c{channel}" for channel in range(n_features)]
            if index and self.reverse_second:
                spike_names.reverse()
            if index == 0 or self.second_has_input:
                spike = {"type": "SpikeCounts", **common}
                spike["n_features"] = n_features
                if self.include_channel_names:
                    spike["ch_names"] = ",".join(spike_names)
                rows.append(spike)
            acoustic = {"type": "AcousticFeatures", "n_features": 2, **common}
            if self.include_channel_names:
                acoustic["ch_names"] = "a0,a1"
            rows.extend(
                [
                    acoustic,
                    {"type": "Sentence", "text": "hello", **common},
                ]
            )
        return pd.DataFrame(rows)


def _collection(source=None, **layout_kwargs):
    return Collection(
        steps={
            "CombineStudies": CombineStudies(
                chains={"member-a": source or _LayoutFrame()}
            ),
            "AssignDataLayouts": AssignDataLayouts(
                inputs=("SpikeCounts",),
                targets=("AcousticFeatures", "Sentence"),
                **layout_kwargs,
            ),
        }
    )


def test_collection_run_returns_annotated_and_validated_events():
    events = _collection().run()

    spikes = events[events["type"] == "SpikeCounts"]
    acoustic = events[events["type"] == "AcousticFeatures"]
    sentence = events[events["type"] == "Sentence"]

    assert spikes[DATA_ROLE].unique().tolist() == ["input"]
    assert acoustic[DATA_ROLE].unique().tolist() == ["target"]
    assert sentence[DATA_ROLE].unique().tolist() == ["target"]
    assert spikes[N_FEATURES].astype(int).unique().tolist() == [3]
    assert acoustic[N_FEATURES].astype(int).unique().tolist() == [2]
    assert sentence[DATA_LAYOUT_KEY].str.endswith("/type=Sentence").all()
    assert sentence[N_FEATURES].isna().all()
    assert spikes[DATA_LAYOUT_KEY].iat[0].endswith("/type=SpikeCounts")
    assert "participant_id=consortium%2FP1" in spikes[DATA_LAYOUT_KEY].iat[0]

    assert "data_rate" not in events
    assert "data_ch_names" not in events
    assert "data_channels" not in events
    assert "ch_names" in events


def test_a_stream_declared_on_both_sides_gets_both_role():
    transform = AssignDataLayouts(inputs=("SpikeCounts",), targets=("SpikeCounts",))
    events = CombineStudies(chains={"member-a": _LayoutFrame()})._run()
    out = transform._run(events)
    assert out.loc[out["type"] == "SpikeCounts", DATA_ROLE].unique().tolist() == [
        "both"
    ]


def test_native_rate_does_not_split_or_invalidate_a_data_layout():
    collection = _collection(
        source=_LayoutFrame(two_sessions=True, second_rate=100.0),
        group_by=("participant_id", "collection_member"),
    )
    events = collection.run()
    spikes = events[events["type"] == "SpikeCounts"]
    assert set(spikes["frequency"].astype(float)) == {50.0, 100.0}
    assert spikes[DATA_LAYOUT_KEY].nunique() == 1


def test_channel_ordering_conflict_fails_automatically_during_collection_run():
    collection = _collection(
        source=_LayoutFrame(two_sessions=True, reverse_second=True),
        group_by=("participant_id", "collection_member"),
    )
    with pytest.raises(ValueError, match="mixes channel orderings"):
        collection.run()


def test_n_features_validates_but_does_not_implicitly_split_a_layout():
    collection = _collection(
        source=_LayoutFrame(two_sessions=True, second_n_features=4),
        group_by=("participant_id", "collection_member"),
    )
    with pytest.raises(ValueError, match="mixes known n_features.*group_by"):
        collection.run()


def test_n_features_only_splits_a_layout_when_explicitly_grouped():
    collection = _collection(
        source=_LayoutFrame(two_sessions=True, second_n_features=4),
        group_by=("participant_id", "collection_member", N_FEATURES),
    )
    with pytest.warns(
        UserWarning,
        match="omitted group_by column 'n_features'.*event type 'Sentence'",
    ):
        events = collection.run()
    spikes = events.query("type == 'SpikeCounts'")
    sentences = events.query("type == 'Sentence'")
    assert spikes[DATA_LAYOUT_KEY].nunique() == 2
    assert spikes[DATA_LAYOUT_KEY].str.contains("n_features=").all()
    assert ~sentences[DATA_LAYOUT_KEY].str.contains("n_features=").any()


def test_partially_populated_grouping_column_is_rejected_per_event_type():
    events = CombineStudies(chains={"member-a": _LayoutFrame(two_sessions=True)})._run()
    sentences = events["type"].eq("Sentence")
    events.loc[sentences, "language"] = ["en", pd.NA]
    transform = AssignDataLayouts(
        inputs=("Sentence",),
        group_by=("participant_id", "collection_member", "language"),
    )
    with pytest.raises(ValueError, match="language.*partially populated.*Sentence"):
        transform._run(events)


def test_channel_names_are_optional_layout_evidence():
    events = _collection(source=_LayoutFrame(include_channel_names=False)).run()
    spikes = events.query("type == 'SpikeCounts'")
    assert spikes[DATA_LAYOUT_KEY].str.len().gt(0).all()
    assert spikes[N_FEATURES].astype(int).unique().tolist() == [3]


def test_every_recording_group_needs_declared_input_data():
    collection = _collection(
        source=_LayoutFrame(two_sessions=True, second_has_input=False)
    )
    with pytest.raises(ValueError, match="have no declared input data"):
        collection.run()


def test_collection_requires_fan_in_first_and_layout_assignment_last():
    layout = AssignDataLayouts(inputs=("SpikeCounts",))
    fan_in = CombineStudies(chains={"member-a": _LayoutFrame()})
    with pytest.raises(ValueError, match="steps cannot be empty"):
        Collection(steps=[])
    with pytest.raises(ValueError, match="start with CombineStudies"):
        Collection(steps=[DummyTransform(), layout])
    with pytest.raises(ValueError, match="end with AssignDataLayouts"):
        Collection(steps=[fan_in, layout, DummyTransform()])
    with pytest.raises(ValueError, match="exactly one"):
        Collection(steps=[fan_in, DummyTransform()])


def test_collection_round_trips_as_a_neuroai_step():
    collection = _collection()
    restored = ns.Step(**collection.model_dump())
    assert isinstance(restored, Collection)
    assert restored.layout_step.inputs == ("SpikeCounts",)
    assert restored.layout_step.targets == ("AcousticFeatures", "Sentence")


def test_collection_does_not_replace_neuroais_default_chain_coercion():
    assert ns.Step._exca_chain_class is ns.Chain


def test_non_dense_event_types_are_valid_targets():
    transform = AssignDataLayouts(inputs=("SpikeCounts",), targets=("Sentence",))
    assert transform.targets == ("Sentence",)


def test_unknown_event_type_is_rejected():
    with pytest.raises(KeyError, match="unknown event type"):
        AssignDataLayouts(inputs=("NotAnEvent",))
