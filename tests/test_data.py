"""Declarative collection-to-DataLoader configuration boundaries."""

import neuralset as ns
import numpy as np
import pandas as pd
import pytest

from neuralshed.collection import Collection
from neuralshed.data import (
    BICData,
    BuiltData,
    CollectionData,
    DataContract,
    MetadataBatchSamplerConfig,
    RaggedCollatorConfig,
    StratifiedBatchSamplerConfig,
)
from neuralshed.dataloader import (
    MetadataBatchSampler,
    PadSpec,
    RaggedCollator,
    RaggedSegmenter,
    StratifiedBatchSampler,
)
from neuralshed.extractors import CharacterEncoder, NeuralFeatureExtractor
from neuralshed.transforms import AssignDataLayouts, CombineStudies


class _DataFrameStep(ns.Step):
    filepath: str = ""

    def _run(self):
        return pd.DataFrame(
            [
                {
                    "type": "SpikeCounts",
                    "timeline": "recording",
                    "start": 0.0,
                    "duration": 3.0,
                    "frequency": 10.0,
                    "filepath": self.filepath,
                    "subject": "P1",
                    "participant_id": "P1",
                    "session": "S1",
                    "feature": "threshold_crossings",
                    "n_features": 2,
                    "unit": "counts/bin",
                    "normalization": "raw",
                },
                {
                    "type": "Sentence",
                    "timeline": "recording",
                    "start": 0.0,
                    "duration": 1.0,
                    "participant_id": "P1",
                    "session": "S1",
                    "text": "hi",
                    "split": "train",
                },
                {
                    "type": "Sentence",
                    "timeline": "recording",
                    "start": 1.0,
                    "duration": 1.0,
                    "participant_id": "P1",
                    "session": "S1",
                    "text": "hello",
                    "split": "val",
                },
                {
                    "type": "Sentence",
                    "timeline": "recording",
                    "start": 2.0,
                    "duration": 1.0,
                    "participant_id": "P1",
                    "session": "S1",
                    "text": "yes",
                    "split": "test",
                },
            ]
        )


def _collection(
    filepath="",
    *,
    inputs=("SpikeCounts",),
    targets=("Sentence",),
):
    return Collection(
        steps={
            "CombineStudies": CombineStudies(
                chains={"synthetic": _DataFrameStep(filepath=filepath)}
            ),
            "AssignDataLayouts": AssignDataLayouts(
                inputs=inputs,
                targets=targets,
                group_by=("participant_id", "collection_member", "session"),
            ),
        }
    )


def _segmenter(metadata_columns=None):
    if metadata_columns is None:
        metadata_columns = (
            "participant_id",
            "collection_member",
            "session",
            "split",
        )
    return RaggedSegmenter(
        extractors={
            "spikes": NeuralFeatureExtractor(event_types="SpikeCounts"),
            "target": CharacterEncoder(
                event_types="Sentence",
                aggregation="trigger",
            ),
        },
        trigger_query="type == 'Sentence'",
        duration=None,
        metadata_columns=metadata_columns,
    )


def _bic_data(*, batch_size=4, **kwargs):
    values = {
        "collection": _collection(),
        "segmenter": _segmenter(),
        "batch_sampler": MetadataBatchSamplerConfig(
            group_by=("participant_id", "collection_member", "session"),
            batch_size=batch_size,
        ),
    }
    values.update(kwargs)
    return BICData(**values)


def test_bic_data_packages_the_notebook_configuration_and_round_trips():
    config = _bic_data()
    restored = BICData.model_validate(config.model_dump())

    assert isinstance(restored.collection, Collection)
    assert isinstance(restored.segmenter, RaggedSegmenter)
    assert restored.segmenter.extractors.keys() == {"spikes", "target"}
    assert restored.batch_sampler is not None
    assert restored.batch_sampler.group_by == (
        "participant_id",
        "collection_member",
        "session",
    )
    assert restored.batch_sampler.batch_size == 4
    assert restored.collator.fields == {
        "spikes": PadSpec(axis=-1, pad_value=0),
        "target": PadSpec(axis=0, pad_value=0),
    }


def test_component_configs_construct_the_existing_runtime_objects():
    config = _bic_data()
    collator = config.collator.make()
    assert isinstance(collator, RaggedCollator)

    with pytest.raises(ValueError, match="batch_size must be a positive int"):
        MetadataBatchSamplerConfig(group_by=("session",), batch_size=0)

    class Dataset:
        metadata = pd.DataFrame(
            {
                "participant_id": ["P1", "P1", "P2"],
                "collection_member": ["a", "a", "b"],
                "session": ["S1", "S1", "S2"],
            }
        )

        def __len__(self):
            return len(self.metadata)

    assert config.batch_sampler is not None
    sampler = config.batch_sampler.make(Dataset(), shuffle=False)
    assert sampler.batch_size == 4
    assert list(sampler) == [[0, 1], [2]]


def test_build_executes_the_generic_pipeline_and_resolves_its_contract(tmp_path):
    filepath = tmp_path / "spikes.npy"
    np.save(filepath, np.arange(60, dtype=np.float32).reshape(2, 30))
    config = _bic_data(collection=_collection(str(filepath)), batch_size=2)

    built = config.build()

    assert isinstance(built, BuiltData)
    assert len(built.dataset) == 3
    assert {name: len(dataset) for name, dataset in built.split_datasets.items()} == {
        "train": 1,
        "val": 1,
        "test": 1,
    }
    assert built["train"] is built.loaders["train"]
    assert "splits: train=1, val=1, test=1" in repr(built)
    assert built.loaders["train"].batch_sampler.shuffle is True
    assert built.loaders["val"].batch_sampler.shuffle is False

    batch = next(iter(built.loaders["train"]))
    assert batch.data["spikes"].shape == (1, 2, 10)
    assert batch.data["target"].shape == (1, 2)

    contract = built.contract
    assert contract.layout_group_by == (
        "participant_id",
        "collection_member",
        "session",
    )
    assert contract.batch_group_by == contract.layout_group_by
    assert contract.splits == ("train", "val", "test")
    spikes = contract.fields["spikes"]
    assert spikes.event_types == ("SpikeCounts",)
    assert spikes.roles == ("input",)
    assert len(spikes.layout_keys) == 1
    assert spikes.feature_widths == {spikes.layout_keys[0]: 2}
    assert spikes.padding == PadSpec(axis=-1, pad_value=0)
    target = contract.fields["target"]
    assert target.roles == ("target",)
    assert len(target.layout_keys) == 1
    assert target.feature_widths == {}
    assert DataContract.model_validate(contract.model_dump()) == contract


def test_collection_data_builds_native_batching_without_metadata_sampler(tmp_path):
    filepath = tmp_path / "spikes.npy"
    np.save(filepath, np.arange(60, dtype=np.float32).reshape(2, 30))
    config = CollectionData(
        collection=_collection(str(filepath)),
        segmenter=_segmenter(),
        collator=RaggedCollatorConfig(
            fields={
                "spikes": PadSpec(axis=-1),
                "target": PadSpec(axis=0),
            }
        ),
        standard_batch_size=2,
        shuffle_splits=(),
    )

    built = config.build()

    assert built.contract.batch_group_by == ()
    assert built.loaders["train"].batch_size == 2
    assert next(iter(built.loaders["val"])).metadata["split"] == ["val"]


def test_build_requires_every_requested_split_to_have_segments(tmp_path):
    filepath = tmp_path / "spikes.npy"
    np.save(filepath, np.zeros((2, 30), dtype=np.float32))
    config = _bic_data(
        collection=_collection(str(filepath)),
        splits=("train", "missing"),
        shuffle_splits=("train",),
    )

    with pytest.raises(ValueError, match="splits contain no segments.*missing"):
        config.build()


def test_bic_data_validates_resolved_input_and_target_roles(tmp_path):
    filepath = tmp_path / "spikes.npy"
    np.save(filepath, np.zeros((2, 30), dtype=np.float32))
    config = _bic_data(
        collection=_collection(
            str(filepath),
            inputs=("Sentence",),
            targets=("SpikeCounts",),
        )
    )

    with pytest.raises(ValueError, match="input field 'spikes'.*input data role"):
        config.build()


def test_collection_data_validates_cross_component_names_and_metadata():
    with pytest.raises(ValueError, match="collator fields.*missing"):
        CollectionData(
            collection=_collection(),
            segmenter=_segmenter(),
            collator=RaggedCollatorConfig(fields={"missing": PadSpec(axis=0)}),
        )

    with pytest.raises(ValueError, match="metadata_columns.*session"):
        BICData(
            collection=_collection(),
            segmenter=_segmenter(
                metadata_columns=("participant_id", "collection_member", "split")
            ),
            batch_sampler=MetadataBatchSamplerConfig(
                group_by=("participant_id", "collection_member", "session")
            ),
        )


def test_bic_data_validates_its_neural_input_and_character_target_contract():
    segmenter = _segmenter()
    segmenter.extractors["target"] = NeuralFeatureExtractor(event_types="SpikeCounts")

    with pytest.raises(TypeError, match="target extractor.*CharacterEncoder"):
        BICData(
            collection=_collection(),
            segmenter=segmenter,
            batch_sampler=MetadataBatchSamplerConfig(
                group_by=("participant_id", "collection_member", "session")
            ),
        )


class _SamplerDataset:
    metadata = pd.DataFrame(
        {
            "participant_id": ["P1"] * 3 + ["P2"] * 5,
            "collection_member": ["m1"] * 3 + ["m2"] * 5,
            "session": ["s1", "s2", "s3", "s1", "s2", "s3", "s4", "s5"],
        }
    )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        return index


def test_stratified_sampler_config_constructs_each_runtime_object():
    config = StratifiedBatchSamplerConfig(
        group_by=("collection_member", "session"),
        weights_by=("collection_member",),
        batch_size=4,
        days_per_batch=2,
        seed=7,
    )

    train_sampler = config.make(_SamplerDataset(), shuffle=True)
    assert isinstance(train_sampler, StratifiedBatchSampler)

    # Eval splits delegate to the homogeneous sampler: a with-replacement day
    # policy must never estimate a metric from a sample of the eval set.
    eval_sampler = config.make(_SamplerDataset(), shuffle=False)
    assert isinstance(eval_sampler, MetadataBatchSampler)
    batches = list(eval_sampler)
    assert sorted(index for batch in batches for index in batch) == list(range(8))


def test_stratified_sampler_config_validates_its_policy():
    with pytest.raises(ValueError, match="divisible"):
        StratifiedBatchSamplerConfig(
            group_by=("collection_member",), batch_size=6, days_per_batch=4
        )
    with pytest.raises(ValueError, match="at least one"):
        StratifiedBatchSamplerConfig(group_by=(), days_per_batch=2)
    with pytest.raises(ValueError, match="weights_by contains duplicates"):
        StratifiedBatchSamplerConfig(
            group_by=("collection_member",),
            weights_by=("collection_member", "collection_member"),
            days_per_batch=2,
        )


def test_collection_data_requires_stratified_columns_in_metadata_columns():
    # `weights_by` columns are read from dataset metadata just like
    # `group_by`, so the segmenter must be configured to preserve them.
    with pytest.raises(ValueError, match="metadata_columns.*session"):
        BICData(
            collection=_collection(),
            segmenter=_segmenter(
                metadata_columns=("participant_id", "collection_member", "split")
            ),
            batch_sampler=StratifiedBatchSamplerConfig(
                group_by=("participant_id", "collection_member"),
                weights_by=("session",),
                days_per_batch=2,
            ),
        )
