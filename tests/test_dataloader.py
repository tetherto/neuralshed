"""Variable-length segmentation at the local neuralshed/neuralset boundary."""

import pickle

import neuralset as ns
import neuralset.segments as ns_segments
import numpy as np
import pandas as pd
import pytest
import torch
from neuralset.events.utils import standardize_events

from neuralshed.dataloader import (
    MetadataBatchSampler,
    PadSpec,
    RaggedBatch,
    RaggedCollator,
    RaggedSegmentDataset,
    RaggedSegmenter,
    SegmentSample,
    StratifiedBatchSampler,
    NeuralshedSegmenter,
)
from neuralshed.extractors import CharacterEncoder, NeuralFeatureExtractor


def _events(tmp_path) -> pd.DataFrame:
    path = tmp_path / "spikes.npy"
    np.save(path, np.arange(60, dtype=np.float32).reshape(2, 30))

    common = dict(
        timeline="recording",
        subject="P1",
        session="S1",
        study="SyntheticStudy",
        collection_member="synthetic",
        split="train",
    )
    rows = [
        dict(
            type="SpikeCounts",
            start=0.0,
            duration=3.0,
            frequency=10.0,
            filepath=str(path),
            feature="threshold_crossings",
            n_features=2,
            unit="counts/bin",
            normalization="raw",
            target_is_valid=False,
            **common,
        ),
        dict(
            type="Sentence",
            start=0.0,
            duration=1.0,
            text="hi",
            canonical_text="hi",
            target_is_valid=True,
            **common,
        ),
        dict(
            type="Sentence",
            start=1.0,
            duration=1.5,
            text="hello",
            canonical_text="hello",
            target_is_valid=True,
            **common,
        ),
    ]
    return standardize_events(pd.DataFrame(rows), auto_fill=False)


def _extractors():
    return {
        "spikes": NeuralFeatureExtractor(
            event_types="SpikeCounts",
            aggregation="cat",
            frequency="native",
            target_unit="native",
        ),
        "target": CharacterEncoder(
            event_types="Sentence",
            aggregation="trigger",
            text_field="canonical_text",
        ),
    }


def _ragged(events: pd.DataFrame) -> RaggedSegmentDataset:
    dataset = RaggedSegmenter(
        extractors=_extractors(),
        trigger_query="target_is_valid == True",
        duration=None,
        metadata_columns=(
            "collection_member",
            "session",
            "split",
            "canonical_text",
        ),
    ).apply(events)
    dataset.prepare()
    return dataset


def test_local_hook_preserves_the_native_default(tmp_path):
    events = _events(tmp_path)
    dataset = NeuralshedSegmenter(
        extractors=_extractors(),
        trigger_query="target_is_valid == True",
        duration=None,
    ).apply(events)

    assert isinstance(dataset, ns.dataloader.SegmentDataset)
    assert [segment.duration for segment in dataset.segments] == [1.0, 1.5]


def test_ragged_items_keep_their_natural_lengths(tmp_path):
    dataset = _ragged(_events(tmp_path))

    short, long = dataset[0], dataset[1]
    assert short.data["spikes"].shape == (2, 10)
    assert long.data["spikes"].shape == (2, 15)
    assert short.data["target"].shape == (2,)
    assert long.data["target"].shape == (5,)
    assert short.segment.trigger is not None
    assert short.segment.trigger.extra["canonical_text"] == "hi"
    assert short.metadata == {
        "collection_member": "synthetic",
        "session": "S1",
        "split": "train",
        "canonical_text": "hi",
    }
    assert dataset.metadata.to_dict("records") == [
        short.metadata,
        long.metadata,
    ]

    # The native dataset rejects these durations before loading either item;
    # the ragged view is precisely the boundary that defers padding.
    with pytest.raises(ValueError, match="different durations"):
        dataset.source[0]


def test_ragged_selection_and_step_round_trip(tmp_path):
    dataset = _ragged(_events(tmp_path))
    selected = dataset.select(1)
    assert isinstance(selected, RaggedSegmentDataset)
    assert len(selected) == 1
    assert selected[0].data["spikes"].shape[-1] == 15

    segmenter = RaggedSegmenter(
        extractors=_extractors(),
        trigger_query="target_is_valid == True",
        duration=None,
        metadata_columns=("session", "split"),
    )
    restored = ns.Step(**segmenter.model_dump())
    assert isinstance(restored, RaggedSegmenter)
    assert restored.metadata_columns == ("session", "split")
    assert selected.metadata["canonical_text"].tolist() == ["hello"]


def test_ragged_segmenter_rejects_missing_metadata_columns(tmp_path):
    with pytest.raises(KeyError, match="missing_column"):
        RaggedSegmenter(
            extractors=_extractors(),
            trigger_query="target_is_valid == True",
            duration=None,
            metadata_columns=("missing_column",),
        ).apply(_events(tmp_path))


def test_ragged_dataset_pickle_keeps_the_native_event_store(tmp_path):
    dataset = _ragged(_events(tmp_path))
    blob = pickle.dumps(dataset)
    old_registry = dict(ns_segments._EventStore._REGISTRY)
    ns_segments._EventStore._REGISTRY.clear()
    try:
        restored = pickle.loads(blob)
        assert restored[1].data["spikes"].shape == (2, 15)
        assert restored[1].segment.trigger is not None
    finally:
        ns_segments._EventStore._REGISTRY.update(old_registry)


def test_ragged_collator_pads_each_configured_axis_with_lengths(tmp_path):
    dataset = _ragged(_events(tmp_path))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        collate_fn=RaggedCollator(
            fields={
                "spikes": PadSpec(axis=-1),
                "target": PadSpec(axis=0, pad_value=-1),
            }
        ),
    )

    batch = next(iter(loader))
    assert isinstance(batch, RaggedBatch)
    assert isinstance(batch, ns.dataloader.Batch)
    assert batch.data["spikes"].shape == (2, 2, 15)
    assert batch.data["target"].shape == (2, 5)
    assert batch.lengths["spikes"].tolist() == [10, 15]
    assert batch.lengths["target"].tolist() == [2, 5]
    assert batch.axes == {"spikes": 1, "target": 0}
    assert torch.count_nonzero(batch.data["spikes"][0, :, 10:]) == 0
    assert batch.data["target"][0, 2:].tolist() == [-1, -1, -1]
    assert batch.valid_mask("spikes").tolist() == [
        [True] * 10 + [False] * 5,
        [True] * 15,
    ]
    assert torch.equal(batch.padding_mask("spikes"), ~batch.valid_mask("spikes"))
    assert [segment.duration for segment in batch.segments] == [1.0, 1.5]
    assert batch.metadata == {
        "collection_member": ["synthetic", "synthetic"],
        "session": ["S1", "S1"],
        "split": ["train", "train"],
        "canonical_text": ["hi", "hello"],
    }


def test_ragged_batch_preserves_metadata_on_to_and_select(tmp_path):
    dataset = _ragged(_events(tmp_path))
    batch = RaggedCollator(
        fields={
            "spikes": PadSpec(axis=-1),
            "target": PadSpec(axis=0),
        }
    )([dataset[0], dataset[1]])

    moved = batch.to("cpu")
    assert isinstance(moved, RaggedBatch)
    assert moved.lengths["spikes"].tolist() == [10, 15]
    assert moved.axes == batch.axes
    assert moved.metadata == batch.metadata

    selected = batch.select(1)
    assert isinstance(selected, RaggedBatch)
    assert selected.data["spikes"].shape == (1, 2, 15)
    assert selected.lengths["spikes"].tolist() == [15]
    assert len(selected.segments) == 1
    assert selected.metadata["canonical_text"] == ["hello"]


def test_ragged_collator_rejects_an_incompatible_fixed_dimension(tmp_path):
    dataset = _ragged(_events(tmp_path))
    first, second = dataset[0], dataset[1]
    incompatible = SegmentSample(
        data={**second.data, "spikes": second.data["spikes"][:1]},
        segment=second.segment,
        metadata=second.metadata,
    )

    collator = RaggedCollator(fields={"spikes": PadSpec(axis=-1)})
    with pytest.raises(ValueError, match="only axis 1 may differ"):
        collator([first, incompatible])


def test_metadata_batch_sampler_groups_without_loading_items():
    class MetadataOnlyDataset:
        metadata = pd.DataFrame(
            {
                "member": ["a", "a", "b", "b", "b"],
                "session": [1, 1, 1, 1, 2],
            }
        )

        def __len__(self):
            return len(self.metadata)

        def __getitem__(self, index):
            raise AssertionError(f"sampler unexpectedly loaded item {index}")

    dataset = MetadataOnlyDataset()
    sampler = MetadataBatchSampler(
        dataset,
        group_by=("member", "session"),
        batch_size=2,
        shuffle=False,
    )

    batches = list(sampler)
    assert batches == [[0, 1], [2, 3], [4]]
    assert len(sampler) == 3
    for batch in batches:
        keys = {
            tuple(dataset.metadata.loc[index, ["member", "session"]]) for index in batch
        }
        assert len(keys) == 1


def test_metadata_batch_sampler_validates_grouping_metadata():
    class Dataset:
        metadata = pd.DataFrame({"group": ["a", None]})

        def __len__(self):
            return len(self.metadata)

    with pytest.raises(ValueError, match="missing values"):
        MetadataBatchSampler(Dataset(), group_by="group", batch_size=2)


def _stratified_metadata() -> pd.DataFrame:
    # Stratum "a": 4 rows over 2 days; stratum "b": 8 rows over 4 days.
    return pd.DataFrame(
        {
            "member": ["a"] * 4 + ["b"] * 8,
            "session": [1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4],
        }
    )


def test_stratified_batch_sampler_never_loads_items():
    class MetadataOnlyDataset:
        metadata = _stratified_metadata()

        def __len__(self):
            return len(self.metadata)

        def __getitem__(self, index):
            raise AssertionError(f"sampler unexpectedly loaded item {index}")

    sampler = StratifiedBatchSampler(
        MetadataOnlyDataset(),
        group_by=("member", "session"),
        weights_by="member",
        batch_size=4,
        days_per_batch=2,
    )
    batches = list(sampler)

    assert len(sampler) == 3
    assert len(batches) == len(sampler)
    for batch in batches:
        assert len(batch) == 4
        days = {
            tuple(sampler.dataset.metadata.loc[index, ("member", "session")])
            for index in batch
        }
        assert len(days) <= 2
        assert all(0 <= index < len(sampler.dataset) for index in batch)


def test_stratified_batch_sampler_reproduces_epochs_from_seed_and_epoch():
    class Dataset:
        metadata = _stratified_metadata()

        def __len__(self):
            return len(self.metadata)

        def __getitem__(self, index):
            return index

    def run(epoch: int) -> list[list[int]]:
        sampler = StratifiedBatchSampler(
            Dataset(),
            group_by=("member", "session"),
            weights_by="member",
            batch_size=4,
            days_per_batch=2,
            seed=7,
        )
        sampler.set_epoch(epoch)
        return list(sampler)

    assert run(3) == run(3)  # one (seed, epoch) pair yields one epoch
    assert run(0) != run(1)  # the epoch is the only input that moved

    # A sampler iterated without set_epoch replays what set_epoch(0) gives:
    # epoch 0 reseeds from `seed` alone, matching the old stream's start.
    fresh = StratifiedBatchSampler(
        Dataset(),
        group_by=("member", "session"),
        weights_by="member",
        batch_size=4,
        days_per_batch=2,
        seed=7,
    )
    assert list(fresh) == run(0)


def test_stratified_batch_sampler_weights_day_slots_by_stratum_size():
    class Dataset:
        metadata = _stratified_metadata()

        def __len__(self):
            return len(self.metadata)

        def __getitem__(self, index):
            return index

    sampler = StratifiedBatchSampler(
        Dataset(),
        group_by=("member", "session"),
        weights_by="member",
        batch_size=4,
        days_per_batch=2,
        seed=11,
    )
    draws = {"a": 0, "b": 0}
    for epoch in range(40):
        sampler.set_epoch(epoch)
        for batch in sampler:
            days = {
                tuple(sampler.dataset.metadata.loc[index, ("member", "session")])
                for index in batch
            }
            for member, _ in days:
                draws[member] += 1

    share = draws["b"] / (draws["a"] + draws["b"])
    # Stratum "b" holds 8 of 12 rows, so it should own roughly two thirds of
    # the day slots; the bounds keep a fair die from looking biased.
    assert 0.5 < share < 0.85


def test_stratified_batch_sampler_validates_batch_geometry_and_metadata():
    class Dataset:
        metadata = _stratified_metadata()

        def __len__(self):
            return len(self.metadata)

    with pytest.raises(ValueError, match="divisible"):
        StratifiedBatchSampler(
            Dataset(), group_by=("member", "session"), batch_size=5, days_per_batch=2
        )
    with pytest.raises(KeyError, match="missing"):
        StratifiedBatchSampler(
            Dataset(), group_by=("nope",), batch_size=4, days_per_batch=2
        )

    class NanDataset:
        metadata = pd.DataFrame({"member": ["a", "b"], "session": [1, None]})

        def __len__(self):
            return 2

    with pytest.raises(ValueError, match="missing values"):
        StratifiedBatchSampler(
            NanDataset(),
            group_by=("member", "session"),
            batch_size=2,
            days_per_batch=2,
        )

    sampler = StratifiedBatchSampler(
        Dataset(), group_by=("member", "session"), batch_size=4, days_per_batch=2
    )
    with pytest.raises(ValueError, match="non-negative"):
        sampler.set_epoch(-1)


# --- StratifiedBatchSampler: continuous seeding + multi-GPU replicas ----------


def _stratified_stub():
    """Small synthetic metadata table: 3 strata, ragged day counts, 600 rows."""
    rows = []
    sizes = {"a": (10, 30), "b": (4, 40), "c": (1, 200)}  # stratum: (days, trials/day)
    for member, (days, per_day) in sizes.items():
        for day in range(days):
            rows.extend({"member": member, "day": day} for _ in range(per_day))
    metadata = pd.DataFrame(rows)

    class _Stub:
        def __init__(self):
            self.metadata = metadata

        def __len__(self):
            return len(self.metadata)

    return _Stub()


def _stratified(ds, *, seed=11, batch=16, days=4, **kw):
    return StratifiedBatchSampler(
        ds,
        group_by=("member", "day"),
        weights_by=("member",),
        batch_size=batch,
        days_per_batch=days,
        seed=seed,
        **kw,
    )


def test_continuous_seeding_is_one_stream_across_epochs():
    ds = _stratified_stub()
    sampler = _stratified(ds, seeding="continuous")
    epoch0, epoch1 = list(sampler), list(sampler)
    assert epoch0 != epoch1  # the stream moves on, it does not restart
    replay = _stratified(ds, seeding="continuous")
    assert list(replay) == epoch0
    assert list(replay) == epoch1


def test_reseed_default_restarts_from_seed_plus_epoch():
    ds = _stratified_stub()
    sampler = _stratified(ds)  # default seeding="reseed": unchanged behavior
    epoch0 = list(sampler)
    epoch1 = list(sampler)
    sampler.set_epoch(0)
    assert list(sampler) == epoch0  # epoch 0 reproduces from (seed, 0)
    fresh = _stratified(ds)
    fresh.set_epoch(1)
    assert list(fresh) == epoch1


def test_replicas_partition_the_continuous_stream():
    ds = _stratified_stub()
    # Global 32/8 over 2 ranks draws the same sub-batches as a single 16/4.
    single = _stratified(ds, batch=16, days=4, seeding="continuous")
    full = list(single)
    r0 = _stratified(ds, batch=32, days=8, seeding="continuous", num_replicas=2, rank=0)
    r1 = _stratified(ds, batch=32, days=8, seeding="continuous", num_replicas=2, rank=1)
    got0, got1 = list(r0), list(r1)
    generated = 2 * len(got0)  # every rank draws the same 2-per-rank count
    assert got0 == full[0:generated:2]
    assert got1 == full[1:generated:2]
    assert len(r0) == len(got0) == len(single) // 2  # __len__ is the per-rank count
    if len(single) % 2:  # an odd tail sub-batch is dropped, never half-assigned
        assert len(got0) + len(got1) < len(single)


def test_replicas_stay_in_sync_across_epochs():
    ds = _stratified_stub()
    single = _stratified(ds, batch=16, days=4, seeding="continuous")
    full = list(single) + list(single)  # one stream, two epochs
    r0 = _stratified(ds, batch=32, days=8, seeding="continuous", num_replicas=2, rank=0)
    r1 = _stratified(ds, batch=32, days=8, seeding="continuous", num_replicas=2, rank=1)
    got0 = list(r0) + list(r0)
    got1 = list(r1) + list(r1)
    assert got0 == full[0 : 2 * len(got0) : 2]
    assert got1 == full[1 : 2 * len(got1) : 2]


def test_set_epoch_is_a_noop_under_continuous_seeding():
    ds = _stratified_stub()
    sampler = _stratified(ds, seeding="continuous")
    epoch0 = list(sampler)
    sampler.set_epoch(99)  # must not jump the stream
    epoch1 = list(sampler)
    assert epoch1 != epoch0
    naive = _stratified(ds, seeding="continuous")  # never touched by set_epoch
    assert list(naive) == epoch0
    assert list(naive) == epoch1


def test_replica_and_seeding_guards():
    ds = _stratified_stub()
    with pytest.raises(ValueError, match="seeding"):
        _stratified(ds, seeding="per-epoch-per-epoch")
    with pytest.raises(ValueError, match="num_replicas"):
        _stratified(ds, batch=32, days=8, num_replicas=3, rank=0)
    with pytest.raises(ValueError, match="num_replicas"):
        _stratified(ds, num_replicas=False)
    with pytest.raises(ValueError, match="rank"):
        _stratified(ds, batch=32, days=8, num_replicas=2, rank=2)
