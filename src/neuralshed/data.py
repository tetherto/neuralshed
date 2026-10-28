"""Serializable data-pipeline configurations for neuralshed experiments.

The classes in this module describe the pieces needed to turn a
:class:`~neuralshed.collection.Collection` into PyTorch batches.
"""

from __future__ import annotations

import dataclasses
import typing as tp

import pandas as pd
import pydantic
import torch

from neuralshed.collection import Collection
from neuralshed.dataloader import (
    MetadataBatchSampler,
    PadSpec,
    RaggedBatch,
    RaggedCollator,
    RaggedSegmentDataset,
    RaggedSegmenter,
    StratifiedBatchSampler,
)
from neuralshed.extractors import CharacterEncoder, NeuralFeatureExtractor
from neuralshed.transforms import DATA_LAYOUT_KEY, DATA_ROLE, N_FEATURES


class DataFieldContract(pydantic.BaseModel):
    """Resolved collection facts for one named extractor output."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    event_types: tuple[str, ...]
    aggregation: str
    query: str | None
    roles: tuple[str, ...]
    layout_keys: tuple[str, ...]
    feature_widths: dict[str, int]
    padding: PadSpec | None


class DataContract(pydantic.BaseModel):
    """Compact model-facing facts resolved from one executed data pipeline.

    This is deliberately descriptive: it contains no loaders, datasets, or
    learned adapters. A later model-specific layer can use these stable facts
    to construct its own adapter configuration.
    """

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    fields: dict[str, DataFieldContract]
    layout_group_by: tuple[str, ...]
    batch_group_by: tuple[str, ...]
    split_column: str
    splits: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class BuiltData:
    """Runtime result of building a :class:`CollectionData` configuration."""

    events: pd.DataFrame
    dataset: RaggedSegmentDataset
    split_datasets: dict[str, RaggedSegmentDataset]
    loaders: dict[str, torch.utils.data.DataLoader[RaggedBatch]]
    contract: DataContract

    def __post_init__(self) -> None:
        split_names = set(self.split_datasets)
        if split_names != set(self.loaders):
            raise ValueError("split_datasets and loaders must have the same split keys")
        if tuple(self.split_datasets) != self.contract.splits:
            raise ValueError(
                "runtime split order does not match the resolved data contract"
            )

    def __getitem__(self, split: str) -> torch.utils.data.DataLoader[RaggedBatch]:
        """Return one loader, preserving the familiar ``built['train']`` access."""
        return self.loaders[split]

    def __repr__(self) -> str:
        sizes = ", ".join(
            f"{name}={len(dataset)}" for name, dataset in self.split_datasets.items()
        )
        fields = ", ".join(self.contract.fields)
        return (
            f"BuiltData({len(self.dataset)} segments; splits: {sizes}; "
            f"fields: {fields})"
        )


class RaggedCollatorConfig(pydantic.BaseModel):
    """Serializable policy for constructing a :class:`RaggedCollator`."""

    model_config = pydantic.ConfigDict(extra="forbid")

    fields: dict[str, PadSpec]

    @pydantic.model_validator(mode="after")
    def _validate_fields(self) -> tp.Self:
        # Keep configuration-time failures identical to the runtime callable.
        RaggedCollator(fields=self.fields)
        return self

    def make(self) -> RaggedCollator:
        """Construct the stateless callable used as ``DataLoader.collate_fn``."""
        return RaggedCollator(fields=self.fields)


class MetadataBatchSamplerConfig(pydantic.BaseModel):
    """Serializable configuration for a dataset-bound :class:`MetadataBatchSampler`.

    The runtime dataset and split-specific ``shuffle`` value are supplied to
    :meth:`make`; every other sampler argument is owned by this configuration.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    group_by: tuple[str, ...]
    batch_size: int = 64
    drop_last: bool = False
    seed: int = 0

    @pydantic.model_validator(mode="after")
    def _validate_policy(self) -> tp.Self:
        if not self.group_by:
            raise ValueError("group_by must contain at least one metadata column")
        if len(self.group_by) != len(set(self.group_by)):
            raise ValueError(f"group_by contains duplicates: {self.group_by}")
        if any(not isinstance(column, str) or not column for column in self.group_by):
            raise ValueError("group_by must contain non-empty strings")
        if (
            not isinstance(self.batch_size, int)
            or isinstance(self.batch_size, bool)
            or self.batch_size <= 0
        ):
            raise ValueError(
                f"batch_size must be a positive int, got {self.batch_size!r}"
            )
        if isinstance(self.seed, bool):
            raise TypeError("seed must be an int, got bool")
        return self

    def make(
        self,
        dataset: tp.Any,
        *,
        shuffle: bool,
    ) -> MetadataBatchSampler:
        """Bind this policy to one concrete dataset and split behavior."""
        return MetadataBatchSampler(
            dataset=dataset,
            group_by=self.group_by,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=self.drop_last,
            seed=self.seed,
        )

    @property
    def required_columns(self) -> tuple[str, ...]:
        """Metadata columns this sampler must be able to group by."""
        return tuple(self.group_by)


class StratifiedBatchSamplerConfig(pydantic.BaseModel):
    """Serializable configuration for a dataset-bound :class:`StratifiedBatchSampler`."""

    model_config = pydantic.ConfigDict(extra="forbid")

    group_by: tuple[str, ...]
    weights_by: tuple[str, ...] | None = None
    batch_size: int = 64
    days_per_batch: int
    drop_last: bool = False
    seed: int = 1

    @pydantic.model_validator(mode="after")
    def _validate_policy(self) -> tp.Self:
        if not self.group_by:
            raise ValueError("group_by must contain at least one metadata column")
        if len(self.group_by) != len(set(self.group_by)):
            raise ValueError(f"group_by contains duplicates: {self.group_by}")
        if any(not isinstance(column, str) or not column for column in self.group_by):
            raise ValueError("group_by must contain non-empty strings")
        if self.weights_by is not None:
            if len(self.weights_by) != len(set(self.weights_by)):
                raise ValueError(f"weights_by contains duplicates: {self.weights_by}")
            if any(
                not isinstance(column, str) or not column for column in self.weights_by
            ):
                raise ValueError("weights_by must contain non-empty strings")
        if (
            not isinstance(self.batch_size, int)
            or isinstance(self.batch_size, bool)
            or self.batch_size <= 0
        ):
            raise ValueError(
                f"batch_size must be a positive int, got {self.batch_size!r}"
            )
        if (
            not isinstance(self.days_per_batch, int)
            or isinstance(self.days_per_batch, bool)
            or self.days_per_batch <= 0
        ):
            raise ValueError(
                f"days_per_batch must be a positive int, got {self.days_per_batch!r}"
            )
        if self.batch_size % self.days_per_batch != 0:
            raise ValueError(
                f"batch_size ({self.batch_size}) must be divisible by "
                f"days_per_batch ({self.days_per_batch})"
            )
        if isinstance(self.seed, bool):
            raise TypeError("seed must be an int, got bool")
        return self

    def make(
        self,
        dataset: tp.Any,
        *,
        shuffle: bool,
    ) -> StratifiedBatchSampler | MetadataBatchSampler:
        """Bind the train policy, or delegate eval splits to the homogeneous sampler."""
        if shuffle:
            return StratifiedBatchSampler(
                dataset=dataset,
                group_by=self.group_by,
                weights_by=self.weights_by,
                batch_size=self.batch_size,
                days_per_batch=self.days_per_batch,
                seed=self.seed,
            )
        return MetadataBatchSampler(
            dataset=dataset,
            group_by=self.group_by,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=self.drop_last,
            seed=self.seed,
        )

    @property
    def required_columns(self) -> tuple[str, ...]:
        """Grouping and stratification columns read from the dataset metadata."""
        return tuple(self.group_by) + tuple(self.weights_by or ())


class CollectionData(pydantic.BaseModel):
    """Declarative neuralshed counterpart of neuraltrain's example ``Data`` config.

    A configuration owns the executable collection and every policy needed at
    the ragged PyTorch boundary. :meth:`build` executes those generic data
    mechanics and returns both familiar split loaders and the resolved runtime
    facts needed by later model-specific adapters.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    collection: Collection
    segmenter: RaggedSegmenter
    collator: RaggedCollatorConfig
    batch_sampler: MetadataBatchSamplerConfig | StratifiedBatchSamplerConfig | None = (
        None
    )
    # Used only by the ordinary DataLoader path when no batch sampler is configured.
    standard_batch_size: int = 64
    num_workers: int = 0
    split_column: str = "split"
    splits: tuple[str, ...] = ("train", "val", "test")
    shuffle_splits: tuple[str, ...] = ("train",)

    @pydantic.model_validator(mode="after")
    def _validate_pipeline_contract(self) -> tp.Self:
        if (
            not isinstance(self.standard_batch_size, int)
            or isinstance(self.standard_batch_size, bool)
            or self.standard_batch_size <= 0
        ):
            raise ValueError(
                "standard_batch_size must be a positive int, got "
                f"{self.standard_batch_size!r}"
            )
        if (
            not isinstance(self.num_workers, int)
            or isinstance(self.num_workers, bool)
            or self.num_workers < 0
        ):
            raise ValueError(
                f"num_workers must be a non-negative int, got {self.num_workers!r}"
            )
        if not self.split_column:
            raise ValueError("split_column must be non-empty")
        if not self.splits or len(self.splits) != len(set(self.splits)):
            raise ValueError(f"splits must be non-empty and unique, got {self.splits}")
        if any(not isinstance(split, str) or not split for split in self.splits):
            raise ValueError("splits must contain non-empty strings")
        if len(self.shuffle_splits) != len(set(self.shuffle_splits)):
            raise ValueError(
                f"shuffle_splits must be unique, got {self.shuffle_splits}"
            )
        if unknown := set(self.shuffle_splits) - set(self.splits):
            raise ValueError(
                "shuffle_splits must be selected by splits; unknown values: "
                f"{sorted(unknown)}"
            )

        extractor_fields = set(self.segmenter.extractors)
        collator_fields = set(self.collator.fields)
        if unknown := collator_fields - extractor_fields:
            raise ValueError(
                "collator fields do not have matching segmenter extractors: "
                f"{sorted(unknown)}"
            )

        metadata_columns = set(self.segmenter.metadata_columns)
        required_metadata = {self.split_column}
        if self.batch_sampler is not None:
            required_metadata.update(self.batch_sampler.required_columns)
        if missing := required_metadata - metadata_columns:
            raise ValueError(
                "segmenter metadata_columns do not preserve required data metadata: "
                f"{sorted(missing)}"
            )
        return self

    @staticmethod
    def _event_types(extractor: tp.Any) -> tuple[str, ...]:
        values = extractor.event_types
        return (values,) if isinstance(values, str) else tuple(values)

    def _resolve_contract(self, events: pd.DataFrame) -> DataContract:
        fields: dict[str, DataFieldContract] = {}
        for name, extractor in self.segmenter.extractors.items():
            event_types = self._event_types(extractor)
            selected = events.loc[events["type"].isin(event_types)]
            query = getattr(extractor, "query", None)
            if query is not None and not selected.empty:
                try:
                    selected = selected.query(query, engine="python")
                except Exception as exc:
                    raise ValueError(
                        f"extractor field {name!r} query {query!r} could not be "
                        "resolved against the collection events"
                    ) from exc

            roles = ()
            if DATA_ROLE in selected:
                roles = tuple(
                    sorted(
                        {
                            str(value)
                            for value in selected[DATA_ROLE]
                            if pd.notna(value) and value != ""
                        }
                    )
                )
            layout_keys = ()
            if DATA_LAYOUT_KEY in selected:
                layout_keys = tuple(
                    sorted(
                        {
                            str(value)
                            for value in selected[DATA_LAYOUT_KEY]
                            if pd.notna(value) and value != ""
                        }
                    )
                )

            feature_widths: dict[str, int] = {}
            if N_FEATURES in selected and DATA_LAYOUT_KEY in selected:
                dense = selected.loc[
                    selected[N_FEATURES].notna()
                    & selected[DATA_LAYOUT_KEY].notna()
                    & selected[DATA_LAYOUT_KEY].ne("")
                ]
                for layout_key, rows in dense.groupby(DATA_LAYOUT_KEY, sort=True):
                    widths = {int(value) for value in rows[N_FEATURES]}
                    if len(widths) != 1:
                        raise ValueError(
                            f"extractor field {name!r} has conflicting feature widths "
                            f"for layout {layout_key!r}: {sorted(widths)}"
                        )
                    feature_widths[str(layout_key)] = widths.pop()

            fields[name] = DataFieldContract(
                event_types=event_types,
                aggregation=str(extractor.aggregation),
                query=query,
                roles=roles,
                layout_keys=layout_keys,
                feature_widths=feature_widths,
                padding=self.collator.fields.get(name),
            )

        return DataContract(
            fields=fields,
            layout_group_by=self.collection.layout_step.group_by,
            batch_group_by=(
                self.batch_sampler.group_by if self.batch_sampler is not None else ()
            ),
            split_column=self.split_column,
            splits=self.splits,
        )

    def _validate_built(self, built: BuiltData) -> None:
        """Model-specific resolved-contract hook."""

    def build(self) -> BuiltData:
        """Execute the collection and construct prepared split DataLoaders.

        Every configured split must contain at least one segment. This makes a
        misspelled or unavailable training/evaluation split fail explicitly;
        experiments that need fewer loaders should narrow :attr:`splits`.
        """
        events = self.collection.run()
        if not isinstance(events, pd.DataFrame):
            raise TypeError(
                "Collection.run() must return a pandas DataFrame, "
                f"got {type(events).__name__}"
            )
        events = events.reset_index(drop=True)

        dataset = self.segmenter.apply(events)
        if not isinstance(dataset, RaggedSegmentDataset):
            raise TypeError(
                "CollectionData.segmenter must return a RaggedSegmentDataset, "
                f"got {type(dataset).__name__}"
            )

        split_values = dataset.metadata[self.split_column]
        missing_splits = [
            split for split in self.splits if not split_values.eq(split).any()
        ]
        if missing_splits:
            available = sorted(
                str(value) for value in split_values.dropna().unique() if value != ""
            )
            raise ValueError(
                f"configured splits contain no segments: {missing_splits}; "
                f"available values: {available}"
            )

        contract = self._resolve_contract(events)
        dataset.prepare()

        split_datasets: dict[str, RaggedSegmentDataset] = {}
        loaders: dict[str, torch.utils.data.DataLoader[RaggedBatch]] = {}
        collator = self.collator.make()
        for split in self.splits:
            split_dataset = dataset.select(split_values.eq(split))
            split_datasets[split] = split_dataset
            shuffle = split in self.shuffle_splits
            if self.batch_sampler is None:
                loaders[split] = torch.utils.data.DataLoader(
                    dataset=split_dataset,
                    batch_size=self.standard_batch_size,
                    shuffle=shuffle,
                    collate_fn=collator,
                    num_workers=self.num_workers,
                )
            else:
                sampler = self.batch_sampler.make(
                    split_dataset,
                    shuffle=shuffle,
                )
                loaders[split] = torch.utils.data.DataLoader(
                    dataset=split_dataset,
                    batch_sampler=sampler,
                    collate_fn=collator,
                    num_workers=self.num_workers,
                )

        built = BuiltData(
            events=events,
            dataset=dataset,
            split_datasets=split_datasets,
            loaders=loaders,
            contract=contract,
        )
        self._validate_built(built)
        return built


def _bic_collator() -> RaggedCollatorConfig:
    return RaggedCollatorConfig(
        fields={
            "spikes": PadSpec(axis=-1, pad_value=0),
            "target": PadSpec(axis=0, pad_value=0),
        }
    )


class BICData(CollectionData):
    """BIC conventions layered on the generic collection-data configuration.

    BIC consumes one or more named neural feature streams and a variable-length
    character target. Architecture, learned adapters, objectives, and batch
    adaptation remain outside this class; generic execution is inherited from
    :class:`CollectionData`.
    """

    collator: RaggedCollatorConfig = pydantic.Field(default_factory=_bic_collator)
    input_fields: tuple[str, ...] = ("spikes",)
    target_field: str = "target"

    @pydantic.model_validator(mode="after")
    def _validate_bic_contract(self) -> tp.Self:
        if self.batch_sampler is None:
            raise ValueError("BICData requires a metadata batch-sampler policy")
        if not self.input_fields or len(self.input_fields) != len(
            set(self.input_fields)
        ):
            raise ValueError(
                f"input_fields must be non-empty and unique, got {self.input_fields}"
            )
        if any(not isinstance(name, str) or not name for name in self.input_fields):
            raise ValueError("input_fields must contain non-empty strings")
        if not self.target_field:
            raise ValueError("target_field must be non-empty")
        if self.target_field in self.input_fields:
            raise ValueError("target_field cannot also be an input field")

        extractors = self.segmenter.extractors
        expected = set(self.input_fields) | {self.target_field}
        if missing := expected - set(extractors):
            raise ValueError(
                f"BIC segmenter is missing extractor fields: {sorted(missing)}"
            )
        if extra := set(extractors) - expected:
            raise ValueError(
                "BIC input_fields and target_field do not classify extractor fields: "
                f"{sorted(extra)}"
            )
        for name in self.input_fields:
            if not isinstance(extractors[name], NeuralFeatureExtractor):
                raise TypeError(
                    f"BIC input extractor {name!r} must be a NeuralFeatureExtractor"
                )
        target = extractors[self.target_field]
        if not isinstance(target, CharacterEncoder):
            raise TypeError("BIC target extractor must be a CharacterEncoder")
        if target.aggregation != "trigger":
            raise ValueError("BIC CharacterEncoder must use aggregation='trigger'")

        missing_padding = expected - set(self.collator.fields)
        if missing_padding:
            raise ValueError(
                f"BIC collator is missing variable fields: {sorted(missing_padding)}"
            )
        return self

    def _validate_built(self, built: BuiltData) -> None:
        for name in self.input_fields:
            field = built.contract.fields[name]
            if not set(field.roles) & {"input", "both"}:
                raise ValueError(
                    f"BIC input field {name!r} does not resolve to an input data role"
                )
            if not field.layout_keys:
                raise ValueError(
                    f"BIC input field {name!r} has no resolved data layout keys"
                )
            if set(field.layout_keys) != set(field.feature_widths):
                raise ValueError(
                    f"BIC input field {name!r} lacks one feature width per data layout"
                )

        target = built.contract.fields[self.target_field]
        if not set(target.roles) & {"target", "both"}:
            raise ValueError(
                f"BIC target field {self.target_field!r} does not resolve to a target "
                "data role"
            )
        if not target.layout_keys:
            raise ValueError(
                f"BIC target field {self.target_field!r} has no resolved data layout keys"
            )


__all__ = [
    "DataFieldContract",
    "DataContract",
    "BuiltData",
    "RaggedCollatorConfig",
    "MetadataBatchSamplerConfig",
    "StratifiedBatchSamplerConfig",
    "CollectionData",
    "BICData",
]
