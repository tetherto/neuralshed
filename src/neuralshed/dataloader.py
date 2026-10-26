"""Variable-length segment datasets built on top of neuralset.

The native :class:`neuralset.dataloader.Segmenter` remains responsible for
constructing and validating temporal windows.  The classes here only replace
the dataset boundary returned afterwards: one item is extracted at its natural
length, leaving padding and collation to a later stage.
"""

from __future__ import annotations

import dataclasses
import random
import typing as tp

import neuralset as ns
import numpy as np
import pandas as pd
import torch


@dataclasses.dataclass
class SegmentSample:
    """Extracted data and trigger metadata for one unpadded segment."""

    data: dict[str, torch.Tensor]
    segment: ns.Segment
    metadata: dict[str, tp.Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class PadSpec:
    """Describe one variable-length axis in an unbatched sample tensor."""

    axis: int
    pad_value: int | float | bool = 0

    def __post_init__(self) -> None:
        if not isinstance(self.axis, int) or isinstance(self.axis, bool):
            raise TypeError(f"axis must be an int, got {type(self.axis).__name__}")


@dataclasses.dataclass(kw_only=True)
class RaggedBatch(ns.dataloader.Batch):
    """A neuralset batch padded to the longest sample in this batch.

    ``lengths`` records the size of each configured variable axis before
    padding. ``axes`` uses the corresponding unbatched tensor coordinates;
    the batch dimension added during collation is therefore not counted.
    """

    lengths: dict[str, torch.Tensor]
    axes: dict[str, int]
    metadata: dict[str, list[tp.Any]] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.lengths, dict):
            raise TypeError(
                f"'lengths' needs to be a dict, got: {type(self.lengths).__name__}"
            )
        if not isinstance(self.axes, dict):
            raise TypeError(
                f"'axes' needs to be a dict, got: {type(self.axes).__name__}"
            )
        if self.lengths.keys() != self.axes.keys():
            raise ValueError("lengths and axes must describe the same data fields")
        if not isinstance(self.metadata, dict):
            raise TypeError(
                f"'metadata' needs to be a dict, got: {type(self.metadata).__name__}"
            )

        batch_size = len(self.segments)
        for name, values in self.metadata.items():
            if not isinstance(name, str):
                raise TypeError(
                    f"metadata names must be strings, got {type(name).__name__}"
                )
            if not isinstance(values, list):
                raise TypeError(
                    f"metadata[{name!r}] must be a list, got {type(values).__name__}"
                )
            if len(values) != batch_size:
                raise ValueError(
                    f"metadata[{name!r}] has {len(values)} values, "
                    f"expected {batch_size}"
                )
        for name, tensor in self.data.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    f"data[{name!r}] must be a torch.Tensor, "
                    f"got {type(tensor).__name__}"
                )
            if tensor.shape[0] != batch_size:
                raise RuntimeError(
                    f"data[{name!r}] has batch size {tensor.shape[0]}, "
                    f"expected {batch_size}"
                )

        for name, values in self.lengths.items():
            if name not in self.data:
                raise ValueError(f"lengths refers to missing data field {name!r}")
            if not isinstance(values, torch.Tensor):
                raise TypeError(
                    f"lengths[{name!r}] must be a torch.Tensor, "
                    f"got {type(values).__name__}"
                )
            if values.ndim != 1 or values.shape[0] != batch_size:
                raise ValueError(
                    f"lengths[{name!r}] must have shape ({batch_size},), "
                    f"got {tuple(values.shape)}"
                )
            if values.dtype != torch.long:
                raise TypeError(
                    f"lengths[{name!r}] must have dtype torch.long, got {values.dtype}"
                )

            sample_ndim = self.data[name].ndim - 1
            axis = self.axes[name]
            if not 0 <= axis < sample_ndim:
                raise ValueError(
                    f"axis {axis} for {name!r} is invalid for an unbatched "
                    f"{sample_ndim}-dimensional tensor"
                )
            padded_length = self.data[name].shape[axis + 1]
            if torch.any(values < 0) or torch.any(values > padded_length):
                raise ValueError(
                    f"lengths[{name!r}] must be between 0 and {padded_length}"
                )

    def valid_mask(self, name: str) -> torch.Tensor:
        """Return ``True`` at original, non-padding positions for one field."""
        if name not in self.lengths:
            raise KeyError(f"No variable-length field named {name!r}")
        max_length = self.data[name].shape[self.axes[name] + 1]
        positions = torch.arange(max_length, device=self.lengths[name].device)
        return positions[None, :] < self.lengths[name][:, None]

    def padding_mask(self, name: str) -> torch.Tensor:
        """Return ``True`` at padding positions for one field."""
        return ~self.valid_mask(name)

    def to(self, device: str | torch.device) -> "RaggedBatch":
        """Create a ragged batch with tensors moved to ``device``."""
        return type(self)(
            data={name: tensor.to(device) for name, tensor in self.data.items()},
            segments=self.segments,
            lengths={name: values.to(device) for name, values in self.lengths.items()},
            axes=self.axes,
            metadata={name: list(values) for name, values in self.metadata.items()},
        )

    def _subselect(self, idx: list[int]) -> "RaggedBatch":
        return type(self)(
            data={name: tensor[idx] for name, tensor in self.data.items()},
            segments=[self.segments[i] for i in idx],
            lengths={name: values[idx] for name, values in self.lengths.items()},
            axes=self.axes,
            metadata={
                name: [values[i] for i in idx] for name, values in self.metadata.items()
            },
        )

    def __repr__(self) -> str:
        shapes = "; ".join(
            f"{name}: {', '.join(str(size) for size in tensor.shape)}"
            for name, tensor in self.data.items()
        )
        length_ranges = "; ".join(
            f"{name}: {int(values.min())}-{int(values.max())}"
            for name, values in self.lengths.items()
        )
        return (
            f"RaggedBatch({len(self.segments)} segments, {self._duration_str()}, "
            f"shapes: {shapes}, lengths: {length_ranges})"
        )


@dataclasses.dataclass(frozen=True)
class RaggedCollator:
    """Pad configured sample axes to their batch-local maximum and stack."""

    fields: tp.Mapping[str, PadSpec]

    def __post_init__(self) -> None:
        if not isinstance(self.fields, tp.Mapping):
            raise TypeError(
                f"fields must be a mapping, got {type(self.fields).__name__}"
            )
        if not self.fields:
            raise ValueError("fields must contain at least one padding specification")
        for name, spec in self.fields.items():
            if not isinstance(name, str):
                raise TypeError(
                    f"field names must be strings, got {type(name).__name__}"
                )
            if not isinstance(spec, PadSpec):
                raise TypeError(
                    f"fields[{name!r}] must be a PadSpec, got {type(spec).__name__}"
                )

    @staticmethod
    def _normalize_axis(name: str, axis: int, ndim: int) -> int:
        normalized = axis if axis >= 0 else ndim + axis
        if not 0 <= normalized < ndim:
            raise ValueError(
                f"axis {axis} for {name!r} is invalid for a {ndim}-dimensional "
                "sample tensor"
            )
        return normalized

    @staticmethod
    def _validate_tensors(name: str, tensors: list[torch.Tensor]) -> None:
        first = tensors[0]
        for index, tensor in enumerate(tensors):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    f"sample {index} data[{name!r}] must be a torch.Tensor, "
                    f"got {type(tensor).__name__}"
                )
            if tensor.ndim != first.ndim:
                raise ValueError(
                    f"data[{name!r}] tensors must have the same rank, got "
                    f"{first.ndim} and {tensor.ndim}"
                )
            if tensor.dtype != first.dtype:
                raise TypeError(
                    f"data[{name!r}] tensors must have the same dtype, got "
                    f"{first.dtype} and {tensor.dtype}"
                )
            if tensor.device != first.device:
                raise ValueError(
                    f"data[{name!r}] tensors must be on the same device, got "
                    f"{first.device} and {tensor.device}"
                )

    def __call__(self, samples: tp.Sequence[SegmentSample]) -> RaggedBatch:
        if not samples:
            raise ValueError("Cannot collate an empty sequence of samples")
        if not all(isinstance(sample, SegmentSample) for sample in samples):
            raise TypeError("RaggedCollator only accepts SegmentSample instances")

        field_names = list(samples[0].data)
        expected_fields = set(field_names)
        for index, sample in enumerate(samples):
            if set(sample.data) != expected_fields:
                raise ValueError(
                    f"sample {index} has data fields {sorted(sample.data)}, "
                    f"expected {sorted(expected_fields)}"
                )
        unknown_fields = set(self.fields) - expected_fields
        if unknown_fields:
            raise ValueError(
                "padding specifications refer to missing data fields: "
                f"{sorted(unknown_fields)}"
            )

        metadata_names = list(samples[0].metadata)
        expected_metadata = set(metadata_names)
        for index, sample in enumerate(samples):
            if set(sample.metadata) != expected_metadata:
                raise ValueError(
                    f"sample {index} has metadata fields {sorted(sample.metadata)}, "
                    f"expected {sorted(expected_metadata)}"
                )

        data: dict[str, torch.Tensor] = {}
        lengths: dict[str, torch.Tensor] = {}
        axes: dict[str, int] = {}
        for name in field_names:
            tensors = [sample.data[name] for sample in samples]
            self._validate_tensors(name, tensors)

            spec = self.fields.get(name)
            if spec is None:
                shapes = [tuple(tensor.shape) for tensor in tensors]
                if len(set(shapes)) != 1:
                    raise ValueError(
                        f"unconfigured field {name!r} cannot be stacked with "
                        f"shapes {shapes}; add a PadSpec for its variable axis"
                    )
                data[name] = torch.stack(tensors, dim=0)
                continue

            axis = self._normalize_axis(name, spec.axis, tensors[0].ndim)
            reference_shape = tensors[0].shape
            for index, tensor in enumerate(tensors[1:], start=1):
                incompatible = [
                    dimension
                    for dimension, (actual, expected) in enumerate(
                        zip(tensor.shape, reference_shape)
                    )
                    if dimension != axis and actual != expected
                ]
                if incompatible:
                    raise ValueError(
                        f"data[{name!r}] sample {index} has incompatible shape "
                        f"{tuple(tensor.shape)}; only axis {axis} may differ from "
                        f"{tuple(reference_shape)}"
                    )

            original_lengths = [tensor.shape[axis] for tensor in tensors]
            max_length = max(original_lengths)
            padded = []
            for tensor, original_length in zip(tensors, original_lengths):
                if original_length == max_length:
                    padded.append(tensor)
                    continue
                padded_shape = list(tensor.shape)
                padded_shape[axis] = max_length
                padded_tensor = torch.full(
                    padded_shape,
                    spec.pad_value,
                    dtype=tensor.dtype,
                    device=tensor.device,
                )
                index = [slice(None)] * tensor.ndim
                index[axis] = slice(0, original_length)
                padded_tensor[tuple(index)] = tensor
                padded.append(padded_tensor)

            data[name] = torch.stack(padded, dim=0)
            lengths[name] = torch.tensor(
                original_lengths,
                dtype=torch.long,
                device=tensors[0].device,
            )
            axes[name] = axis

        return RaggedBatch(
            data=data,
            segments=[sample.segment for sample in samples],
            lengths=lengths,
            axes=axes,
            metadata={
                name: [sample.metadata[name] for sample in samples]
                for name in metadata_names
            },
        )


class RaggedSegmentDataset(torch.utils.data.Dataset[SegmentSample]):
    """An unpadded item view over a native ``SegmentDataset``.

    The native dataset is retained as ``source`` rather than copied.  Besides
    keeping selection and extractor preparation in neuralset, this preserves
    the event-store references needed when a dataset is pickled for DataLoader
    workers.
    """

    def __init__(
        self,
        source: ns.dataloader.SegmentDataset,
        metadata_columns: tp.Sequence[str] = (),
    ) -> None:
        if not isinstance(source, ns.dataloader.SegmentDataset):
            raise TypeError(
                "source must be a neuralset SegmentDataset, "
                f"got {type(source).__name__}"
            )
        columns = tuple(metadata_columns)
        if len(columns) != len(set(columns)):
            raise ValueError(f"metadata_columns contains duplicates: {columns}")
        if any(not isinstance(column, str) for column in columns):
            raise TypeError("metadata_columns must contain only strings")

        self.source = source
        self.metadata_columns = columns
        if columns:
            triggers = source.triggers
            missing = set(columns) - set(triggers.columns)
            if missing:
                raise KeyError(
                    "metadata columns are missing from the segment triggers: "
                    f"{sorted(missing)}"
                )
            self._metadata = (
                triggers.loc[:, list(columns)].reset_index(drop=True).copy()
            )
        else:
            self._metadata = pd.DataFrame(index=pd.RangeIndex(len(source)))

    @property
    def segments(self) -> tp.Sequence[ns.Segment]:
        """The temporal windows produced by the native segmenter."""
        return self.source.segments

    @property
    def extractors(self) -> tp.Mapping[str, ns.BaseExtractor]:
        """The extractors applied to every segment."""
        return self.source.extractors

    @property
    def triggers(self) -> pd.DataFrame:
        """The trigger event for every segment, in dataset order."""
        return self.source.triggers

    @property
    def events(self) -> pd.DataFrame:
        """All events spanned by the dataset's segments."""
        return self.source.events

    @property
    def metadata(self) -> pd.DataFrame:
        """Requested trigger columns, aligned positionally with dataset items."""
        return self._metadata

    def prepare(self) -> None:
        """Prepare the native extractors against these segments."""
        self.source.prepare()

    def select(
        self,
        idx: int | list[int] | list[bool] | np.ndarray | pd.Series,
    ) -> "RaggedSegmentDataset":
        """Return a ragged view over a native dataset subselection."""
        return type(self)(
            self.source.select(idx), metadata_columns=self.metadata_columns
        )

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, idx: int) -> SegmentSample:
        if not isinstance(idx, int):
            raise TypeError(f"idx must be an int, got {type(idx).__name__}")

        segment = self.segments[idx]
        data = {
            name: extractor(
                segment.ns_events,
                start=segment.start,
                duration=segment.duration,
                trigger=segment.trigger,
            )
            for name, extractor in self.extractors.items()
        }

        # Preserve the native dataset transform contract: transforms receive
        # a one-item batch.  Remove that temporary dimension before exposing
        # the naturally shaped sample.
        for name, transform in self.source.transforms.items():
            transformed = transform(data[name][None, ...])
            if transformed.shape[0] != 1:
                raise ValueError(
                    f"transform for {name!r} changed the temporary batch size "
                    f"from 1 to {transformed.shape[0]}"
                )
            data[name] = transformed[0]

        metadata = self._metadata.iloc[idx].to_dict()
        return SegmentSample(data=data, segment=segment, metadata=metadata)

    def __repr__(self) -> str:
        extractors = ", ".join(self.extractors) or "none"
        metadata = ", ".join(self.metadata_columns) or "none"
        return (
            f"RaggedSegmentDataset({len(self)} segments, extractors: {extractors}, "
            f"metadata: {metadata})"
        )


class StratifiedBatchSampler(torch.utils.data.Sampler[list[int]]):
    """Heterogeneous multi-day batches: stratified day draws from a metadata table.
    Recording days are sampled and distributed proportionally to dataset composition.
    """

    def __init__(
        self,
        dataset: tp.Any,
        *,
        group_by: str | tp.Sequence[str],
        weights_by: str | tuple[str, ...] | None = None,
        batch_size: int,
        days_per_batch: int,
        seed: int = 1,
        seeding: str = "reseed",
        num_replicas: int = 1,
        rank: int = 0,
    ) -> None:
        metadata = getattr(dataset, "metadata", None)
        if not isinstance(metadata, pd.DataFrame):
            raise TypeError("dataset must expose metadata as a pandas DataFrame")
        if len(metadata) != len(dataset):
            raise ValueError(
                f"dataset metadata has {len(metadata)} rows for {len(dataset)} items"
            )

        columns = (group_by,) if isinstance(group_by, str) else tuple(group_by)
        if not columns:
            raise ValueError("group_by must contain at least one metadata column")
        if len(columns) != len(set(columns)):
            raise ValueError(f"group_by contains duplicates: {columns}")
        missing = set(columns) - set(metadata.columns)
        if missing:
            raise KeyError(f"grouping metadata columns are missing: {sorted(missing)}")

        # Optional metadata column name(s) saying which layer contributes to each
        # `days_per_batch` slots, proportionally to its size (i.e., the different subjects in cross-dataset training)
        if weights_by is None:
            weight_columns: tuple[str, ...] = ()
        else:
            weight_columns = (
                (weights_by,) if isinstance(weights_by, str) else tuple(weights_by)
            )
            if len(weight_columns) != len(set(weight_columns)):
                raise ValueError(f"weights_by contains duplicates: {weight_columns}")
            missing_w = set(weight_columns) - set(metadata.columns)
            if missing_w:
                raise KeyError(
                    f"weighting metadata columns are missing: {sorted(missing_w)}"
                )
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError(f"batch_size must be a positive int, got {batch_size!r}")
        # Ensure valid quantities for stratification (divisibilty guard)
        if (
            not isinstance(days_per_batch, int)
            or isinstance(days_per_batch, bool)
            or days_per_batch <= 0
        ):
            raise ValueError(
                f"days_per_batch must be a positive int, got {days_per_batch!r}"
            )
        if batch_size % days_per_batch != 0:
            raise ValueError(
                f"batch_size ({batch_size}) must be divisible by "
                f"days_per_batch ({days_per_batch})"
            )

        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError(f"seed must be an int, got {seed.__class__.__name__}")

        grouped_values = metadata.loc[
            :, list(dict.fromkeys((*columns, *weight_columns)))
        ]
        if grouped_values.isna().to_numpy().any():
            missing_rows = grouped_values.isna().any(axis=1)
            raise ValueError(
                f"grouping metadata contains missing values in "
                f"{int(missing_rows.sum())} rows"
            )

        self.dataset = dataset
        self.group_by = columns
        self.weights_by = weight_columns
        self.batch_size = batch_size
        self.days_per_batch = days_per_batch
        self.samples_per_day = batch_size // days_per_batch
        if seeding not in ("reseed", "continuous"):
            raise ValueError(
                f"seeding must be 'reseed' or 'continuous', got {seeding!r}"
            )
        if (
            not isinstance(num_replicas, int)
            or isinstance(num_replicas, bool)
            or num_replicas < 1
        ):
            raise ValueError(
                f"num_replicas must be a positive int, got {num_replicas!r}"
            )
        if (
            not isinstance(rank, int)
            or isinstance(rank, bool)
            or not 0 <= rank < num_replicas
        ):
            raise ValueError(f"rank must be an int in [0, num_replicas), got {rank!r}")
        if num_replicas > 1 and (
            batch_size % num_replicas or days_per_batch % num_replicas
        ):
            raise ValueError(
                f"Batch_size ({batch_size}) and days_per_batch "
                f"({days_per_batch}) must both be divisible by num_replicas "
                f"({num_replicas})"
            )
        self.seed = seed
        self.seeding = seeding
        self.num_replicas = num_replicas
        self.rank = rank
        # DDP contract:
        self._sub_batch_days = days_per_batch // num_replicas
        if num_replicas > 1:
            self._per_rank = (
                len(dataset) // (batch_size // num_replicas)
            ) // num_replicas
            self.num_batches = self._per_rank * num_replicas
        else:
            self._per_rank = len(dataset) // batch_size
            self.num_batches = self._per_rank

        day_values = metadata.loc[:, list(columns)]
        # Map the dataset positions belonging to each day
        self._groups: dict[tuple[tp.Any, ...], list[int]] = {}
        for index, values in enumerate(day_values.itertuples(index=False, name=None)):
            try:
                hash(values)
            except TypeError as exc:
                raise TypeError(
                    f"grouping metadata at row {index} is not hashable: {values!r}"
                ) from exc
            self._groups.setdefault(values, []).append(index)

        if weight_columns:
            # Subset containing only the weighting columns
            weight_values = metadata.loc[:, list(weight_columns)]
            # Count how many rows each "stratum" (i.e., different subj/datasets) has
            stratum_sizes: dict[tuple[tp.Any, ...], int] = {}
            # Map to which "stratum" each day belongs to
            group_stratum: dict[tuple[tp.Any, ...], tuple[tp.Any, ...]] = {}
            for row_pos, (day_key, w_key) in enumerate(
                zip(
                    day_values.itertuples(index=False, name=None),
                    weight_values.itertuples(index=False, name=None),
                )
            ):
                stratum_sizes[w_key] = stratum_sizes.get(w_key, 0) + 1
                known = group_stratum.setdefault(day_key, w_key)
                if known != w_key:
                    raise ValueError(
                        f"day group {day_key!r} spans strata {known!r} and "
                        f"{w_key!r} (row {row_pos})"
                    )
        else:
            stratum_sizes = {(): len(metadata)}
            group_stratum = {day_key: () for day_key in self._groups}

        # Invert the mapping: {stratum: [day_keys]}
        self._strata: dict[tuple[tp.Any, ...], list[tuple[tp.Any, ...]]] = {}
        for day_key, stratum in group_stratum.items():
            self._strata.setdefault(stratum, []).append(day_key)
        # IDs for each stratum
        self._stratum_ids = list(stratum_sizes)
        # Two different sources of randomness used:
        # `trng` for weighted day-slot draws and `rng` for uniform day/sample draws.
        # Under `seeding="reseed"` both are rebuilt per iteration from
        # `seed + epoch`, so one epoch reproduces from that pair; see
        # `set_epoch`. Under `seeding="continuous"` they are built once here
        # and run for the sampler's whole life.
        self.epoch = 0
        if self.seeding == "continuous":
            self._trng: torch.Generator | None = torch.Generator().manual_seed(seed)
            self._rng: random.Random | None = random.Random(seed)
        else:
            self._trng = None
            self._rng = None
        # Compute the percentage of contribution for each day
        total = len(metadata)
        self._probs = {key: n / total for key, n in stratum_sizes.items()}

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch the draw streams are reseeded from.

        A no-op under ``seeding="continuous"``: those streams run across
        epochs by design and are never restarted.
        """
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise ValueError(f"epoch must be a non-negative int, got {epoch!r}")
        if self.seeding == "continuous":
            return
        self.epoch = epoch

    def __iter__(self) -> tp.Iterator[list[int]]:
        if self.seeding == "continuous":
            assert self._trng is not None and self._rng is not None
            trng, rng = self._trng, self._rng
        else:
            trng = torch.Generator().manual_seed(self.seed + self.epoch)
            rng = random.Random(self.seed + self.epoch)
            self.epoch += 1
        probs = torch.tensor([self._probs[k] for k in self._stratum_ids])
        for global_index in range(self.num_batches):
            batch: list[int] = []
            slot_ids = torch.multinomial(
                probs,
                num_samples=self._sub_batch_days,
                replacement=True,
                generator=trng,
            )
            counts = torch.bincount(slot_ids, minlength=len(self._stratum_ids))
            for key, n_days in zip(self._stratum_ids, counts):
                days = rng.choices(self._strata[key], k=int(n_days.item()))
                for day in days:
                    batch.extend(rng.choices(self._groups[day], k=self.samples_per_day))
            if global_index % self.num_replicas == self.rank:
                yield batch

    def __len__(self) -> int:
        return self._per_rank


class MetadataBatchSampler(torch.utils.data.Sampler[list[int]]):
    """Build homogeneous batches from columns in a dataset metadata table.

    The dataset must expose a ``metadata`` DataFrame whose rows align with its
    positional indices. Grouping never calls ``dataset[i]``, so it does not
    trigger data extraction merely to discover compatible samples.
    """

    def __init__(
        self,
        dataset: tp.Any,
        *,
        group_by: str | tp.Sequence[str],
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
    ) -> None:
        metadata = getattr(dataset, "metadata", None)
        if not isinstance(metadata, pd.DataFrame):
            raise TypeError("dataset must expose metadata as a pandas DataFrame")
        if len(metadata) != len(dataset):
            raise ValueError(
                f"dataset metadata has {len(metadata)} rows for {len(dataset)} items"
            )

        columns = (group_by,) if isinstance(group_by, str) else tuple(group_by)
        if not columns:
            raise ValueError("group_by must contain at least one metadata column")
        if len(columns) != len(set(columns)):
            raise ValueError(f"group_by contains duplicates: {columns}")
        missing = set(columns) - set(metadata.columns)
        if missing:
            raise KeyError(f"grouping metadata columns are missing: {sorted(missing)}")
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError(f"batch_size must be a positive int, got {batch_size!r}")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError(f"seed must be an int, got {type(seed).__name__}")

        grouped_values = metadata.loc[:, list(columns)]
        if grouped_values.isna().to_numpy().any():
            missing_rows = grouped_values.isna().any(axis=1)
            raise ValueError(
                f"grouping metadata contains missing values in "
                f"{int(missing_rows.sum())} rows"
            )

        self.dataset = dataset
        self.group_by = columns
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        self._groups: dict[tuple[tp.Any, ...], list[int]] = {}
        for index, values in enumerate(
            grouped_values.itertuples(index=False, name=None)
        ):
            try:
                hash(values)
            except TypeError as exc:
                raise TypeError(
                    f"grouping metadata at row {index} is not hashable: {values!r}"
                ) from exc
            self._groups.setdefault(values, []).append(index)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch used to derive the deterministic shuffle order."""
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise ValueError(f"epoch must be a non-negative int, got {epoch!r}")
        self.epoch = epoch

    def __iter__(self) -> tp.Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        batches: list[list[int]] = []
        for indices in self._groups.values():
            order = rng.permutation(indices).tolist() if self.shuffle else list(indices)
            for start in range(0, len(order), self.batch_size):
                batch = order[start : start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        return iter(batches)

    def __len__(self) -> int:
        if self.drop_last:
            return sum(
                len(indices) // self.batch_size for indices in self._groups.values()
            )
        return sum(
            (len(indices) + self.batch_size - 1) // self.batch_size
            for indices in self._groups.values()
        )


class NeuralshedSegmenter(ns.dataloader.Segmenter):
    """Local dataset-construction hook over neuralset's native segmenter."""

    def _build_dataset(
        self,
        dataset: ns.dataloader.SegmentDataset,
    ) -> tp.Any:
        """Convert the native result; subclasses override this hook."""
        return dataset

    def apply(self, events: pd.DataFrame) -> tp.Any:
        """Run native segmentation, then pass its dataset through the local hook."""
        return self._build_dataset(super().apply(events))


class RaggedSegmenter(NeuralshedSegmenter):
    """Return naturally sized segment samples instead of globally padded batches."""

    padding: tp.Literal[None] = None
    metadata_columns: tuple[str, ...] = ()

    def _build_dataset(
        self,
        dataset: ns.dataloader.SegmentDataset,
    ) -> RaggedSegmentDataset:
        return RaggedSegmentDataset(dataset, metadata_columns=self.metadata_columns)


__all__ = [
    "SegmentSample",
    "PadSpec",
    "RaggedBatch",
    "RaggedCollator",
    "RaggedSegmentDataset",
    "MetadataBatchSampler",
    "StratifiedBatchSampler",
    "NeuralshedSegmenter",
    "RaggedSegmenter",
]
