import hashlib
import inspect
import typing as tp

import numpy as np
import pydantic
import torch
from exca import MapInfra
from neuralset.base import TimedArray
from neuralset.events import etypes, eventquery
from neuralset.extractors.base import BaseExtractor, BaseStatic

from neuralshed import units

# importing the event types registers them so the `event_types` strings resolve below
from neuralshed.events import (  # noqa: F401
    BandPower,
    BehavioralSeries,
    DenseArrayEvent,
    GraspSeries,
    OrientationSeries,
    PositionSeries,
    SpikeCounts,
)
from neuralshed.text import CharVocab, PhonemeVocab


def _item_uid(event: DenseArrayEvent) -> str:
    """Cache key for one event: everything `_preprocess` reads, plus the code reading it.

    The fields come from `DenseArrayEvent.CACHE_FIELDS`.`_IMPL_HASH` is resolved at
    call time because it cannot be built until both extractor classes below exist.
    """
    parts = [_IMPL_HASH, str(event.study_relative_path())]
    for name in type(event).CACHE_FIELDS:
        value = getattr(event, name)
        parts.append(f"{value:g}" if isinstance(value, float) else str(value))
    return "|".join(parts)


class BaseDenseArrayExtractor(BaseExtractor):
    """Windows a `DenseArrayEvent` into a `(n_rows, T)` TimedArray, resampled to `frequency`.

    Abstract — the `Base` prefix keeps neuralset from registering it. Concrete subclasses
    set the `event_types` default and may override `_ch_names`. `offset` shifts the window
    (e.g. a neural→behavior lag). `query` selects logical streams among events of those
    types, for example one brain region among simultaneous `SpikeCounts` events. `_get_data`
    is cached per event in RAM via `exca.MapInfra`.
    """

    frequency: tp.Literal["native"] | float = "native"
    offset: float = 0.0
    query: str | None = None
    infra: MapInfra = MapInfra(keep_in_ram=True)  # cache _get_data per event (RAM)

    _predicate: tp.Callable[[etypes.Event], bool] | None = pydantic.PrivateAttr(None)

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        if self.query is None:
            return
        if self.aggregation == "trigger":
            raise ValueError(
                f"{self.name}: aggregation='trigger' takes the trigger event directly, "
                "bypassing selection, so query would be silently ignored"
            )
        self._predicate = eventquery.compile_event_query(self.query)

    def _exclude_from_cache_uid(self) -> list[str]:
        # offset is applied per-window in _get_timed_array; query picks which events are
        # read, not how one event is read. Neither changes the cached _get_data output.
        return super()._exclude_from_cache_uid() + ["offset", "query"]

    def _select(self, events: list[etypes.Event]) -> list[etypes.Event]:
        """Restrict candidate events to those matching :attr:`query`."""
        if self._predicate is None:
            return events
        try:
            return [event for event in events if self._predicate(event)]
        except ValueError as exc:
            raise ValueError(
                f"{self.name}: query {self.query!r} is invalid here: {exc}"
            ) from exc

    def prepare(self, obj: tp.Any) -> None:
        """Prepare only events selected by :attr:`query`."""
        events = self._event_types_helper.extract(obj)
        self._check_query_fields(events)
        selected = self._select(events)
        if events and not selected:
            raise ValueError(
                f"{self.name}: query {self.query!r} matched none of the "
                f"{len(events)} event(s) it was prepared on"
            )
        super().prepare(selected)

    def _check_query_fields(self, events: list[etypes.Event]) -> None:
        """Require every candidate event to carry every field named by the query.

        Evaluation already raises on a missing field, but ``and``/``or`` short-circuit.
        Compiling against the fields common to all candidates catches a missing field even
        when its branch would not be evaluated for a particular event.
        """
        if self.query is None or not events:
            return
        schemas = [set(type(event).model_fields) | set(event.extra) for event in events]
        common_fields = set.intersection(*schemas)
        eventquery.compile_event_query(self.query, fields=common_fields)

    def _get_relevant_events(
        self,
        events: tp.Any,
        trigger: etypes.Event | None,
        *,
        start: float,
        duration: float,
    ) -> list[etypes.Event]:
        """Filter by query before native window selection and aggregation."""
        selected = self._select(self._event_types_helper.extract(events))
        return super()._get_relevant_events(
            selected,
            trigger,
            start=start,
            duration=duration,
        )

    def _ch_names(self, event: DenseArrayEvent, n: int) -> str:
        """Channel-name header; default `feature:i`. Overridden for named-axis events."""
        return ",".join(f"{event.feature}:{i}" for i in range(n))

    @staticmethod
    def _resample(data: np.ndarray, src_freq: float, dst_freq: float) -> np.ndarray:
        # linear interp; fine for smooth/oversampled signals (cast counts to float first)
        n_src = data.shape[-1]
        if dst_freq == src_freq or n_src == 0:
            return data
        n_dst = max(1, int(round(dst_freq * n_src / src_freq)))
        if n_dst == n_src:
            return data
        src_t = np.arange(n_src) / src_freq
        dst_t = np.arange(n_dst) / dst_freq
        return np.stack([np.interp(dst_t, src_t, row) for row in data]).astype(
            np.float32
        )

    def _select_channels(self, data: np.ndarray, event: DenseArrayEvent) -> np.ndarray:
        """Hook applied to the read array before rescaling; identity here.
        See `NeuralFeatureExtractor` for the channel pick."""
        return data

    def _transform_values(
        self,
        data: np.ndarray,
        event: DenseArrayEvent,
        dst_freq: float,
    ) -> np.ndarray:
        """Hook applied before resampling; identity here. See `NeuralFeatureExtractor`."""
        return data

    def _unit(self, event: DenseArrayEvent, dst_freq: float) -> str:
        """Unit of the values this extractor yields, for the `TimedArray` header."""
        return event.unit

    def _preprocess(self, event: DenseArrayEvent) -> TimedArray:
        src = float(event.frequency)
        dst = src if self.frequency == "native" else float(self.frequency)
        data = np.asarray(event.read(), dtype=np.float32)
        data = self._select_channels(data, event)
        data = self._transform_values(data, event, dst)
        data = self._resample(data=data, src_freq=src, dst_freq=dst)
        return TimedArray(
            data=data,
            frequency=dst,
            start=event.start,
            header={
                "ch_names": self._ch_names(event, data.shape[0]),
                "unit": self._unit(event, dst),
                "normalization": event.normalization,
            },
        )

    @infra.apply(
        item_uid=_item_uid,
        exclude_from_cache_uid="method:_exclude_from_cache_uid",
    )
    def _get_data(
        self,
        events: tp.Sequence[DenseArrayEvent],
    ) -> tp.Iterator[TimedArray]:
        for event in events:
            yield self._preprocess(event)

    def _get_timed_arrays(
        self,
        events: list[DenseArrayEvent],
        start: float,
        duration: float,
    ) -> tp.Iterable[TimedArray]:
        for event in events:
            yield self._get_timed_array(event, start, duration)

    def _get_timed_array(
        self,
        event: DenseArrayEvent,
        start: float,
        duration: float,
    ) -> TimedArray:
        start += self.offset
        ta = next(self._get_data([event])).copy(start=event.start)
        tdata = ta.overlap(start=start, duration=duration)
        if tdata.start == start and tdata.duration == duration:
            out = tdata
        else:  # partial coverage: zero-pad to exact shape
            out = TimedArray(frequency=ta.frequency, start=start, duration=duration)
            out += tdata
        out.start -= self.offset
        return out


class BehavioralSeriesExtractor(BaseDenseArrayExtractor):
    """Behavioral series -> `(n_dims, T)`; `event_types` selects one stream type. Channel
    names come from the event's `dims`."""

    event_types: str | tuple[str, ...] = "PositionSeries"

    def _ch_names(self, event: BehavioralSeries, n: int) -> str:
        return ",".join(event.dims)


class NeuralFeatureExtractor(BaseDenseArrayExtractor):
    """Pre-binned neural features (`SpikeCounts`/`BandPower`) -> `(n_channels, T)`. Combine
    streams with `aggregation="cat"` (concatenated along channels in event/df order).

    `target_unit` rescales to `"Hz"` or `"counts/bin"` from `event.unit`, the source rate
    and the destination rate, before resampling. Non-convertible units pass through and an
    unregistered unit raises. `"native"` keeps the source's own unit, and inspects nothing
    at all when the rate is unchanged. `event.normalization` gates the conversion
    (`units.preserves_unit`); this extractor never changes it.

    Resampling is area-preserving rather than the base class's interpolation.

    `channels` picks and orders the array rows after the event is read. Rows are
    channels and columns are time bins, so the pick commutes with the rescaling and
    resampling below.

    `pair_trigger_field` re-anchors the downsampling to the trial instead of the
    recording's grid. Pairs of `native_rate / frequency` bins are taken with the
    phase of the trigger-event time, spikes summed and band power averaged per pair, and
    a trailing partial pair is dropped. Pairing requires a numeric `frequency`
    and an integer ratio.
    """

    event_types: str | tuple[str, ...] = ("SpikeCounts", "BandPower")
    target_unit: units.TargetUnit = "native"
    channels: tuple[int, ...] | None = None
    pair_trigger_field: str | None = None
    trim_leading_bins: int = 0

    # the pairing anchor, stashed per call from the trigger by `_get_relevant_events`
    _pair_anchor: float | None = pydantic.PrivateAttr(None)
    # the whole-pair count the current call emitted, to size the assembly window
    _paired_bins: int | None = pydantic.PrivateAttr(None)
    # True while `prepare()` runs its dummy window (whose trigger lacks stamped fields)
    _in_prepare: bool = pydantic.PrivateAttr(False)

    @pydantic.field_validator("channels")
    def _check_channels(cls, value: tuple[int, ...] | None) -> tuple[int, ...] | None:
        if value is not None:
            if not value:
                raise ValueError(
                    "Channels must not be empty; leave it None to keep all rows"
                )
            if min(value) < 0:
                raise ValueError(
                    f"Channels must be non-negative positions, got {value}"
                )
        return value

    @pydantic.field_validator("pair_trigger_field")
    def _check_pair_trigger_field(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError(
                "`pair_trigger_field` must name a trigger field, or be None"
            )
        return value

    @pydantic.field_validator("trim_leading_bins")
    def _check_trim_leading_bins(cls, value: int) -> int:
        if value < 0:
            raise ValueError(f"`trim_leading_bins must` be >= 0, got {value}")
        return value

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        if self.pair_trigger_field is not None and self.frequency == "native":
            raise ValueError(
                f"{self.name}: `pair_trigger_field` re-pairs bins on the way down, so it "
                f"needs a numeric frequency, not 'native'"
            )

    def prepare(self, obj: tp.Any) -> None:
        self._in_prepare = True
        try:
            super().prepare(obj)
        finally:
            self._in_prepare = False

    def _exclude_from_cache_uid(self) -> list[str]:
        # Trim and offset are applied per-window in `_get_timed_array`; they don't affect `_get_data` caching.
        # `pair_trigger_field` is  excluded from caching, since pairing would modify the cached payload.
        return super()._exclude_from_cache_uid() + ["trim_leading_bins"]

    def _get_relevant_events(
        self,
        events: tp.Any,
        trigger: etypes.Event | None,
        *,
        start: float,
        duration: float,
    ) -> list[etypes.Event]:
        self._pair_anchor = None
        self._paired_bins = None
        if (
            self.pair_trigger_field is not None
            and trigger is not None
            and not self._in_prepare
        ):
            try:
                self._pair_anchor = float(
                    trigger._get_field_or_extra(self.pair_trigger_field)
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{self.name}: pair_trigger_field={self.pair_trigger_field!r} did "
                    f"not resolve to a time on the trigger event: {exc}"
                ) from exc
        return super()._get_relevant_events(
            events, trigger, start=start, duration=duration
        )

    def _preprocess(self, event: DenseArrayEvent) -> TimedArray:
        if self.pair_trigger_field is None:
            return super()._preprocess(event)
        data = np.asarray(event.read(), dtype=np.float32)
        data = self._select_channels(data, event)
        return TimedArray(
            data=data,
            frequency=float(event.frequency),
            start=event.start,
            header={
                "ch_names": self._ch_names(event, data.shape[0]),
                "unit": event.unit,
                "normalization": event.normalization,
            },
        )

    def _select_channels(self, data: np.ndarray, event: DenseArrayEvent) -> np.ndarray:
        """Subset rows by source position; out-of-range positions raise."""
        if self.channels is None:
            return data
        n = data.shape[0]
        bad = [c for c in self.channels if c >= n]
        if bad:
            raise ValueError(
                f"{self.name}: channels {bad} are outside the {n} channels of "
                f"{type(event).__name__}(feature={event.feature!r})"
            )
        return data[list(self.channels)]

    def _ch_names(self, event: DenseArrayEvent, n: int) -> str:
        if self.channels is None:
            return super()._ch_names(event, n)
        return ",".join(f"{event.feature}:{c}" for c in self.channels)

    @staticmethod
    def _resample(data: np.ndarray, src_freq: float, dst_freq: float) -> np.ndarray:
        """Area-preserving rebinning: an output bin averages the input over its own span.

        Exact in either direction and at any ratio. An integer downsample merges whole bins
        instead of reading one and dropping the rest, and an upsample splits a bin evenly.
        The running integral of a piecewise-constant signal is piecewise-linear, so
        interpolating *it* is exact rather than an approximation.

        Values are rates here in all but a constant factor, which is why an average is the
        right operator: `_transform_values` has already folded in the count/rate scaling.
        """
        n_src = data.shape[-1]
        if dst_freq == src_freq or n_src == 0:
            return data
        n_dst = max(1, int(round(dst_freq * n_src / src_freq)))
        src_edges = np.arange(n_src + 1) / src_freq
        dst_edges = np.arange(n_dst + 1) / dst_freq
        # float64: a long block's running integral loses the small differences in float32
        integral = np.concatenate(
            [
                np.zeros((data.shape[0], 1)),
                np.cumsum(data, axis=1, dtype=np.float64) / src_freq,
            ],
            axis=1,
        )
        at = np.stack([np.interp(dst_edges, src_edges, row) for row in integral])
        # np.interp clamps past the end, so a trailing partial bin averages only what it covers
        span = np.diff(np.clip(dst_edges, 0.0, src_edges[-1]))
        return (np.diff(at, axis=1) / span).astype(np.float32)

    def _target_unit(self, event: DenseArrayEvent) -> str:
        """The unit the output is expressed in. `native` keeps the source's own string."""
        return event.unit if self.target_unit == "native" else self.target_unit

    def _converts(self, event: DenseArrayEvent, dst_freq: float) -> bool:
        """Whether this event's values need scaling to land in `_target_unit` at `dst_freq`."""
        if self.target_unit == "native" and dst_freq == float(event.frequency):
            return False  # the factor is 1 whatever the unit is, so inspect nothing
        try:
            preserved = units.preserves_unit(event.normalization)
            convertible = units.classify(event.unit) != "non_convertible"
        except ValueError as exc:
            raise ValueError(
                f"{type(event).__name__}(feature={event.feature!r}, "
                f"timeline={event.timeline!r}): {exc}"
            ) from exc
        return preserved and convertible

    def _transform_values(
        self,
        data: np.ndarray,
        event: DenseArrayEvent,
        dst_freq: float,
    ) -> np.ndarray:
        """Unit conversion, including the change of bin width the resampling is about to make.

        Scale the values to `_target_unit` at `dst_freq` if the event's unit is convertible
        and the normalization preserves the unit. Otherwise, return the values unchanged.
        """
        if not self._converts(event, dst_freq):
            return data
        factor = units.scale_factor(
            unit=event.unit,
            target=self._target_unit(event),
            frequency=float(event.frequency),
            target_frequency=dst_freq,
        )
        if factor == 1.0:
            return data
        return np.multiply(data, factor, dtype=np.float64).astype(np.float32)

    def _unit(self, event: DenseArrayEvent, dst_freq: float) -> str:
        return (
            self._target_unit(event) if self._converts(event, dst_freq) else event.unit
        )

    def _get_timed_array(
        self,
        event: DenseArrayEvent,
        start: float,
        duration: float,
    ) -> TimedArray:
        if self.pair_trigger_field is not None and not self._in_prepare:
            out = self._get_paired_timed_array(event, start + self.offset, duration)
            out.start -= self.offset
            return out
        return super()._get_timed_array(event, start, duration)

    def _get_paired_timed_array(
        self,
        event: DenseArrayEvent,
        start: float,
        duration: float,
    ) -> TimedArray:
        """Window re-paired from the trigger's own anchor instead of the recording grid."""
        anchor = self._pair_anchor
        if anchor is None:
            raise ValueError(
                f"{self.name}: pairing needs the trigger's {self.pair_trigger_field!r} "
                "as anchor, but no anchor was stashed for this call"
            )
        f_n = float(event.frequency)
        f_t = float(self.frequency)
        ratio = f_n / f_t
        if abs(ratio - round(ratio)) > 1e-9 or round(ratio) < 2:
            raise ValueError(
                f"{self.name}: pairing needs an integer factor >= 2, got "
                f"{f_n:g} Hz -> {f_t:g} Hz"
            )
        f = round(ratio)
        ta = next(self._get_data([event])).copy(start=event.start)
        t0 = float(ta.start)
        a_bin = round((anchor - t0) * f_n)
        s_bin = round((start - t0) * f_n)
        if s_bin < a_bin:
            raise ValueError(
                f"{self.name}: window start {start:g} s is before the pairing anchor "
                f"{anchor:g} s of {type(event).__name__}(feature={event.feature!r})"
            )
        first = a_bin + f * ((s_bin - a_bin) // f)
        e_bin = round((start - t0) * f_n) + round(duration * f_n)
        n_pairs = (e_bin - first) // f
        if self._paired_bins is not None and self._paired_bins != n_pairs:
            raise ValueError(
                f"{self.name}: paired streams disagree on the pair count "
                f"({self._paired_bins} vs {n_pairs}) — a window one stream does "
                "not cover should have zero-padded to the same shape"
            )
        self._paired_bins = n_pairs
        win_start = t0 + first / f_n
        win_duration = n_pairs * f / f_n
        tdata = ta.overlap(start=win_start, duration=win_duration)
        if tdata.start != win_start or tdata.duration != win_duration:
            # partial coverage: zero-pad to exact shape (same as the grid path)
            padded = TimedArray(
                frequency=ta.frequency, start=win_start, duration=win_duration
            )
            padded += tdata
            tdata = padded
        pairs = tdata.data[..., : n_pairs * f].reshape(-1, n_pairs, f)
        if isinstance(event, SpikeCounts):
            paired = pairs.sum(axis=2)
        elif isinstance(event, BandPower):
            paired = pairs.mean(axis=2)
        else:
            raise TypeError(
                f"{self.name}: pair_trigger_field is defined for SpikeCounts (sum)"
                f"and BandPower (mean) events, got {type(event).__name__}"
            )
        if self._converts(event, f_t):
            factor = units.scale_factor(
                unit=event.unit,
                target=self._target_unit(event),
                frequency=f_n,
                target_frequency=f_t,
            )
            if factor != 1.0:
                paired = np.multiply(paired, factor, dtype=np.float64).astype(
                    np.float32
                )
        return TimedArray(
            data=paired.astype(np.float32),
            frequency=f_t,
            start=start,
            header={
                "ch_names": ta.header["ch_names"],
                "unit": self._unit(event, f_t),
                "normalization": event.normalization,
            },
        )

    def _tarrays_to_tensor(
        self,
        tarrays: list[TimedArray],
        start: float,
        duration: float,
        frequency: float,
        aggregation: str,
    ) -> torch.Tensor:
        if self.pair_trigger_field is not None and self._in_prepare and tarrays:
            frequency = float(tarrays[0].frequency)
        elif (
            self.pair_trigger_field is not None
            and self._paired_bins is not None
            and not self._in_prepare
        ):
            duration = self._paired_bins / float(frequency)
        tensor = BaseExtractor._tarrays_to_tensor(
            tarrays, start, duration, frequency, aggregation
        )
        if self.trim_leading_bins and not self._in_prepare:
            n = tensor.shape[-1] if frequency else 1
            if n <= self.trim_leading_bins:
                raise ValueError(
                    f"{self.name}: trim_leading_bins={self.trim_leading_bins} leaves "
                    f"nothing of the {n}-bin window of this segment"
                )
            tensor = tensor[..., self.trim_leading_bins :]
        return tensor


# Fingerprint of the code `_preprocess` runs, so changing it re-keys every cache entry
_IMPL_HASH = hashlib.md5(
    "".join(
        inspect.getsource(f)
        for f in (
            BaseDenseArrayExtractor._preprocess,
            BaseDenseArrayExtractor._select_channels,
            BaseDenseArrayExtractor._resample,
            NeuralFeatureExtractor._preprocess,
            NeuralFeatureExtractor._select_channels,
            NeuralFeatureExtractor._ch_names,
            NeuralFeatureExtractor._resample,
            NeuralFeatureExtractor._transform_values,
            NeuralFeatureExtractor._converts,
            NeuralFeatureExtractor._unit,
        )
    ).encode()
).hexdigest()[:8]


def read_text_field(event: etypes.Event, field: str) -> str:
    """The event's `field`, whether it is a declared attribute or an extra column.

    neuralset's `Event.from_dict` funnels any column outside the type's schema into the
    `extra` dict, so a transform-added column like `canonical_text` is never an attribute.
    """
    value = getattr(event, field, None)
    if value is None:
        value = event.extra.get(field)
    if value is None:
        raise KeyError(
            f"event {type(event).__name__} has no {field!r} "
            f"(fields: {sorted(type(event).model_fields)}, extra: {sorted(event.extra)})"
        )
    return str(value)


class CharacterEncoder(BaseStatic):
    """Text target -> a variable-length `(L,)` int64 tensor of character ids for CTC decoding.

    `charset_version` pins a frozen charset, so `n_classes` is a property of the code rather
    than of whichever collection happened to be assembled, and nothing is fitted. Set it to
    `None` to fall back to the old behaviour, fitting a `CharVocab` from the texts in
    `prepare()`; that vocabulary lives in a private attribute and does not survive a config
    round-trip, so it is only for exploration.

    `text_field` names the string to read, looked up as an attribute first and then in the
    event's `extra` dict, which is where an events transform's added column lands.

    Output length varies per segment, so batching needs a CTC-style collate (concatenate
    targets, track lengths), not the default `SegmentDataset.collate_fn`.
    """

    event_types: str | tuple[str, ...] = "Sentence"
    text_field: str = "text"
    charset_version: str | None = "v1"

    _vocab: CharVocab | None = pydantic.PrivateAttr(None)

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        if self.charset_version is not None:
            self._vocab = CharVocab.canonical(self.charset_version)

    @property
    def vocab(self) -> CharVocab:
        if self._vocab is None:
            raise ValueError(
                f"{self.name}: call prepare(events) before using the extractor."
            )
        return self._vocab

    def prepare(self, obj: tp.Any) -> None:
        events = self._event_types_helper.extract(obj)
        if not events:
            raise ValueError(f"No events found for {self.name}")
        if self.charset_version is None:
            self._vocab = CharVocab.from_texts(
                read_text_field(e, self.text_field) for e in events
            )
        # run once to populate the output shape (mirrors BaseStatic.prepare)
        self(events[0], events[0].start, duration=0.001, trigger=events[0])

    def get_static(self, event: etypes.Event) -> torch.Tensor:
        ids = self.vocab.encode(read_text_field(event, self.text_field))
        return torch.from_numpy(ids)


class PhonemeEncoder(BaseStatic):
    """Phoneme target -> a variable-length `(L,)` int64 tensor of phoneme ids for CTC decoding.

    Reads the space-separated ARPAbet string in `text_field` (card_2024 emits it as the
    `phonemes` column of each `Sentence`) and maps every symbol through the frozen
    `PhonemeVocab` named by `vocab_version`, so `n_classes` is 41 wherever it is used. A
    symbol outside the vocabulary raises rather than being dropped.

    Output length varies per segment, so batching needs a CTC-style collate.
    """

    event_types: str | tuple[str, ...] = "Sentence"
    text_field: str = "phonemes"
    vocab_version: str = "v1"

    _vocab: PhonemeVocab | None = pydantic.PrivateAttr(None)

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        self._vocab = PhonemeVocab.canonical(self.vocab_version)

    @property
    def vocab(self) -> PhonemeVocab:
        assert self._vocab is not None
        return self._vocab

    def prepare(self, obj: tp.Any) -> None:
        events = self._event_types_helper.extract(obj)
        if not events:
            raise ValueError(f"No events found for {self.name}")
        # run once to populate the output shape (mirrors BaseStatic.prepare)
        self(events[0], events[0].start, duration=0.001, trigger=events[0])

    def get_static(self, event: etypes.Event) -> torch.Tensor:
        ids = self.vocab.encode(read_text_field(event, self.text_field))
        return torch.from_numpy(ids)


class WhisperLabelEncoder(BaseStatic):
    """Text target -> a variable-length `(L,)` int64 tensor of Whisper token ids for the CE head.

    The extractor emits natural length with no padding at all,
    the ragged collator's `PadSpec(axis=-1, pad_value=-100)` writes the ignore values.
    """

    event_types: str | tuple[str, ...] = "Sentence"
    text_field: str = "text"
    whisper_name: str = "openai/whisper-tiny.en"
    max_length: int = 448

    _tokenizer: tp.Any = pydantic.PrivateAttr(None)

    @property
    def tokenizer(self) -> tp.Any:
        if self._tokenizer is None:
            from transformers import WhisperTokenizer

            try:
                self._tokenizer = WhisperTokenizer.from_pretrained(
                    self.whisper_name,
                    language="en",
                    task="transcribe",
                    local_files_only=True,
                )
            except Exception:
                self._tokenizer = WhisperTokenizer.from_pretrained(
                    self.whisper_name, language="en", task="transcribe"
                )
        return self._tokenizer

    def prepare(self, obj: tp.Any) -> None:
        events = self._event_types_helper.extract(obj)
        if not events:
            raise ValueError(f"No events found for {self.name}")
        # run once to populate the output shape (mirrors BaseStatic.prepare)
        self(events[0], events[0].start, duration=0.001, trigger=events[0])

    def get_static(self, event: etypes.Event) -> torch.Tensor:
        encoded = self.tokenizer(
            read_text_field(event, self.text_field),
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        # Stip the prefixed sentence-start token
        return encoded["input_ids"][0, 1:].to(torch.int64)


__all__ = [
    "BaseDenseArrayExtractor",
    "BehavioralSeriesExtractor",
    "NeuralFeatureExtractor",
    "CharacterEncoder",
    "PhonemeEncoder",
    "WhisperLabelEncoder",
    "read_text_field",
]
