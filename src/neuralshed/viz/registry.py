"""Event-detail view registration and the context passed to renderers."""

from __future__ import annotations

import dataclasses
import math
import typing as tp

import pandas as pd
from neuralset.events import etypes

EventRenderer = tp.Callable[["EventViewContext"], tp.Any]
DEFAULT_TIME_REFERENCE_COLOR = "#d1495b"


@dataclasses.dataclass(frozen=True)
class TimeReference:
    """One labeled, colored time coordinate shared across explorer views."""

    id: str
    time: float
    color: str = DEFAULT_TIME_REFERENCE_COLOR

    def __post_init__(self) -> None:
        reference_id = str(self.id).strip()
        time = float(self.time)
        color = str(self.color).strip()
        if not reference_id:
            raise ValueError("time reference id must not be empty")
        if not math.isfinite(time):
            raise ValueError("time reference must be finite")
        if not color:
            raise ValueError("time reference color must not be empty")
        object.__setattr__(self, "id", reference_id)
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "color", color)


class EventDataProvider(tp.Protocol):
    """How a detail renderer obtains the payload behind a data event."""

    def read(self, event: etypes.Event) -> tp.Any: ...


class DirectEventDataProvider:
    """Read through the event itself.

    This is the current notebook behaviour. A future provider can implement source-level
    windowing without changing the explorer or renderer registry.
    """

    def read(self, event: etypes.Event) -> tp.Any:
        if not isinstance(event, etypes.BaseDataEvent):
            raise TypeError(f"{type(event).__name__} is not a data event")
        return event.read()


class CachingEventDataProvider:
    """Cache whole-event reads for the lifetime of one explorer.

    Zooming changes which portion a renderer displays, not the underlying event payload.
    Keeping that payload in memory avoids repeating an expensive lazy read after every
    timeline range change. A future window-aware provider can replace this wrapper.
    """

    def __init__(self, provider: EventDataProvider | None = None) -> None:
        self.provider = provider or DirectEventDataProvider()
        self._cache: dict[tuple[type[etypes.Event], str], tp.Any] = {}

    @staticmethod
    def _key(event: etypes.Event) -> tuple[type[etypes.Event], str]:
        return type(event), event.model_dump_json()

    def read(self, event: etypes.Event) -> tp.Any:
        key = self._key(event)
        if key not in self._cache:
            self._cache[key] = self.provider.read(event)
        return self._cache[key]

    def clear(self) -> None:
        """Release all cached event payloads."""
        self._cache.clear()


@dataclasses.dataclass(frozen=True)
class EventViewContext:
    """Everything a detail renderer may need about the selected event."""

    events: pd.DataFrame
    row_position: int
    row: pd.Series
    event: etypes.Event
    visible_time_range: tuple[float, float]
    data_provider: EventDataProvider
    reference_time: float | None = None
    references: tuple[TimeReference, ...] = ()

    @property
    def time_references(self) -> tuple[TimeReference, ...]:
        """All references, including a legacy scalar reference when supplied."""
        if self.references:
            return self.references
        if self.reference_time is None:
            return ()
        return (TimeReference(id="R1", time=self.reference_time),)

    @property
    def start(self) -> float:
        return float(self.row["start"])

    @property
    def stop(self) -> float:
        duration = self.row.get("duration", 0.0)
        duration = 0.0 if pd.isna(duration) else float(duration)
        return self.start + duration

    def display_window(self) -> tuple[float, float]:
        """Window a data view should show, clipped to the selected event."""
        start, stop = self.visible_time_range
        if self.event.duration > 0:
            start = max(start, self.event.start)
            stop = min(stop, self.event.stop)
        if stop <= start:
            return (self.event.start, self.event.stop)
        return (start, stop)

    def event_from_row(self, row: pd.Series) -> etypes.Event:
        fields = row.drop(labels=["_event_id"], errors="ignore")
        return etypes.Event.from_dict(fields.to_dict())

    def read(self, event: etypes.Event | None = None) -> tp.Any:
        return self.data_provider.read(self.event if event is None else event)

    def related(
        self,
        event_types: str | type[etypes.Event] | tp.Sequence[str],
        *,
        overlapping: bool = True,
        include_selected: bool = False,
    ) -> pd.DataFrame:
        """Return related rows from this timeline, optionally overlapping the selection."""
        helper = etypes.EventTypesHelper(event_types)
        mask = self.events["type"].isin(helper.names)
        if overlapping:
            starts = self.events["start"].astype(float)
            durations = self.events["duration"].fillna(0).astype(float)
            stops = starts + durations
            mask &= (starts <= self.stop) & (stops >= self.start)
        if not include_selected:
            positions = pd.Series(range(len(self.events)), index=self.events.index)
            mask &= positions != self.row_position
        return self.events.loc[mask]


@dataclasses.dataclass(frozen=True)
class EventViewSpec:
    """One named detail view for an event class."""

    event_class: type[etypes.Event]
    name: str
    renderer: EventRenderer
    label: str
    default: bool = False
    time_sensitive: bool = False
    order: int = 0


class EventViewRegistry:
    """Inheritance-aware registry of the views available for an event.

    Parent-class views remain available. If a subclass registers the same view name, its
    implementation replaces the parent implementation for that event.
    """

    def __init__(self) -> None:
        self._specs: list[EventViewSpec] = []

    def register(
        self,
        event_class: type[etypes.Event],
        renderer: EventRenderer | None = None,
        *,
        name: str = "default",
        label: str | None = None,
        default: bool = False,
        time_sensitive: bool = False,
    ) -> EventRenderer | tp.Callable[[EventRenderer], EventRenderer]:
        """Register directly or use as a decorator."""

        def add(func: EventRenderer) -> EventRenderer:
            self._specs.append(
                EventViewSpec(
                    event_class=event_class,
                    name=name,
                    renderer=func,
                    label=label or name.replace("_", " ").title(),
                    default=default,
                    time_sensitive=time_sensitive,
                    order=len(self._specs),
                )
            )
            return func

        return add(renderer) if renderer is not None else add

    def views_for(self, event: etypes.Event) -> list[EventViewSpec]:
        """All applicable views, with subclass implementations taking precedence."""
        mro = type(event).mro()
        by_name: dict[str, tuple[int, EventViewSpec]] = {}
        for spec in self._specs:
            if not isinstance(event, spec.event_class):
                continue
            distance = (
                mro.index(spec.event_class) if spec.event_class in mro else len(mro)
            )
            previous = by_name.get(spec.name)
            if previous is None or distance < previous[0]:
                by_name[spec.name] = (distance, spec)
        ranked = by_name.values()
        return [
            spec
            for _, spec in sorted(
                ranked,
                key=lambda item: (not item[1].default, item[0], item[1].order),
            )
        ]


__all__ = [
    "CachingEventDataProvider",
    "DEFAULT_TIME_REFERENCE_COLOR",
    "DirectEventDataProvider",
    "EventDataProvider",
    "EventRenderer",
    "EventViewContext",
    "EventViewRegistry",
    "EventViewSpec",
    "TimeReference",
]
