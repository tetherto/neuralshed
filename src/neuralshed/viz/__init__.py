"""Interactive and static visualization helpers for neuralshed objects.

The generic static study/overlap plots remain owned by ``neuralset``. neuralshed adds an
event-aware timeline and a notebook explorer whose detail renderers understand neuralshed's
custom event classes.
"""

from neuralset.events.viz import plot_overlap, plot_study

from .defaults import TIME_REFERENCE_COLOR, TIME_REFERENCE_COLORS
from .defaults import default_event_view_registry
from .explorer import EventExplorer, ExplorerState, explore_events
from .registry import (
    CachingEventDataProvider,
    DirectEventDataProvider,
    EventDataProvider,
    EventViewContext,
    EventViewRegistry,
    EventViewSpec,
    TimeReference,
)
from .timeline import plot_event_timeline

# Familiar neuralset spelling, upgraded here with compact traces and event ids.
plot_events = plot_event_timeline

__all__ = [
    "CachingEventDataProvider",
    "DirectEventDataProvider",
    "EventDataProvider",
    "EventExplorer",
    "ExplorerState",
    "EventViewContext",
    "EventViewRegistry",
    "EventViewSpec",
    "TIME_REFERENCE_COLOR",
    "TIME_REFERENCE_COLORS",
    "TimeReference",
    "default_event_view_registry",
    "explore_events",
    "plot_event_timeline",
    "plot_events",
    "plot_overlap",
    "plot_study",
]
