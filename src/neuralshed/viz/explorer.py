"""Live-kernel event explorer for Jupyter notebooks."""

from __future__ import annotations

import asyncio
import dataclasses
import html
import typing as tp

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from neuralset.events import etypes

from .defaults import TIME_REFERENCE_COLORS, default_event_view_registry
from .registry import CachingEventDataProvider, EventDataProvider, EventViewContext
from .registry import EventViewRegistry, EventViewSpec, TimeReference
from .timeline import plot_event_timeline


def _widgets() -> tuple[tp.Any, tp.Any]:
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError as exc:  # pragma: no cover - depends on the optional environment
        raise ImportError(
            "EventExplorer needs the optional visualization dependencies. "
            "Install neuralshed with `pip install -e '.[viz]'`."
        ) from exc
    return widgets, display


@dataclasses.dataclass
class ExplorerState:
    """User-controlled state shared by the sidebar, timeline, and detail pane."""

    timeline: tp.Any
    query: str
    search: str
    visible_types: set[str]
    lane_order: list[str]
    time_range: tuple[float, float]
    references: list[TimeReference] = dataclasses.field(default_factory=list)
    selected_event_id: int | None = None
    active_view: str | None = None

    @property
    def reference_time(self) -> float | None:
        """First reference time retained for scalar API compatibility."""
        return self.references[0].time if self.references else None


class EventExplorer:
    """Dashboard for filtering timelines and opening event-specific detail views."""

    def __init__(
        self,
        events: pd.DataFrame,
        *,
        registry: EventViewRegistry | None = None,
        lanes: tp.Sequence[str] | None = None,
        label: str | None = None,
        data_provider: EventDataProvider | None = None,
        zoom_debounce: float = 0.2,
        query: str | None = None,
        search: str | None = None,
        time_range: tuple[float, float] | None = None,
        visible_types: tp.Collection[str] | None = None,
        default_views: tp.Mapping[str, str] | None = None,
        initial_timeline: tp.Any | None = None,
        show_controls: bool = True,
        top_row_height: int = 600,
        event_type_list_height: int = 180,
        reference_list_height: int = 128,
        references: (
            tp.Sequence[TimeReference | float | tp.Mapping[str, tp.Any]] | None
        ) = None,
        reference_time: float | None = None,
    ) -> None:
        widgets, _ = _widgets()
        if events.empty:
            raise ValueError("EventExplorer needs at least one event")
        required = {"timeline", "type", "start", "duration"}
        missing = required - set(events.columns)
        if missing:
            raise ValueError(f"events is missing required columns: {sorted(missing)}")

        self.all_events = events.reset_index(drop=True).copy()
        self.all_events["_event_id"] = np.arange(len(self.all_events))
        self._timeline_values = list(self.all_events["timeline"].dropna().unique())
        if not self._timeline_values:
            raise ValueError("events contains no timeline values")
        if initial_timeline is None:
            initial_timeline = self._timeline_values[0]
        if initial_timeline not in self._timeline_values:
            raise ValueError(f"unknown initial timeline: {initial_timeline!r}")

        self.registry = registry or default_event_view_registry()
        self.data_provider = (
            data_provider
            if isinstance(data_provider, CachingEventDataProvider)
            else CachingEventDataProvider(data_provider)
        )
        self.zoom_debounce = max(0.0, float(zoom_debounce))
        self.label = label
        self.default_views = dict(default_views or {})
        self.show_controls = bool(show_controls)
        self.top_row_height = max(320, int(top_row_height))
        self.event_type_list_height = max(80, int(event_type_list_height))
        self.reference_list_height = max(72, int(reference_list_height))
        self._next_reference_number = 1
        self._requested_visible_types = (
            set(visible_types) if visible_types is not None else None
        )
        self._initial_lane_order = list(lanes or ())

        self.events = self._events_for_timeline(initial_timeline)
        self._full_time_range = self._time_extent(self.events)
        initial_range = self._coerce_time_range(time_range or self._full_time_range)
        available_types = set(self.events["type"].astype(str))
        chosen_types = (
            available_types
            if self._requested_visible_types is None
            else available_types & self._requested_visible_types
        )
        self.state = ExplorerState(
            timeline=initial_timeline,
            query=query or "",
            search=search or "",
            visible_types=chosen_types,
            lane_order=self._ordered_types(available_types),
            time_range=initial_range,
            references=self._coerce_references(references, reference_time),
        )
        self._query_filtered_events = self._filter_by_text(self.events)
        self.visible_events = self._filter_visible_types(self._query_filtered_events)

        self._context: EventViewContext | None = None
        self._views: dict[str, EventViewSpec] = {}
        self._active_view: EventViewSpec | None = None
        self._zoom_task: asyncio.Task[None] | None = None
        self._reference_task: asyncio.Task[None] | None = None
        self._updating_picker = False
        self._updating_controls = False
        self._updating_reference = False

        self.detail_title = widgets.HTML(
            "<div style='padding:8px 0;color:#666'>Click an event to inspect it.</div>"
        )
        self.view_picker = widgets.Dropdown(
            description="View",
            options=[],
            layout=widgets.Layout(width="320px"),
        )
        self.view_picker.layout.display = "none"
        self.detail_content = widgets.VBox(
            layout=widgets.Layout(border="1px solid #ddd", padding="8px")
        )
        detail_children = [self.detail_title]
        if not self.show_controls:
            detail_children.append(self.view_picker)
        detail_children.append(self.detail_content)
        self.detail = widgets.VBox(detail_children)
        self.timeline = self._new_timeline_widget()
        self.timeline_host = widgets.VBox(
            [self.timeline],
            layout=widgets.Layout(
                width="auto",
                min_width="0",
                height=f"{self.top_row_height}px",
                flex="1 1 auto",
                overflow="hidden",
            ),
        )
        self.sidebar = self._build_sidebar()
        top_children = (
            [self.sidebar, self.timeline_host]
            if self.show_controls
            else [self.timeline_host]
        )
        self.top_row = widgets.HBox(
            top_children,
            layout=widgets.Layout(
                width="100%",
                height=f"{self.top_row_height}px",
                align_items="stretch",
                overflow="hidden",
            ),
        )
        self.widget = widgets.VBox(
            [self.top_row, self.detail],
            layout=widgets.Layout(width="100%"),
        )
        self.view_picker.observe(self._on_view_change, names="value")

    @property
    def visible_time_range(self) -> tuple[float, float]:
        return self.state.time_range

    @visible_time_range.setter
    def visible_time_range(self, value: tuple[float, float]) -> None:
        self.state.time_range = value

    @property
    def selected_event_id(self) -> int | None:
        return self.state.selected_event_id

    @selected_event_id.setter
    def selected_event_id(self, value: int | None) -> None:
        self.state.selected_event_id = value

    @property
    def references(self) -> tuple[TimeReference, ...]:
        """Current time references in display order."""
        return tuple(self.state.references)

    def _events_for_timeline(self, timeline: tp.Any) -> pd.DataFrame:
        return self.all_events.loc[self.all_events["timeline"] == timeline].copy()

    @staticmethod
    def _time_extent(events: pd.DataFrame) -> tuple[float, float]:
        starts = events["start"].astype(float)
        stops = starts + events["duration"].fillna(0).astype(float)
        start, stop = float(starts.min()), float(stops.max())
        return (start, stop if stop > start else start + 1.0)

    def _coerce_time_range(self, value: tp.Sequence[float]) -> tuple[float, float]:
        if len(value) != 2:
            raise ValueError("time_range must contain exactly (start, stop)")
        start, stop = float(value[0]), float(value[1])
        if not np.isfinite([start, stop]).all() or stop <= start:
            raise ValueError("time_range must contain finite values with stop > start")
        full_start, full_stop = self._full_time_range
        clipped = (max(start, full_start), min(stop, full_stop))
        if clipped[1] <= clipped[0]:
            raise ValueError(
                f"time_range {value!r} does not overlap {self._full_time_range!r}"
            )
        return clipped

    def _coerce_reference_time(self, value: float | None) -> float | None:
        if value is None:
            return None
        reference = float(value)
        if not np.isfinite(reference):
            raise ValueError("reference_time must be finite")
        full_start, full_stop = self._full_time_range
        if not full_start <= reference <= full_stop:
            raise ValueError(
                f"reference_time {reference} outside {self._full_time_range!r}"
            )
        return reference

    def _coerce_references(
        self,
        values: tp.Sequence[TimeReference | float | tp.Mapping[str, tp.Any]] | None,
        legacy_reference_time: float | None,
    ) -> list[TimeReference]:
        if values is not None and legacy_reference_time is not None:
            raise ValueError("pass either references or reference_time, not both")
        raw_values: tp.Sequence[TimeReference | float | tp.Mapping[str, tp.Any]]
        raw_values = (
            [legacy_reference_time]
            if legacy_reference_time is not None
            else list(values or ())
        )
        references: list[TimeReference] = []
        used_ids: set[str] = set()
        for index, value in enumerate(raw_values):
            if isinstance(value, TimeReference):
                reference_id = value.id
                time = value.time
                color = value.color
            elif isinstance(value, tp.Mapping):
                reference_id = str(value.get("id") or f"R{index + 1}")
                if "time" not in value:
                    raise ValueError("each reference mapping needs a time")
                time = tp.cast(float, value["time"])
                color = str(
                    value.get("color")
                    or TIME_REFERENCE_COLORS[index % len(TIME_REFERENCE_COLORS)]
                )
            else:
                reference_id = f"R{index + 1}"
                time = float(value)
                color = TIME_REFERENCE_COLORS[index % len(TIME_REFERENCE_COLORS)]
            if reference_id in used_ids:
                raise ValueError(f"duplicate time reference id: {reference_id!r}")
            used_ids.add(reference_id)
            references.append(
                TimeReference(
                    id=reference_id,
                    time=tp.cast(float, self._coerce_reference_time(time)),
                    color=color,
                )
            )
        self._next_reference_number = len(references) + 1
        return references

    def _ordered_types(self, available: set[str]) -> list[str]:
        named = [name for name in self._initial_lane_order if name in available]
        return named + sorted(available.difference(named))

    def _filter_by_text(
        self,
        events: pd.DataFrame,
        *,
        query: str | None = None,
        search: str | None = None,
    ) -> pd.DataFrame:
        query = self.state.query if query is None else query
        search = self.state.search if search is None else search
        filtered = events
        if query.strip():
            filtered = filtered.query(query, engine="python")
        if search.strip():
            mask = pd.Series(False, index=filtered.index)
            for column in ("text", "description", "code", "feature"):
                if column in filtered:
                    mask |= (
                        filtered[column]
                        .fillna("")
                        .astype(str)
                        .str.contains(search, case=False, regex=False)
                    )
            filtered = filtered.loc[mask]
        return filtered.copy()

    def _filter_visible_types(self, events: pd.DataFrame) -> pd.DataFrame:
        return events.loc[
            events["type"].astype(str).isin(self.state.visible_types)
        ].copy()

    def _timeline_label(self) -> str:
        if self.label is not None and len(self._timeline_values) == 1:
            return self.label
        return str(self.state.timeline)

    def _empty_timeline_figure(self) -> go.Figure:
        figure = go.Figure()
        figure.add_annotation(
            text="No events match the current filters",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"color": "#777"},
        )
        figure.update_layout(
            title=self._timeline_label(),
            xaxis_title="Time (s)",
            xaxis_range=list(self.visible_time_range),
            height=max(250, len(self.state.lane_order) * 50 + 100),
            margin={"l": 120, "r": 30, "t": 50, "b": 50},
            plot_bgcolor="white",
        )
        return figure

    def _new_timeline_widget(self) -> tp.Any:
        figure = (
            self._empty_timeline_figure()
            if self.visible_events.empty
            else plot_event_timeline(
                self.visible_events,
                lanes=self.state.lane_order,
                label=self._timeline_label(),
            )
        )
        figure.update_layout(height=self.top_row_height)
        figure.update_xaxes(range=list(self.visible_time_range))
        for reference in self.state.references:
            self._add_reference_shape(figure, reference)
        try:
            timeline = go.FigureWidget(figure)
        except ImportError as exc:
            raise ImportError(
                "EventExplorer needs `anywidget`; install with "
                "`pip install -e '.[viz]'`."
            ) from exc
        for trace in timeline.data:
            trace.on_click(self._on_click)
        timeline.layout.on_change(self._on_visible_range, ("xaxis", "range"))
        timeline.layout.on_change(self._on_autorange, ("xaxis", "autorange"))
        self._connect_reference_shapes(timeline)
        return timeline

    @staticmethod
    def _add_reference_shape(figure: tp.Any, reference: TimeReference) -> None:
        figure.add_shape(
            type="line",
            name=f"time-reference:{reference.id}",
            x0=reference.time,
            x1=reference.time,
            y0=0,
            y1=1,
            yref="paper",
            editable=True,
            layer="above",
            line={"color": reference.color, "width": 3},
        )

    @staticmethod
    def _reference_id_from_shape(shape: tp.Any) -> str | None:
        name = str(shape.name or "")
        prefix = "time-reference:"
        return name[len(prefix) :] if name.startswith(prefix) else None

    def _reference_shape(
        self,
        reference_id: str | None = None,
        timeline: tp.Any | None = None,
    ) -> tp.Any | None:
        timeline = self.timeline if timeline is None else timeline
        return next(
            (
                shape
                for shape in timeline.layout.shapes
                if self._reference_id_from_shape(shape) is not None
                and (
                    reference_id is None
                    or self._reference_id_from_shape(shape) == reference_id
                )
            ),
            None,
        )

    def _connect_reference_shapes(self, timeline: tp.Any | None = None) -> None:
        timeline = self.timeline if timeline is None else timeline
        for shape in timeline.layout.shapes:
            if self._reference_id_from_shape(shape) is not None:
                shape.on_change(self._on_reference_shape_change, "x0", "x1")

    def _sync_reference_shapes(self) -> None:
        expected = {reference.id: reference for reference in self.state.references}
        self._updating_reference = True
        try:
            retained = tuple(
                shape
                for shape in self.timeline.layout.shapes
                if (
                    (reference_id := self._reference_id_from_shape(shape)) is None
                    or reference_id in expected
                )
            )
            if retained != tuple(self.timeline.layout.shapes):
                self.timeline.layout.shapes = retained
            for reference in self.state.references:
                shape = self._reference_shape(reference.id)
                if shape is None:
                    self._add_reference_shape(self.timeline, reference)
                    shape = self._reference_shape(reference.id)
                    if shape is not None:
                        shape.on_change(
                            self._on_reference_shape_change,
                            "x0",
                            "x1",
                        )
                elif (
                    shape.x0 != reference.time
                    or shape.x1 != reference.time
                    or shape.line.color != reference.color
                ):
                    with self.timeline.batch_update():
                        shape.x0 = reference.time
                        shape.x1 = reference.time
                        shape.line.color = reference.color
        finally:
            self._updating_reference = False

    def _replace_timeline(self) -> None:
        previous = self.timeline
        self.timeline = self._new_timeline_widget()
        self.timeline_host.children = (self.timeline,)
        previous.close()

    @staticmethod
    def _section_title(text: str) -> tp.Any:
        widgets, _ = _widgets()
        return widgets.HTML(
            "<div style='font-weight:600;margin:10px 0 4px'>"
            f"{html.escape(text)}</div>"
        )

    def _build_sidebar(self) -> tp.Any:
        widgets, _ = _widgets()
        full_start, full_stop = self._full_time_range
        step = max((full_stop - full_start) / 1000, 0.001)

        self.timeline_picker = widgets.Dropdown(
            options=[(str(value), value) for value in self._timeline_values],
            value=self.state.timeline,
            description="Timeline",
            layout=widgets.Layout(width="100%"),
        )
        self.timeline_picker.layout.display = (
            "" if len(self._timeline_values) > 1 else "none"
        )
        self.query_input = widgets.Textarea(
            value=self.state.query,
            placeholder='e.g. type == "Action" and duration > 1',
            layout=widgets.Layout(width="100%", height="56px", margin="0"),
        )
        self.search_input = widgets.Text(
            value=self.state.search,
            placeholder="Search text, description, code, feature",
            layout=widgets.Layout(width="100%", margin="0"),
        )
        self.apply_filter_button = widgets.Button(
            description="Apply", button_style="primary"
        )
        self.reset_filter_button = widgets.Button(description="Reset")
        self.query_status = widgets.HTML(layout=widgets.Layout(margin="0"))

        self.type_box = widgets.VBox(
            layout=widgets.Layout(
                height=f"{self.event_type_list_height}px",
                overflow="auto",
                border="1px solid #ddd",
                padding="0 8px",
                margin="0 0 4px",
            )
        )
        self.lane_picker = widgets.Select(
            options=self.state.lane_order,
            rows=min(4, max(2, len(self.state.lane_order))),
            layout=widgets.Layout(width="100%", margin="0"),
        )
        self.lane_up_button = widgets.Button(description="↑", tooltip="Move lane up")
        self.lane_down_button = widgets.Button(
            description="↓", tooltip="Move lane down"
        )

        self.time_slider = widgets.FloatRangeSlider(
            value=self.visible_time_range,
            min=full_start,
            max=full_stop,
            step=step,
            readout=False,
            continuous_update=False,
            layout=widgets.Layout(width="100%"),
        )
        self.time_start_input = widgets.FloatText(
            value=self.visible_time_range[0],
            layout=widgets.Layout(width="auto", min_width="0", flex="1 1 0"),
        )
        self.time_stop_input = widgets.FloatText(
            value=self.visible_time_range[1],
            layout=widgets.Layout(width="auto", min_width="0", flex="1 1 0"),
        )
        self.reset_time_button = widgets.Button(
            description="Full",
            tooltip="Show the full timeline",
            layout=widgets.Layout(width="auto", min_width="0", flex="1 1 0"),
        )
        self.time_input_row = widgets.HBox(
            [self.time_start_input, self.time_stop_input, self.reset_time_button],
            layout=widgets.Layout(width="100%", grid_gap="4px"),
        )
        self.time_status = widgets.HTML()

        self.add_reference_button = widgets.Button(
            description="Add reference",
            icon="plus",
            tooltip="Add a reference at the center of the visible time window",
            layout=widgets.Layout(width="100%", margin="0"),
        )
        self.reference_box = widgets.VBox(
            layout=widgets.Layout(
                width="100%",
                height=f"{self.reference_list_height}px",
                overflow="auto",
                border="1px solid #ddd",
                padding="4px",
                margin="0",
            ),
        )
        self.reference_status = widgets.HTML(layout=widgets.Layout(margin="0"))

        self.selection_summary = widgets.HTML(
            "<span style='color:#666'>No event selected</span>"
        )
        self.view_picker.description = "View"
        self.view_picker.layout.width = "100%"

        self.filter_actions = widgets.HBox(
            [self.apply_filter_button, self.reset_filter_button],
            layout=widgets.Layout(margin="0"),
        )
        self.lane_actions = widgets.HBox(
            [self.lane_up_button, self.lane_down_button],
            layout=widgets.Layout(margin="0"),
        )

        self.filter_tab = widgets.VBox(
            [
                self.timeline_picker,
                self.query_input,
                self.search_input,
                self.filter_actions,
                self.query_status,
                self.type_box,
                self.lane_picker,
                self.lane_actions,
            ],
            layout=widgets.Layout(width="100%", padding="4px 4px 0"),
        )
        self.view_tab = widgets.VBox(
            [
                self._section_title("Time window"),
                self.time_slider,
                self.time_input_row,
                self.time_status,
                self.add_reference_button,
                self.reference_box,
                self.reference_status,
                self.selection_summary,
                self.view_picker,
            ],
            layout=widgets.Layout(width="100%", padding="4px 4px 0"),
        )
        self.sidebar_tabs = widgets.Tab(
            children=[self.filter_tab, self.view_tab],
            layout=widgets.Layout(width="100%", height=f"{self.top_row_height - 48}px"),
        )
        self.sidebar_tabs.set_title(0, "Filter")
        self.sidebar_tabs.set_title(1, "View")

        sidebar = widgets.VBox(
            [
                widgets.HTML(
                    "<div style='font-size:1.15em;font-weight:650;padding:0 4px'>"
                    "Explore</div>"
                ),
                self.sidebar_tabs,
            ],
            layout=widgets.Layout(
                width="320px",
                min_width="320px",
                height=f"{self.top_row_height}px",
                padding="8px 8px 8px 4px",
                overflow="hidden",
            ),
        )

        self.timeline_picker.observe(self._on_timeline_change, names="value")
        self.apply_filter_button.on_click(self._on_apply_filters)
        self.reset_filter_button.on_click(self._on_reset_filters)
        self.lane_up_button.on_click(lambda _button: self._move_lane(-1))
        self.lane_down_button.on_click(lambda _button: self._move_lane(1))
        self.time_slider.observe(self._on_time_slider, names="value")
        self.time_start_input.observe(self._on_time_input, names="value")
        self.time_stop_input.observe(self._on_time_input, names="value")
        self.reset_time_button.on_click(self._on_reset_time)
        self.add_reference_button.on_click(self._on_add_reference)
        self._rebuild_type_controls()
        self._rebuild_reference_controls()
        self._update_filter_status()
        return sidebar

    def _rebuild_type_controls(self) -> None:
        widgets, _ = _widgets()
        counts = self._query_filtered_events["type"].astype(str).value_counts()
        checks = []
        self._type_checks: dict[str, tp.Any] = {}
        for event_type in self.state.lane_order:
            check = widgets.Checkbox(
                value=event_type in self.state.visible_types,
                description=f"{event_type} ({int(counts.get(event_type, 0))})",
                indent=False,
                layout=widgets.Layout(width="100%", height="24px", margin="0"),
            )
            check.observe(
                lambda change, name=event_type: self._on_type_change(name, change),
                names="value",
            )
            self._type_checks[event_type] = check
            checks.append(check)
        self.type_box.children = tuple(checks)
        current = (
            self.lane_picker.value
            if self.lane_picker.value in self.state.lane_order
            else None
        )
        self.lane_picker.options = self.state.lane_order
        if current is not None:
            self.lane_picker.value = current

    def _rebuild_reference_controls(self) -> None:
        if not hasattr(self, "reference_box"):
            return
        widgets, _ = _widgets()
        self._reference_time_inputs: dict[str, tp.Any] = {}
        self._reference_color_inputs: dict[str, tp.Any] = {}
        self._reference_delete_buttons: dict[str, tp.Any] = {}
        rows = []
        for reference in self.state.references:
            label = widgets.HTML(
                f"<b>{html.escape(reference.id)}</b>",
                layout=widgets.Layout(
                    width="28px",
                    min_width="28px",
                    margin="0",
                ),
            )
            time_input = widgets.FloatText(
                value=reference.time,
                layout=widgets.Layout(
                    width="auto",
                    min_width="0",
                    flex="1 1 auto",
                    margin="0",
                ),
            )
            color_input = widgets.ColorPicker(
                value=reference.color,
                concise=True,
                description="",
                layout=widgets.Layout(
                    width="40px",
                    min_width="40px",
                    margin="0",
                ),
            )
            delete_button = widgets.Button(
                description="",
                icon="trash",
                tooltip=f"Delete {reference.id}",
                layout=widgets.Layout(
                    width="36px",
                    min_width="36px",
                    margin="0",
                ),
            )
            time_input.observe(
                lambda change, reference_id=reference.id: (
                    self._on_reference_input(reference_id, change)
                ),
                names="value",
            )
            color_input.observe(
                lambda change, reference_id=reference.id: (
                    self._on_reference_color(reference_id, change)
                ),
                names="value",
            )
            delete_button.on_click(
                lambda _button, reference_id=reference.id: (
                    self.remove_reference(reference_id)
                )
            )
            self._reference_time_inputs[reference.id] = time_input
            self._reference_color_inputs[reference.id] = color_input
            self._reference_delete_buttons[reference.id] = delete_button
            rows.append(
                widgets.HBox(
                    [label, time_input, color_input, delete_button],
                    layout=widgets.Layout(
                        width="100%",
                        min_height="36px",
                        align_items="center",
                        grid_gap="4px",
                    ),
                )
            )
        if not rows:
            rows.append(
                widgets.HTML(
                    "<span style='color:#666'>No time references</span>",
                    layout=widgets.Layout(padding="4px"),
                )
            )
        previous = self.reference_box.children
        self.reference_box.children = tuple(rows)
        for child in previous:
            child.close()

    def _on_type_change(self, event_type: str, change: dict[str, tp.Any]) -> None:
        if self._updating_controls:
            return
        if bool(change["new"]):
            self.state.visible_types.add(event_type)
        else:
            self.state.visible_types.discard(event_type)
        self.visible_events = self._filter_visible_types(self._query_filtered_events)
        self._after_visibility_change()

    def _move_lane(self, direction: int) -> None:
        selected = self.lane_picker.value
        if selected is None:
            return
        index = self.state.lane_order.index(str(selected))
        target = index + direction
        if not 0 <= target < len(self.state.lane_order):
            return
        self.state.lane_order[index], self.state.lane_order[target] = (
            self.state.lane_order[target],
            self.state.lane_order[index],
        )
        self._rebuild_type_controls()
        self.lane_picker.value = selected
        self._replace_timeline()

    def _on_apply_filters(self, _button: tp.Any) -> None:
        query = self.query_input.value.strip()
        search = self.search_input.value.strip()
        try:
            filtered = self._filter_by_text(
                self.events,
                query=query,
                search=search,
            )
        except Exception as exc:
            self.query_status.value = (
                "<span style='color:#a33'>"
                f"Could not apply query: {html.escape(str(exc))}</span>"
            )
            return
        self.state.query = query
        self.state.search = search
        self._query_filtered_events = filtered
        self.visible_events = self._filter_visible_types(filtered)
        self._rebuild_type_controls()
        self._after_visibility_change()

    def _on_reset_filters(self, _button: tp.Any) -> None:
        self.query_input.value = ""
        self.search_input.value = ""
        self.state.query = ""
        self.state.search = ""
        self._query_filtered_events = self.events.copy()
        self.state.visible_types = set(self.events["type"].astype(str))
        self.visible_events = self.events.copy()
        self._rebuild_type_controls()
        self._after_visibility_change()

    def _after_visibility_change(self) -> None:
        visible_ids = set(self.visible_events["_event_id"].astype(int))
        if (
            self.selected_event_id is not None
            and self.selected_event_id not in visible_ids
        ):
            self._clear_selection()
        self._replace_timeline()
        self._update_filter_status()
        self._update_selection_controls()

    def _update_filter_status(self) -> None:
        self.query_status.value = (
            "<span style='color:#666'>"
            f"Showing {len(self.visible_events)} of {len(self.events)} events</span>"
        )

    def _on_timeline_change(self, change: dict[str, tp.Any]) -> None:
        if self._updating_controls or change.get("new") == self.state.timeline:
            return
        previous_types = set(self.events["type"].astype(str))
        showing_all_types = self.state.visible_types == previous_types
        previous_visible_types = set(self.state.visible_types)
        self.state.timeline = change["new"]
        self.events = self._events_for_timeline(self.state.timeline)
        self._full_time_range = self._time_extent(self.events)
        self.state.references.clear()
        self._next_reference_number = 1
        available_types = set(self.events["type"].astype(str))
        self.state.visible_types = (
            available_types
            if showing_all_types
            else available_types & previous_visible_types
        )
        self.state.lane_order = self._ordered_types(available_types)
        try:
            self._query_filtered_events = self._filter_by_text(self.events)
        except Exception as exc:
            self.state.query = ""
            self.query_input.value = ""
            self._query_filtered_events = self._filter_by_text(self.events, query="")
            self.query_status.value = (
                "<span style='color:#a33'>Query was reset for this timeline: "
                f"{html.escape(str(exc))}</span>"
            )
        self.visible_events = self._filter_visible_types(self._query_filtered_events)
        self.visible_time_range = self._full_time_range
        self._clear_selection()
        self._rebuild_type_controls()
        self._sync_time_controls()
        self._sync_reference_controls()
        self._replace_timeline()
        self._update_filter_status()

    def _sync_time_controls(self) -> None:
        if not hasattr(self, "time_slider"):
            return
        self._updating_controls = True
        try:
            full_start, full_stop = self._full_time_range
            start, stop = self.visible_time_range
            slider_start = min(max(start, full_start), full_stop)
            slider_stop = min(max(stop, full_start), full_stop)
            if slider_stop <= slider_start:
                slider_start, slider_stop = full_start, full_stop
            self.time_slider.min = min(self.time_slider.min, full_start, slider_start)
            self.time_slider.max = max(self.time_slider.max, full_stop, slider_stop)
            self.time_slider.value = (slider_start, slider_stop)
            self.time_slider.min = full_start
            self.time_slider.max = full_stop
            self.time_slider.step = max((full_stop - full_start) / 1000, 0.001)
            self.time_start_input.value = start
            self.time_stop_input.value = stop
            self.time_status.value = ""
        finally:
            self._updating_controls = False

    def _on_time_slider(self, change: dict[str, tp.Any]) -> None:
        if not self._updating_controls:
            self._apply_time_control(change["new"])

    def _on_time_input(self, _change: dict[str, tp.Any]) -> None:
        if not self._updating_controls:
            self._apply_time_control(
                (self.time_start_input.value, self.time_stop_input.value)
            )

    def _on_reset_time(self, _button: tp.Any) -> None:
        self._apply_time_control(self._full_time_range)

    def _apply_time_control(self, value: tp.Sequence[float]) -> None:
        try:
            time_range = self._coerce_time_range(value)
        except ValueError as exc:
            self.time_status.value = (
                f"<span style='color:#a33'>{html.escape(str(exc))}</span>"
            )
            return
        self._set_visible_range(time_range)
        with self.timeline.batch_update():
            self.timeline.layout.xaxis.autorange = False
            self.timeline.layout.xaxis.range = list(time_range)

    def _on_add_reference(self, _button: tp.Any) -> None:
        self.add_reference()

    def _on_reference_input(
        self,
        reference_id: str,
        change: dict[str, tp.Any],
    ) -> None:
        if self._updating_reference:
            return
        try:
            self.update_reference(reference_id, time=float(change["new"]))
        except ValueError as exc:
            self._sync_reference_controls()
            self.reference_status.value = (
                f"<span style='color:#a33'>{html.escape(str(exc))}</span>"
            )

    def _on_reference_color(
        self,
        reference_id: str,
        change: dict[str, tp.Any],
    ) -> None:
        if not self._updating_reference:
            self.update_reference(reference_id, color=str(change["new"]))

    def _on_reference_shape_change(self, shape: tp.Any, x0: tp.Any, x1: tp.Any) -> None:
        if self._updating_reference or x0 is None or x1 is None:
            return
        reference_id = self._reference_id_from_shape(shape)
        if reference_id is None:
            return
        current = next(
            (
                reference
                for reference in self.state.references
                if reference.id == reference_id
            ),
            None,
        )
        if current is None:
            return
        full_start, full_stop = self._full_time_range
        x0, x1 = float(x0), float(x1)
        if np.isclose(x0, x1):
            candidate = x0
        elif np.isclose(x0, current.time):
            candidate = x1
        elif np.isclose(x1, current.time):
            candidate = x0
        else:
            candidate = (x0 + x1) / 2
        time = min(max(candidate, full_start), full_stop)
        self.update_reference(reference_id, time=time, update_timeline=False)
        if not np.isclose(time, x0) or not np.isclose(time, x1):
            self._sync_reference_shapes()

    def _new_reference_id(self) -> tuple[str, int]:
        used = {reference.id for reference in self.state.references}
        while f"R{self._next_reference_number}" in used:
            self._next_reference_number += 1
        number = self._next_reference_number
        self._next_reference_number += 1
        return f"R{number}", number

    def add_reference(
        self,
        time: float | None = None,
        color: str | None = None,
    ) -> TimeReference:
        """Add a reference, defaulting to the visible time-window midpoint."""
        if time is None:
            full_start, full_stop = self._full_time_range
            midpoint = sum(self.visible_time_range) / 2
            time = min(max(midpoint, full_start), full_stop)
        reference_id, number = self._new_reference_id()
        reference = TimeReference(
            id=reference_id,
            time=tp.cast(float, self._coerce_reference_time(time)),
            color=color
            or TIME_REFERENCE_COLORS[(number - 1) % len(TIME_REFERENCE_COLORS)],
        )
        self.state.references.append(reference)
        self._rebuild_reference_controls()
        self._sync_reference_shapes()
        self._schedule_reference_refresh()
        return reference

    def update_reference(
        self,
        reference_id: str,
        *,
        time: float | None = None,
        color: str | None = None,
        update_timeline: bool = True,
    ) -> TimeReference:
        """Update one reference by its stable identifier."""
        index = next(
            (
                index
                for index, reference in enumerate(self.state.references)
                if reference.id == reference_id
            ),
            None,
        )
        if index is None:
            raise KeyError(f"unknown time reference: {reference_id!r}")
        previous = self.state.references[index]
        reference = TimeReference(
            id=previous.id,
            time=(
                previous.time
                if time is None
                else tp.cast(float, self._coerce_reference_time(time))
            ),
            color=previous.color if color is None else color,
        )
        if reference == previous:
            self._sync_reference_controls()
            if update_timeline:
                self._sync_reference_shapes()
            return reference
        self.state.references[index] = reference
        self._sync_reference_controls()
        if update_timeline:
            self._sync_reference_shapes()
        self._schedule_reference_refresh()
        return reference

    def remove_reference(self, reference_id: str) -> None:
        """Remove one reference by its stable identifier."""
        remaining = [
            reference
            for reference in self.state.references
            if reference.id != reference_id
        ]
        if len(remaining) == len(self.state.references):
            raise KeyError(f"unknown time reference: {reference_id!r}")
        self.state.references[:] = remaining
        self._rebuild_reference_controls()
        self._sync_reference_shapes()
        self._schedule_reference_refresh()

    def clear_references(self) -> None:
        """Remove all time references."""
        if not self.state.references:
            return
        self.state.references.clear()
        self._rebuild_reference_controls()
        self._sync_reference_shapes()
        self._schedule_reference_refresh()

    def _set_reference_time(
        self,
        value: float | None,
        *,
        update_timeline: bool = True,
    ) -> None:
        if value is None:
            self.clear_references()
            return
        if self.state.references:
            self.update_reference(
                self.state.references[0].id,
                time=value,
                update_timeline=update_timeline,
            )
        else:
            self.add_reference(time=value)

    def set_reference_time(self, value: float | None) -> None:
        """Set or clear the timeline reference from Python."""
        self._set_reference_time(value)

    def _sync_reference_controls(self) -> None:
        if not hasattr(self, "reference_box"):
            return
        expected_ids = {reference.id for reference in self.state.references}
        if expected_ids != set(self._reference_time_inputs):
            self._rebuild_reference_controls()
            return
        self._updating_reference = True
        try:
            for reference in self.state.references:
                self._reference_time_inputs[reference.id].value = reference.time
                self._reference_color_inputs[reference.id].value = reference.color
            self.reference_status.value = ""
        finally:
            self._updating_reference = False

    def _schedule_reference_refresh(self) -> None:
        if self._active_view is None or not self._active_view.time_sensitive:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._refresh_active_view()
            return
        if self._reference_task is not None and not self._reference_task.done():
            self._reference_task.cancel()
        self._reference_task = loop.create_task(self._refresh_after_reference_change())

    async def _refresh_after_reference_change(self) -> None:
        try:
            await asyncio.sleep(self.zoom_debounce)
        except asyncio.CancelledError:
            return
        self._refresh_active_view()

    def _update_selection_controls(self) -> None:
        if not hasattr(self, "selection_summary"):
            return
        selected = self.selected_event_id
        if selected is None:
            self.selection_summary.value = (
                "<span style='color:#666'>No event selected</span>"
            )
            return
        match = self.events.loc[self.events["_event_id"] == selected]
        if match.empty:
            return
        row = match.iloc[0]
        start = float(row["start"])
        duration_value = row.get("duration", 0)
        duration = 0.0 if pd.isna(duration_value) else float(duration_value)
        self.selection_summary.value = (
            f"<b>{html.escape(str(row['type']))}</b><br>"
            f"{start:.3f}–{start + duration:.3f} s · event {selected}"
        )

    def _clear_selection(self) -> None:
        if self._zoom_task is not None and not self._zoom_task.done():
            self._zoom_task.cancel()
        if self._reference_task is not None and not self._reference_task.done():
            self._reference_task.cancel()
        self.selected_event_id = None
        self.state.active_view = None
        self._context = None
        self._views = {}
        self._active_view = None
        self.detail_title.value = (
            "<div style='padding:8px 0;color:#666'>Click an event to inspect it.</div>"
        )
        self._updating_picker = True
        self.view_picker.options = []
        self.view_picker.layout.display = "none"
        self._updating_picker = False
        previous = self.detail_content.children
        self.detail_content.children = ()
        for child in previous:
            child.close()
        self._update_selection_controls()

    def _on_visible_range(self, _layout: tp.Any, value: tp.Any) -> None:
        if value is not None and len(value) == 2:
            self._set_visible_range((float(value[0]), float(value[1])))

    def _on_autorange(self, _layout: tp.Any, value: tp.Any) -> None:
        if value is True:
            self._set_visible_range(self._full_time_range)

    def _set_visible_range(self, value: tuple[float, float]) -> None:
        if value == self.visible_time_range:
            return
        self.visible_time_range = value
        self._sync_time_controls()
        self._schedule_zoom_refresh()

    def _schedule_zoom_refresh(self) -> None:
        if self._active_view is None or not self._active_view.time_sensitive:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._refresh_active_view()
            return
        if self._zoom_task is not None and not self._zoom_task.done():
            self._zoom_task.cancel()
        self._zoom_task = loop.create_task(self._refresh_after_zoom())

    async def _refresh_after_zoom(self) -> None:
        try:
            await asyncio.sleep(self.zoom_debounce)
        except asyncio.CancelledError:
            return
        self._refresh_active_view()

    def _refresh_active_view(self) -> None:
        if self._active_view is not None and self._active_view.time_sensitive:
            self._render(self._active_view)

    def _on_click(self, trace: tp.Any, points: tp.Any, _state: tp.Any) -> None:
        if not points.point_inds:
            return
        point = int(points.point_inds[0])
        event_id = int(trace.customdata[point])
        self.select_event(event_id)

    def select_event(self, event_id: int) -> None:
        """Select an event by its stable id; useful from code and tests too."""
        positions = np.flatnonzero(
            self.events["_event_id"].astype(int).to_numpy() == int(event_id)
        )
        if not len(positions):
            raise IndexError(
                f"event id {event_id} is not part of timeline {self.state.timeline!r}"
            )
        row_position = int(positions[0])
        row = self.events.iloc[row_position]
        event_fields = row.drop(labels=["_event_id"], errors="ignore")
        event = etypes.Event.from_dict(event_fields.to_dict())
        self.selected_event_id = event_id
        self._context = EventViewContext(
            events=self.events,
            row_position=row_position,
            row=row,
            event=event,
            visible_time_range=self.visible_time_range,
            data_provider=self.data_provider,
            reference_time=self.state.reference_time,
            references=self.references,
        )
        views = self.registry.views_for(event)
        self._views = {view.name: view for view in views}
        if not views:
            raise LookupError(
                f"no event detail view registered for {type(event).__name__}"
            )

        self.detail_title.value = (
            "<div style='padding:8px 0'>"
            f"<b>{html.escape(type(event).__name__)}</b>"
            f" · {event.start:.3f}–{event.stop:.3f} s"
            "</div>"
        )
        self._updating_picker = True
        self.view_picker.options = [(view.label, view.name) for view in views]
        desired = self.default_views.get(type(event).__name__)
        selected_view = next(
            (view for view in views if view.name == desired),
            views[0],
        )
        self.view_picker.value = selected_view.name
        self.view_picker.layout.display = "" if len(views) > 1 else "none"
        self._updating_picker = False
        self._render(selected_view)
        self._update_selection_controls()

    def _on_view_change(self, change: dict[str, tp.Any]) -> None:
        if self._updating_picker or change.get("new") is None:
            return
        self._render(self._views[str(change["new"])])

    def _render(self, spec: EventViewSpec) -> None:
        if self._context is None:
            return
        self._active_view = spec
        self.state.active_view = spec.name
        # The time range may have changed since the event was selected.
        context = dataclasses.replace(
            self._context,
            visible_time_range=self.visible_time_range,
            reference_time=self.state.reference_time,
            references=self.references,
        )
        try:
            detail = self._as_detail_widget(spec.renderer(context))
        except Exception as exc:  # render errors belong in this explorer's pane
            widgets, _ = _widgets()
            detail = widgets.HTML(
                "<div style='color:#a33'>"
                f"Could not render {html.escape(spec.label)}: "
                f"{html.escape(str(exc))}</div>"
            )
        self._replace_detail(detail)

    @staticmethod
    def _as_detail_widget(value: tp.Any) -> tp.Any:
        """Normalize renderer output without using the notebook output channel."""
        widgets, _ = _widgets()
        if isinstance(value, widgets.Widget):
            return value
        if isinstance(value, go.Figure):
            return go.FigureWidget(value)
        if value is None:
            return widgets.HTML("<i>This view did not return any content.</i>")
        repr_html = getattr(value, "_repr_html_", None)
        if callable(repr_html):
            rendered = repr_html()
            if rendered is not None:
                return widgets.HTML(value=str(rendered))
        return widgets.HTML(
            value=(
                "<pre style='white-space:pre-wrap;margin:0'>"
                f"{html.escape(str(value))}</pre>"
            )
        )

    def _replace_detail(self, detail: tp.Any) -> None:
        """Replace only this explorer's child and release the superseded widgets."""
        previous = self.detail_content.children
        self.detail_content.children = (detail,)
        for child in previous:
            if child is not detail:
                child.close()

    def _ipython_display_(self) -> None:
        """Let the explorer itself be the last expression in a notebook cell."""
        _, display = _widgets()
        display(self.widget)


def explore_events(
    events: pd.DataFrame,
    *,
    registry: EventViewRegistry | None = None,
    lanes: tp.Sequence[str] | None = None,
    label: str | None = None,
    data_provider: EventDataProvider | None = None,
    zoom_debounce: float = 0.2,
    query: str | None = None,
    search: str | None = None,
    time_range: tuple[float, float] | None = None,
    visible_types: tp.Collection[str] | None = None,
    default_views: tp.Mapping[str, str] | None = None,
    initial_timeline: tp.Any | None = None,
    show_controls: bool = True,
    top_row_height: int = 600,
    event_type_list_height: int = 180,
    reference_list_height: int = 128,
    references: (
        tp.Sequence[TimeReference | float | tp.Mapping[str, tp.Any]] | None
    ) = None,
    reference_time: float | None = None,
) -> EventExplorer:
    """Create a live-kernel dashboard for one or more event timelines.

    Query, search, and type controls filter visible marks only. Detail renderers retain
    the complete active timeline so ``EventViewContext.related`` can still find hidden
    contextual streams.
    """
    return EventExplorer(
        events,
        registry=registry,
        lanes=lanes,
        label=label,
        data_provider=data_provider,
        zoom_debounce=zoom_debounce,
        query=query,
        search=search,
        time_range=time_range,
        visible_types=visible_types,
        default_views=default_views,
        initial_timeline=initial_timeline,
        show_controls=show_controls,
        top_row_height=top_row_height,
        event_type_list_height=event_type_list_height,
        reference_list_height=reference_list_height,
        references=references,
        reference_time=reference_time,
    )


__all__ = ["EventExplorer", "ExplorerState", "explore_events"]
