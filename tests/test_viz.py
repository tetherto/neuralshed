"""Clickable event timeline and renderer dispatch, without loading source datasets."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from neuralset.events import etypes

from neuralshed.events import PositionSeries, SpikeCounts
from neuralshed.viz import EventViewContext, EventViewRegistry, TimeReference
from neuralshed.viz import explore_events
from neuralshed.viz import plot_event_timeline
from neuralshed.viz.defaults import default_event_view_registry
from neuralshed.viz.registry import CachingEventDataProvider
from neuralshed.viz.registry import DirectEventDataProvider


def simple_events() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "type": "Stimulus",
                "timeline": "t1",
                "start": 0.0,
                "duration": 1.0,
                "code": 1,
                "description": "cue",
            },
            {
                "type": "Sentence",
                "timeline": "t1",
                "start": 1.0,
                "duration": 2.0,
                "text": "hello world",
                "language": "en",
            },
            {
                "type": "Stimulus",
                "timeline": "t1",
                "start": 3.0,
                "duration": 0.0,
                "code": 2,
                "description": "done",
            },
        ]
    )


def test_timeline_groups_rows_and_preserves_click_identity():
    events = simple_events()
    fig = plot_event_timeline(events, lanes=["Stimulus", "Sentence"])

    assert len(fig.data) == 3  # Stimulus spans, Stimulus points, Sentence spans
    event_ids = sorted(int(x) for trace in fig.data for x in trace.customdata)
    assert event_ids == list(range(len(events)))
    assert list(fig.layout.yaxis.categoryarray) == ["Sentence", "Stimulus"]


def test_timeline_uses_stable_ids_from_a_filtered_frame():
    events = simple_events().assign(_event_id=[10, 20, 30]).iloc[[0, 2]]
    fig = plot_event_timeline(events)

    event_ids = sorted(int(x) for trace in fig.data for x in trace.customdata)
    assert event_ids == [10, 30]


def test_registry_keeps_parent_views_and_overrides_names_by_specificity():
    registry = EventViewRegistry()
    registry.register(etypes.Event, lambda ctx: "fields", name="fields")
    registry.register(etypes.BaseText, lambda ctx: "parent", name="main", default=True)
    registry.register(etypes.Sentence, lambda ctx: "child", name="main", default=True)

    event = etypes.Sentence(timeline="t1", start=0.0, text="hello")
    views = registry.views_for(event)

    assert [view.name for view in views] == ["main", "fields"]
    assert views[0].renderer(None) == "child"


def test_registry_records_whether_a_view_follows_time():
    registry = EventViewRegistry()
    registry.register(
        etypes.Event,
        lambda ctx: "plot",
        name="plot",
        time_sensitive=True,
    )

    event = etypes.Stimulus(timeline="t1", start=0.0)
    assert registry.views_for(event)[0].time_sensitive is True


def test_context_finds_overlapping_related_events():
    events = simple_events()
    row = events.iloc[1]
    event = etypes.Event.from_dict(row.to_dict())
    context = EventViewContext(
        events=events,
        row_position=1,
        row=row,
        event=event,
        visible_time_range=(0.0, 3.0),
        data_provider=DirectEventDataProvider(),
    )

    related = context.related("Stimulus")
    assert related.description.tolist() == ["cue", "done"]


def test_default_dense_views_read_a_synthetic_event(tmp_path):
    path = tmp_path / "position.npy"
    np.save(path, np.array([[0, 1, 2], [2, 1, 0]], dtype=np.float32))
    row = pd.Series(
        PositionSeries(
            timeline="t1",
            subject="s1",
            start=0.0,
            duration=3.0,
            frequency=1.0,
            filepath=str(path),
            feature="cursor",
            unit="a.u.",
            dims=("x", "y"),
            n_features=2,
        ).to_dict()
    )
    context = EventViewContext(
        events=pd.DataFrame([row]),
        row_position=0,
        row=row,
        event=etypes.Event.from_dict(row.to_dict()),
        visible_time_range=(0.0, 3.0),
        data_provider=DirectEventDataProvider(),
        references=(
            TimeReference(id="R1", time=0.9, color="#d1495b"),
            TimeReference(id="R2", time=2.1, color="#0072b2"),
        ),
    )
    views = default_event_view_registry().views_for(context.event)

    assert [view.name for view in views] == [
        "trajectory",
        "traces",
        "heatmap",
        "fields",
    ]
    assert [view.time_sensitive for view in views] == [True, True, True, False]
    rendered = {view.name: view.renderer(context) for view in views[:3]}
    assert isinstance(rendered["trajectory"], go.Figure)
    assert [trace.mode for trace in rendered["trajectory"].data[-2:]] == [
        "markers",
        "markers",
    ]
    assert [trace.customdata[0] for trace in rendered["trajectory"].data[-2:]] == [
        1.0,
        2.0,
    ]
    assert [trace.marker.color for trace in rendered["trajectory"].data[-2:]] == [
        "#d1495b",
        "#0072b2",
    ]
    assert all(trace.mode == "lines" for trace in rendered["traces"].data)
    assert [shape.x0 for shape in rendered["traces"].layout.shapes] == [0.9, 2.1]
    assert [shape.x0 for shape in rendered["heatmap"].layout.shapes] == [0.9, 2.1]


def test_context_keeps_legacy_scalar_reference_compatible():
    events = simple_events()
    row = events.iloc[1]
    context = EventViewContext(
        events=events,
        row_position=1,
        row=row,
        event=etypes.Event.from_dict(row.to_dict()),
        visible_time_range=(0.0, 3.0),
        data_provider=DirectEventDataProvider(),
        reference_time=1.5,
    )

    assert context.time_references == (
        TimeReference(id="R1", time=1.5, color="#d1495b"),
    )


def test_explorer_selects_an_event_and_offers_its_views():
    explorer = explore_events(simple_events())
    explorer.select_event(1)

    assert explorer.selected_event_id == 1
    assert explorer._context is not None
    assert explorer._context.event.text == "hello world"
    assert [value for _, value in explorer.view_picker.options] == ["text", "fields"]


def test_explorer_accepts_default_views_and_keeps_picker_in_compact_mode():
    explorer = explore_events(
        simple_events(),
        default_views={"Sentence": "fields"},
        show_controls=False,
    )
    explorer.select_event(1)

    assert explorer.state.active_view == "fields"
    assert explorer.view_picker.value == "fields"
    assert explorer.view_picker in explorer.detail.children


def test_dashboard_query_filters_marks_but_preserves_full_context():
    explorer = explore_events(simple_events(), query='type == "Stimulus"')

    assert explorer.visible_events["_event_id"].tolist() == [0, 2]
    plotted_ids = sorted(
        int(event_id)
        for trace in explorer.timeline.data
        for event_id in trace.customdata
    )
    assert plotted_ids == [0, 2]

    explorer.select_event(2)
    assert explorer.selected_event_id == 2
    assert explorer._context is not None
    assert len(explorer._context.events) == 3


def test_dashboard_controls_filter_types_and_synchronize_time():
    explorer = explore_events(simple_events(), time_range=(0.5, 2.5))

    assert explorer.time_slider.value == (0.5, 2.5)
    explorer._type_checks["Sentence"].value = False
    assert set(explorer.visible_events["type"]) == {"Stimulus"}

    explorer.time_slider.value = (1.0, 2.0)
    assert explorer.visible_time_range == (1.0, 2.0)
    assert tuple(explorer.timeline.layout.xaxis.range) == (1.0, 2.0)


def test_dashboard_reference_editor_adds_edits_drags_and_removes_lines():
    explorer = explore_events(
        simple_events(),
        references=[
            {"time": 0.5, "color": "#d1495b"},
            {"time": 1.5, "color": "#0072b2"},
        ],
        reference_list_height=112,
    )

    assert [reference.id for reference in explorer.references] == ["R1", "R2"]
    assert explorer.state.reference_time == 0.5
    assert explorer.reference_box.layout.height == "112px"
    assert len(explorer.reference_box.children) == 2
    assert set(explorer._reference_time_inputs) == {"R1", "R2"}
    assert [shape.x0 for shape in explorer.timeline.layout.shapes] == [0.5, 1.5]
    assert all(shape.editable is True for shape in explorer.timeline.layout.shapes)

    explorer._reference_time_inputs["R2"].value = 1.75
    assert explorer.references[1].time == 1.75
    assert explorer._reference_shape("R2").x0 == 1.75

    explorer._reference_color_inputs["R1"].value = "#123456"
    assert explorer.references[0].color == "#123456"
    assert explorer._reference_shape("R1").line.color == "#123456"

    shape = explorer._reference_shape("R2")
    shape.x0 = 1.25
    shape.x1 = 1.25
    assert explorer.references[1].time == 1.25
    assert explorer._reference_time_inputs["R2"].value == 1.25

    added = explorer.add_reference()
    assert added.id == "R3"
    assert added.time == sum(explorer.visible_time_range) / 2
    assert explorer._reference_shape("R3") is not None

    explorer._reference_delete_buttons["R1"].click()
    assert [reference.id for reference in explorer.references] == ["R2", "R3"]
    assert explorer._reference_shape("R1") is None

    explorer.clear_references()
    assert explorer.references == ()
    assert explorer._reference_shape() is None
    assert "No time references" in explorer.reference_box.children[0].value

    explorer._on_visible_range(explorer.timeline.layout, (-10.0, -5.0))
    assert explorer.add_reference().time == 0.0


def test_dashboard_keeps_legacy_reference_time_api():
    explorer = explore_events(simple_events(), reference_time=1.5)

    assert explorer.references == (TimeReference(id="R1", time=1.5, color="#d1495b"),)
    explorer.set_reference_time(2.25)
    assert explorer.references[0].time == 2.25
    explorer.set_reference_time(None)
    assert explorer.references == ()


def test_dashboard_rejects_plural_and_legacy_reference_arguments_together():
    with pytest.raises(ValueError, match="either references or reference_time"):
        explore_events(simple_events(), references=[1.0], reference_time=2.0)


def test_reference_change_refreshes_only_time_sensitive_detail():
    rendered_references = []
    registry = EventViewRegistry()

    @registry.register(
        etypes.Sentence,
        name="window",
        default=True,
        time_sensitive=True,
    )
    def render_window(context):
        rendered_references.append(
            tuple(reference.time for reference in context.time_references)
        )
        return go.Figure()

    explorer = explore_events(simple_events(), registry=registry, zoom_debounce=0)
    explorer.select_event(1)
    explorer.add_reference(1.5)
    explorer.add_reference(2.0)

    assert rendered_references == [(), (1.5,), (1.5, 2.0)]


def test_dashboard_uses_two_rows_tabs_and_bounded_event_types():
    explorer = explore_events(
        simple_events(),
        top_row_height=520,
        event_type_list_height=132,
        reference_list_height=116,
    )

    assert explorer.widget.children == (explorer.top_row, explorer.detail)
    assert explorer.top_row.children == (explorer.sidebar, explorer.timeline_host)
    assert explorer.detail.children == (
        explorer.detail_title,
        explorer.detail_content,
    )
    assert explorer.top_row.layout.height == "520px"
    assert explorer.timeline.layout.height == 520
    assert explorer.sidebar_tabs.get_title(0) == "Filter"
    assert explorer.sidebar_tabs.get_title(1) == "View"
    assert explorer.filter_tab.children == (
        explorer.timeline_picker,
        explorer.query_input,
        explorer.search_input,
        explorer.filter_actions,
        explorer.query_status,
        explorer.type_box,
        explorer.lane_picker,
        explorer.lane_actions,
    )
    assert explorer.type_box in explorer.filter_tab.children
    assert explorer.type_box.layout.height == "132px"
    assert explorer.type_box.layout.overflow == "auto"
    assert all(
        checkbox.layout.height == "24px" for checkbox in explorer._type_checks.values()
    )
    assert explorer.time_input_row in explorer.view_tab.children
    assert explorer.time_input_row.children == (
        explorer.time_start_input,
        explorer.time_stop_input,
        explorer.reset_time_button,
    )
    assert explorer.time_start_input.description == ""
    assert explorer.time_stop_input.description == ""
    assert explorer.reset_time_button.description == "Full"
    assert explorer.time_input_row.layout.grid_gap == "4px"
    assert {control.layout.flex for control in explorer.time_input_row.children} == {
        "1 1 0"
    }
    assert {control.layout.width for control in explorer.time_input_row.children} == {
        "auto"
    }
    assert explorer.add_reference_button in explorer.view_tab.children
    assert explorer.reference_box in explorer.view_tab.children
    assert explorer.reference_box.layout.height == "116px"
    assert explorer.reference_box.layout.overflow == "auto"
    assert explorer.selection_summary in explorer.view_tab.children
    assert not hasattr(explorer, "previous_button")
    assert not hasattr(explorer, "next_button")
    assert all(
        "Selected event" not in (getattr(child, "value", "") or "")
        for child in explorer.view_tab.children
    )


def test_dashboard_switches_between_timelines_and_clears_selection():
    first = simple_events()
    second = simple_events().assign(timeline="t2", start=lambda frame: frame.start + 10)
    events = pd.concat([first, second], ignore_index=True)
    explorer = explore_events(events, reference_time=1.5)
    explorer.select_event(1)

    explorer.timeline_picker.value = "t2"

    assert explorer.state.timeline == "t2"
    assert set(explorer.events["timeline"]) == {"t2"}
    assert explorer.visible_time_range == (10.0, 13.0)
    assert explorer.selected_event_id is None
    assert explorer.state.reference_time is None
    assert explorer.references == ()


def test_explorer_refreshes_active_time_view_after_range_change():
    rendered_ranges = []
    registry = EventViewRegistry()

    @registry.register(
        etypes.Sentence,
        name="window",
        default=True,
        time_sensitive=True,
    )
    def render_window(context):
        rendered_ranges.append(context.visible_time_range)
        return "detail"

    explorer = explore_events(simple_events(), registry=registry, zoom_debounce=0)
    explorer.select_event(1)
    explorer._on_visible_range(explorer.timeline.layout, (1.25, 2.25))

    assert rendered_ranges == [(0.0, 3.0), (1.25, 2.25)]


def test_zoom_replaces_only_the_originating_explorers_detail():
    left_ranges = []
    right_ranges = []

    def registry_for(rendered_ranges):
        registry = EventViewRegistry()

        @registry.register(
            etypes.Sentence,
            name="window",
            default=True,
            time_sensitive=True,
        )
        def render_window(context):
            rendered_ranges.append(context.visible_time_range)
            return go.Figure(go.Scatter(x=[0, 1], y=[0, 1]))

        return registry

    left = explore_events(
        simple_events(), registry=registry_for(left_ranges), zoom_debounce=0
    )
    right = explore_events(
        simple_events(), registry=registry_for(right_ranges), zoom_debounce=0
    )
    left.select_event(1)
    right.select_event(1)
    left_before = left.detail_content.children[0]
    right_before = right.detail_content.children[0]

    left._on_visible_range(left.timeline.layout, (1.25, 2.25))

    assert isinstance(left.detail_content.children[0], go.FigureWidget)
    assert left.detail_content.children[0] is not left_before
    assert right.detail_content.children[0] is right_before
    assert left_ranges == [(0.0, 3.0), (1.25, 2.25)]
    assert right_ranges == [(0.0, 3.0)]


def test_cached_provider_reads_each_event_once(tmp_path):
    path = tmp_path / "position.npy"
    np.save(path, np.zeros((2, 3), dtype=np.float32))
    event = PositionSeries(
        timeline="t1",
        subject="s1",
        start=0,
        duration=3,
        frequency=1,
        filepath=str(path),
        feature="cursor",
        unit="a.u.",
        dims=("x", "y"),
        n_features=2,
    )

    class CountingProvider:
        def __init__(self):
            self.calls = 0

        def read(self, _event):
            self.calls += 1
            return np.ones((2, 3))

    provider = CountingProvider()
    cached = CachingEventDataProvider(provider)

    first = cached.read(event)
    second = cached.read(etypes.Event.from_dict(event.to_dict()))

    assert provider.calls == 1
    assert first is second


def test_unrelated_event_class_still_has_fields_view(tmp_path):
    path = tmp_path / "spikes.npy"
    np.save(path, np.zeros((1, 1), dtype=np.float32))
    event = SpikeCounts(
        timeline="t1",
        subject="s1",
        start=0,
        duration=1,
        frequency=1,
        filepath=str(path),
        feature="spikes",
        n_features=1,
        unit="counts/bin",
    )
    names = [view.name for view in default_event_view_registry().views_for(event)]
    assert names == ["heatmap", "fields"]
