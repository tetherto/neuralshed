"""Plotly timeline for standardized neuralset event frames."""

from __future__ import annotations

import html
import typing as tp

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from neuralset.events import etypes

# Register neuralshed's custom event classes before inspecting Event._CLASSES.
from neuralshed import events as _neuralshed_events  # noqa: F401

_REQUIRED_COLUMNS = {"timeline", "type", "start", "duration"}


def _validate(events: pd.DataFrame) -> None:
    missing = _REQUIRED_COLUMNS - set(events.columns)
    if missing:
        raise ValueError(f"events is missing required columns: {sorted(missing)}")
    timelines = events["timeline"].dropna().unique()
    if len(timelines) != 1:
        raise ValueError(
            f"plot_event_timeline expects one timeline, got {len(timelines)}: "
            f"{list(timelines)}"
        )


def _is_subclass(event_type: str, parent: type[etypes.Event]) -> bool:
    cls = etypes.Event._CLASSES.get(event_type)
    return cls is not None and issubclass(cls, parent)


def _label(row: pd.Series) -> str:
    if _is_subclass(str(row["type"]), etypes.BaseText):
        value = row.get("text", "")
        return "" if pd.isna(value) else str(value)
    return ""


def _hover(row: pd.Series) -> str:
    start = float(row["start"])
    duration = row.get("duration", 0.0)
    duration = 0.0 if pd.isna(duration) else float(duration)
    lines = [
        f"<b>{html.escape(str(row['type']))}</b>",
        f"start: {start:.3f}s",
        f"duration: {duration:.3f}s",
    ]
    for field in ("text", "description", "code", "feature", "unit"):
        value = row.get(field)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        text = str(value)
        if text:
            lines.append(f"{field}: {html.escape(text)}")
    return "<br>".join(lines)


def _title(events: pd.DataFrame, label: str | None) -> str:
    if label is not None:
        return label
    timeline = str(events["timeline"].iloc[0])
    subject = events.get("subject")
    if subject is not None and len(values := subject.dropna().unique()):
        return f"{values[0]} · {timeline}"
    return timeline


def plot_event_timeline(
    events: pd.DataFrame,
    *,
    lanes: tp.Sequence[str] | None = None,
    label: str | None = None,
) -> go.Figure:
    """Plot one timeline with stable event ids attached to every clickable item.

    Unlike neuralset's current ``plot_events``, events of one type share a trace. This keeps
    the trace count proportional to event types rather than event rows. ``customdata`` maps
    back to ``_event_id`` when present, or otherwise to the dataframe row position.
    """
    _validate(events)
    frame = events.reset_index(drop=True)
    event_ids = (
        frame["_event_id"].astype(int).to_numpy()
        if "_event_id" in frame
        else np.arange(len(frame))
    )
    event_types = sorted(frame["type"].astype(str).unique())
    colors = {
        event_type: f"hsl({i * 360 // max(len(event_types), 1)}, 65%, 55%)"
        for i, event_type in enumerate(event_types)
    }
    fig = go.Figure()

    for event_type in event_types:
        positions = np.flatnonzero(frame["type"].astype(str).to_numpy() == event_type)
        subset = frame.iloc[positions]
        durations = subset["duration"].fillna(0).astype(float)

        span_mask = durations.to_numpy() > 0
        if span_mask.any():
            spans = subset.iloc[np.flatnonzero(span_mask)]
            span_positions = event_ids[positions[span_mask]]
            fig.add_trace(
                go.Bar(
                    x=spans["duration"].astype(float),
                    y=[event_type] * len(spans),
                    base=spans["start"].astype(float),
                    customdata=span_positions,
                    ids=[str(x) for x in span_positions],
                    orientation="h",
                    marker_color=colors[event_type],
                    text=[_label(row) or None for _, row in spans.iterrows()],
                    textposition="inside",
                    insidetextanchor="middle",
                    textfont_size=9,
                    hovertext=[_hover(row) for _, row in spans.iterrows()],
                    hoverinfo="text",
                    name=event_type,
                    showlegend=False,
                    width=0.6,
                )
            )

        point_mask = ~span_mask
        if point_mask.any():
            points = subset.iloc[np.flatnonzero(point_mask)]
            point_positions = event_ids[positions[point_mask]]
            fig.add_trace(
                go.Scatter(
                    x=points["start"].astype(float),
                    y=[event_type] * len(points),
                    customdata=point_positions,
                    ids=[str(x) for x in point_positions],
                    mode="markers",
                    marker={
                        "symbol": "line-ns",
                        "size": 12,
                        "color": colors[event_type],
                        "line_width": 2,
                    },
                    hovertext=[_hover(row) for _, row in points.iterrows()],
                    hoverinfo="text",
                    name=event_type,
                    showlegend=False,
                )
            )

    present = set(event_types)
    named = [name for name in lanes or () if name in present]
    lane_order = named + sorted(present.difference(named))
    starts = frame["start"].astype(float)
    stops = starts + frame["duration"].fillna(0).astype(float)
    fig.update_layout(
        title=_title(frame, label),
        xaxis_title="Time (s)",
        xaxis_range=[float(starts.min()), float(stops.max())],
        yaxis={
            "categoryorder": "array",
            "categoryarray": list(reversed(lane_order)),
        },
        barmode="overlay",
        height=max(250, len(event_types) * 50 + 100),
        margin={"l": 120, "r": 30, "t": 50, "b": 50},
        plot_bgcolor="white",
        clickmode="event+select",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eee")
    fig.update_yaxes(showgrid=False)
    return fig


__all__ = ["plot_event_timeline"]
