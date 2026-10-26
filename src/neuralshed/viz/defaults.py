"""Default neuralshed event-detail renderers."""

from __future__ import annotations

import html
import math
import typing as tp

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from neuralset.events import etypes

from neuralshed.events import BehavioralSeries, DenseArrayEvent, PositionSeries

from .registry import DEFAULT_TIME_REFERENCE_COLOR, EventViewContext, EventViewRegistry
from .registry import TimeReference

TIME_REFERENCE_COLOR = DEFAULT_TIME_REFERENCE_COLOR
TIME_REFERENCE_COLORS = (
    TIME_REFERENCE_COLOR,
    "#0072b2",
    "#e69f00",
    "#009e73",
    "#cc79a7",
    "#56b4e9",
)


def _is_missing(value: tp.Any) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing) if isinstance(missing, (bool, np.bool_)) else False


def fields_view(ctx: EventViewContext) -> tp.Any:
    """A compact table of non-empty fields for every event."""
    from ipywidgets import HTML

    rows = []
    for key, value in ctx.row.items():
        if _is_missing(value) or key == "filepath" or str(key).startswith("_"):
            continue
        rendered = html.escape(str(value))
        if len(rendered) > 500:
            rendered = rendered[:497] + "…"
        rows.append(
            "<tr>"
            f"<th style='text-align:left;vertical-align:top;padding:3px 12px 3px 0'>"
            f"{html.escape(str(key))}</th>"
            f"<td style='padding:3px 0'>{rendered}</td>"
            "</tr>"
        )
    return HTML("<table>" + "".join(rows) + "</table>")


def text_view(ctx: EventViewContext) -> tp.Any:
    """Human-readable text and optional phoneme/context fields."""
    from ipywidgets import HTML

    text = html.escape(str(ctx.row.get("text", "")))
    details = []
    for key in ("phonemes", "language", "modality", "context"):
        value = ctx.row.get(key)
        if not _is_missing(value) and str(value):
            details.append(f"<b>{key}</b>: {html.escape(str(value))}")
    suffix = "<br>".join(details)
    return HTML(
        f"<blockquote style='font-size:1.15em;margin:8px 0'>{text}</blockquote>"
        + (f"<div>{suffix}</div>" if suffix else "")
    )


def _slice(
    event: DenseArrayEvent, ctx: EventViewContext
) -> tuple[np.ndarray, np.ndarray]:
    data = np.asarray(ctx.read(event))
    if data.ndim != 2:
        raise ValueError(f"expected two-dimensional data, got {data.shape}")
    start, stop = ctx.display_window()
    frequency = float(event.frequency)
    i0 = max(0, int(math.floor((start - event.start) * frequency)))
    i1 = min(data.shape[1], int(math.ceil((stop - event.start) * frequency)))
    if i1 <= i0:
        i0, i1 = 0, data.shape[1]
    times = event.start + np.arange(i0, i1) / frequency
    return times, data[:, i0:i1]


def _reference_index(
    reference: TimeReference,
    ctx: EventViewContext,
    times: np.ndarray,
) -> int | None:
    if not len(times):
        return None
    start, stop = ctx.display_window()
    if not start <= reference.time <= stop:
        return None
    return int(np.argmin(np.abs(times - reference.time)))


def _add_reference_lines(
    figure: go.Figure, ctx: EventViewContext, times: np.ndarray
) -> None:
    for reference in ctx.time_references:
        if _reference_index(reference, ctx, times) is not None:
            figure.add_vline(
                x=reference.time,
                line={"color": reference.color, "width": 2},
            )


def heatmap_view(ctx: EventViewContext) -> go.Figure:
    """Channel/dimension by time heatmap for any neuralshed dense-array event."""
    event = tp.cast(DenseArrayEvent, ctx.event)
    times, data = _slice(event, ctx)
    labels = list(getattr(event, "dims", ())) or list(range(data.shape[0]))
    fig = go.Figure(
        go.Heatmap(
            x=times,
            y=labels,
            z=data,
            colorscale="Viridis",
            colorbar={"title": event.unit or "value"},
        )
    )
    fig.update_layout(
        title=f"{type(event).__name__} · {event.feature}",
        xaxis_title="Time (s)",
        yaxis_title="Dimension" if getattr(event, "dims", ()) else "Channel",
        margin={"l": 70, "r": 30, "t": 50, "b": 50},
        height=360,
    )
    _add_reference_lines(fig, ctx, times)
    return fig


def traces_view(ctx: EventViewContext) -> go.Figure:
    """Named traces for a behavioral series."""
    event = tp.cast(BehavioralSeries, ctx.event)
    times, data = _slice(event, ctx)
    fig = go.Figure()
    for index, values in enumerate(data):
        label = event.dims[index] if index < len(event.dims) else str(index)
        fig.add_scatter(x=times, y=values, mode="lines", name=label)
    fig.update_layout(
        title=type(event).__name__,
        xaxis_title="Time (s)",
        yaxis_title=event.unit or "Value",
        margin={"l": 70, "r": 30, "t": 50, "b": 50},
        height=340,
    )
    _add_reference_lines(fig, ctx, times)
    return fig


def trajectory_view(ctx: EventViewContext) -> go.Figure:
    """First two dimensions of a position series as a spatial trajectory."""
    event = tp.cast(PositionSeries, ctx.event)
    times, data = _slice(event, ctx)
    if data.shape[0] < 2:
        return traces_view(ctx)
    fig = go.Figure(
        go.Scatter(
            x=data[0],
            y=data[1],
            mode="lines",
            name=type(event).__name__,
            line={"color": "#3b7dd8"},
        )
    )
    fig.update_layout(
        title=type(event).__name__,
        xaxis_title=event.dims[0],
        yaxis_title=event.dims[1],
        yaxis={"scaleanchor": "x", "scaleratio": 1},
        margin={"l": 70, "r": 30, "t": 50, "b": 50},
        height=400,
    )
    for reference in ctx.time_references:
        reference_index = _reference_index(reference, ctx, times)
        if reference_index is None:
            continue
        fig.add_scatter(
            x=[data[0, reference_index]],
            y=[data[1, reference_index]],
            mode="markers",
            marker={
                "color": reference.color,
                "size": 11,
                "line": {"color": "white", "width": 1.5},
            },
            customdata=[times[reference_index]],
            hovertemplate=(
                f"{html.escape(reference.id)}: " "%{customdata:.3f} s<extra></extra>"
            ),
            name=reference.id,
            showlegend=False,
        )
    return fig


def default_event_view_registry() -> EventViewRegistry:
    """Fresh registry containing generic and neuralshed-specific views."""
    registry = EventViewRegistry()
    registry.register(etypes.Event, fields_view, name="fields", label="Fields")
    registry.register(
        etypes.BaseText,
        text_view,
        name="text",
        label="Text",
        default=True,
    )
    registry.register(
        DenseArrayEvent,
        heatmap_view,
        name="heatmap",
        label="Heatmap",
        default=True,
        time_sensitive=True,
    )
    registry.register(
        BehavioralSeries,
        traces_view,
        name="traces",
        label="Traces",
        default=True,
        time_sensitive=True,
    )
    registry.register(
        PositionSeries,
        trajectory_view,
        name="trajectory",
        label="Trajectory",
        default=True,
        time_sensitive=True,
    )
    return registry


__all__ = [
    "TIME_REFERENCE_COLOR",
    "TIME_REFERENCE_COLORS",
    "default_event_view_registry",
    "fields_view",
    "heatmap_view",
    "text_view",
    "traces_view",
    "trajectory_view",
]
