from __future__ import annotations

from io import BytesIO
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative


def assign_lanes(df_annotator: pd.DataFrame) -> pd.DataFrame:
    """Place overlapping intervals from one annotator on separate visual lanes."""
    ordered = df_annotator.sort_values(
        ["start", "end", "event"]
    ).reset_index(drop=True)

    lane_ends: list[float] = []
    lane_assignments: list[int] = []

    for _, row in ordered.iterrows():
        lane = None
        for index, lane_end in enumerate(lane_ends):
            if row["start"] >= lane_end:
                lane = index
                lane_ends[index] = row["end"]
                break

        if lane is None:
            lane = len(lane_ends)
            lane_ends.append(row["end"])

        lane_assignments.append(lane)

    result = ordered.copy()
    result["lane"] = lane_assignments
    return result


def prepare_plot_data(
    df: pd.DataFrame,
    annotator_order: list[str],
) -> tuple[pd.DataFrame, dict[str, int], dict[str, float], float]:
    parts = []
    max_lanes: dict[str, int] = {}

    for annotator in annotator_order:
        subset = df[df["annotator"] == annotator].copy()
        if subset.empty:
            continue
        subset = assign_lanes(subset)
        parts.append(subset)
        max_lanes[annotator] = int(subset["lane"].max()) + 1

    if not parts:
        return pd.DataFrame(), {}, {}, 0.0

    plotted = pd.concat(parts, ignore_index=True)

    y_base: dict[str, float] = {}
    current_y = 0.0
    for annotator in annotator_order:
        if annotator not in max_lanes:
            continue
        y_base[annotator] = current_y
        current_y += max_lanes[annotator] + 1.0

    plotted["y"] = plotted.apply(
        lambda row: y_base[row["annotator"]] + row["lane"] + 0.45,
        axis=1,
    )
    return plotted, max_lanes, y_base, current_y


def format_seconds(value: float) -> str:
    total = max(0, int(round(value)))
    hours = total // 3600
    minutes = (total % 3600) // 60
    seconds = total % 60
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def wrap_annotator_name(
    value: str,
    width: int = 18,
    html: bool = True,
) -> str:
    """Wrap long filenames so the timeline keeps more horizontal space."""
    lines = textwrap.wrap(
        str(value),
        width=width,
        break_long_words=True,
        break_on_hyphens=True,
    ) or [str(value)]
    return ("<br>" if html else "\n").join(lines)


def _tick_step(max_time: float) -> int:
    candidates = [10, 20, 30, 60, 120, 300, 600, 900, 1800]
    target = max(max_time / 8.0, 1.0)
    return min(candidates, key=lambda value: abs(value - target))


def _plotly_color_to_hex(value: str) -> str:
    """Convert Plotly rgb(...) colors to hex for Streamlit and Matplotlib."""
    if value.startswith("#"):
        return value
    if value.startswith("rgb(") and value.endswith(")"):
        red, green, blue = [
            int(part.strip())
            for part in value[4:-1].split(",")
        ]
        return f"#{red:02x}{green:02x}{blue:02x}"
    return value


def default_event_colors(events: list[str]) -> dict[str, str]:
    """Return stable, editable default colors for event classes."""
    palette = [
        _plotly_color_to_hex(color)
        for color in (
            qualitative.Safe
            + qualitative.Vivid
            + qualitative.Pastel
            + qualitative.Bold
        )
    ]
    return {
        event: palette[index % len(palette)]
        for index, event in enumerate(events)
    }


def create_timeline_figure(
    df: pd.DataFrame,
    annotator_order: list[str],
    visual_extension: float = 0.0,
    color_map: dict[str, str] | None = None,
    reference_annotator: str | None = None,
) -> go.Figure:
    plotted, max_lanes, y_base, current_y = prepare_plot_data(
        df,
        annotator_order,
    )

    if plotted.empty:
        return go.Figure()

    events = sorted(plotted["event"].unique())
    resolved_colors = default_event_colors(events)
    if color_map:
        resolved_colors.update(
            {event: color_map[event] for event in events if event in color_map}
        )

    figure = go.Figure()

    for event in events:
        event_df = plotted[plotted["event"] == event].copy()
        display_duration = (
            event_df["duration"].astype(float) + visual_extension
        ).clip(lower=0.05)

        custom_data = [
            [
                row.annotator,
                format_seconds(row.start),
                format_seconds(row.end),
                float(row.duration),
            ]
            for row in event_df.itertuples()
        ]

        figure.add_trace(
            go.Bar(
                name=event,
                y=event_df["y"],
                x=display_duration,
                base=event_df["start"],
                orientation="h",
                width=0.78,
                marker={
                    "color": resolved_colors[event],
                    "line": {"color": "black", "width": 1},
                },
                customdata=custom_data,
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "Annotator: %{customdata[0]}<br>"
                    "Start: %{customdata[1]}<br>"
                    "End: %{customdata[2]}<br>"
                    "True duration: %{customdata[3]:.1f} s"
                    "<extra></extra>"
                ),
            )
        )

    tick_values: list[float] = []
    tick_text: list[str] = []
    separator_values: list[float] = []

    visible_annotators = [
        annotator for annotator in annotator_order if annotator in max_lanes
    ]
    for index, annotator in enumerate(visible_annotators):
        center = y_base[annotator] + max_lanes[annotator] / 2.0
        tick_values.append(center)
        wrapped_name = wrap_annotator_name(annotator, width=18, html=True)
        if annotator == reference_annotator:
            tick_text.append(f"<b>EXPERT / REFERENCE</b><br>{wrapped_name}")
        else:
            tick_text.append(wrapped_name)
        if index > 0:
            separator_values.append(y_base[annotator] - 0.5)

    max_time = float((plotted["end"] + visual_extension).max())
    step = _tick_step(max_time)
    x_tick_values = np.arange(0, max_time + step, step)
    x_tick_text = [format_seconds(value) for value in x_tick_values]

    for separator in separator_values:
        figure.add_hline(
            y=separator,
            line_width=2,
            line_color="black",
        )

    total_lanes = sum(max_lanes.values())
    figure.update_layout(
        title="Annotation timeline",
        barmode="overlay",
        dragmode="pan",
        uirevision="annotation-timeline",
        height=max(500, 90 * total_lanes + 95 * len(max_lanes)),
        margin={"l": 85, "r": 25, "t": 65, "b": 125},
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.20,
            "xanchor": "center",
            "x": 0.5,
        },
        xaxis={
            "title": "Time",
            "tickmode": "array",
            "tickvals": x_tick_values,
            "ticktext": x_tick_text,
            "gridcolor": "rgba(0,0,0,0.15)",
            "range": [0, max_time * 1.01 if max_time else 1],
            "fixedrange": False,
            "rangeslider": {"visible": True, "thickness": 0.06},
        },
        yaxis={
            "tickmode": "array",
            "tickvals": tick_values,
            "ticktext": tick_text,
            "tickfont": {"size": 11},
            "range": [-0.5, current_y - 0.5],
            "showgrid": False,
            "title": None,
            "fixedrange": True,
            "automargin": True,
        },
        hovermode="closest",
    )
    return figure


def render_timeline_png(
    df: pd.DataFrame,
    annotator_order: list[str],
    visual_extension: float = 0.0,
    color_map: dict[str, str] | None = None,
    reference_annotator: str | None = None,
    dpi: int = 180,
) -> bytes:
    plotted, max_lanes, y_base, current_y = prepare_plot_data(
        df,
        annotator_order,
    )

    if plotted.empty:
        return b""

    events = sorted(plotted["event"].unique())
    resolved_colors = default_event_colors(events)
    if color_map:
        resolved_colors.update(
            {event: color_map[event] for event in events if event in color_map}
        )

    total_lanes = sum(max_lanes.values())
    figure_height = max(5.0, total_lanes * 0.8 + len(max_lanes) * 0.5)
    figure, axis = plt.subplots(
        figsize=(18, figure_height),
        dpi=dpi,
    )

    for row in plotted.itertuples():
        display_duration = max(
            float(row.duration) + visual_extension,
            0.05,
        )
        y = y_base[row.annotator] + row.lane
        axis.broken_barh(
            [(row.start, display_duration)],
            (y, 0.82),
            facecolors=resolved_colors[row.event],
            edgecolor="black",
            linewidth=1.0,
        )

    y_ticks: list[float] = []
    y_tick_labels: list[str] = []
    visible_annotators = [
        annotator for annotator in annotator_order if annotator in max_lanes
    ]

    for index, annotator in enumerate(visible_annotators):
        y_ticks.append(y_base[annotator] + max_lanes[annotator] / 2.0)
        wrapped_name = wrap_annotator_name(annotator, width=18, html=False)
        if annotator == reference_annotator:
            wrapped_name = f"EXPERT / REFERENCE\n{wrapped_name}"
        y_tick_labels.append(wrapped_name)
        if index > 0:
            axis.axhline(
                y=y_base[annotator] - 0.5,
                color="black",
                linewidth=1.5,
            )

    max_time = float((plotted["end"] + visual_extension).max())
    step = _tick_step(max_time)
    ticks = np.arange(0, max_time + step, step)

    axis.set_xlim(0, max_time * 1.01 if max_time else 1)
    axis.set_ylim(-0.5, current_y - 0.5)
    axis.set_xticks(ticks)
    axis.set_xticklabels([format_seconds(value) for value in ticks])
    axis.set_yticks(y_ticks)
    axis.set_yticklabels(y_tick_labels, fontweight="bold", fontsize=9)
    axis.set_xlabel("Time")
    axis.set_title("Annotation timeline", fontweight="bold")
    axis.grid(axis="x", linestyle="--", alpha=0.4)
    axis.set_axisbelow(True)

    handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=resolved_colors[event],
            edgecolor="black",
        )
        for event in events
    ]
    legend_columns = min(5, max(1, len(events)))
    axis.legend(
        handles,
        events,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=legend_columns,
        frameon=True,
    )

    figure.tight_layout()
    output = BytesIO()
    figure.savefig(
        output,
        format="png",
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(figure)
    output.seek(0)
    return output.getvalue()
