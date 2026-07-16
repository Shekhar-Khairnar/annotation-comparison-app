from __future__ import annotations

import hashlib

import pandas as pd
import plotly.express as px
import streamlit as st

from annotation_app.metrics import compare_annotators, pairwise_iou_matrix
from annotation_app.parser import parse_uploaded_files
from annotation_app.plots import (
    create_timeline_figure,
    default_event_colors,
    render_timeline_png,
)


st.set_page_config(
    page_title="Temporal Annotation Comparator",
    page_icon="📊",
    layout="wide",
)

st.title("Temporal Annotation Comparator")
st.caption(
    "Compare the timing and occurrence of surgical workflow annotations against "
    "an expert/reference annotation. All uploaded data are processed locally."
)

with st.sidebar:
    st.header("Display and matching settings")

    visual_extension = st.slider(
        "Extra display width for each bar",
        min_value=0.0,
        max_value=10.0,
        value=0.0,
        step=0.5,
        format="%.1f seconds",
        help=(
            "Adds visual width to the right side of each bar so very short events "
            "remain visible. This does not change event times or any score."
        ),
    )
    st.caption(
        f"Display only: each bar is drawn {visual_extension:.1f} seconds wider. "
        "The original start/end times are still used for all calculations."
    )

    segment_iou_threshold = st.slider(
        "Required overlap for two segments to count as a match",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.05,
        help=(
            "Used only for segment-level precision, recall, and F1. Two segments "
            "must have the same event name and meet this temporal IoU threshold."
        ),
    )
    st.caption(
        f"At {segment_iou_threshold:.2f}, the shared time must be at least "
        f"{segment_iou_threshold * 100:.0f}% of the total time covered by either "
        "segment. A higher setting is stricter."
    )

uploaded_files = st.file_uploader(
    "Upload annotation files",
    type=["txt"],
    accept_multiple_files=True,
    help="Expected line format: Event Name: (HH:MM:SS, HH:MM:SS)",
)

if not uploaded_files:
    st.info(
        "Upload at least two `.txt` files. Each filename becomes the annotator "
        "name. Example line: `Suture 1: (00:03:31, 00:08:42)`."
    )
    st.stop()

df, diagnostics = parse_uploaded_files(uploaded_files)

if diagnostics:
    diagnostics_df = pd.DataFrame(diagnostics)
    with st.expander(
        "File validation",
        expanded=bool(diagnostics_df["skipped_lines"].sum()),
    ):
        st.dataframe(diagnostics_df, width="stretch", hide_index=True)

if df.empty:
    st.error("No valid annotation rows were found.")
    st.stop()

annotators = list(df["annotator"].drop_duplicates())
if len(annotators) < 2:
    st.warning("At least two valid annotation files are required for comparison.")
    st.stop()

all_events = sorted(df["event"].dropna().unique())

with st.sidebar:
    with st.expander("Event colors", expanded=False):
        st.caption("Choose the color used for each event in interactive and PNG plots.")
        default_colors = default_event_colors(all_events)
        color_map: dict[str, str] = {}

        reset_colors = st.button("Reset event colors", width="stretch")
        if reset_colors:
            for event in all_events:
                digest = hashlib.md5(event.encode("utf-8")).hexdigest()[:10]
                st.session_state.pop(f"event_color_{digest}", None)
            st.rerun()

        for event in all_events:
            digest = hashlib.md5(event.encode("utf-8")).hexdigest()[:10]
            color_map[event] = st.color_picker(
                event,
                value=default_colors[event],
                key=f"event_color_{digest}",
            )

control_col1, control_col2 = st.columns([1, 2])
with control_col1:
    expert = st.selectbox(
        "Expert / reference annotator",
        annotators,
        help=(
            "This file is treated as the reference for overlap, missed segments, "
            "extra segments, and timing errors."
        ),
    )
with control_col2:
    default_comparisons = [a for a in annotators if a != expert]
    comparisons = st.multiselect(
        "Annotators to compare with the expert",
        options=[a for a in annotators if a != expert],
        default=default_comparisons,
        help="Select one or several comparison annotators.",
    )

selected_events = st.multiselect(
    "Events included in the plots and scores",
    options=all_events,
    default=all_events,
    help=(
        "Select one event to isolate it, or select several events for a focused "
        "comparison. This filter applies to the timeline and all agreement scores."
    ),
)

if not selected_events:
    st.warning("Select at least one event to analyze.")
    st.stop()

analysis_df = df[df["event"].isin(selected_events)].copy()
st.caption(
    f"Analyzing {len(selected_events)} of {len(all_events)} event types: "
    + ", ".join(selected_events)
)

tabs = st.tabs(
    [
        "Timeline",
        "Expert comparison",
        "Pairwise agreement",
        "Parsed data",
        "Metric definitions",
    ]
)

with tabs[0]:
    timeline_choices = st.multiselect(
        "Annotators shown in the timeline",
        annotators,
        default=[expert] + comparisons,
        key="timeline_choices",
    )

    if timeline_choices:
        timeline_df = analysis_df[
            analysis_df["annotator"].isin(timeline_choices)
        ].copy()
        fig = create_timeline_figure(
            timeline_df,
            annotator_order=timeline_choices,
            visual_extension=visual_extension,
            color_map=color_map,
            reference_annotator=expert,
        )
        st.info(
            "Timeline controls: drag left or right to move through time, use the "
            "mouse wheel to zoom, drag the small navigator below the axis to change "
            "the visible time range, and double-click the plot to reset."
        )
        st.plotly_chart(
            fig,
            width="stretch",
            config={
                "displaylogo": False,
                "displayModeBar": True,
                "scrollZoom": True,
                "doubleClick": "reset",
                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                "toImageButtonOptions": {
                    "format": "png",
                    "filename": "annotation_timeline",
                    "scale": 2,
                },
            },
        )
        st.caption(
            "Long filenames are wrapped to preserve plotting space. The camera icon "
            "saves the current interactive view as PNG."
        )

        html_bytes = fig.to_html(include_plotlyjs=True).encode("utf-8")
        png_bytes = render_timeline_png(
            timeline_df,
            annotator_order=timeline_choices,
            visual_extension=visual_extension,
            color_map=color_map,
            reference_annotator=expert,
        )

        download_col1, download_col2 = st.columns(2)
        with download_col1:
            st.download_button(
                "Download interactive HTML",
                data=html_bytes,
                file_name="annotation_timeline.html",
                mime="text/html",
                width="stretch",
            )
        with download_col2:
            st.download_button(
                "Download static PNG",
                data=png_bytes,
                file_name="annotation_timeline.png",
                mime="image/png",
                width="stretch",
            )
    else:
        st.info("Select at least one annotator.")

with tabs[1]:
    st.info(
        "Overlap and F1 scores range from 0 to 1; higher is better. Timing errors "
        "are measured in seconds; lower is better."
    )

    if not comparisons:
        st.info("Select at least one annotator to compare with the expert.")
    else:
        summary_rows = []
        event_tables = {}

        for comparison in comparisons:
            summary, per_event = compare_annotators(
                df=analysis_df,
                reference=expert,
                comparison=comparison,
                segment_iou_threshold=segment_iou_threshold,
            )
            summary_rows.append(summary)
            event_tables[comparison] = per_event

        summary_df = pd.DataFrame(summary_rows)

        st.subheader("Overall comparison")
        display_summary = summary_df.rename(
                    columns={
                        "comparison": "Comparison annotator",
                        "macro_temporal_iou": "Average event overlap",
                        "micro_duration_f1": "Time coverage F1",
                        "segment_f1": "Segment detection F1",
                        "boundary_start_mae_s": "Average start error (s)",
                        "boundary_end_mae_s": "Average end error (s)",
                        "unmatched_reference_segments": "Missed expert segments",
                        "unmatched_comparison_segments": "Extra comparison segments",
                    }
                )
        display_summary = display_summary.drop(
                columns=[
                    "reference",
                    "micro_temporal_iou",
                    "matched_segments",
                ],
                errors="ignore",
            ).round(3)

        st.dataframe(
            display_summary,
            width="stretch",
            hide_index=True,
            column_config={
                "Average event overlap": st.column_config.NumberColumn(
                    help=(
                        "Temporal IoU calculated for each selected event and then "
                        "averaged. Every event type has equal weight."
                    ),
                    format="%.3f",
                ),
                "Time coverage F1": st.column_config.NumberColumn(
                    help=(
                        "Balances missed expert-annotated time and extra time marked "
                        "by the comparison annotator."
                    ),
                    format="%.3f",
                ),
                "Segment detection F1": st.column_config.NumberColumn(
                    help=(
                        "Measures whether individual event occurrences were found. "
                        "The matching strictness is controlled by the sidebar slider."
                    ),
                    format="%.3f",
                ),
                "Average start error (s)": st.column_config.NumberColumn(
                    help="Average absolute difference between matched segment start times.",
                    format="%.2f",
                ),
                "Average end error (s)": st.column_config.NumberColumn(
                    help="Average absolute difference between matched segment end times.",
                    format="%.2f",
                ),
            },
        )

        summary_csv = display_summary.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download summary metrics CSV",
            data=summary_csv,
            file_name="expert_comparison_summary.csv",
            mime="text/csv",
        )

        st.subheader("Results by event")
        selected_comparison = st.selectbox(
            "Comparison annotator",
            comparisons,
            key="per_event_comparison",
        )
        per_event = event_tables[selected_comparison].rename(
            columns={
                "event": "Event",
                "reference_duration_s": "Expert time (s)",
                "comparison_duration_s": "Comparison time (s)",
                "overlap_duration_s": "Shared time (s)",
                "temporal_iou": "Time overlap (IoU)",
                "duration_precision": "Time precision",
                "duration_recall": "Time recall",
                "duration_f1": "Time coverage F1",
                "reference_segments": "Expert segments",
                "comparison_segments": "Comparison segments",
                "matched_segments": "Matched segments",
                "segment_precision": "Segment precision",
                "segment_recall": "Segment recall",
                "segment_f1": "Segment F1",
                "start_mae_s": "Start error (s)",
                "end_mae_s": "End error (s)",
                "duration_mae_s": "Duration error (s)",
            }
        ).round(3)

        per_event = per_event.drop(
                columns=[
                    "Time precision",
                    "Time recall",
                    "Segment precision",
                    "Segment recall",
                    "Matched segments",
                    "Duration error (s)",
                ],
                errors="ignore",
            )

        st.dataframe(
            per_event,
            width="stretch",
            hide_index=True,
            column_config={
                "Time overlap (IoU)": st.column_config.NumberColumn(
                    help=(
                        "Shared annotated time divided by all time covered by either "
                        "annotator for this event."
                    ),
                    format="%.3f",
                ),
                "Segment F1": st.column_config.NumberColumn(
                    help="Combined segment precision and segment recall.",
                    format="%.3f",
                ),
            },
        )

        st.download_button(
            "Download per-event metrics CSV",
            data=per_event.to_csv(index=False).encode("utf-8"),
            file_name=f"{expert}_vs_{selected_comparison}_per_event.csv",
            mime="text/csv",
        )

with tabs[2]:
    st.caption(
        "This matrix uses only the selected events. Each cell is the average "
        "event-level temporal IoU between two annotators."
    )
    matrix = pairwise_iou_matrix(analysis_df, annotators)
    heatmap = px.imshow(
        matrix,
        text_auto=".2f",
        zmin=0,
        zmax=1,
        aspect="auto",
        labels={
            "x": "Annotator",
            "y": "Annotator",
            "color": "Average event overlap",
        },
        title="Pairwise average event overlap",
    )
    heatmap.update_layout(height=max(450, 70 * len(annotators)))
    st.plotly_chart(
        heatmap,
        width="stretch",
        config={"displaylogo": False},
    )

    st.dataframe(matrix.style.format("{:.3f}"), width="stretch")
    st.download_button(
        "Download pairwise matrix CSV",
        data=matrix.to_csv().encode("utf-8"),
        file_name="pairwise_temporal_iou.csv",
        mime="text/csv",
    )

with tabs[3]:
    show_only_selected = st.checkbox(
        "Show only events selected for analysis",
        value=True,
    )
    display_df = analysis_df.copy() if show_only_selected else df.copy()
    display_df["start_time"] = display_df["start"].map(
        lambda x: (
            f"{int(x // 3600):02d}:"
            f"{int((x % 3600) // 60):02d}:"
            f"{int(x % 60):02d}"
        )
    )
    display_df["end_time"] = display_df["end"].map(
        lambda x: (
            f"{int(x // 3600):02d}:"
            f"{int((x % 3600) // 60):02d}:"
            f"{int(x % 60):02d}"
        )
    )
    st.dataframe(display_df, width="stretch", hide_index=True)
    st.download_button(
        "Download parsed annotations CSV",
        data=display_df.to_csv(index=False).encode("utf-8"),
        file_name="parsed_annotations.csv",
        mime="text/csv",
    )

with tabs[4]:
    st.markdown(
        """
### How to interpret the scores

#### Time overlap (temporal IoU)

This defines: **How much of the annotated time is shared by both annotators?**

It is calculated separately for each event:

```text
shared time / total time marked by either annotator
```

A score of `1.00` means the two intervals agree exactly. A score of `0.00`
means they do not overlap.

#### Time coverage F1

Measures how closely the total event timing agrees between the two annotators.

It penalizes both:

- expert-annotated time that was missed
- extra time marked by the comparison annotator

A score of `1.00` means complete agreement.

#### Segment F1

Measures whether the individual occurrences of an event were identified.

Two segments count as a match only when they have the same event name and their
temporal IoU reaches the threshold selected in the sidebar.

Missed expert segments and extra comparison segments both reduce this score.


#### Start-time and end-time error

For matched segments, these are the average absolute timing differences in
seconds. Smaller values are better. They do not indicate whether the comparison
annotator was early or late; they measure only error magnitude.

### What the sliders do

- **Extra display width:** changes only how wide bars look. It never changes
  annotations or scores.
- **Required segment overlap:** controls how strict segment matching is. A higher
  value requires more precise start/end agreement before two segments count as
  the same occurrence.
        """
    )
