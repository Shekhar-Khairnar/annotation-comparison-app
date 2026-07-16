from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


Interval = tuple[float, float]


def merge_intervals(intervals: Iterable[Interval]) -> list[Interval]:
    valid = sorted((float(s), float(e)) for s, e in intervals if e > s)
    if not valid:
        return []

    merged = [valid[0]]
    for start, end in valid[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def total_duration(intervals: Iterable[Interval]) -> float:
    return float(sum(end - start for start, end in merge_intervals(intervals)))


def intersection_duration(
    first: Iterable[Interval],
    second: Iterable[Interval],
) -> float:
    a = merge_intervals(first)
    b = merge_intervals(second)
    i = 0
    j = 0
    overlap = 0.0

    while i < len(a) and j < len(b):
        start = max(a[i][0], b[j][0])
        end = min(a[i][1], b[j][1])
        overlap += max(0.0, end - start)

        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1

    return float(overlap)


def interval_iou(first: Interval, second: Interval) -> float:
    overlap = max(0.0, min(first[1], second[1]) - max(first[0], second[0]))
    union = (first[1] - first[0]) + (second[1] - second[0]) - overlap
    return overlap / union if union > 0 else 0.0


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def _f1(precision: float, recall: float) -> float:
    return (
        float(2 * precision * recall / (precision + recall))
        if precision + recall > 0
        else 0.0
    )


def _segments_for_event(
    df: pd.DataFrame,
    annotator: str,
    event: str,
) -> list[Interval]:
    subset = df[
        (df["annotator"] == annotator)
        & (df["event"] == event)
        & (df["end"] > df["start"])
    ]
    return list(zip(subset["start"].astype(float), subset["end"].astype(float)))


def match_segments(
    reference_segments: list[Interval],
    comparison_segments: list[Interval],
    iou_threshold: float,
) -> list[dict]:
    if not reference_segments or not comparison_segments:
        return []

    iou_matrix = np.zeros(
        (len(reference_segments), len(comparison_segments)),
        dtype=float,
    )

    for i, reference_segment in enumerate(reference_segments):
        for j, comparison_segment in enumerate(comparison_segments):
            iou_matrix[i, j] = interval_iou(reference_segment, comparison_segment)

    row_indices, column_indices = linear_sum_assignment(1.0 - iou_matrix)

    matches = []
    for row_index, column_index in zip(row_indices, column_indices):
        iou = float(iou_matrix[row_index, column_index])
        if iou < iou_threshold:
            continue

        reference_segment = reference_segments[row_index]
        comparison_segment = comparison_segments[column_index]

        matches.append(
            {
                "reference_index": int(row_index),
                "comparison_index": int(column_index),
                "iou": iou,
                "start_error_s": abs(
                    reference_segment[0] - comparison_segment[0]
                ),
                "end_error_s": abs(
                    reference_segment[1] - comparison_segment[1]
                ),
                "duration_error_s": abs(
                    (reference_segment[1] - reference_segment[0])
                    - (comparison_segment[1] - comparison_segment[0])
                ),
            }
        )

    return matches


def compare_annotators(
    df: pd.DataFrame,
    reference: str,
    comparison: str,
    segment_iou_threshold: float = 0.5,
) -> tuple[dict, pd.DataFrame]:
    relevant = df[df["annotator"].isin([reference, comparison])]
    events = sorted(relevant["event"].dropna().unique())

    per_event_rows = []
    all_start_errors: list[float] = []
    all_end_errors: list[float] = []

    for event in events:
        reference_segments = _segments_for_event(df, reference, event)
        comparison_segments = _segments_for_event(df, comparison, event)

        reference_duration = total_duration(reference_segments)
        comparison_duration = total_duration(comparison_segments)
        overlap_duration = intersection_duration(
            reference_segments,
            comparison_segments,
        )
        union_duration = (
            reference_duration + comparison_duration - overlap_duration
        )

        temporal_iou = _safe_ratio(overlap_duration, union_duration)
        duration_precision = _safe_ratio(
            overlap_duration,
            comparison_duration,
        )
        duration_recall = _safe_ratio(
            overlap_duration,
            reference_duration,
        )
        duration_f1 = _f1(duration_precision, duration_recall)

        matches = match_segments(
            reference_segments,
            comparison_segments,
            iou_threshold=segment_iou_threshold,
        )
        matched_count = len(matches)
        comparison_count = len(comparison_segments)
        reference_count = len(reference_segments)

        segment_precision = _safe_ratio(matched_count, comparison_count)
        segment_recall = _safe_ratio(matched_count, reference_count)
        segment_f1 = _f1(segment_precision, segment_recall)

        start_errors = [match["start_error_s"] for match in matches]
        end_errors = [match["end_error_s"] for match in matches]
        duration_errors = [match["duration_error_s"] for match in matches]

        all_start_errors.extend(start_errors)
        all_end_errors.extend(end_errors)

        per_event_rows.append(
            {
                "event": event,
                "reference_duration_s": reference_duration,
                "comparison_duration_s": comparison_duration,
                "overlap_duration_s": overlap_duration,
                "temporal_iou": temporal_iou,
                "duration_precision": duration_precision,
                "duration_recall": duration_recall,
                "duration_f1": duration_f1,
                "reference_segments": reference_count,
                "comparison_segments": comparison_count,
                "matched_segments": matched_count,
                "segment_precision": segment_precision,
                "segment_recall": segment_recall,
                "segment_f1": segment_f1,
                "start_mae_s": (
                    float(np.mean(start_errors)) if start_errors else np.nan
                ),
                "end_mae_s": (
                    float(np.mean(end_errors)) if end_errors else np.nan
                ),
                "duration_mae_s": (
                    float(np.mean(duration_errors))
                    if duration_errors
                    else np.nan
                ),
            }
        )

    per_event = pd.DataFrame(per_event_rows)

    if per_event.empty:
        summary = {
            "reference": reference,
            "comparison": comparison,
            "macro_temporal_iou": 0.0,
            "micro_temporal_iou": 0.0,
            "micro_duration_f1": 0.0,
            "segment_f1": 0.0,
            "boundary_start_mae_s": np.nan,
            "boundary_end_mae_s": np.nan,
            "matched_segments": 0,
            "unmatched_reference_segments": 0,
            "unmatched_comparison_segments": 0,
        }
        return summary, per_event

    total_reference_duration = float(per_event["reference_duration_s"].sum())
    total_comparison_duration = float(
        per_event["comparison_duration_s"].sum()
    )
    total_overlap_duration = float(per_event["overlap_duration_s"].sum())
    total_union_duration = (
        total_reference_duration
        + total_comparison_duration
        - total_overlap_duration
    )

    micro_precision = _safe_ratio(
        total_overlap_duration,
        total_comparison_duration,
    )
    micro_recall = _safe_ratio(
        total_overlap_duration,
        total_reference_duration,
    )

    matched_segments = int(per_event["matched_segments"].sum())
    total_reference_segments = int(per_event["reference_segments"].sum())
    total_comparison_segments = int(per_event["comparison_segments"].sum())

    global_segment_precision = _safe_ratio(
        matched_segments,
        total_comparison_segments,
    )
    global_segment_recall = _safe_ratio(
        matched_segments,
        total_reference_segments,
    )

    summary = {
        "reference": reference,
        "comparison": comparison,
        "macro_temporal_iou": float(per_event["temporal_iou"].mean()),
        "micro_temporal_iou": _safe_ratio(
            total_overlap_duration,
            total_union_duration,
        ),
        "micro_duration_f1": _f1(micro_precision, micro_recall),
        "segment_f1": _f1(
            global_segment_precision,
            global_segment_recall,
        ),
        "boundary_start_mae_s": (
            float(np.mean(all_start_errors))
            if all_start_errors
            else np.nan
        ),
        "boundary_end_mae_s": (
            float(np.mean(all_end_errors))
            if all_end_errors
            else np.nan
        ),
        "matched_segments": matched_segments,
        "unmatched_reference_segments": (
            total_reference_segments - matched_segments
        ),
        "unmatched_comparison_segments": (
            total_comparison_segments - matched_segments
        ),
    }

    return summary, per_event


def pairwise_iou_matrix(
    df: pd.DataFrame,
    annotators: list[str],
) -> pd.DataFrame:
    matrix = pd.DataFrame(
        np.eye(len(annotators), dtype=float),
        index=annotators,
        columns=annotators,
    )

    for i, first in enumerate(annotators):
        for j in range(i + 1, len(annotators)):
            second = annotators[j]
            summary, _ = compare_annotators(
                df,
                reference=first,
                comparison=second,
                segment_iou_threshold=0.5,
            )
            value = summary["macro_temporal_iou"]
            matrix.loc[first, second] = value
            matrix.loc[second, first] = value

    return matrix
