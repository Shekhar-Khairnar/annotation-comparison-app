from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd


LINE_PATTERN = re.compile(
    r"^(.+?)\s*:\s*[\(\（]\s*"
    r"(\d{1,2}:\d{2}:\d{2})\s*,\s*"
    r"(\d{1,2}:\d{2}:\d{2})\s*[\)\）]\s*$"
)


def time_to_seconds(value: str) -> int:
    hours, minutes, seconds = map(int, value.split(":"))
    if minutes >= 60 or seconds >= 60:
        raise ValueError(f"Invalid time value: {value}")
    return hours * 3600 + minutes * 60 + seconds


def _unique_annotator_name(base_name: str, used_names: set[str]) -> str:
    candidate = base_name
    counter = 2
    while candidate in used_names:
        candidate = f"{base_name} ({counter})"
        counter += 1
    used_names.add(candidate)
    return candidate


def parse_uploaded_files(uploaded_files: Iterable) -> tuple[pd.DataFrame, list[dict]]:
    rows: list[dict] = []
    diagnostics: list[dict] = []
    used_names: set[str] = set()

    for uploaded_file in uploaded_files:
        filename = getattr(uploaded_file, "name", "annotation.txt")
        stem = Path(filename).stem.strip() or "annotator"
        annotator = _unique_annotator_name(stem, used_names)

        raw = uploaded_file.getvalue()
        text = raw.decode("utf-8", errors="ignore")

        valid_lines = 0
        skipped_lines = 0
        messages: list[str] = []

        for line_number, raw_line in enumerate(text.splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue

            match = LINE_PATTERN.match(line)
            if not match:
                skipped_lines += 1
                messages.append(f"Line {line_number}: unsupported format")
                continue

            event, start_text, end_text = match.groups()

            try:
                start = time_to_seconds(start_text)
                end = time_to_seconds(end_text)
            except ValueError as exc:
                skipped_lines += 1
                messages.append(f"Line {line_number}: {exc}")
                continue

            if end < start:
                skipped_lines += 1
                messages.append(f"Line {line_number}: end time is before start time")
                continue

            rows.append(
                {
                    "annotator": annotator,
                    "source_file": filename,
                    "event": event.strip(),
                    "start": float(start),
                    "end": float(end),
                    "duration": float(end - start),
                }
            )
            valid_lines += 1

        diagnostics.append(
            {
                "file": filename,
                "annotator": annotator,
                "valid_lines": valid_lines,
                "skipped_lines": skipped_lines,
                "details": "; ".join(messages[:8])
                + ("; ..." if len(messages) > 8 else ""),
            }
        )

    columns = [
        "annotator",
        "source_file",
        "event",
        "start",
        "end",
        "duration",
    ]
    return pd.DataFrame(rows, columns=columns), diagnostics
