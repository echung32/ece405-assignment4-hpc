from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# Tokens used to identify numeric columns for formatting.
# Union of all patterns used across sections.
_NUMERIC_TOKENS: tuple[str, ...] = ("seconds", "_ms", "_gb", "_mb", "_bytes")


def _flatten(prefix: str, value: object, row: dict[str, object]) -> None:
    """Recursively flatten nested dicts into dot-separated column names."""
    if isinstance(value, dict):
        for key, nested_value in value.items():
            nested_prefix = f"{prefix}_{key}" if prefix else key
            _flatten(nested_prefix, nested_value, row)
        return
    if isinstance(value, list):
        return
    row[prefix] = value


def load_results(input_root: Path, campaign: str | None) -> list[dict[str, object]]:
    """Find all result.json files under input_root (optionally scoped to campaign) and flatten them."""
    if campaign is None:
        result_paths = sorted(input_root.glob("*/**/result.json"))
    else:
        result_paths = sorted((input_root / campaign).glob("*/result.json"))

    rows: list[dict[str, object]] = []
    for result_path in result_paths:
        payload = json.loads(result_path.read_text())
        row: dict[str, object] = {
            "result_path": str(result_path),
            "run_dir": str(result_path.parent),
        }
        for key, value in payload.items():
            _flatten(key, value, row)
        rows.append(row)
    return rows


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table."""
    if frame.empty:
        return ""

    headers = [str(column) for column in frame.columns]
    rows = [
        ["" if pd.isna(value) else str(value) for value in row]
        for row in frame.itertuples(index=False, name=None)
    ]
    separator = ["---"] * len(headers)

    def format_row(values: list[str]) -> str:
        return "| " + " | ".join(values) + " |"

    lines = [format_row(headers), format_row(separator)]
    lines.extend(format_row(row) for row in rows)
    return "\n".join(lines)


def format_numeric(
    frame: pd.DataFrame,
    tokens: tuple[str, ...] = _NUMERIC_TOKENS,
) -> pd.DataFrame:
    """Return a copy of frame with float columns (identified by token substrings) formatted to 6 decimal places."""
    formatted = frame.copy()
    for column in formatted.columns:
        if any(token in column for token in tokens):
            formatted[column] = formatted[column].map(
                lambda value: f"{value:.6f}" if pd.notna(value) else ""
            )
    return formatted


def write_report(
    frame: pd.DataFrame,
    analysis_dir: Path,
    stem: str = "timing_summary",
    tokens: tuple[str, ...] = _NUMERIC_TOKENS,
) -> None:
    """Write frame to <analysis_dir>/<stem>.csv and <analysis_dir>/<stem>.md."""
    analysis_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(analysis_dir / f"{stem}.csv", index=False)
    (analysis_dir / f"{stem}.md").write_text(
        dataframe_to_markdown(format_numeric(frame, tokens)) + "\n"
    )
