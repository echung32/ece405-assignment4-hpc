from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _flatten(prefix: str, value: object, row: dict[str, object]) -> None:
    if isinstance(value, dict):
        for key, nested_value in value.items():
            nested_prefix = f"{prefix}_{key}" if prefix else key
            _flatten(nested_prefix, nested_value, row)
        return
    if isinstance(value, list):
        return
    row[prefix] = value


def load_results(input_root: Path, campaign: str | None) -> list[dict[str, object]]:
    result_paths = sorted((input_root / campaign).glob("*/result.json")) if campaign else sorted(input_root.glob("*/**/result.json"))
    rows: list[dict[str, object]] = []
    for result_path in result_paths:
        payload = json.loads(result_path.read_text())
        row = {"result_path": str(result_path), "run_dir": str(result_path.parent)}
        for key, value in payload.items():
            _flatten(key, value, row)
        rows.append(row)
    return rows


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    headers = [str(column) for column in frame.columns]
    rows = [["" if pd.isna(value) else str(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    separator = ["---"] * len(headers)
    format_row = lambda values: "| " + " | ".join(values) + " |"
    return "\n".join([format_row(headers), format_row(separator), *(format_row(row) for row in rows)])


def format_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    formatted = frame.copy()
    for column in formatted.columns:
        if any(token in column for token in ("seconds", "_gb", "_mb")):
            formatted[column] = formatted[column].map(lambda value: f"{value:.6f}" if pd.notna(value) else "")
    return formatted


def write_outputs(frame: pd.DataFrame, analysis_dir: Path) -> None:
    analysis_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(analysis_dir / "timing_summary.csv", index=False)
    (analysis_dir / "timing_summary.md").write_text(dataframe_to_markdown(format_numeric(frame)) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Section 2 benchmark results")
    parser.add_argument("--input-root", default="data/section2")
    parser.add_argument("--campaign", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    rows = load_results(input_root, args.campaign)
    if not rows:
        raise FileNotFoundError(f"No Section 2 result files found under {input_root}")
    frame = pd.DataFrame(rows)
    campaign_name = args.campaign if args.campaign else "all"
    analysis_dir = input_root / "analysis" / campaign_name
    write_outputs(frame, analysis_dir)
    print(dataframe_to_markdown(format_numeric(frame)))


if __name__ == "__main__":
    main()