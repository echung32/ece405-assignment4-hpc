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


def flatten_result(payload: dict[str, object], result_path: Path) -> dict[str, object]:
    row: dict[str, object] = {
        "result_path": str(result_path),
        "run_dir": str(result_path.parent),
    }
    for key, value in payload.items():
        _flatten(key, value, row)
    return row


def load_results(input_root: Path, campaign: str | None) -> list[dict[str, object]]:
    if campaign is None:
        result_paths = sorted(input_root.glob("*/**/result.json"))
    else:
        result_paths = sorted((input_root / campaign).glob("*/result.json"))

    rows: list[dict[str, object]] = []
    for result_path in result_paths:
        payload = json.loads(result_path.read_text())
        rows.append(flatten_result(payload, result_path))
    return rows


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""

    headers = [str(column) for column in frame.columns]
    rows = [["" if pd.isna(value) else str(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    separator = ["---"] * len(headers)

    def format_row(values: list[str]) -> str:
        return "| " + " | ".join(values) + " |"

    lines = [format_row(headers), format_row(separator)]
    lines.extend(format_row(row) for row in rows)
    return "\n".join(lines)


def format_numeric_columns(frame: pd.DataFrame) -> pd.DataFrame:
    formatted = frame.copy()
    for column in formatted.columns:
        if any(token in column for token in ("seconds", "_ms", "_gb")):
            formatted[column] = formatted[column].map(lambda value: f"{value:.6f}" if pd.notna(value) else "")
    return formatted


def build_summary_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    preferred_columns = [
        "config_implementation",
        "config_dtype",
        "config_sequence_length",
        "config_d_model",
        "config_model_size",
        "config_mode",
        "status",
        "forward_mean_seconds",
        "backward_mean_seconds",
        "forward_ms",
        "backward_ms",
        "end_to_end_ms",
        "mean_seconds",
        "stdev_seconds",
    ]
    summary_columns = [column for column in preferred_columns if column in frame.columns]
    if not summary_columns:
        summary_columns = [column for column in frame.columns if column not in {"result_path", "run_dir"}]
    sort_columns = [
        column
        for column in (
            "config_implementation",
            "config_model_size",
            "config_mode",
            "config_dtype",
            "config_d_model",
            "config_sequence_length",
        )
        if column in frame.columns
    ]
    summary = frame.loc[:, summary_columns].sort_values(by=sort_columns) if sort_columns else frame.loc[:, summary_columns]

    manifest_columns = [column for column in ("run_dir", "result_path", "config_campaign", "status") if column in frame.columns]
    manifest = frame.loc[:, manifest_columns].sort_values(by=["run_dir"]) if manifest_columns else frame[["run_dir", "result_path"]]
    return summary, manifest


def write_outputs(summary: pd.DataFrame, manifest: pd.DataFrame, analysis_dir: Path) -> None:
    analysis_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(analysis_dir / "timing_summary.csv", index=False)
    manifest.to_csv(analysis_dir / "artifact_manifest.csv", index=False)
    (analysis_dir / "timing_summary.md").write_text(dataframe_to_markdown(format_numeric_columns(summary)) + "\n")
    (analysis_dir / "artifact_manifest.md").write_text(dataframe_to_markdown(manifest.fillna("")) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Section 1.2/1.3 benchmark artifacts into analysis tables")
    parser.add_argument("--input-root", default="data/section1_2")
    parser.add_argument("--campaign", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    rows = load_results(input_root, args.campaign)
    if not rows:
        raise FileNotFoundError(f"No result files found under {input_root}")

    frame = pd.DataFrame(rows)
    summary, manifest = build_summary_tables(frame)
    campaign_name = args.campaign if args.campaign is not None else "all"
    analysis_dir = input_root / "analysis" / campaign_name
    write_outputs(summary, manifest, analysis_dir)
    print(dataframe_to_markdown(format_numeric_columns(summary)))


if __name__ == "__main__":
    main()