from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def flatten_result(payload: dict[str, object], result_path: Path) -> dict[str, object]:
    config = payload.get("config", {})
    model_spec = payload.get("model_spec", {})
    row = {
        "result_path": str(result_path),
        "run_dir": str(result_path.parent),
    }
    for prefix, values in (("config", config), ("model", model_spec)):
        for key, value in values.items():
            row[f"{prefix}_{key}"] = value
    for key, value in payload.items():
        if key in {"config", "model_spec", "timings_seconds"}:
            continue
        row[key] = value
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


def format_numeric_columns(frame: pd.DataFrame) -> pd.DataFrame:
    formatted = frame.copy()
    for column in ("mean_seconds", "stdev_seconds", "peak_reserved_gb", "peak_device_used_gb"):
        if column in formatted.columns:
            formatted[column] = formatted[column].map(lambda value: f"{value:.6f}" if pd.notna(value) else "")
    return formatted


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


def build_summary_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_columns = [
        "config_model_size",
        "config_context_length",
        "config_batch_size",
        "config_mode",
        "config_precision",
        "mean_seconds",
        "stdev_seconds",
        "peak_reserved_gb",
        "peak_device_used_gb",
        "snapshot_path",
    ]
    available_summary_columns = [column for column in summary_columns if column in frame.columns]
    summary = frame.loc[:, available_summary_columns].sort_values(
        by=[
            "config_model_size",
            "config_context_length",
            "config_batch_size",
            "config_mode",
            "config_precision",
        ]
    )

    manifest_columns = [
        "run_dir",
        "result_path",
        "config_campaign",
        "config_model_size",
        "config_mode",
        "config_precision",
        "snapshot_path",
    ]
    available_manifest_columns = [column for column in manifest_columns if column in frame.columns]
    manifest = frame.loc[:, available_manifest_columns].sort_values(by=["run_dir"])
    return summary, manifest


def write_outputs(summary: pd.DataFrame, manifest: pd.DataFrame, analysis_dir: Path) -> None:
    analysis_dir.mkdir(parents=True, exist_ok=True)

    summary.to_csv(analysis_dir / "timing_summary.csv", index=False)
    manifest.to_csv(analysis_dir / "artifact_manifest.csv", index=False)

    formatted_summary = format_numeric_columns(summary)
    formatted_manifest = manifest.fillna("")

    (analysis_dir / "timing_summary.md").write_text(dataframe_to_markdown(formatted_summary) + "\n")
    (analysis_dir / "artifact_manifest.md").write_text(dataframe_to_markdown(formatted_manifest) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Section 1.1 run artifacts into analysis tables")
    parser.add_argument("--input-root", default="data/section1_1")
    parser.add_argument("--campaign", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    rows = load_results(input_root, args.campaign)
    if not rows:
        raise FileNotFoundError(f"No Section 1.1 result files found under {input_root}")

    frame = pd.DataFrame(rows)
    summary, manifest = build_summary_tables(frame)

    campaign_name = args.campaign if args.campaign is not None else "all"
    analysis_dir = input_root / "analysis" / campaign_name
    write_outputs(summary, manifest, analysis_dir)

    print(dataframe_to_markdown(format_numeric_columns(summary)))


if __name__ == "__main__":
    main()