from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.report_utils import dataframe_to_markdown, format_numeric, write_report


def flatten_result(payload: dict[str, object], result_path: Path) -> dict[str, object]:
    """Flatten a Section 1.1 result.json payload into a flat dict.

    Maps ``config.*`` → ``config_*`` and ``model_spec.*`` → ``model_*`` (one level),
    and promotes remaining top-level scalar keys directly.
    """
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
    write_report(summary, analysis_dir, "timing_summary")
    write_report(manifest, analysis_dir, "artifact_manifest")


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

    print(dataframe_to_markdown(format_numeric(summary)))


if __name__ == "__main__":
    main()