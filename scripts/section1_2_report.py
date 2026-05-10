from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.report_utils import dataframe_to_markdown, format_numeric, load_results, write_report



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
    write_report(summary, analysis_dir, "timing_summary")
    write_report(manifest, analysis_dir, "artifact_manifest")


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
    print(dataframe_to_markdown(format_numeric(summary)))


if __name__ == "__main__":
    main()