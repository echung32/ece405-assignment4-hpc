from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.report_utils import dataframe_to_markdown, format_numeric, load_results, write_report


def write_outputs(frame: pd.DataFrame, analysis_dir: Path) -> None:
    write_report(frame, analysis_dir)


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