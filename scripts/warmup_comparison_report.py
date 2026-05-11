"""Generate warmup comparison table for the report.

Loads results from data/section1_1/assignment_warmup_w{N}_full/ for N in {0,1,2,5},
builds a pivot table of mean timing (ms) vs warmup_steps, and writes:
  - data/section1_1/analysis/warmup_comparison/warmup_comparison.csv
  - data/section1_1/analysis/warmup_comparison/warmup_comparison.md
  - stdout: a LaTeX table for pasting into the report
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
import statistics

WARMUP_VARIANTS = [0, 1, 2, 5]
DATA_ROOT = Path("data/section1_1")
ANALYSIS_DIR = DATA_ROOT / "analysis" / "warmup_comparison"
CONTEXT_LENGTHS = [128, 256, 512, 1024]
MODE_LABELS = {"forward": "Forward", "forward_backward": "Fwd+Bwd"}


def load_all() -> list[dict]:
    rows = []
    for warmup in WARMUP_VARIANTS:
        campaign_dir = DATA_ROOT / f"assignment_warmup_w{warmup}_full"
        for result_path in campaign_dir.rglob("result.json"):
            data = json.loads(result_path.read_text())
            cfg = data["config"]
            timings = data["timings_seconds"]
            rows.append({
                "warmup_steps": cfg["warmup_steps"],
                "context_length": cfg["context_length"],
                "mode": cfg["mode"],
                "mean_ms": data["mean_seconds"] * 1000,
                # Steady-state mean: drop first timing (cold-start spike)
                "steady_ms": statistics.mean(timings[1:]) * 1000 if len(timings) > 1 else data["mean_seconds"] * 1000,
                "stdev_ms": data["stdev_seconds"] * 1000,
            })
    return rows


def build_pivot(rows: list[dict], value_key: str) -> dict:
    """Return nested dict: pivot[mode][ctx][warmup] = value."""
    pivot: dict = {}
    for row in rows:
        mode = row["mode"]
        ctx = row["context_length"]
        warmup = row["warmup_steps"]
        pivot.setdefault(mode, {}).setdefault(ctx, {})[warmup] = row[value_key]
    return pivot


def write_csv(rows: list[dict], out_path: Path) -> None:
    import csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["mode"], r["context_length"], r["warmup_steps"])))


def make_latex_table(pivot_mean: dict, pivot_steady: dict) -> str:
    """Build a LaTeX table: rows=(mode,ctx), cols=warmup 0,1,2,5."""
    header = (
        r"\begin{table}[h!]" "\n"
        r"\centering" "\n"
        r"\caption{Effect of warm-up steps on mean timing (ms) for the small model, FP32, batch size 1. "
        r"``Steady'' shows the mean of steps 2--10 only (excluding the cold-start first step).}" "\n"
        r"\label{tab:warmup_comparison}" "\n"
        r"\begin{tabular}{ll|rrrr|rrrr}" "\n"
        r"\hline" "\n"
        r"& & \multicolumn{4}{c|}{All-steps mean (ms)} & \multicolumn{4}{c}{Steady-state mean (ms, steps 2--10)} \\" "\n"
        r"Mode & Ctx & $w{=}0$ & $w{=}1$ & $w{=}2$ & $w{=}5$ & $w{=}0$ & $w{=}1$ & $w{=}2$ & $w{=}5$ \\" "\n"
        r"\hline" "\n"
    )
    body_lines = []
    for mode in ["forward", "forward_backward"]:
        label = MODE_LABELS[mode]
        for ctx in CONTEXT_LENGTHS:
            mean_vals = [pivot_mean.get(mode, {}).get(ctx, {}).get(w, float("nan")) for w in WARMUP_VARIANTS]
            steady_vals = [pivot_steady.get(mode, {}).get(ctx, {}).get(w, float("nan")) for w in WARMUP_VARIANTS]
            mean_str = " & ".join(f"{v:.1f}" for v in mean_vals)
            steady_str = " & ".join(f"{v:.1f}" for v in steady_vals)
            body_lines.append(f"\t{label} & {ctx} & {mean_str} & {steady_str} \\\\")
        body_lines.append(r"\hline")
    footer = (
        r"\end{tabular}" "\n"
        r"\end{table}"
    )
    return header + "\n".join(body_lines) + "\n" + footer


def main() -> None:
    rows = load_all()
    if not rows:
        print("ERROR: no result.json files found. Run the campaign first.", file=sys.stderr)
        sys.exit(1)

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(rows, ANALYSIS_DIR / "warmup_comparison.csv")

    pivot_mean = build_pivot(rows, "mean_ms")
    pivot_steady = build_pivot(rows, "steady_ms")

    # Write markdown table
    md_lines = ["| Mode | Ctx | w=0 (all) | w=1 (all) | w=2 (all) | w=5 (all) | w=0 (steady) | w=5 (steady) |",
                "|------|-----|-----------|-----------|-----------|-----------|--------------|--------------|"]
    for mode in ["forward", "forward_backward"]:
        label = MODE_LABELS[mode]
        for ctx in CONTEXT_LENGTHS:
            m = pivot_mean.get(mode, {}).get(ctx, {})
            s = pivot_steady.get(mode, {}).get(ctx, {})
            md_lines.append(
                f"| {label} | {ctx} | "
                f"{m.get(0, float('nan')):.1f} | {m.get(1, float('nan')):.1f} | "
                f"{m.get(2, float('nan')):.1f} | {m.get(5, float('nan')):.1f} | "
                f"{s.get(0, float('nan')):.1f} | {s.get(5, float('nan')):.1f} |"
            )
    (ANALYSIS_DIR / "warmup_comparison.md").write_text("\n".join(md_lines) + "\n")

    latex = make_latex_table(pivot_mean, pivot_steady)
    (ANALYSIS_DIR / "warmup_comparison_table.tex").write_text(latex + "\n")
    print(latex)
    print(f"\n\nOutputs written to {ANALYSIS_DIR}/", file=sys.stderr)


if __name__ == "__main__":
    main()
