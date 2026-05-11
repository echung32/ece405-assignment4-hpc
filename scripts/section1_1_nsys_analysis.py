"""Analyse nsys SQLite profiles to extract CUDA kernel statistics.

Usage:
    uv run python -m scripts.section1_1_nsys_analysis [--campaign CAMPAIGN]

Reads profile.sqlite files produced by the nsys campaign, filters to the
'measurement' NVTX scope (excluding warmup), and reports top kernels and
GEMM fractions for the forward and train_step modes.

Outputs:
    data/section1_1/analysis/<campaign>/nsys_kernel_summary.json
    data/section1_1/analysis/<campaign>/nsys_kernel_summary.md
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


GEMM_PATTERNS = ("cutlass", "cublas", "gemm", "bmm", "sgemm", "hgemm", "dgemm")
SCOPE_EXCLUDE = ("forward", "train_step", "forward_backward", "backward",
                 "loss", "optimizer_step", "backprop_step", "forward_step",
                 "warmup", "measurement")


def is_gemm(name: str) -> bool:
    low = name.lower()
    return any(p in low for p in GEMM_PATTERNS)


def is_scope_label(name: str) -> bool:
    """Return True if this is a Python-scope annotation rather than a real kernel."""
    return name in SCOPE_EXCLUDE


def get_measurement_range_ns(db: sqlite3.Connection) -> tuple[int, int] | None:
    """Return (start_ns, end_ns) for the 'measurement' NVTX range, or None.

    nsys 2026.x stores NVTX text directly in the `text` column (textId is NULL).
    Older versions stored it via StringIds joined on textId.
    """
    # Try direct text column first (nsys 2026+)
    rows = db.execute("""
        SELECT start, end FROM NVTX_EVENTS
        WHERE text = 'measurement'
        ORDER BY start LIMIT 1
    """).fetchall()
    if rows:
        return int(rows[0][0]), int(rows[0][1])
    # Fallback: join via StringIds (older nsys)
    rows = db.execute("""
        SELECT e.start, e.end
        FROM NVTX_EVENTS e
        JOIN StringIds s ON e.textId = s.id
        WHERE s.value = 'measurement'
        ORDER BY e.start LIMIT 1
    """).fetchall()
    if not rows:
        return None
    return int(rows[0][0]), int(rows[0][1])


def get_nvtx_scope_range_ns(
    db: sqlite3.Connection, scope_name: str, after_ns: int, before_ns: int
) -> tuple[int, int] | None:
    """Return the first NVTX range with this name within [after_ns, before_ns]."""
    # Try direct text column first (nsys 2026+)
    rows = db.execute("""
        SELECT start, end FROM NVTX_EVENTS
        WHERE text = ? AND start >= ? AND end <= ?
        ORDER BY start LIMIT 1
    """, (scope_name, after_ns, before_ns)).fetchall()
    if rows:
        return int(rows[0][0]), int(rows[0][1])
    # Fallback: join via StringIds
    rows = db.execute("""
        SELECT e.start, e.end
        FROM NVTX_EVENTS e
        JOIN StringIds s ON e.textId = s.id
        WHERE s.value = ? AND e.start >= ? AND e.end <= ?
        ORDER BY e.start LIMIT 1
    """, (scope_name, after_ns, before_ns)).fetchall()
    if not rows:
        return None
    return int(rows[0][0]), int(rows[0][1])


def query_kernels(
    db: sqlite3.Connection,
    start_ns: int,
    end_ns: int,
    top_n: int = 20,
) -> tuple[list[dict], float]:
    """Query top kernels within [start_ns, end_ns].

    Returns (kernel_list, total_gpu_ms) where total_gpu_ms is the total GPU
    time of all kernels in the window.
    """
    rows = db.execute("""
        SELECT sd.value as name,
               COUNT(*) as cnt,
               SUM(k.end - k.start) / 1e6 as total_ms
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds sd ON k.demangledName = sd.id
        WHERE k.start >= ? AND k.end <= ?
        GROUP BY k.demangledName
        ORDER BY total_ms DESC
    """, (start_ns, end_ns)).fetchall()

    total_ms = sum(r[2] for r in rows)
    if total_ms == 0.0:
        return [], 0.0

    result = []
    for row in rows[:top_n]:
        name, cnt, ms = row[0], row[1], row[2]
        result.append({
            "name": name,
            "count": cnt,
            "total_ms": round(ms, 4),
            "pct": round(ms / total_ms * 100, 2),
            "is_gemm": is_gemm(name),
        })
    return result, round(total_ms, 4)


def analyse_sqlite(path: Path, model_size: str, context_length: int, mode: str) -> dict | None:
    """Analyse a single profile.sqlite file."""
    if not path.exists():
        return None

    db = sqlite3.connect(str(path))

    # Check kernel table populated
    cnt = db.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
    if cnt == 0:
        db.close()
        return {"error": "empty kernel table", "model_size": model_size,
                "context_length": context_length, "mode": mode}

    # Get measurement NVTX scope to filter out warmup
    meas_range = get_measurement_range_ns(db)
    if meas_range is None:
        # Fall back to full trace
        all_rows = db.execute(
            "SELECT MIN(start), MAX(end) FROM CUPTI_ACTIVITY_KIND_KERNEL"
        ).fetchone()
        start_ns, end_ns = int(all_rows[0]), int(all_rows[1])
    else:
        start_ns, end_ns = meas_range

    kernels, total_ms = query_kernels(db, start_ns, end_ns)
    gemm_ms = sum(k["total_ms"] for k in kernels if k["is_gemm"])

    db.close()
    return {
        "model_size": model_size,
        "context_length": context_length,
        "mode": mode,
        "total_gpu_ms": total_ms,
        "gemm_ms": round(gemm_ms, 4),
        "gemm_pct": round(gemm_ms / total_ms * 100, 2) if total_ms > 0 else 0.0,
        "top_kernels": kernels,
        "filtered_to_measurement": meas_range is not None,
        "measurement_wall_ms": round((meas_range[1] - meas_range[0]) / 1e6, 2) if meas_range else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", default="assignment_nsys_full")
    parser.add_argument("--data-root", default="data/section1_1")
    parser.add_argument("--top-n", type=int, default=15)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    campaign_dir = data_root / args.campaign
    analysis_dir = data_root / "analysis" / args.campaign
    analysis_dir.mkdir(parents=True, exist_ok=True)

    model_sizes = ["small", "medium", "large", "xl", "2.7b"]
    context_lengths = [128, 256, 512, 1024]
    modes = ["forward", "forward_backward", "train_step"]

    results: list[dict] = []
    missing: list[str] = []

    for model_size in model_sizes:
        for context_length in context_lengths:
            for mode in modes:
                # Prefer the canonical _bs1_ naming (matches benchmarking_script output)
                run_name = f"{model_size}_ctx{context_length}_bs1_{mode}_fp32"
                sqlite_path = campaign_dir / run_name / "profile.sqlite"
                if not sqlite_path.exists():
                    # Fall back to old naming without _bs1_ (pre-fix runs)
                    run_name = f"{model_size}_ctx{context_length}_{mode}_fp32"
                    sqlite_path = campaign_dir / run_name / "profile.sqlite"
                if not sqlite_path.exists():
                    missing.append(f"{model_size}_ctx{context_length}_{mode}_fp32")
                    continue

                rec = analyse_sqlite(sqlite_path, model_size, context_length, mode)
                if rec:
                    results.append(rec)
                    print(f"  {model_size} ctx={context_length} {mode}: "
                          f"{rec.get('total_gpu_ms', 0):.1f} ms total, "
                          f"GEMM={rec.get('gemm_pct', 0):.1f}%")
                else:
                    missing.append(run_name)

    if missing:
        print(f"\nMissing/failed ({len(missing)}):", ", ".join(missing[:10]))

    # Save JSON
    summary = {
        "campaign": args.campaign,
        "results": results,
        "missing": missing,
    }
    out_json = analysis_dir / "nsys_kernel_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_json}")

    # Print focused summary for report
    print("\n=== SMALL MODEL KERNEL SUMMARY ===")
    for mode in ["forward", "train_step"]:
        recs = [r for r in results if r["model_size"] == "small"
                and r["context_length"] == 128 and r["mode"] == mode
                and "error" not in r]
        if not recs:
            print(f"  {mode}: no data")
            continue
        rec = recs[0]
        print(f"\n  {mode} (ctx=128): total GPU={rec['total_gpu_ms']:.1f} ms, GEMM={rec['gemm_pct']:.1f}%")
        for k in rec["top_kernels"][:8]:
            tag = " [GEMM]" if k["is_gemm"] else ""
            print(f"    {k['pct']:5.1f}%  {k['count']:4d}x  {k['name'][:80]}{tag}")

    # Markdown table for GEMM% across models and contexts
    md_lines = [
        "# nsys CUDA Kernel Analysis\n",
        "## GEMM fraction by mode (small model)\n",
        "| ctx | forward GEMM% | fwd+bwd GEMM% | train_step GEMM% |",
        "|-----|--------------|---------------|-----------------|",
    ]
    for ctx in context_lengths:
        row = [f"| {ctx}"]
        for mode in ["forward", "forward_backward", "train_step"]:
            recs = [r for r in results if r["model_size"] == "small"
                    and r["context_length"] == ctx and r["mode"] == mode
                    and "error" not in r]
            val = f"{recs[0]['gemm_pct']:.1f}%" if recs else "—"
            row.append(val)
        md_lines.append(" | ".join(row) + " |")

    out_md = analysis_dir / "nsys_kernel_summary.md"
    out_md.write_text("\n".join(md_lines) + "\n")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
