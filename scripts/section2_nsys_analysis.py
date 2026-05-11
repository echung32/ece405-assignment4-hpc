"""Section 2 DDP nsys analysis script.

Reads SQLite profiles from the DDP nsys campaign and extracts:
- NCCL kernel counts during backward vs comm_sync NVTX windows
- Phase durations (forward, backward, comm_sync)
- Overlap fraction: NCCL kernels during backward / total NCCL kernels

Usage:
    uv run python -m scripts.section2_nsys_analysis
    uv run python -m scripts.section2_nsys_analysis --campaign assignment_section2_ddp_nsys_smoke
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


NCCL_PREFIXES = ["ncclDev", "nccl"]
CAMPAIGN_DEFAULT = "assignment_section2_ddp_nsys_full"
DATA_ROOT = Path("data/section2")


def is_nccl_kernel(name: str) -> bool:
    return any(name.startswith(p) or p in name for p in NCCL_PREFIXES)


def get_nvtx_ranges(con: sqlite3.Connection, label: str) -> list[tuple[int, int]]:
    """Return (start, end) pairs for NVTX ranges with the given text label."""
    rows = con.execute(
        "SELECT start, end FROM NVTX_EVENTS WHERE text = ? AND end IS NOT NULL ORDER BY start",
        (label,),
    ).fetchall()
    return rows


def pair_ranks(rows: list[tuple[int, int]], gap_ns: int = 500_000_000) -> list[tuple[int, int]]:
    """Collapse rank-0 / rank-1 pairs for each step, returning the rank-0 (earlier) range."""
    paired: list[tuple[int, int]] = []
    i = 0
    while i < len(rows):
        if i + 1 < len(rows) and abs(rows[i + 1][0] - rows[i][0]) < gap_ns:
            paired.append(rows[i])  # take rank-0 (smaller start)
            i += 2
        else:
            paired.append(rows[i])
            i += 1
    return paired


def kernel_stats_in_window(con: sqlite3.Connection, t_start: int, t_end: int) -> dict[str, int]:
    """Count NCCL and compute kernels within the given ns time window."""
    rows = con.execute(
        """
        SELECT s.value AS name, COUNT(*) AS cnt
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.shortName = s.id
        WHERE k.start >= ? AND k.start < ?
        GROUP BY k.shortName
        """,
        (t_start, t_end),
    ).fetchall()
    total = sum(cnt for _, cnt in rows)
    nccl = sum(cnt for name, cnt in rows if is_nccl_kernel(name))
    return {"total": total, "nccl": nccl, "compute": total - nccl}


def analyse_profile(db_path: Path) -> dict[str, object]:
    """Analyse a single DDP nsys SQLite profile."""
    con = sqlite3.connect(str(db_path))

    # Extract NVTX ranges for each phase (rank 0 = earlier of each pair)
    phases: dict[str, list[tuple[int, int]]] = {}
    for label in ("measurement", "forward", "backward", "comm_sync"):
        rows = get_nvtx_ranges(con, label)
        phases[label] = pair_ranks(rows)

    # Separate warmup and measurement steps
    # measurement NVTX wraps only the measure_steps, so we can use it to identify step indices
    n_measurement = len(phases["measurement"])

    # Total steps = warmup_steps + measure_steps; backward has all steps
    n_total_steps = len(phases["backward"])
    n_warmup = n_total_steps - n_measurement

    step_results: list[dict[str, object]] = []
    for step_i in range(n_total_steps):
        is_measurement = step_i >= n_warmup
        bwd_start, bwd_end = phases["backward"][step_i]
        bwd_dur_ms = (bwd_end - bwd_start) / 1e6

        fwd_dur_ms = None
        if step_i < len(phases["forward"]):
            fwd_start, fwd_end = phases["forward"][step_i]
            fwd_dur_ms = (fwd_end - fwd_start) / 1e6

        sync_dur_ms = None
        nccl_during_sync = 0
        if step_i < len(phases["comm_sync"]):
            cs_start, cs_end = phases["comm_sync"][step_i]
            sync_dur_ms = (cs_end - cs_start) / 1e6
            nccl_during_sync = kernel_stats_in_window(con, cs_start, cs_end)["nccl"]

        bwd_kernels = kernel_stats_in_window(con, bwd_start, bwd_end)

        step_results.append(
            {
                "step": step_i,
                "is_measurement": is_measurement,
                "fwd_dur_ms": fwd_dur_ms,
                "bwd_dur_ms": bwd_dur_ms,
                "sync_dur_ms": sync_dur_ms,
                "bwd_total_kernels": bwd_kernels["total"],
                "bwd_nccl_kernels": bwd_kernels["nccl"],
                "bwd_compute_kernels": bwd_kernels["compute"],
                "bwd_nccl_pct": 100 * bwd_kernels["nccl"] / bwd_kernels["total"] if bwd_kernels["total"] else 0,
                "sync_nccl_kernels": nccl_during_sync,
            }
        )

    # Summary over measurement steps only
    meas = [s for s in step_results if s["is_measurement"]]
    if meas:
        avg_fwd = sum(s["fwd_dur_ms"] for s in meas if s["fwd_dur_ms"] is not None) / len(meas)
        avg_bwd = sum(s["bwd_dur_ms"] for s in meas) / len(meas)
        avg_sync = sum(s["sync_dur_ms"] for s in meas if s["sync_dur_ms"] is not None) / len(meas)
        avg_nccl_pct = sum(s["bwd_nccl_pct"] for s in meas) / len(meas)
        avg_bwd_nccl = sum(s["bwd_nccl_kernels"] for s in meas) / len(meas)
        avg_sync_nccl = sum(s["sync_nccl_kernels"] for s in meas) / len(meas)
        total_nccl = avg_bwd_nccl + avg_sync_nccl
        overlap_fraction = avg_bwd_nccl / total_nccl if total_nccl > 0 else 0
    else:
        avg_fwd = avg_bwd = avg_sync = avg_nccl_pct = overlap_fraction = 0

    con.close()
    return {
        "db_path": str(db_path),
        "steps": step_results,
        "summary": {
            "n_measurement_steps": len(meas),
            "avg_fwd_ms": avg_fwd,
            "avg_bwd_ms": avg_bwd,
            "avg_sync_ms": avg_sync,
            "avg_nccl_pct_during_bwd": avg_nccl_pct,
            "overlap_fraction": overlap_fraction,
        },
    }


def run_analysis(campaign: str) -> dict[str, object]:
    campaign_dir = DATA_ROOT / campaign
    analysis_dir = DATA_ROOT / "analysis" / campaign
    analysis_dir.mkdir(parents=True, exist_ok=True)

    profiles: list[dict[str, object]] = []
    for run_dir in sorted(campaign_dir.iterdir()):
        db = run_dir / "profile.sqlite"
        if not db.exists():
            continue
        print(f"Analysing {run_dir.name} ...")
        result = analyse_profile(db)
        result["run_name"] = run_dir.name
        profiles.append(result)

    output = {"campaign": campaign, "profiles": profiles}
    json_path = analysis_dir / "ddp_nsys_summary.json"
    json_path.write_text(json.dumps(output, indent=2))

    # Human-readable report
    lines = [f"# DDP nsys Analysis — {campaign}", ""]
    for p in profiles:
        name = p["run_name"]
        s = p["summary"]
        lines.append(f"## {name}")
        lines.append(f"  fwd={s['avg_fwd_ms']:.1f}ms  bwd={s['avg_bwd_ms']:.1f}ms  sync={s['avg_sync_ms']:.1f}ms")
        lines.append(
            f"  NCCL kernels during backward: {s['avg_nccl_pct_during_bwd']:.1f}%"
            f"  overlap_fraction={s['overlap_fraction']:.2f}"
        )
        lines.append("")

    md_path = analysis_dir / "ddp_nsys_summary.md"
    md_path.write_text("\n".join(lines))
    print(f"\nOutput written to {json_path} and {md_path}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse DDP nsys SQLite profiles")
    parser.add_argument("--campaign", default=CAMPAIGN_DEFAULT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_analysis(args.campaign)
    # Print summary table
    print("\n=== Summary ===")
    for p in result["profiles"]:
        s = p["summary"]
        print(
            f"{p['run_name']:<55}  "
            f"bwd={s['avg_bwd_ms']:6.1f}ms  "
            f"sync={s['avg_sync_ms']:5.1f}ms  "
            f"nccl_during_bwd={s['avg_nccl_pct_during_bwd']:4.1f}%  "
            f"overlap={s['overlap_fraction']:.2f}"
        )


if __name__ == "__main__":
    main()
