"""Section 2 DDP nsys profiling campaign.

Profiles naive vs overlap_individual DDP implementations with nsys to show
the sequential vs overlapping communication pattern.

Usage:
    uv run python -m scripts.section2_nsys_campaigns --smoke --execute
    uv run python -m scripts.section2_nsys_campaigns --execute
    uv run python -m scripts.section2_nsys_campaigns  # dry-run
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from scripts.campaign_utils import ensure_layout


@dataclass(frozen=True)
class NsysDDPRun:
    name: str
    implementation: str
    model_size: str
    context_length: int
    batch_size: int
    warmup_steps: int
    measure_steps: int


def build_runs(smoke: bool) -> list[NsysDDPRun]:
    if smoke:
        return [
            NsysDDPRun(
                name=f"ddp_{impl}_small_ctx128",
                implementation=impl,
                model_size="small",
                context_length=128,
                batch_size=4,
                warmup_steps=1,
                measure_steps=1,
            )
            for impl in ("naive", "overlap_individual")
        ]
    # Full run: XL model at ctx=128 to match assignment spec (1 node, 2 GPUs, XL)
    runs: list[NsysDDPRun] = []
    for impl in ("naive", "overlap_individual"):
        runs.append(
            NsysDDPRun(
                name=f"ddp_{impl}_xl_ctx128",
                implementation=impl,
                model_size="xl",
                context_length=128,
                batch_size=4,
                warmup_steps=2,
                measure_steps=2,
            )
        )
    return runs


def build_ddp_command(run: NsysDDPRun, campaign: str, data_dir: Path, master_port: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "cs336_systems.section2.benchmarks",
        "ddp",
        "--implementation", run.implementation,
        "--backend", "nccl",
        "--device-type", "cuda",
        "--world-size", "2",
        "--model-size", run.model_size,
        "--context-length", str(run.context_length),
        "--batch-size", str(run.batch_size),
        "--warmup-steps", str(run.warmup_steps),
        "--measure-steps", str(run.measure_steps),
        "--campaign", campaign,
        "--output-root", str(data_dir.parent),
        "--master-port", str(master_port),
    ]


def build_nsys_command(
    inner_command: list[str],
    profile_prefix: str,
    nsys_path: str,
) -> list[str]:
    return [
        nsys_path,
        "profile",
        "--trace=cuda,nvtx,nccl",
        "--sample=none",
        "--force-overwrite=true",
        "--export=sqlite",
        "-o", profile_prefix,
        *inner_command,
    ]


def run_campaign(args: argparse.Namespace) -> dict[str, object]:
    campaign = args.campaign or f"assignment_section2_ddp_nsys_{'smoke' if args.smoke else 'full'}"
    runs = build_runs(smoke=args.smoke)

    log_dir, data_dir, analysis_dir = ensure_layout("section2", campaign)

    (analysis_dir / "run_plan.json").write_text(
        json.dumps(
            {
                "campaign": campaign,
                "smoke": args.smoke,
                "runs": [
                    {
                        "name": r.name,
                        "implementation": r.implementation,
                        "model_size": r.model_size,
                        "context_length": r.context_length,
                    }
                    for r in runs
                ],
            },
            indent=2,
        )
    )

    orchestrator_lines = [
        f"campaign={campaign}",
        f"smoke={args.smoke}",
        f"run_count={len(runs)}",
    ]
    manifest: list[dict[str, object]] = []

    for run_idx, run in enumerate(runs):
        master_port = 29800 + run_idx

        # The DDP benchmark stores results under data/section2/{campaign}/{run_name}/
        # run_name mirrors the format in run_ddp():
        impl_run_name = (
            f"{run.implementation}_nccl_cuda_ws2_{run.model_size}"
            f"_ctx{run.context_length}_bs{run.batch_size}"
        )
        profile_dir = data_dir / impl_run_name
        profile_dir.mkdir(parents=True, exist_ok=True)
        profile_prefix = str(profile_dir / "profile")

        inner_cmd = build_ddp_command(run, campaign, data_dir, master_port)
        command = build_nsys_command(inner_cmd, profile_prefix, args.nsys_path)
        command_str = shlex.join(command)
        log_file = log_dir / f"{run.name}.log"

        entry: dict[str, object] = {
            "name": run.name,
            "command": command,
            "command_str": command_str,
            "log_file": str(log_file),
            "profile_prefix": profile_prefix,
            "status": "planned",
        }
        orchestrator_lines.append(f"planned {run.name}: {command_str}")

        if args.execute:
            with log_file.open("w") as handle:
                completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, text=True)
            entry["return_code"] = completed.returncode
            entry["status"] = "completed" if completed.returncode == 0 else "failed"
            orchestrator_lines.append(f"finished {run.name}: return_code={completed.returncode}")

        manifest.append(entry)

    (log_dir / "orchestrator.log").write_text("\n".join(orchestrator_lines) + "\n")
    manifest_path = analysis_dir / "campaign_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return {
        "campaign": campaign,
        "log_dir": str(log_dir),
        "data_dir": str(data_dir),
        "analysis_dir": str(analysis_dir),
        "manifest_path": str(manifest_path),
        "run_count": len(runs),
        "executed": args.execute,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Section 2 DDP nsys profiling campaign")
    parser.add_argument("--campaign", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--nsys-path", default="nsys")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_campaign(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
