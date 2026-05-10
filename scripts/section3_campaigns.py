from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.campaign_utils import CampaignRun, ensure_layout, execute_runs


def build_runs(kind: str, smoke: bool) -> list[CampaignRun]:
    if smoke:
        optimizer_impls = ("standard", "sharded")
        return [
            CampaignRun(
                name=f"{kind}_{optimizer_impl}_smoke",
                module="cs336_systems.section3.benchmarks",
                args=(
                    kind,
                    "--optimizer-impl",
                    optimizer_impl,
                    "--backend",
                    "gloo",
                    "--device-type",
                    "cpu",
                    "--world-size",
                    "2",
                    "--model-size",
                    "small",
                    "--context-length",
                    "32",
                    "--batch-size",
                    "2",
                    "--warmup-steps",
                    "1",
                    "--measure-steps",
                    "1",
                    "--campaign",
                    "{campaign}",
                ),
            )
            for optimizer_impl in optimizer_impls
        ]

    return [
        CampaignRun(
            name=f"{kind}_standard",
            module="cs336_systems.section3.benchmarks",
            args=(
                kind,
                "--optimizer-impl",
                "standard",
                "--backend",
                "nccl",
                "--device-type",
                "cuda",
                "--world-size",
                "2",
                "--model-size",
                "xl",
                "--campaign",
                "{campaign}",
            ),
        ),
        CampaignRun(
            name=f"{kind}_sharded",
            module="cs336_systems.section3.benchmarks",
            args=(
                kind,
                "--optimizer-impl",
                "sharded",
                "--backend",
                "nccl",
                "--device-type",
                "cuda",
                "--world-size",
                "2",
                "--model-size",
                "xl",
                "--campaign",
                "{campaign}",
            ),
        ),
    ]


def run_campaign(args: argparse.Namespace) -> dict[str, object]:
    campaign = args.campaign or f"assignment_section3_{args.kind}_{'smoke' if args.smoke else 'full'}"
    runs = build_runs(args.kind, smoke=args.smoke)
    log_dir, data_dir, analysis_dir = ensure_layout("section3", campaign)

    (analysis_dir / "run_plan.json").write_text(
        json.dumps(
            {"kind": args.kind, "campaign": campaign, "smoke": args.smoke, "runs": [{"name": run.name, "module": run.module, "args": list(run.args)} for run in runs]},
            indent=2,
        )
    )

    _manifest, manifest_path = execute_runs(
        runs=runs,
        campaign=campaign,
        log_dir=log_dir,
        analysis_dir=analysis_dir,
        execute=args.execute,
        # Unique port per run to avoid conflicts between consecutive spawned workers.
        header_lines=[f"campaign={campaign}", f"kind={args.kind}", f"smoke={args.smoke}", f"run_count={len(runs)}"],
        base_port=29800,
    )
    return {
        "campaign": campaign,
        "kind": args.kind,
        "log_dir": str(log_dir),
        "data_dir": str(data_dir),
        "analysis_dir": str(analysis_dir),
        "manifest_path": str(manifest_path),
        "run_count": len(runs),
        "executed": args.execute,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch Section 3 benchmark campaigns")
    parser.add_argument("kind", choices=("timing", "memory"))
    parser.add_argument("--campaign", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run_campaign(parse_args()), indent=2))


if __name__ == "__main__":
    main()