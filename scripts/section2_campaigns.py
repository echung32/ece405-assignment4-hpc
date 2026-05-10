from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.campaign_utils import CampaignRun, ensure_layout, execute_runs


def build_allreduce_runs(smoke: bool) -> list[CampaignRun]:
    if smoke:
        return [
            CampaignRun(
                name="allreduce_gloo_cpu_smoke",
                module="cs336_systems.section2.benchmarks",
                args=(
                    "allreduce",
                    "--backend",
                    "gloo",
                    "--device-type",
                    "cpu",
                    "--world-size",
                    "2",
                    "--size-mb",
                    "1",
                    "--warmup-steps",
                    "1",
                    "--measure-steps",
                    "2",
                    "--campaign",
                    "{campaign}",
                ),
            )
        ]

    sizes = (1, 10, 100, 1024)
    world_sizes = (2, 4, 6)
    runs: list[CampaignRun] = []
    for backend, device_type in (("gloo", "cpu"), ("nccl", "cuda")):
        for world_size in world_sizes:
            for size_mb in sizes:
                runs.append(
                    CampaignRun(
                        name=f"allreduce_{backend}_{device_type}_ws{world_size}_{size_mb}mb",
                        module="cs336_systems.section2.benchmarks",
                        args=(
                            "allreduce",
                            "--backend",
                            backend,
                            "--device-type",
                            device_type,
                            "--world-size",
                            str(world_size),
                            "--size-mb",
                            str(size_mb),
                            "--campaign",
                            "{campaign}",
                        ),
                    )
                )
    return runs


def build_ddp_runs(smoke: bool) -> list[CampaignRun]:
    if smoke:
        implementations = (
            ("naive", ()),
            ("flat", ()),
            ("overlap_individual", ()),
            ("overlap_bucketed", ("--bucket-size-mb", "1")),
        )
        return [
            CampaignRun(
                name=f"ddp_{implementation}_smoke",
                module="cs336_systems.section2.benchmarks",
                args=(
                    "ddp",
                    "--implementation",
                    implementation,
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
                    *extra_args,
                    "--campaign",
                    "{campaign}",
                ),
            )
            for implementation, extra_args in implementations
        ]

    runs: list[CampaignRun] = []
    for implementation in ("naive", "flat", "overlap_individual"):
        runs.append(
            CampaignRun(
                name=f"ddp_{implementation}_xl",
                module="cs336_systems.section2.benchmarks",
                args=(
                    "ddp",
                    "--implementation",
                    implementation,
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
            )
        )
    for bucket_size_mb in (1, 10, 100, 1000):
        runs.append(
            CampaignRun(
                name=f"ddp_overlap_bucketed_{bucket_size_mb}mb",
                module="cs336_systems.section2.benchmarks",
                args=(
                    "ddp",
                    "--implementation",
                    "overlap_bucketed",
                    "--backend",
                    "nccl",
                    "--device-type",
                    "cuda",
                    "--world-size",
                    "2",
                    "--model-size",
                    "xl",
                    "--bucket-size-mb",
                    str(bucket_size_mb),
                    "--campaign",
                    "{campaign}",
                ),
            )
        )
    return runs


def build_runs(kind: str, smoke: bool) -> list[CampaignRun]:
    if kind == "allreduce":
        return build_allreduce_runs(smoke=smoke)
    if kind == "ddp":
        return build_ddp_runs(smoke=smoke)
    raise ValueError(f"Unsupported Section 2 campaign kind: {kind}")


def run_campaign(args: argparse.Namespace) -> dict[str, object]:
    campaign = args.campaign or f"assignment_section2_{args.kind}_{'smoke' if args.smoke else 'full'}"
    runs = build_runs(args.kind, smoke=args.smoke)
    log_dir, data_dir, analysis_dir = ensure_layout("section2", campaign)

    (analysis_dir / "run_plan.json").write_text(
        json.dumps(
            {
                "kind": args.kind,
                "campaign": campaign,
                "smoke": args.smoke,
                "runs": [{"name": run.name, "module": run.module, "args": list(run.args)} for run in runs],
            },
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
        base_port=29700,
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
    parser = argparse.ArgumentParser(description="Launch Section 2 benchmark campaigns")
    parser.add_argument("kind", choices=("allreduce", "ddp"))
    parser.add_argument("--campaign", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run_campaign(parse_args()), indent=2))


if __name__ == "__main__":
    main()