from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CampaignRun:
    name: str
    area: str
    module: str
    args: tuple[str, ...]


def ensure_layout(area: str, campaign: str) -> tuple[Path, Path, Path]:
    log_dir = Path("logs") / area / campaign
    data_dir = Path("data") / area / campaign
    analysis_dir = Path("data") / area / "analysis" / campaign
    log_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    return log_dir, data_dir, analysis_dir


def build_attention_runs(smoke: bool, device: str) -> list[CampaignRun]:
    common_args = (
        "--device",
        device,
        "--campaign",
        "{campaign}",
        "--output-root",
        "data/section1_2",
    )
    if smoke:
        return [
            CampaignRun(
                name="naive_smoke",
                area="section1_2",
                module="cs336_systems.section1.pytorch_attention",
                args=("--implementation", "naive", "--smoke", *common_args),
            ),
            CampaignRun(
                name="compiled_naive_smoke",
                area="section1_2",
                module="cs336_systems.section1.pytorch_attention",
                args=("--implementation", "compiled_naive", "--smoke", *common_args),
            ),
            CampaignRun(
                name="flash_triton_smoke",
                area="section1_2",
                module="cs336_systems.section1.pytorch_attention",
                args=("--implementation", "flash_triton", "--smoke", *common_args),
            ),
        ]

    return [
        CampaignRun(
            name="naive_fp32",
            area="section1_2",
            module="cs336_systems.section1.pytorch_attention",
            args=("--implementation", "naive", *common_args),
        ),
        CampaignRun(
            name="compiled_naive_fp32",
            area="section1_2",
            module="cs336_systems.section1.pytorch_attention",
            args=("--implementation", "compiled_naive", *common_args),
        ),
        CampaignRun(
            name="flash_pytorch_fp32",
            area="section1_2",
            module="cs336_systems.section1.pytorch_attention",
            args=("--implementation", "flash_pytorch", *common_args),
        ),
        CampaignRun(
            name="flash_triton_fp32",
            area="section1_2",
            module="cs336_systems.section1.pytorch_attention",
            args=("--implementation", "flash_triton", *common_args),
        ),
    ]


def build_model_compile_runs(smoke: bool, device: str) -> list[CampaignRun]:
    common_prefix = (
        "--device",
        device,
        "--campaign",
        "{campaign}",
        "--output-root",
        "data/section1_3",
    )
    model_sizes = ["small"] if smoke else ["small", "medium", "large", "xl", "2.7b"]
    modes = ["forward"] if smoke else ["forward", "forward_backward", "train_step"]
    smoke_flag = ("--smoke",) if smoke else ()
    runs: list[CampaignRun] = []
    for model_size in model_sizes:
        for mode in modes:
            runs.append(
                CampaignRun(
                    name=f"{model_size}_{mode}_eager",
                    area="section1_3",
                    module="cs336_systems.section1.torch_compile",
                    args=("--model-size", model_size, "--mode", mode, *smoke_flag, *common_prefix),
                )
            )
            runs.append(
                CampaignRun(
                    name=f"{model_size}_{mode}_compiled",
                    area="section1_3",
                    module="cs336_systems.section1.torch_compile",
                    args=("--model-size", model_size, "--mode", mode, "--compiled", *smoke_flag, *common_prefix),
                )
            )
    return runs


def build_flash_runs(smoke: bool, device: str) -> list[CampaignRun]:
    common_prefix = (
        "--device",
        device,
        "--campaign",
        "{campaign}",
        "--output-root",
        "data/section1_3",
        "--is-causal",
    )
    dtypes = ["bf16"] if smoke else ["bf16", "fp32"]
    implementations = ["naive", "flash_triton"]
    runs: list[CampaignRun] = []
    for dtype in dtypes:
        for implementation in implementations:
            smoke_flag = ("--smoke",) if smoke else ()
            runs.append(
                CampaignRun(
                    name=f"{implementation}_{dtype}",
                    area="section1_3",
                    module="cs336_systems.section1.flash_benchmark",
                    args=("--implementation", implementation, "--dtype", dtype, *smoke_flag, *common_prefix),
                )
            )
    return runs


def build_runs(kind: str, smoke: bool, device: str) -> list[CampaignRun]:
    if kind == "attention":
        return build_attention_runs(smoke=smoke, device=device)
    if kind == "model-compile":
        return build_model_compile_runs(smoke=smoke, device=device)
    if kind == "flash":
        return build_flash_runs(smoke=smoke, device=device)
    raise ValueError(f"Unsupported campaign kind: {kind}")


def run_campaign(args: argparse.Namespace) -> dict[str, object]:
    runs = build_runs(args.kind, smoke=args.smoke, device=args.device)
    area = runs[0].area
    campaign = args.campaign or f"assignment_{args.kind}_{'smoke' if args.smoke else 'full'}"
    log_dir, data_dir, analysis_dir = ensure_layout(area, campaign)

    plan_payload = {
        "kind": args.kind,
        "campaign": campaign,
        "device": args.device,
        "smoke": args.smoke,
        "runs": [
            {
                "name": run.name,
                "area": run.area,
                "module": run.module,
                "args": list(run.args),
            }
            for run in runs
        ],
    }
    (analysis_dir / "run_plan.json").write_text(json.dumps(plan_payload, indent=2))

    orchestrator_lines = [
        f"campaign={campaign}",
        f"kind={args.kind}",
        f"area={area}",
        f"device={args.device}",
        f"smoke={args.smoke}",
        f"run_count={len(runs)}",
    ]
    manifest: list[dict[str, object]] = []

    for run in runs:
        rendered_args = tuple(campaign if arg == "{campaign}" else arg for arg in run.args)
        command = [sys.executable, "-m", run.module, *rendered_args]
        command_str = shlex.join(command)
        log_file = log_dir / f"{run.name}.log"
        entry = {
            "name": run.name,
            "command": command,
            "command_str": command_str,
            "log_file": str(log_file),
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
        "kind": args.kind,
        "area": area,
        "log_dir": str(log_dir),
        "data_dir": str(data_dir),
        "analysis_dir": str(analysis_dir),
        "manifest_path": str(manifest_path),
        "run_count": len(runs),
        "executed": args.execute,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch Section 1.2/1.3 benchmark campaigns")
    parser.add_argument("kind", choices=("attention", "model-compile", "flash"))
    parser.add_argument("--campaign", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run_campaign(parse_args()), indent=2))


if __name__ == "__main__":
    main()