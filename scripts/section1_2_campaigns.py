from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.campaign_utils import CampaignRun, ensure_layout, execute_runs

_AREA_FOR_KIND: dict[str, str] = {
    "attention": "section1_2",
    "model-compile": "section1_3",
    "flash": "section1_3",
}

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
                module="cs336_systems.section1.pytorch_attention",
                args=("--implementation", "naive", "--smoke", *common_args),
            ),
            CampaignRun(
                name="compiled_naive_smoke",
                module="cs336_systems.section1.pytorch_attention",
                args=("--implementation", "compiled_naive", "--smoke", *common_args),
            ),
            CampaignRun(
                name="flash_triton_smoke",
                module="cs336_systems.section1.pytorch_attention",
                args=("--implementation", "flash_triton", "--smoke", *common_args),
            ),
        ]

    return [
        CampaignRun(
            name="naive_fp32",
            module="cs336_systems.section1.pytorch_attention",
            args=("--implementation", "naive", *common_args),
        ),
        CampaignRun(
            name="compiled_naive_fp32",
            module="cs336_systems.section1.pytorch_attention",
            args=("--implementation", "compiled_naive", *common_args),
        ),
        CampaignRun(
            name="flash_pytorch_fp32",
            module="cs336_systems.section1.pytorch_attention",
            args=("--implementation", "flash_pytorch", *common_args),
        ),
        CampaignRun(
            name="flash_triton_fp32",
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
                    module="cs336_systems.section1.torch_compile",
                    args=("--model-size", model_size, "--mode", mode, *smoke_flag, *common_prefix),
                )
            )
            runs.append(
                CampaignRun(
                    name=f"{model_size}_{mode}_compiled",
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
    area = _AREA_FOR_KIND[args.kind]
    campaign = args.campaign or f"assignment_{args.kind}_{'smoke' if args.smoke else 'full'}"
    log_dir, data_dir, analysis_dir = ensure_layout(area, campaign)

    (analysis_dir / "run_plan.json").write_text(
        json.dumps(
            {
                "kind": args.kind,
                "campaign": campaign,
                "area": area,
                "device": args.device,
                "smoke": args.smoke,
                "runs": [{"name": run.name, "module": run.module, "args": list(run.args)} for run in runs],
            },
            indent=2,
        )
    )

    header_lines = [
        f"campaign={campaign}",
        f"kind={args.kind}",
        f"area={area}",
        f"device={args.device}",
        f"smoke={args.smoke}",
        f"run_count={len(runs)}",
    ]
    _manifest, manifest_path = execute_runs(
        runs=runs,
        campaign=campaign,
        log_dir=log_dir,
        analysis_dir=analysis_dir,
        execute=args.execute,
        header_lines=header_lines,
    )

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