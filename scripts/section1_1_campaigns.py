from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from scripts.campaign_utils import ensure_layout as _ensure_layout


@dataclass(frozen=True)
class CampaignRun:
    name: str
    model_size: str
    context_length: int
    batch_size: int
    mode: str
    precision: str
    warmup_steps: int
    measure_steps: int
    annotate_attention: bool = False
    profile_memory: bool = False


def build_nsys_runs(smoke: bool, batch_size: int) -> list[CampaignRun]:
    if smoke:
        return [
            CampaignRun(
                name="small_ctx128_train_step_fp32",
                model_size="small",
                context_length=128,
                batch_size=batch_size,
                mode="train_step",
                precision="fp32",
                warmup_steps=1,
                measure_steps=1,
                annotate_attention=True,
            )
        ]

    model_sizes = ["small", "medium", "large", "xl", "2.7b"]
    context_lengths = [128, 256, 512, 1024]
    modes = ["forward", "forward_backward", "train_step"]
    runs: list[CampaignRun] = []
    for model_size in model_sizes:
        for context_length in context_lengths:
            for mode in modes:
                runs.append(
                    CampaignRun(
                        name=f"{model_size}_ctx{context_length}_{mode}_fp32",
                        model_size=model_size,
                        context_length=context_length,
                        batch_size=batch_size,
                        mode=mode,
                        precision="fp32",
                        warmup_steps=5,
                        measure_steps=1,
                        annotate_attention=True,
                    )
                )
    return runs


def build_memory_runs(smoke: bool, batch_size: int) -> list[CampaignRun]:
    if smoke:
        return [
            CampaignRun(
                name="small_ctx128_forward_fp32",
                model_size="small",
                context_length=128,
                batch_size=batch_size,
                mode="forward",
                precision="fp32",
                warmup_steps=1,
                measure_steps=1,
                profile_memory=True,
            )
        ]

    context_lengths = [128, 256, 512]
    modes = ["forward", "train_step"]
    precisions = ["fp32", "bf16"]
    runs: list[CampaignRun] = []
    for context_length in context_lengths:
        for mode in modes:
            for precision in precisions:
                runs.append(
                    CampaignRun(
                        name=f"2.7b_ctx{context_length}_{mode}_{precision}",
                        model_size="2.7b",
                        context_length=context_length,
                        batch_size=batch_size,
                        mode=mode,
                        precision=precision,
                        warmup_steps=5,
                        measure_steps=1,
                        profile_memory=True,
                    )
                )
    return runs


def build_timing_runs(smoke: bool, batch_size: int) -> list[CampaignRun]:
    if smoke:
        return [
            CampaignRun(
                name="small_ctx128_bs1_forward_fp32",
                model_size="small",
                context_length=128,
                batch_size=batch_size,
                mode="forward",
                precision="fp32",
                warmup_steps=5,
                measure_steps=10,
            )
        ]

    model_sizes = ["small", "medium", "large", "xl", "2.7b"]
    context_lengths = [128, 256, 512, 1024]
    modes = ["forward", "forward_backward", "train_step"]
    precisions = ["fp32", "bf16"]
    runs: list[CampaignRun] = []
    for model_size in model_sizes:
        for context_length in context_lengths:
            for mode in modes:
                for precision in precisions:
                    runs.append(
                        CampaignRun(
                            name=f"{model_size}_ctx{context_length}_{mode}_{precision}",
                            model_size=model_size,
                            context_length=context_length,
                            batch_size=batch_size,
                            mode=mode,
                            precision=precision,
                            warmup_steps=5,
                            measure_steps=10,
                        )
                    )
    return runs


def build_runs(kind: str, smoke: bool, batch_size: int) -> list[CampaignRun]:
    if kind == "timing":
        return build_timing_runs(smoke=smoke, batch_size=batch_size)
    if kind == "nsys":
        return build_nsys_runs(smoke=smoke, batch_size=batch_size)
    if kind == "memory":
        return build_memory_runs(smoke=smoke, batch_size=batch_size)
    raise ValueError(f"Unsupported campaign kind: {kind}")


def ensure_layout(kind: str, campaign: str) -> tuple[Path, Path, Path]:
    return _ensure_layout("section1_1", campaign)


def build_section_command(run: CampaignRun, campaign: str, output_root: str, device: str, vram_limit_gb: float) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "cs336_systems.section1.benchmarking_script",
        "--model-size",
        run.model_size,
        "--context-length",
        str(run.context_length),
        "--batch-size",
        str(run.batch_size),
        "--mode",
        run.mode,
        "--warmup-steps",
        str(run.warmup_steps),
        "--measure-steps",
        str(run.measure_steps),
        "--precision",
        run.precision,
        "--device",
        device,
        "--campaign",
        campaign,
        "--output-root",
        output_root,
        "--vram-limit-gb",
        str(vram_limit_gb),
    ]
    if run.annotate_attention:
        command.append("--annotate-attention")
    if run.profile_memory:
        command.append("--profile-memory")
    return command


def build_execution_command(
    kind: str,
    run: CampaignRun,
    campaign: str,
    output_root: str,
    device: str,
    vram_limit_gb: float,
    nsys_path: str,
) -> tuple[list[str], str | None]:
    base_command = build_section_command(run, campaign, output_root, device, vram_limit_gb)
    if kind != "nsys":
        return base_command, None

    run_dir = Path(output_root) / campaign / run.name
    run_dir.mkdir(parents=True, exist_ok=True)
    report_prefix = str(run_dir / "profile")
    command = [
        nsys_path,
        "profile",
        "--trace=cuda,nvtx,osrt",
        "--sample=none",
        "--force-overwrite=true",
        "-o",
        report_prefix,
        *base_command,
    ]
    return command, report_prefix


def run_campaign(args: argparse.Namespace) -> dict[str, object]:
    campaign = args.campaign or f"assignment_{args.kind}_{'smoke' if args.smoke else 'full'}"
    runs = build_runs(args.kind, smoke=args.smoke, batch_size=args.batch_size)
    log_dir, data_dir, analysis_dir = ensure_layout(args.kind, campaign)

    plan_payload = {
        "kind": args.kind,
        "campaign": campaign,
        "device": args.device,
        "vram_limit_gb": args.vram_limit_gb,
        "smoke": args.smoke,
        "runs": [asdict(run) for run in runs],
    }
    (analysis_dir / "run_plan.json").write_text(json.dumps(plan_payload, indent=2))

    orchestrator_log = log_dir / "orchestrator.log"
    orchestrator_lines = [
        f"campaign={campaign}",
        f"kind={args.kind}",
        f"device={args.device}",
        f"smoke={args.smoke}",
        f"run_count={len(runs)}",
    ]

    manifest: list[dict[str, object]] = []
    for run in runs:
        command, report_prefix = build_execution_command(
            kind=args.kind,
            run=run,
            campaign=campaign,
            output_root=str(data_dir.parent),
            device=args.device,
            vram_limit_gb=args.vram_limit_gb,
            nsys_path=args.nsys_path,
        )
        command_str = shlex.join(command)
        log_file = log_dir / f"{run.name}.log"
        entry = {
            "name": run.name,
            "command": command,
            "command_str": command_str,
            "log_file": str(log_file),
            "report_prefix": report_prefix,
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

    orchestrator_log.write_text("\n".join(orchestrator_lines) + "\n")
    manifest_path = analysis_dir / "campaign_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return {
        "campaign": campaign,
        "kind": args.kind,
        "log_dir": str(log_dir),
        "analysis_dir": str(analysis_dir),
        "manifest_path": str(manifest_path),
        "run_count": len(runs),
        "executed": args.execute,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch assignment-aligned Section 1.1 campaigns")
    parser.add_argument("kind", choices=("nsys", "memory", "timing"))
    parser.add_argument("--campaign", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--vram-limit-gb", type=float, default=50.0)
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