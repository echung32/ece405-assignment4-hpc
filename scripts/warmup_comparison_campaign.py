"""Run warmup comparison campaign for the small model.

Compares warmup_steps in {0, 1, 2, 5} at ctx in {128, 256, 512, 1024}
for forward and forward_backward modes with FP32 precision.

Each warmup count gets its own campaign directory so results don't overwrite.
Results land in data/section1_1/assignment_warmup_w{N}_full/ per AGENTS.md conventions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.campaign_utils import ensure_layout, execute_runs, CampaignRun

WARMUP_VARIANTS = [0, 1, 2, 5]
CONTEXT_LENGTHS = [128, 256, 512, 1024]
MODES = ["forward", "forward_backward"]
MEASURE_STEPS = 10


def build_runs(warmup: int, device: str) -> list[CampaignRun]:
    campaign_token = f"assignment_warmup_w{warmup}_full"
    runs: list[CampaignRun] = []
    for ctx in CONTEXT_LENGTHS:
        for mode in MODES:
            runs.append(
                CampaignRun(
                    name=f"small_ctx{ctx}_{mode}_fp32",
                    module="cs336_systems.section1.benchmarking_script",
                    args=(
                        "--model-size", "small",
                        "--context-length", str(ctx),
                        "--batch-size", "1",
                        "--mode", mode,
                        "--warmup-steps", str(warmup),
                        "--measure-steps", str(MEASURE_STEPS),
                        "--precision", "fp32",
                        "--device", device,
                        "--campaign", "{campaign}",
                        "--output-root", "data/section1_1",
                    ),
                )
            )
    return runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Warmup comparison campaign for small model")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Run only first config per warmup for quick validation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for warmup in WARMUP_VARIANTS:
        campaign = f"assignment_warmup_w{warmup}_full"
        runs = build_runs(warmup, args.device)
        if args.smoke:
            runs = runs[:1]

        log_dir, data_dir, analysis_dir = ensure_layout("section1_1", campaign)

        (analysis_dir / "run_plan.json").write_text(
            json.dumps(
                {
                    "campaign": campaign,
                    "warmup_steps": warmup,
                    "context_lengths": CONTEXT_LENGTHS,
                    "modes": MODES,
                    "measure_steps": MEASURE_STEPS,
                    "runs": [{"name": r.name, "module": r.module, "args": list(r.args)} for r in runs],
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
            header_lines=[
                f"campaign={campaign}",
                f"warmup_steps={warmup}",
                f"device={args.device}",
                f"run_count={len(runs)}",
            ],
        )
        print(json.dumps({"campaign": campaign, "warmup_steps": warmup, "run_count": len(runs), "manifest_path": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
