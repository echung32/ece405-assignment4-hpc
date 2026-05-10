"""Memory profiling harness for Section 1.1.6 (memory_profiling problem).

This module provides the memory-profiling-focused CLI entry point.  The
underlying benchmark infrastructure is shared with benchmarking_script.py;
this module just exposes a more ergonomic interface for the memory-profiling
workflow (--profile-memory is on by default, default mode is train_step).
"""
from __future__ import annotations

import argparse
import json

from cs336_systems.section1.benchmarking_script import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_MEASURE_STEPS,
    DEFAULT_VOCAB_SIZE,
    DEFAULT_VRAM_LIMIT_GB,
    DEFAULT_WARMUP_STEPS,
    MODEL_SPECS,
    RunConfig,
    run_benchmark,
)


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Section 1.1.6 memory profiling harness.  "
            "Wraps the standard benchmarking harness with --profile-memory "
            "enabled by default so PyTorch memory snapshots are always captured."
        )
    )
    parser.add_argument("--model-size", choices=sorted(MODEL_SPECS.keys()), default="2.7b")
    parser.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT_LENGTH)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--mode",
        choices=("forward", "forward_backward", "train_step"),
        default="train_step",
    )
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument("--measure-steps", type=int, default=1)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vram-limit-gb", type=float, default=DEFAULT_VRAM_LIMIT_GB)
    parser.add_argument(
        "--no-profile-memory",
        action="store_true",
        help="Disable memory snapshot capture (enabled by default for this entry point).",
    )
    parser.add_argument("--annotate-attention", action="store_true")
    parser.add_argument("--campaign", default="default")
    parser.add_argument("--output-root", default="data/section1_1")
    args = parser.parse_args()
    return RunConfig(
        model_size=args.model_size,
        context_length=args.context_length,
        batch_size=args.batch_size,
        mode=args.mode,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        precision=args.precision,
        device=args.device,
        seed=args.seed,
        vram_limit_gb=args.vram_limit_gb,
        profile_memory=not args.no_profile_memory,
        annotate_attention=args.annotate_attention,
        campaign=args.campaign,
        output_root=args.output_root,
    )


def main() -> None:
    config = parse_args()
    result = run_benchmark(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
