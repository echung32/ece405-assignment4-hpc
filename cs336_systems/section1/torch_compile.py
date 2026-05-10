from __future__ import annotations

import argparse
import json
import statistics
import timeit
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_systems.section1.benchmarking_script import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_MEASURE_STEPS,
    DEFAULT_VOCAB_SIZE,
    DEFAULT_WARMUP_STEPS,
    MODEL_SPECS,
    ModelSpec,
    create_model,
    generate_batch,
    get_autocast_context,
    maybe_raise_for_vram_limit,
    sanitize_name,
    synchronize,
)


@dataclass(frozen=True)
class ModelCompileBenchmarkConfig:
    model_size: str
    context_length: int
    batch_size: int
    mode: str
    compiled: bool
    warmup_steps: int
    measure_steps: int
    precision: str
    device: str
    campaign: str
    output_root: str
    vram_limit_gb: float


def make_output_dir(config: ModelCompileBenchmarkConfig) -> Path:
    run_name = (
        f"{sanitize_name(config.model_size)}_ctx{config.context_length}_bs{config.batch_size}_"
        f"{config.mode}_{config.precision}_{'compiled' if config.compiled else 'eager'}"
    )
    output_dir = Path(config.output_root) / config.campaign / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run_step(
    model: torch.nn.Module,
    optimizer: AdamW | None,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    config: ModelCompileBenchmarkConfig,
) -> None:
    device = inputs.device
    if config.mode != "forward":
        model.zero_grad(set_to_none=True)
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    with get_autocast_context(device, config.precision):
        if config.mode == "forward":
            with torch.no_grad():
                model(inputs)
        else:
            logits = model(inputs)
            loss = cross_entropy(logits, targets)
            loss.backward()
            if config.mode == "train_step":
                if optimizer is None:
                    raise ValueError("Optimizer is required for train_step mode")
                optimizer.step()

    synchronize(device)
    maybe_raise_for_vram_limit(device, config.vram_limit_gb)


def benchmark_single_config(config: ModelCompileBenchmarkConfig) -> dict[str, object]:
    if config.model_size not in MODEL_SPECS:
        raise ValueError(f"Unknown model size: {config.model_size}")

    device = torch.device(config.device)
    if device.type != "cuda":
        raise ValueError("Model compile benchmarking is intended for CUDA devices")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    spec: ModelSpec = MODEL_SPECS[config.model_size]
    model = create_model(spec, config.context_length, DEFAULT_VOCAB_SIZE, device)
    eager_model = model
    if config.compiled:
        model = torch.compile(model)
    optimizer = AdamW(eager_model.parameters(), lr=1e-3) if config.mode == "train_step" else None
    inputs, targets = generate_batch(config.batch_size, config.context_length, DEFAULT_VOCAB_SIZE, device)

    for _ in range(config.warmup_steps):
        run_step(model, optimizer, inputs, targets, config)

    timings: list[float] = []
    for _ in range(config.measure_steps):
        start_time = timeit.default_timer()
        run_step(model, optimizer, inputs, targets, config)
        timings.append(timeit.default_timer() - start_time)

    result = {
        "config": asdict(config),
        "model_spec": asdict(spec),
        "timings_seconds": timings,
        "mean_seconds": statistics.fmean(timings),
        "stdev_seconds": statistics.stdev(timings) if len(timings) > 1 else 0.0,
    }

    result_path = make_output_dir(config) / "result.json"
    result_path.write_text(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark eager vs compiled full-model execution for Section 1.3")
    parser.add_argument("--model-size", choices=sorted(MODEL_SPECS.keys()), default="small")
    parser.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT_LENGTH)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--mode", choices=("forward", "forward_backward", "train_step"), default="forward")
    parser.add_argument("--compiled", action="store_true")
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument("--measure-steps", type=int, default=DEFAULT_MEASURE_STEPS)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--campaign", default="default")
    parser.add_argument("--output-root", default="data/section1_3")
    parser.add_argument("--vram-limit-gb", type=float, default=40.0)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ModelCompileBenchmarkConfig(
        model_size=args.model_size,
        context_length=args.context_length,
        batch_size=args.batch_size,
        mode=args.mode,
        compiled=args.compiled,
        warmup_steps=1 if args.smoke else args.warmup_steps,
        measure_steps=1 if args.smoke else args.measure_steps,
        precision=args.precision,
        device=args.device,
        campaign=args.campaign,
        output_root=args.output_root,
        vram_limit_gb=args.vram_limit_gb,
    )
    print(json.dumps(benchmark_single_config(config), indent=2))


if __name__ == "__main__":
    main()
