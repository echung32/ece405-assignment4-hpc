from __future__ import annotations

import argparse
import json
import math
import statistics
import timeit
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from cs336_systems.section1.flash_attention import (
    get_flashattention_autograd_function_pytorch,
    get_flashattention_autograd_function_triton,
    naive_scaled_dot_product_attention,
)

DEFAULT_BATCH_SIZE = 8
DEFAULT_D_MODEL_VALUES = (16, 32, 64, 128)
DEFAULT_SEQUENCE_LENGTH_VALUES = (256, 1024, 4096, 8192, 16384)
DEFAULT_WARMUP_STEPS = 10
DEFAULT_MEASURE_STEPS = 100


@dataclass(frozen=True)
class AttentionBenchmarkConfig:
    implementation: str
    batch_size: int
    d_model: int
    sequence_length: int
    warmup_steps: int
    measure_steps: int
    device: str
    dtype: str
    is_causal: bool
    campaign: str
    output_root: str


def parse_dtype(value: str) -> torch.dtype:
    if value == "fp32":
        return torch.float32
    if value == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {value}")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


def sanitize_name(value: str) -> str:
    return value.replace(":", "-").replace("/", "-")


def make_output_dir(config: AttentionBenchmarkConfig) -> Path:
    run_name = (
        f"{config.implementation}_seq{config.sequence_length}_d{config.d_model}_"
        f"bs{config.batch_size}_{config.dtype}_{'causal' if config.is_causal else 'full'}"
    )
    output_dir = Path(config.output_root) / config.campaign / sanitize_name(run_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def make_attention_callable(implementation: str):
    if implementation == "naive":
        return lambda q, k, v, is_causal: naive_scaled_dot_product_attention(q, k, v, is_causal=is_causal)[0]
    if implementation == "compiled_naive":
        compiled = torch.compile(
            lambda q, k, v, is_causal: naive_scaled_dot_product_attention(q, k, v, is_causal=is_causal)[0]
        )
        return compiled
    if implementation == "flash_pytorch":
        flash = get_flashattention_autograd_function_pytorch()
        return lambda q, k, v, is_causal: flash.apply(q, k, v, is_causal)
    if implementation == "flash_triton":
        flash = get_flashattention_autograd_function_triton()
        return lambda q, k, v, is_causal: flash.apply(q, k, v, is_causal)
    raise ValueError(f"Unsupported implementation: {implementation}")


def prepare_inputs(config: AttentionBenchmarkConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = torch.device(config.device)
    dtype = parse_dtype(config.dtype)
    q = torch.randn(
        config.batch_size,
        config.sequence_length,
        config.d_model,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    k = torch.randn(
        config.batch_size,
        config.sequence_length,
        config.d_model,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    v = torch.randn(
        config.batch_size,
        config.sequence_length,
        config.d_model,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    grad_output = torch.randn(
        config.batch_size,
        config.sequence_length,
        config.d_model,
        device=device,
        dtype=dtype,
    )
    return q, k, v, grad_output


def clear_gradients(*tensors: torch.Tensor) -> None:
    for tensor in tensors:
        tensor.grad = None


def benchmark_forward(
    attention_impl,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool,
    warmup_steps: int,
    measure_steps: int,
) -> list[float]:
    device = q.device
    with torch.no_grad():
        for _ in range(warmup_steps):
            attention_impl(q, k, v, is_causal)
            synchronize(device)

        timings: list[float] = []
        for _ in range(measure_steps):
            start_time = timeit.default_timer()
            attention_impl(q, k, v, is_causal)
            synchronize(device)
            timings.append(timeit.default_timer() - start_time)
    return timings


def benchmark_backward(
    attention_impl,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    is_causal: bool,
    warmup_steps: int,
    measure_steps: int,
) -> tuple[list[float], list[int], list[int]]:
    device = q.device
    allocated_bytes: list[int] = []
    reserved_bytes: list[int] = []

    for _ in range(warmup_steps):
        clear_gradients(q, k, v)
        output = attention_impl(q, k, v, is_causal)
        synchronize(device)
        output.backward(grad_output)
        synchronize(device)

    timings: list[float] = []
    for _ in range(measure_steps):
        clear_gradients(q, k, v)
        output = attention_impl(q, k, v, is_causal)
        synchronize(device)
        if device.type == "cuda":
            allocated_bytes.append(torch.cuda.memory_allocated(device=device))
            reserved_bytes.append(torch.cuda.memory_reserved(device=device))
        start_time = timeit.default_timer()
        output.backward(grad_output)
        synchronize(device)
        timings.append(timeit.default_timer() - start_time)

    return timings, allocated_bytes, reserved_bytes


def summarize_timings(values: list[float]) -> dict[str, float]:
    return {
        "mean_seconds": statistics.fmean(values),
        "stdev_seconds": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min_seconds": min(values),
        "max_seconds": max(values),
    }


def summarize_memory(values: list[int]) -> dict[str, float | None]:
    if not values:
        return {
            "mean_bytes": None,
            "max_bytes": None,
            "mean_gb": None,
            "max_gb": None,
        }
    return {
        "mean_bytes": statistics.fmean(values),
        "max_bytes": max(values),
        "mean_gb": statistics.fmean(values) / (1024**3),
        "max_gb": max(values) / (1024**3),
    }


def run_single_config(config: AttentionBenchmarkConfig) -> dict[str, object]:
    device = torch.device(config.device)
    if device.type != "cuda":
        raise ValueError("Attention benchmarking is intended for CUDA devices")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    output_dir = make_output_dir(config)
    attention_impl = make_attention_callable(config.implementation)
    q, k, v, grad_output = prepare_inputs(config)

    try:
        forward_timings = benchmark_forward(
            attention_impl,
            q,
            k,
            v,
            is_causal=config.is_causal,
            warmup_steps=config.warmup_steps,
            measure_steps=config.measure_steps,
        )
        backward_timings, backward_allocated, backward_reserved = benchmark_backward(
            attention_impl,
            q,
            k,
            v,
            grad_output,
            is_causal=config.is_causal,
            warmup_steps=config.warmup_steps,
            measure_steps=config.measure_steps,
        )
        result: dict[str, object] = {
            "config": asdict(config),
            "status": "ok",
            "forward": {
                **summarize_timings(forward_timings),
                "timings_seconds": forward_timings,
            },
            "backward": {
                **summarize_timings(backward_timings),
                "timings_seconds": backward_timings,
                "memory_allocated_before_backward": summarize_memory(backward_allocated),
                "memory_reserved_before_backward": summarize_memory(backward_reserved),
            },
            "attention_scores_bytes": (
                config.batch_size * (config.sequence_length**2) * torch.tensor([], dtype=parse_dtype(config.dtype)).element_size()
            ),
        }
    except RuntimeError as error:
        if "out of memory" not in str(error).lower():
            raise
        if device.type == "cuda":
            torch.cuda.empty_cache()
        result = {
            "config": asdict(config),
            "status": "oom",
            "error": str(error),
        }

    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark attention implementations for Sections 1.2 and 1.3")
    parser.add_argument(
        "--implementation",
        choices=("naive", "compiled_naive", "flash_pytorch", "flash_triton"),
        default="naive",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--d-model-values", type=int, nargs="+", default=DEFAULT_D_MODEL_VALUES)
    parser.add_argument("--sequence-length-values", type=int, nargs="+", default=DEFAULT_SEQUENCE_LENGTH_VALUES)
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument("--measure-steps", type=int, default=DEFAULT_MEASURE_STEPS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--is-causal", action="store_true")
    parser.add_argument("--campaign", default="default")
    parser.add_argument("--output-root", default="data/section1_2")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    d_model_values = [args.d_model_values[0]] if args.smoke else list(args.d_model_values)
    sequence_length_values = [args.sequence_length_values[0]] if args.smoke else list(args.sequence_length_values)
    warmup_steps = 1 if args.smoke else args.warmup_steps
    measure_steps = 1 if args.smoke else args.measure_steps

    results = []
    for d_model in d_model_values:
        for sequence_length in sequence_length_values:
            config = AttentionBenchmarkConfig(
                implementation=args.implementation,
                batch_size=args.batch_size,
                d_model=d_model,
                sequence_length=sequence_length,
                warmup_steps=warmup_steps,
                measure_steps=measure_steps,
                device=args.device,
                dtype=args.dtype,
                is_causal=args.is_causal,
                campaign=args.campaign,
                output_root=args.output_root,
            )
            results.append(run_single_config(config))

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
