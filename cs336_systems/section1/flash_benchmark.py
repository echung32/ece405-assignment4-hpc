from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import triton

from cs336_systems.section1.flash_attention import (
    get_flashattention_autograd_function_triton,
    naive_scaled_dot_product_attention,
)

DEFAULT_BATCH_SIZE = 1
DEFAULT_D_MODEL_VALUES = (16, 32, 64, 128)
DEFAULT_SEQUENCE_LENGTH_VALUES = tuple(2**power for power in range(7, 17))
DEFAULT_WARMUP_MS = 100
DEFAULT_REP_MS = 200


@dataclass(frozen=True)
class FlashBenchmarkConfig:
    implementation: str
    batch_size: int
    d_model: int
    sequence_length: int
    warmup_ms: int
    rep_ms: int
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


def sanitize_name(value: str) -> str:
    return value.replace(":", "-").replace("/", "-")


def make_output_dir(config: FlashBenchmarkConfig) -> Path:
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
    if implementation == "flash_triton":
        flash = get_flashattention_autograd_function_triton()
        return lambda q, k, v, is_causal: flash.apply(q, k, v, is_causal)
    raise ValueError(f"Unsupported implementation: {implementation}")


def prepare_inputs(config: FlashBenchmarkConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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


def benchmark_forward(attention_impl, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, is_causal: bool, warmup_ms: int, rep_ms: int) -> float:
    with torch.no_grad():
        return float(triton.testing.do_bench(lambda: attention_impl(q, k, v, is_causal), warmup=warmup_ms, rep=rep_ms))


def benchmark_backward(attention_impl, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, grad_output: torch.Tensor, *, is_causal: bool, warmup_ms: int, rep_ms: int) -> float:
    output = attention_impl(q, k, v, is_causal)

    def backward_only() -> None:
        clear_gradients(q, k, v)
        output.backward(grad_output, retain_graph=True)

    return float(triton.testing.do_bench(backward_only, warmup=warmup_ms, rep=rep_ms))


def benchmark_end_to_end(attention_impl, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, grad_output: torch.Tensor, *, is_causal: bool, warmup_ms: int, rep_ms: int) -> float:
    def forward_backward() -> None:
        clear_gradients(q, k, v)
        output = attention_impl(q, k, v, is_causal)
        output.backward(grad_output)

    return float(triton.testing.do_bench(forward_backward, warmup=warmup_ms, rep=rep_ms))


def run_single_config(config: FlashBenchmarkConfig) -> dict[str, object]:
    device = torch.device(config.device)
    if device.type != "cuda":
        raise ValueError("Flash benchmarking is intended for CUDA devices")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    output_dir = make_output_dir(config)
    attention_impl = make_attention_callable(config.implementation)
    q, k, v, grad_output = prepare_inputs(config)

    try:
        result: dict[str, object] = {
            "config": asdict(config),
            "status": "ok",
            "forward_ms": benchmark_forward(
                attention_impl,
                q,
                k,
                v,
                is_causal=config.is_causal,
                warmup_ms=config.warmup_ms,
                rep_ms=config.rep_ms,
            ),
            "backward_ms": benchmark_backward(
                attention_impl,
                q,
                k,
                v,
                grad_output,
                is_causal=config.is_causal,
                warmup_ms=config.warmup_ms,
                rep_ms=config.rep_ms,
            ),
            "end_to_end_ms": benchmark_end_to_end(
                attention_impl,
                q,
                k,
                v,
                grad_output,
                is_causal=config.is_causal,
                warmup_ms=config.warmup_ms,
                rep_ms=config.rep_ms,
            ),
        }
    except RuntimeError as error:
        if "out of memory" not in str(error).lower():
            raise
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
    parser = argparse.ArgumentParser(description="Benchmark FlashAttention vs naive attention for Section 1.3")
    parser.add_argument("--implementation", choices=("naive", "flash_triton"), default="flash_triton")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--d-model-values", type=int, nargs="+", default=DEFAULT_D_MODEL_VALUES)
    parser.add_argument("--sequence-length-values", type=int, nargs="+", default=DEFAULT_SEQUENCE_LENGTH_VALUES)
    parser.add_argument("--warmup-ms", type=int, default=DEFAULT_WARMUP_MS)
    parser.add_argument("--rep-ms", type=int, default=DEFAULT_REP_MS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--is-causal", action="store_true")
    parser.add_argument("--campaign", default="default")
    parser.add_argument("--output-root", default="data/section1_3")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    d_model_values = [args.d_model_values[0]] if args.smoke else list(args.d_model_values)
    sequence_length_values = [args.sequence_length_values[0]] if args.smoke else list(args.sequence_length_values)
    warmup_ms = 25 if args.smoke else args.warmup_ms
    rep_ms = 50 if args.smoke else args.rep_ms

    results = []
    for d_model in d_model_values:
        for sequence_length in sequence_length_values:
            config = FlashBenchmarkConfig(
                implementation=args.implementation,
                batch_size=args.batch_size,
                d_model=d_model,
                sequence_length=sequence_length,
                warmup_ms=warmup_ms,
                rep_ms=rep_ms,
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