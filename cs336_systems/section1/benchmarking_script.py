from __future__ import annotations

import argparse
import json
import math
import statistics
import timeit
from contextlib import ExitStack, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.cuda.nvtx as nvtx

import cs336_basics.model as basics_model
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy, softmax
from cs336_basics.optimizer import AdamW

DEFAULT_BATCH_SIZE = 4
DEFAULT_CONTEXT_LENGTH = 128
DEFAULT_MEASURE_STEPS = 10
DEFAULT_WARMUP_STEPS = 5
DEFAULT_VOCAB_SIZE = 10_000
DEFAULT_ROPE_THETA = 10_000.0
DEFAULT_VRAM_LIMIT_GB = 40.0


@dataclass(frozen=True)
class ModelSpec:
    name: str
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int


MODEL_SPECS: dict[str, ModelSpec] = {
    "small": ModelSpec("small", d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": ModelSpec("medium", d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": ModelSpec("large", d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": ModelSpec("xl", d_model=1600, d_ff=6400, num_layers=48, num_heads=25),
    "2.7b": ModelSpec("2.7b", d_model=2560, d_ff=10_240, num_layers=32, num_heads=32),
}


@dataclass(frozen=True)
class RunConfig:
    model_size: str
    context_length: int
    batch_size: int
    mode: str
    warmup_steps: int
    measure_steps: int
    precision: str
    device: str
    seed: int
    vram_limit_gb: float
    profile_memory: bool
    annotate_attention: bool
    campaign: str
    output_root: str


def annotated_scaled_dot_product_attention(*args, **kwargs) -> torch.Tensor:
    if args:
        if len(args) not in {3, 4}:
            raise TypeError(f"Expected 3 or 4 positional arguments, got {len(args)}")
        Q, K, V = args[:3]
        mask = args[3] if len(args) == 4 else None
    else:
        Q = kwargs.pop("Q", kwargs.pop("q", None))
        K = kwargs.pop("K", kwargs.pop("k", None))
        V = kwargs.pop("V", kwargs.pop("v", None))
        mask = kwargs.pop("mask", None)
        if kwargs:
            unexpected = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unexpected keyword arguments: {unexpected}")
        if Q is None or K is None or V is None:
            raise TypeError("Q, K, and V must be provided")

    d_k = K.shape[-1]
    with nvtx.range("computing attention scores"):
        attention_scores = torch.einsum("...qd,...kd->...qk", Q, K) / math.sqrt(d_k)
        if mask is not None:
            attention_scores = torch.where(mask, attention_scores, float("-inf"))

    with nvtx.range("computing softmax"):
        attention_weights = softmax(attention_scores, dim=-1)

    with nvtx.range("final matmul"):
        return torch.einsum("...qk,...kd->...qd", attention_weights, V)


def sanitize_name(value: str) -> str:
    return value.replace(":", "-").replace("/", "-")


def make_output_dir(config: RunConfig) -> Path:
    run_name = (
        f"{sanitize_name(config.model_size)}_"
        f"ctx{config.context_length}_"
        f"bs{config.batch_size}_"
        f"{config.mode}_"
        f"{config.precision}"
    )
    output_dir = Path(config.output_root) / config.campaign / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def create_model(spec: ModelSpec, context_length: int, vocab_size: int, device: torch.device) -> BasicsTransformerLM:
    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=spec.d_model,
        num_layers=spec.num_layers,
        num_heads=spec.num_heads,
        d_ff=spec.d_ff,
        rope_theta=DEFAULT_ROPE_THETA,
    )
    return model.to(device=device)


def generate_batch(
    batch_size: int,
    context_length: int,
    vocab_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.randint(0, vocab_size, (batch_size, context_length), device=device, dtype=torch.long)
    targets = torch.randint(0, vocab_size, (batch_size, context_length), device=device, dtype=torch.long)
    return inputs, targets


def get_autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    if precision != "bf16":
        raise ValueError(f"Unsupported precision: {precision}")
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


def get_reserved_memory(device: torch.device) -> int:
    if device.type != "cuda":
        return 0
    return torch.cuda.memory_reserved(device=device)


def get_peak_reserved_memory(device: torch.device) -> int:
    if device.type != "cuda":
        return 0
    return torch.cuda.max_memory_reserved(device=device)


def get_total_device_memory_used(device: torch.device) -> int:
    if device.type != "cuda":
        return 0
    # Use process-level reserved memory so the guard is not affected by other
    # users' processes that may be sharing the same GPU.
    return torch.cuda.memory_reserved(device=device)


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device=device)


def maybe_raise_for_vram_limit(device: torch.device, vram_limit_gb: float) -> None:
    if device.type != "cuda":
        return
    limit_bytes = int(vram_limit_gb * (1024**3))
    used_bytes = get_total_device_memory_used(device)
    if used_bytes > limit_bytes:
        used_gb = used_bytes / (1024**3)
        raise RuntimeError(
            f"VRAM guard exceeded on {device}: total_used={used_gb:.2f} GB, limit={vram_limit_gb:.2f} GB"
        )


def run_model_step(
    model: BasicsTransformerLM,
    optimizer: AdamW | None,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    config: RunConfig,
) -> dict[str, float | None]:
    device = inputs.device
    loss_value: float | None = None

    if config.mode != "forward":
        model.zero_grad(set_to_none=True)

    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    with ExitStack() as stack:
        if config.mode != "forward":
            stack.enter_context(nvtx.range("backprop_step"))
        else:
            stack.enter_context(nvtx.range("forward_step"))
            stack.enter_context(torch.no_grad())
        stack.enter_context(get_autocast_context(device, config.precision))

        with nvtx.range("forward"):
            logits = model(inputs)

        if config.mode != "forward":
            with nvtx.range("loss"):
                loss = cross_entropy(logits, targets)
                loss_value = float(loss.detach().item())

            with nvtx.range("backward"):
                loss.backward()

            if config.mode == "train_step":
                if optimizer is None:
                    raise ValueError("Optimizer is required for train_step mode")
                with nvtx.range("optimizer_step"):
                    optimizer.step()

    synchronize(device)
    maybe_raise_for_vram_limit(device, config.vram_limit_gb)
    return {"loss": loss_value}


def install_attention_annotations(enabled: bool):
    original_attention = basics_model.scaled_dot_product_attention
    if enabled:
        basics_model.scaled_dot_product_attention = annotated_scaled_dot_product_attention
    return original_attention


def restore_attention_annotations(original_attention) -> None:
    basics_model.scaled_dot_product_attention = original_attention


def get_supported_cuda_arches() -> set[int]:
    supported_arches: set[int] = set()
    for arch in torch.cuda.get_arch_list():
        if not arch.startswith("sm_"):
            continue
        supported_arches.add(int(arch.removeprefix("sm_")))
    return supported_arches


def ensure_supported_device(device: torch.device) -> None:
    if device.type != "cuda":
        return
    capability = torch.cuda.get_device_capability(device=device)
    arch_code = int(f"{capability[0]}{capability[1]}")
    supported_arches = get_supported_cuda_arches()
    if arch_code not in supported_arches:
        supported_list = ", ".join(f"sm_{arch}" for arch in sorted(supported_arches))
        raise RuntimeError(
            "The installed PyTorch build does not support the selected CUDA device. "
            f"Device capability is sm_{arch_code}, but this build only supports: {supported_list}."
        )


def capture_memory_snapshot(snapshot_path: Path) -> None:
    torch.cuda.memory._dump_snapshot(str(snapshot_path))


def run_benchmark(config: RunConfig) -> dict[str, object]:
    if config.model_size not in MODEL_SPECS:
        raise ValueError(f"Unknown model size: {config.model_size}")

    device = torch.device(config.device)
    if device.type != "cuda":
        raise ValueError("Section 1.1 harness requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    ensure_supported_device(device)

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    output_dir = make_output_dir(config)
    spec = MODEL_SPECS[config.model_size]
    model = create_model(spec, config.context_length, DEFAULT_VOCAB_SIZE, device)
    optimizer = AdamW(model.parameters(), lr=1e-3) if config.mode == "train_step" else None
    inputs, targets = generate_batch(config.batch_size, config.context_length, DEFAULT_VOCAB_SIZE, device)

    original_attention = install_attention_annotations(config.annotate_attention)

    reset_peak_memory(device)
    timings: list[float] = []
    snapshot_path: Path | None = None
    peak_device_used_bytes = 0

    try:
        with nvtx.range("warmup"):
            for _ in range(config.warmup_steps):
                run_model_step(model, optimizer, inputs, targets, config)

        if config.profile_memory:
            torch.cuda.memory._record_memory_history(max_entries=1_000_000)
            snapshot_path = output_dir / "memory_snapshot.pickle"

        reset_peak_memory(device)
        with nvtx.range("measurement"):
            for step_index in range(config.measure_steps):
                start_time = timeit.default_timer()
                metrics = run_model_step(model, optimizer, inputs, targets, config)
                elapsed = timeit.default_timer() - start_time
                timings.append(elapsed)
                peak_device_used_bytes = max(peak_device_used_bytes, get_total_device_memory_used(device))
                if config.profile_memory and step_index == 0 and snapshot_path is not None:
                    capture_memory_snapshot(snapshot_path)

        peak_reserved_bytes = get_peak_reserved_memory(device)
        result = {
            "config": asdict(config),
            "model_spec": asdict(spec),
            "timings_seconds": timings,
            "mean_seconds": statistics.fmean(timings),
            "stdev_seconds": statistics.stdev(timings) if len(timings) > 1 else 0.0,
            "peak_reserved_bytes": peak_reserved_bytes,
            "peak_reserved_gb": peak_reserved_bytes / (1024**3),
            "peak_device_used_bytes": peak_device_used_bytes,
            "peak_device_used_gb": peak_device_used_bytes / (1024**3),
            "snapshot_path": str(snapshot_path) if snapshot_path is not None else None,
            "parameter_count": model.get_num_params(non_embedding=False),
            "loss": metrics["loss"] if config.mode != "forward" else None,
        }
    finally:
        if config.profile_memory:
            torch.cuda.memory._record_memory_history(enabled=None)
        restore_attention_annotations(original_attention)

    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2))
    return result


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description="Section 1.1 benchmarking and profiling harness")
    parser.add_argument("--model-size", choices=sorted(MODEL_SPECS.keys()), default="small")
    parser.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT_LENGTH)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--mode",
        choices=("forward", "forward_backward", "train_step"),
        default="forward",
    )
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument("--measure-steps", type=int, default=DEFAULT_MEASURE_STEPS)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vram-limit-gb", type=float, default=DEFAULT_VRAM_LIMIT_GB)
    parser.add_argument("--profile-memory", action="store_true")
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
        profile_memory=args.profile_memory,
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