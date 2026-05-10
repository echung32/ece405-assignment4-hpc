from __future__ import annotations

import argparse
import json
import random
import timeit
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_basics.nn_utils import cross_entropy
from cs336_systems.benchmark_utils import (
    aggregate_step_times,
    cleanup_process_group,
    gradient_nbytes,
    optimizer_state_nbytes,
    parameter_nbytes,
    setup_process_group,
    summarize_timings,
    synchronize_device,
)
from cs336_systems.section2.ddp import DDPBucketed
from cs336_systems.section1.benchmarking_script import DEFAULT_VOCAB_SIZE, MODEL_SPECS, create_model, generate_batch
from cs336_systems.section3.sharded_optimizer import ShardedOptimizer

DEFAULT_MASTER_PORT = 29710


@dataclass(frozen=True)
class Section3BenchmarkConfig:
    mode: str
    optimizer_impl: str
    backend: str
    device_type: str
    world_size: int
    model_size: str
    context_length: int
    batch_size: int
    warmup_steps: int
    measure_steps: int
    bucket_size_mb: float
    master_port: int
    campaign: str
    output_root: str


def sanitize_name(value: str) -> str:
    return value.replace(":", "-").replace("/", "-")


def make_output_dir(output_root: str, campaign: str, run_name: str) -> Path:
    output_dir = Path(output_root) / campaign / sanitize_name(run_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _preflight_cuda(device_type: str, world_size: int) -> str | None:
    if device_type != "cuda":
        return None
    if not torch.cuda.is_available():
        return "CUDA is not available"
    if torch.cuda.device_count() < world_size:
        return f"Requested world_size={world_size} CUDA ranks but only found {torch.cuda.device_count()} GPUs"
    return None


def _write_result(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2))


def _memory_snapshot(label: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> dict[str, object]:
    parameters = list(model.parameters())
    snapshot = {
        "label": label,
        "parameter_bytes": parameter_nbytes(parameters),
        "gradient_bytes": gradient_nbytes(parameters),
        "optimizer_state_bytes": optimizer_state_nbytes(optimizer),
    }
    if device.type == "cuda":
        snapshot.update(
            {
                "cuda_allocated_bytes": torch.cuda.memory_allocated(device=device),
                "cuda_reserved_bytes": torch.cuda.memory_reserved(device=device),
                "cuda_max_allocated_bytes": torch.cuda.max_memory_allocated(device=device),
            }
        )
    return snapshot


def _run_training_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int,
    context_length: int,
    step_seed: int,
    collect_memory: bool,
) -> tuple[float, dict[str, object] | None, dict[str, object] | None]:
    if hasattr(model, "start_gradient_synchronization"):
        model.start_gradient_synchronization()
    optimizer.zero_grad(set_to_none=True)

    torch.manual_seed(step_seed)
    random.seed(step_seed)
    inputs, targets = generate_batch(batch_size, context_length, DEFAULT_VOCAB_SIZE, device)

    start = timeit.default_timer()
    logits = model(inputs)
    loss = cross_entropy(logits, targets)
    loss.backward()
    model.finish_gradient_synchronization()
    pre_step_snapshot = _memory_snapshot("pre_step", model, optimizer, device) if collect_memory else None
    optimizer.step()
    synchronize_device(device)
    post_step_snapshot = _memory_snapshot("post_step", model, optimizer, device) if collect_memory else None
    return timeit.default_timer() - start, pre_step_snapshot, post_step_snapshot


def _worker(rank: int, config: Section3BenchmarkConfig, result_path: str) -> None:
    device = setup_process_group(rank, config.world_size, config.backend, config.device_type, config.master_port)
    try:
        torch.manual_seed(42)
        random.seed(42)
        base_model = create_model(MODEL_SPECS[config.model_size], config.context_length, DEFAULT_VOCAB_SIZE, device)
        model = DDPBucketed(base_model, bucket_size_mb=config.bucket_size_mb)
        if config.optimizer_impl == "standard":
            optimizer: torch.optim.Optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        elif config.optimizer_impl == "sharded":
            optimizer = ShardedOptimizer(model.parameters(), torch.optim.AdamW, lr=1e-3)
        else:
            raise ValueError(f"Unsupported optimizer implementation: {config.optimizer_impl}")

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device=device)

        init_snapshot = _memory_snapshot("post_init", model, optimizer, device)

        timings: list[float] = []
        pre_step_snapshot: dict[str, object] | None = None
        post_step_snapshot: dict[str, object] | None = None
        total_steps = config.warmup_steps + config.measure_steps
        for step_idx in range(total_steps):
            step_seconds, maybe_pre_step, maybe_post_step = _run_training_step(
                model,
                optimizer,
                device,
                config.batch_size,
                config.context_length,
                step_seed=2000 + step_idx * config.world_size + rank,
                collect_memory=config.mode == "memory" and step_idx == total_steps - 1,
            )
            if step_idx >= config.warmup_steps:
                timings.append(step_seconds)
            if maybe_pre_step is not None:
                pre_step_snapshot = maybe_pre_step
            if maybe_post_step is not None:
                post_step_snapshot = maybe_post_step

        aggregated_timings = aggregate_step_times(timings)
        if rank == 0:
            payload: dict[str, object] = {
                "config": asdict(config),
                "model_spec": asdict(MODEL_SPECS[config.model_size]),
                "status": "ok",
                "timings_seconds": aggregated_timings,
                **summarize_timings(aggregated_timings),
            }
            if config.mode == "memory":
                payload["memory"] = {
                    "post_init": init_snapshot,
                    "pre_step": pre_step_snapshot,
                    "post_step": post_step_snapshot,
                }
            _write_result(Path(result_path), payload)
    finally:
        cleanup_process_group()


def run_benchmark(config: Section3BenchmarkConfig) -> dict[str, object]:
    run_name = (
        f"{config.mode}_{config.optimizer_impl}_{config.backend}_{config.device_type}_ws{config.world_size}_"
        f"{config.model_size}_ctx{config.context_length}_bs{config.batch_size}"
    )
    output_dir = make_output_dir(config.output_root, config.campaign, run_name)
    result_path = output_dir / "result.json"
    preflight_error = _preflight_cuda(config.device_type, config.world_size)
    if preflight_error is not None:
        result = {"config": asdict(config), "status": "skipped", "error": preflight_error}
        _write_result(result_path, result)
        return result

    mp.spawn(_worker, args=(config, str(result_path)), nprocs=config.world_size, join=True)
    return json.loads(result_path.read_text())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Section 3 timing and memory benchmarks")
    parser.add_argument("mode", choices=("timing", "memory"))
    parser.add_argument("--optimizer-impl", choices=("standard", "sharded"), required=True)
    parser.add_argument("--backend", choices=("gloo", "nccl"), required=True)
    parser.add_argument("--device-type", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--model-size", choices=sorted(MODEL_SPECS.keys()), default="xl")
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--measure-steps", type=int, default=5)
    parser.add_argument("--bucket-size-mb", type=float, default=25.0)
    parser.add_argument("--master-port", type=int, default=DEFAULT_MASTER_PORT)
    parser.add_argument("--campaign", default="default")
    parser.add_argument("--output-root", default="data/section3")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_benchmark(
        Section3BenchmarkConfig(
            mode=args.mode,
            optimizer_impl=args.optimizer_impl,
            backend=args.backend,
            device_type=args.device_type,
            world_size=args.world_size,
            model_size=args.model_size,
            context_length=args.context_length,
            batch_size=args.batch_size,
            warmup_steps=args.warmup_steps,
            measure_steps=args.measure_steps,
            bucket_size_mb=args.bucket_size_mb,
            master_port=args.master_port,
            campaign=args.campaign,
            output_root=args.output_root,
        )
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()