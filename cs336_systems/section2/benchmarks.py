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
    setup_process_group,
    summarize_timings,
    synchronize_device,
)
from cs336_systems.section2.ddp import DDPBucketed, DDPIndividualParameters
from cs336_systems.section1.benchmarking_script import DEFAULT_VOCAB_SIZE, MODEL_SPECS, create_model, generate_batch

DEFAULT_MASTER_PORT = 29610


@dataclass(frozen=True)
class AllReduceBenchmarkConfig:
    backend: str
    device_type: str
    world_size: int
    size_mb: int
    warmup_steps: int
    measure_steps: int
    master_port: int
    campaign: str
    output_root: str


@dataclass(frozen=True)
class DDPBenchmarkConfig:
    implementation: str
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


def _allreduce_worker(rank: int, config: AllReduceBenchmarkConfig, result_path: str) -> None:
    device = setup_process_group(rank, config.world_size, config.backend, config.device_type, config.master_port)
    try:
        num_elements = (config.size_mb * 1024 * 1024) // torch.tensor([], dtype=torch.float32).element_size()
        tensor = torch.randn(num_elements, device=device, dtype=torch.float32)

        for _ in range(config.warmup_steps):
            dist.all_reduce(tensor, async_op=False)
            synchronize_device(device)

        timings: list[float] = []
        for _ in range(config.measure_steps):
            tensor.uniform_()
            start = timeit.default_timer()
            dist.all_reduce(tensor, async_op=False)
            synchronize_device(device)
            timings.append(timeit.default_timer() - start)

        aggregated_timings = aggregate_step_times(timings)
        if rank == 0:
            _write_result(
                Path(result_path),
                {
                    "config": asdict(config),
                    "status": "ok",
                    "timings_seconds": aggregated_timings,
                    **summarize_timings(aggregated_timings),
                },
            )
    finally:
        cleanup_process_group()


def run_allreduce(config: AllReduceBenchmarkConfig) -> dict[str, object]:
    run_name = f"{config.backend}_{config.device_type}_ws{config.world_size}_{config.size_mb}mb"
    output_dir = make_output_dir(config.output_root, config.campaign, run_name)
    result_path = output_dir / "result.json"
    preflight_error = _preflight_cuda(config.device_type, config.world_size)
    if preflight_error is not None:
        result = {"config": asdict(config), "status": "skipped", "error": preflight_error}
        _write_result(result_path, result)
        return result

    mp.spawn(_allreduce_worker, args=(config, str(result_path)), nprocs=config.world_size, join=True)
    return json.loads(result_path.read_text())


def _broadcast_parameters(module: torch.nn.Module) -> None:
    for parameter in module.state_dict().values():
        dist.broadcast(parameter, src=0)


def _sync_gradients_naive(module: torch.nn.Module, world_size: int) -> float:
    start = timeit.default_timer()
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        dist.all_reduce(parameter.grad, async_op=False)
        parameter.grad.div_(world_size)
    return timeit.default_timer() - start


def _sync_gradients_flat(module: torch.nn.Module, world_size: int) -> float:
    gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
    if not gradients:
        return 0.0

    start = timeit.default_timer()
    flat_gradient = torch._utils._flatten_dense_tensors(gradients)
    dist.all_reduce(flat_gradient, async_op=False)
    flat_gradient.div_(world_size)
    synchronized = torch._utils._unflatten_dense_tensors(flat_gradient, gradients)
    for gradient, synchronized_gradient in zip(gradients, synchronized):
        gradient.copy_(synchronized_gradient)
    return timeit.default_timer() - start


def _ddp_worker(rank: int, config: DDPBenchmarkConfig, result_path: str) -> None:
    device = setup_process_group(rank, config.world_size, config.backend, config.device_type, config.master_port)
    try:
        torch.manual_seed(42)
        random.seed(42)
        base_model = create_model(MODEL_SPECS[config.model_size], config.context_length, DEFAULT_VOCAB_SIZE, device)

        if config.implementation == "naive":
            model = base_model
            _broadcast_parameters(model)
        elif config.implementation == "flat":
            model = base_model
            _broadcast_parameters(model)
        elif config.implementation == "overlap_individual":
            model = DDPIndividualParameters(base_model)
        elif config.implementation == "overlap_bucketed":
            model = DDPBucketed(base_model, bucket_size_mb=config.bucket_size_mb)
        else:
            raise ValueError(f"Unsupported DDP implementation: {config.implementation}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        step_timings: list[float] = []
        communication_timings: list[float] = []

        total_steps = config.warmup_steps + config.measure_steps
        for step_idx in range(total_steps):
            if hasattr(model, "start_gradient_synchronization"):
                model.start_gradient_synchronization()
            optimizer.zero_grad(set_to_none=True)

            torch.manual_seed(1000 + step_idx * config.world_size + rank)
            inputs, targets = generate_batch(config.batch_size, config.context_length, DEFAULT_VOCAB_SIZE, device)

            start = timeit.default_timer()
            logits = model(inputs)
            loss = cross_entropy(logits, targets)
            loss.backward()

            if config.implementation == "naive":
                communication_seconds = _sync_gradients_naive(model, config.world_size)
            elif config.implementation == "flat":
                communication_seconds = _sync_gradients_flat(model, config.world_size)
            else:
                comm_start = timeit.default_timer()
                model.finish_gradient_synchronization()
                communication_seconds = timeit.default_timer() - comm_start

            optimizer.step()
            synchronize_device(device)
            step_seconds = timeit.default_timer() - start

            if step_idx >= config.warmup_steps:
                step_timings.append(step_seconds)
                communication_timings.append(communication_seconds)

        aggregated_step_timings = aggregate_step_times(step_timings)
        aggregated_comm_timings = aggregate_step_times(communication_timings)
        if rank == 0:
            _write_result(
                Path(result_path),
                {
                    "config": asdict(config),
                    "model_spec": asdict(MODEL_SPECS[config.model_size]),
                    "status": "ok",
                    "step_timings_seconds": aggregated_step_timings,
                    "communication_timings_seconds": aggregated_comm_timings,
                    "step": summarize_timings(aggregated_step_timings),
                    "communication": summarize_timings(aggregated_comm_timings),
                },
            )
    finally:
        cleanup_process_group()


def run_ddp(config: DDPBenchmarkConfig) -> dict[str, object]:
    run_name = (
        f"{config.implementation}_{config.backend}_{config.device_type}_ws{config.world_size}_"
        f"{config.model_size}_ctx{config.context_length}_bs{config.batch_size}"
    )
    if config.implementation == "overlap_bucketed":
        run_name += f"_bucket{config.bucket_size_mb:g}mb"
    output_dir = make_output_dir(config.output_root, config.campaign, run_name)
    result_path = output_dir / "result.json"
    preflight_error = _preflight_cuda(config.device_type, config.world_size)
    if preflight_error is not None:
        result = {"config": asdict(config), "status": "skipped", "error": preflight_error}
        _write_result(result_path, result)
        return result

    mp.spawn(_ddp_worker, args=(config, str(result_path)), nprocs=config.world_size, join=True)
    return json.loads(result_path.read_text())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Section 2 communication and DDP benchmarks")
    subparsers = parser.add_subparsers(dest="command", required=True)

    allreduce = subparsers.add_parser("allreduce")
    allreduce.add_argument("--backend", choices=("gloo", "nccl"), required=True)
    allreduce.add_argument("--device-type", choices=("cpu", "cuda"), required=True)
    allreduce.add_argument("--world-size", type=int, default=2)
    allreduce.add_argument("--size-mb", type=int, default=1)
    allreduce.add_argument("--warmup-steps", type=int, default=5)
    allreduce.add_argument("--measure-steps", type=int, default=10)
    allreduce.add_argument("--master-port", type=int, default=DEFAULT_MASTER_PORT)
    allreduce.add_argument("--campaign", default="default")
    allreduce.add_argument("--output-root", default="data/section2")

    ddp = subparsers.add_parser("ddp")
    ddp.add_argument("--implementation", choices=("naive", "flat", "overlap_individual", "overlap_bucketed"), required=True)
    ddp.add_argument("--backend", choices=("gloo", "nccl"), required=True)
    ddp.add_argument("--device-type", choices=("cpu", "cuda"), required=True)
    ddp.add_argument("--world-size", type=int, default=2)
    ddp.add_argument("--model-size", choices=sorted(MODEL_SPECS.keys()), default="xl")
    ddp.add_argument("--context-length", type=int, default=128)
    ddp.add_argument("--batch-size", type=int, default=4)
    ddp.add_argument("--warmup-steps", type=int, default=5)
    ddp.add_argument("--measure-steps", type=int, default=10)
    ddp.add_argument("--bucket-size-mb", type=float, default=25.0)
    ddp.add_argument("--master-port", type=int, default=DEFAULT_MASTER_PORT + 10)
    ddp.add_argument("--campaign", default="default")
    ddp.add_argument("--output-root", default="data/section2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "allreduce":
        result = run_allreduce(
            AllReduceBenchmarkConfig(
                backend=args.backend,
                device_type=args.device_type,
                world_size=args.world_size,
                size_mb=args.size_mb,
                warmup_steps=args.warmup_steps,
                measure_steps=args.measure_steps,
                master_port=args.master_port,
                campaign=args.campaign,
                output_root=args.output_root,
            )
        )
    else:
        result = run_ddp(
            DDPBenchmarkConfig(
                implementation=args.implementation,
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