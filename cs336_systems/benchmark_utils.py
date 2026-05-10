from __future__ import annotations

import os
import statistics
from collections.abc import Iterable

import torch
import torch.distributed as dist


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


def setup_process_group(
    rank: int,
    world_size: int,
    backend: str,
    device_type: str,
    master_port: int,
) -> torch.device:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)

    if device_type == "cuda":
        local_rank = rank
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        # Disable NCCL direct P2P access (PHB topology hangs without this)
        os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    else:
        device = torch.device("cpu")

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return device


def cleanup_process_group() -> None:
    dist.barrier()
    dist.destroy_process_group()


def summarize_timings(values: list[float]) -> dict[str, float]:
    return {
        "mean_seconds": statistics.fmean(values),
        "stdev_seconds": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min_seconds": min(values),
        "max_seconds": max(values),
    }


def aggregate_step_times(local_values: list[float]) -> list[float]:
    gathered: list[list[float] | None] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, local_values)
    return [max(rank_values[step] for rank_values in gathered if rank_values is not None) for step in range(len(local_values))]


def unique_tensors(parameters: Iterable[torch.nn.Parameter]) -> list[torch.nn.Parameter]:
    seen: set[int] = set()
    unique_parameters: list[torch.nn.Parameter] = []
    for parameter in parameters:
        parameter_id = id(parameter)
        if parameter_id in seen:
            continue
        seen.add(parameter_id)
        unique_parameters.append(parameter)
    return unique_parameters


def tensor_nbytes(tensor: torch.Tensor | None) -> int:
    if tensor is None:
        return 0
    return tensor.numel() * tensor.element_size()


def optimizer_state_nbytes(optimizer: torch.optim.Optimizer) -> int:
    if hasattr(optimizer, "_local_optimizer") and getattr(optimizer, "_local_optimizer") is not None:
        return optimizer_state_nbytes(getattr(optimizer, "_local_optimizer"))

    total = 0
    for state in optimizer.state.values():
        if isinstance(state, dict):
            for value in state.values():
                if torch.is_tensor(value):
                    total += tensor_nbytes(value)
    return total


def parameter_nbytes(parameters: Iterable[torch.nn.Parameter]) -> int:
    return sum(tensor_nbytes(parameter.data) for parameter in unique_tensors(parameters))


def gradient_nbytes(parameters: Iterable[torch.nn.Parameter]) -> int:
    return sum(tensor_nbytes(parameter.grad) for parameter in unique_tensors(parameters))