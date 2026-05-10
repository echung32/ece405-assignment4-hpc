from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import torch
import torch.distributed as dist


def _iter_unique_parameters(module: torch.nn.Module) -> Iterable[torch.nn.Parameter]:
    seen: set[int] = set()
    for parameter in module.parameters():
        parameter_id = id(parameter)
        if parameter_id in seen:
            continue
        seen.add(parameter_id)
        yield parameter


class DDPIndividualParameters(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized before constructing DDPIndividualParameters")

        self.module = module
        self.world_size = dist.get_world_size()
        self._pending_handles: dict[int, dist.Work] = {}

        self._register_parameter_hooks()
        self._broadcast_module_state()

    def _broadcast_module_state(self) -> None:
        for tensor in self.module.state_dict().values():
            dist.broadcast(tensor, src=0)

    def _register_parameter_hooks(self) -> None:
        for parameter in _iter_unique_parameters(self.module):
            if not parameter.requires_grad:
                continue
            parameter.register_post_accumulate_grad_hook(self._make_grad_sync_hook(parameter))

    def _make_grad_sync_hook(self, parameter: torch.nn.Parameter):
        parameter_id = id(parameter)

        def hook(_parameter: torch.nn.Parameter) -> None:
            if parameter.grad is None or parameter_id in self._pending_handles:
                return
            parameter.grad.div_(self.world_size)
            self._pending_handles[parameter_id] = dist.all_reduce(parameter.grad, async_op=True)

        return hook

    def finish_gradient_synchronization(self) -> None:
        for handle in self._pending_handles.values():
            handle.wait()
        self._pending_handles.clear()

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)


@dataclass
class _GradientBucket:
    parameters: tuple[torch.nn.Parameter, ...]
    ready_parameter_ids: set[int] = field(default_factory=set)
    flat_gradient: torch.Tensor | None = None
    handle: dist.Work | None = None

    def reset(self) -> None:
        self.ready_parameter_ids.clear()
        self.flat_gradient = None
        self.handle = None


class DDPBucketed(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float | None):
        super().__init__()
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized before constructing DDPBucketed")

        self.module = module
        self.world_size = dist.get_world_size()
        self.bucket_size_bytes = None if bucket_size_mb is None else max(1, int(bucket_size_mb * 1024 * 1024))
        self.buckets = self._build_buckets()
        self._parameter_to_bucket = {
            id(parameter): bucket
            for bucket in self.buckets
            for parameter in bucket.parameters
        }

        self._register_parameter_hooks()
        self._broadcast_module_state()

    def _broadcast_module_state(self) -> None:
        for tensor in self.module.state_dict().values():
            dist.broadcast(tensor, src=0)

    def _build_buckets(self) -> list[_GradientBucket]:
        trainable_parameters = [parameter for parameter in _iter_unique_parameters(self.module) if parameter.requires_grad]
        reversed_parameters = list(reversed(trainable_parameters))
        if not reversed_parameters:
            return []

        buckets: list[_GradientBucket] = []
        current_parameters: list[torch.nn.Parameter] = []
        current_size = 0

        for parameter in reversed_parameters:
            parameter_size = parameter.numel() * parameter.element_size()
            would_overflow = (
                self.bucket_size_bytes is not None
                and current_parameters
                and current_size + parameter_size > self.bucket_size_bytes
            )
            if would_overflow:
                buckets.append(_GradientBucket(parameters=tuple(current_parameters)))
                current_parameters = []
                current_size = 0

            current_parameters.append(parameter)
            current_size += parameter_size

        if current_parameters:
            buckets.append(_GradientBucket(parameters=tuple(current_parameters)))
        return buckets

    def _register_parameter_hooks(self) -> None:
        for parameter in _iter_unique_parameters(self.module):
            if not parameter.requires_grad:
                continue
            parameter.register_post_accumulate_grad_hook(self._make_grad_sync_hook(parameter))

    def _make_grad_sync_hook(self, parameter: torch.nn.Parameter):
        parameter_id = id(parameter)

        def hook(_parameter: torch.nn.Parameter) -> None:
            if parameter.grad is None:
                return
            bucket = self._parameter_to_bucket[parameter_id]
            if parameter_id in bucket.ready_parameter_ids:
                return

            bucket.ready_parameter_ids.add(parameter_id)
            if len(bucket.ready_parameter_ids) != len(bucket.parameters):
                return

            gradients = [bucket_parameter.grad for bucket_parameter in bucket.parameters]
            if any(gradient is None for gradient in gradients):
                raise RuntimeError("Encountered a bucket with missing gradients during synchronization")

            flat_gradient = torch._utils._flatten_dense_tensors(gradients)
            flat_gradient.div_(self.world_size)
            bucket.flat_gradient = flat_gradient
            bucket.handle = dist.all_reduce(flat_gradient, async_op=True)

        return hook

    def start_gradient_synchronization(self) -> None:
        for bucket in self.buckets:
            bucket.reset()

    def finish_gradient_synchronization(self) -> None:
        for bucket in self.buckets:
            if bucket.handle is None or bucket.flat_gradient is None:
                continue
            bucket.handle.wait()
            synchronized_gradients = torch._utils._unflatten_dense_tensors(
                bucket.flat_gradient,
                [parameter.grad for parameter in bucket.parameters],
            )
            for parameter, synchronized_gradient in zip(bucket.parameters, synchronized_gradients):
                parameter.grad.copy_(synchronized_gradient)
            bucket.handle = None
            bucket.flat_gradient = None

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)