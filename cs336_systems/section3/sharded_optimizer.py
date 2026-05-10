from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any, Type

import torch
import torch.distributed as dist


class ShardedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls: Type[torch.optim.Optimizer], **kwargs: Any):
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = dict(kwargs)
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self._parameter_owners: dict[int, int] = {}
        self._next_parameter_index = 0
        self._local_pending_param_groups: list[dict[str, Any]] = []
        self._local_optimizer: torch.optim.Optimizer | None = None
        self._initializing = True

        super().__init__(params, kwargs)
        self.state = defaultdict(dict)

        self._initializing = False
        if self._local_pending_param_groups:
            self._local_optimizer = self.optimizer_cls(self._local_pending_param_groups, **self.optimizer_kwargs)

    def _normalize_param_group(self, param_group: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(param_group)
        params = normalized["params"]
        if isinstance(params, torch.Tensor):
            normalized["params"] = [params]
        else:
            normalized["params"] = list(params)
        return normalized

    def _build_local_param_group(self, normalized_group: dict[str, Any]) -> dict[str, Any] | None:
        local_params: list[torch.nn.Parameter] = []
        for parameter in normalized_group["params"]:
            parameter_id = id(parameter)
            if parameter_id not in self._parameter_owners:
                self._parameter_owners[parameter_id] = self._next_parameter_index % self.world_size
                self._next_parameter_index += 1
            if self._parameter_owners[parameter_id] == self.rank:
                local_params.append(parameter)

        if not local_params:
            return None

        local_group = {key: value for key, value in normalized_group.items() if key != "params"}
        local_group["params"] = local_params
        return local_group

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        normalized_group = self._normalize_param_group(param_group)
        local_group = self._build_local_param_group(normalized_group)
        super().add_param_group(normalized_group)

        if local_group is None:
            return
        if self._initializing:
            self._local_pending_param_groups.append(local_group)
            return
        if self._local_optimizer is None:
            self._local_optimizer = self.optimizer_cls([local_group], **self.optimizer_kwargs)
            return
        self._local_optimizer.add_param_group(local_group)

    @torch.no_grad()
    def step(self, closure=None, **kwargs: Any):
        loss = None
        if self._local_optimizer is not None:
            loss = self._local_optimizer.step(closure=closure, **kwargs)
        elif closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self.world_size == 1:
            return loss

        for param_group in self.param_groups:
            for parameter in param_group["params"]:
                dist.broadcast(parameter.data, src=self._parameter_owners[id(parameter)])
        return loss
