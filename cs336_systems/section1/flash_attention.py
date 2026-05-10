from __future__ import annotations

import math
from dataclasses import dataclass

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

DEFAULT_QUERY_TILE_SIZE = 128
DEFAULT_KEY_TILE_SIZE = 128


def naive_scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    if is_causal:
        query_positions = torch.arange(q.shape[-2], device=q.device).view(-1, 1)
        key_positions = torch.arange(k.shape[-2], device=k.device).view(1, -1)
        scores = scores.masked_fill(query_positions < key_positions, float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v), lse


def _math_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype in {torch.float16, torch.bfloat16}:
        return torch.float32
    return dtype


@dataclass(frozen=True)
class _FlashAttentionTiling:
    query_tile_size: int = DEFAULT_QUERY_TILE_SIZE
    key_tile_size: int = DEFAULT_KEY_TILE_SIZE


def _select_tile_size(sequence_length: int, d_model: int) -> int:
    if d_model >= 128:
        return 32
    return 64 if sequence_length >= 64 else 32


def _flatten_attention_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...], int, int]:
    if q.ndim < 2 or k.ndim < 2 or v.ndim < 2:
        raise ValueError("q, k, and v must each have at least 2 dimensions")
    if q.shape[:-2] != k.shape[:-2] or k.shape[:-2] != v.shape[:-2]:
        raise ValueError("q, k, and v must share the same leading dimensions")
    if k.shape[-2] != v.shape[-2]:
        raise ValueError("k and v must have the same key sequence length")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k must have the same head dimension")

    leading_shape = q.shape[:-2]
    q_len, d_model = q.shape[-2:]
    k_len = k.shape[-2]
    batch = math.prod(leading_shape) if leading_shape else 1
    return (
        q.reshape(batch, q_len, d_model),
        k.reshape(batch, k_len, d_model),
        v.reshape(batch, k_len, v.shape[-1]),
        leading_shape,
        q_len,
        k_len,
    )


def _causal_mask_block(
    scores: torch.Tensor,
    query_start: int,
    key_start: int,
) -> torch.Tensor:
    query_positions = torch.arange(query_start, query_start + scores.shape[1], device=scores.device)
    key_positions = torch.arange(key_start, key_start + scores.shape[2], device=scores.device)
    return scores.masked_fill(query_positions[:, None] < key_positions[None, :], float("-inf"))


if triton is not None:

    @triton.jit
    def _flash_attention_forward_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        l_ptr,
        stride_qb,
        stride_qq,
        stride_qd,
        stride_kb,
        stride_kk,
        stride_kd,
        stride_vb,
        stride_vk,
        stride_vd,
        stride_ob,
        stride_oq,
        stride_od,
        stride_lb,
        stride_lq,
        n_queries,
        n_keys,
        scale,
        is_causal: tl.constexpr,
        d_model: tl.constexpr,
        d_value: tl.constexpr,
        query_tile_size: tl.constexpr,
        key_tile_size: tl.constexpr,
    ):
        query_tile_index = tl.program_id(0)
        batch_index = tl.program_id(1)
        query_start = query_tile_index * query_tile_size

        q_block_ptr = tl.make_block_ptr(
            q_ptr + batch_index * stride_qb,
            shape=(n_queries, d_model),
            strides=(stride_qq, stride_qd),
            offsets=(query_start, 0),
            block_shape=(query_tile_size, d_model),
            order=(1, 0),
        )
        k_block_ptr = tl.make_block_ptr(
            k_ptr + batch_index * stride_kb,
            shape=(n_keys, d_model),
            strides=(stride_kk, stride_kd),
            offsets=(0, 0),
            block_shape=(key_tile_size, d_model),
            order=(1, 0),
        )
        v_block_ptr = tl.make_block_ptr(
            v_ptr + batch_index * stride_vb,
            shape=(n_keys, d_value),
            strides=(stride_vk, stride_vd),
            offsets=(0, 0),
            block_shape=(key_tile_size, d_value),
            order=(1, 0),
        )
        o_block_ptr = tl.make_block_ptr(
            o_ptr + batch_index * stride_ob,
            shape=(n_queries, d_value),
            strides=(stride_oq, stride_od),
            offsets=(query_start, 0),
            block_shape=(query_tile_size, d_value),
            order=(1, 0),
        )
        l_block_ptr = tl.make_block_ptr(
            l_ptr + batch_index * stride_lb,
            shape=(n_queries,),
            strides=(stride_lq,),
            offsets=(query_start,),
            block_shape=(query_tile_size,),
            order=(0,),
        )

        q = tl.load(q_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        key_offsets = tl.arange(0, key_tile_size)
        query_offsets = query_start + tl.arange(0, query_tile_size)

        row_max = tl.full((query_tile_size,), -float("inf"), dtype=tl.float32)
        row_sum = tl.zeros((query_tile_size,), dtype=tl.float32)
        row_output = tl.zeros((query_tile_size, d_value), dtype=tl.float32)

        max_key_tile_count = tl.cdiv(n_keys, key_tile_size)
        if is_causal:
            max_key_tile_count = tl.cdiv(tl.minimum(n_keys, query_start + query_tile_size), key_tile_size)

        for key_start in range(0, max_key_tile_count):
            key_block_start = key_start * key_tile_size

            k = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
            v = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
            scores = tl.dot(q, tl.trans(k)) * scale

            key_positions = key_block_start + key_offsets
            valid_mask = key_positions[None, :] < n_keys
            if is_causal:
                valid_mask = valid_mask & (query_offsets[:, None] >= key_positions[None, :])
            scores = tl.where(valid_mask, scores, -1.0e6)

            block_max = tl.max(scores, axis=1)
            updated_max = tl.maximum(row_max, block_max)
            probabilities = tl.exp(scores - updated_max[:, None])
            rescale = tl.exp(row_max - updated_max)

            row_sum = row_sum * rescale + tl.sum(probabilities, axis=1)
            row_output = row_output * rescale[:, None] + tl.dot(probabilities, v)
            row_max = updated_max

            k_block_ptr = k_block_ptr.advance((key_tile_size, 0))
            v_block_ptr = v_block_ptr.advance((key_tile_size, 0))

        normalized_output = row_output / row_sum[:, None]
        logsumexp = row_max + tl.log(row_sum)
        tl.store(o_block_ptr, normalized_output.to(o_ptr.type.element_ty), boundary_check=(0, 1))
        tl.store(l_block_ptr, logsumexp, boundary_check=(0,))


def flash_attention_forward_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    *,
    tiling: _FlashAttentionTiling | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if triton is None or tl is None:
        raise RuntimeError("Triton is not installed in the current environment")
    if q.device.type != "cuda" or k.device.type != "cuda" or v.device.type != "cuda":
        raise ValueError("FlashAttention Triton forward requires CUDA tensors")

    q_flat, k_flat, v_flat, leading_shape, q_len, k_len = _flatten_attention_inputs(q, k, v)
    q_flat = q_flat.contiguous()
    k_flat = k_flat.contiguous()
    v_flat = v_flat.contiguous()

    tiling = tiling or _FlashAttentionTiling(
        query_tile_size=_select_tile_size(q_len, q.shape[-1]),
        key_tile_size=_select_tile_size(k_len, q.shape[-1]),
    )

    batch = q_flat.shape[0]
    output = torch.empty((batch, q_len, v_flat.shape[-1]), device=q.device, dtype=v.dtype)
    lse = torch.empty((batch, q_len), device=q.device, dtype=_math_dtype(q.dtype))
    scale = 1.0 / math.sqrt(q.shape[-1])

    grid = (triton.cdiv(q_len, tiling.query_tile_size), batch)
    _flash_attention_forward_kernel[grid](
        q_flat,
        k_flat,
        v_flat,
        output,
        lse,
        q_flat.stride(0),
        q_flat.stride(1),
        q_flat.stride(2),
        k_flat.stride(0),
        k_flat.stride(1),
        k_flat.stride(2),
        v_flat.stride(0),
        v_flat.stride(1),
        v_flat.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        lse.stride(0),
        lse.stride(1),
        n_queries=q_len,
        n_keys=k_len,
        scale=scale,
        is_causal=is_causal,
        d_model=q_flat.shape[-1],
        d_value=v_flat.shape[-1],
        query_tile_size=tiling.query_tile_size,
        key_tile_size=tiling.key_tile_size,
    )
    return output.reshape(*leading_shape, q_len, v_flat.shape[-1]), lse.reshape(*leading_shape, q_len)


def flash_attention_forward_tiled(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    *,
    tiling: _FlashAttentionTiling | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    tiling = tiling or _FlashAttentionTiling()
    q_flat, k_flat, v_flat, leading_shape, q_len, k_len = _flatten_attention_inputs(q, k, v)

    original_dtype = q.dtype
    math_dtype = _math_dtype(original_dtype)
    q_math = q_flat.to(dtype=math_dtype)
    k_math = k_flat.to(dtype=math_dtype)
    v_math = v_flat.to(dtype=math_dtype)

    batch = q_math.shape[0]
    scale = 1.0 / math.sqrt(q_math.shape[-1])
    output = torch.empty((batch, q_len, v_math.shape[-1]), device=q.device, dtype=math_dtype)
    lse = torch.empty((batch, q_len), device=q.device, dtype=math_dtype)

    for query_start in range(0, q_len, tiling.query_tile_size):
        query_end = min(query_start + tiling.query_tile_size, q_len)
        q_block = q_math[:, query_start:query_end, :]
        row_max = torch.full((batch, query_end - query_start), float("-inf"), device=q.device, dtype=math_dtype)
        row_sum = torch.zeros((batch, query_end - query_start), device=q.device, dtype=math_dtype)
        row_output = torch.zeros(
            (batch, query_end - query_start, v_math.shape[-1]),
            device=q.device,
            dtype=math_dtype,
        )

        for key_start in range(0, k_len, tiling.key_tile_size):
            key_end = min(key_start + tiling.key_tile_size, k_len)
            k_block = k_math[:, key_start:key_end, :]
            v_block = v_math[:, key_start:key_end, :]

            scores = torch.matmul(q_block, k_block.transpose(-1, -2)) * scale
            if is_causal:
                scores = _causal_mask_block(scores, query_start=query_start, key_start=key_start)

            block_max = torch.max(scores, dim=-1).values
            updated_max = torch.maximum(row_max, block_max)
            block_exp = torch.exp(scores - updated_max.unsqueeze(-1))
            max_rescale = torch.exp(row_max - updated_max)
            row_sum = row_sum * max_rescale + torch.sum(block_exp, dim=-1)
            row_output = row_output * max_rescale.unsqueeze(-1) + torch.matmul(block_exp, v_block)
            row_max = updated_max

        output[:, query_start:query_end, :] = row_output / row_sum.unsqueeze(-1)
        lse[:, query_start:query_end] = row_max + torch.log(row_sum)

    output = output.reshape(*leading_shape, q_len, v_flat.shape[-1]).to(dtype=original_dtype)
    lse = lse.reshape(*leading_shape, q_len)
    return output, lse


def flash_attention_backward_tiled(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    grad_output: torch.Tensor,
    is_causal: bool = False,
    *,
    tiling: _FlashAttentionTiling | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tiling = tiling or _FlashAttentionTiling()
    q_flat, k_flat, v_flat, leading_shape, q_len, k_len = _flatten_attention_inputs(q, k, v)
    output_flat = output.reshape(-1, q_len, output.shape[-1])
    lse_flat = lse.reshape(-1, q_len)
    grad_output_flat = grad_output.reshape(-1, q_len, grad_output.shape[-1])

    original_dtype = q.dtype
    math_dtype = _math_dtype(original_dtype)
    q_math = q_flat.to(dtype=math_dtype)
    k_math = k_flat.to(dtype=math_dtype)
    v_math = v_flat.to(dtype=math_dtype)
    output_math = output_flat.to(dtype=math_dtype)
    lse_math = lse_flat.to(dtype=math_dtype)
    grad_output_math = grad_output_flat.to(dtype=math_dtype)

    scale = 1.0 / math.sqrt(q_math.shape[-1])
    grad_q = torch.zeros_like(q_math)
    grad_k = torch.zeros_like(k_math)
    grad_v = torch.zeros_like(v_math)
    delta = torch.sum(grad_output_math * output_math, dim=-1)

    for query_start in range(0, q_len, tiling.query_tile_size):
        query_end = min(query_start + tiling.query_tile_size, q_len)
        q_block = q_math[:, query_start:query_end, :]
        do_block = grad_output_math[:, query_start:query_end, :]
        lse_block = lse_math[:, query_start:query_end]
        delta_block = delta[:, query_start:query_end]

        for key_start in range(0, k_len, tiling.key_tile_size):
            key_end = min(key_start + tiling.key_tile_size, k_len)
            k_block = k_math[:, key_start:key_end, :]
            v_block = v_math[:, key_start:key_end, :]

            scores = torch.matmul(q_block, k_block.transpose(-1, -2)) * scale
            if is_causal:
                scores = _causal_mask_block(scores, query_start=query_start, key_start=key_start)

            probs = torch.exp(scores - lse_block.unsqueeze(-1))
            grad_probs = torch.matmul(do_block, v_block.transpose(-1, -2))
            grad_scores = probs * (grad_probs - delta_block.unsqueeze(-1))

            grad_q[:, query_start:query_end, :] += torch.matmul(grad_scores, k_block) * scale
            grad_k[:, key_start:key_end, :] += torch.matmul(grad_scores.transpose(-1, -2), q_block) * scale
            grad_v[:, key_start:key_end, :] += torch.matmul(probs.transpose(-1, -2), do_block)

    grad_q = grad_q.reshape(*leading_shape, q_len, q.shape[-1]).to(dtype=original_dtype)
    grad_k = grad_k.reshape(*leading_shape, k_len, k.shape[-1]).to(dtype=original_dtype)
    grad_v = grad_v.reshape(*leading_shape, k_len, v.shape[-1]).to(dtype=v.dtype)
    return grad_q, grad_k, grad_v


class FlashAttention2PyTorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        output, lse = flash_attention_forward_tiled(q, k, v, is_causal=is_causal)
        ctx.save_for_backward(q, k, v, output, lse)
        ctx.is_causal = is_causal
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        q, k, v, output, lse = ctx.saved_tensors
        grad_q, grad_k, grad_v = flash_attention_backward_tiled(
            q,
            k,
            v,
            output,
            lse,
            grad_output,
            is_causal=ctx.is_causal,
        )
        return grad_q, grad_k, grad_v, None


class FlashAttention2Triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        output, lse = flash_attention_forward_triton(q, k, v, is_causal=is_causal)
        ctx.save_for_backward(q, k, v, output, lse)
        ctx.is_causal = is_causal
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        q, k, v, output, lse = ctx.saved_tensors
        grad_q, grad_k, grad_v = flash_attention_backward_tiled(
            q,
            k,
            v,
            output,
            lse,
            grad_output,
            is_causal=ctx.is_causal,
        )
        return grad_q, grad_k, grad_v, None


def get_flashattention_autograd_function_pytorch() -> type[FlashAttention2PyTorch]:
    return FlashAttention2PyTorch


def get_flashattention_autograd_function_triton() -> type[FlashAttention2Triton]:
    return FlashAttention2Triton
