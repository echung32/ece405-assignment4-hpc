"""Profile CUDA kernel statistics for nsys report questions b, d, e.

Runs torch.profiler on:
  - small model, ctx=128, forward (for question b: top kernels in fwd pass)
  - small model, ctx=128, train_step (for question d: GEMM fraction change)
  - small model, ctx=128, forward with NVTX attention annotations (for question e: softmax vs matmul)

Outputs JSON to data/section1_1/analysis/kernel_profiling/.
"""
from __future__ import annotations

import json
import math
import pathlib
from contextlib import contextmanager

import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile, record_function

# ── import model from cs336-basics ──────────────────────────────────────────
from cs336_basics.model import BasicsTransformerLM

OUT_DIR = pathlib.Path("data/section1_1/analysis/kernel_profiling")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# small model spec (same as benchmarking_script)
MODEL_SPEC = dict(
    vocab_size=50257,
    context_length=128,
    d_model=768,
    num_layers=12,
    num_heads=12,
    d_ff=3072,
    rope_theta=10000.0,
)

DEVICE = torch.device("cuda:0")
WARMUP = 5
CTX = 128
BATCH = 1


def make_model():
    m = BasicsTransformerLM(**MODEL_SPEC).to(DEVICE)
    return m


def make_inputs():
    return torch.randint(0, MODEL_SPEC["vocab_size"], (BATCH, CTX), device=DEVICE)


def top_kernels(prof, n=15):
    """Return list of {name, count, self_cuda_us, pct} for top n GPU kernels."""
    avgs = prof.key_averages()
    # filter to CUDA kernels (no Python overhead entries)
    kernel_events = [e for e in avgs if e.self_device_time_total > 0]
    total_us = sum(e.self_device_time_total for e in kernel_events)
    kernel_events.sort(key=lambda e: e.self_device_time_total, reverse=True)
    results = []
    for e in kernel_events[:n]:
        results.append({
            "name": e.key,
            "count": e.count,
            "self_cuda_ms": round(e.self_device_time_total / 1000, 3),
            "pct": round(e.self_device_time_total / total_us * 100, 2),
        })
    return results, round(total_us / 1000, 3)


def run_forward_profile():
    model = make_model()
    inputs = make_inputs()

    # warmup
    for _ in range(WARMUP):
        with torch.no_grad():
            model(inputs)
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False, with_stack=False) as prof:
        for _ in range(3):
            with torch.no_grad():
                with record_function("forward"):
                    model(inputs)
        torch.cuda.synchronize()

    kernels, total_ms = top_kernels(prof)
    return {"mode": "forward", "ctx": CTX, "total_gpu_ms": total_ms, "top_kernels": kernels}


def run_train_step_profile():
    model = make_model()
    inputs = make_inputs()
    labels = torch.randint(0, MODEL_SPEC["vocab_size"], (BATCH, CTX), device=DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for _ in range(WARMUP):
        optimizer.zero_grad()
        logits = model(inputs)
        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False, with_stack=False) as prof:
        for _ in range(3):
            with record_function("train_step"):
                optimizer.zero_grad()
                with record_function("forward"):
                    logits = model(inputs)
                with record_function("loss"):
                    loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
                with record_function("backward"):
                    loss.backward()
                with record_function("optimizer_step"):
                    optimizer.step()
        torch.cuda.synchronize()

    kernels, total_ms = top_kernels(prof)
    return {"mode": "train_step", "ctx": CTX, "total_gpu_ms": total_ms, "top_kernels": kernels}


def run_attention_profile():
    """Profile individual attention ops for question e (softmax vs matmul)."""
    d_head = MODEL_SPEC["d_model"] // MODEL_SPEC["num_heads"]  # 64
    seq = CTX
    heads = MODEL_SPEC["num_heads"]

    Q = torch.randn(BATCH, heads, seq, d_head, device=DEVICE)
    K = torch.randn(BATCH, heads, seq, d_head, device=DEVICE)
    V = torch.randn(BATCH, heads, seq, d_head, device=DEVICE)
    scale = math.sqrt(d_head)

    def qk_matmul():
        return torch.matmul(Q, K.transpose(-2, -1)) / scale

    def softmax_op(scores):
        return torch.softmax(scores, dim=-1)

    def av_matmul(weights):
        return torch.matmul(weights, V)

    # warmup
    for _ in range(WARMUP):
        scores = qk_matmul()
        weights = softmax_op(scores)
        _ = av_matmul(weights)
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False) as prof:
        for _ in range(10):
            with record_function("qk_matmul"):
                scores = qk_matmul()
            with record_function("softmax"):
                weights = softmax_op(scores)
            with record_function("av_matmul"):
                _ = av_matmul(weights)
        torch.cuda.synchronize()

    avgs = prof.key_averages()
    ops = {}
    for e in avgs:
        if e.key in ("qk_matmul", "softmax", "av_matmul"):
            ops[e.key] = {
                "count": e.count,
                "cuda_ms": round(e.device_time_total / 1000, 4),
                "self_cuda_ms": round(e.self_device_time_total / 1000, 4),
            }
    return {"mode": "attention_ops", "ctx": CTX, "d_head": d_head, "heads": heads, "ops": ops}


def main():
    print("Running forward profile...")
    fwd = run_forward_profile()
    (OUT_DIR / "forward_kernels.json").write_text(json.dumps(fwd, indent=2))
    print(f"  Total GPU time: {fwd['total_gpu_ms']:.1f} ms (3 steps)")
    print("  Top 5 kernels:")
    for k in fwd["top_kernels"][:5]:
        print(f"    {k['pct']:5.1f}%  {k['count']:4d}x  {k['self_cuda_ms']:.2f} ms  {k['name'][:80]}")

    print("\nRunning train_step profile...")
    ts = run_train_step_profile()
    (OUT_DIR / "train_step_kernels.json").write_text(json.dumps(ts, indent=2))
    print(f"  Total GPU time: {ts['total_gpu_ms']:.1f} ms (3 steps)")
    print("  Top 5 kernels:")
    for k in ts["top_kernels"][:5]:
        print(f"    {k['pct']:5.1f}%  {k['count']:4d}x  {k['self_cuda_ms']:.2f} ms  {k['name'][:80]}")

    print("\nRunning attention op profile...")
    attn = run_attention_profile()
    (OUT_DIR / "attention_ops.json").write_text(json.dumps(attn, indent=2))
    print(f"  d_head={attn['d_head']}, heads={attn['heads']}, ctx={attn['ctx']}, 10 reps each")
    for op, v in attn["ops"].items():
        print(f"    {op:15s}: cuda_ms={v['cuda_ms']:.4f}  count={v['count']}")

    print(f"\nOutputs written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
