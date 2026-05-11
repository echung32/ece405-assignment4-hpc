# Sections 1, 2, 3 Runbook

This file tracks what still needs to be run, and which experiments remain for Sections 1, 2, and 3.

## Section 1

### Core correctness checks

Run these after touching attention code:

```bash
uv run pytest tests/test_attention.py
```

### Smoke runs

Attention benchmark smoke:

```bash
uv run python -m scripts.section1_2_campaigns attention --smoke --execute
uv run python -m scripts.section1_1_report --input-root data/section1_2 --campaign assignment_attention_smoke
```

FlashAttention benchmark smoke:

```bash
uv run python -m scripts.section1_2_campaigns flash --smoke --execute
uv run python -m scripts.section1_2_report --input-root data/section1_3 --campaign assignment_flash_smoke
```

Full-model compile smoke:

```bash
uv run python -m scripts.section1_2_campaigns model-compile --smoke --execute
uv run python -m scripts.section1_2_report --input-root data/section1_3 --campaign assignment_model-compile_smoke
```

### Full experiments still to run

PyTorch attention and compiled attention grids:

```bash
uv run python -m scripts.section1_2_campaigns attention --execute
uv run python -m scripts.section1_1_report --input-root data/section1_2 --campaign assignment_attention_full
```

Full-model eager vs compiled runs:

```bash
uv run python -m scripts.section1_2_campaigns model-compile --execute
uv run python -m scripts.section1_2_report --input-root data/section1_3 --campaign assignment_model-compile_full
```

FlashAttention benchmark grid:

```bash
uv run python -m scripts.section1_2_campaigns flash --execute
uv run python -m scripts.section1_2_report --input-root data/section1_3 --campaign assignment_flash_full
```

### Section 1 deliverables still to inspect manually

- OOM thresholds for vanilla attention at large sequence lengths.
- Forward, backward, and end-to-end latency tables.
- Comparison of eager vs compiled model timings.
- FlashAttention vs vanilla attention latency tables across dtype and sequence length.

## Section 2

### Core correctness checks

```bash
uv run pytest tests/test_ddp_individual_parameters.py
uv run pytest tests/test_ddp.py
```

### Remaining implementation work

- Nsight-assisted benchmark workflow for overlap comparisons.

### Smoke runs

All-reduce smoke:

```bash
uv run python -m scripts.section2_campaigns allreduce --smoke --execute
uv run python -m scripts.section2_report --input-root data/section2 --campaign assignment_section2_allreduce_smoke
```

DDP smoke:

```bash
uv run python -m scripts.section2_campaigns ddp --smoke --execute
uv run python -m scripts.section2_report --input-root data/section2 --campaign assignment_section2_ddp_smoke
```

### Full experiments to run

```bash
uv run python -m scripts.section2_campaigns allreduce --execute
uv run python -m scripts.section2_report --input-root data/section2 --campaign assignment_section2_allreduce_full

uv run python -m scripts.section2_campaigns ddp --execute
uv run python -m scripts.section2_report --input-root data/section2 --campaign assignment_section2_ddp_full
```

## Section 3

### Core correctness checks

```bash
uv run pytest tests/test_sharded_optimizer.py
```

### Smoke runs

Timing smoke:

```bash
uv run python -m scripts.section3_campaigns timing --smoke --execute
uv run python -m scripts.section3_report --input-root data/section3 --campaign assignment_section3_timing_smoke
```

Memory smoke:

```bash
uv run python -m scripts.section3_campaigns memory --smoke --execute
uv run python -m scripts.section3_report --input-root data/section3 --campaign assignment_section3_memory_smoke
```

### Full experiments to run

```bash
uv run python -m scripts.section3_campaigns timing --execute
uv run python -m scripts.section3_report --input-root data/section3 --campaign assignment_section3_timing_full

uv run python -m scripts.section3_campaigns memory --execute
uv run python -m scripts.section3_report --input-root data/section3 --campaign assignment_section3_memory_full
```

## Suggested execution order

1. Keep Section 1 tests green.
2. Run the Section 2 smoke campaigns and inspect the generated summary tables.
3. Run the Section 3 smoke campaigns and inspect the generated summary tables.
4. Run the full Section  and 3 campaigns in tmux for long jobs.
5. Run the remaining full Section 1 campaigns if the writeup tables are still missing.

## Running progress log

### 2026-05-04 status snapshot

**Section 1 — all full campaigns complete.**

| Campaign | Status | Artifacts |
|---|---|---|
| 1.1 Timing full | 120 / 120 | `data/section1_1/assignment_timing_full` |
| 1.1 Memory full | 12 / 12 | `data/section1_1/assignment_memory_full` |
| 1.1 Nsys full | 56 / 60 (4 have `.qdstrm` only — nsys importer unavailable) | `data/section1_1/assignment_nsys_full` |
| 1.1 Mixed-precision (1.1.5) | Completed and validated | — |
| 1.2 Attention full | 4 / 4 | `data/section1_2/assignment_attention_full` |
| 1.3 Flash full | 4 / 4 | `data/section1_3/assignment_flash_full` |
| 1.3 Model-compile full | 30 / 30 | `data/section1_3/assignment_model-compile_full` |

All OOM failures were retried at `--vram-limit-gb 85` on cuda:1 once GPUs were free; all passed. Smoke data/log directories removed.

**Section 2 — ALL COMPLETE (2026-05-04).**

Root cause of NCCL ws2 hangs: PHB GPU topology (no NVLink) caused NCCL P2P to hang.
Fix applied: `os.environ.setdefault("NCCL_P2P_DISABLE", "1")` in `cs336_systems/benchmark_utils.py` `setup_process_group()`.
Note: ws4/ws6 NCCL runs are `skipped` (only 2 GPUs available; preflight check rejects world_size > device_count).

| Campaign | Status | Artifacts |
|---|---|---|
| 2 Allreduce full | 24 / 24 (ws4/ws6 NCCL = skipped/no-op) | `data/section2/assignment_section2_allreduce_full` |
| 2 DDP full | 7 / 7 | `data/section2/assignment_section2_ddp_full` |

**Section 3 — ALL COMPLETE (2026-05-04).**

| Campaign | Status | Artifacts |
|---|---|---|
| 3 Timing full | 2 / 2 | `data/section3/assignment_section3_timing_full` |
| 3 Memory full | 2 / 2 | `data/section3/assignment_section3_memory_full` |

Reports generated for all Section 2/3 campaigns.

### Section 2/3 key result highlights

**DDP step time (NCCL/CUDA/ws2/xl/ctx128/bs4):**
- Naive: 2.19s compute + 0.044s comm
- Flat all-reduce: 2.19s compute + 0.039s comm
- Overlap individual: 1.94s compute + 0.002s comm
- Overlap bucketed 1MB: 1.94s compute + 0.006s comm
- Overlap bucketed 10MB: 1.95s compute + 0.006s comm
- Overlap bucketed 100MB: 1.91s compute + 0.022s comm
- Overlap bucketed 1000MB: 1.86s compute + 0.020s comm

**Section 3 timing (NCCL/CUDA/ws2/xl):**
- Standard optimizer: 1.91s/step
- Sharded optimizer: 2.70s/step (~41% slower due to gather/scatter overhead)

**Section 3 memory (NCCL/CUDA/ws2/xl, post_step cuda_reserved):**
- Standard: 42.7 GB optimizer state (full copy per rank)
- Sharded: 31.3 GB optimizer state (~27% reduction with 2 ranks)