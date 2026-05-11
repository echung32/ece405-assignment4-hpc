# DDP nsys Analysis — assignment_section2_ddp_nsys_full

## naive_nccl_cuda_ws2_medium_ctx128_bs4
  fwd=37.9ms  bwd=39.6ms  sync=13.0ms
  NCCL kernels during backward: 0.0%  overlap_fraction=0.00

## naive_nccl_cuda_ws2_medium_ctx256_bs4
  fwd=37.7ms  bwd=97.5ms  sync=16.1ms
  NCCL kernels during backward: 0.0%  overlap_fraction=0.00

## overlap_individual_nccl_cuda_ws2_medium_ctx128_bs4
  fwd=36.3ms  bwd=68.5ms  sync=1.7ms
  NCCL kernels during backward: 1.2%  overlap_fraction=1.00

## overlap_individual_nccl_cuda_ws2_medium_ctx256_bs4
  fwd=36.0ms  bwd=97.3ms  sync=2.2ms
  NCCL kernels during backward: 1.2%  overlap_fraction=0.92
