# nsys CUDA Kernel Analysis

## GEMM fraction by mode (small model)

| ctx | forward GEMM% | fwd+bwd GEMM% | train_step GEMM% |
|-----|--------------|---------------|-----------------|
| 128 | 69.9% | 55.9% | 34.0% |
| 256 | 76.6% | 65.8% | 44.9% |
| 512 | 77.6% | 71.0% | 57.3% |
| 1024 | 78.9% | 64.7% | 58.3% |
