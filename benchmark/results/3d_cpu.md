# 3D Benchmarks - CPU

### Uniform Noise

| Grid Size | Backend | Prep (ms) | Compute (ms) | Accuracy |
|---|---|---|---|---|
| 16x16x16 | FlashCubical (C++) | 0.02 | 2.72 | Baseline |
| 16x16x16 | JAX (Parallel Forest) | 1.31 | 18.15 | PASS |
| 16x16x16 | PyTorch (Parallel Forest) | 0.42 | 8.90 | PASS |
| 32x32x32 | FlashCubical (C++) | 0.01 | 25.27 | Baseline |
| 32x32x32 | JAX (Parallel Forest) | 0.46 | 111.57 | PASS |
| 32x32x32 | PyTorch (Parallel Forest) | 0.24 | 95.84 | PASS |
| 64x64x64 | FlashCubical (C++) | 0.01 | 236.70 | Baseline |
| 64x64x64 | JAX (Parallel Forest) | 0.65 | 845.30 | PASS |
| 64x64x64 | PyTorch (Parallel Forest) | 2.29 | 474.00 | PASS |
| 128x128x128 | FlashCubical (C++) | 0.06 | 3197.27 | Baseline |
| 128x128x128 | JAX (Parallel Forest) | 1.48 | 7317.96 | PASS |
| 128x128x128 | PyTorch (Parallel Forest) | 6.80 | 4119.36 | PASS |
### Gaussian Noise

| Grid Size | Backend | Prep (ms) | Compute (ms) | Accuracy |
|---|---|---|---|---|
| 16x16x16 | FlashCubical (C++) | 0.07 | 2.73 | Baseline |
| 16x16x16 | JAX (Parallel Forest) | 1.22 | 16.86 | PASS |
| 16x16x16 | PyTorch (Parallel Forest) | 0.36 | 12.71 | PASS |
| 32x32x32 | FlashCubical (C++) | 0.01 | 24.61 | Baseline |
| 32x32x32 | JAX (Parallel Forest) | 0.31 | 104.85 | PASS |
| 32x32x32 | PyTorch (Parallel Forest) | 0.37 | 86.59 | PASS |
| 64x64x64 | FlashCubical (C++) | 0.01 | 232.21 | Baseline |
| 64x64x64 | JAX (Parallel Forest) | 0.75 | 828.05 | PASS |
| 64x64x64 | PyTorch (Parallel Forest) | 4.01 | 514.24 | PASS |
| 128x128x128 | FlashCubical (C++) | 0.13 | 3095.49 | Baseline |
| 128x128x128 | JAX (Parallel Forest) | 1.44 | 7376.90 | PASS |
| 128x128x128 | PyTorch (Parallel Forest) | 16.75 | 3859.04 | PASS |
