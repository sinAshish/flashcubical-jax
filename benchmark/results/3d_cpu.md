# 3D Benchmarks - CPU

### Uniform Noise

| Grid Size | Backend | Prep (ms) | Compute (ms) | Accuracy |
|---|---|---|---|---|
| 16x16x16 | FlashCubical (C++) | 0.01 | 2.77 | Baseline |
| 16x16x16 | JAX (Parallel Forest) | 1.23 | 21.42 | PASS |
| 16x16x16 | PyTorch (Parallel Forest) | 0.36 | 396.02 | PASS |
| 32x32x32 | FlashCubical (C++) | 0.02 | 25.38 | Baseline |
| 32x32x32 | JAX (Parallel Forest) | 1.05 | 121.01 | PASS |
| 32x32x32 | PyTorch (Parallel Forest) | 0.23 | 1171.73 | PASS |
| 64x64x64 | FlashCubical (C++) | 0.03 | 232.89 | Baseline |
| 64x64x64 | JAX (Parallel Forest) | 0.76 | 877.98 | PASS |
| 64x64x64 | PyTorch (Parallel Forest) | 15.91 | 3033.34 | PASS |
| 128x128x128 | FlashCubical (C++) | 0.06 | 3230.40 | Baseline |
| 128x128x128 | JAX (Parallel Forest) | 1.31 | 7605.92 | PASS |
| 128x128x128 | PyTorch (Parallel Forest) | 10.94 | 4265.07 | PASS |
### Gaussian Noise

| Grid Size | Backend | Prep (ms) | Compute (ms) | Accuracy |
|---|---|---|---|---|
| 16x16x16 | FlashCubical (C++) | 0.06 | 3.24 | Baseline |
| 16x16x16 | JAX (Parallel Forest) | 1.19 | 11.83 | PASS |
| 16x16x16 | PyTorch (Parallel Forest) | 0.20 | 11.86 | PASS |
| 32x32x32 | FlashCubical (C++) | 0.01 | 28.57 | Baseline |
| 32x32x32 | JAX (Parallel Forest) | 0.52 | 125.13 | PASS |
| 32x32x32 | PyTorch (Parallel Forest) | 0.27 | 111.33 | PASS |
| 64x64x64 | FlashCubical (C++) | 0.01 | 234.04 | Baseline |
| 64x64x64 | JAX (Parallel Forest) | 0.66 | 862.90 | PASS |
| 64x64x64 | PyTorch (Parallel Forest) | 1.09 | 583.23 | PASS |
| 128x128x128 | FlashCubical (C++) | 0.05 | 2935.04 | Baseline |
| 128x128x128 | JAX (Parallel Forest) | 1.44 | 7583.84 | PASS |
| 128x128x128 | PyTorch (Parallel Forest) | 20.03 | 4362.25 | PASS |
