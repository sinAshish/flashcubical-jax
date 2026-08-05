# 3D Benchmarks - GPU (A100)

### Uniform Noise

| Grid Size | Backend | Prep (ms) | Compute (ms) | Accuracy |
|---|---|---|---|---|
| 16x16x16 | FlashCubical (C++) | 0.02 | 3.55 | Baseline |
| 16x16x16 | JAX (Parallel Forest) | 1.91 | 12.20 | PASS |
| 16x16x16 | PyTorch (Parallel Forest) | 1.42 | 6.81 | PASS |
| 32x32x32 | FlashCubical (C++) | 0.02 | 25.64 | Baseline |
| 32x32x32 | JAX (Parallel Forest) | 1.65 | 43.33 | PASS |
| 32x32x32 | PyTorch (Parallel Forest) | 1.04 | 36.46 | PASS |
| 64x64x64 | FlashCubical (C++) | 0.03 | 254.39 | Baseline |
| 64x64x64 | JAX (Parallel Forest) | 1.72 | 342.01 | PASS |
| 64x64x64 | PyTorch (Parallel Forest) | 25.33 | 325.44 | PASS |
| 128x128x128 | FlashCubical (C++) | 0.07 | 2908.20 | Baseline |
| 128x128x128 | JAX (Parallel Forest) | 1.72 | 3263.89 | PASS |
| 128x128x128 | PyTorch (Parallel Forest) | 37.71 | 2910.15 | PASS |
### Gaussian Noise

| Grid Size | Backend | Prep (ms) | Compute (ms) | Accuracy |
|---|---|---|---|---|
| 16x16x16 | FlashCubical (C++) | 0.09 | 2.71 | Baseline |
| 16x16x16 | JAX (Parallel Forest) | 2.25 | 6.18 | PASS |
| 16x16x16 | PyTorch (Parallel Forest) | 0.49 | 6.08 | PASS |
| 32x32x32 | FlashCubical (C++) | 0.01 | 24.90 | Baseline |
| 32x32x32 | JAX (Parallel Forest) | 0.48 | 36.35 | PASS |
| 32x32x32 | PyTorch (Parallel Forest) | 0.52 | 34.61 | PASS |
| 64x64x64 | FlashCubical (C++) | 0.01 | 223.80 | Baseline |
| 64x64x64 | JAX (Parallel Forest) | 0.83 | 338.44 | PASS |
| 64x64x64 | PyTorch (Parallel Forest) | 8.70 | 320.76 | PASS |
| 128x128x128 | FlashCubical (C++) | 0.09 | 3054.93 | Baseline |
| 128x128x128 | JAX (Parallel Forest) | 1.60 | 3196.97 | PASS |
| 128x128x128 | PyTorch (Parallel Forest) | 21.27 | 2768.37 | PASS |
