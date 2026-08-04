# 3D Benchmarks - GPU (A100)

### Uniform Noise

| Grid Size | Backend | Prep (ms) | Compute (ms) | Accuracy |
|---|---|---|---|---|
| 16x16x16 | FlashCubical (C++) | 0.01 | 2.73 | Baseline |
| 16x16x16 | JAX (Parallel Forest) | 2.25 | 11.98 | PASS |
| 16x16x16 | PyTorch (Parallel Forest) | 0.33 | 14.03 | PASS |
| 32x32x32 | FlashCubical (C++) | 0.01 | 25.65 | Baseline |
| 32x32x32 | JAX (Parallel Forest) | 0.70 | 41.34 | PASS |
| 32x32x32 | PyTorch (Parallel Forest) | 0.42 | 95.76 | PASS |
| 64x64x64 | FlashCubical (C++) | 0.02 | 261.80 | Baseline |
| 64x64x64 | JAX (Parallel Forest) | 2.62 | 346.70 | PASS |
| 64x64x64 | PyTorch (Parallel Forest) | 4.04 | 479.74 | PASS |
| 128x128x128 | FlashCubical (C++) | 0.07 | 3122.83 | Baseline |
| 128x128x128 | JAX (Parallel Forest) | 3.37 | 3175.15 | PASS |
| 128x128x128 | PyTorch (Parallel Forest) | 19.22 | 4115.58 | PASS |
### Gaussian Noise

| Grid Size | Backend | Prep (ms) | Compute (ms) | Accuracy |
|---|---|---|---|---|
| 16x16x16 | FlashCubical (C++) | 0.06 | 2.84 | Baseline |
| 16x16x16 | JAX (Parallel Forest) | 2.67 | 6.60 | PASS |
| 16x16x16 | PyTorch (Parallel Forest) | 0.25 | 38.65 | PASS |
| 32x32x32 | FlashCubical (C++) | 0.01 | 27.06 | Baseline |
| 32x32x32 | JAX (Parallel Forest) | 0.84 | 40.72 | PASS |
| 32x32x32 | PyTorch (Parallel Forest) | 0.33 | 90.29 | PASS |
| 64x64x64 | FlashCubical (C++) | 0.01 | 224.43 | Baseline |
| 64x64x64 | JAX (Parallel Forest) | 0.84 | 339.50 | PASS |
| 64x64x64 | PyTorch (Parallel Forest) | 1.09 | 531.09 | PASS |
| 128x128x128 | FlashCubical (C++) | 0.04 | 3206.80 | Baseline |
| 128x128x128 | JAX (Parallel Forest) | 1.57 | 3155.29 | PASS |
| 128x128x128 | PyTorch (Parallel Forest) | 4.63 | 4314.62 | PASS |
