# 3D Benchmarks - CPU

### Uniform Noise

| Grid Size | Backend | Prep (ms) | Compute (ms) | Accuracy |
|---|---|---|---|---|
| 16x16x16 | FlashCubical (C++) | 0.01 | 2.78 | Baseline |
| 16x16x16 | JAX (Parallel Forest) | 1.15 | 10.66 | PASS |
| 16x16x16 | PyTorch (Parallel Forest) | 0.22 | 144.55 | PASS |
| 16x16x16 | Triton (Parallel Forest) | 0.07 | 3.36 | PASS |
| 16x16x16 | CUDA C++ (Numba-free) | 0.04 | 3.01 | PASS |
| 32x32x32 | FlashCubical (C++) | 0.01 | 25.47 | Baseline |
| 32x32x32 | JAX (Parallel Forest) | 0.44 | 69.63 | PASS |
| 32x32x32 | PyTorch (Parallel Forest) | 0.19 | 277.19 | PASS |
| 32x32x32 | Triton (Parallel Forest) | 0.07 | 26.30 | PASS |
| 32x32x32 | CUDA C++ (Numba-free) | 0.07 | 24.62 | PASS |
| 64x64x64 | FlashCubical (C++) | 0.01 | 238.75 | Baseline |
| 64x64x64 | JAX (Parallel Forest) | 0.58 | 491.11 | PASS |
| 64x64x64 | PyTorch (Parallel Forest) | 15.54 | 536.18 | PASS |
| 64x64x64 | Triton (Parallel Forest) | 7.33 | 246.68 | PASS |
| 64x64x64 | CUDA C++ (Numba-free) | 1.92 | 223.45 | PASS |
| 128x128x128 | FlashCubical (C++) | 0.05 | 3185.47 | Baseline |
| 128x128x128 | JAX (Parallel Forest) | 1.39 | 5000.96 | PASS |
| 128x128x128 | PyTorch (Parallel Forest) | 10.68 | 5062.42 | PASS |
| 128x128x128 | Triton (Parallel Forest) | 8.41 | 3285.04 | PASS |
| 128x128x128 | CUDA C++ (Numba-free) | 4.31 | 3253.42 | PASS |
### Gaussian Noise

| Grid Size | Backend | Prep (ms) | Compute (ms) | Accuracy |
|---|---|---|---|---|
| 16x16x16 | FlashCubical (C++) | 0.08 | 2.71 | Baseline |
| 16x16x16 | JAX (Parallel Forest) | 1.32 | 6.18 | PASS |
| 16x16x16 | PyTorch (Parallel Forest) | 0.32 | 42.09 | PASS |
| 16x16x16 | Triton (Parallel Forest) | 0.06 | 3.25 | PASS |
| 16x16x16 | CUDA C++ (Numba-free) | 0.04 | 2.77 | PASS |
| 32x32x32 | FlashCubical (C++) | 0.01 | 24.56 | Baseline |
| 32x32x32 | JAX (Parallel Forest) | 0.31 | 50.97 | PASS |
| 32x32x32 | PyTorch (Parallel Forest) | 0.26 | 442.55 | PASS |
| 32x32x32 | Triton (Parallel Forest) | 0.09 | 31.99 | PASS |
| 32x32x32 | CUDA C++ (Numba-free) | 0.14 | 29.03 | PASS |
| 64x64x64 | FlashCubical (C++) | 0.02 | 281.66 | Baseline |
| 64x64x64 | JAX (Parallel Forest) | 0.64 | 488.11 | PASS |
| 64x64x64 | PyTorch (Parallel Forest) | 1.13 | 716.72 | PASS |
| 64x64x64 | Triton (Parallel Forest) | 3.07 | 677.39 | PASS |
| 64x64x64 | CUDA C++ (Numba-free) | 0.87 | 285.35 | PASS |
| 128x128x128 | FlashCubical (C++) | 0.06 | 3050.95 | Baseline |
| 128x128x128 | JAX (Parallel Forest) | 1.30 | 4880.43 | PASS |
| 128x128x128 | PyTorch (Parallel Forest) | 15.55 | 4478.44 | PASS |
| 128x128x128 | Triton (Parallel Forest) | 8.83 | 4225.11 | PASS |
| 128x128x128 | CUDA C++ (Numba-free) | 15.94 | 3414.08 | PASS |
