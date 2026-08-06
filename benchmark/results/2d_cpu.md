# 2D Benchmarks - CPU

### Uniform Noise

| Grid Size | Backend | Prep (ms) | Compute (ms) | Accuracy |
|---|---|---|---|---|
| 32x32 | FlashCubical (C++) | 0.02 | 0.08 | Baseline |
| 32x32 | JAX (Parallel Forest) | 215.36 | 0.98 | PASS |
| 32x32 | PyTorch (Parallel Forest) | 1.62 | 1.39 | PASS |
| 32x32 | Triton (Parallel Forest) | 0.05 | 0.22 | PASS |
| 32x32 | CUDA C++ (Numba-free) | 0.04 | 0.07 | PASS |
| 64x64 | FlashCubical (C++) | 0.01 | 0.46 | Baseline |
| 64x64 | JAX (Parallel Forest) | 0.41 | 3.67 | PASS |
| 64x64 | PyTorch (Parallel Forest) | 0.21 | 62.18 | PASS |
| 64x64 | Triton (Parallel Forest) | 0.06 | 0.75 | PASS |
| 64x64 | CUDA C++ (Numba-free) | 0.05 | 0.49 | PASS |
| 128x128 | FlashCubical (C++) | 0.01 | 2.06 | Baseline |
| 128x128 | JAX (Parallel Forest) | 0.60 | 15.93 | PASS |
| 128x128 | PyTorch (Parallel Forest) | 0.23 | 63.99 | PASS |
| 128x128 | Triton (Parallel Forest) | 0.09 | 2.31 | PASS |
| 128x128 | CUDA C++ (Numba-free) | 0.05 | 2.06 | PASS |
| 256x256 | FlashCubical (C++) | 0.01 | 8.28 | Baseline |
| 256x256 | JAX (Parallel Forest) | 0.44 | 66.50 | PASS |
| 256x256 | PyTorch (Parallel Forest) | 15.93 | 137.23 | PASS |
| 256x256 | Triton (Parallel Forest) | 0.58 | 38.41 | PASS |
| 256x256 | CUDA C++ (Numba-free) | 0.45 | 8.28 | PASS |
| 512x512 | FlashCubical (C++) | 0.03 | 34.22 | Baseline |
| 512x512 | JAX (Parallel Forest) | 3.06 | 308.15 | PASS |
| 512x512 | PyTorch (Parallel Forest) | 2.81 | 397.19 | PASS |
| 512x512 | Triton (Parallel Forest) | 2.01 | 40.56 | PASS |
| 512x512 | CUDA C++ (Numba-free) | 1.15 | 34.19 | PASS |
| 1024x1024 | FlashCubical (C++) | 0.03 | 169.95 | Baseline |
| 1024x1024 | JAX (Parallel Forest) | 0.74 | 1429.50 | PASS |
| 1024x1024 | PyTorch (Parallel Forest) | 18.43 | 1576.04 | PASS |
| 1024x1024 | Triton (Parallel Forest) | 0.76 | 170.75 | PASS |
| 1024x1024 | CUDA C++ (Numba-free) | 1.35 | 150.80 | PASS |
### Gaussian Noise

| Grid Size | Backend | Prep (ms) | Compute (ms) | Accuracy |
|---|---|---|---|---|
| 32x32 | FlashCubical (C++) | 0.08 | 0.07 | Baseline |
| 32x32 | JAX (Parallel Forest) | 1.91 | 0.91 | PASS |
| 32x32 | PyTorch (Parallel Forest) | 0.20 | 1.40 | PASS |
| 32x32 | Triton (Parallel Forest) | 0.04 | 0.21 | PASS |
| 32x32 | CUDA C++ (Numba-free) | 0.04 | 0.07 | PASS |
| 64x64 | FlashCubical (C++) | 0.01 | 0.46 | Baseline |
| 64x64 | JAX (Parallel Forest) | 0.25 | 3.81 | PASS |
| 64x64 | PyTorch (Parallel Forest) | 0.06 | 25.46 | PASS |
| 64x64 | Triton (Parallel Forest) | 0.04 | 0.75 | PASS |
| 64x64 | CUDA C++ (Numba-free) | 0.04 | 0.48 | PASS |
| 128x128 | FlashCubical (C++) | 0.01 | 2.06 | Baseline |
| 128x128 | JAX (Parallel Forest) | 0.31 | 15.74 | PASS |
| 128x128 | PyTorch (Parallel Forest) | 0.07 | 61.62 | PASS |
| 128x128 | Triton (Parallel Forest) | 0.06 | 2.30 | PASS |
| 128x128 | CUDA C++ (Numba-free) | 0.05 | 2.04 | PASS |
| 256x256 | FlashCubical (C++) | 0.01 | 8.41 | Baseline |
| 256x256 | JAX (Parallel Forest) | 0.43 | 78.11 | PASS |
| 256x256 | PyTorch (Parallel Forest) | 2.14 | 87.91 | PASS |
| 256x256 | Triton (Parallel Forest) | 0.36 | 40.78 | PASS |
| 256x256 | CUDA C++ (Numba-free) | 0.37 | 8.43 | PASS |
| 512x512 | FlashCubical (C++) | 0.02 | 33.87 | Baseline |
| 512x512 | JAX (Parallel Forest) | 1.00 | 317.35 | PASS |
| 512x512 | PyTorch (Parallel Forest) | 16.04 | 432.61 | PASS |
| 512x512 | Triton (Parallel Forest) | 1.39 | 74.46 | PASS |
| 512x512 | CUDA C++ (Numba-free) | 1.26 | 33.50 | PASS |
| 1024x1024 | FlashCubical (C++) | 0.05 | 147.01 | Baseline |
| 1024x1024 | JAX (Parallel Forest) | 1.25 | 1433.88 | PASS |
| 1024x1024 | PyTorch (Parallel Forest) | 9.04 | 1453.98 | PASS |
| 1024x1024 | Triton (Parallel Forest) | 1.42 | 162.13 | PASS |
| 1024x1024 | CUDA C++ (Numba-free) | 8.77 | 142.26 | PASS |
