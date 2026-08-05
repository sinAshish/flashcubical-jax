# 2D Benchmarks - GPU (A100)

### Uniform Noise

| Grid Size | Backend | Prep (ms) | Compute (ms) | Accuracy |
|---|---|---|---|---|
| 32x32 | FlashCubical (C++) | 0.01 | 0.08 | Baseline |
| 32x32 | JAX (Parallel Forest) | 1929.38 | 2.56 | PASS |
| 32x32 | PyTorch (Parallel Forest) | 1.96 | 1.51 | PASS |
| 32x32 | Triton (Parallel Forest) | 0.06 | 0.09 | PASS |
| 32x32 | CUDA C++ (Numba-free) | 0.06 | 0.07 | PASS |
| 64x64 | FlashCubical (C++) | 0.01 | 0.48 | Baseline |
| 64x64 | JAX (Parallel Forest) | 0.55 | 5.00 | PASS |
| 64x64 | PyTorch (Parallel Forest) | 0.14 | 397.90 | PASS |
| 64x64 | Triton (Parallel Forest) | 0.19 | 0.68 | PASS |
| 64x64 | CUDA C++ (Numba-free) | 0.14 | 0.47 | PASS |
| 128x128 | FlashCubical (C++) | 0.02 | 2.06 | Baseline |
| 128x128 | JAX (Parallel Forest) | 1.61 | 15.53 | PASS |
| 128x128 | PyTorch (Parallel Forest) | 0.16 | 431.83 | PASS |
| 128x128 | Triton (Parallel Forest) | 0.11 | 2.10 | PASS |
| 128x128 | CUDA C++ (Numba-free) | 0.06 | 2.07 | PASS |
| 256x256 | FlashCubical (C++) | 0.02 | 8.43 | Baseline |
| 256x256 | JAX (Parallel Forest) | 1.53 | 53.57 | PASS |
| 256x256 | PyTorch (Parallel Forest) | 8.08 | 91.04 | PASS |
| 256x256 | Triton (Parallel Forest) | 4.50 | 8.30 | PASS |
| 256x256 | CUDA C++ (Numba-free) | 0.35 | 8.24 | PASS |
| 512x512 | FlashCubical (C++) | 0.02 | 34.11 | Baseline |
| 512x512 | JAX (Parallel Forest) | 1.26 | 255.28 | PASS |
| 512x512 | PyTorch (Parallel Forest) | 9.17 | 393.05 | PASS |
| 512x512 | Triton (Parallel Forest) | 6.18 | 34.68 | PASS |
| 512x512 | CUDA C++ (Numba-free) | 2.04 | 40.93 | PASS |
| 1024x1024 | FlashCubical (C++) | 0.13 | 201.10 | Baseline |
| 1024x1024 | JAX (Parallel Forest) | 8.56 | 1054.37 | PASS |
| 1024x1024 | PyTorch (Parallel Forest) | 16.67 | 3443.78 | PASS |
| 1024x1024 | Triton (Parallel Forest) | 8.96 | 142.89 | PASS |
| 1024x1024 | CUDA C++ (Numba-free) | 8.08 | 144.85 | PASS |
### Gaussian Noise

| Grid Size | Backend | Prep (ms) | Compute (ms) | Accuracy |
|---|---|---|---|---|
| 32x32 | FlashCubical (C++) | 0.07 | 0.07 | Baseline |
| 32x32 | JAX (Parallel Forest) | 2.01 | 3.06 | PASS |
| 32x32 | PyTorch (Parallel Forest) | 0.35 | 1.40 | PASS |
| 32x32 | Triton (Parallel Forest) | 0.05 | 0.08 | PASS |
| 32x32 | CUDA C++ (Numba-free) | 0.04 | 0.07 | PASS |
| 64x64 | FlashCubical (C++) | 0.00 | 0.45 | Baseline |
| 64x64 | JAX (Parallel Forest) | 0.32 | 4.89 | PASS |
| 64x64 | PyTorch (Parallel Forest) | 0.05 | 379.74 | PASS |
| 64x64 | Triton (Parallel Forest) | 0.07 | 0.55 | PASS |
| 64x64 | CUDA C++ (Numba-free) | 0.04 | 0.48 | PASS |
| 128x128 | FlashCubical (C++) | 0.01 | 2.09 | Baseline |
| 128x128 | JAX (Parallel Forest) | 0.91 | 13.23 | PASS |
| 128x128 | PyTorch (Parallel Forest) | 0.08 | 431.93 | PASS |
| 128x128 | Triton (Parallel Forest) | 0.10 | 2.08 | PASS |
| 128x128 | CUDA C++ (Numba-free) | 0.05 | 2.08 | PASS |
| 256x256 | FlashCubical (C++) | 0.01 | 8.25 | Baseline |
| 256x256 | JAX (Parallel Forest) | 2.13 | 50.61 | PASS |
| 256x256 | PyTorch (Parallel Forest) | 16.17 | 1711.67 | PASS |
| 256x256 | Triton (Parallel Forest) | 16.64 | 8.49 | PASS |
| 256x256 | CUDA C++ (Numba-free) | 15.98 | 8.36 | PASS |
| 512x512 | FlashCubical (C++) | 0.02 | 33.66 | Baseline |
| 512x512 | JAX (Parallel Forest) | 1.11 | 215.43 | PASS |
| 512x512 | PyTorch (Parallel Forest) | 8.63 | 5282.91 | PASS |
| 512x512 | Triton (Parallel Forest) | 15.62 | 33.90 | PASS |
| 512x512 | CUDA C++ (Numba-free) | 9.48 | 33.55 | PASS |
| 1024x1024 | FlashCubical (C++) | 0.05 | 143.50 | Baseline |
| 1024x1024 | JAX (Parallel Forest) | 6.48 | 1087.69 | PASS |
| 1024x1024 | PyTorch (Parallel Forest) | 10.83 | 2933.21 | PASS |
| 1024x1024 | Triton (Parallel Forest) | 9.16 | 165.06 | PASS |
| 1024x1024 | CUDA C++ (Numba-free) | 16.16 | 146.31 | PASS |
