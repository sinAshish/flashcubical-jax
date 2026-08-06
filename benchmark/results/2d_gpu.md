# 2D Benchmarks - GPU (A100)

### Uniform Noise

| Grid Size | Backend | Prep (ms) | Compute (ms) | Accuracy |
|---|---|---|---|---|
| 32x32 | FlashCubical (C++) | 0.03 | 0.19 | Baseline |
| 32x32 | JAX (Parallel Forest) | 1068.54 | 3.44 | PASS |
| 32x32 | PyTorch (Parallel Forest) | 3.15 | 2.33 | PASS |
| 32x32 | Triton (Parallel Forest) | 0.14 | 2.63 | PASS |
| 32x32 | CUDA C++ (Numba-free) | 0.12 | 0.13 | PASS |
| 64x64 | FlashCubical (C++) | 0.01 | 0.47 | Baseline |
| 64x64 | JAX (Parallel Forest) | 0.96 | 10.32 | PASS |
| 64x64 | PyTorch (Parallel Forest) | 0.79 | 6.16 | PASS |
| 64x64 | Triton (Parallel Forest) | 0.11 | 6.67 | PASS |
| 64x64 | CUDA C++ (Numba-free) | 0.12 | 0.54 | PASS |
| 128x128 | FlashCubical (C++) | 0.01 | 2.02 | Baseline |
| 128x128 | JAX (Parallel Forest) | 2.81 | 17.72 | PASS |
| 128x128 | PyTorch (Parallel Forest) | 0.65 | 21.19 | PASS |
| 128x128 | Triton (Parallel Forest) | 0.14 | 22.85 | PASS |
| 128x128 | CUDA C++ (Numba-free) | 0.13 | 2.14 | PASS |
| 256x256 | FlashCubical (C++) | 0.02 | 8.34 | Baseline |
| 256x256 | JAX (Parallel Forest) | 0.74 | 67.89 | PASS |
| 256x256 | PyTorch (Parallel Forest) | 39.86 | 57.13 | PASS |
| 256x256 | Triton (Parallel Forest) | 12.75 | 96.39 | PASS |
| 256x256 | CUDA C++ (Numba-free) | 20.46 | 8.53 | PASS |
| 512x512 | FlashCubical (C++) | 0.02 | 33.98 | Baseline |
| 512x512 | JAX (Parallel Forest) | 1.22 | 304.43 | PASS |
| 512x512 | PyTorch (Parallel Forest) | 18.09 | 272.28 | PASS |
| 512x512 | Triton (Parallel Forest) | 17.57 | 496.23 | PASS |
| 512x512 | CUDA C++ (Numba-free) | 12.08 | 43.83 | PASS |
| 1024x1024 | FlashCubical (C++) | 0.11 | 161.44 | Baseline |
| 1024x1024 | JAX (Parallel Forest) | 8.63 | 1425.88 | PASS |
| 1024x1024 | PyTorch (Parallel Forest) | 36.32 | 1210.03 | PASS |
| 1024x1024 | Triton (Parallel Forest) | 9.81 | 2206.20 | PASS |
| 1024x1024 | CUDA C++ (Numba-free) | 11.45 | 146.16 | PASS |
### Gaussian Noise

| Grid Size | Backend | Prep (ms) | Compute (ms) | Accuracy |
|---|---|---|---|---|
| 32x32 | FlashCubical (C++) | 0.07 | 0.08 | Baseline |
| 32x32 | JAX (Parallel Forest) | 1.40 | 3.16 | PASS |
| 32x32 | PyTorch (Parallel Forest) | 0.28 | 2.27 | PASS |
| 32x32 | Triton (Parallel Forest) | 0.10 | 2.46 | PASS |
| 32x32 | CUDA C++ (Numba-free) | 0.09 | 0.13 | PASS |
| 64x64 | FlashCubical (C++) | 0.01 | 0.46 | Baseline |
| 64x64 | JAX (Parallel Forest) | 0.25 | 5.55 | PASS |
| 64x64 | PyTorch (Parallel Forest) | 0.12 | 4.70 | PASS |
| 64x64 | Triton (Parallel Forest) | 0.09 | 6.41 | PASS |
| 64x64 | CUDA C++ (Numba-free) | 0.10 | 0.52 | PASS |
| 128x128 | FlashCubical (C++) | 0.01 | 1.99 | Baseline |
| 128x128 | JAX (Parallel Forest) | 0.25 | 17.36 | PASS |
| 128x128 | PyTorch (Parallel Forest) | 0.20 | 14.29 | PASS |
| 128x128 | Triton (Parallel Forest) | 0.13 | 22.10 | PASS |
| 128x128 | CUDA C++ (Numba-free) | 0.13 | 2.12 | PASS |
| 256x256 | FlashCubical (C++) | 0.01 | 8.13 | Baseline |
| 256x256 | JAX (Parallel Forest) | 0.35 | 69.48 | PASS |
| 256x256 | PyTorch (Parallel Forest) | 1.10 | 56.90 | PASS |
| 256x256 | Triton (Parallel Forest) | 2.12 | 93.54 | PASS |
| 256x256 | CUDA C++ (Numba-free) | 16.79 | 8.53 | PASS |
| 512x512 | FlashCubical (C++) | 0.01 | 33.57 | Baseline |
| 512x512 | JAX (Parallel Forest) | 0.46 | 293.31 | PASS |
| 512x512 | PyTorch (Parallel Forest) | 3.45 | 264.75 | PASS |
| 512x512 | Triton (Parallel Forest) | 17.11 | 526.70 | PASS |
| 512x512 | CUDA C++ (Numba-free) | 15.82 | 39.46 | PASS |
| 1024x1024 | FlashCubical (C++) | 0.04 | 167.40 | Baseline |
| 1024x1024 | JAX (Parallel Forest) | 1.16 | 1534.26 | PASS |
| 1024x1024 | PyTorch (Parallel Forest) | 10.31 | 1300.48 | PASS |
| 1024x1024 | Triton (Parallel Forest) | 7.20 | 2496.88 | PASS |
| 1024x1024 | CUDA C++ (Numba-free) | 8.08 | 167.32 | PASS |
