# 2D Benchmarks - CPU

### Uniform Noise

| Grid Size | Backend | Prep (ms) | Compute (ms) | Accuracy |
|---|---|---|---|---|
| 32x32 | FlashCubical (C++) | 0.01 | 0.08 | Baseline |
| 32x32 | JAX (Parallel Forest) | 171.59 | 2.62 | PASS |
| 32x32 | PyTorch (Parallel Forest) | 1.78 | 1.27 | PASS |
| 32x32 | Triton (Parallel Forest) | 0.05 | 0.09 | PASS |
| 32x32 | CUDA C++ (Numba-free) | 0.04 | 0.07 | PASS |
| 64x64 | FlashCubical (C++) | 0.01 | 0.45 | Baseline |
| 64x64 | JAX (Parallel Forest) | 0.41 | 10.26 | PASS |
| 64x64 | PyTorch (Parallel Forest) | 0.25 | 11.30 | PASS |
| 64x64 | Triton (Parallel Forest) | 0.05 | 0.55 | PASS |
| 64x64 | CUDA C++ (Numba-free) | 0.05 | 0.51 | PASS |
| 128x128 | FlashCubical (C++) | 0.01 | 2.02 | Baseline |
| 128x128 | JAX (Parallel Forest) | 0.40 | 49.50 | PASS |
| 128x128 | PyTorch (Parallel Forest) | 0.32 | 42.37 | PASS |
| 128x128 | Triton (Parallel Forest) | 0.06 | 2.07 | PASS |
| 128x128 | CUDA C++ (Numba-free) | 0.05 | 2.04 | PASS |
| 256x256 | FlashCubical (C++) | 0.01 | 8.30 | Baseline |
| 256x256 | JAX (Parallel Forest) | 0.61 | 154.41 | PASS |
| 256x256 | PyTorch (Parallel Forest) | 4.63 | 83.72 | PASS |
| 256x256 | Triton (Parallel Forest) | 0.51 | 8.31 | PASS |
| 256x256 | CUDA C++ (Numba-free) | 0.51 | 8.30 | PASS |
| 512x512 | FlashCubical (C++) | 0.06 | 34.25 | Baseline |
| 512x512 | JAX (Parallel Forest) | 3.46 | 600.91 | PASS |
| 512x512 | PyTorch (Parallel Forest) | 4.72 | 404.94 | PASS |
| 512x512 | Triton (Parallel Forest) | 1.94 | 41.13 | PASS |
| 512x512 | CUDA C++ (Numba-free) | 1.16 | 36.53 | PASS |
| 1024x1024 | FlashCubical (C++) | 0.04 | 204.27 | Baseline |
| 1024x1024 | JAX (Parallel Forest) | 1.72 | 2538.05 | PASS |
| 1024x1024 | PyTorch (Parallel Forest) | 10.94 | 2896.72 | PASS |
| 1024x1024 | Triton (Parallel Forest) | 8.07 | 152.23 | PASS |
| 1024x1024 | CUDA C++ (Numba-free) | 3.36 | 149.46 | PASS |
### Gaussian Noise

| Grid Size | Backend | Prep (ms) | Compute (ms) | Accuracy |
|---|---|---|---|---|
| 32x32 | FlashCubical (C++) | 0.05 | 0.08 | Baseline |
| 32x32 | JAX (Parallel Forest) | 1.26 | 1.73 | PASS |
| 32x32 | PyTorch (Parallel Forest) | 0.20 | 1.42 | PASS |
| 32x32 | Triton (Parallel Forest) | 0.04 | 0.08 | PASS |
| 32x32 | CUDA C++ (Numba-free) | 0.03 | 0.07 | PASS |
| 64x64 | FlashCubical (C++) | 0.00 | 0.46 | Baseline |
| 64x64 | JAX (Parallel Forest) | 0.24 | 8.18 | PASS |
| 64x64 | PyTorch (Parallel Forest) | 0.10 | 402.16 | PASS |
| 64x64 | Triton (Parallel Forest) | 0.07 | 0.57 | PASS |
| 64x64 | CUDA C++ (Numba-free) | 0.04 | 0.50 | PASS |
| 128x128 | FlashCubical (C++) | 0.01 | 2.07 | Baseline |
| 128x128 | JAX (Parallel Forest) | 0.46 | 33.65 | PASS |
| 128x128 | PyTorch (Parallel Forest) | 0.10 | 415.85 | PASS |
| 128x128 | Triton (Parallel Forest) | 0.08 | 2.15 | PASS |
| 128x128 | CUDA C++ (Numba-free) | 0.07 | 2.12 | PASS |
| 256x256 | FlashCubical (C++) | 0.02 | 8.28 | Baseline |
| 256x256 | JAX (Parallel Forest) | 0.98 | 130.78 | PASS |
| 256x256 | PyTorch (Parallel Forest) | 8.22 | 4752.93 | PASS |
| 256x256 | Triton (Parallel Forest) | 30.46 | 9.62 | PASS |
| 256x256 | CUDA C++ (Numba-free) | 16.34 | 8.49 | PASS |
| 512x512 | FlashCubical (C++) | 0.03 | 35.14 | Baseline |
| 512x512 | JAX (Parallel Forest) | 1.49 | 599.42 | PASS |
| 512x512 | PyTorch (Parallel Forest) | 17.87 | 1914.73 | PASS |
| 512x512 | Triton (Parallel Forest) | 8.31 | 33.60 | PASS |
| 512x512 | CUDA C++ (Numba-free) | 7.85 | 33.30 | PASS |
| 1024x1024 | FlashCubical (C++) | 0.03 | 144.99 | Baseline |
| 1024x1024 | JAX (Parallel Forest) | 1.38 | 2396.93 | PASS |
| 1024x1024 | PyTorch (Parallel Forest) | 16.09 | 3671.10 | PASS |
| 1024x1024 | Triton (Parallel Forest) | 16.83 | 190.96 | PASS |
| 1024x1024 | CUDA C++ (Numba-free) | 10.48 | 157.14 | PASS |
