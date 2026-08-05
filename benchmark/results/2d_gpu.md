# 2D Benchmarks - GPU (A100)

### Uniform Noise

| Grid Size | Backend | Prep (ms) | Compute (ms) | Accuracy |
|---|---|---|---|---|
| 32x32 | FlashCubical (C++) | 0.01 | 0.09 | Baseline |
| 32x32 | JAX (Parallel Forest) | 929.71 | 2.11 | PASS |
| 32x32 | PyTorch (Parallel Forest) | 31.39 | 2.59 | PASS |
| 32x32 | Triton (Parallel Forest) | 0.15 | 0.35 | FAIL |
| 32x32 | CUDA C++ (Numba-free) | 0.14 | 0.18 | PASS |
| 64x64 | FlashCubical (C++) | 0.03 | 0.58 | Baseline |
| 64x64 | JAX (Parallel Forest) | 0.80 | 8.93 | PASS |
| 64x64 | PyTorch (Parallel Forest) | 0.73 | 8.50 | PASS |
| 64x64 | Triton (Parallel Forest) | 0.27 | 0.41 | FAIL |
| 64x64 | CUDA C++ (Numba-free) | 0.18 | 0.53 | PASS |
| 128x128 | FlashCubical (C++) | 0.01 | 2.01 | Baseline |
| 128x128 | JAX (Parallel Forest) | 0.69 | 16.12 | PASS |
| 128x128 | PyTorch (Parallel Forest) | 0.45 | 12.19 | PASS |
| 128x128 | Triton (Parallel Forest) | 0.12 | 1.20 | FAIL |
| 128x128 | CUDA C++ (Numba-free) | 0.14 | 2.11 | PASS |
| 256x256 | FlashCubical (C++) | 0.01 | 9.21 | Baseline |
| 256x256 | JAX (Parallel Forest) | 0.56 | 59.47 | PASS |
| 256x256 | PyTorch (Parallel Forest) | 25.54 | 58.65 | PASS |
| 256x256 | Triton (Parallel Forest) | 18.96 | 0.49 | FAIL |
| 256x256 | CUDA C++ (Numba-free) | 8.27 | 8.49 | PASS |
| 512x512 | FlashCubical (C++) | 0.01 | 36.05 | Baseline |
| 512x512 | JAX (Parallel Forest) | 0.79 | 258.35 | PASS |
| 512x512 | PyTorch (Parallel Forest) | 18.63 | 248.09 | PASS |
| 512x512 | Triton (Parallel Forest) | 20.69 | 4.94 | FAIL |
| 512x512 | CUDA C++ (Numba-free) | 10.85 | 39.86 | PASS |
| 1024x1024 | FlashCubical (C++) | 0.09 | 190.21 | Baseline |
| 1024x1024 | JAX (Parallel Forest) | 17.79 | 1158.58 | PASS |
| 1024x1024 | PyTorch (Parallel Forest) | 36.29 | 1035.64 | PASS |
| 1024x1024 | Triton (Parallel Forest) | 19.23 | 7.09 | FAIL |
| 1024x1024 | CUDA C++ (Numba-free) | 14.52 | 159.25 | PASS |
### Gaussian Noise

| Grid Size | Backend | Prep (ms) | Compute (ms) | Accuracy |
|---|---|---|---|---|
| 32x32 | FlashCubical (C++) | 0.06 | 0.08 | Baseline |
| 32x32 | JAX (Parallel Forest) | 1.66 | 4.02 | PASS |
| 32x32 | PyTorch (Parallel Forest) | 0.37 | 2.37 | PASS |
| 32x32 | Triton (Parallel Forest) | 0.12 | 0.30 | FAIL |
| 32x32 | CUDA C++ (Numba-free) | 0.10 | 0.12 | PASS |
| 64x64 | FlashCubical (C++) | 0.01 | 0.45 | Baseline |
| 64x64 | JAX (Parallel Forest) | 0.53 | 9.97 | PASS |
| 64x64 | PyTorch (Parallel Forest) | 0.41 | 4.47 | PASS |
| 64x64 | Triton (Parallel Forest) | 0.14 | 0.31 | FAIL |
| 64x64 | CUDA C++ (Numba-free) | 0.14 | 0.51 | PASS |
| 128x128 | FlashCubical (C++) | 0.01 | 2.01 | Baseline |
| 128x128 | JAX (Parallel Forest) | 0.70 | 21.48 | PASS |
| 128x128 | PyTorch (Parallel Forest) | 0.49 | 17.79 | PASS |
| 128x128 | Triton (Parallel Forest) | 0.19 | 0.38 | FAIL |
| 128x128 | CUDA C++ (Numba-free) | 0.10 | 2.09 | PASS |
| 256x256 | FlashCubical (C++) | 0.01 | 8.13 | Baseline |
| 256x256 | JAX (Parallel Forest) | 0.48 | 50.47 | PASS |
| 256x256 | PyTorch (Parallel Forest) | 20.60 | 55.82 | PASS |
| 256x256 | Triton (Parallel Forest) | 9.85 | 0.60 | FAIL |
| 256x256 | CUDA C++ (Numba-free) | 12.89 | 12.71 | PASS |
| 512x512 | FlashCubical (C++) | 0.04 | 36.68 | Baseline |
| 512x512 | JAX (Parallel Forest) | 0.58 | 257.13 | PASS |
| 512x512 | PyTorch (Parallel Forest) | 14.79 | 225.61 | PASS |
| 512x512 | Triton (Parallel Forest) | 20.53 | 3.02 | FAIL |
| 512x512 | CUDA C++ (Numba-free) | 9.03 | 36.39 | PASS |
| 1024x1024 | FlashCubical (C++) | 0.06 | 158.88 | Baseline |
| 1024x1024 | JAX (Parallel Forest) | 1.53 | 1123.31 | PASS |
| 1024x1024 | PyTorch (Parallel Forest) | 19.92 | 1039.20 | PASS |
| 1024x1024 | Triton (Parallel Forest) | 20.54 | 14.25 | FAIL |
| 1024x1024 | CUDA C++ (Numba-free) | 20.06 | 191.72 | PASS |
