# 2D Benchmarks - CPU

### Uniform Noise

| Grid Size | Backend | Prep (ms) | Compute (ms) | Accuracy |
|---|---|---|---|---|
| 32x32 | FlashCubical (C++) | 0.01 | 0.08 | Baseline |
| 32x32 | JAX (Parallel Forest) | 144.88 | 2.12 | PASS |
| 32x32 | PyTorch (Parallel Forest) | 3.43 | 2.23 | PASS |
| 32x32 | Triton (Parallel Forest) | 0.05 | 0.08 | PASS |
| 32x32 | CUDA C++ (Numba-free) | 0.04 | 0.07 | PASS |
| 64x64 | FlashCubical (C++) | 0.01 | 0.45 | Baseline |
| 64x64 | JAX (Parallel Forest) | 0.91 | 10.35 | PASS |
| 64x64 | PyTorch (Parallel Forest) | 0.36 | 6.06 | PASS |
| 64x64 | Triton (Parallel Forest) | 0.09 | 0.52 | PASS |
| 64x64 | CUDA C++ (Numba-free) | 0.04 | 0.47 | PASS |
| 128x128 | FlashCubical (C++) | 0.01 | 2.01 | Baseline |
| 128x128 | JAX (Parallel Forest) | 0.43 | 49.44 | PASS |
| 128x128 | PyTorch (Parallel Forest) | 0.39 | 23.18 | PASS |
| 128x128 | Triton (Parallel Forest) | 0.07 | 2.06 | PASS |
| 128x128 | CUDA C++ (Numba-free) | 0.11 | 2.02 | PASS |
| 256x256 | FlashCubical (C++) | 0.01 | 8.24 | Baseline |
| 256x256 | JAX (Parallel Forest) | 0.48 | 143.05 | PASS |
| 256x256 | PyTorch (Parallel Forest) | 1.48 | 91.82 | PASS |
| 256x256 | Triton (Parallel Forest) | 0.49 | 8.40 | PASS |
| 256x256 | CUDA C++ (Numba-free) | 8.17 | 8.33 | PASS |
| 512x512 | FlashCubical (C++) | 0.01 | 35.92 | Baseline |
| 512x512 | JAX (Parallel Forest) | 1.26 | 587.78 | PASS |
| 512x512 | PyTorch (Parallel Forest) | 16.36 | 376.75 | PASS |
| 512x512 | Triton (Parallel Forest) | 5.18 | 34.07 | PASS |
| 512x512 | CUDA C++ (Numba-free) | 1.24 | 34.78 | PASS |
| 1024x1024 | FlashCubical (C++) | 0.06 | 163.85 | Baseline |
| 1024x1024 | JAX (Parallel Forest) | 3.73 | 2420.48 | PASS |
| 1024x1024 | PyTorch (Parallel Forest) | 7.26 | 1390.03 | PASS |
| 1024x1024 | Triton (Parallel Forest) | 8.71 | 162.28 | PASS |
| 1024x1024 | CUDA C++ (Numba-free) | 2.87 | 157.41 | PASS |
### Gaussian Noise

| Grid Size | Backend | Prep (ms) | Compute (ms) | Accuracy |
|---|---|---|---|---|
| 32x32 | FlashCubical (C++) | 0.06 | 0.09 | Baseline |
| 32x32 | JAX (Parallel Forest) | 1.32 | 2.93 | PASS |
| 32x32 | PyTorch (Parallel Forest) | 0.27 | 1.37 | PASS |
| 32x32 | Triton (Parallel Forest) | 0.04 | 0.08 | PASS |
| 32x32 | CUDA C++ (Numba-free) | 0.04 | 0.07 | PASS |
| 64x64 | FlashCubical (C++) | 0.00 | 0.46 | Baseline |
| 64x64 | JAX (Parallel Forest) | 0.22 | 10.95 | PASS |
| 64x64 | PyTorch (Parallel Forest) | 0.11 | 9.46 | PASS |
| 64x64 | Triton (Parallel Forest) | 0.04 | 0.54 | PASS |
| 64x64 | CUDA C++ (Numba-free) | 0.04 | 0.50 | PASS |
| 128x128 | FlashCubical (C++) | 0.01 | 2.05 | Baseline |
| 128x128 | JAX (Parallel Forest) | 0.32 | 35.78 | PASS |
| 128x128 | PyTorch (Parallel Forest) | 0.14 | 25.37 | PASS |
| 128x128 | Triton (Parallel Forest) | 0.06 | 2.07 | PASS |
| 128x128 | CUDA C++ (Numba-free) | 0.04 | 2.04 | PASS |
| 256x256 | FlashCubical (C++) | 0.01 | 8.26 | Baseline |
| 256x256 | JAX (Parallel Forest) | 0.28 | 124.95 | PASS |
| 256x256 | PyTorch (Parallel Forest) | 0.95 | 79.24 | PASS |
| 256x256 | Triton (Parallel Forest) | 0.51 | 8.88 | PASS |
| 256x256 | CUDA C++ (Numba-free) | 0.44 | 8.18 | PASS |
| 512x512 | FlashCubical (C++) | 0.02 | 33.54 | Baseline |
| 512x512 | JAX (Parallel Forest) | 1.14 | 560.39 | PASS |
| 512x512 | PyTorch (Parallel Forest) | 17.73 | 327.51 | PASS |
| 512x512 | Triton (Parallel Forest) | 2.76 | 33.93 | PASS |
| 512x512 | CUDA C++ (Numba-free) | 0.85 | 34.28 | PASS |
| 1024x1024 | FlashCubical (C++) | 0.05 | 156.22 | Baseline |
| 1024x1024 | JAX (Parallel Forest) | 0.91 | 2404.10 | PASS |
| 1024x1024 | PyTorch (Parallel Forest) | 5.46 | 1346.53 | PASS |
| 1024x1024 | Triton (Parallel Forest) | 7.98 | 148.96 | PASS |
| 1024x1024 | CUDA C++ (Numba-free) | 4.38 | 144.77 | PASS |
