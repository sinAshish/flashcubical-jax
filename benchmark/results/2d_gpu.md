# 2D Benchmarks - GPU (A100)

### Uniform Noise

| Grid Size | Backend | Prep (ms) | Compute (ms) | Accuracy |
|---|---|---|---|---|
| 32x32 | FlashCubical (C++) | 0.01 | 0.08 | Baseline |
| 32x32 | JAX (Parallel Forest) | 1186.39 | 3.75 | PASS |
| 32x32 | PyTorch (Parallel Forest) | 3.60 | 2.25 | PASS |
| 32x32 | Triton (Parallel Forest) | 0.07 | 0.15 | PASS |
| 32x32 | CUDA C++ (Numba-free) | 0.06 | 0.12 | PASS |
| 64x64 | FlashCubical (C++) | 0.02 | 0.45 | Baseline |
| 64x64 | JAX (Parallel Forest) | 0.41 | 9.14 | PASS |
| 64x64 | PyTorch (Parallel Forest) | 0.29 | 12.17 | PASS |
| 64x64 | Triton (Parallel Forest) | 0.04 | 0.54 | PASS |
| 64x64 | CUDA C++ (Numba-free) | 0.05 | 0.51 | PASS |
| 128x128 | FlashCubical (C++) | 0.01 | 2.03 | Baseline |
| 128x128 | JAX (Parallel Forest) | 0.38 | 16.69 | PASS |
| 128x128 | PyTorch (Parallel Forest) | 0.79 | 39.12 | PASS |
| 128x128 | Triton (Parallel Forest) | 0.08 | 2.11 | PASS |
| 128x128 | CUDA C++ (Numba-free) | 0.05 | 2.02 | PASS |
| 256x256 | FlashCubical (C++) | 0.01 | 8.28 | Baseline |
| 256x256 | JAX (Parallel Forest) | 0.50 | 53.60 | PASS |
| 256x256 | PyTorch (Parallel Forest) | 3.27 | 106.21 | PASS |
| 256x256 | Triton (Parallel Forest) | 7.08 | 8.23 | PASS |
| 256x256 | CUDA C++ (Numba-free) | 7.02 | 8.22 | PASS |
| 512x512 | FlashCubical (C++) | 0.04 | 34.63 | Baseline |
| 512x512 | JAX (Parallel Forest) | 2.30 | 240.30 | PASS |
| 512x512 | PyTorch (Parallel Forest) | 13.02 | 322.08 | PASS |
| 512x512 | Triton (Parallel Forest) | 1.16 | 38.69 | PASS |
| 512x512 | CUDA C++ (Numba-free) | 1.29 | 36.22 | PASS |
| 1024x1024 | FlashCubical (C++) | 0.04 | 191.47 | Baseline |
| 1024x1024 | JAX (Parallel Forest) | 8.05 | 1093.98 | PASS |
| 1024x1024 | PyTorch (Parallel Forest) | 8.11 | 1406.10 | PASS |
| 1024x1024 | Triton (Parallel Forest) | 8.25 | 193.45 | PASS |
| 1024x1024 | CUDA C++ (Numba-free) | 4.40 | 168.87 | PASS |
### Gaussian Noise

| Grid Size | Backend | Prep (ms) | Compute (ms) | Accuracy |
|---|---|---|---|---|
| 32x32 | FlashCubical (C++) | 0.05 | 0.08 | Baseline |
| 32x32 | JAX (Parallel Forest) | 1.38 | 3.55 | PASS |
| 32x32 | PyTorch (Parallel Forest) | 0.20 | 1.62 | PASS |
| 32x32 | Triton (Parallel Forest) | 0.04 | 0.08 | PASS |
| 32x32 | CUDA C++ (Numba-free) | 0.03 | 0.07 | PASS |
| 64x64 | FlashCubical (C++) | 0.01 | 0.45 | Baseline |
| 64x64 | JAX (Parallel Forest) | 0.29 | 5.33 | PASS |
| 64x64 | PyTorch (Parallel Forest) | 0.05 | 6.76 | PASS |
| 64x64 | Triton (Parallel Forest) | 0.04 | 0.53 | PASS |
| 64x64 | CUDA C++ (Numba-free) | 0.04 | 0.49 | PASS |
| 128x128 | FlashCubical (C++) | 0.01 | 2.47 | Baseline |
| 128x128 | JAX (Parallel Forest) | 0.50 | 13.32 | PASS |
| 128x128 | PyTorch (Parallel Forest) | 0.22 | 30.97 | PASS |
| 128x128 | Triton (Parallel Forest) | 0.10 | 2.60 | PASS |
| 128x128 | CUDA C++ (Numba-free) | 0.09 | 2.25 | PASS |
| 256x256 | FlashCubical (C++) | 0.01 | 8.65 | Baseline |
| 256x256 | JAX (Parallel Forest) | 0.60 | 53.98 | PASS |
| 256x256 | PyTorch (Parallel Forest) | 0.92 | 173.05 | PASS |
| 256x256 | Triton (Parallel Forest) | 0.99 | 9.02 | PASS |
| 256x256 | CUDA C++ (Numba-free) | 0.43 | 9.62 | PASS |
| 512x512 | FlashCubical (C++) | 0.01 | 35.63 | Baseline |
| 512x512 | JAX (Parallel Forest) | 0.74 | 253.54 | PASS |
| 512x512 | PyTorch (Parallel Forest) | 1.84 | 344.75 | PASS |
| 512x512 | Triton (Parallel Forest) | 1.17 | 34.25 | PASS |
| 512x512 | CUDA C++ (Numba-free) | 1.11 | 33.52 | PASS |
| 1024x1024 | FlashCubical (C++) | 0.03 | 154.23 | Baseline |
| 1024x1024 | JAX (Parallel Forest) | 0.79 | 1073.04 | PASS |
| 1024x1024 | PyTorch (Parallel Forest) | 18.30 | 1373.23 | PASS |
| 1024x1024 | Triton (Parallel Forest) | 16.89 | 161.20 | PASS |
| 1024x1024 | CUDA C++ (Numba-free) | 7.94 | 162.72 | PASS |
