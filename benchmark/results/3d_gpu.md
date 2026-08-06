# 3D Benchmarks - GPU (A100)

### Uniform Noise

| Grid Size | Backend | Prep (ms) | Compute (ms) | Accuracy |
|---|---|---|---|---|
| 16x16x16 | FlashCubical (C++) | 0.01 | 3.44 | Baseline |
| 16x16x16 | JAX (Parallel Forest) | 1.62 | 7.75 | PASS |
| 16x16x16 | PyTorch (Parallel Forest) | 0.31 | 6.70 | PASS |
| 16x16x16 | Triton (Parallel Forest) | ERROR | ERROR | NameError |
| 16x16x16 | CUDA C++ (Numba-free) | 0.10 | 2.28 | PASS |
| 32x32x32 | FlashCubical (C++) | 0.01 | 24.60 | Baseline |
| 32x32x32 | JAX (Parallel Forest) | 0.62 | 48.80 | PASS |
| 32x32x32 | PyTorch (Parallel Forest) | 0.40 | 44.07 | PASS |
| 32x32x32 | Triton (Parallel Forest) | ERROR | ERROR | NameError |
| 32x32x32 | CUDA C++ (Numba-free) | 0.17 | 3.04 | PASS |
| 64x64x64 | FlashCubical (C++) | 0.01 | 233.79 | Baseline |
| 64x64x64 | JAX (Parallel Forest) | 1.05 | 460.06 | PASS |
| 64x64x64 | PyTorch (Parallel Forest) | 5.13 | 415.40 | PASS |
| 64x64x64 | Triton (Parallel Forest) | ERROR | ERROR | NameError |
| 64x64x64 | CUDA C++ (Numba-free) | 19.30 | 8.32 | PASS |
| 128x128x128 | FlashCubical (C++) | 0.05 | 3746.88 | Baseline |
| 128x128x128 | JAX (Parallel Forest) | 1.71 | 4346.90 | PASS |
| 128x128x128 | PyTorch (Parallel Forest) | 15.71 | 4062.30 | PASS |
| 128x128x128 | Triton (Parallel Forest) | ERROR | ERROR | NameError |
| 128x128x128 | CUDA C++ (Numba-free) | 20.73 | 49.10 | PASS |
### Gaussian Noise

| Grid Size | Backend | Prep (ms) | Compute (ms) | Accuracy |
|---|---|---|---|---|
| 16x16x16 | FlashCubical (C++) | 0.05 | 2.63 | Baseline |
| 16x16x16 | JAX (Parallel Forest) | 1.79 | 6.66 | PASS |
| 16x16x16 | PyTorch (Parallel Forest) | 0.28 | 6.61 | PASS |
| 16x16x16 | Triton (Parallel Forest) | ERROR | ERROR | NameError |
| 16x16x16 | CUDA C++ (Numba-free) | 0.10 | 2.40 | PASS |
| 32x32x32 | FlashCubical (C++) | 0.01 | 24.89 | Baseline |
| 32x32x32 | JAX (Parallel Forest) | 0.77 | 51.46 | PASS |
| 32x32x32 | PyTorch (Parallel Forest) | 0.43 | 48.14 | PASS |
| 32x32x32 | Triton (Parallel Forest) | ERROR | ERROR | NameError |
| 32x32x32 | CUDA C++ (Numba-free) | 0.15 | 2.49 | PASS |
| 64x64x64 | FlashCubical (C++) | 0.01 | 249.79 | Baseline |
| 64x64x64 | JAX (Parallel Forest) | 0.51 | 454.50 | PASS |
| 64x64x64 | PyTorch (Parallel Forest) | 10.71 | 412.09 | PASS |
| 64x64x64 | Triton (Parallel Forest) | ERROR | ERROR | NameError |
| 64x64x64 | CUDA C++ (Numba-free) | 8.12 | 6.88 | PASS |
| 128x128x128 | FlashCubical (C++) | 0.06 | 2789.50 | Baseline |
| 128x128x128 | JAX (Parallel Forest) | 1.41 | 4294.16 | PASS |
| 128x128x128 | PyTorch (Parallel Forest) | 10.95 | 3696.72 | PASS |
| 128x128x128 | Triton (Parallel Forest) | ERROR | ERROR | NameError |
| 128x128x128 | CUDA C++ (Numba-free) | 20.33 | 48.72 | PASS |
