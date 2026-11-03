# 2D Benchmarks - CPU

### Uniform Noise

| Grid Size   | Backend                   | Prep (ms) | Compute (ms) | Accuracy |
|-------------|---------------------------|-----------|--------------|----------|
| 32x32       | FlashCubical (C++)        | 0.00      | 0.05         | Baseline |
| 32x32       | JAX (Parallel Forest)     | 0.20      | 0.06         | PASS     |
| 32x32       | PyTorch (Parallel Forest) | 0.04      | 0.16         | PASS     |
| 32x32       | Triton (Parallel Forest)  | 0.01      | 0.06         | PASS     |
| 32x32       | CUDA C++ (Numba-free)     | 0.01      | 0.05         | PASS     |
| 64x64       | FlashCubical (C++)        | 0.00      | 0.22         | Baseline |
| 64x64       | JAX (Parallel Forest)     | 0.18      | 0.25         | PASS     |
| 64x64       | PyTorch (Parallel Forest) | 0.04      | 0.52         | PASS     |
| 64x64       | Triton (Parallel Forest)  | 0.01      | 0.27         | PASS     |
| 64x64       | CUDA C++ (Numba-free)     | 0.01      | 0.19         | PASS     |
| 128x128     | FlashCubical (C++)        | 0.00      | 1.47         | Baseline |
| 128x128     | JAX (Parallel Forest)     | 0.22      | 0.74         | PASS     |
| 128x128     | PyTorch (Parallel Forest) | 0.04      | 1.09         | PASS     |
| 128x128     | Triton (Parallel Forest)  | 0.01      | 1.44         | PASS     |
| 128x128     | CUDA C++ (Numba-free)     | 0.01      | 1.30         | PASS     |
| 256x256     | FlashCubical (C++)        | 0.00      | 5.89         | Baseline |
| 256x256     | JAX (Parallel Forest)     | 0.28      | 2.46         | PASS     |
| 256x256     | PyTorch (Parallel Forest) | 0.04      | 4.46         | PASS     |
| 256x256     | Triton (Parallel Forest)  | 0.02      | 6.17         | PASS     |
| 256x256     | CUDA C++ (Numba-free)     | 0.03      | 6.30         | PASS     |
| 512x512     | FlashCubical (C++)        | 0.00      | 24.98        | Baseline |
| 512x512     | JAX (Parallel Forest)     | 0.30      | 9.83         | PASS     |
| 512x512     | PyTorch (Parallel Forest) | 0.06      | 12.40        | PASS     |
| 512x512     | Triton (Parallel Forest)  | 0.02      | 25.61        | PASS     |
| 512x512     | CUDA C++ (Numba-free)     | 0.03      | 25.55        | PASS     |
| 1024x1024   | FlashCubical (C++)        | 0.01      | 106.34       | Baseline |
| 1024x1024   | JAX (Parallel Forest)     | 0.35      | 44.45        | PASS     |
| 1024x1024   | PyTorch (Parallel Forest) | 0.10      | 51.65        | PASS     |
| 1024x1024   | Triton (Parallel Forest)  | 0.04      | 114.14       | PASS     |
| 1024x1024   | CUDA C++ (Numba-free)     | 0.05      | 109.75       | PASS     |

### Gaussian Noise

| Grid Size   | Backend                   | Prep (ms) | Compute (ms) | Accuracy |
|-------------|---------------------------|-----------|--------------|----------|
| 32x32       | FlashCubical (C++)        | 0.01      | 0.05         | Baseline |
| 32x32       | JAX (Parallel Forest)     | 0.44      | 0.06         | PASS     |
| 32x32       | PyTorch (Parallel Forest) | 0.04      | 0.15         | PASS     |
| 32x32       | Triton (Parallel Forest)  | 0.01      | 0.05         | PASS     |
| 32x32       | CUDA C++ (Numba-free)     | 0.00      | 0.05         | PASS     |
| 64x64       | FlashCubical (C++)        | 0.00      | 0.24         | Baseline |
| 64x64       | JAX (Parallel Forest)     | 0.12      | 0.27         | PASS     |
| 64x64       | PyTorch (Parallel Forest) | 0.01      | 0.58         | PASS     |
| 64x64       | Triton (Parallel Forest)  | 0.01      | 0.20         | PASS     |
| 64x64       | CUDA C++ (Numba-free)     | 0.00      | 0.19         | PASS     |
| 128x128     | FlashCubical (C++)        | 0.00      | 1.36         | Baseline |
| 128x128     | JAX (Parallel Forest)     | 0.11      | 0.67         | PASS     |
| 128x128     | PyTorch (Parallel Forest) | 0.01      | 1.06         | PASS     |
| 128x128     | Triton (Parallel Forest)  | 0.01      | 1.45         | PASS     |
| 128x128     | CUDA C++ (Numba-free)     | 0.01      | 1.29         | PASS     |
| 256x256     | FlashCubical (C++)        | 0.00      | 6.05         | Baseline |
| 256x256     | JAX (Parallel Forest)     | 0.24      | 2.42         | PASS     |
| 256x256     | PyTorch (Parallel Forest) | 0.03      | 3.94         | PASS     |
| 256x256     | Triton (Parallel Forest)  | 0.01      | 6.11         | PASS     |
| 256x256     | CUDA C++ (Numba-free)     | 0.01      | 5.97         | PASS     |
| 512x512     | FlashCubical (C++)        | 0.00      | 24.55        | Baseline |
| 512x512     | JAX (Parallel Forest)     | 0.26      | 9.31         | PASS     |
| 512x512     | PyTorch (Parallel Forest) | 0.04      | 12.21        | PASS     |
| 512x512     | Triton (Parallel Forest)  | 0.03      | 25.68        | PASS     |
| 512x512     | CUDA C++ (Numba-free)     | 0.02      | 26.02        | PASS     |
| 1024x1024   | FlashCubical (C++)        | 0.01      | 124.39       | Baseline |
| 1024x1024   | JAX (Parallel Forest)     | 0.30      | 39.73        | PASS     |
| 1024x1024   | PyTorch (Parallel Forest) | 0.06      | 60.90        | PASS     |
| 1024x1024   | Triton (Parallel Forest)  | 0.04      | 110.61       | PASS     |
| 1024x1024   | CUDA C++ (Numba-free)     | 0.04      | 108.42       | PASS     |
