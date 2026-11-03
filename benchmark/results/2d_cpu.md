# 2D Benchmarks - CPU

### Uniform Noise

| Grid Size   | Backend                   | Prep (ms) | Compute (ms) | Accuracy |
|-------------|---------------------------|-----------|--------------|----------|
| 32x32       | FlashCubical (C++)        | 0.00      | 0.05         | Baseline |
| 32x32       | JAX (Parallel Forest)     | 0.17      | 0.39         | PASS     |
| 32x32       | PyTorch (Parallel Forest) | 0.03      | 0.55         | PASS     |
| 32x32       | Triton (Parallel Forest)  | 0.01      | 0.06         | PASS     |
| 32x32       | CUDA C++ (Numba-free)     | 0.00      | 0.05         | PASS     |
| 64x64       | FlashCubical (C++)        | 0.00      | 0.24         | Baseline |
| 64x64       | JAX (Parallel Forest)     | 0.16      | 1.53         | PASS     |
| 64x64       | PyTorch (Parallel Forest) | 0.03      | 2.66         | PASS     |
| 64x64       | Triton (Parallel Forest)  | 0.01      | 0.25         | PASS     |
| 64x64       | CUDA C++ (Numba-free)     | 0.01      | 0.20         | PASS     |
| 128x128     | FlashCubical (C++)        | 0.00      | 1.32         | Baseline |
| 128x128     | JAX (Parallel Forest)     | 0.21      | 6.87         | PASS     |
| 128x128     | PyTorch (Parallel Forest) | 0.03      | 8.26         | PASS     |
| 128x128     | Triton (Parallel Forest)  | 0.01      | 1.44         | PASS     |
| 128x128     | CUDA C++ (Numba-free)     | 0.01      | 1.32         | PASS     |
| 256x256     | FlashCubical (C++)        | 0.00      | 5.82         | Baseline |
| 256x256     | JAX (Parallel Forest)     | 0.24      | 30.51        | PASS     |
| 256x256     | PyTorch (Parallel Forest) | 0.04      | 32.50        | PASS     |
| 256x256     | Triton (Parallel Forest)  | 0.01      | 6.07         | PASS     |
| 256x256     | CUDA C++ (Numba-free)     | 0.01      | 6.10         | PASS     |
| 512x512     | FlashCubical (C++)        | 0.00      | 24.59        | Baseline |
| 512x512     | JAX (Parallel Forest)     | 0.26      | 148.62       | PASS     |
| 512x512     | PyTorch (Parallel Forest) | 0.05      | 147.66       | PASS     |
| 512x512     | Triton (Parallel Forest)  | 0.02      | 25.66        | PASS     |
| 512x512     | CUDA C++ (Numba-free)     | 0.02      | 25.64        | PASS     |
| 1024x1024   | FlashCubical (C++)        | 0.01      | 104.96       | Baseline |
| 1024x1024   | JAX (Parallel Forest)     | 0.30      | 723.24       | PASS     |
| 1024x1024   | PyTorch (Parallel Forest) | 0.06      | 706.44       | PASS     |
| 1024x1024   | Triton (Parallel Forest)  | 0.05      | 108.08       | PASS     |
| 1024x1024   | CUDA C++ (Numba-free)     | 0.03      | 108.43       | PASS     |

### Gaussian Noise

| Grid Size   | Backend                   | Prep (ms) | Compute (ms) | Accuracy |
|-------------|---------------------------|-----------|--------------|----------|
| 32x32       | FlashCubical (C++)        | 0.01      | 0.05         | Baseline |
| 32x32       | JAX (Parallel Forest)     | 0.37      | 0.39         | PASS     |
| 32x32       | PyTorch (Parallel Forest) | 0.04      | 0.55         | PASS     |
| 32x32       | Triton (Parallel Forest)  | 0.01      | 0.05         | PASS     |
| 32x32       | CUDA C++ (Numba-free)     | 0.01      | 0.05         | PASS     |
| 64x64       | FlashCubical (C++)        | 0.00      | 0.24         | Baseline |
| 64x64       | JAX (Parallel Forest)     | 0.12      | 1.54         | PASS     |
| 64x64       | PyTorch (Parallel Forest) | 0.01      | 2.37         | PASS     |
| 64x64       | Triton (Parallel Forest)  | 0.00      | 0.25         | PASS     |
| 64x64       | CUDA C++ (Numba-free)     | 0.00      | 0.20         | PASS     |
| 128x128     | FlashCubical (C++)        | 0.00      | 1.33         | Baseline |
| 128x128     | JAX (Parallel Forest)     | 0.12      | 6.98         | PASS     |
| 128x128     | PyTorch (Parallel Forest) | 0.03      | 8.50         | PASS     |
| 128x128     | Triton (Parallel Forest)  | 0.01      | 1.46         | PASS     |
| 128x128     | CUDA C++ (Numba-free)     | 0.01      | 1.30         | PASS     |
| 256x256     | FlashCubical (C++)        | 0.00      | 5.83         | Baseline |
| 256x256     | JAX (Parallel Forest)     | 0.24      | 30.67        | PASS     |
| 256x256     | PyTorch (Parallel Forest) | 0.04      | 33.27        | PASS     |
| 256x256     | Triton (Parallel Forest)  | 0.01      | 6.13         | PASS     |
| 256x256     | CUDA C++ (Numba-free)     | 0.01      | 6.06         | PASS     |
| 512x512     | FlashCubical (C++)        | 0.00      | 24.65        | Baseline |
| 512x512     | JAX (Parallel Forest)     | 0.26      | 145.14       | PASS     |
| 512x512     | PyTorch (Parallel Forest) | 0.05      | 144.77       | PASS     |
| 512x512     | Triton (Parallel Forest)  | 0.02      | 25.89        | PASS     |
| 512x512     | CUDA C++ (Numba-free)     | 0.02      | 25.79        | PASS     |
| 1024x1024   | FlashCubical (C++)        | 0.01      | 103.36       | Baseline |
| 1024x1024   | JAX (Parallel Forest)     | 0.28      | 718.85       | PASS     |
| 1024x1024   | PyTorch (Parallel Forest) | 0.07      | 709.47       | PASS     |
| 1024x1024   | Triton (Parallel Forest)  | 0.03      | 106.98       | PASS     |
| 1024x1024   | CUDA C++ (Numba-free)     | 0.03      | 106.50       | PASS     |
