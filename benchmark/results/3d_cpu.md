# 3D Benchmarks - CPU

### Uniform Noise

| Grid Size   | Backend                   | Prep (ms) | Compute (ms) | Accuracy |
|-------------|---------------------------|-----------|--------------|----------|
| 16x16x16    | FlashCubical (C++)        | 0.00      | 1.67         | Baseline |
| 16x16x16    | JAX (Parallel Forest)     | 0.19      | 2.15         | PASS     |
| 16x16x16    | PyTorch (Parallel Forest) | 0.04      | 3.32         | PASS     |
| 16x16x16    | Triton (Parallel Forest)  | 0.01      | 0.71         | PASS     |
| 16x16x16    | CUDA C++ (Numba-free)     | 0.01      | 0.62         | PASS     |
| 32x32x32    | FlashCubical (C++)        | 0.00      | 18.25        | Baseline |
| 32x32x32    | JAX (Parallel Forest)     | 0.25      | 22.33        | PASS     |
| 32x32x32    | PyTorch (Parallel Forest) | 0.04      | 24.05        | PASS     |
| 32x32x32    | Triton (Parallel Forest)  | 0.01      | 8.99         | PASS     |
| 32x32x32    | CUDA C++ (Numba-free)     | 0.01      | 8.97         | PASS     |
| 64x64x64    | FlashCubical (C++)        | 0.00      | 174.97       | Baseline |
| 64x64x64    | JAX (Parallel Forest)     | 0.28      | 218.26       | PASS     |
| 64x64x64    | PyTorch (Parallel Forest) | 0.08      | 223.54       | PASS     |
| 64x64x64    | Triton (Parallel Forest)  | 0.02      | 76.67        | PASS     |
| 64x64x64    | CUDA C++ (Numba-free)     | 0.03      | 76.39        | PASS     |
| 128x128x128 | FlashCubical (C++)        | 0.02      | 2596.63      | Baseline |
| 128x128x128 | JAX (Parallel Forest)     | 0.38      | 2323.61      | PASS     |
| 128x128x128 | PyTorch (Parallel Forest) | 0.13      | 2306.31      | PASS     |
| 128x128x128 | Triton (Parallel Forest)  | 0.04      | 668.20       | PASS     |
| 128x128x128 | CUDA C++ (Numba-free)     | 0.04      | 667.20       | PASS     |

### Gaussian Noise

| Grid Size   | Backend                   | Prep (ms) | Compute (ms) | Accuracy |
|-------------|---------------------------|-----------|--------------|----------|
| 16x16x16    | FlashCubical (C++)        | 0.01      | 1.82         | Baseline |
| 16x16x16    | JAX (Parallel Forest)     | 0.45      | 2.51         | PASS     |
| 16x16x16    | PyTorch (Parallel Forest) | 0.04      | 3.46         | PASS     |
| 16x16x16    | Triton (Parallel Forest)  | 0.01      | 0.84         | PASS     |
| 16x16x16    | CUDA C++ (Numba-free)     | 0.01      | 0.63         | PASS     |
| 32x32x32    | FlashCubical (C++)        | 0.00      | 18.54        | Baseline |
| 32x32x32    | JAX (Parallel Forest)     | 0.26      | 22.57        | PASS     |
| 32x32x32    | PyTorch (Parallel Forest) | 0.04      | 24.72        | PASS     |
| 32x32x32    | Triton (Parallel Forest)  | 0.01      | 8.89         | PASS     |
| 32x32x32    | CUDA C++ (Numba-free)     | 0.01      | 8.83         | PASS     |
| 64x64x64    | FlashCubical (C++)        | 0.00      | 180.83       | Baseline |
| 64x64x64    | JAX (Parallel Forest)     | 0.27      | 216.36       | PASS     |
| 64x64x64    | PyTorch (Parallel Forest) | 0.05      | 217.50       | PASS     |
| 64x64x64    | Triton (Parallel Forest)  | 0.02      | 76.46        | PASS     |
| 64x64x64    | CUDA C++ (Numba-free)     | 0.03      | 88.70        | PASS     |
| 128x128x128 | FlashCubical (C++)        | 0.01      | 2644.76      | Baseline |
| 128x128x128 | JAX (Parallel Forest)     | 1.80      | 2557.18      | PASS     |
| 128x128x128 | PyTorch (Parallel Forest) | 0.42      | 2583.75      | PASS     |
| 128x128x128 | Triton (Parallel Forest)  | 0.09      | 679.56       | PASS     |
| 128x128x128 | CUDA C++ (Numba-free)     | 0.06      | 672.79       | PASS     |
