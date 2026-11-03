# 3D Benchmarks - CPU

### Uniform Noise

| Grid Size   | Backend                   | Prep (ms) | Compute (ms) | Accuracy |
|-------------|---------------------------|-----------|--------------|----------|
| 16x16x16    | FlashCubical (C++)        | 0.00      | 2.07         | Baseline |
| 16x16x16    | JAX (Parallel Forest)     | 0.28      | 0.40         | PASS     |
| 16x16x16    | PyTorch (Parallel Forest) | 0.04      | 0.72         | PASS     |
| 16x16x16    | Triton (Parallel Forest)  | 0.01      | 1.23         | PASS     |
| 16x16x16    | CUDA C++ (Numba-free)     | 0.01      | 0.72         | PASS     |
| 32x32x32    | FlashCubical (C++)        | 0.00      | 18.59        | Baseline |
| 32x32x32    | JAX (Parallel Forest)     | 0.29      | 1.66         | PASS     |
| 32x32x32    | PyTorch (Parallel Forest) | 0.04      | 2.75         | PASS     |
| 32x32x32    | Triton (Parallel Forest)  | 0.01      | 9.06         | PASS     |
| 32x32x32    | CUDA C++ (Numba-free)     | 0.02      | 8.97         | PASS     |
| 64x64x64    | FlashCubical (C++)        | 0.00      | 181.50       | Baseline |
| 64x64x64    | JAX (Parallel Forest)     | 0.28      | 11.87        | PASS     |
| 64x64x64    | PyTorch (Parallel Forest) | 0.04      | 15.58        | PASS     |
| 64x64x64    | Triton (Parallel Forest)  | 0.03      | 77.00        | PASS     |
| 64x64x64    | CUDA C++ (Numba-free)     | 0.04      | 77.83        | PASS     |
| 128x128x128 | FlashCubical (C++)        | 0.01      | 2410.94      | Baseline |
| 128x128x128 | JAX (Parallel Forest)     | 0.32      | 111.25       | PASS     |
| 128x128x128 | PyTorch (Parallel Forest) | 0.06      | 137.45       | PASS     |
| 128x128x128 | Triton (Parallel Forest)  | 0.04      | 736.93       | PASS     |
| 128x128x128 | CUDA C++ (Numba-free)     | 0.04      | 661.71       | PASS     |

### Gaussian Noise

| Grid Size   | Backend                   | Prep (ms) | Compute (ms) | Accuracy |
|-------------|---------------------------|-----------|--------------|----------|
| 16x16x16    | FlashCubical (C++)        | 0.01      | 1.74         | Baseline |
| 16x16x16    | JAX (Parallel Forest)     | 0.28      | 0.38         | PASS     |
| 16x16x16    | PyTorch (Parallel Forest) | 0.04      | 0.75         | PASS     |
| 16x16x16    | Triton (Parallel Forest)  | 0.01      | 0.82         | PASS     |
| 16x16x16    | CUDA C++ (Numba-free)     | 0.01      | 0.64         | PASS     |
| 32x32x32    | FlashCubical (C++)        | 0.00      | 18.25        | Baseline |
| 32x32x32    | JAX (Parallel Forest)     | 0.26      | 1.61         | PASS     |
| 32x32x32    | PyTorch (Parallel Forest) | 0.04      | 2.44         | PASS     |
| 32x32x32    | Triton (Parallel Forest)  | 0.01      | 8.90         | PASS     |
| 32x32x32    | CUDA C++ (Numba-free)     | 0.01      | 8.86         | PASS     |
| 64x64x64    | FlashCubical (C++)        | 0.00      | 177.75       | Baseline |
| 64x64x64    | JAX (Parallel Forest)     | 0.29      | 11.83        | PASS     |
| 64x64x64    | PyTorch (Parallel Forest) | 0.05      | 15.81        | PASS     |
| 64x64x64    | Triton (Parallel Forest)  | 0.02      | 76.06        | PASS     |
| 64x64x64    | CUDA C++ (Numba-free)     | 0.02      | 76.51        | PASS     |
| 128x128x128 | FlashCubical (C++)        | 0.01      | 2315.54      | Baseline |
| 128x128x128 | JAX (Parallel Forest)     | 0.30      | 135.08       | PASS     |
| 128x128x128 | PyTorch (Parallel Forest) | 0.09      | 172.89       | PASS     |
| 128x128x128 | Triton (Parallel Forest)  | 0.04      | 733.70       | PASS     |
| 128x128x128 | CUDA C++ (Numba-free)     | 0.05      | 854.16       | PASS     |
