# 2D Benchmarks - GPU (A100)

### Uniform Noise

| Grid Size   | Backend                   | Prep (ms) | Compute (ms) | Accuracy |
|-------------|---------------------------|-----------|--------------|----------|
| 32x32       | FlashCubical (C++)        | 0.02      | 0.20         | Baseline |
| 32x32       | JAX (Parallel Forest)     | 876.84    | 2.63         | PASS     |
| 32x32       | PyTorch (Parallel Forest) | 2.18      | 2.68         | PASS     |
| 32x32       | Triton (Parallel Forest)  | 0.11      | 2.92         | PASS     |
| 32x32       | CUDA C++ (Numba-free)     | 0.10      | 0.16         | PASS     |
| 64x64       | FlashCubical (C++)        | 0.01      | 0.53         | Baseline |
| 64x64       | JAX (Parallel Forest)     | 0.57      | 5.42         | PASS     |
| 64x64       | PyTorch (Parallel Forest) | 0.28      | 5.13         | PASS     |
| 64x64       | Triton (Parallel Forest)  | 0.11      | 6.64         | PASS     |
| 64x64       | CUDA C++ (Numba-free)     | 0.13      | 0.60         | PASS     |
| 128x128     | FlashCubical (C++)        | 0.02      | 2.03         | Baseline |
| 128x128     | JAX (Parallel Forest)     | 0.58      | 24.61        | PASS     |
| 128x128     | PyTorch (Parallel Forest) | 0.43      | 21.52        | PASS     |
| 128x128     | Triton (Parallel Forest)  | 0.18      | 23.90        | PASS     |
| 128x128     | CUDA C++ (Numba-free)     | 0.14      | 2.14         | PASS     |
| 256x256     | FlashCubical (C++)        | 0.01      | 8.39         | Baseline |
| 256x256     | JAX (Parallel Forest)     | 0.65      | 98.88        | PASS     |
| 256x256     | PyTorch (Parallel Forest) | 34.27     | 59.48        | PASS     |
| 256x256     | Triton (Parallel Forest)  | 12.27     | 94.15        | PASS     |
| 256x256     | CUDA C++ (Numba-free)     | 14.06     | 8.49         | PASS     |
| 512x512     | FlashCubical (C++)        | 0.01      | 41.41        | Baseline |
| 512x512     | JAX (Parallel Forest)     | 0.96      | 402.13       | PASS     |
| 512x512     | PyTorch (Parallel Forest) | 15.00     | 338.19       | PASS     |
| 512x512     | Triton (Parallel Forest)  | 16.36     | 518.25       | PASS     |
| 512x512     | CUDA C++ (Numba-free)     | 20.87     | 39.05        | PASS     |
| 1024x1024   | FlashCubical (C++)        | 0.05      | 176.94       | Baseline |
| 1024x1024   | JAX (Parallel Forest)     | 7.59      | 1585.48      | PASS     |
| 1024x1024   | PyTorch (Parallel Forest) | 45.09     | 1200.05      | PASS     |
| 1024x1024   | Triton (Parallel Forest)  | 19.19     | 2222.72      | PASS     |
| 1024x1024   | CUDA C++ (Numba-free)     | 14.96     | 142.36       | PASS     |
### Gaussian Noise

| Grid Size   | Backend                   | Prep (ms) | Compute (ms) | Accuracy |
|-------------|---------------------------|-----------|--------------|----------|
| 32x32       | FlashCubical (C++)        | 0.06      | 0.07         | Baseline |
| 32x32       | JAX (Parallel Forest)     | 3.58      | 2.92         | PASS     |
| 32x32       | PyTorch (Parallel Forest) | 0.26      | 2.21         | PASS     |
| 32x32       | Triton (Parallel Forest)  | 0.08      | 2.51         | PASS     |
| 32x32       | CUDA C++ (Numba-free)     | 0.09      | 0.13         | PASS     |
| 64x64       | FlashCubical (C++)        | 0.01      | 0.46         | Baseline |
| 64x64       | JAX (Parallel Forest)     | 0.34      | 5.23         | PASS     |
| 64x64       | PyTorch (Parallel Forest) | 0.13      | 4.50         | PASS     |
| 64x64       | Triton (Parallel Forest)  | 0.08      | 6.33         | PASS     |
| 64x64       | CUDA C++ (Numba-free)     | 0.10      | 0.51         | PASS     |
| 128x128     | FlashCubical (C++)        | 0.01      | 2.02         | Baseline |
| 128x128     | JAX (Parallel Forest)     | 0.41      | 16.28        | PASS     |
| 128x128     | PyTorch (Parallel Forest) | 0.34      | 14.21        | PASS     |
| 128x128     | Triton (Parallel Forest)  | 0.13      | 22.52        | PASS     |
| 128x128     | CUDA C++ (Numba-free)     | 0.13      | 2.13         | PASS     |
| 256x256     | FlashCubical (C++)        | 0.01      | 8.34         | Baseline |
| 256x256     | JAX (Parallel Forest)     | 0.48      | 70.64        | PASS     |
| 256x256     | PyTorch (Parallel Forest) | 10.36     | 60.71        | PASS     |
| 256x256     | Triton (Parallel Forest)  | 10.28     | 108.01       | PASS     |
| 256x256     | CUDA C++ (Numba-free)     | 10.02     | 8.86         | PASS     |
| 512x512     | FlashCubical (C++)        | 0.02      | 38.64        | Baseline |
| 512x512     | JAX (Parallel Forest)     | 0.57      | 310.69       | PASS     |
| 512x512     | PyTorch (Parallel Forest) | 17.45     | 327.76       | PASS     |
| 512x512     | Triton (Parallel Forest)  | 18.28     | 482.50       | PASS     |
| 512x512     | CUDA C++ (Numba-free)     | 19.08     | 37.25        | PASS     |
| 1024x1024   | FlashCubical (C++)        | 0.04      | 165.05       | Baseline |
| 1024x1024   | JAX (Parallel Forest)     | 1.27      | 1489.86      | PASS     |
| 1024x1024   | PyTorch (Parallel Forest) | 13.05     | 1492.73      | PASS     |
| 1024x1024   | Triton (Parallel Forest)  | 13.72     | 2442.05      | PASS     |
| 1024x1024   | CUDA C++ (Numba-free)     | 13.39     | 153.15       | PASS     |
