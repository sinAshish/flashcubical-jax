# 3D Benchmarks - GPU (A100)

### Uniform Noise

| Grid Size   | Backend                   | Prep (ms) | Compute (ms) | Accuracy |
|-------------|---------------------------|-----------|--------------|----------|
| 16x16x16    | FlashCubical (C++)        | 0.01      | 2.78         | Baseline |
| 16x16x16    | JAX (Parallel Forest)     | 1.55      | 7.10         | PASS     |
| 16x16x16    | PyTorch (Parallel Forest) | 0.45      | 394.54       | PASS     |
| 32x32x32    | FlashCubical (C++)        | 0.02      | 24.99        | Baseline |
| 32x32x32    | JAX (Parallel Forest)     | 1.19      | 38.93        | PASS     |
| 32x32x32    | PyTorch (Parallel Forest) | 0.31      | 1138.06      | PASS     |
| 64x64x64    | FlashCubical (C++)        | 0.03      | 324.14       | Baseline |
| 64x64x64    | JAX (Parallel Forest)     | 1.45      | 400.97       | PASS     |
| 64x64x64    | PyTorch (Parallel Forest) | 19.17     | 2535.15      | PASS     |
| 128x128x128 | FlashCubical (C++)        | 0.06      | 3096.03      | Baseline |
| 128x128x128 | JAX (Parallel Forest)     | 1.77      | 3131.49      | PASS     |
| 128x128x128 | PyTorch (Parallel Forest) | 25.74     | 6088.28      | PASS     |
### Gaussian Noise

| Grid Size   | Backend                   | Prep (ms) | Compute (ms) | Accuracy |
|-------------|---------------------------|-----------|--------------|----------|
| 16x16x16    | FlashCubical (C++)        | 0.04      | 2.77         | Baseline |
| 16x16x16    | JAX (Parallel Forest)     | 1.74      | 5.62         | PASS     |
| 16x16x16    | PyTorch (Parallel Forest) | 0.19      | 404.83       | PASS     |
| 32x32x32    | FlashCubical (C++)        | 0.01      | 26.11        | Baseline |
| 32x32x32    | JAX (Parallel Forest)     | 0.84      | 40.02        | PASS     |
| 32x32x32    | PyTorch (Parallel Forest) | 0.40      | 3356.68      | PASS     |
| 64x64x64    | FlashCubical (C++)        | 0.03      | 461.82       | Baseline |
| 64x64x64    | JAX (Parallel Forest)     | 1.77      | 373.28       | PASS     |
| 64x64x64    | PyTorch (Parallel Forest) | 12.91     | 2511.98      | PASS     |
| 128x128x128 | FlashCubical (C++)        | 0.10      | 3034.02      | Baseline |
| 128x128x128 | JAX (Parallel Forest)     | 2.46      | 2927.63      | PASS     |
| 128x128x128 | PyTorch (Parallel Forest) | 19.76     | 6103.05      | PASS     |
