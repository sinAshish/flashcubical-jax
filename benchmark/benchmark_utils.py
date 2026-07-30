import numpy as np
from typing import Literal

def generate_2d_grid(size: int, noise_type: Literal["uniform", "gaussian"]) -> np.ndarray:
    if noise_type == "uniform":
        return np.random.rand(size, size).astype(np.float64)
    elif noise_type == "gaussian":
        return np.random.randn(size, size).astype(np.float64)
    raise ValueError(f"Unknown noise_type {noise_type}")

def generate_3d_grid(size: int, noise_type: Literal["uniform", "gaussian"]) -> np.ndarray:
    if noise_type == "uniform":
        return np.random.rand(size, size, size).astype(np.float64)
    return np.random.randn(size, size, size).astype(np.float64)

def verify_persistence(baseline: np.ndarray, test_res: np.ndarray) -> bool:
    # We only benchmark H0 for the GPU implementations
    base_h0 = baseline[baseline[:, 2] == 0]
    test_h0 = test_res[test_res[:, 2] == 0]
    
    # Filter zero persistence
    base_filtered = base_h0[base_h0[:, 0] != base_h0[:, 1]]
    test_filtered = test_h0[test_h0[:, 0] != test_h0[:, 1]]
    
    if len(base_filtered) != len(test_filtered):
        return False
        
    # Sort both arrays lexicographically by birth, death
    base_sorted = base_filtered[np.lexsort((base_filtered[:, 1], base_filtered[:, 0]))]
    test_sorted = test_filtered[np.lexsort((test_filtered[:, 1], test_filtered[:, 0]))]
    
    return np.allclose(base_sorted[:, :2], test_sorted[:, :2], atol=1e-5)