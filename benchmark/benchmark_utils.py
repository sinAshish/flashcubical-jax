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
    # Filter zero persistence
    base_filtered = baseline[baseline[:, 0] != baseline[:, 1]]
    test_filtered = test_res[test_res[:, 0] != test_res[:, 1]]
    
    # Sort both arrays lexicographically by dimension, birth, death
    base_sorted = base_filtered[np.lexsort((base_filtered[:, 1], base_filtered[:, 0], base_filtered[:, 2]))]
    test_sorted = test_filtered[np.lexsort((test_filtered[:, 1], test_filtered[:, 0], test_filtered[:, 2]))]
    
    return np.allclose(base_sorted, test_sorted, atol=1e-6)