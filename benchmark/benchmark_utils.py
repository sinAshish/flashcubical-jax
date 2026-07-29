import numpy as np
from typing import Literal

def generate_2d_grid(size: int, noise_type: Literal["uniform", "gaussian"]) -> np.ndarray:
    if noise_type == "uniform":
        return np.random.rand(size, size).astype(np.float64)
    elif noise_type == "gaussian":
        return np.random.randn(size, size).astype(np.float64)
    raise ValueError(f"Unknown noise_type {noise_type}")
