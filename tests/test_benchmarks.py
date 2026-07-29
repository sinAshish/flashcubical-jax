import numpy as np
from benchmark.backend_baseline import FlashCubicalBackend
from benchmark.benchmark_utils import generate_2d_grid

def test_baseline_backend():
    grid = generate_2d_grid(64, "uniform")
    backend = FlashCubicalBackend()
    res = backend.compute_ph(grid)
    assert res.shape[1] == 3
    assert res.dtype == np.float64
