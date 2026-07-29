import pytest
import numpy as np
from benchmark.backend_baseline import FlashCubicalBackend
from benchmark.backend_jax import JaxBackend
from benchmark.benchmark_utils import generate_2d_grid, verify_persistence

def test_jax_backend_basic():
    grid = generate_2d_grid(32, "uniform")
    baseline = FlashCubicalBackend().compute_ph(grid)
    jax_res = JaxBackend().compute_ph(grid)
    assert verify_persistence(baseline, jax_res) is True