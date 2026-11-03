import numpy as np
from flash_cubical.backends.backend_baseline import FlashCubicalBackend
from flash_cubical.backends.backend_jax import JaxBackend
from flash_cubical.backends.benchmark_utils import generate_2d_grid, verify_persistence


def test_jax_backend_basic():
    grid = generate_2d_grid(32, "uniform")
    baseline = FlashCubicalBackend().compute_ph(grid)
    jax_res = JaxBackend().compute_ph(grid)
    assert verify_persistence(baseline, jax_res) is True
