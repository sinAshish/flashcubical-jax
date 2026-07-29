import pytest
import numpy as np
from benchmark.benchmark_utils import generate_2d_grid, verify_persistence

def test_generate_2d_grid():
    grid = generate_2d_grid(128, "uniform")
    assert grid.shape == (128, 128)
    assert grid.dtype == np.float64

def test_verify_persistence():
    baseline = np.array([[0.1, 0.5, 0], [0.2, 0.3, 1], [0.4, 0.4, 0]]) # third is zero-persistence
    test_res = np.array([[0.2, 0.3, 1], [0.1, 0.5, 0], [0.6, 0.6, 0]])
    assert verify_persistence(baseline, test_res) is True
