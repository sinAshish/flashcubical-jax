
import pytest
import numpy as np
import flash_cubical as fc
import torch

def test_3d_backends():
    # Create a small 3D volume
    volume = np.random.rand(8, 8, 8).astype(np.float64)
    
    # Compute baseline H0 using CPU C++
    ph_cpu = fc.compute(volume, backend="cpu", h1=False).values
    baseline_h0 = ph_cpu[ph_cpu[:, 2] == 0]
    
    # Sort baseline
    baseline_h0 = baseline_h0[baseline_h0[:, 0] != baseline_h0[:, 1]]
    baseline_h0 = baseline_h0[np.lexsort((baseline_h0[:, 1], baseline_h0[:, 0]))]
    
    # Test PyTorch
    ph_torch = fc.compute(volume, backend="pytorch").values
    test_h0_pt = ph_torch[ph_torch[:, 2] == 0]
    test_h0_pt = test_h0_pt[test_h0_pt[:, 0] != test_h0_pt[:, 1]]
    test_h0_pt = test_h0_pt[np.lexsort((test_h0_pt[:, 1], test_h0_pt[:, 0]))]
    
    assert np.allclose(baseline_h0[:, :2], test_h0_pt[:, :2], atol=1e-5), "PyTorch 3D failed"
    
    # Test JAX
    ph_jax = fc.compute(volume, backend="jax").values
    test_h0_jax = ph_jax[ph_jax[:, 2] == 0]
    test_h0_jax = test_h0_jax[test_h0_jax[:, 0] != test_h0_jax[:, 1]]
    test_h0_jax = test_h0_jax[np.lexsort((test_h0_jax[:, 1], test_h0_jax[:, 0]))]
    
    assert np.allclose(baseline_h0[:, :2], test_h0_jax[:, :2], atol=1e-5), "JAX 3D failed"
