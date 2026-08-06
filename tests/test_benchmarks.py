import pytest
import numpy as np
from flash_cubical.backends.benchmark_utils import generate_2d_grid, verify_persistence
from flash_cubical.backends.backend_baseline import FlashCubicalBackend
from flash_cubical.backends.backend_jax import JaxBackend
from flash_cubical.backends.backend_pytorch import PyTorchBackend
from flash_cubical.backends.backend_triton import TritonBackend
from flash_cubical.backends.backend_cuda import CudaBackend

# --- Utils Tests ---

@pytest.mark.parametrize("size", [16, 32, 64])
@pytest.mark.parametrize("noise_type", ["uniform", "gaussian"])
def test_generate_2d_grid(size, noise_type):
    grid = generate_2d_grid(size, noise_type)
    assert grid.shape == (size, size)
    assert grid.dtype == np.float64
    
    if noise_type == "uniform":
        assert np.all((grid >= 0.0) & (grid <= 1.0))

def test_generate_2d_grid_invalid():
    with pytest.raises(ValueError):
        generate_2d_grid(32, "unknown_noise")

def test_verify_persistence():
    # Only H0 matches are tested, H1/H2 will be ignored. 
    # Zero-persistence pairs should be filtered out.
    baseline = np.array([
        [0.1, 0.5, 0], 
        [0.2, 0.3, 0], 
        [0.4, 0.4, 0], # zero persistence
        [0.1, 0.8, 1]  # H1 ignored
    ])
    
    test_res = np.array([
        [0.2, 0.3, 0], 
        [0.1, 0.5, 0], 
        [0.6, 0.6, 0]  # zero persistence
    ])
    
    assert verify_persistence(baseline, test_res) is True

def test_verify_persistence_fail():
    baseline = np.array([[0.1, 0.5, 0]])
    test_res = np.array([[0.1, 0.6, 0]]) # death mismatch
    assert verify_persistence(baseline, test_res) is False


# --- Backend Protocol Tests ---

def test_baseline_backend():
    grid = generate_2d_grid(32, "uniform")
    backend = FlashCubicalBackend()
    res = backend.compute_ph(grid)
    assert res.shape[1] == 3
    assert res.dtype == np.float64
    assert "FlashCubical" in backend.name()

# --- Accuracy Tests ---

@pytest.fixture
def backends():
    bs = [JaxBackend(), PyTorchBackend()]
    
    # Triton and CUDA backends might fail to initialize on macOS/CPU
    try:
        from flash_cubical.backends.backend_triton import HAS_TRITON
        if HAS_TRITON:
            bs.append(TritonBackend())
        else:
            # We still append it to test the CPU fallback logic!
            bs.append(TritonBackend())
    except Exception:
        pass
        
    try:
        bs.append(CudaBackend())
    except Exception:
        pass
        
    return bs

@pytest.fixture
def baseline_backend():
    return FlashCubicalBackend()

@pytest.mark.parametrize("size", [16, 32])
@pytest.mark.parametrize("noise_type", ["uniform", "gaussian"])
def test_backends_accuracy_random(backends, baseline_backend, size, noise_type):
    """Test accuracy against baseline on standard random distributions."""
    grid = generate_2d_grid(size, noise_type)
    baseline_res = baseline_backend.compute_ph(grid)
    
    for backend in backends:
        prepared_grid = backend.prepare(grid) if hasattr(backend, "prepare") else grid
        res = backend.compute_ph(prepared_grid)
        if hasattr(res, "cpu"): res = res.cpu().numpy()
        elif hasattr(res, "device_buffer"): res = np.array(res)
        elif not isinstance(res, np.ndarray): res = np.array(res)
        assert verify_persistence(baseline_res, res), f"{backend.name()} failed on {noise_type} {size}x{size}"

def test_backends_accuracy_flat_image(backends, baseline_backend):
    """Test edge case: flat image (all values zero). 
    Everything is born at 0 and merges immediately, leaving a single essential class at (0, inf)."""
    grid = np.zeros((32, 32), dtype=np.float64)
    baseline_res = baseline_backend.compute_ph(grid)
    
    for backend in backends:
        prepared_grid = backend.prepare(grid) if hasattr(backend, "prepare") else grid
        res = backend.compute_ph(prepared_grid)
        if hasattr(res, "cpu"): res = res.cpu().numpy()
        elif hasattr(res, "device_buffer"): res = np.array(res)
        elif not isinstance(res, np.ndarray): res = np.array(res)
        assert verify_persistence(baseline_res, res), f"{backend.name()} failed on flat image"

def test_backends_accuracy_checkerboard(backends, baseline_backend):
    """Test edge case: highly degenerate checkerboard pattern."""
    grid = np.indices((32, 32)).sum(axis=0) % 2
    grid = grid.astype(np.float64)
    baseline_res = baseline_backend.compute_ph(grid)
    
    for backend in backends:
        prepared_grid = backend.prepare(grid) if hasattr(backend, "prepare") else grid
        res = backend.compute_ph(prepared_grid)
        if hasattr(res, "cpu"): res = res.cpu().numpy()
        elif hasattr(res, "device_buffer"): res = np.array(res)
        elif not isinstance(res, np.ndarray): res = np.array(res)
        assert verify_persistence(baseline_res, res), f"{backend.name()} failed on checkerboard image"

def test_backends_accuracy_single_peak(backends, baseline_backend):
    """Test edge case: Single peak in the center (mountain)."""
    grid = np.zeros((32, 32), dtype=np.float64)
    grid[16, 16] = 1.0
    baseline_res = baseline_backend.compute_ph(grid)
    
    for backend in backends:
        prepared_grid = backend.prepare(grid) if hasattr(backend, "prepare") else grid
        res = backend.compute_ph(prepared_grid)
        if hasattr(res, "cpu"): res = res.cpu().numpy()
        elif hasattr(res, "device_buffer"): res = np.array(res)
        elif not isinstance(res, np.ndarray): res = np.array(res)
        assert verify_persistence(baseline_res, res), f"{backend.name()} failed on single peak"
