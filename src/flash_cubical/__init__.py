"""flash_cubical – fast cubical persistent homology for 2D and 3D scalar grids."""

import warnings
from importlib import resources

import numpy as np

from ._core import compute as _compute_native
from ._core import initialize_lookup_tables as _initialize_lookup_tables
from .persistence import Persistence

__all__ = ["compute", "Persistence"]


def _load_packaged_lookup_tables() -> None:
    data = resources.files(__package__).joinpath("data")
    path2d = data.joinpath("zero_lookup_2d.bin")
    path3d = data.joinpath("zero_lookup_3d.bin")
    with resources.as_file(path2d) as p2d, resources.as_file(path3d) as p3d:
        _initialize_lookup_tables(str(p2d), str(p3d))


_load_packaged_lookup_tables()


def compute(x, *, min_persistence: float = 0.0, h1: bool = True, backend: str = "cpu") -> Persistence:
    """Compute cubical persistent homology of a 2D or 3D scalar array.

    Parameters
    ----------
    x : array-like
        Input scalar field.  Converted to a C-contiguous float64 ndarray.
        Must be 2D (rows × cols) or 3D (depth × rows × cols).
    min_persistence : float, optional
        Minimum persistence threshold.  Currently not implemented by the
        backend; any value other than 0.0 emits a RuntimeWarning and is
        treated as 0.0.
    h1 : bool, optional
        Whether to compute H1 (loops / tunnels).
        - For 3D inputs: defaults to True; pass False to skip H1.
        - For 2D inputs: must be True (raising ValueError otherwise) when using the 'cpu' backend.
    backend : str, optional
        The computational backend to use. Options are:
        - "cpu" (default): The highly-optimized C++ native implementation. Supports 2D and 3D.
        - "pytorch": PyTorch GPU-accelerated Parallel Forest (2D H0 only).
        - "jax": JAX GPU-accelerated Parallel Forest (2D H0 only).
        - "triton": Triton GPU-accelerated Parallel Forest (2D H0 only).
        - "cuda": CUDA C++ GPU-accelerated Parallel Forest (2D H0 only).

    Returns
    -------
    Persistence
        Object with a ``.values`` attribute (float64 array of shape (n, 3),
        columns: birth, death, homology dimension) and a ``.plot()`` method.

    Raises
    ------
    ValueError
        If ``x`` is not 2D or 3D, if ``min_persistence < 0``, or if
        invalid parameter combinations are requested for the chosen backend.
    RuntimeWarning
        If ``min_persistence != 0.0`` (backend does not yet support this).
    """
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float64))

    if x.ndim < 2 or x.ndim > 3:
        raise ValueError(
            f"Input must be 2D or 3D, got {x.ndim}D array."
        )

    if min_persistence < 0.0:
        raise ValueError(
            f"min_persistence must be non-negative, got {min_persistence}."
        )

    if min_persistence != 0.0:
        warnings.warn(
            "min_persistence != 0.0 is not yet supported by the backend; "
            "running as if min_persistence=0.0.",
            RuntimeWarning,
            stacklevel=2,
        )

    backend = backend.lower()

    if backend == "cpu":
        if x.ndim == 2 and not h1:
            raise ValueError(
                "h1=False is only valid for 3D inputs.  The 2D CPU backend always "
                "computes H0 and H1 together."
            )
        arr = _compute_native(x, h1)
        return Persistence(arr)

    if x.ndim != 2 and backend in ["unknown_placeholder_to_force_skip"]:
        raise ValueError(f"The '{backend}' backend currently only supports 2D inputs for H0 computation. 3D coming soon.")

    # Lazy import backends to avoid making deep learning frameworks hard dependencies
    b_inst = None
    if backend == "pytorch":
        from .backends.backend_pytorch import PyTorchBackend
        b_inst = PyTorchBackend()
    elif backend == "jax":
        from .backends.backend_jax import JaxBackend
        b_inst = JaxBackend()
    elif backend == "triton":
        from .backends.backend_triton import TritonBackend
        b_inst = TritonBackend()
    elif backend == "cuda":
        from .backends.backend_cuda import CudaBackend
        b_inst = CudaBackend()
    else:
        raise ValueError(f"Unknown backend '{backend}'. Choose from: 'cpu', 'pytorch', 'jax', 'triton', 'cuda'.")

    prepared = b_inst.prepare(x) if hasattr(b_inst, "prepare") else x
    res = b_inst.compute_ph(prepared)
    
    # Convert result back to CPU numpy array if needed
    if hasattr(res, "cpu"):
        res = res.cpu().numpy()
    elif hasattr(res, "device_buffer"):
        res = np.array(res)
    elif not isinstance(res, np.ndarray):
        res = np.array(res)

    return Persistence(res)
