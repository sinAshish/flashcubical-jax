from pathlib import Path
from typing import Any
import numpy as np
import torch
from .backend_protocol import PHBackend

# PyTorch provides on-the-fly compilation of CUDA C++ extensions via 'load'
try:
    from torch.utils.cpp_extension import load
    HAS_CUDA_BUILDER = True
except ImportError:
    HAS_CUDA_BUILDER = False


class CudaBackend(PHBackend):
    def __init__(self):
        self.module = None
        # Only compile the CUDA C++ code if a GPU and nvcc compiler are active
        if HAS_CUDA_BUILDER and torch.cuda.is_available():
            try:
                cu_source = Path(__file__).resolve().parents[3] / "cpp" / "cuda" / "flash_cubical_cuda.cu"
                if not cu_source.exists():
                    cu_source = Path("cpp/cuda/flash_cubical_cuda.cu").resolve()
                self.module = load(
                    name="flash_cubical_cuda",
                    sources=[str(cu_source)],
                    verbose=True,
                )
            except Exception as e:
                print(f"Warning: Failed to compile CUDA extension: {e}")

    def prepare(self, image: Any) -> Any:
        arr = torch.as_tensor(image, dtype=torch.float64)
        if torch.cuda.is_available():
            arr = arr.cuda()
        return arr

    def sync(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def compute_ph(self, arr: Any) -> np.ndarray:
        if not isinstance(arr, torch.Tensor):
            arr = self.prepare(arr)

        # Fallback to C++ CPU (H0 only) if CUDA compiling fails or no GPU is present
        if self.module is None or not torch.cuda.is_available():
            import flash_cubical as fc
            np_arr = arr.cpu().numpy() if hasattr(arr, "cpu") else np.asarray(arr)
            res = fc.compute(np_arr, h1=(np_arr.ndim == 2)).values
            return res[res[:, 2] == 0]

        if not arr.is_cuda:
            arr = arr.cuda()

        # GPU execution: Call the compiled CUDA C++ shared library
        births = arr.flatten().contiguous()

        if arr.ndim == 2:
            deaths = self.module.compute_h0_cuda(births, arr.shape[0], arr.shape[1])
        else:
            deaths = self.module.compute_h0_cuda_3d(births, arr.shape[0], arr.shape[1], arr.shape[2])

        pairs = torch.stack([births, deaths, torch.zeros_like(births)], dim=1)
        return pairs.cpu().numpy()

    def name(self) -> str:
        return "CUDA C++ (Numba-free)"
