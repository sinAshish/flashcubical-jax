import torch
import numpy as np
from typing import Any
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
        # We only compile the CUDA C++ code if a GPU and nvcc compiler are active
        if HAS_CUDA_BUILDER and torch.cuda.is_available():
            try:
                self.module = load(
                    name="flash_cubical_cuda",
                    sources=["cpp/cuda/flash_cubical_cuda.cu"],
                    verbose=True
                )
            except Exception as e:
                # Compile errors will be caught and logged safely
                print(f"Warning: Failed to compile CUDA extension: {e}")

    def prepare(self, image: Any) -> Any:
        arr = torch.tensor(image, dtype=torch.float64)
        if torch.cuda.is_available():
            arr = arr.cuda()
        return arr

    def sync(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def compute_ph(self, arr: Any) -> np.ndarray:
        # Fallback to JAX or C++ CPU if CUDA compiling fails or no GPU is present
        if self.module is None or not torch.cuda.is_available():
            import flash_cubical as fc
            return fc.compute(arr.cpu().numpy()).values
            
        # GPU execution: Call the compiled CUDA C++ shared library
        births = arr.flatten()
        v = births.numel()
        
        if arr.ndim == 2:
            deaths = self.module.compute_h0_cuda(births, arr.shape[0], arr.shape[1])
        else:
            deaths = self.module.compute_h0_cuda_3d(births, arr.shape[0], arr.shape[1], arr.shape[2])
            
        pairs = torch.stack([births, deaths, torch.zeros_like(births)], dim=1)
        return pairs.cpu().numpy()

    def name(self) -> str:
        return "CUDA C++ (Numba-free)"
