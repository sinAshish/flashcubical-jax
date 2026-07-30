import torch
import numpy as np
from typing import Any
from benchmark.backend_protocol import PHBackend

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

    def compute_ph(self, image: Any) -> np.ndarray:
        arr = torch.tensor(image, dtype=torch.float64)
        
        # Fallback to JAX or C++ CPU if CUDA compiling fails or no GPU is present
        if self.module is None or not torch.cuda.is_available():
            import flash_cubical as fc
            return fc.compute(arr.numpy()).values
            
        # GPU execution: Call the compiled CUDA C++ shared library
        arr = arr.cuda()
        births = arr.flatten()
        
        # Call bindings compiled on-the-fly
        deaths = self.module.compute_h0_cuda(births, arr.shape[0], arr.shape[1])
        
        pairs = torch.stack([births, deaths, torch.zeros_like(births)], dim=1)
        return pairs.cpu().numpy()

    def name(self) -> str:
        return "CUDA C++ (Numba-free)"
