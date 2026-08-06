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
                import traceback
                traceback.print_exc()

    def prepare(self, image: Any) -> Any:
        arr = torch.tensor(image, dtype=torch.float64)
        if torch.cuda.is_available():
            arr = arr.cuda()
        return arr

    def sync(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def compute_ph(self, arr: Any) -> np.ndarray:
        if self.module is None or not torch.cuda.is_available():
            import flash_cubical as fc
            return fc.compute(arr.cpu().numpy()).values
            
        births = arr.flatten()
        v = births.numel()
        
        parent = torch.arange(v, dtype=torch.int32, device=arr.device)
        deaths = torch.full((v,), float('inf'), dtype=torch.float64, device=arr.device)
        has_zp = torch.zeros(v, dtype=torch.bool, device=arr.device)
        
        if arr.ndim == 2:
            n, m = arr.shape
            self.module.compute_h0_cuda(births, parent, deaths, has_zp, n, m)
            r, c = torch.meshgrid(torch.arange(n, device=arr.device), torch.arange(m, device=arr.device), indexing='ij')
            u = torch.cat([(r[:, :-1] * m + c[:, :-1]).flatten(), (r[:-1, :] * m + c[:-1, :]).flatten()])
            v_ = torch.cat([(r[:, :-1] * m + c[:, 1:]).flatten(), (r[1:, :] * m + c[1:, :]).flatten()])
        else:
            d, n, m = arr.shape
            self.module.compute_h0_cuda_3d(births, parent, deaths, has_zp, d, n, m)
            z, r, c = torch.meshgrid(torch.arange(d, device=arr.device), torch.arange(n, device=arr.device), torch.arange(m, device=arr.device), indexing='ij')
            u_z = (z[:-1, :, :] * n * m + r[:-1, :, :] * m + c[:-1, :, :]).flatten()
            v_z = ((z[:-1, :, :] + 1) * n * m + r[:-1, :, :] * m + c[:-1, :, :]).flatten()
            u_y = (z[:, :-1, :] * n * m + r[:, :-1, :] * m + c[:, :-1, :]).flatten()
            v_y = (z[:, :-1, :] * n * m + (r[:, :-1, :] + 1) * m + c[:, :-1, :]).flatten()
            u_x = (z[:, :, :-1] * n * m + r[:, :, :-1] * m + c[:, :, :-1]).flatten()
            v_x = (z[:, :, :-1] * n * m + r[:, :, :-1] * m + (c[:, :, :-1] + 1)).flatten()
            u = torch.cat([u_z, u_y, u_x])
            v_ = torch.cat([v_z, v_y, v_x])
            
        parent_flat = parent.long()
        for _ in range(21):
            new_parent = parent_flat[parent_flat]
            if torch.equal(parent_flat, new_parent):
                break
            parent_flat = new_parent
        parent = parent_flat.int()
        
        edge_vals = torch.maximum(births[u], births[v_])
        root_u = parent[u]
        root_v = parent[v_]
        
        valid_saddles = (root_u != root_v)
        root_u = root_u[valid_saddles]
        root_v = root_v[valid_saddles]
        saddle_vals = edge_vals[valid_saddles]
        
        b_u_s = births[root_u.long()]
        b_v_s = births[root_v.long()]
        diff_s = b_u_s - b_v_s
        
        younger_s = torch.where(diff_s > 0, root_u, torch.where(diff_s < 0, root_v, torch.maximum(root_u, root_v)))
        older_s = torch.where(diff_s > 0, root_v, torch.where(diff_s < 0, root_u, torch.minimum(root_u, root_v)))
        
        order = torch.argsort(saddle_vals, stable=True)
        younger_s = younger_s[order].int()
        older_s = older_s[order].int()
        saddle_vals = saddle_vals[order]
        
        self.module.parallel_uf_cuda(parent, deaths, younger_s, older_s, saddle_vals, births, v)
        
        pairs = torch.stack([births, deaths, torch.zeros_like(births)], dim=1)
        return pairs.cpu().numpy()

    def name(self) -> str:
        return "CUDA C++ (Numba-free)"
