import torch
import numpy as np
from typing import Any, Tuple, List
from benchmark.backend_protocol import PHBackend

def _compute_h0_torch(image: torch.Tensor) -> torch.Tensor:
    n, m = image.shape
    v = n * m
    births = image.flatten()
    
    r, c = torch.meshgrid(torch.arange(n, device=image.device), torch.arange(m, device=image.device), indexing='ij')
    u_h = (r[:, :-1] * m + c[:, :-1]).flatten()
    v_h = (r[:, :-1] * m + c[:, 1:]).flatten()
    u_v = (r[:-1, :] * m + c[:-1, :]).flatten()
    v_v = (r[1:, :] * m + c[1:, :]).flatten()
    
    u = torch.cat([u_h, u_v])
    v_ = torch.cat([v_h, v_v])
    
    edge_vals = torch.maximum(births[u], births[v_])
    
    max_uv = torch.maximum(u, v_)
    min_uv = torch.minimum(u, v_)
    
    order = torch.argsort(max_uv, stable=True)
    min_uv = min_uv[order]
    edge_vals = edge_vals[order]
    u = u[order]
    v_ = v_[order]
    
    order2 = torch.argsort(min_uv, stable=True)
    edge_vals = edge_vals[order2]
    u = u[order2]
    v_ = v_[order2]
    
    order3 = torch.argsort(edge_vals, stable=True)
    u = u[order3]
    v_ = v_[order3]
    edge_vals = edge_vals[order3]
    
    # Fast path for python lists
    u_list: List[int] = u.tolist()
    v_list: List[int] = v_.tolist()
    edge_vals_list: List[float] = edge_vals.tolist()
    births_list: List[float] = births.tolist()
    
    parent = list(range(v))
    deaths = [float('inf')] * v
    
    for i in range(len(u_list)):
        ui = u_list[i]
        vi = v_list[i]
        val = edge_vals_list[i]
        
        root_u = ui
        while parent[root_u] != root_u:
            parent[root_u] = parent[parent[root_u]]
            root_u = parent[root_u]
            
        root_v = vi
        while parent[root_v] != root_v:
            parent[root_v] = parent[parent[root_v]]
            root_v = parent[root_v]
            
        if root_u != root_v:
            b_u = births_list[root_u]
            b_v = births_list[root_v]
            diff = b_u - b_v
            
            if diff < 0:
                older = root_u
                younger = root_v
            elif diff > 0:
                older = root_v
                younger = root_u
            else:
                older = max(root_u, root_v)
                younger = min(root_u, root_v)
                
            parent[younger] = older
            deaths[younger] = val
            
    deaths_tensor = torch.tensor(deaths, dtype=image.dtype, device=image.device)
    pairs = torch.stack([births, deaths_tensor, torch.zeros(v, device=image.device, dtype=image.dtype)], dim=1)
    return pairs

class PyTorchBackend(PHBackend):
    def compute_ph(self, image: Any) -> np.ndarray:
        arr = torch.tensor(image, dtype=torch.float64)
        if torch.cuda.is_available():
            arr = arr.cuda()
        # MPS does not support float64, fallback to CPU
        res = _compute_h0_torch(arr)
        return res.cpu().numpy()

    def name(self) -> str:
        return "PyTorch (Hybrid)"
