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
    
    b_u = births[u]
    b_v = births[v_]
    
    diff = b_u - b_v
    
    younger = torch.where(diff > 0, u, torch.where(diff < 0, v_, torch.maximum(u, v_)))
    older = torch.where(diff > 0, v_, torch.where(diff < 0, u, torch.minimum(u, v_)))
    
    # Prune zero persistence pairs
    best_older = torch.full((v,), fill_value=v, dtype=torch.int64, device=image.device)
    best_older.scatter_reduce_(0, younger, older, reduce="amin", include_self=False)
    
    is_zp = (older == best_older[younger])
    
    # Drop them
    mask = ~is_zp
    
    younger_rem = younger[mask]
    older_rem = older[mask]
    edge_vals_rem = edge_vals[mask]
    
    # Sort remaining edges
    order = torch.argsort(older_rem, stable=True)
    younger_rem = younger_rem[order]
    edge_vals_rem = edge_vals_rem[order]
    older_rem = older_rem[order]
    
    order2 = torch.argsort(younger_rem, stable=True)
    edge_vals_rem = edge_vals_rem[order2]
    younger_rem = younger_rem[order2]
    older_rem = older_rem[order2]
    
    order3 = torch.argsort(edge_vals_rem, stable=True)
    younger_rem = younger_rem[order3]
    older_rem = older_rem[order3]
    edge_vals_rem = edge_vals_rem[order3]
    
    younger_list = younger_rem.tolist()
    older_list = older_rem.tolist()
    edge_vals_list = edge_vals_rem.tolist()
    births_list = births.tolist()
    
    # initialize parent
    parent = best_older.clone()
    no_best = (parent == v)
    parent[no_best] = torch.arange(v, device=image.device)[no_best]
    parent = parent.tolist()
    
    deaths = torch.full((v,), float('inf'), device=image.device, dtype=image.dtype)
    deaths[~no_best] = births[~no_best]
    deaths = deaths.tolist()
    
    for i in range(len(younger_list)):
        root_u = younger_list[i]
        while parent[root_u] != root_u:
            parent[root_u] = parent[parent[root_u]]
            root_u = parent[root_u]
            
        root_v = older_list[i]
        while parent[root_v] != root_v:
            parent[root_v] = parent[parent[root_v]]
            root_v = parent[root_v]
            
        if root_u != root_v:
            b_u = births_list[root_u]
            b_v = births_list[root_v]
            diff = b_u - b_v
            
            if diff < 0:
                old_r = root_u
                yng_r = root_v
            elif diff > 0:
                old_r = root_v
                yng_r = root_u
            else:
                old_r = max(root_u, root_v)
                yng_r = min(root_u, root_v)
                
            parent[yng_r] = old_r
            deaths[yng_r] = edge_vals_list[i]
            
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
