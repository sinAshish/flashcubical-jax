import torch
import numpy as np
from typing import Any
from .backend_protocol import PHBackend

class PyTorchBackend(PHBackend):
    def prepare(self, image: Any) -> Any:
        arr = torch.tensor(image, dtype=torch.float64)
        if torch.cuda.is_available():
            arr = arr.cuda()
        return arr

    def sync(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def compute_ph(self, arr: Any) -> np.ndarray:
        n, m = arr.shape
        v = n * m
        births = arr.flatten()
        
        # Novel PyTorch algorithm: Parallel Zero-Persistence Gradient Flow + Sequential Saddle Merge
        r, c = torch.meshgrid(torch.arange(n, device=arr.device), torch.arange(m, device=arr.device), indexing='ij')
        u = torch.cat([(r[:, :-1] * m + c[:, :-1]).flatten(), (r[:-1, :] * m + c[:-1, :]).flatten()])
        v_ = torch.cat([(r[:, :-1] * m + c[:, 1:]).flatten(), (r[1:, :] * m + c[1:, :]).flatten()])
        
        edge_vals = torch.maximum(births[u], births[v_])
        
        b_u = births[u]
        b_v = births[v_]
        diff = b_u - b_v
        
        younger = torch.where(diff > 0, u, torch.where(diff < 0, v_, torch.maximum(u, v_)))
        older = torch.where(diff > 0, v_, torch.where(diff < 0, u, torch.minimum(u, v_)))
        
        # Phase 1: Parallel Zero-Persistence Pruning
        best_older = torch.full((v,), fill_value=v, dtype=torch.int64, device=arr.device)
        best_older.scatter_reduce_(0, younger, older, reduce="amin", include_self=False)
        
        is_zp = (older == best_older[younger])
        
        # Phase 2: Parallel Pointer Jumping
        parent = best_older.clone()
        no_best = (parent == v)
        parent[no_best] = torch.arange(v, dtype=torch.int64, device=arr.device)[no_best]
        
        # Path doubling flattens the forest natively in tensors
        for _ in range(21):
            new_parent = parent[parent]
            if torch.equal(parent, new_parent):
                break
            parent = new_parent
            
        # Phase 3: Extract and map saddle edges
        mask = ~is_zp
        younger_rem = younger[mask]
        older_rem = older[mask]
        edge_vals_rem = edge_vals[mask]
        
        root_u = parent[younger_rem]
        root_v = parent[older_rem]
        
        valid_saddles = (root_u != root_v)
        root_u = root_u[valid_saddles]
        root_v = root_v[valid_saddles]
        saddle_vals = edge_vals_rem[valid_saddles]
        
        b_u_s = births[root_u]
        b_v_s = births[root_v]
        diff_s = b_u_s - b_v_s
        
        younger_s = torch.where(diff_s > 0, root_u, torch.where(diff_s < 0, root_v, torch.maximum(root_u, root_v)))
        older_s = torch.where(diff_s > 0, root_v, torch.where(diff_s < 0, root_u, torch.minimum(root_u, root_v)))
        
        # Phase 4: Stable Sort Saddle Edges
        order = torch.argsort(saddle_vals, stable=True)
        
        younger_s = younger_s[order]
        older_s = older_s[order]
        saddle_vals = saddle_vals[order]
        
        # Phase 5: Sequential Union-Find strictly on the minimal saddle topology
        root_u_list = younger_s.tolist()
        root_v_list = older_s.tolist()
        saddle_vals_list = saddle_vals.tolist()
        births_list = births.tolist()
        
        uf_parent = list(range(v))
        deaths = [float('inf')] * v
        
        for i in range(len(root_u_list)):
            ru = root_u_list[i]
            while uf_parent[ru] != ru:
                uf_parent[ru] = uf_parent[uf_parent[ru]]
                ru = uf_parent[ru]
                
            rv = root_v_list[i]
            while uf_parent[rv] != rv:
                uf_parent[rv] = uf_parent[uf_parent[rv]]
                rv = uf_parent[rv]
                
            if ru != rv:
                bu = births_list[ru]
                bv = births_list[rv]
                diff_b = bu - bv
                
                if diff_b < 0:
                    old_r = ru
                    yng_r = rv
                elif diff_b > 0:
                    old_r = rv
                    yng_r = ru
                else:
                    old_r = max(ru, rv)
                    yng_r = min(ru, rv)
                    
                uf_parent[yng_r] = old_r
                deaths[yng_r] = saddle_vals_list[i]
                
        deaths_tensor = torch.tensor(deaths, dtype=arr.dtype, device=arr.device)
        deaths_tensor[~no_best] = births[~no_best]
        pairs = torch.stack([births, deaths_tensor, torch.zeros(v, device=arr.device, dtype=arr.dtype)], dim=1)
        
        return pairs

    def name(self) -> str:
        return "PyTorch (Parallel Forest)"
