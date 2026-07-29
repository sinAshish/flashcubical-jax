import torch
import numpy as np
from typing import Any
from numba import njit
from benchmark.backend_protocol import PHBackend

@njit
def _lower_star_numba_2d(births, order, n, m):
    v = n * m
    parent = np.arange(v, dtype=np.int32)
    deaths = np.full(v, np.inf, dtype=np.float64)
    active = np.zeros(v, dtype=np.bool_)
    roots = np.empty(4, dtype=np.int32)
    
    for i in range(v):
        curr = order[i]
        curr_birth = births[curr]
        active[curr] = True
        
        r = curr // m
        c = curr % m
        
        num_roots = 0
        if r > 0:
            nbr = curr - m
            if active[nbr]:
                while parent[nbr] != nbr:
                    parent[nbr] = parent[parent[nbr]]
                    nbr = parent[nbr]
                roots[num_roots] = nbr
                num_roots += 1
                    
        if r < n - 1:
            nbr = curr + m
            if active[nbr]:
                while parent[nbr] != nbr:
                    parent[nbr] = parent[parent[nbr]]
                    nbr = parent[nbr]
                present = False
                for j in range(num_roots):
                    if roots[j] == nbr:
                        present = True
                        break
                if not present:
                    roots[num_roots] = nbr
                    num_roots += 1
                    
        if c > 0:
            nbr = curr - 1
            if active[nbr]:
                while parent[nbr] != nbr:
                    parent[nbr] = parent[parent[nbr]]
                    nbr = parent[nbr]
                present = False
                for j in range(num_roots):
                    if roots[j] == nbr:
                        present = True
                        break
                if not present:
                    roots[num_roots] = nbr
                    num_roots += 1
                    
        if c < m - 1:
            nbr = curr + 1
            if active[nbr]:
                while parent[nbr] != nbr:
                    parent[nbr] = parent[parent[nbr]]
                    nbr = parent[nbr]
                present = False
                for j in range(num_roots):
                    if roots[j] == nbr:
                        present = True
                        break
                if not present:
                    roots[num_roots] = nbr
                    num_roots += 1
                    
        if num_roots > 0:
            best_root = roots[0]
            best_birth = births[best_root]
            
            for j in range(1, num_roots):
                r_n = roots[j]
                b = births[r_n]
                if b < best_birth or (b == best_birth and r_n > best_root):
                    best_root = r_n
                    best_birth = b
                    
            parent[curr] = best_root
            deaths[curr] = curr_birth
            
            for j in range(num_roots):
                r_n = roots[j]
                if r_n != best_root:
                    parent[r_n] = best_root
                    deaths[r_n] = curr_birth
                    
    return deaths

class PyTorchBackend(PHBackend):
    def compute_ph(self, image: Any) -> np.ndarray:
        arr = torch.tensor(image, dtype=torch.float64)
        
        n, m = arr.shape
        v = n * m
        births = arr.flatten()
        
        # CPU Fast Path (Lower-Star + Numba)
        order = torch.argsort(births, stable=True)
        
        # Numba lower star algorithm exactly mirroring C++ baseline logic
        deaths = _lower_star_numba_2d(births.numpy(), order.numpy(), n, m)
        deaths_tensor = torch.tensor(deaths, dtype=torch.float64)
        
        pairs = torch.stack([births, deaths_tensor, torch.zeros(v, dtype=torch.float64)], dim=1)
        return pairs.numpy()

    def name(self) -> str:
        return "PyTorch (Numba+LowerStar)"
