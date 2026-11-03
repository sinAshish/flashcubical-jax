import torch
import numpy as np
from typing import Any
from .backend_protocol import PHBackend


class PyTorchBackend(PHBackend):
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

        v = arr.numel()
        births = arr.flatten()

        # Parallel Zero-Persistence Gradient Flow + Sequential Saddle Merge
        if arr.ndim == 2:
            n, m = arr.shape
            r, c = torch.meshgrid(torch.arange(n, device=arr.device), torch.arange(m, device=arr.device), indexing='ij')
            u = torch.cat([(r[:, :-1] * m + c[:, :-1]).flatten(), (r[:-1, :] * m + c[:-1, :]).flatten()])
            v_ = torch.cat([(r[:, :-1] * m + c[:, 1:]).flatten(), (r[1:, :] * m + c[1:, :]).flatten()])
        else:
            d, n, m = arr.shape
            z, r, c = torch.meshgrid(torch.arange(d, device=arr.device), torch.arange(n, device=arr.device), torch.arange(m, device=arr.device), indexing='ij')
            u_z = (z[:-1, :, :] * n * m + r[:-1, :, :] * m + c[:-1, :, :]).flatten()
            v_z = ((z[:-1, :, :] + 1) * n * m + r[:-1, :, :] * m + c[:-1, :, :]).flatten()
            u_y = (z[:, :-1, :] * n * m + r[:, :-1, :] * m + c[:, :-1, :]).flatten()
            v_y = (z[:, :-1, :] * n * m + (r[:, :-1, :] + 1) * m + c[:, :-1, :]).flatten()
            u_x = (z[:, :, :-1] * n * m + r[:, :, :-1] * m + c[:, :, :-1]).flatten()
            v_x = (z[:, :, :-1] * n * m + r[:, :, :-1] * m + (c[:, :, :-1] + 1)).flatten()
            u = torch.cat([u_z, u_y, u_x])
            v_ = torch.cat([v_z, v_y, v_x])

        b_u = births[u]
        b_v = births[v_]
        edge_vals = torch.maximum(b_u, b_v)

        younger = torch.where(b_u > b_v, u, torch.where(b_u < b_v, v_, torch.maximum(u, v_)))
        older = torch.where(b_u > b_v, v_, torch.where(b_u < b_v, u, torch.minimum(u, v_)))

        # Phase 1: Parallel Zero-Persistence Pruning
        best_older = torch.full((v,), fill_value=v, dtype=torch.int64, device=arr.device)
        best_older.scatter_reduce_(0, younger, older, reduce="amin", include_self=False)

        is_zp = (older == best_older[younger])

        # Phase 2: Parallel Pointer Jumping
        parent = best_older.clone()
        no_best = (parent == v)
        parent[no_best] = torch.arange(v, dtype=torch.int64, device=arr.device)[no_best]

        # Path doubling flattens the forest natively in tensors without per-step device sync
        num_steps = max(1, int(np.ceil(np.log2(max(v, 2)))))
        for _ in range(num_steps):
            parent = parent[parent]

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

        younger_s = torch.where(b_u_s > b_v_s, root_u, torch.where(b_u_s < b_v_s, root_v, torch.maximum(root_u, root_v)))
        older_s = torch.where(b_u_s > b_v_s, root_v, torch.where(b_u_s < b_v_s, root_u, torch.minimum(root_u, root_v)))

        # Phase 4: Stable sort on host CPU using NumPy
        saddle_vals_np = saddle_vals.cpu().numpy()
        younger_s_np = younger_s.cpu().numpy()
        older_s_np = older_s.cpu().numpy()

        order = np.argsort(saddle_vals_np, kind="stable")

        younger_s_list = younger_s_np[order].tolist()
        older_s_list = older_s_np[order].tolist()
        saddle_vals_list = saddle_vals_np[order].tolist()
        births_list = births.cpu().tolist()

        # Phase 5: Sequential Union-Find on minimal saddle topology
        uf_parent = list(range(v))
        deaths = [float('inf')] * v

        for i in range(len(younger_s_list)):
            ru = younger_s_list[i]
            while uf_parent[ru] != ru:
                uf_parent[ru] = uf_parent[uf_parent[ru]]
                ru = uf_parent[ru]

            rv = older_s_list[i]
            while uf_parent[rv] != rv:
                uf_parent[rv] = uf_parent[uf_parent[rv]]
                rv = uf_parent[rv]

            if ru != rv:
                bu = births_list[ru]
                bv = births_list[rv]

                if bu < bv:
                    old_r = ru
                    yng_r = rv
                elif bu > bv:
                    old_r = rv
                    yng_r = ru
                else:
                    old_r = min(ru, rv)
                    yng_r = max(ru, rv)

                uf_parent[yng_r] = old_r
                deaths[yng_r] = saddle_vals_list[i]

        deaths_tensor = torch.tensor(deaths, dtype=arr.dtype, device=arr.device)
        deaths_tensor[~no_best] = births[~no_best]
        pairs = torch.stack([births, deaths_tensor, torch.zeros(v, device=arr.device, dtype=arr.dtype)], dim=1)

        return pairs

    def name(self) -> str:
        return "PyTorch (Parallel Forest)"
