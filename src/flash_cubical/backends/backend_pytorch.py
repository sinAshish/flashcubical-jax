import torch
import torch.nn.functional as F
import numpy as np
from typing import Any
from .._core import resolve_saddles as _resolve_saddles
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
        dev = arr.device

        if arr.ndim == 2:
            n, m = arr.shape
            idx = torch.arange(v, dtype=torch.int32, device=dev).reshape(n, m)

            cond_up = F.pad(arr[:-1, :] <= arr[1:, :], (0, 0, 1, 0))
            cond_left = F.pad(arr[:, :-1] <= arr[:, 1:], (1, 0, 0, 0))
            cond_right = F.pad(arr[:, 1:] < arr[:, :-1], (0, 1, 0, 0))
            cond_down = F.pad(arr[1:, :] < arr[:-1, :], (0, 0, 0, 1))

            # Phase 1: Parallel Zero-Persistence Pruning
            parent = idx.clone()
            parent = torch.where(cond_down, idx + m, parent)
            parent = torch.where(cond_right, idx + 1, parent)
            parent = torch.where(cond_left, idx - 1, parent)
            parent = torch.where(cond_up, idx - m, parent)
            parent_flat = parent.reshape(-1).long()

            # Phase 2: Parallel Pointer Jumping
            num_steps = max(1, int(np.ceil(np.log2(max(v, 2)))))
            for _ in range(num_steps):
                parent_flat = parent_flat[parent_flat]
            parent_i32 = parent_flat.to(torch.int32)
            p2d = parent_i32.reshape(n, m)

            # Phase 3: Parallel Saddle Neighbor Bitmask
            s_up = cond_up & F.pad(p2d[:-1, :] != p2d[1:, :], (0, 0, 1, 0))
            s_left = cond_left & F.pad(p2d[:, :-1] != p2d[:, 1:], (1, 0, 0, 0))
            s_right = cond_right & F.pad(p2d[:, 1:] != p2d[:, :-1], (0, 1, 0, 0))
            s_down = cond_down & F.pad(p2d[1:, :] != p2d[:-1, :], (0, 0, 0, 1))

            saddle_mask = (
                s_up.to(torch.uint8)
                | (s_left.to(torch.uint8) << 1)
                | (s_right.to(torch.uint8) << 2)
                | (s_down.to(torch.uint8) << 3)
            ).reshape(-1)

            plane = 0
            cols = int(m)
        else:
            d, n, m = arr.shape
            plane_sz = n * m
            idx = torch.arange(v, dtype=torch.int32, device=dev).reshape(d, n, m)

            cond_z_up = F.pad(arr[:-1, :, :] <= arr[1:, :, :], (0, 0, 0, 0, 1, 0))
            cond_y_up = F.pad(arr[:, :-1, :] <= arr[:, 1:, :], (0, 0, 1, 0, 0, 0))
            cond_x_left = F.pad(arr[:, :, :-1] <= arr[:, :, 1:], (1, 0, 0, 0, 0, 0))
            cond_x_right = F.pad(arr[:, :, 1:] < arr[:, :, :-1], (0, 1, 0, 0, 0, 0))
            cond_y_down = F.pad(arr[:, 1:, :] < arr[:, :-1, :], (0, 0, 0, 1, 0, 0))
            cond_z_down = F.pad(arr[1:, :, :] < arr[:-1, :, :], (0, 0, 0, 0, 0, 1))

            parent = idx.clone()
            parent = torch.where(cond_z_down, idx + plane_sz, parent)
            parent = torch.where(cond_y_down, idx + m, parent)
            parent = torch.where(cond_x_right, idx + 1, parent)
            parent = torch.where(cond_x_left, idx - 1, parent)
            parent = torch.where(cond_y_up, idx - m, parent)
            parent = torch.where(cond_z_up, idx - plane_sz, parent)
            parent_flat = parent.reshape(-1).long()

            num_steps = max(1, int(np.ceil(np.log2(max(v, 2)))))
            for _ in range(num_steps):
                parent_flat = parent_flat[parent_flat]
            parent_i32 = parent_flat.to(torch.int32)
            p3d = parent_i32.reshape(d, n, m)

            s_z_up = cond_z_up & F.pad(p3d[:-1, :, :] != p3d[1:, :, :], (0, 0, 0, 0, 1, 0))
            s_y_up = cond_y_up & F.pad(p3d[:, :-1, :] != p3d[:, 1:, :], (0, 0, 1, 0, 0, 0))
            s_x_left = cond_x_left & F.pad(p3d[:, :, :-1] != p3d[:, :, 1:], (1, 0, 0, 0, 0, 0))
            s_x_right = cond_x_right & F.pad(p3d[:, :, 1:] != p3d[:, :, :-1], (0, 1, 0, 0, 0, 0))
            s_y_down = cond_y_down & F.pad(p3d[:, 1:, :] != p3d[:, :-1, :], (0, 0, 0, 1, 0, 0))
            s_z_down = cond_z_down & F.pad(p3d[1:, :, :] != p3d[:-1, :, :], (0, 0, 0, 0, 0, 1))

            saddle_mask = (
                s_z_up.to(torch.uint8)
                | (s_y_up.to(torch.uint8) << 1)
                | (s_x_left.to(torch.uint8) << 2)
                | (s_x_right.to(torch.uint8) << 3)
                | (s_y_down.to(torch.uint8) << 4)
                | (s_z_down.to(torch.uint8) << 5)
            ).reshape(-1)

            plane = int(plane_sz)
            cols = int(m)

        births_np = arr.detach().cpu().numpy().reshape(-1)
        parent_np = parent_i32.detach().cpu().numpy()
        mask_np = saddle_mask.detach().cpu().numpy()

        return _resolve_saddles(births_np, parent_np, mask_np, plane, cols)

    def name(self) -> str:
        return "PyTorch (Parallel Forest)"
