import torch
import torch.nn.functional as F
import numpy as np
from typing import Any
from .._core import resolve_saddles as _resolve_saddles
from .backend_protocol import PHBackend

# OpenAI Triton is strictly compiled and executed on NVIDIA GPUs.
# We import triton inside a conditional block to allow the benchmark suite
# to load and pass baseline tests on CPU-only or non-NVIDIA machines.
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

if HAS_TRITON:
    @triton.jit
    def triton_prepass_2d_kernel(
        births_ptr, parent_ptr, deaths_ptr, has_zp_ptr, n, m,
        LUT_ptr, BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        v = n * m
        mask = offsets < v

        curr = offsets
        curr_birth = tl.load(births_ptr + curr, mask=mask)

        r = curr // m
        c = curr % m

        # Build 4-bit neighbor mask (4-connected in 2D)
        n_mask = 0

        # Up (r > 0)
        valid_up = (r > 0) & mask
        birth_up = tl.load(births_ptr + (curr - m), mask=valid_up, other=float('inf'))
        is_up_older = (birth_up < curr_birth) | ((birth_up == curr_birth) & ((curr - m) < curr))
        n_mask = tl.where(is_up_older & valid_up, n_mask | 1, n_mask)

        # Down (r < n - 1)
        valid_down = (r < n - 1) & mask
        birth_down = tl.load(births_ptr + (curr + m), mask=valid_down, other=float('inf'))
        is_down_older = (birth_down < curr_birth) | ((birth_down == curr_birth) & ((curr + m) < curr))
        n_mask = tl.where(is_down_older & valid_down, n_mask | 2, n_mask)

        # Left (c > 0)
        valid_left = (c > 0) & mask
        birth_left = tl.load(births_ptr + (curr - 1), mask=valid_left, other=float('inf'))
        is_left_older = (birth_left < curr_birth) | ((birth_left == curr_birth) & ((curr - 1) < curr))
        n_mask = tl.where(is_left_older & valid_left, n_mask | 4, n_mask)

        # Right (c < m - 1)
        valid_right = (c < m - 1) & mask
        birth_right = tl.load(births_ptr + (curr + 1), mask=valid_right, other=float('inf'))
        is_right_older = (birth_right < curr_birth) | ((birth_right == curr_birth) & ((curr + 1) < curr))
        n_mask = tl.where(is_right_older & valid_right, n_mask | 8, n_mask)

        zp_val = tl.load(LUT_ptr + n_mask, mask=(n_mask > 0) & mask, other=255)

        best_older = curr
        best_older = tl.where(zp_val == 0, curr - m, best_older)
        best_older = tl.where(zp_val == 1, curr + m, best_older)
        best_older = tl.where(zp_val == 2, curr - 1, best_older)
        best_older = tl.where(zp_val == 3, curr + 1, best_older)

        has_zp = (zp_val != 255) & (n_mask > 0)

        tl.store(parent_ptr + curr, tl.where(has_zp, best_older, curr), mask=mask)
        tl.store(deaths_ptr + curr, tl.where(has_zp, curr_birth, float('inf')), mask=mask)
        tl.store(has_zp_ptr + curr, has_zp, mask=mask)


    @triton.jit
    def triton_prepass_3d_kernel(
        births_ptr, parent_ptr, deaths_ptr, has_zp_ptr, d, n, m,
        BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        v = d * n * m
        mask = offsets < v

        curr = offsets
        curr_birth = tl.load(births_ptr + curr, mask=mask)

        z = curr // (n * m)
        r = (curr // m) % n
        c = curr % m

        best_older = curr
        min_older_birth = curr_birth
        found_older = tl.zeros([BLOCK_SIZE], dtype=tl.int1)

        # Z Up (z > 0)
        valid = (z > 0) & mask
        nbr = curr - (n * m)
        birth = tl.load(births_ptr + nbr, mask=valid, other=float('inf'))
        is_older = (birth < curr_birth) | ((birth == curr_birth) & (nbr < curr))
        is_better = (birth < min_older_birth) | ((birth == min_older_birth) & (nbr < best_older))
        update = is_older & (~found_older | is_better) & valid
        best_older = tl.where(update, nbr, best_older)
        min_older_birth = tl.where(update, birth, min_older_birth)
        found_older = found_older | (is_older & valid)

        # Z Down (z < d - 1)
        valid = (z < d - 1) & mask
        nbr = curr + (n * m)
        birth = tl.load(births_ptr + nbr, mask=valid, other=float('inf'))
        is_older = (birth < curr_birth) | ((birth == curr_birth) & (nbr < curr))
        is_better = (birth < min_older_birth) | ((birth == min_older_birth) & (nbr < best_older))
        update = is_older & (~found_older | is_better) & valid
        best_older = tl.where(update, nbr, best_older)
        min_older_birth = tl.where(update, birth, min_older_birth)
        found_older = found_older | (is_older & valid)

        # Y Up (r > 0)
        valid = (r > 0) & mask
        nbr = curr - m
        birth = tl.load(births_ptr + nbr, mask=valid, other=float('inf'))
        is_older = (birth < curr_birth) | ((birth == curr_birth) & (nbr < curr))
        is_better = (birth < min_older_birth) | ((birth == min_older_birth) & (nbr < best_older))
        update = is_older & (~found_older | is_better) & valid
        best_older = tl.where(update, nbr, best_older)
        min_older_birth = tl.where(update, birth, min_older_birth)
        found_older = found_older | (is_older & valid)

        # Y Down (r < n - 1)
        valid = (r < n - 1) & mask
        nbr = curr + m
        birth = tl.load(births_ptr + nbr, mask=valid, other=float('inf'))
        is_older = (birth < curr_birth) | ((birth == curr_birth) & (nbr < curr))
        is_better = (birth < min_older_birth) | ((birth == min_older_birth) & (nbr < best_older))
        update = is_older & (~found_older | is_better) & valid
        best_older = tl.where(update, nbr, best_older)
        min_older_birth = tl.where(update, birth, min_older_birth)
        found_older = found_older | (is_older & valid)

        # X Left (c > 0)
        valid = (c > 0) & mask
        nbr = curr - 1
        birth = tl.load(births_ptr + nbr, mask=valid, other=float('inf'))
        is_older = (birth < curr_birth) | ((birth == curr_birth) & (nbr < curr))
        is_better = (birth < min_older_birth) | ((birth == min_older_birth) & (nbr < best_older))
        update = is_older & (~found_older | is_better) & valid
        best_older = tl.where(update, nbr, best_older)
        min_older_birth = tl.where(update, birth, min_older_birth)
        found_older = found_older | (is_older & valid)

        # X Right (c < m - 1)
        valid = (c < m - 1) & mask
        nbr = curr + 1
        birth = tl.load(births_ptr + nbr, mask=valid, other=float('inf'))
        is_older = (birth < curr_birth) | ((birth == curr_birth) & (nbr < curr))
        is_better = (birth < min_older_birth) | ((birth == min_older_birth) & (nbr < best_older))
        update = is_older & (~found_older | is_better) & valid
        best_older = tl.where(update, nbr, best_older)
        min_older_birth = tl.where(update, birth, min_older_birth)
        found_older = found_older | (is_older & valid)

        # Store results
        tl.store(parent_ptr + curr, tl.where(found_older, best_older, curr), mask=mask)
        tl.store(deaths_ptr + curr, tl.where(found_older, curr_birth, float('inf')), mask=mask)
        tl.store(has_zp_ptr + curr, found_older, mask=mask)


class TritonBackend(PHBackend):
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

        # Fallback to C++ CPU (H0 only) if Triton is not compiled or CUDA is missing
        if not HAS_TRITON or not torch.cuda.is_available():
            import flash_cubical as fc
            np_arr = arr.cpu().numpy() if hasattr(arr, "cpu") else np.asarray(arr)
            res = fc.compute(np_arr, h1=(np_arr.ndim == 2)).values
            return res[res[:, 2] == 0]

        if not arr.is_cuda:
            arr = arr.cuda()

        v = arr.numel()
        births = arr.flatten()
        parent = torch.arange(v, dtype=torch.int32, device=arr.device)
        deaths = torch.full((v,), float('inf'), dtype=torch.float64, device=arr.device)
        has_zp = torch.zeros(v, dtype=torch.bool, device=arr.device)

        grid = lambda meta: (triton.cdiv(v, meta['BLOCK_SIZE']),)

        if arr.ndim == 2:
            n, m = arr.shape
            lut = torch.full((256,), fill_value=255, dtype=torch.uint8, device=arr.device)
            for mask_val in range(1, 16):
                if mask_val & 1:
                    lut[mask_val] = 0
                elif mask_val & 4:
                    lut[mask_val] = 2
                elif mask_val & 8:
                    lut[mask_val] = 3
                elif mask_val & 2:
                    lut[mask_val] = 1
            triton_prepass_2d_kernel[grid](
                births, parent, deaths, has_zp, n, m, lut, BLOCK_SIZE=512
            )
        else:
            d, n, m = arr.shape
            triton_prepass_3d_kernel[grid](
                births, parent, deaths, has_zp, d, n, m, BLOCK_SIZE=512
            )

        parent_flat = parent.long()
        num_steps = max(1, int(np.ceil(np.log2(max(v, 2)))))
        for _ in range(num_steps):
            parent_flat = parent_flat[parent_flat]
        parent_i32 = parent_flat.to(torch.int32)

        if arr.ndim == 2:
            n, m = arr.shape
            p2d = parent_i32.reshape(n, m)
            cond_up = F.pad(arr[:-1, :] <= arr[1:, :], (0, 0, 1, 0))
            cond_left = F.pad(arr[:, :-1] <= arr[:, 1:], (1, 0, 0, 0))
            cond_right = F.pad(arr[:, 1:] < arr[:, :-1], (0, 1, 0, 0))
            cond_down = F.pad(arr[1:, :] < arr[:-1, :], (0, 0, 0, 1))

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
            p3d = parent_i32.reshape(d, n, m)
            cond_z_up = F.pad(arr[:-1, :, :] <= arr[1:, :, :], (0, 0, 0, 0, 1, 0))
            cond_y_up = F.pad(arr[:, :-1, :] <= arr[:, 1:, :], (0, 0, 1, 0, 0, 0))
            cond_x_left = F.pad(arr[:, :, :-1] <= arr[:, :, 1:], (1, 0, 0, 0, 0, 0))
            cond_x_right = F.pad(arr[:, :, 1:] < arr[:, :, :-1], (0, 1, 0, 0, 0, 0))
            cond_y_down = F.pad(arr[:, 1:, :] < arr[:, :-1, :], (0, 0, 0, 1, 0, 0))
            cond_z_down = F.pad(arr[1:, :, :] < arr[:-1, :, :], (0, 0, 0, 0, 0, 1))

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
            plane = int(n * m)
            cols = int(m)

        births_np = births.detach().cpu().numpy()
        parent_np = parent_i32.detach().cpu().numpy()
        mask_np = saddle_mask.detach().cpu().numpy()

        return _resolve_saddles(births_np, parent_np, mask_np, plane, cols)

    def name(self) -> str:
        return "Triton (Parallel Forest)"
