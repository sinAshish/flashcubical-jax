import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from typing import Any
from .._core import resolve_saddles as _resolve_saddles
from .backend_protocol import PHBackend


def phase_1_4(image):
    if image.ndim == 2:
        n, m = image.shape
        v = n * m
        idx = jnp.arange(v, dtype=jnp.int32).reshape(n, m)

        cond_up = jnp.pad(image[:-1, :] <= image[1:, :], ((1, 0), (0, 0)))
        cond_left = jnp.pad(image[:, :-1] <= image[:, 1:], ((0, 0), (1, 0)))
        cond_right = jnp.pad(image[:, 1:] < image[:, :-1], ((0, 0), (0, 1)))
        cond_down = jnp.pad(image[1:, :] < image[:-1, :], ((0, 1), (0, 0)))

        # Phase 1: Parallel Zero-Persistence Pruning (lowest-index older neighbor wins)
        parent = idx
        parent = jnp.where(cond_down, idx + m, parent)
        parent = jnp.where(cond_right, idx + 1, parent)
        parent = jnp.where(cond_left, idx - 1, parent)
        parent = jnp.where(cond_up, idx - m, parent)
        parent_flat = parent.reshape(-1)

        # Phase 2: Parallel Pointer Jumping
        def cond(state):
            curr, prev = state
            return jnp.any(curr != prev)

        def body(state):
            curr, _ = state
            return curr[curr], curr

        parent_flat, _ = jax.lax.while_loop(
            cond, body, (parent_flat, jnp.full(v, -1, dtype=jnp.int32))
        )
        p2d = parent_flat.reshape(n, m)

        # Phase 3: Parallel Saddle Neighbor Bitmask
        s_up = cond_up & jnp.pad(p2d[:-1, :] != p2d[1:, :], ((1, 0), (0, 0)))
        s_left = cond_left & jnp.pad(p2d[:, :-1] != p2d[:, 1:], ((0, 0), (1, 0)))
        s_right = cond_right & jnp.pad(p2d[:, 1:] != p2d[:, :-1], ((0, 0), (0, 1)))
        s_down = cond_down & jnp.pad(p2d[1:, :] != p2d[:-1, :], ((0, 1), (0, 0)))

        saddle_mask = (
            s_up.astype(jnp.uint8)
            | (s_left.astype(jnp.uint8) << 1)
            | (s_right.astype(jnp.uint8) << 2)
            | (s_down.astype(jnp.uint8) << 3)
        ).reshape(-1)
        return parent_flat, saddle_mask
    else:
        d, n, m = image.shape
        plane = n * m
        v = d * plane
        idx = jnp.arange(v, dtype=jnp.int32).reshape(d, n, m)

        cond_z_up = jnp.pad(image[:-1, :, :] <= image[1:, :, :], ((1, 0), (0, 0), (0, 0)))
        cond_y_up = jnp.pad(image[:, :-1, :] <= image[:, 1:, :], ((0, 0), (1, 0), (0, 0)))
        cond_x_left = jnp.pad(image[:, :, :-1] <= image[:, :, 1:], ((0, 0), (0, 0), (1, 0)))
        cond_x_right = jnp.pad(image[:, :, 1:] < image[:, :, :-1], ((0, 0), (0, 0), (0, 1)))
        cond_y_down = jnp.pad(image[:, 1:, :] < image[:, :-1, :], ((0, 0), (0, 1), (0, 0)))
        cond_z_down = jnp.pad(image[1:, :, :] < image[:-1, :, :], ((0, 1), (0, 0), (0, 0)))

        parent = idx
        parent = jnp.where(cond_z_down, idx + plane, parent)
        parent = jnp.where(cond_y_down, idx + m, parent)
        parent = jnp.where(cond_x_right, idx + 1, parent)
        parent = jnp.where(cond_x_left, idx - 1, parent)
        parent = jnp.where(cond_y_up, idx - m, parent)
        parent = jnp.where(cond_z_up, idx - plane, parent)
        parent_flat = parent.reshape(-1)

        def cond(state):
            curr, prev = state
            return jnp.any(curr != prev)

        def body(state):
            curr, _ = state
            return curr[curr], curr

        parent_flat, _ = jax.lax.while_loop(
            cond, body, (parent_flat, jnp.full(v, -1, dtype=jnp.int32))
        )
        p3d = parent_flat.reshape(d, n, m)

        s_z_up = cond_z_up & jnp.pad(p3d[:-1, :, :] != p3d[1:, :, :], ((1, 0), (0, 0), (0, 0)))
        s_y_up = cond_y_up & jnp.pad(p3d[:, :-1, :] != p3d[:, 1:, :], ((0, 0), (1, 0), (0, 0)))
        s_x_left = cond_x_left & jnp.pad(p3d[:, :, :-1] != p3d[:, :, 1:], ((0, 0), (0, 0), (1, 0)))
        s_x_right = cond_x_right & jnp.pad(p3d[:, :, 1:] != p3d[:, :, :-1], ((0, 0), (0, 0), (0, 1)))
        s_y_down = cond_y_down & jnp.pad(p3d[:, 1:, :] != p3d[:, :-1, :], ((0, 0), (0, 1), (0, 0)))
        s_z_down = cond_z_down & jnp.pad(p3d[1:, :, :] != p3d[:-1, :, :], ((0, 1), (0, 0), (0, 0)))

        saddle_mask = (
            s_z_up.astype(jnp.uint8)
            | (s_y_up.astype(jnp.uint8) << 1)
            | (s_x_left.astype(jnp.uint8) << 2)
            | (s_x_right.astype(jnp.uint8) << 3)
            | (s_y_down.astype(jnp.uint8) << 4)
            | (s_z_down.astype(jnp.uint8) << 5)
        ).reshape(-1)
        return parent_flat, saddle_mask


_jit_phases_1_4 = jax.jit(phase_1_4)


class JaxBackend(PHBackend):
    def prepare(self, image: Any) -> Any:
        arr = jax.device_put(jnp.asarray(image, dtype=jnp.float64))
        jax.block_until_ready(arr)
        return arr

    def sync(self):
        pass

    def compute_ph(self, arr: Any) -> Any:
        if not isinstance(arr, jax.Array):
            arr = jnp.asarray(arr, dtype=jnp.float64)

        out = _jit_phases_1_4(arr)
        jax.block_until_ready(out)
        parent_jax, saddle_mask_jax = out

        births_np = np.asarray(arr, dtype=np.float64).reshape(-1)
        parent_np = np.asarray(parent_jax, dtype=np.int32)
        mask_np = np.asarray(saddle_mask_jax, dtype=np.uint8)

        if arr.ndim == 2:
            plane = 0
            cols = int(arr.shape[1])
        else:
            plane = int(arr.shape[1] * arr.shape[2])
            cols = int(arr.shape[2])

        return _resolve_saddles(births_np, parent_np, mask_np, plane, cols)

    def name(self) -> str:
        return "JAX (Parallel Forest)"
