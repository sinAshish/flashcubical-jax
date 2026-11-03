import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from typing import Any
from .backend_protocol import PHBackend


def phase_1_4(image):
    v = image.size
    births = image.flatten()

    if image.ndim == 2:
        n, m = image.shape
        r, c = jnp.meshgrid(jnp.arange(n), jnp.arange(m), indexing='ij')
        u = jnp.concatenate([(r[:, :-1] * m + c[:, :-1]).flatten(), (r[:-1, :] * m + c[:-1, :]).flatten()])
        v_ = jnp.concatenate([(r[:, :-1] * m + c[:, 1:]).flatten(), (r[1:, :] * m + c[1:, :]).flatten()])
    else:
        d, n, m = image.shape
        z, r, c = jnp.meshgrid(jnp.arange(d), jnp.arange(n), jnp.arange(m), indexing='ij')
        u_z = (z[:-1, :, :] * n * m + r[:-1, :, :] * m + c[:-1, :, :]).flatten()
        v_z = ((z[:-1, :, :] + 1) * n * m + r[:-1, :, :] * m + c[:-1, :, :]).flatten()
        u_y = (z[:, :-1, :] * n * m + r[:, :-1, :] * m + c[:, :-1, :]).flatten()
        v_y = (z[:, :-1, :] * n * m + (r[:, :-1, :] + 1) * m + c[:, :-1, :]).flatten()
        u_x = (z[:, :, :-1] * n * m + r[:, :, :-1] * m + c[:, :, :-1]).flatten()
        v_x = (z[:, :, :-1] * n * m + r[:, :, :-1] * m + (c[:, :, :-1] + 1)).flatten()
        u = jnp.concatenate([u_z, u_y, u_x])
        v_ = jnp.concatenate([v_z, v_y, v_x])

    b_u = births[u]
    b_v = births[v_]
    edge_vals = jnp.maximum(b_u, b_v)

    younger = jnp.where(b_u > b_v, u, jnp.where(b_u < b_v, v_, jnp.maximum(u, v_)))
    older = jnp.where(b_u > b_v, v_, jnp.where(b_u < b_v, u, jnp.minimum(u, v_)))

    # Phase 1: Parallel Zero-Persistence Pruning
    best_older = jax.ops.segment_min(older, younger, num_segments=v)
    MAX_INT = jnp.iinfo(best_older.dtype).max
    has_zp = (best_older != MAX_INT)

    is_zp = (older == best_older[younger])

    # Phase 2: Parallel Pointer Jumping
    parent = jnp.where(has_zp, best_older, jnp.arange(v))

    def cond(state):
        curr, prev = state
        return jnp.any(curr != prev)

    def body(state):
        curr, _ = state
        return curr[curr], curr

    parent, _ = jax.lax.while_loop(cond, body, (parent, jnp.full(v, -1)))

    # Phase 3: Extract and map saddle edges
    root_u = parent[younger]
    root_v = parent[older]

    is_saddle = (~is_zp) & (root_u != root_v)

    b_u_s = births[root_u]
    b_v_s = births[root_v]

    younger_s = jnp.where(b_u_s > b_v_s, root_u, jnp.where(b_u_s < b_v_s, root_v, jnp.maximum(root_u, root_v)))
    older_s = jnp.where(b_u_s > b_v_s, root_v, jnp.where(b_u_s < b_v_s, root_u, jnp.minimum(root_u, root_v)))

    return younger_s, older_s, edge_vals, is_saddle, births, has_zp


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
        v = arr.size

        # Run JIT-compiled Phases 1-3 on device
        out = _jit_phases_1_4(arr)
        jax.block_until_ready(out)
        younger_s_jax, older_s_jax, edge_vals_jax, is_saddle_jax, births_jax, has_zp_jax = out

        # Convert to numpy arrays on CPU and filter valid saddles before sorting
        is_saddle_np = np.asarray(is_saddle_jax)
        younger_s_np = np.asarray(younger_s_jax)[is_saddle_np]
        older_s_np = np.asarray(older_s_jax)[is_saddle_np]
        saddle_vals_np = np.asarray(edge_vals_jax)[is_saddle_np]
        births = np.asarray(births_jax)
        has_zp = np.asarray(has_zp_jax)

        # Phase 4: Stable sort valid saddle edges on CPU
        order = np.argsort(saddle_vals_np, kind="stable")
        root_u_list = younger_s_np[order].tolist()
        root_v_list = older_s_np[order].tolist()
        saddle_vals_list = saddle_vals_np[order].tolist()
        births_list = births.tolist()

        # Phase 5: Sequential Union-Find strictly on the minimal saddle topology on CPU
        uf_parent = list(range(v))
        deaths = [float('inf')] * v

        for i in range(len(root_u_list)):
            val = saddle_vals_list[i]
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
                deaths[yng_r] = val

        deaths_arr = np.array(deaths, dtype=np.float64)
        deaths_arr[has_zp] = births[has_zp]

        pairs = np.stack([births, deaths_arr, np.zeros(v)], axis=1)
        return pairs

    def name(self) -> str:
        return "JAX (Parallel Forest)"
