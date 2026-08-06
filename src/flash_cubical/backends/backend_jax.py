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
    
    edge_vals = jnp.maximum(births[u], births[v_])
    
    b_u = births[u]
    b_v = births[v_]
    diff = b_u - b_v
    
    younger = jnp.where(diff > 0, u, jnp.where(diff < 0, v_, jnp.maximum(u, v_)))
    older = jnp.where(diff > 0, v_, jnp.where(diff < 0, u, jnp.minimum(u, v_)))
    
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
    
    # Only process valid saddles
    is_saddle = (~is_zp) & (root_u != root_v)
    saddle_vals = jnp.where(is_saddle, edge_vals, jnp.inf)
    
    b_u_s = births[root_u]
    b_v_s = births[root_v]
    diff_s = b_u_s - b_v_s
    
    younger_s = jnp.where(diff_s > 0, root_u, jnp.where(diff_s < 0, root_v, jnp.maximum(root_u, root_v)))
    older_s = jnp.where(diff_s > 0, root_v, jnp.where(diff_s < 0, root_u, jnp.minimum(root_u, root_v)))
    
    return younger_s, older_s, saddle_vals, births, has_zp

_jit_phases_1_4 = jax.jit(phase_1_4)

class JaxBackend(PHBackend):
    def prepare(self, image: Any) -> Any:
        return jax.device_put(jnp.asarray(image, dtype=jnp.float64))

    def sync(self):
        pass

    def compute_ph(self, arr: Any) -> Any:
        v = arr.size
        
        # Run JIT-compiled Phases 1-4 on GPU
        out = _jit_phases_1_4(arr)
        # Block until ready
        jax.block_until_ready(out)
        younger_s_jax, older_s_jax, saddle_vals_jax, births_jax, has_zp_jax = out
        
        # Convert to numpy arrays on CPU
        younger_s_np = np.array(younger_s_jax)
        older_s_np = np.array(older_s_jax)
        saddle_vals_np = np.array(saddle_vals_jax)
        births = np.array(births_jax)
        has_zp = np.array(has_zp_jax)
        
        # Sort on CPU via NumPy to avoid platform-specific stability bugs
        order = np.lexsort((older_s_np, younger_s_np, saddle_vals_np))
        younger_s = younger_s_np[order]
        older_s = older_s_np[order]
        saddle_vals = saddle_vals_np[order]
        
        # Truncate at inf
        valid_count = np.searchsorted(saddle_vals, np.inf)
        younger_s = younger_s[:valid_count]
        older_s = older_s[:valid_count]
        saddle_vals = saddle_vals[:valid_count]
        
        # Phase 5: Sequential Union-Find strictly on the minimal saddle topology on CPU
        root_u_list = younger_s.tolist()
        root_v_list = older_s.tolist()
        saddle_vals_list = saddle_vals.tolist()
        births_list = births.tolist()
        
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
                deaths[yng_r] = val
                
        deaths_arr = np.array(deaths, dtype=np.float64)
        deaths_arr[has_zp] = births[has_zp]
        
        pairs = np.stack([births, deaths_arr, np.zeros(v)], axis=1)
        return pairs

    def name(self) -> str:
        return "JAX (Parallel Forest)"
