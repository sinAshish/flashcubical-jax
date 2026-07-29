import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from typing import Any
from benchmark.backend_protocol import PHBackend

def compute_h0(image):
    n, m = image.shape
    v = n * m
    births = image.flatten()
    
    r, c = jnp.meshgrid(jnp.arange(n), jnp.arange(m), indexing='ij')
    u = jnp.concatenate([(r[:, :-1] * m + c[:, :-1]).flatten(), (r[:-1, :] * m + c[:-1, :]).flatten()])
    v_ = jnp.concatenate([(r[:, :-1] * m + c[:, 1:]).flatten(), (r[1:, :] * m + c[1:, :]).flatten()])
    edge_vals = jnp.maximum(births[u], births[v_])
    
    b_u = births[u]
    b_v = births[v_]
    diff = b_u - b_v
    
    younger = jnp.where(diff > 0, u, jnp.where(diff < 0, v_, jnp.maximum(u, v_)))
    older = jnp.where(diff > 0, v_, jnp.where(diff < 0, u, jnp.minimum(u, v_)))
    
    # zero-persistence pruning via segmented min
    best_older = jax.ops.segment_min(older, younger, num_segments=v)
    MAX_INT = jnp.iinfo(best_older.dtype).max
    has_zp = (best_older != MAX_INT)
    
    is_zp = (older == best_older[younger])
    
    # Push ZP edges to the end of the sort by setting their filtration value to infinity
    edge_vals_pruned = jnp.where(is_zp, jnp.inf, edge_vals)
    
    order = jnp.lexsort((older, younger, edge_vals_pruned))
    younger = younger[order]
    older = older[order]
    edge_vals_sorted = edge_vals_pruned[order]
    
    # Any vertex with a zero-persistence edge starts already merged to its best older neighbor
    parent = jnp.where(has_zp, best_older, jnp.arange(v))
    deaths = jnp.where(has_zp, births, jnp.inf)
    
    def find(p, i):
        def cond(state):
            curr, p_arr = state
            return p_arr[curr] != curr
        def body(state):
            curr, p_arr = state
            nxt = p_arr[curr]
            p_arr = p_arr.at[curr].set(p_arr[nxt])
            return p_arr[curr], p_arr
        root, p = jax.lax.while_loop(cond, body, (i, p))
        return root, p

    def step(state, edge_idx):
        p, d = state
        ui = younger[edge_idx]
        vi = older[edge_idx]
        val = edge_vals_sorted[edge_idx]
        
        def do_work():
            root_u, p_new = find(p, ui)
            root_v, p_new = find(p_new, vi)
            
            def do_union():
                b_u = births[root_u]
                b_v = births[root_v]
                diff = b_u - b_v
                older_root = jnp.where(diff < 0, root_u, jnp.where(diff > 0, root_v, jnp.maximum(root_u, root_v)))
                younger_root = jnp.where(diff < 0, root_v, jnp.where(diff > 0, root_u, jnp.minimum(root_u, root_v)))
                
                p_out = p_new.at[younger_root].set(older_root)
                d_out = d.at[younger_root].set(val)
                return p_out, d_out
                
            return jax.lax.cond(root_u != root_v, do_union, lambda: (p_new, d))
            
        p, d = jax.lax.cond(val != jnp.inf, do_work, lambda: (p, d))
        return (p, d), None
        
    (final_p, final_d), _ = jax.lax.scan(step, (parent, deaths), jnp.arange(len(younger)))
    pairs = jnp.stack([births, final_d, jnp.zeros(v)], axis=1)
    return pairs

_compute_jit = jax.jit(compute_h0)

class JaxBackend(PHBackend):
    def compute_ph(self, image: Any) -> np.ndarray:
        arr = jnp.asarray(image, dtype=jnp.float64)
        return np.array(_compute_jit(arr))

    def name(self) -> str:
        return "JAX"