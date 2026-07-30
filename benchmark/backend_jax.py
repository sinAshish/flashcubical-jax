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
    # Push ZP edges to the end
    edge_vals_pruned = jnp.where(is_zp, jnp.inf, edge_vals)
    
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
    
    # Phase 4: Stable Sort Saddle Edges
    order = jnp.lexsort((older_s, younger_s, saddle_vals))
    younger_s = younger_s[order]
    older_s = older_s[order]
    saddle_vals = saddle_vals[order]
    
    deaths = jnp.where(has_zp, births, jnp.inf)
    
    # Phase 5: Sequential Union-Find strictly on the sorted saddle edges
    def find_uf(p, i):
        def c(state):
            curr, p_arr = state
            return p_arr[curr] != curr
        def b(state):
            curr, p_arr = state
            nxt = p_arr[curr]
            p_arr = p_arr.at[curr].set(p_arr[nxt])
            return p_arr[curr], p_arr
        root, p = jax.lax.while_loop(c, b, (i, p))
        return root, p

    def step(state, edge_idx):
        p, d = state
        ui = younger_s[edge_idx]
        vi = older_s[edge_idx]
        val = saddle_vals[edge_idx]
        
        def do_work():
            root_u, p_new = find_uf(p, ui)
            root_v, p_new = find_uf(p_new, vi)
            
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
        
    (final_p, final_d), _ = jax.lax.scan(step, (jnp.arange(v), deaths), jnp.arange(len(younger_s)))
    pairs = jnp.stack([births, final_d, jnp.zeros(v)], axis=1)
    return pairs

_compute_jit = jax.jit(compute_h0)

class JaxBackend(PHBackend):
    def prepare(self, image: Any) -> Any:
        return jax.device_put(jnp.asarray(image, dtype=jnp.float64))

    def sync(self):
        pass # jax automatically syncs block_until_ready() if called

    def compute_ph(self, arr: Any) -> Any:
        res = _compute_jit(arr)
        res.block_until_ready()
        return res

    def name(self) -> str:
        return "JAX (Parallel Forest)"
