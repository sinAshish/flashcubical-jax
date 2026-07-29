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
    u_h = (r[:, :-1] * m + c[:, :-1]).flatten()
    v_h = (r[:, :-1] * m + c[:, 1:]).flatten()
    u_v = (r[:-1, :] * m + c[:-1, :]).flatten()
    v_v = (r[1:, :] * m + c[1:, :]).flatten()
    
    u = jnp.concatenate([u_h, u_v])
    v_ = jnp.concatenate([v_h, v_v])
    
    edge_vals = jnp.maximum(births[u], births[v_])
    
    order = jnp.lexsort((jnp.maximum(u, v_), jnp.minimum(u, v_), edge_vals))
    u = u[order]
    v_ = v_[order]
    edge_vals = edge_vals[order]
    
    parent = jnp.arange(v)
    deaths = jnp.full(v, jnp.inf)
    
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
        ui = u[edge_idx]
        vi = v_[edge_idx]
        val = edge_vals[edge_idx]
        
        root_u, p = find(p, ui)
        root_v, p = find(p, vi)
        
        def do_union(p, d):
            b_u = births[root_u]
            b_v = births[root_v]
            diff = b_u - b_v
            
            older = jnp.where(diff < 0, root_u, jnp.where(diff > 0, root_v, jnp.maximum(root_u, root_v)))
            younger = jnp.where(diff < 0, root_v, jnp.where(diff > 0, root_u, jnp.minimum(root_u, root_v)))
            
            p = p.at[younger].set(older)
            d = d.at[younger].set(val)
            return p, d
            
        p, d = jax.lax.cond(root_u != root_v, do_union, lambda p, d: (p, d), p, d)
        return (p, d), None
        
    (final_p, final_d), _ = jax.lax.scan(step, (parent, deaths), jnp.arange(len(u)))
    pairs = jnp.stack([births, final_d, jnp.zeros(v)], axis=1)
    return pairs

_compute_jit = jax.jit(compute_h0)

class JaxBackend(PHBackend):
    def compute_ph(self, image: Any) -> np.ndarray:
        arr = jnp.asarray(image, dtype=jnp.float64)
        return np.array(_compute_jit(arr))

    def name(self) -> str:
        return "JAX"
