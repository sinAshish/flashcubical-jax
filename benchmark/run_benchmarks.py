import time
import numpy as np
from benchmark.benchmark_utils import generate_2d_grid, verify_persistence
from benchmark.backend_baseline import FlashCubicalBackend
from benchmark.backend_jax import JaxBackend
from benchmark.backend_pytorch import PyTorchBackend

def run_suite():
    backends = [FlashCubicalBackend(), JaxBackend(), PyTorchBackend()]
    grid = generate_2d_grid(128, "uniform")
    
    print("Running benchmarks on 128x128 2D Grid (Uniform Noise)...")
    baseline_res = None
    
    for backend in backends:
        # Warm-up
        for _ in range(3):
            _ = backend.compute_ph(grid)
        
        # Timed run
        times = []
        for _ in range(10):
            start = time.perf_counter()
            res = backend.compute_ph(grid)
            times.append(time.perf_counter() - start)
            
        avg_time = np.mean(times)
        
        if isinstance(backend, FlashCubicalBackend):
            baseline_res = res
            print(f"{backend.name()}: {avg_time*1000:.2f} ms (Baseline)")
        else:
            is_valid = verify_persistence(baseline_res, res)
            valid_str = "PASS" if is_valid else "FAIL"
            print(f"{backend.name()}: {avg_time*1000:.2f} ms | Accuracy: {valid_str}")

if __name__ == "__main__":
    run_suite()