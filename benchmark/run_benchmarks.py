import time
import numpy as np
from benchmark.benchmark_utils import generate_2d_grid, verify_persistence
from benchmark.backend_baseline import FlashCubicalBackend
from benchmark.backend_jax import JaxBackend
from benchmark.backend_pytorch import PyTorchBackend
from benchmark.backend_triton import TritonBackend
from benchmark.backend_cuda import CudaBackend

def run_suite():
    backends = [
        FlashCubicalBackend(), 
        JaxBackend(), 
        PyTorchBackend(), 
        TritonBackend(), 
        CudaBackend()
    ]
    sizes = [32, 64, 128]
    noise_types = ["uniform", "gaussian"]
    
    print(f"{'Grid Size':<12} | {'Noise':<10} | {'Backend':<22} | {'Time (ms)':<10} | {'Accuracy'}")
    print("-" * 75)
    
    for size in sizes:
        for noise in noise_types:
            try:
                grid = generate_2d_grid(size, noise)
            except MemoryError:
                print(f"{size}x{size:<7} | {noise:<10} | OOM during data generation")
                continue
                
            baseline_res = None
            for backend in backends:
                try:
                    # Warm-up
                    for _ in range(3):
                        _ = backend.compute_ph(grid)
                    
                    # Timed run
                    times = []
                    for _ in range(5 if size >= 512 else 10):
                        start = time.perf_counter()
                        res = backend.compute_ph(grid)
                        times.append(time.perf_counter() - start)
                        
                    avg_time = np.mean(times)
                    
                    if isinstance(backend, FlashCubicalBackend):
                        baseline_res = res
                        print(f"{size}x{size:<7} | {noise:<10} | {backend.name():<22} | {avg_time*1000:<10.2f} | Baseline")
                    else:
                        is_valid = verify_persistence(baseline_res, res)
                        valid_str = "PASS" if is_valid else "FAIL"
                        print(f"{size}x{size:<7} | {noise:<10} | {backend.name():<22} | {avg_time*1000:<10.2f} | {valid_str}")
                except Exception as e:
                    if "out of memory" in str(e).lower() or isinstance(e, MemoryError):
                        print(f"{size}x{size:<7} | {noise:<10} | {backend.name():<22} | {'OOM':<10} | N/A")
                    else:
                        print(f"{size}x{size:<7} | {noise:<10} | {backend.name():<22} | {'ERROR':<10} | {type(e).__name__}")
                        
            print("-" * 75)

if __name__ == "__main__":
    run_suite()