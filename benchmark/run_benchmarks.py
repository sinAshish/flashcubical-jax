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
    sizes = [32, 64, 128, 256, 512, 1024]
    noise_types = ["uniform", "gaussian"]
    
    print(f"{'Grid Size':<12} | {'Noise':<10} | {'Backend':<22} | {'Prep (ms)':<9} | {'Compute (ms)':<12} | {'Accuracy'}")
    print("-" * 85)
    
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
                    # Pre-load data to GPU outside the compute timer
                    t_prep_start = time.perf_counter()
                    prepared_grid = backend.prepare(grid) if hasattr(backend, "prepare") else grid
                    if hasattr(backend, "sync"):
                        backend.sync()
                    prep_time = time.perf_counter() - t_prep_start
                    
                    # Warm-up
                    for _ in range(3):
                        _ = backend.compute_ph(prepared_grid)
                    
                    if hasattr(backend, "sync"):
                        backend.sync()
                        
                    # Timed run for compute only
                    times = []
                    for _ in range(5 if size >= 512 else 10):
                        if hasattr(backend, "sync"):
                            backend.sync()
                        start = time.perf_counter()
                        res = backend.compute_ph(prepared_grid)
                        if hasattr(backend, "sync"):
                            backend.sync()
                        times.append(time.perf_counter() - start)
                        
                    avg_time = float(np.mean(times))
                    
                    # Convert to numpy for verification
                    if hasattr(res, "cpu"):
                        res_np = res.cpu().numpy()
                    elif hasattr(res, "device_buffer"):
                        res_np = np.array(res)
                    elif not isinstance(res, np.ndarray):
                        res_np = np.array(res)
                    else:
                        res_np = res
                    
                    if isinstance(backend, FlashCubicalBackend):
                        baseline_res = res_np
                        print(f"{size}x{size:<7} | {noise:<10} | {backend.name():<22} | {prep_time*1000:<9.2f} | {avg_time*1000:<12.2f} | Baseline")
                    else:
                        is_valid = verify_persistence(baseline_res, res_np)
                        valid_str = "PASS" if is_valid else "FAIL"
                        print(f"{size}x{size:<7} | {noise:<10} | {backend.name():<22} | {prep_time*1000:<9.2f} | {avg_time*1000:<12.2f} | {valid_str}")
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    if "out of memory" in str(e).lower() or isinstance(e, MemoryError):
                        print(f"{size}x{size:<7} | {noise:<10} | {backend.name():<22} | {'OOM':<9} | {'OOM':<12} | N/A")
                    else:
                        print(f"{size}x{size:<7} | {noise:<10} | {backend.name():<22} | {'ERROR':<9} | {'ERROR':<12} | {type(e).__name__}")
                        # traceback.print_exc()
                        
            print("-" * 85)

if __name__ == "__main__":
    run_suite()