import os
import time
import warnings
warnings.filterwarnings('ignore', message='.*CUDA initialization.*')
import numpy as np

from flash_cubical.backends.benchmark_utils import generate_2d_grid, generate_3d_grid, verify_persistence
from flash_cubical.backends.backend_baseline import FlashCubicalBackend
from flash_cubical.backends.backend_jax import JaxBackend
from flash_cubical.backends.backend_pytorch import PyTorchBackend
from flash_cubical.backends.backend_triton import TritonBackend
from flash_cubical.backends.backend_cuda import CudaBackend

def run_benchmark(dim, use_gpu):
    os.makedirs("benchmark/results", exist_ok=True)
    
    backends = [FlashCubicalBackend(), JaxBackend(), PyTorchBackend()]
    if dim == 2:
        backends.extend([TritonBackend(), CudaBackend()])
        sizes = [32, 64, 128, 256, 512, 1024]
    else:
        backends.extend([TritonBackend(), CudaBackend()])
        sizes = [16, 32, 64, 128]
        
    noise_types = ["uniform", "gaussian"]
    
    env_name = "GPU (A100)" if use_gpu else "CPU"
    filename = f"benchmark/results/{dim}d_{'gpu' if use_gpu else 'cpu'}.md"
    
    with open(filename, "w") as f:
        f.write(f"# {dim}D Benchmarks - {env_name}\n\n")
        
        for noise in noise_types:
            f.write(f"### {noise.capitalize()} Noise\n\n")
            f.write(f"| Grid Size | Backend | Prep (ms) | Compute (ms) | Accuracy |\n")
            f.write(f"|---|---|---|---|---|\n")
            
            for size in sizes:
                try:
                    if dim == 2:
                        grid = generate_2d_grid(size, noise)
                    else:
                        grid = generate_3d_grid(size, noise)
                except MemoryError:
                    f.write(f"| {size}^{dim} | OOM during data generation | - | - | - |\n")
                    continue
                    
                baseline_res = None
                for backend in backends:
                    try:
                        t_prep_start = time.perf_counter()
                        prepared_grid = backend.prepare(grid) if hasattr(backend, "prepare") else grid
                        if hasattr(backend, "sync"): backend.sync()
                        prep_time = time.perf_counter() - t_prep_start
                        
                        # Warm-up
                        for _ in range(2):
                            _ = backend.compute_ph(prepared_grid)
                        if hasattr(backend, "sync"): backend.sync()
                        
                        times = []
                        iters = 5
                        if dim == 3 and size >= 128: iters = 3
                        
                        for _ in range(iters):
                            if hasattr(backend, "sync"): backend.sync()
                            start = time.perf_counter()
                            res = backend.compute_ph(prepared_grid)
                            if hasattr(backend, "sync"): backend.sync()
                            times.append(time.perf_counter() - start)
                            
                        avg_time = float(np.mean(times))
                        
                        if hasattr(res, "cpu"): res_np = res.cpu().numpy()
                        elif hasattr(res, "device_buffer"): res_np = np.array(res)
                        elif not isinstance(res, np.ndarray): res_np = np.array(res)
                        else: res_np = res
                        
                        if isinstance(backend, FlashCubicalBackend):
                            # For 3D, C++ computes H0, H1, H2. We filter to H0 for comparison.
                            baseline_h0 = res_np[res_np[:, 2] == 0]
                            baseline_res = baseline_h0
                            acc_str = "Baseline"
                        else:
                            # Python backends output H0 pairs directly
                            is_valid = verify_persistence(baseline_res, res_np)
                            acc_str = "PASS" if is_valid else "FAIL"
                            
                        b_name = backend.name()
                        sz_str = f"{size}x{size}" if dim == 2 else f"{size}x{size}x{size}"
                        f.write(f"| {sz_str} | {b_name} | {prep_time*1000:.2f} | {avg_time*1000:.2f} | {acc_str} |\n")
                        print(f"{sz_str} | {noise} | {b_name} | {avg_time*1000:.2f} ms")
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        b_name = backend.name()
                        sz_str = f"{size}x{size}" if dim == 2 else f"{size}x{size}x{size}"
                        if "out of memory" in str(e).lower() or isinstance(e, MemoryError):
                            f.write(f"| {sz_str} | {b_name} | OOM | OOM | N/A |\n")
                        else:
                            f.write(f"| {sz_str} | {b_name} | ERROR | ERROR | {type(e).__name__} |\n")

if __name__ == "__main__":
    import sys
    use_gpu = "--gpu" in sys.argv
    run_benchmark(2, use_gpu)
    run_benchmark(3, use_gpu)
