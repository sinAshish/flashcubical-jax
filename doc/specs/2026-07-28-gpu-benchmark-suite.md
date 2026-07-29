# GPU Persistent Homology Benchmarking Suite

## Architecture Overview
The benchmark suite is a standalone Python framework designed to compare the execution time and accuracy of different persistent homology algorithms on GPUs (JAX, PyTorch/Triton) versus the existing C++ baseline. It will use the existing `flash_cubical` as the C++ baseline and provide pluggable backends for JAX and PyTorch/Triton.

## Components and Responsibilities
- **Data Generator**: Creates 2D and 3D scalar grids of fixed sizes (2D: 128x128, 256x256, 512x512, 1024x1024; 3D: 32x32x32, 64x64x64, 128x128x128). Data will consist of uniform noise in [0, 1] and standard Gaussian blobs to test best/average/worst-case inputs.
- **Backend Protocol**: A common interface (`compute_ph(image) -> np.ndarray`) that all implementations must satisfy. The `flash_cubical` baseline is version `0.1.0`. The interface expects the input as a C-contiguous `float64` array and returns a `float64` NumPy array of shape `(n, 3)` where columns are `birth`, `death`, and `homology dimension`.
- **Verification Engine**: Compares the resulting persistence pairs against the C++ baseline. It matches births, deaths, and dimensions, ignoring local zero-persistence pairs.
- **Profiling Engine**: Measures wall-clock execution time, separating GPU memory transfer, JIT/compilation overhead (warm-up runs), and pure execution latency.

## GPU Implementations
Two primary hardware-accelerated algorithms will be implemented:

1. **JAX Parallel Pointer Jumping**:
   - Uses purely tensor-native operations.
   - Extracts and sorts edges based on filtration values.
   - Implements hooking via `jax.ops.segment_min` (or scatter) to allow higher-valued roots to attach to lower-valued ones.
   - Flattens trees using `jax.lax.while_loop` (pointer jumping: `parent[i] = parent[parent[i]]`), flattening the Union-Find structure in logarithmic time.

2. **PyTorch Triton Kernel**:
   - Leverages OpenAI's Triton to write custom GPU kernels in Python.
   - Utilizes `tl.atomic_min` and `tl.atomic_cas` to implement a batched parallel Kruskal approach, explicitly handling union-find tree merges directly in GPU shared/global memory.

## Error Handling and Edge Cases
- **Memory Limits**: Large 3D grids may exceed GPU memory during edge-sorting. The suite will gracefully catch Out-Of-Memory (OOM) exceptions and record them in the benchmark matrix instead of crashing.
- **Zero-Persistence Pairs**: GPU algorithms might output pairs with birth = death. The verifier will explicitly filter these out before comparing with the C++ backend.
- **JIT Compilation**: JAX and Triton incur significant first-run compilation times. The suite will run 3 "warm-up" iterations before starting the timer, then compute the average of 10 execution iterations to exclude compilation latency from benchmark results.

## Testing Approach
- Unit tests will assert that both GPU backends output exactly the same persistence diagram as `flash_cubical.compute` on a set of fixed seeds.

## Out of Scope
- Framework self-testing using a mock backend.
- Benchmarking non-baseline CPU implementations.

## Documentation impact
- Feature / user-facing docs introduced: none
- Materially amended existing docs: none
- Derived / memory docs invalidated: none
