# Native GPU Kernels for V-Filtration Cubical Persistent Homology

## Architecture Overview
This specification details the design and integration of two separate GPU-accelerated backends for `FlashCubical`: a native C++/CUDA C++ extension and a Python-native Triton kernel. Both backends will implement the exact lower-star 2D and 3D persistent homology (H0/connected components) algorithm, utilizing hardware-level parallelism.

The C++/CUDA and Triton backends will serve as pluggable alternatives to the existing C++ CPU baseline, extending performance gains to massive 3D scalar volumes.

## Components and Responsibilities

### 1. C++/CUDA Backend (`flash_cubical_cuda`)
- **Lookup Table Initialization**: Loads the 2D (256-entry) or 3D zero-persistence byte table directly into GPU `__constant__` memory at runtime depending on the input dimensions.
- **Prepass CUDA Kernel**: Spawns 1 thread per pixel/voxel to extract local neighbor masks (up to 4 neighbors in 2D, 6 neighbors in 3D), query constant memory, and flag zero-persistence vertices.
- **Atomic Union-Find CUDA Kernel**: Runs a custom Parallel Union-Find block utilizing `atomicCAS` (Compare-And-Swap) to merge the remaining saddle edges and compute births and deaths directly in GPU VRAM.

### 2. OpenAI Triton Backend (`flash_cubical_triton`)
- **Shared Memory Lookup**: Allocates and caches the 2D or 3D lookup table in GPU L1 Shared Memory per thread block.
- **Triton Union-Find Kernel**: A pure Python-written Triton GPU kernel that loads coalesced batches of edges (2D grid edges or 3D volume edges), performs parallel path compression, and updates parents using `tl.atomic_cas` and `tl.atomic_min`.

### 3. Unified PyTorch Interface
Both backends will adhere to the common `PHBackend` protocol.

The interface will strictly accept a standard PyTorch GPU Tensor of type `float32` or `float64` of shape `(H, W)` (2D image) or `(D, H, W)` (3D volume). If the input tensor is located on the CPU, the backend will automatically move it to the active GPU device. Tensors with any other dimensions will raise a `ValueError` immediately. The interface will return an `(n, 3)` `float64` tensor containing birth, death, and homology dimension.

## Data Flow
1. **Host-to-Device Copy**: Input 2D or 3D grid tensor is moved to GPU VRAM (if not already there).
2. **Zero-Persistence Prepass**: Parallel neighbor analysis (4-way for 2D, 6-way for 3D) assigns zero-persistence status.
3. **Parallel Union-Find (Atomic)**: CUDA/Triton threads execute parallel tree-merging with hardware atomics.
4. **Homogenization**: Births, deaths, and dimension (0.0) are structured into a 2D tensor of shape `(n, 3)`.
5. **Output**: Result is returned as a GPU-resident PyTorch tensor to prevent Device-to-Host transfer bottlenecks.

## Error Handling and Edge Cases
- **Warp Serialization**: In heavily degenerate inputs (such as checkerboard or flat grids), multiple threads attempt to update the same roots simultaneously. The kernels will resolve this using bounded path-halving arrays with a fixed upper step limit of 32 to guarantee termination and prevent infinite CAS polling.
- **Device Availability**: If no NVIDIA CUDA-compatible GPU is available, the backends will raise a descriptive `RuntimeError` on import or initialization, rather than failing silently.

## Testing Approach
- We will verify complete topological accuracy against the `FlashCubical` C++ CPU baseline using 2D and 3D Uniform Noise, Gaussian Blobs, flat grids, and checkerboard volumes up to $1024 \times 1024$ (for 2D) and $128 \times 128 \times 128$ (for 3D) within PyTest suites.
- The tests will assert that all three dimensions (birth, death, and homology dimension) match the baseline results exactly for integer outputs, and within a $10^{-5}$ tolerance for floating-point outputs.

## Flagged External References

This specification relies on several load-bearing external contracts, files, and academic works that are not fully inlined in this document:

1. **The `PHBackend` Protocol**:
   - *Reference*: Currently imported from `benchmark.backend_protocol` in `benchmark/backend_jax.py` but the protocol file is not fully defined or present in the current codebase.
   - *Recommendation*: Inline the concrete Python class definition of `PHBackend` directly into this specification and guarantee that `benchmark/backend_protocol.py` is present.

2. **The V-Filtration Algorithm Paper (arXiv:2606.04801)**:
   - *Reference*: `https://arxiv.org/abs/2606.04801` (proposed ideas for V-filtration cubical persistent homology and lookup tables).
   - *Recommendation*: Inline or explicitly document the core math, neighbor-extraction formulas, and local zero-persistence definitions from the paper within this specification to make it self-contained.

3. **Precomputed Lookup Table Binaries**:
   - *Reference*: The binary files `src/flash_cubical/data/zero_lookup_2d.bin` and `src/flash_cubical/data/zero_lookup_3d.bin`.
   - *Recommendation*: Document the exact file formats, structures, and schema versions of these tables within the specification to ensure the GPU kernels parse them consistently.

4. **The Existing C++ CPU Baseline (`flash_cubical.compute` / `cpp/compute_2d_all.cpp`)**:
   - *Reference*: The baseline implementation of the lower-star persistent homology algorithm.
   - *Recommendation*: Inline the key CPU step descriptions (e.g. how neighbor queries map to edge/saddle-points) to ensure complete alignment between the CPU baseline and GPU kernels.

## Out of Scope
- Homology dimensions other than $H_0$ (connected components) for these GPU kernels (H1/H2 will be ignored).
- AMD/ROCm compilation or execution support for the CUDA C++ backend (the CUDA backend will strictly target NVIDIA hardware).
- Automatic differentiation / backpropagation (no backward pass or gradient calculation will be implemented for these kernels).
- Multi-GPU execution or distributed GPU workloads (execution is strictly restricted to a single GPU device).
- Custom memory allocation pools beyond the standard PyTorch CUDA caching allocator.
- Integration with machine learning or training loops beyond raw PyTorch tensor inputs (e.g., no integration with PyTorch Lightning or HuggingFace).
- Support for non-cubical grids, such as simplicial complexes or general graphs.
- Run-time generation or reconstruction of the zero-lookup binary tables (the kernels will strictly load precomputed binary tables).
- Automatic fallback compilation from CUDA to CPU when no NVIDIA GPU is present (the kernels will raise a runtime error directly).

## Documentation Impact
- No new user-facing documentation is introduced.
- Existing documentation is not materially amended.
- No derived documentation is invalidated.
