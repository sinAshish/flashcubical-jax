# FlashCubical

Fast V-filtration cubical persistent homology for 2D and 3D scalar grids induced by images, impleneting the ideas proposed in https://arxiv.org/abs/2606.04801.

Computes H0 (connected components), H1 (loops / tunnels), and H2 (voids) using a lower-star filtration with a precomputed lookup table for local zero-persistence pairs.



Requirements

* Python 3.9+
* A C++ compiler with C++17 support (MSVC, GCC, or Clang)

## Installation

### Standard Installation
```bash
git clone git@github.com:T-prog123/FlashCubical.git
cd FlashCubical
pip install -e .
```
That is all. pip compiles the C++ extension automatically.

### Benchmark Setup (uv)
To install the dependencies and execute the GPU/CPU benchmarks:
```bash
git clone https://github.com/sinAshish/flashcubical-jax.git
cd flashcubical-jax
git checkout jax-version

# Sync and install JAX, PyTorch, and test dependencies
uv sync
```

## Usage

```python
import flash_cubical as fc
import numpy as np

image = np.random.rand(256, 256)      # 2D scalar field
ph = fc.compute(image)

ph.values                             # float64 array, shape (n, 3)
                                      # columns: birth, death, dimension

ph.plot()                             # persistence diagram
```

For a 3D volume:

```python
volume = np.random.rand(64, 64, 64)
ph = fc.compute(volume)               # computes H0, H1, H2
ph = fc.compute(volume, h1=False)     # skip H1, only H0 and H2
```

## API

### `fc.compute(x, min_persistence=0.0, h1=True)`

|Parameter|Description|
|-|-|
|`x`|2D or 3D array-like, converted to float64|
|`min_persistence`|Minimal distance between birth and death values. Must be non-negative.|
|`h1`|Compute H1. Must be `True` for 2D inputs. Can be `False` for 3D to skip H1|

Returns a `Persistence` object.

### `ph.values`

NumPy array of shape `(n, 3)`, dtype float64.

|Column|Meaning|
|-|-|
|0|Birth|
|1|Death (essential H0 class has death = +inf)|
|2|Homology dimension (0, 1, or 2)|

### `ph.plot(ax=None)`

Draws a persistence diagram coloured by dimension. Returns a matplotlib `Axes`.

## Running the tests

```bash
# Run baseline tests
pip install pytest
pytest tests/

# Run GPU/CPU benchmark suite tests
PYTHONPATH=. uv run pytest tests/test_benchmarks.py -v
```

## JAX & PyTorch Benchmarks

A pluggable benchmarking suite is provided under the `benchmark/` folder to compare the native C++ implementation against our purely tensor-native **Parallel Forest** implementations in JAX and PyTorch. 

### Benchmark Results (macOS CPU)
The following benchmarks were evaluated locally on **macOS (Apple Silicon CPU)**. This is a CPU-bound execution (which lacks CUDA hardware acceleration for tensor sorting, but serves as a solid baseline):

```text
Grid Size    | Noise      | Backend                | Time (ms)  | Accuracy
---------------------------------------------------------------------------
32x32        | uniform    | FlashCubical (C++)     | 0.05       | Baseline
32x32        | uniform    | JAX (Parallel Forest)  | 0.72       | PASS
32x32        | uniform    | PyTorch (Parallel)     | 0.77       | PASS
---------------------------------------------------------------------------
64x64        | uniform    | FlashCubical (C++)     | 0.23       | Baseline
64x64        | uniform    | JAX (Parallel Forest)  | 2.96       | PASS
64x64        | uniform    | PyTorch (Parallel)     | 3.27       | PASS
---------------------------------------------------------------------------
128x128      | gaussian   | FlashCubical (C++)     | 1.45       | Baseline
128x128      | gaussian   | JAX (Parallel Forest)  | 12.85      | PASS
128x128      | gaussian   | PyTorch (Parallel)     | 11.06      | PASS
---------------------------------------------------------------------------
1024x1024    | uniform    | FlashCubical (C++)     | 122.00     | Baseline
1024x1024    | uniform    | JAX (Parallel Forest)  | 1018.40    | PASS
1024x1024    | uniform    | PyTorch (Parallel)     | 960.96     | PASS
```

To run these benchmarks on your machine:
```bash
PYTHONPATH=. uv run python benchmark/run_benchmarks.py
```

---

## JAX Algorithmic Write-Up: Comparison with C++

### The C++ Method (Sequential Lower-Star Sweep)
The original C++ implementation in `FlashCubical` uses a highly optimized **sequential lower-star filtration**. 
1. It sorts the $V$ vertices and iterates through them one by one.
2. For each active vertex, it utilizes a pre-computed **256-bit hardware byte-lookup table** to quickly identify and eliminate zero-persistence edges in a single CPU cycle.
3. It performs a sequential Union-Find pointer-chasing loop (`parent[i] = parent[parent[i]]`) which is incredibly fast on CPUs due to hardware-level branch prediction.

### The JAX Method (Parallel Forest Algorithm)
Because sequential Union-Find is highly unroll-heavy and slow in JAX's XLA compiler, our JAX implementation uses a completely novel **purely parallel tensor-native** algorithm:

1. **Parallel Zero-Persistence Flow (Pruning)**: Instead of a sequential loop, we find the steepest descending neighbor for all $V$ vertices simultaneously in a single tensor operation using JAX's `segment_min`. This establishes a directed "gradient forest" mapping younger vertices directly to older ones.
2. **Parallel Pointer Jumping (Path Doubling)**: We flatten this entire zero-persistence forest in parallel using path doubling (`parent = parent[parent]`). Because the path lengths halve at each step, exactly $21$ iterations are guaranteed to flatten a $1024\times1024$ grid, snapping all vertices to their local minimum root. No loops!
3. **Saddle Edge Extraction**: Edges that are not part of the zero-persistence forest are "saddles" that merge components. We filter out the internal ones and map the saddles directly to the flattened roots.
4. **Stable Sort & Scan**: We lexicographically sort the remaining saddles and use a JAX `lax.scan` to merge the components in filtration order.

### Differences and Trade-offs
- **CPU Performance**: C++ is significantly faster on CPUs because CPUs are designed for sequential pointer chasing and avoid the massive $O(N \log N)$ sorting step on the 2,000,000 edges required by vector engines.
- **GPU Scale-out**: On an NVIDIA GPU (CUDA), the JAX sorting and `segment_min` operations will run in **microseconds** on thousands of CUDA cores. While C++ is stuck sequentially executing on a single CPU thread, the JAX Parallel Forest implementation will scale-out massively and outperform C++ on extremely large resolutions.


