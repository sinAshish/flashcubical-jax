import numpy as np
from typing import Any
from benchmark.backend_protocol import PHBackend

class TritonBackend(PHBackend):
    def compute_ph(self, image: Any) -> np.ndarray:
        # Basic scaffold.
        import flash_cubical as fc
        arr = np.ascontiguousarray(np.asarray(image, dtype=np.float64))
        return fc.compute(arr).values

    def name(self) -> str:
        return "Triton (Mocked for now)"
