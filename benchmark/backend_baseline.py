import numpy as np
from benchmark.backend_protocol import PHBackend
import flash_cubical as fc
from typing import Any

class FlashCubicalBackend(PHBackend):
    def compute_ph(self, image: Any) -> np.ndarray:
        arr = np.ascontiguousarray(np.asarray(image, dtype=np.float64))
        ph = fc.compute(arr)
        return ph.values
    def name(self) -> str:
        return "FlashCubical (C++)"
