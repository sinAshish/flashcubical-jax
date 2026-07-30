import numpy as np
from benchmark.backend_protocol import PHBackend
import flash_cubical as fc
from typing import Any

class FlashCubicalBackend(PHBackend):
    def prepare(self, image: Any) -> Any:
        return np.ascontiguousarray(np.asarray(image, dtype=np.float64))

    def sync(self):
        pass

    def compute_ph(self, arr: Any) -> np.ndarray:
        ph = fc.compute(arr)
        return ph.values

    def name(self) -> str:
        return "FlashCubical (C++)"
