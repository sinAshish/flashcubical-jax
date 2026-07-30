from typing import Protocol, Any
import numpy as np

class PHBackend(Protocol):
    def compute_ph(self, image: Any) -> np.ndarray:
        ...
    def name(self) -> str:
        ...
