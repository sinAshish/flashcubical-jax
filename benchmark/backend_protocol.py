from typing import Protocol, Any

class PHBackend(Protocol):
    def compute_ph(self, image: Any) -> Any:
        ...
    def name(self) -> str:
        ...

