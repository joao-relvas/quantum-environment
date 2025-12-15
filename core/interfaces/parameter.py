from dataclasses import dataclass
from typing import Optional

@dataclass
class Parameter:
    name: str
    value: Optional[float] = None

    @property
    def is_symbolic(self) -> bool:
        return self.value is None