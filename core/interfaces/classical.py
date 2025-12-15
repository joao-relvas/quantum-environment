from dataclasses import dataclass

@dataclass(frozen=True)
class ClassicalCondition:
    bit: int
    value: int  # usually 0 or 1