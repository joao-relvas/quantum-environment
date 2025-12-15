from enum import Enum, auto

class InstructionType(Enum):
    UNITARY = auto()
    MEASUREMENT = auto()
    RESET = auto()
    NOISE = auto()
    BARRIER = auto()