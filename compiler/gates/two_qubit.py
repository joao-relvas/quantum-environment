import numpy as np
import math
from core.interfaces.gate import Gate
from typing import Tuple

# ─────────────────────────────
#              CX
# ─────────────────────────────
class CXGate(Gate):
    def __init__(self) -> None:
        super().__init__(
            name="CX",
            arity=2,
            matrix=np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
            ]),
            parameter_spec=(),
            is_composite=False,
        )

        
    def to_instructions(self, qubits, parameters, classical_condition=None):
        self.validate(qubits, parameters)
        return [
            {
                "op": "CX",
                "qubits": qubits,
                "condition": classical_condition
            }
        ]
        
# ─────────────────────────────
#            CZ
# ─────────────────────────────
class CZGate(Gate):
    def __init__(self) -> None:
        super().__init__(
            name = "CZ",
            arity = 2,
            matrix = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, -1]
            ]),
            parameter_spec = (),
            is_composite = False
        )
        
    def to_instructions(self, qubits, parameters, classical_condition=None):
        self.validate(qubits, parameters)
        return [
            {
                "op": "CZ",
                "qubits": qubits,
                "condition": classical_condition
            }
        ]
        
# ─────────────────────────────
#            SWAP
# ─────────────────────────────
class SWAPGate(Gate):
    def __init__(self) -> None:
        super().__init__(
            name = "SWAP",
            arity = 2,
            matrix = np.array([
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1]
            ]),
            parameter_spec = (),
            is_composite = False
        )
        
    def to_instructions(self, qubits, parameters, classical_condition=None):
        self.validate(qubits, parameters)
        return [
            {
                "op": "SWAP",
                "qubits": qubits,
                "condition": classical_condition
            }
        ]
        