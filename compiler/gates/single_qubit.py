import numpy as np
import math
import cmath
from core.interfaces.gate import Gate
from typing import Tuple
from core.constants import PI, E_I_PI4


# ─────────────────────────────
#              X
# ─────────────────────────────
class XGate(Gate):
    def __init__(self) -> None:
        super().__init__(
            name = "X",
            arity = 1,
            matrix = np.array([
                [0, 1],
                [1, 0]
            ]),
            parameter_spec = (),
            is_composite = False
        )
        
    def to_instructions(self, qubits, parameters, classical_condition=None):
        self.validate(qubits, parameters)
        return [
            {
                "op": "X",
                "qubits": qubits,
                "condition": classical_condition
            }
        ]
        
# ─────────────────────────────
#              Y
# ─────────────────────────────
class YGate(Gate):
    def __init__(self) -> None:
        super().__init__(
            name = "Y",
            arity = 1,
            matrix = np.array([
                [0, -1j], 
                [1j, 0 ]
            ]),
            parameter_spec = (),
            is_composite = True
        )
        
    def to_instructions(self, qubits, parameters, classical_condition=None):
        self.validate(qubits, parameters)
        return [
            {
                "op": "Y",
                "qubits": qubits,
                "condition": classical_condition
            }
        ]

# ─────────────────────────────
#              Z
# ─────────────────────────────
class ZGate(Gate):
    def __init__(self) -> None:
        super().__init__(
            name = "Z",
            arity = 1,
            matrix = np.array([
                [1, 0], 
                [0, -1]
            ]),
            parameter_spec = (),
            is_composite = False            
        )
        
    def to_instructions(self, qubits, parameters, classical_condition=None):
        self.validate(qubits, parameters)
        return [
            {
                "op": "Z",
                "qubits": qubits,
                "condition": classical_condition
            }
        ]

# ─────────────────────────────
#          Hadamard
# ─────────────────────────────
class HGate(Gate):
    def __init__(self):
        super().__init__(
            name = "H",
            arity = 1,
            matrix = (1/math.sqrt(2)) * np.array([  [1, 1], 
                                                    [1, -1] ]),
            parameter_spec = (),
            is_composite = False
        )
        
    def to_instructions(self, qubits, parameters, classical_condition=None):
        self.validate(qubits, parameters)
        return [
            {
                "op": "H",
                "qubits": qubits,
                "condition": classical_condition
            }
        ]

# ─────────────────────────────
#          Phase (S)
# ─────────────────────────────
class SGate(Gate):
    def __init__(self):
        super().__init__(
            name = "H",
            arity = 1,
            matrix = np.array([ 
                [1, 0], 
                [0, 1j]
            ]),
            parameter_spec = (),
            is_composite = True
        )
        
    def to_instructions(self, qubits, parameters, classical_condition=None):
        self.validate(qubits, parameters)
        return [
            {
                "op": "S",
                "qubits": qubits,
                "condition": classical_condition
            }
        ]
        
# ─────────────────────────────
#             T
# ─────────────────────────────
class TGate(Gate):
    def __init__(self):
        super().__init__(
            name = "T",
            arity = 1,
            matrix = np.array([  
                [1, 0], 
                [0, E_I_PI4]
            ]),
            parameter_spec = (),
            is_composite = True
        )
        
    def to_instructions(self, qubits, parameters, classical_condition=None):
        self.validate(qubits, parameters)
        return [
            {
                "op": "T",
                "qubits": qubits,
                "condition": classical_condition
            }
        ]
