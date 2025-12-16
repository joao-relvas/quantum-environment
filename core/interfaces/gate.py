from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
from core.linalg.matrices import is_unitary_matrix

class Gate(ABC):
    def __init__(
        self,
        *,
        name: str,
        arity: int,
        matrix: np.ndarray,
        parameter_spec: Tuple[str, ...],
        is_composite: bool
    ) -> None:
        self._name = name
        self._arity = arity
        self._matrix = matrix
        self._parameter_spec = parameter_spec or ()
        self._is_composite = is_composite
        self._is_unitary = True
        
        if not is_unitary_matrix(self._matrix):
            raise ValueError("Matrix should be unitary.")
        
    # ─────────────────────────────
    #            Getters
    # ─────────────────────────────
    
    def name(self):
        return self._name
    
    def arity(self):
        return self._arity
    
    def parameter_spec(self):
        return self._parameter_spec
    
    def is_composite(self):
        return self._is_composite
    
    def is_unitary(self):
        return self._is_unitary
    
    def matrix(self):
        return self._matrix
    
    # ─────────────────────────────
    #           Validation
    # ─────────────────────────────
    
    def validate(self, qubits: Tuple[int, ...], parameters):
        if (len(qubits) != self._arity) or (set(parameters.keys()) != set(self.parameter_spec())):
            raise ValueError
    
    # ─────────────────────────────
    #       Abstract Methods
    # ─────────────────────────────
    
    @abstractmethod    
    def to_instructions(self, qubits, parameters, classical_condition=None):
        pass
    
    
    
