import numpy as np
from core.interfaces.gate import Gate
from typing import List

def comp_gates(gates: List[Gate]) -> np.array:
    n_matrices = len(gates)
    gates = tuple(reversed(gates))
    result = gates[0].matrix()
    for i in range(n_matrices - 1):
        result = np.matmul(result, gates[i + 1].matrix())
    return result

def get_all_params(gates):
    params = []
    for gate in gates:
        for param in gate.parameter_spec():
            if param not in params:
                params.append(param)
    return params