
from typing import Tuple
from core.interfaces.gate import Gate
from compiler.gates.base import comp_gates, get_all_params

class CompositeGate(Gate):
    def __init__(self, gates: Tuple[Gate, ...]) -> None:
        super().__init__(
            name = "CMP",
            arity = gates[0].arity(),
            matrix = comp_gates(gates),
            parameter_spec = get_all_params(gates),
            is_composite = True
        )
        self._gates = gates
        
    def to_instructions(self, qubits, parameters, classical_condition=None):
        self.validate(qubits, parameters)

        instructions = []

        for gate in self._gates:
            instructions.extend(
                gate.to_instructions(
                    qubits=qubits,
                    parameters=parameters,
                    classical_condition=classical_condition
                )
            )

        return instructions
