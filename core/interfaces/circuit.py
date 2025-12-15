from parameter import Parameter
from typing import Tuple, Mapping, Optional
from enums import InstructionType
from classical import ClassicalCondition

class Circuit:

    def __init__(self, *, num_qubits, num_clbits):
        self._num_qubits = num_qubits
        self._num_clbits = num_clbits
        self._instructions = []

    def _validate_instruction(self, instruction):
        # quantum wires
        for q in instruction.touched_qubits():
            if not (0 <= q < self._num_qubits):
                raise ValueError("Qubit index out of bounds")

        # classical outputs
        for c in instruction.classical_outputs:
            if not (0 <= c < self._num_clbits):
                raise ValueError("Classical bit index out of bounds")

        # classical condition
        if instruction.classical_condition:
            bit = instruction.classical_condition.bit
            if not (0 <= bit < self._num_clbits):
                raise ValueError("Classical condition bit out of bounds")

    def add_instruction(self, instruction):
        self._validate_instruction(instruction)
        self._instructions.append(instruction)

    def extend(self, instructions):
        for instr in instructions:
            self.add_instruction(instr)
