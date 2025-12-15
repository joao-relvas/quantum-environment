from abc import ABC, abstractmethod
from parameter import Parameter
from typing import Tuple, Mapping, Optional
from enums import InstructionType
from classical import ClassicalCondition


class Instruction(ABC):

    def __init__(
        self,
        *,
        instruction_type: InstructionType,
        opcode: str,
        qubits: Tuple[int, ...],
        parameters: Optional[Mapping[str, Parameter]] = None,
        classical_outputs: Tuple[int, ...] = (),
        classical_condition: Optional[ClassicalCondition] = None,
        duration: Optional[float] = None,
    ) -> None:
        self._instruction_type = instruction_type
        self._opcode = opcode
        self._qubits = qubits
        self._parameters = parameters or {}
        self._classical_outputs = classical_outputs
        self._classical_condition = classical_condition
        self._duration = duration

    # ─────────────────────────────
    # Identity & semantics
    # ─────────────────────────────

    @property
    def instruction_type(self) -> InstructionType:
        return self._instruction_type

    @property
    def opcode(self) -> str:
        return self._opcode

    @property
    def is_unitary(self) -> bool:
        return self._instruction_type is InstructionType.UNITARY

    @property
    def is_reversible(self) -> bool:
        return self._instruction_type is InstructionType.UNITARY

    # ─────────────────────────────
    # Quantum operands
    # ─────────────────────────────

    @property
    def qubits(self) -> Tuple[int, ...]:
        return self._qubits

    @abstractmethod
    def touched_qubits(self) -> Tuple[int, ...]:
        """
        Returns all qubits read or written by this instruction.
        Used for dependency analysis and scheduling.
        """
        pass

    # ─────────────────────────────
    # Parameters
    # ─────────────────────────────

    @property
    def parameters(self) -> Mapping[str, Parameter]:
        return self._parameters

    @property
    def is_parameterized(self) -> bool:
        return bool(self._parameters)

    @property
    def has_unbound_parameters(self) -> bool:
        return any(p.is_symbolic for p in self._parameters.values())

    # ─────────────────────────────
    # Classical interaction
    # ─────────────────────────────

    @property
    def classical_outputs(self) -> Tuple[int, ...]:
        return self._classical_outputs

    @property
    def classical_condition(self) -> Optional[ClassicalCondition]:
        return self._classical_condition

    # ─────────────────────────────
    # Scheduling / dependency metadata
    # ─────────────────────────────

    @property
    def duration(self) -> Optional[float]:
        """
        Abstract duration of the instruction.
        May be None if unknown or backend-defined.
        """
        return self._duration

    def conflicts_with(self, other: "Instruction") -> bool:
        """
        Returns True if this instruction cannot be scheduled
        in parallel with another instruction.
        """

        # Quantum resource conflict
        if set(self.touched_qubits()) & set(other.touched_qubits()):
            return True

        # Classical dependency conflict
        if (
            self.classical_outputs
            and other.classical_condition
            and other.classical_condition.bit in self.classical_outputs
        ):
            return True

        return False

    # ─────────────────────────────
    # Equality & hashing (structural)
    # ─────────────────────────────

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Instruction):
            return NotImplemented

        return (
            self.instruction_type == other.instruction_type
            and self.opcode == other.opcode
            and self.qubits == other.qubits
            and self.parameters == other.parameters
            and self.classical_outputs == other.classical_outputs
            and self.classical_condition == other.classical_condition
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.instruction_type,
                self.opcode,
                self.qubits,
                frozenset(self.parameters.items()),
                self.classical_outputs,
                self.classical_condition,
            )
        )

    

    
    