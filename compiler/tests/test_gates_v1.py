import numpy as np
import pytest

from compiler.gates.single_qubit import XGate, HGate
from compiler.gates.two_qubit import CXGate
from compiler.gates.composite import CompositeGate
from core.interfaces.gate import Gate


# ─────────────────────────────
# Primitive gate creation
# ─────────────────────────────

def test_x_gate_creation():
    g = XGate()
    assert isinstance(g, Gate)
    assert g.name() == "X"
    assert g.arity() == 1
    assert g.is_unitary() is True
    assert g.is_composite() is False


def test_h_gate_creation():
    g = HGate()
    assert g.name() == "H"
    assert g.arity() == 1
    assert g.is_unitary() is True
    assert g.is_composite() is False


def test_cx_gate_creation():
    g = CXGate()
    assert g.name() == "CX"
    assert g.arity() == 2
    assert g.is_unitary() is True
    assert g.is_composite() is False


# ─────────────────────────────
# Unitarity enforcement
# ─────────────────────────────

def test_non_unitary_gate_is_rejected():
    bad_matrix = np.array([[1, 1],
                           [1, 1]])  # NOT unitary

    class BadGate(Gate):
        def __init__(self):
            super().__init__(
                name="BAD",
                arity=1,
                matrix=bad_matrix,
                parameter_spec=(),
                is_composite=False,
            )

        def to_instructions(self, qubits, parameters, classical_condition=None):
            return []

    with pytest.raises(ValueError):
        BadGate()


# ─────────────────────────────
# Composite gate behavior
# ─────────────────────────────

def test_composite_gate_creation():
    g = CompositeGate((HGate(), XGate(), HGate()))

    assert isinstance(g, Gate)
    assert g.name() == "CMP"
    assert g.arity() == 1
    assert g.is_unitary() is True
    assert g.is_composite() is True


# ─────────────────────────────
# Composition math correctness
# ─────────────────────────────

def test_h_x_h_equals_z():
    h = HGate()
    x = XGate()
    composite = CompositeGate((h, x, h))

    z_matrix = np.array([
        [1,  0],
        [0, -1]
    ])

    assert np.allclose(composite.matrix(), z_matrix, atol=1e-10)


# ─────────────────────────────
# Instruction expansion
# ─────────────────────────────

def test_composite_expands_instructions():
    h = HGate()
    x = XGate()
    composite = CompositeGate((h, x, h))

    instr = composite.to_instructions(
        qubits=(0,),
        parameters={}
    )

    assert isinstance(instr, list)
    assert len(instr) == 3

    assert instr[0]["op"] == "H"
    assert instr[1]["op"] == "X"
    assert instr[2]["op"] == "H"
