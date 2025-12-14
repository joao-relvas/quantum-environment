class QuantumPlatformError(Exception):
    """
    Base class for all custom errors in the Quantum Platform.
    Every module must raise subclasses of this type.
    """
    pass


# ===============================================================
#  GATE / CIRCUIT ERRORS
# ===============================================================

class InvalidGateError(QuantumPlatformError):
    """
    Raised when a gate is invalid:
    - non-unitary matrix
    - wrong matrix dimensions
    - unexpected dtype
    - malformed gate definition
    """
    pass


class InvalidCircuitError(QuantumPlatformError):
    """
    Raised when a circuit contains invalid elements:
    - gate applied to nonexistent qubit index
    - out-of-range qubit reference
    - undefined gate
    - structurally invalid circuit representation
    """
    pass


# ===============================================================
#  CONFIG / SETUP ERRORS
# ===============================================================

class ConfigError(QuantumPlatformError):
    """
    Raised when the platform configuration is invalid:
    - missing required fields
    - unsupported values
    - incompatible parameters
    """
    pass


# ===============================================================
#  NUMERICAL / MATH ERRORS
# ===============================================================

class MathError(QuantumPlatformError):
    """
    Raised for numerical instability:
    - non-finite values (NaN/inf)
    - floating point explosions
    - norm drift beyond tolerance
    - attempted inversion of a singular matrix
    """
    pass


# ===============================================================
#  SIMULATOR ERRORS
# ===============================================================

class SimulatorError(QuantumPlatformError):
    """
    Raised by the simulator backend:
    - tensor dimension mismatch
    - invalid state vector shape
    - attempt to simulate too many qubits unsupported by backend
    - stride-based gate application failure
    """
    pass


# ===============================================================
#  COMPILER ERRORS
# ===============================================================

class CompilerError(QuantumPlatformError):
    """
    Raised during compilation stages:
    - IR transformation failures
    - gate fusion inconsistencies
    - illegal optimization rewrites
    - pass ordering problems
    """
    pass
