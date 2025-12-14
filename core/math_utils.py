import numpy as np
from constants import I2, CX
import scipy.linalg as spla
from functools import reduce
import math

# ==== VECTOR OPERATIONS ====

# Vector Module
def get_norm(vector):
    return np.linalg.norm(vector, ord=None)

# Vector Normalization
def normalize_vector(vector):
    vec_norm = get_norm(vector)
    if vec_norm != 0:
        return vector/vec_norm
    return np.zeros_like(vector)

# Inner Product
def inner_product(vec_a, vec_b):
    return np.vdot(vec_a, vec_b)

# Outer Product
def outer_product(vec_a, vec_b):
    return np.outer(vec_a, vec_b)

# Complex Conjugation
def complex_conj(vec):
    return np.conjugate(vec, None)

# ==== MATRIX OPERATIONS ====

# Identity Matrix Generator
def identity(n):
    return np.identity(n)

# Tensor / Kronecker Product
def kron_product(A, B):
    return np.kron(A, B)

def kron_all(factors):
    return reduce(kron_product, factors)

# Matrix Transpose
def transpose_matrix(matrix):
    return np.transpose(matrix, None)

# Conjugate Transpose
def conj_transpose(matrix):
    return np.conjugate(matrix).T

# Matrix exponentiation
def matrix_exp(matrix):
    return spla.expm(matrix)

# ==== MATRIX VERIFICATIONS ====

# Unitarity verification
def is_unitary(matrix, atol=1e-12):
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return False
    I = identity(matrix.shape[0])
    product = conj_transpose(matrix) @ matrix
    return np.allclose(product, I, atol=atol)

# Hermitian verification
def is_hermitian(matrix, atol=1e-12):
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return False
    return np.allclose(matrix, conj_transpose(matrix), atol=atol)

# Positive semi-definite verification
def is_positive_semi(matrix, atol=1e-12):
    if not is_hermitian(matrix):
        return False
    eigenvalues = np.linalg.eigvalsh(matrix)
    return np.all(eigenvalues >= -atol)

# ==== PROBABILITIES AND STATE EVOLUTION UTILITIES ====

# State probabilities
def state_probs(state_vec):
    A = np.asarray(state_vec, dtype=complex).ravel()
    return (np.abs(A) ** 2).tolist()

# State value
def calc_state(state_vec):
    return max(state_probs(state_vec))

# Apply gate to state vector
def apply_gate(gate, vec):
    A = np.asarray(vec, dtype=complex).ravel()
    return (gate @ A).reshape(-1, 1) # Returns the vector in a Column form

# Apply multi-qubit gate to multi-qubit system
def build_adj_operator(n_qubits, gate, k_adj_targets):
    factors = []
    factors.append(gate)
    for i in range(n_qubits - k_adj_targets):
        factors.append(I2)
    return kron_all(factors)

def permute_targets(state, targets):
    N = int(np.log2(state.size))
    return targets + [i for i in range(N) if i not in targets]

def permute_state(state, new_order):
    N = state.size
    n_qubits = int(math.log2(N))
    state_new = np.zeros_like(state)

    for i in range(N):
        # --- pegar bits antigos ---
        # bits_old[q] = bit do qubit q no índice i
        bits_old = [(i >> (n_qubits - 1 - q)) & 1 for q in range(n_qubits)]

        # --- rearranjar bits segundo new_order ---
        bits_new = [bits_old[new_order[k]] for k in range(n_qubits)]

        # --- converter bits_new para um novo índice j ---
        j = 0
        for q in range(n_qubits):
            j |= (bits_new[q] << (n_qubits - 1 - q))


        # --- mover amplitude ---
        state_new[j] = state[i]

    return state_new

def invert_order(new_order):
    n = len(new_order)
    inv = [0] * n
    for new_pos, old_pos in enumerate(new_order):
        inv[old_pos] = new_pos
    return inv


def apply_gate_multi_qubit(gate, state, targets):
    n_qubits = int(np.log2(state.size))
    k = int(np.log2(gate.shape[0]))
    
    new_order = permute_targets(state, targets)
    state_perm = permute_state(state, new_order)
    
    U = build_adj_operator(n_qubits, gate, k)
    gate_applied = U @ state_perm
    old_order = invert_order(new_order)
    
    return permute_state(gate_applied, old_order)
    
state = np.array([1, 2, 3, 4, 5, 6, 7, 8])
targets = [2, 0]
print(apply_gate_multi_qubit(CX, state, targets))
        