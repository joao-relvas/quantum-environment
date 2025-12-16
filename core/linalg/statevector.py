import numpy as np
import math

from constants import I2
from matrices import kron_all


# ==== PROBABILITIES AND STATE EVOLUTION UTILITIES ====

# State probabilities
def state_probs(state_vec: np.array):
    A = np.asarray(state_vec, dtype=complex).ravel()
    return (np.abs(A) ** 2).tolist()

# State value
def calc_state(state_vec: np.array):
    return max(state_probs(state_vec))

# Apply gate to state vector
def apply_gate(gate: np.array, vec: np.array):
    A = np.asarray(vec, dtype=complex).ravel()
    return (gate @ A).reshape(-1, 1) # Returns the vector in a Column form

# Apply multi-qubit gate to multi-qubit system
def build_adj_operator(n_qubits: int, gate: np.array, k_adj_targets: int):
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

def invert_order(new_order: np.array):
    n = len(new_order)
    inv = [0] * n
    for new_pos, old_pos in enumerate(new_order):
        inv[old_pos] = new_pos
    return inv


def apply_gate_multi_qubit(gate: np.array, state: np.array, targets: np.array):
    n_qubits = int(np.log2(state.size))
    k = int(np.log2(gate.shape[0]))
    
    new_order = permute_targets(state, targets)
    state_perm = permute_state(state, new_order)
    
    U = build_adj_operator(n_qubits, gate, k)
    gate_applied = U @ state_perm
    old_order = invert_order(new_order)
    
    return permute_state(gate_applied, old_order)