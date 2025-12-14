import numpy as np
import random

# ==== RANDOM UTILITIES ====

# Default RNG Wrapper
rng = np.random.default_rng()

# Seed manager
def set_seed(seed: int):
    global rng
    rng = np.random.default_rng(seed)
    
# Random sampling unity
def sample_prob(probs, size=1):
    return rng.choice(len(probs), p=probs, size=size)

# Measure state from state vector
def measure_state(state, shots=1):
    probs = np.abs(state)**2
    return sample_prob(probs, size=shots)

# Random state generator
def random_state(n_qubits):
    dim = 2**n_qubits
    vec = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    vec /= np.linalg.norm(vec)
    return vec