import numpy as np

# ==== VECTOR OPERATIONS ====

# Vector Module
def get_norm(vector: np.array):
    return np.linalg.norm(vector, ord=None)

# Vector Normalization
def normalize_vector(vector: np.array):
    vec_norm = get_norm(vector)
    if vec_norm != 0:
        return vector/vec_norm
    return np.zeros_like(vector)

# Inner Product
def inner_product(vec_a: np.array, vec_b: np.array):
    return np.vdot(vec_a, vec_b)

# Outer Product
def outer_product(vec_a: np.array, vec_b: np.array):
    return np.outer(vec_a, vec_b)

# Complex Conjugation
def complex_conj(vec: np.array):
    return np.conjugate(vec, None)