import numpy as np
from functools import reduce
import scipy.linalg as spla

# ==== MATRIX OPERATIONS ====

# Identity Matrix Generator
def identity(n: int):
    return np.identity(n)

# Tensor / Kronecker Product
def kron_product(A: np.array, B: np.array):
    return np.kron(A, B)

def kron_all(factors):
    return reduce(kron_product, factors)

# Matrix Transpose
def transpose_matrix(matrix: np.array):
    return np.transpose(matrix, None)

# Conjugate Transpose
def conj_transpose(matrix: np.array):
    return np.conjugate(matrix).T

# Matrix exponentiation
def matrix_exp(matrix: np.array):
    return spla.expm(matrix)

# ==== MATRIX VERIFICATIONS ====

# Unitarity verification
def is_unitary_matrix(matrix: np.array, atol=1e-12):
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return False
    I = identity(matrix.shape[0])
    product = conj_transpose(matrix) @ matrix
    return np.allclose(product, I, atol=atol)

# Hermitian verification
def is_hermitian(matrix: np.array, atol=1e-12):
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return False
    return np.allclose(matrix, conj_transpose(matrix), atol=atol)

# Positive semi-definite verification
def is_positive_semi(matrix: np.array, atol=1e-12):
    if not is_hermitian(matrix):
        return False
    eigenvalues = np.linalg.eigvalsh(matrix)
    return np.all(eigenvalues >= -atol)