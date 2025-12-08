import math
from math import sqrt
import numpy as np

# Math Constants
PI = math.pi
E = math.e

# Quantum Gates
X = np.array(
    [[0, 1], 
     [1, 0]])
Y = np.array(
    [[0, -1j], 
     [1j, 0]])
Z = np.array(
    [[1, 0], 
     [0, -1]])
H = (1/math.sqrt(2)) * np.array([[1, 1], 
                                [1, -1]])
CX = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
])

G = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, -1]
])

CNOT_0_2 = np.array([
    [1,0,0,0,0,0,0,0],
    [0,1,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0],
    [0,0,0,1,0,0,0,0],
    [0,0,0,0,0,0,1,0],
    [0,0,0,0,0,0,0,1],
    [0,0,0,0,1,0,0,0],
    [0,0,0,0,0,1,0,0]
], dtype=complex)

I2 = np.identity(2)
