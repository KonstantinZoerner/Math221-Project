import numpy as np
import scipy

import sys
import os
sys.path.append(os.path.abspath('.'))
import utilities.helpers as helpers

def generate_hilbert(shape):
    m, n = shape
    if n >= m:
        A = scipy.linalg.hilbert(n)    
    else:
        A = scipy.linalg.hilbert(m)
    return A[:m, :n]

def generate_spread_singular_values(shape):
    m, n = shape
    U = helpers.get_random_orth_matrix(m, min(m, n))
    S = np.diag(np.logspace(0, -10, min(m, n)))  # Eigenvalues range from 1 to 1e-10
    V = helpers.get_random_orth_matrix(min(m, n), n)
    return U @ S @ V

def generate_spread_singular_values_with_noise(shape, noise_level=1e-3):
    A = generate_spread_singular_values(shape)
    A += np.random.standard_normal(shape) * noise_level
    return A

def generate_multicollinerarity(shape):
    m, n = shape
    base = np.random.rand(m)
    A = np.column_stack([base + 1e-6 * np.random.rand(m) for _ in range(n)])
    return A



