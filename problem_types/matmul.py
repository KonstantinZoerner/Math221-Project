import sys
import os
sys.path.append(os.path.abspath('.'))

import numpy as np
import sketching_methods.sketching as sketching
import scipy


def matmul_exact(A, B):
    """
    Multiply two matrices exactly.

    Parameters:
    A (numpy.ndarray): The first input matrix.
    B (numpy.ndarray): The second input matrix.

    Returns:
    numpy.ndarray: The product A.T @ B of the two input matrices.
    """
    assert A.shape[0] == B.shape[0]
    return A.T @ B

def matmul_sketch_sketching_matrix(A, B, k, sketching_matrix_function):
    assert A.shape[0] == B.shape[0]
    F = sketching_matrix_function(k, A.shape[0])
    # F = np.random.randn(k, A.shape[0]) / np.sqrt(k)
    As = F @ A
    Bs = F @ B
    return As.T @ Bs

def matmul_sketch_sketching_function(A, B, k, sketching_function):
    assert A.shape[0] == B.shape[0]
    As = sketching_function(k, A)
    Bs = sketching_function(k, B)
    return As.T @ Bs

def compute_rel_error(k = 50, m = 1024, n = 500, p = 200, sketching_matrix_function = sketching.orthogonal_sketching_matrix):
    A = np.random.rand(n, m)
    B = np.random.rand(n, p)

    C_exact = matmul_exact(A, B)
    C_sketch_sketching_matrix = matmul_sketch_sketching_matrix(A, B, k, sketching_matrix_function)
    difference_matrix = np.linalg.norm(C_exact - C_sketch_sketching_matrix, ord='fro')
    C_norm = np.linalg.norm(C_exact, ord='fro')
    rel_error = difference_matrix / (C_norm*np.linalg.norm(A, ord='fro')* np.linalg.norm(B, ord='fro'))
    return rel_error






if __name__ == "__main__":
    np.random.seed(0)
    for k in range(10, 200, 10):
        print(compute_rel_error(sketching_matrix_function=sketching.orthogonal_sketching_matrix, k = k))




    
