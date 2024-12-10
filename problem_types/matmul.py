import sys
import os
sys.path.append(os.path.abspath('.'))

import numpy as np
import sketching_methods.sketching as sketching


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
    As = F @ A
    Bs = F @ B
    return As.T @ Bs

def matmul_sketch_sketching_function(A, B, k, sketching_function):
    assert A.shape[0] == B.shape[0]
    As = sketching_function(k, A)
    Bs = sketching_function(k, B)
    return As.T @ Bs

if __name__ == "__main__":
    np.random.seed(0)
    m = 1000
    n = 500
    k = 500
    A = np.random.standard_normal((m, n))
    B = np.random.standard_normal((m, n))
    C_exact = matmul_exact(A, B)
    C_sketch_sketching_matrix = matmul_sketch_sketching_matrix(A, B, k, sketching.orthogonal_sketching_matrix)
    C_sketch_sketching_function = matmul_sketch_sketching_function(A, B, k, sketching.sketch_orthogonal)
    difference_matrix = np.linalg.norm(C_exact - C_sketch_sketching_matrix, ord='fro')
    print("difference_matrix", difference_matrix)
    difference_function = np.linalg.norm(C_exact - C_sketch_sketching_function, ord='fro')
    print("difference_function", difference_function)
    C_norm = np.linalg.norm(C_exact, ord='fro')
    print("C_norm", C_norm)

    
