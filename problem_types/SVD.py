import sys
import os
sys.path.append(os.path.abspath('.'))

import sketching_methods.sketching as sketching
import numpy as np
import scipy
import scipy.linalg

def rank_approximation_exact(A, rank):
    """
    Compute the rank-rank approximation of a matrix A.

    Parameters:
    A (numpy.ndarray): The input matrix A.
    rank (int): The rank of the approximation.

    Returns:
    numpy.ndarray: The rank-rank approximation of A.
    """
    U, s, V = np.linalg.svd(A, full_matrices=True)
    return U[:, :rank] @ np.diag(s[:rank]) @ V[:rank, :]

def rank_approximation_sketch(A, rank, k, sketching_matrix_function):
    """
    Compute the rank-rank approximation of a matrix A using a sketched version of A.

    Parameters:
    A (numpy.ndarray): The input matrix A.
    rank (int): The rank of the approximation.
    k (int): The number of rows to sample from the transformed matrix.
    sketching_matrix_function (function): The sketching matrix function to use.

    Returns:
    numpy.ndarray: The rank-rank approximation of A.
    """
    F = sketching_matrix_function(k, A.shape[0])
    A_sketched = F @ A
    return rank_approximation_exact(A_sketched, rank)

def compute_epsilon(A, k, rank, sketching_matrix_function):
    A_r_sketch_norm = np.linalg.norm(A - rank_approximation_sketch(A, rank, k, sketching_matrix_function))
    A_r_norm = np.linalg.norm(A - rank_approximation_exact(A, rank))

    return A_r_sketch_norm / A_r_norm - 1

if __name__ == "__main__":
    A = np.random.standard_normal((1000, 500))
    rank = 50
    k = 60
    print(compute_epsilon(A, k, rank, sketching.orthogonal_sketching_matrix))

