import sys
import os
sys.path.append(os.path.abspath('.'))

import sketching_methods.sketching as sketching
import numpy as np
import scipy
import scipy.linalg




def solve_QR_exact(A, b):
    """
    Solve the least squares problem min_x ||Ax - b||_2 using the QR decomposition.

    Parameters:
    A (numpy.ndarray): The input matrix A.
    b (numpy.ndarray): The input vector b.

    Returns:
    numpy.ndarray: The solution to the least squares problem.
    """
    
    Q, R = scipy.linalg.qr(A, mode='economic')
    return scipy.linalg.solve_triangular(R, Q.T @ b)

def solve_QR_sketch(A, b, k, sketching_matrix_function):
    """
    Solve the least squares problem min_x ||Ax - b||_2 using a sketched version of A.

    Parameters:
    A (numpy.ndarray): The input matrix A.
    b (numpy.ndarray): The input vector b.
    k (int): The number of rows to sample from the transformed matrix.
    sketching_matrix_function (function): The sketching matrix function to use.

    Returns:
    numpy.ndarray: The solution to the least squares problem.
    """
    F = sketching_matrix_function(k, A.shape[0])
    A_sketched = F @ A
    b_sketched = F @ b
    return solve_QR_exact(A_sketched, b_sketched)



if __name__ == "__main__":
    np.random.seed(0)
    m = 1000
    n = 50
    k = 60
    A = np.random.standard_normal((m, n))
    b = np.random.standard_normal(m)
    sol_exact = solve_QR_exact(A, b)
    sol_sketched = solve_QR_sketch(A, b, k, sketching.orthogonal_sketching_matrix)

    difference = np.linalg.norm(sol_exact - sol_sketched)
    print(difference)