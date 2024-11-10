import numpy as np
import scipy

def get_random_orth_matrix(k, m):
    """
    Generate a random orthogonal matrix using QR decomposition.

    This function generates a random orthogonal matrix by performing QR decomposition 
    on a Gaussian iid matrix of size k x m. Depending on the dimensions, the function 
    ensures that the resulting matrix has orthogonal rows or columns.

    Parameters:
    k (int): Number of rows of the resulting orthogonal matrix.
    m (int): Number of columns of the resulting orthogonal matrix.

    Returns:
    numpy.ndarray: A k x m orthogonal matrix.
    """
    

    if k < m: # orthogonal rows
        transpose = True
        A = np.random.standard_normal((m, k))
    else: # orthogonal colums
        transpose = False
        A = np.random.standard_normal((k, m))  
    Q, _ = scipy.linalg.qr(A)
    if transpose:
        return Q[:m , :k].T
    return Q[:k , :m]

