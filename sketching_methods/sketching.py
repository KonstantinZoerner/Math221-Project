import numpy as np
import scipy
import utils

#这个JLT是什么
#def sketch_SRTT(k, p,n):
"""
    Perform a Subsampled Randomized Trigonometric Transform (SRTT) sketch of the matrix A.

    Parameters:
    k (int): The number of rows to sample from the transformed matrix.
    A (numpy.ndarray): The input matrix to be sketched.

    Returns:
    numpy.ndarray: A k x n matrix which is a sketch of the input matrix A.
    
    Raises:
    AssertionError: If k is not less than the number of rows in A."""
    #for sname in sketches:
     #   stype=sketches[sname]
      #  S=stype(n,k+p,defouttype="SharedMatrix")
       # '''m = A.shape[0]
        #assert k < m
        #angle = np.random.uniform(0, 2*np.pi, m)
        #D = np.diag(np.exp(angle * 1j))
        #FFT = np.fft.fft(D @ A)
        #return FFT[np.random.choice(FFT.shape[0], k, replace=False)]'''
        #return S
#return relevant matrices that we need here

def sketch_hadamard(k, A):
    """
    Perform a Hadamard transform sketch of the matrix A.

    Parameters:
    k (int): The number of rows to sample from the transformed matrix.
    A (numpy.ndarray): The input matrix to be sketched.

    Returns:
    numpy.ndarray: A k x n matrix which is a sketch of the input matrix A.
    
    Raises:
    AssertionError: If the number of rows in A is not a power of 2.
    """
    
    m = A.shape[0]
    assert np.log2(m) - np.floor(np.log2(m)) == 0
    H = scipy.linalg.hadamard(m)
    angle = np.random.uniform(0, 2*np.pi, m)
    D = np.diag(np.exp(angle * 1j))
    transformed = H @ D @ A
    return transformed[np.random.choice(transformed.shape[0], k, replace=False)]

def sketch_orthogonal(k, A):
    """
    Perform an orthogonal transform sketch of the matrix A.

    Parameters:
    k (int): The number of rows to sample from the transformed matrix.
    A (numpy.ndarray): The input matrix to be sketched.

    Returns:
    numpy.ndarray: A k x n matrix which is a sketch of the input matrix A.
    
    Raises:
    AssertionError: If k is not less than the number of rows in A.
    """
    
    m = A.shape[0]
    assert k < m
    F = utils.helpers.get_random_orth_matrix(k, m)
    return np.sqrt(m/k)*F @ A

def gaussian_sketching_matrix(m, n):
    """
    Generates a Gaussian sketching matrix of dimensions m x n.
    
    Parameters:
    m (int): Number of rows (target dimensionality).
    n (int): Number of columns (original dimensionality).
    
    Returns:
    numpy.ndarray: Gaussian sketching matrix of shape (m, n).
    """
    # Create an m x n matrix with entries sampled from a standard normal distribution
    sketching_matrix = np.random.normal(0, 1, (m, n))
    return sketching_matrix


def uniform_sketching_matrix(m, n, low=-1, high=1):
    """
    Generates a uniform sketching matrix of dimensions m x n.
    
    Parameters:
    m (int): Number of rows (target dimensionality).
    n (int): Number of columns (original dimensionality).
    low (float): Lower bound of the uniform distribution.
    high (float): Upper bound of the uniform distribution.
    
    Returns:
    numpy.ndarray: Uniform sketching matrix of shape (m, n).
    """
    # Create an m x n matrix with entries sampled from a uniform distribution in [low, high]
    sketching_matrix = np.random.uniform(low, high, (m, n))
    return sketching_matrix

