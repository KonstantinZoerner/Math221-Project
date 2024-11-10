import numpy as np
import scipy
import utils
import utils.helpers

def sketch_SRTT(k, A):
    """
    Perform a Subsampled Randomized Trigonometric Transform (SRTT) sketch of the matrix A.

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
    angle = np.random.uniform(0, 2*np.pi, m)
    D = np.diag(np.exp(angle * 1j))
    FFT = np.fft.fft(D @ A)
    return FFT[np.random.choice(FFT.shape[0], k, replace=False)]

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