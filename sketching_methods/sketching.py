import numpy as np
import scipy
import utils
#import utils.helpers
from skylark.sketch import JLT,CWT
#import two kinds of sketching matrices
sketches={ "JLT" : JLT, "CWT" : CWT }
#这个JLT是什么
def sketch_SRTT(k, p,n):
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
    for sname in sketches:
        stype=sketches[sname]
        S=stype(n,k+p,defouttype="SharedMatrix")
        '''m = A.shape[0]
        assert k < m
        angle = np.random.uniform(0, 2*np.pi, m)
        D = np.diag(np.exp(angle * 1j))
        FFT = np.fft.fft(D @ A)
        return FFT[np.random.choice(FFT.shape[0], k, replace=False)]'''
        return S
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

import numpy as np
from sklearn.random_projection import johnson_lindenstrauss_min_dim
from sklearn.random_projection import GaussianRandomProjection
def johnson_lindenstrauss_transform(data, n_components):
    """
    Apply the Johnson-Lindenstrauss Transform using Gaussian Random Projection.
    
    Parameters:
    - data: np.ndarray
        High-dimensional data of shape (n_samples, n_features).
    - n_components: int
        The target dimensionality for the transformation.

    Returns:
    - transformed_data: np.ndarray
        The data in the lower-dimensional space of shape (n_samples, n_components).
    """
    # Validate input dimensions
    if n_components > data.shape[1]:
        raise ValueError("n_components must be less than or equal to the original number of features.")
    
    # Initialize Gaussian Random Projection
    transformer = GaussianRandomProjection(n_components=n_components)
    
    # Transform the data
    transformed_data = transformer.fit_transform(data)
    return transformed_data

# Example usage
if __name__ == "__main__":
    # Generate random high-dimensional data
    np.random.seed(42)
    high_dim_data = np.random.rand(1000, 500)  # 1000 samples, 500 features
    
    # Calculate the minimum number of components to approximately preserve distances
    eps = 0.1  # Error tolerance
    min_components = johnson_lindenstrauss_min_dim(n_samples=high_dim_data.shape[0], eps=eps)
    print(f"Minimum components to preserve distances (eps={eps}): {min_components}")
    
    # Apply the Johnson-Lindenstrauss Transform
    low_dim_data = johnson_lindenstrauss_transform(high_dim_data, n_components=min_components)
    print(f"Original shape: {high_dim_data.shape}, Transformed shape: {low_dim_data.shape}")
