import numpy as np
import scipy
import scipy.linalg
import utils
from numpy.random import Generator as Generator
from scipy.sparse import csc_matrix

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

'''def sketch_hadamard(k, A):
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
    return transformed[np.random.choice(transformed.shape[0], k, replace=False)]'''


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

def rademacher_sketch_matrix(m,n):
    sketching_matrix=np.random.choice([-1,1],(m,n))
    return sketching_matrix
def check_random_state(seed):
    """Turn `seed` into a `np.random.RandomState` instance.

    Parameters
    ----------
    seed : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional
        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.

    Returns
    -------
    seed : {`numpy.random.Generator`, `numpy.random.RandomState`}
        Random number generator.

    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral | np.integer):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState | np.random.Generator):
        return seed

    raise ValueError(f"'{seed}' cannot be used to seed a numpy.random.RandomState"
                     " instance")

def rng_integers(gen, low, high=None, size=None, dtype='int64',
                 endpoint=False):
    """
    Return random integers from low (inclusive) to high (exclusive), or if
    endpoint=True, low (inclusive) to high (inclusive). Replaces
    `RandomState.randint` (with endpoint=False) and
    `RandomState.random_integers` (with endpoint=True).

    Return random integers from the "discrete uniform" distribution of the
    specified dtype. If high is None (the default), then results are from
    0 to low.

    Parameters
    ----------
    gen : {None, np.random.RandomState, np.random.Generator}
        Random number generator. If None, then the np.random.RandomState
        singleton is used.
    low : int or array-like of ints
        Lowest (signed) integers to be drawn from the distribution (unless
        high=None, in which case this parameter is 0 and this value is used
        for high).
    high : int or array-like of ints
        If provided, one above the largest (signed) integer to be drawn from
        the distribution (see above for behavior if high=None). If array-like,
        must contain integer values.
    size : array-like of ints, optional
        Output shape. If the given shape is, e.g., (m, n, k), then m * n * k
        samples are drawn. Default is None, in which case a single value is
        returned.
    dtype : {str, dtype}, optional
        Desired dtype of the result. All dtypes are determined by their name,
        i.e., 'int64', 'int', etc, so byteorder is not available and a specific
        precision may have different C types depending on the platform.
        The default value is 'int64'.
    endpoint : bool, optional
        If True, sample from the interval [low, high] instead of the default
        [low, high) Defaults to False.

    Returns
    -------
    out: int or ndarray of ints
        size-shaped array of random integers from the appropriate distribution,
        or a single such random int if size not provided.
    """
    if isinstance(gen, Generator):
        return gen.integers(low, high=high, size=size, dtype=dtype,
                            endpoint=endpoint)
    else:
        if gen is None:
            # default is RandomState singleton used by np.random.
            gen = np.random.mtrand._rand
        if endpoint:
            # inclusive of endpoint
            # remember that low and high can be arrays, so don't modify in
            # place
            if high is None:
                return gen.randint(low + 1, size=size, dtype=dtype)
            if high is not None:
                return gen.randint(low, high=high + 1, size=size, dtype=dtype)

        # exclusive
        return gen.randint(low, high=high, size=size, dtype=dtype)
def cwt_sketch_matrix(m,n,rng):
    rng = check_random_state(rng)
    rows = rng_integers(rng, 0, m, n)
    cols = np.arange(n+1)
    signs = rng.choice([1, -1], n)
    S = csc_matrix((signs, rows, cols), shape=(m, n))
    return S

