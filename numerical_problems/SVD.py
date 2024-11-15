import numpy as np
import math
from sklearn.datasets import make_classification
import time
from sklearn.metrics import pairwise_distances

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



def matrix_generation(t,r,m,n):
    H = np.random.rand(m,n)
    u, s, vh = np.linalg.svd(H, full_matrices=False)
    mat_U = u @ vh
    H_2 = np.random.rand(n, n)
    u_2, s_2, vh_2 = np.linalg.svd(H_2, full_matrices=False)
    mat_V = u_2 @ vh_2
    mat_sigma=np.eye(n)
    for i in range(math.ceil(n/t)):
        for k in range(t):
            mat_sigma[i*t+k][i*t+k]=math.pow(r,i)
    return mat_U@mat_sigma@mat_V.T

def random_matrix_generation(m,n):
    X, _ = make_classification(n_samples=1000, n_features=500, random_state=42)

def evaluate_sketching(sketching_matrices,input_matrix):
    
    start_time = time.time()
    sketched_matrix=sketching_matrices@input_matrix
    sketching_time = time.time() - start_time
    U_a,S_a,V_a=np.linalg.svd(sketched_matrix)
    approx_matrix=U_a@S_a@V_a
    original_distances = pairwise_distances(input_matrix)
    projected_distances = pairwise_distances(approx_matrix)
    difference = np.abs(original_distances - projected_distances) / (original_distances + 1e-9)  # avoid div by zero
    return difference,sketching_time

