import numpy as np
import scipy
import matplotlib.pyplot as plt
import time

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

def save_plot(name):
    """
    Save a plot to a file.

    Parameters:
    name (str): The name of the file.
    path (str): The path to save the file.
    """
    name = name + str(int(time.time()%100000000)) + ".pdf"
    path = "figures/"
    print("Saving plot to", path + name)
    plt.savefig(path + name, bbox_inches='tight')

if __name__ == "__main__":
    print("(10, 100)", get_random_orth_matrix(10, 100).shape)
    print("(100, 10)", get_random_orth_matrix(100, 10).shape)
    A = get_random_orth_matrix(2, 5)
    B = get_random_orth_matrix(5, 2)
    print(A @ A.T)
    print(B.T @ B)