import numpy as np
import scipy

def get_random_orth_matrix(k, m):
    """ Get random orthogonal matrix via QR decomposition 
    of a gaussian iid matrix of size k x m"""

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