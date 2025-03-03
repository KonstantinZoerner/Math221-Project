import numpy as np
from scipy.linalg import svd
from numpy.linalg import qr

def randomized_svd_sketched_right_multiplication(Y,A,k,n_iter):

    '''
    get the randomized svd using sketched matrices

    Parameters:
    Y: sketched matrices
    A: input matrices
    k: truncated rank
    n_iter: the iteration times
    '''
    # Step 1: Random sampling
    A=A.astype(np.float64)
    m, n = A.shape
   
    

    # Step 2: Power iteration (optional, improves accuracy for ill-conditioned matrices)
    for _ in range(n_iter):
        Y = A @ (A.T @ Y)

    # Step 3: QR decomposition
    Q, _ = qr(Y, mode='reduced')
    

    # Step 4: Project matrix to lower-dimensional space
    B = Q.T @ A

    # Step 5: Compute SVD on the smaller matrix
    U_tilde, S, Vt = svd(B, full_matrices=False)

    # Step 6: Reconstruct the left singular vectors
    U = Q @ U_tilde

    return U[:, :k], S[:k], Vt[:k, :]

def randomized_svd_left_multiplication(A,k,random_matrix,p=5,n_iter=2):
    A=A.astype(np.float64)
    m,n=A.shape

    Y=A.T@random_matrix@A

    for _ in range(n_iter):
        Y=A.T@(A@Y)

    Q,_=qr(Y,mode='reduced')

    B=Q.T@A

    U_tilde, S, Vt = svd(B, full_matrices=False)

    # Step 6: Reconstruct the left singular vectors
    U = Q @ U_tilde

    return U[:, :k], S[:k], Vt[:k, :]




    return
def randomized_svd_right_multiplication(A, k, random_matrix,p=5, n_iter=2):
    """
    Perform Randomized SVD on matrix A from right multiplication
    
    Parameters:
        A: ndarray, shape (m, n)
           Input matrix to decompose.
        k: int
           Target rank (number of singular values/vectors to compute).
        p: int, optional (default=5)
           Oversampling parameter to improve approximation accuracy.
        n_iter: int, optional (default=2)
           Number of iterations for power iteration (improves accuracy).
           
    Returns:
        U: ndarray, shape (m, k)
           Approximated left singular vectors.
        S: ndarray, shape (k,)
           Approximated singular values.
        V: ndarray, shape (k, n)
           Approximated right singular vectors.
    """
    # Step 1: Random sampling
    A=A.astype(np.float64)
    m, n = A.shape
   
    Y = A @ random_matrix

    # Step 2: Power iteration (optional, improves accuracy for ill-conditioned matrices)
    for _ in range(n_iter):
        Y = A @ (A.T @ Y)

    # Step 3: QR decomposition
    Q, _ = qr(Y, mode='reduced')
    

    # Step 4: Project matrix to lower-dimensional space
    B = Q.T @ A

    # Step 5: Compute SVD on the smaller matrix
    U_tilde, S, Vt = svd(B, full_matrices=False)

    # Step 6: Reconstruct the left singular vectors
    U = Q @ U_tilde

    return U[:, :k], S[:k], Vt[:k, :]