import numpy as np
from scipy.linalg import svd
from sklearn.utils.extmath import randomized_svd

def cur_decomposition(A, c, r, rank=None):
    """
    Perform CUR decomposition on a matrix A.
    
    Parameters:
        A (ndarray): Input matrix of shape (m, n).
        c (int): Number of columns to sample.
        r (int): Number of rows to sample.
        rank (int, optional): Rank for approximation. Default is min(c, r).
    
    Returns:
        C (ndarray): Matrix of selected columns.
        U (ndarray): Core matrix.
        R (ndarray): Matrix of selected rows.
    """
    # Step 1: Column Sampling
    col_norms = np.linalg.norm(A, axis=0)**2
    col_probs = col_norms / np.sum(col_norms)
    col_indices = np.random.choice(A.shape[1], size=c, replace=False, p=col_probs)
    C = A[:, col_indices] / np.sqrt(c * col_probs[col_indices])
    
    # Step 2: Row Sampling
    row_norms = np.linalg.norm(A, axis=1)**2
    row_probs = row_norms / np.sum(row_norms)
    row_indices = np.random.choice(A.shape[0], size=r, replace=False, p=row_probs)
    R = A[row_indices, :] / np.sqrt(r * row_probs[row_indices])    
    
    # Step 3: Core Matrix U
    W = C[row_indices, :]  # Intersection of selected rows and columns
    if rank is None:
        rank = min(c, r)
    
    U, S, VT = svd(W, full_matrices=False)
    S_inv = np.diag(1 / S[:rank])  # Inverse of singular values (truncate to `rank`)
    U = VT[:rank, :].T @ S_inv @ U[:, :rank].T  # Core matrix U
    
    return C, U, R

# Example usage
A = np.random.rand(6, 5)  # Example matrix
c = 3  # Number of columns to sample
r = 3  # Number of rows to sample

C, U, R = cur_decomposition(A, c, r)

# Verify decomposition
A_approx = C @ U @ R
print("Original Matrix A:\n", A)
print("Approximated Matrix A_approx:\n", A_approx)
