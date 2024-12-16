import numpy as np
import scipy

def SRFT_real_sketch_matrix(k, m):
    """ compare Chapter 2.5 of Randomized Linear Algebra"""
    D = np.diag(np.random.choice([-1, 1], m))
    F = np.sqrt(1/m)*np.fft.fft(np.eye(m))
    #F=np.fft.fft(D)
    R = np.zeros((k, m))
    selected_rows = np.random.choice(m, k, replace=False)
    for i, row in enumerate(selected_rows):
        R[i, row] = 1
    print("R:", R)
    print("F:", F)
    print("D:", D)
    print("selected_rows:", selected_rows)

    return np.sqrt(m/k)*R @ F @ D
    #return R @ F @ D

def sparse_sign_embedding_sketch_matrix(k, m, zeta=8):
    """
    Create a sparse sign embedding sketch matrix S of shape (k, m),
    where each column has exactly 'zeta' non-zero entries with random signs.
    """
    # Initialize row indices, column indices, and data (values)
    row_indices = np.repeat(np.arange(m), zeta)  # Each column has 'zeta' entries
    col_indices = np.concatenate([np.random.choice(k, zeta, replace=False) for _ in range(m)])
    data = np.random.choice([-1, 1], size=m * zeta)

    # Create sparse matrix in COO format
    S = scipy.sparse.coo_matrix((data, (col_indices, row_indices)), shape=(k, m))

    # Scale the matrix
    return S.todense() * (1 / np.sqrt(zeta))

if __name__ == "__main__":
    k = 3
    m = 6
    #print(SRFT_real_sketch_matrix(k, m))
    print(sparse_sign_embedding_sketch_matrix(k, m, zeta=2))