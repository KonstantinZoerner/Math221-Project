from SVD import gaussian_sketching_matrix,uniform_sketching_matrix
import numpy as np
def eigendecomposition(A):

    eigenvalues, eigenvectors = np.linalg.eig(A)
    # Diagonal matrix of eigenvalues
    D = np.diag(eigenvalues)

# Reconstruct the matrix
    A_reconstructed = eigenvectors @ D @ np.linalg.inv(eigenvectors)

    return A_reconstructed

def hermitian_matrix_generation(m):
    A=np.random.rand(m,m)
    symmetric_matrix=(A+A.T)/2
    return symmetric_matrix
    
