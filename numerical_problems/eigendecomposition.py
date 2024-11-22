from SVD import gaussian_sketching_matrix,uniform_sketching_matrix
import numpy as np
def eigendecomposition(A):

    eigenvalues, eigenvectors = np.linalg.eig(A)
    # Diagonal matrix of eigenvalues
    D = np.diag(eigenvalues)

# Reconstruct the matrix
    A_reconstructed = eigenvectors @ D @ np.linalg.inv(eigenvectors)

    return A_reconstructed

#Low-rank eigendecomposition under QB algorithms
def randomized_eigendecomposition(Q,B):
    C=B@Q
    eig_value,eig_vector=np.linalg.eig(C)
    V=Q@eig_vector
    return V,eig_value
    
