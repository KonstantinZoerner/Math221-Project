# Evaluation function
import numpy as np
from numpy.linalg import svd, norm
from numpy.linalg import qr
import matplotlib.pyplot as plt
from sketching_methods.sketching import JLT_sketching_matrix, orthogonal_sketching_matrix,gaussian_sketching_matrix,uniform_sketching_matrix,rademacher_sketch_matrix,SRFT_sketch_matrix,cwt_sketch_matrix,SRTT_sketch_matrix,hadamard_sketch_matrix
def randomized_svd(A, k, sketching_matrix_func,n_iter,sketch_size):
     # Step 1: Random sampling
    n=A.shape[1]
    random_matrix=sketching_matrix_func(n,sketch_size)
    A=A.astype(np.float64)
    print(random_matrix.shape)
    Y = A @ random_matrix
    #I mainly consider right mutiplication
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

def evaluate_sketching(A, k, sketching_matrix_func,n_iter,sketch_size):
    U, Sigma, Vt = randomized_svd(A, k, sketching_matrix_func,n_iter,sketch_size)
    A_approx = (U * Sigma) @ Vt  # Reconstruction
    error = norm(A - A_approx, 'fro') / norm(A, 'fro')
    return error

# Main script
if __name__ == "__main__":
    # Generate a synthetic matrix
    np.random.seed(42)
    m, n, k = 500, 300, 10
    A = np.random.randn(m, n)

    # Sketching methods
    methods = {
       "Orthogonal": orthogonal_sketching_matrix, 
        "Gaussian": gaussian_sketching_matrix,
        "Uniform": uniform_sketching_matrix,
        "Rademacher": rademacher_sketch_matrix,
        #"SRFT": SRFT_sketch_matrix,
        #"SRTT": SRTT_sketch_matrix,
        "CWT": cwt_sketch_matrix,
        #"Hadamard": hadamard_sketch_matrix, 
        'JLT':JLT_sketching_matrix,
    }
    n_iter=2
    # Evaluate each method
    # Evaluate each method for different sketch sizes
    sketch_sizes = range(k + 5, 300, 10)
    results = {name: [] for name in methods.keys()}
    for sketch_size in sketch_sizes:
        for name, func in methods.items():
            error = evaluate_sketching(A, k, func,n_iter,sketch_size)
            results[name].append(error)

    # Plot results
    plt.figure(figsize=(10, 6))
    for name, errors in results.items():
        plt.plot(sketch_sizes, errors, label=name)
    
    plt.xlabel("Sketch Size")
    plt.ylabel("Relative Error")
    plt.title("Performance of Sketching Methods")
    plt.legend()
    plt.grid()
    plt.show()