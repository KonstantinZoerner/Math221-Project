# Evaluation function
import numpy as np
from numpy.linalg import svd, norm
from numpy.linalg import qr
import matplotlib.pyplot as plt
from sketching_methods.sketching import JLT_sketching_matrix, orthogonal_sketching_matrix,gaussian_sketching_matrix,uniform_sketching_matrix,rademacher_sketch_matrix,SRFT_real_sketch_matrix,cwt_sketch_matrix,SRFT_complex_sketch_matrix,hadamard_sketch_matrix,sparse_sign_embedding_sketch_matrix
def randomized_svd(A, sketching_matrix_func,n_iter,sketch_size):
    
    # Step 1: Random sampling
    A=A.astype(np.float64)
    m, n = A.shape
    if sketching_matrix_func==SRFT_complex_sketch_matrix or sketching_matrix_func==SRFT_real_sketch_matrix or sketching_matrix_func==hadamard_sketch_matrix:
        random_matrix=sketching_matrix_func(sketch_size,n).T
    else:
        random_matrix=sketching_matrix_func(n,sketch_size)
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

def randomized_svd_left(A,k,sketching_matrix_func,sketch_size,n_iter):
    m = A.shape[0]
    S = sketching_matrix_func(sketch_size,m).T  # Adjust sketch size
    Y=S
    for _ in range(n_iter):
        Y = A@A.T@Y
    Q, _ = np.linalg.qr(Y)  # Orthonormal basis
    B = Q.T @ A
    U_tilde, Sigma, Vt = svd(B, full_matrices=False)
    U = Q @ U_tilde
    return U[:, :k], Sigma[:k], Vt[:k, :]
def evaluate_sketching(A, sketching_matrix_func,n_iter,sketch_size):
    U, Sigma, Vt = randomized_svd(A,sketching_matrix_func,n_iter,sketch_size)
    A_approx = U@np.diag(Sigma) @ Vt  # Reconstruction
    U,S,V=svd(A,full_matrices=False)
    approx_svd=U[:,:sketch_size]@np.diag(S[:sketch_size])@V[:sketch_size,:]
    error = norm(approx_svd - A_approx, 'fro') / norm(A, 'fro')
    return error

# Main script
if __name__ == "__main__":
    # Generate a synthetic matrix
    np.random.seed(42)
    m, n, k = 512,1024, 300
    A = np.random.randn(m, n)
    #Generate
    # Sketching methods
    n_iter=4
    methods = {
       "Orthogonal": orthogonal_sketching_matrix, 
        "Gaussian": gaussian_sketching_matrix,
        "Uniform": uniform_sketching_matrix,
        "Rademacher": rademacher_sketch_matrix,
        #"SRFT": SRFT_complex_sketch_matrix,
        #"SRTT": SRFT_real_sketch_matrix,
        "CWT": cwt_sketch_matrix,
        #"Hadamard": hadamard_sketch_matrix, 
        'JLT':JLT_sketching_matrix,
        'SSE':sparse_sign_embedding_sketch_matrix
    }
  
    sketch_sizes = range(100, 700, 10)
    results = {name: [] for name in methods.keys()}
    results['svd']=[]
    for sketch_size in sketch_sizes:
        for name, func in methods.items():
            error = evaluate_sketching(A, func,n_iter,sketch_size)
            results[name].append(error)
        
        '''rror_svd=norm(approx_svd-A,'fro')/norm(A,'fro')
        results['svd'].append(error_svd)'''
            #print(name)
    # Plot results
    plt.figure(figsize=(10, 6))
    for name, errors in results.items():
        plt.plot(sketch_sizes, errors, label=name)
    plt.yscale("log")
    plt.xlabel("Sketch Size")
    plt.ylabel("Relative Error")
    plt.title("Performance of Sketching Methods on m1")
    plt.legend()
    plt.grid()
    plt.savefig('m1_low_rank_2.pdf')

    #Check the singular values
    print('A1',np.linalg.cond(A))

    from generate_test_matrices.genrate_A import generate_hilbert,generate_multicollinerarity,generate_spread_singular_values,generate_spread_singular_values_with_noise
    A_2=generate_spread_singular_values_with_noise((512,1024))
  
    sketch_sizes = range(100, 700, 10)
    results = {name: [] for name in methods.keys()}
    results['svd']=[]
    for sketch_size in sketch_sizes:
        for name, func in methods.items():
            error = evaluate_sketching(A_2, func,n_iter,sketch_size)
            results[name].append(error)
        #U,S,V=svd(A_2,full_matrices=False)
        #approx_svd=U[:,:sketch_size]@np.diag(S[:sketch_size])@V[:sketch_size,:]
        #error_svd=norm(approx_svd-A_2,'fro')/norm(A_2,'fro')
        #results['svd'].append(error_svd)
            #print(name)
    # Plot results
    plt.figure(figsize=(10, 6))
    for name, errors in results.items():
        plt.plot(sketch_sizes, errors, label=name)
    plt.yscale("log")
    plt.xlabel("Sketch Size")
    plt.ylabel("Relative Error")
    plt.title("Performance of Sketching Methods on m2")
    plt.legend()
    plt.grid()
    plt.savefig('m2_low_rank_2.pdf')
    print('A2',np.linalg.cond(A_2))

    A_3=generate_hilbert((512,1024))
    n_iter=2
  
    
    sketch_sizes = range(100, 700, 10)
    results = {name: [] for name in methods.keys()}
    results['svd']=[]
    for sketch_size in sketch_sizes:
        for name, func in methods.items():
            error = evaluate_sketching(A_3, func,n_iter,sketch_size)
            results[name].append(error)
    ''' U,S,V=svd(A_3,full_matrices=False)
        approx_svd=U[:,:sketch_size]@np.diag(S[:sketch_size])@V[:sketch_size,:]
        error_svd=norm(approx_svd-A_3,'fro')/norm(A_3,'fro')
        results['svd'].append(error_svd)'''
            #print(name)
    # Plot results
    plt.figure(figsize=(10, 6))
    for name, errors in results.items():
        plt.plot(sketch_sizes, errors, label=name)
    plt.yscale("log")
    plt.xlabel("Sketch Size")
    plt.ylabel("Relative Error")
    plt.title("Performance of Sketching Methods on m3")
    plt.legend()
    plt.grid()
    plt.savefig('m3_low_rank_2.pdf')
    print('A3',np.linalg.cond(A_3))
    A_4=generate_multicollinerarity((512,1024))
    n_iter=2
  
    sketch_sizes = range(100, 700, 10)
    results = {name: [] for name in methods.keys()}
    results['svd']=[]
    for sketch_size in sketch_sizes:
        for name, func in methods.items():
            error = evaluate_sketching(A_4, func,n_iter,sketch_size)
            results[name].append(error)
            #print(name)
        '''U,S,V=svd(A_4,full_matrices=False)
        approx_svd=U[:,:sketch_size]@np.diag(S[:sketch_size])@V[:sketch_size,:]
        error_svd=norm(approx_svd-A_4,'fro')/norm(A_4,'fro')'''
        '''results['svd'].append(error_svd)'''
    # Plot results
    plt.figure(figsize=(10, 6))
    for name, errors in results.items():
        plt.plot(sketch_sizes, errors, label=name)
    plt.yscale("log")
    plt.xlabel("Sketch Size")
    plt.ylabel("Relative Error")
    plt.title("Performance of Sketching Methods on m4")
    plt.legend()
    plt.grid()
    plt.savefig('m4_low_rank_2.pdf')

    print('A4',np.linalg.cond(A_4))
    A_5=generate_spread_singular_values((512,1024))
    n_iter=2
  
    sketch_sizes = range(100, 700, 10)
    results = {name: [] for name in methods.keys()}
    results['svd']=[]
    for sketch_size in sketch_sizes:
        for name, func in methods.items():
            error = evaluate_sketching(A_5, func,n_iter,sketch_size)
            results[name].append(error)
        '''U,S,V=svd(A_5,full_matrices=False)
        approx_svd=U[:,:sketch_size]@np.diag(S[:sketch_size])@V[:sketch_size,:]
        error_svd=norm(approx_svd-A_5,'fro')/norm(A_5,'fro')
        results['svd'].append(error_svd)'''
            #print(name)
    # Plot results
    plt.figure(figsize=(10, 6))
    for name, errors in results.items():
        plt.plot(sketch_sizes, errors, label=name)
    plt.yscale("log")
    plt.xlabel("Sketch Size")
    plt.ylabel("Relative Error")
    plt.title("Performance of Sketching Methods on m5")
    plt.legend()
    plt.grid()
    plt.savefig('m5_low_rank_2.pdf')
    print('A5',np.linalg.cond(A_5))
