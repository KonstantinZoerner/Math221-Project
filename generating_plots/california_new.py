import sys
import os
sys.path.append(os.path.abspath('.'))

import numpy as np
import problem_types.QR as QR
import sketching_methods.sketching as sketching
import matplotlib.pyplot as plt
import utilities.helpers as helpers
import generate_test_matrices.genrate_A as genrate_A
import generate_test_matrices.californian_housing_prices as california
np.random.seed(0)

def compute_true_result(A, b):
    x = QR.solve_QR_exact(A, b)
    return x

def compute_sketch_size_vs_error(A, b, sol_exact, k_range, loops = 20,\
                                sketching_matrix_function = sketching.orthogonal_sketching_matrix,\
                                SVD=False, compute_residual = True):
    results = np.zeros(len(k_range))
    for i in range(loops):
        print("Itteration:", i)
        for k_index, k in enumerate(k_range):
            if SVD:
                sol_sketched = QR.solve_least_squares_sketch_SVD(A, b, k, sketching_matrix_function)
            else:
                sol_sketched = QR.solve_QR_sketch(A, b, k, sketching_matrix_function)

            if compute_residual:
                min_exact = np.linalg.norm(A @ sol_exact - b)
                min_sketched = np.linalg.norm(A @ sol_sketched - b)
                rel_error = (min_sketched - min_exact) / min_exact
                results[k_index] += rel_error
            else:
                rel_difference = np.linalg.norm(sol_exact - sol_sketched)/np.linalg.norm(sol_exact)
                results[k_index] += rel_difference
                results[k_index] += rel_difference
            
    results /= loops
    return results


def plot_sketch_size_vs_error(A, b, sol_exact, k_range, loops = 20,\
                                sketching_matrix_function_dict = sketching.sketching_matricies_dict,\
                                A_generator=np.random.standard_normal, title = "sketch_size_vs_error", SVD=False,\
                                compute_residual = True):
    
    # compute the results
    for key in sketching_matrix_function_dict:
        print(key)
        results = compute_sketch_size_vs_error(A, b,sol_exact, k_range, loops, sketching_matrix_function_dict[key], SVD=SVD, compute_residual=compute_residual)
        plt.plot(k_range, results, label=key)
    
    # Asthetics
    plt.legend()
    plt.xlabel("Sketch size k")
    if compute_residual:
        plt.ylabel("Relative Error of residual in 2-Norm")
    else:
        plt.ylabel("Relative Difference of x in 2-Norm")
    plt.rcParams['text.usetex'] = True
    plt.title(r"Least squares for for Californian housing data")
    plt.rcParams['text.usetex'] = False

    # Save and show
    helpers.save_plot(f"{title}_loops{loops}_california")
    plt.show()

def plot_california(loops = 20, k_range = range(50, 1000, 50), compute_residual=True):
    sketching_matrix_functions = {"Orthogonal": sketching.orthogonal_sketching_matrix, 
                            "Gaussian": sketching.gaussian_sketching_matrix,
                            "Uniform": sketching.uniform_sketching_matrix,
                            "Rademacher": sketching.rademacher_sketch_matrix,
                            "SRFT (real)": sketching.SRFT_real_sketch_matrix,
                            "SRFT (complex)": sketching.SRFT_complex_sketch_matrix,
                            "Hadamard": sketching.hadamard_sketch_matrix, 
                            # "CWT": sketching.cwt_sketch_matrix,
                            "SSE": sketching.sparse_sign_embedding_sketch_matrix}
    
    A, b = california.load_california_housing()
    sol_exact = compute_true_result(A, b)
    plot_sketch_size_vs_error(A=A, b=b, sol_exact=sol_exact, \
                                sketching_matrix_function_dict=sketching_matrix_functions,\
                                title="qr_err_vs_k_california", k_range=k_range,\
                                loops=loops, SVD=False, compute_residual=compute_residual)

if __name__ == "__main__":
    plot_california(loops=20)