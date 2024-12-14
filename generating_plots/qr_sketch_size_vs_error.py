import sys
import os
sys.path.append(os.path.abspath('.'))

import numpy as np
import problem_types.QR as QR
import sketching_methods.sketching as sketching
import matplotlib.pyplot as plt
import utilities.helpers as helpers
import generate_test_matrices.genrate_A as genrate_A


np.random.seed(0)


def compute_sketch_size_vs_error(m = 1000, n = 50, k_range = [60, 80, 100, 125, 250, 500], loops = 20,\
                                 sketching_matrix_function = sketching.orthogonal_sketching_matrix,\
                                    A_generator=np.random.standard_normal, SVD=False):
    results = np.zeros(len(k_range))
    for i in range(loops):
        print("Itteration:", i)
        for k_index, k in enumerate(k_range):
            A = A_generator((m, n))
            b = np.random.standard_normal(m)

            if SVD:
                sol_exact = QR.solve_least_squares_exact_SVD(A, b)
                sol_sketched = QR.solve_least_squares_sketch_SVD(A, b, k, sketching_matrix_function)
            else:
                sol_exact = QR.solve_QR_exact(A, b)
                sol_sketched = QR.solve_QR_sketch(A, b, k, sketching_matrix_function)

            min_exact = np.linalg.norm(A @ sol_exact - b)
            min_sketched = np.linalg.norm(A @ sol_sketched - b)

            # rel_difference = np.linalg.norm(sol_exact - sol_sketched)/np.linalg.norm(sol_exact)
            #print(rel_difference)
            #results[k_index] += rel_difference
            rel_error = (min_sketched - min_exact) / min_exact
            results[k_index] += rel_error
    results /= loops
    return results

def plot_sketch_size_vs_error(m = 1024, n = 50, k_range = [60, 80, 100, 125, 250, 500], loops = 20,\
                                sketching_matrix_function_dict = sketching.sketching_matricies_dict,\
                                A_generator=np.random.standard_normal, title = "sketch_size_vs_error", SVD=False):
    for key in sketching_matrix_function_dict:
        print(key)
        results = compute_sketch_size_vs_error(m, n, k_range, loops, sketching_matrix_function_dict[key], A_generator=A_generator, SVD=SVD)
        plt.plot(k_range, results, label=key)
    plt.legend()
    plt.rcParams['text.usetex'] = True

    plt.xlabel("Sketch size k")
    plt.ylabel("Relative Error in 2-Norm")
    plt.title(r"Least squares for $A \in \mathbb{R}^{1024 \times 50}, b \in \mathbb{R}^{1024}$")

    helpers.save_plot(f"{title}_loops{loops}_m{m}_n{n}")
    plt.show()

def plot_sketch_size_vs_error_hilbert(loops = 20):
    m = 256
    n = 20
    k_range = range(40, 100, 5)
    sketching_matrix_functions = sketching_matricies_dict = {"Orthogonal": sketching.orthogonal_sketching_matrix, 
                            "Gaussian": sketching.gaussian_sketching_matrix,
                            "Uniform": sketching.uniform_sketching_matrix,
                            "Rademacher": sketching.rademacher_sketch_matrix,
                            "SRFT (real)": sketching.SRFT_real_sketch_matrix,
                            "SRFT (complex)": sketching.SRFT_complex_sketch_matrix,
                            "Hadamard": sketching.hadamard_sketch_matrix, 
                            #"CWT": sketching.cwt_sketch_matrix,
                            "SSE": sketching.sparse_sign_embedding_sketch_matrix}
    plot_sketch_size_vs_error(A_generator=genrate_A.generate_hilbert, sketching_matrix_function_dict=sketching_matrix_functions, title="qr_err_vs_k_hilbert", m=m, n=n, k_range=k_range, loops=loops, SVD=True)

if __name__ == "__main__":
    #k_range = range(100, 500, 20)
    # plot_sketch_size_vs_error(k_range = k_range)
    #plot_sketch_size_vs_error(A_generator=genrate_A.generate_hilbert)
    plot_sketch_size_vs_error_hilbert()