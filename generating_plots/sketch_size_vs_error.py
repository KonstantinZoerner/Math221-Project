import sys
import os
sys.path.append(os.path.abspath('.'))

import numpy as np
import problem_types.QR as QR
import sketching_methods.sketching as sketching
import matplotlib.pyplot as plt


np.random.seed(0)


def compute_sketch_size_vs_error(m = 1000, n = 50, k_range = [60, 80, 100, 125, 250, 500], loops = 20, sketching_matrix_function = sketching.orthogonal_sketching_matrix):
    results = np.zeros(len(k_range))
    for i in range(loops):
        for k_index, k in enumerate(k_range):
            A = np.random.standard_normal((m, n))
            b = np.random.standard_normal(m)

            sol_exact = QR.solve_QR_exact(A, b)
            sol_sketched = QR.solve_QR_sketch(A, b, k, sketching_matrix_function)

            difference = np.linalg.norm(sol_exact - sol_sketched)
            results[k_index] += difference
    results /= loops
    return results

def plot_sketch_size_vs_error(m = 1000, n = 50, k_range = [60, 80, 100, 125, 250, 500], loops = 5):
    results_orth = compute_sketch_size_vs_error(m, n, k_range, loops, sketching.orthogonal_sketching_matrix)
    results_gaussian = compute_sketch_size_vs_error(m, n, k_range, loops, sketching.gaussian_sketching_matrix)
    #results_hadamard = compute_sketch_size_vs_error(m, n, k_range, loops, sketching.hadamard_sketch_matrix)
    results_uniform = compute_sketch_size_vs_error(m, n, k_range, loops, sketching.uniform_sketching_matrix)
    #results_SRTT = compute_sketch_size_vs_error(m, n, k_range, loops, sketching.sketch_SRTT)
    plt.plot(k_range, results_orth, label="Orthogonal")
    plt.plot(k_range, results_gaussian, label="Gaussian")
    #plt.plot(k_range, results_hadamard, label="Hadamard")
    plt.plot(k_range, results_uniform, label="Uniform")
    #plt.plot(k_range, results_SRTT, label="SRTT")
    plt.legend()

    plt.xlabel("Sketch size k")
    plt.ylabel("Error in 2-Norm")
    plt.title("Sketch size vs Error (m = 1000, n = 50)")
    plt.show()

if __name__ == "__main__":
    k_range = range(60, 500, 20)
    plot_sketch_size_vs_error(k_range = k_range)