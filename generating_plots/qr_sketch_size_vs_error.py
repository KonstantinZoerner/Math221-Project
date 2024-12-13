import sys
import os
sys.path.append(os.path.abspath('.'))

import numpy as np
import problem_types.QR as QR
import sketching_methods.sketching as sketching
import matplotlib.pyplot as plt
import utilities.helpers as helpers


np.random.seed(0)


def compute_sketch_size_vs_error(m = 1000, n = 50, k_range = [60, 80, 100, 125, 250, 500], loops = 20, sketching_matrix_function = sketching.orthogonal_sketching_matrix):
    results = np.zeros(len(k_range))
    for i in range(loops):
        print("Itteration:", i)
        for k_index, k in enumerate(k_range):
            A = np.random.standard_normal((m, n))
            b = np.random.standard_normal(m)

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

def plot_sketch_size_vs_error(m = 1024, n = 50, k_range = [60, 80, 100, 125, 250, 500], loops = 20, sketching_matrix_function_dict = sketching.sketching_matricies_dict):
    for key in sketching_matrix_function_dict:
        print(key)
        results = compute_sketch_size_vs_error(m, n, k_range, loops, sketching_matrix_function_dict[key])
        plt.plot(k_range, results, label=key)
    plt.legend()
    plt.rcParams['text.usetex'] = True

    plt.xlabel("Sketch size k")
    plt.ylabel("Relative Error in 2-Norm")
    plt.title(r"Least squares for $A \in \mathbb{R}^{1024 \times 50}, b \in \mathbb{R}^{1024}$")

    helpers.save_plot("sketch_size_vs_error")
    plt.show()

if __name__ == "__main__":
    k_range = range(100, 500, 20)
    plot_sketch_size_vs_error(k_range = k_range)
    # plot_sketch_size_vs_error()