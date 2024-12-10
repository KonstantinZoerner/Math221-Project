import sys
import os
sys.path.append(os.path.abspath('.'))

import numpy as np
import problem_types.QR as QR
import sketching_methods.sketching as sketching
import matplotlib.pyplot as plt
import utils.helpers as helpers
import problem_types.distortion as distortion


np.random.seed(0)


def compute_sketch_size_vs_distortion(m = 1000, n = 50, k_range = [60, 80, 100, 125, 250, 500], loops = 1, sketching_function = sketching.sketch_orthogonal):
    results = np.zeros(len(k_range))
    for i in range(loops):
        for k_index, k in enumerate(k_range):
            A = np.random.standard_normal((m, n))
            As = sketching_function(k, A)
            dist = distortion.compute_distortion(A, As)
            results[k_index] += dist
    results /= loops
    return results

def plot_sketch_size_vs_distortion(m = 1024, n = 50, k_range = [60, 80, 100, 125, 250, 500], loops = 5):
    results_orth = compute_sketch_size_vs_distortion(m, n, k_range, loops, sketching.sketch_orthogonal)
    #results_gaussian = compute_sketch_size_vs_distortion(m, n, k_range, loops, sketching.sketch_gaussian)
    #results_hadamard = compute_sketch_size_vs_distortion(m, n, k_range, loops, sketching.sketch_hadamard)
    #results_uniform = compute_sketch_size_vs_distortion(m, n, k_range, loops, sketching.sketch_uniform)
    results_SRTT = compute_sketch_size_vs_distortion(m, n, k_range, loops, sketching.sketch_SRTT)
    plt.plot(k_range, results_orth, label="Orthogonal")
    #plt.plot(k_range, results_gaussian, label="Gaussian")
    #plt.plot(k_range, results_hadamard, label="Hadamard")
    #plt.plot(k_range, results_uniform, label="Uniform")
    plt.plot(k_range, results_SRTT, label="SRTT")
    plt.legend()

    plt.xlabel("Sketch size k")
    plt.ylabel("Distortion")
    plt.title(f"Sketch size vs distortion (m = {m}, n = {n})")

    helpers.save_plot("sketch_size_vs_distortion")
    plt.show()

if __name__ == "__main__":
    k_range = range(60, 500, 20)
    plot_sketch_size_vs_distortion(k_range = k_range)