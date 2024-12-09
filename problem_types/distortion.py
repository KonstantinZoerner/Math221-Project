import sys
import os
sys.path.append(os.path.abspath('.'))

import sketching_methods.sketching as sketching
import numpy as np
import scipy
import scipy.linalg

def compute_distortion(A, sketched_A):
    ATA = (A.T @ A)
    AsTAs = sketched_A.T @ sketched_A
    eigvals, eigvecs = scipy.linalg.eigh(ATA)
    ATA_1_2 = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

    distortion_matrix = np.eye(A.shape[1]) - ATA_1_2 @ AsTAs @ ATA_1_2
    distortion = np.linalg.norm(distortion_matrix, ord=2)
    return distortion
