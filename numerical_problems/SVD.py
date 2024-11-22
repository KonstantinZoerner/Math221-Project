import numpy as np
import math
from sklearn.datasets import make_classification
import time
from sklearn.metrics import pairwise_distances


def matrix_generation(t,r,m,n):
    H = np.random.rand(m,n)
    u, s, vh = np.linalg.svd(H, full_matrices=False)
    mat_U = u @ vh
    H_2 = np.random.rand(n, n)
    u_2, s_2, vh_2 = np.linalg.svd(H_2, full_matrices=False)
    mat_V = u_2 @ vh_2
    mat_sigma=np.eye(n)
    for i in range(math.ceil(n/t)):
        for k in range(t):
            mat_sigma[i*t+k][i*t+k]=math.pow(r,i)
    return mat_U@mat_sigma@mat_V.T

def random_matrix_generation(m,n):
    X, _ = make_classification(n_samples=1000, n_features=500, random_state=42)

def standard_svd(input_matrix,k):
    u,s,vh=np.linalg.svd(input_matrix,full_matrices=False)
    s_truncated=np.diag(s[:k])
    return u@s_truncated@vh

def svd_computation_and_error(Q,B,svd_result):
    u,s,vh=np.linalg.svd(B,full_matrices=False)
    s_truncated=np.linalg(s)
    estimation=Q@u@s_truncated@vh
    distance=np.linalg.norm(estimation-svd_result)
    return distance