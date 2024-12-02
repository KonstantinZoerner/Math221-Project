import numpy as np
import math
from sklearn.datasets import make_classification
import time
from sklearn.metrics import pairwise_distances


def standard_svd(input_matrix):
    u,s,vh=np.linalg.svd(input_matrix,full_matrices=False)
    s_truncated=np.diag(s)
    return u@s_truncated@vh

#after we got QB decomposition, do SVD do get how well the approximation is, and how the singular value changes
def svd_computation_and_error(Q,B,svd_result):
    u,s,vh=np.linalg.svd(B,full_matrices=False)
    s_truncated=np.linalg(s)
    estimation=Q@u@s_truncated@vh
    distance=np.linalg.norm(estimation-svd_result)
    return distance, s_truncated