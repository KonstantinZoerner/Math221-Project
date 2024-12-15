import sys
import os
sys.path.append(os.path.abspath('.'))

from sklearn.datasets import fetch_california_housing
import numpy as np
import problem_types.QR as QR

def load_california_housing():
    data = fetch_california_housing()
    A = data.data  # Features
    b = data.target  # Target variable
    return A, b


if __name__ == "__main__":
    A, b = load_california_housing()
    print(A.shape, b.shape)
    x = QR.solve_QR_exact(A, b)
    print(x, np.linalg.norm(A @ x - b))
