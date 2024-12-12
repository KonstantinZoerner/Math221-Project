import numpy as np
from scipy.sparse import random
from scipy.linalg import svd

# Generate random matrices A and B
m, n, p, r = 1000, 500, 200, 100  # Original and sketching dimensions
A = np.random.rand(m, n)
B = np.random.rand(n, p)

# Step 1: Create a random sketching matrix S
S = np.random.randn(r, n) / np.sqrt(r)  # Gaussian sketch (rows normalized)

# Step 2: Compute the sketched versions of A and B
A_sketched = A @ S.T  # Project A to a lower-dimensional space
B_sketched = S @ B    # Project B to a lower-dimensional space

# Step 3: Perform approximate multiplication
C_approx = A_sketched @ B_sketched  # Approximation of A @ B

# Compare with true multiplication
C_exact = A @ B
error = np.linalg.norm(C_exact - C_approx, ord="fro") / np.linalg.norm(C_exact, ord="fro")

print(f"Approximation Error: {error:.4f}")