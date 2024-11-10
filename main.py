import numpy as np
import sketching_methods.sketching

A = np.random.standard_normal((100, 100))
A_sketched = sketching_methods.sketching.sketch_orthogonal(10, A)
print(A_sketched)