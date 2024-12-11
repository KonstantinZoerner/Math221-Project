import sys
import os
sys.path.append(os.path.abspath('.'))

import sketching_methods.sketching as sketching
import numpy as np
import scipy
import scipy.linalg
import matplotlib.pyplot as plt

ratio = []
loops = 1000
for i in range(loops):
    m = 256
    n = 200
    k = 50
    print(i)
    A = np.random.standard_normal((m, n))

    As = sketching.sketch_uniform(k, A)
    x = np.random.standard_normal(n)

    Ax_norm2 = np.linalg.norm(A @ x)**2
    Asx_norm2 = np.linalg.norm(As @ x)**2
    
    ratio.append(Asx_norm2 / Ax_norm2)

plt.plot(ratio)
plt.show()

print(np.mean(ratio))

