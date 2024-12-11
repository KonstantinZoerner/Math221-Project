import sys
import os
sys.path.append(os.path.abspath('.'))

import sketching_methods.sketching as sketching
import numpy as np
import scipy
import scipy.linalg
import matplotlib.pyplot as plt

ratio = []
loops = 200
for i in range(200):
    m = 400
    n = 200
    k = 50
    print(i)
    A = np.random.standard_normal((m, n))

    As = sketching.sketch_SRTT_2(k, A)
    x = np.random.standard_normal(n)

    Ax_norm2 = np.linalg.norm(A @ x)**2
    Asx_norm2 = np.linalg.norm(As @ x)**2
    
    ratio.append(Asx_norm2 / Ax_norm2)

plt.plot(ratio)
plt.show()

