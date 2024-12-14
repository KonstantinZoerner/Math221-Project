import numpy as np
import scipy

def generate_hilbert(shape):
    m, n = shape
    if n >= m:
        A = scipy.linalg.hilbert(n)    
    else:
        A = scipy.linalg.hilbert(m)
    return A[:m, :n]
