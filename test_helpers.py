import unittest
import numpy as np
from numpy.testing import assert_almost_equal
from utils.helpers import get_random_orth_matrix

class TestHelpers(unittest.TestCase):

    def test_get_random_orth_matrix(self):
        k, m = 5, 5
        orth_matrix = get_random_orth_matrix(k, m)
        
        # Check if the product of the matrix and its transpose is the identity matrix
        identity_matrix = np.eye(k)
        product_matrix = np.dot(orth_matrix, orth_matrix.T)
        
        assert_almost_equal(product_matrix, identity_matrix, decimal=6)

if __name__ == '__main__':
    unittest.main()