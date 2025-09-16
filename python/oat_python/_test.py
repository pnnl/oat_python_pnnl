


import unittest
from . import dissimilarity
from . import matrix




class TestCase(unittest.TestCase):
    def test_dissimilarity(self):
        dissimilarity.test_dissimilarity_matrix(max_grid_size=3)

    def test_minmax_(self):
        matrix.test_minmax()

    def test_validate_square_and_symmetric_(self):
        matrix.test_validate_square_and_symmetric_matrix()


    def test_validate_dissimilarity_matrix_(self):
        dissimilarity.test_validate_dissimilarity_matrix()




if __name__ == '__main__':
    unittest.main()        


