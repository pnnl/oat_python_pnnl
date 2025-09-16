"""
Functions for dense and sparse matrices.
"""


from cmath import inf
import numpy as np
import sklearn
import sklearn.metrics
from . import point_cloud
import scipy
from sklearn.neighbors import radius_neighbors_graph
import itertools
import warnings



def triplets_for_csr( matrix, sorted=False ):
    """
    Returns all :term:`structural nonzero <sparse matrix>`
    triplets (i, j, value) for a given sciyp CSR matrix.

    This function is a wrapper around a simple block of code:

    .. code-block:: python

        points = matrix.tocoo()
        triplets = list(zip(points.row, points.col, points.data))

    Parameters
    ----------
    matrix : scipy.sparse.csr_matrix
        The input CSR matrix.
    sorted : bool, optional
        If True, the triplets are sorted by (i, j) before returning. Default is False.

    Returns
    -------
    triplets : list of tuples
        A list of tuples (i, j, value) representing the
        structural non-zero entries of the matrix.

        - If `sorted` is True, the triplets are sorted lexicographically.
        - If `matrix` has sorted column indices (within each row),
          then triplets will be in sorted order even if `sorted` is False.
          See the cautionary note on sorting in :term:`glossary <scipy sparse csr matrices>`
          for more details.

    Example
    -------
    .. code-block:: python

        import numpy as np
        import scipy.sparse
        from oat_python.matrix import triplets_for_csr

        # Start with a dense matrix
        dense = np.array([
            [1, 0, 0],
            [0, 2, 0],
            [0, 0, 3]
        ])
        print("Dense matrix:")
        print(dense)

        # Convert to CSR
        matrix = scipy.sparse.csr_matrix(dense)

        # Get triplets
        triplets = triplets_for_csr(matrix)
        print("Triplets:", triplets)
        # Output: [(0, 0, 1), (1, 1, 2), (2, 2, 3)]         
    """
    if not isinstance(matrix, scipy.sparse.csr_matrix):
        raise TypeError("Input must be a scipy.sparse.csr_matrix.")
    
    points = matrix.tocoo()
    triplets = list(zip(points.row, points.col, points.data))

    if sorted:
        triplets.sort()
    
    return triplets


def assert_symmetric_triplets_csr( csr_matrix, err_msg='' ):
    """
    Assert that the triplets of a CSR matrix are symmetric.

    This function checks that for every triplet (i, j, value) in the CSR matrix,
    there exists a corresponding triplet (j, i, value) in the same matrix.
    If the matrix is not symmetric, an AssertionError is raised.

    Parameters
    ----------
    csr_matrix : scipy.sparse.csr_matrix
        The CSR matrix to check for symmetry.
    err_msg : str, optional
        The error message to display in case of failure.

    Raises
    ------
    AssertionError
        If the CSR matrix is not symmetric.
    """
    triplets = triplets_for_csr(csr_matrix)
    triplet_set = set(triplets)
    
    for i, j, value in triplets:
        if (j, i, value) not in triplet_set:
            raise AssertionError(f"{err_msg}Matrix is not symmetric at ({i}, {j}) with value {value}.")



def assert_almost_equal_csr( a, b, decimal, err_msg='' ):
    """
    Assert that two CSR (Compressed Sparse Row) matrices are almost equal.

    This function checks that the input matrices `a` and `b` have the same sparsity pattern (i.e., identical `indptr` and `indices` arrays)
    and that their explicitly stored data entries are equal up to a specified decimal precision.

    Parameters
    ----------
    a : scipy.sparse.csr_matrix
        The first CSR matrix to compare.
    b : scipy.sparse.csr_matrix
        The second CSR matrix to compare.
    decimal : int
        The number of decimal places to which the data entries are compared.
    err_msg : str, optional
        The error message to display in case of failure.

    Raises
    ------
    AssertionError
        If the matrices do not have the same sparsity pattern or their data entries differ beyond the specified precision.
    """
    np.testing.assert_equal(a.indptr, b.indptr, err_msg=err_msg)
    np.testing.assert_equal(a.indices, b.indices, err_msg=err_msg)    
    np.testing.assert_almost_equal(a.data, b.data, decimal=decimal, err_msg=err_msg)




def validate_square_and_symmetric_matrix(matrix):
    """
    Validates that the input matrix is a square and symmetric matrix.

    This function checks that the input is either a NumPy ndarray or a SciPy CSR sparse matrix,
    that it is two-dimensional and square (number of rows equals number of columns),
    and that it is exactly symmetric (i.e., ``matrix[i, j] == matrix[j, i]`` for all ``i, j``).

    Parameters
    ----------
    matrix : numpy.ndarray or scipy.sparse.csr_matrix
        The matrix to validate.

    Raises
    ------
    TypeError
        If the input is not a NumPy ndarray or a SciPy CSR sparse matrix.
    Exception
        If the input is not two-dimensional, not square, or not exactly symmetric.
    """    
    if not (isinstance(matrix, np.ndarray) or isinstance(matrix, scipy.sparse.csr_matrix)):
        raise TypeError("Matrix must be a numpy ndarray or scipy.sparse.csr_matrix.")
    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        raise Exception("Matrix must be square.")
    if isinstance(matrix, np.ndarray):
        for i in range(matrix.shape[0]):
            for j in range(i + 1, matrix.shape[1]):
                if matrix[i, j] != matrix[j, i]:
                    raise Exception(f"Dense matrix is not symmetric: M[{i},{j}]={matrix[i, j]}, but M[{j},{i}]={matrix[j, i]}.")
    elif isinstance(matrix, scipy.sparse.csr_matrix):
        assert_symmetric_triplets_csr(matrix, err_msg="Sparse matrix validation failed: ")

def test_validate_square_and_symmetric_matrix():
    """
    Unit test to check that a matrix is a numpy ndarray or scipy.sparse.csr_matrix,
    and that it is square and symmetric.
    """

    # Valid dense
    arr = np.array([[1, 2], [2, 1]])
    validate_square_and_symmetric_matrix(arr)

    # Invalid dense (not symmetric)
    arr2 = np.array([[1, 0], [2, 1]])
    try:
        validate_square_and_symmetric_matrix(arr2)
        assert False, "Should have raised Exception for non-symmetric dense"
    except Exception:
        pass

    # Valid sparse
    data = np.array([1, 2, 2, 1])
    row = np.array([0, 0, 1, 1])
    col = np.array([0, 1, 0, 1])
    sp = scipy.sparse.csr_matrix((data, (row, col)), shape=(2, 2))
    validate_square_and_symmetric_matrix(sp)

    # Invalid sparse (not symmetric)
    data = np.array([1, 0, 2, 1])
    row = np.array([0, 0, 1, 1])
    col = np.array([0, 1, 0, 1])
    sp2 = scipy.sparse.csr_matrix((data, (row, col)), shape=(2, 2))
    try:
        validate_square_and_symmetric_matrix(sp2)
        assert False, "Should have raised Exception for non-symmetric sparse"
    except Exception:
        pass

    # Not square
    arr3 = np.ones((2, 3))
    try:
        validate_square_and_symmetric_matrix(arr3)
        assert False, "Should have raised Exception for non-square"
    except Exception:
        pass

    # Not ndarray or csr
    try:
        validate_square_and_symmetric_matrix([[1, 2], [2, 1]])
        assert False, "Should have raised TypeError for wrong type"
    except TypeError:
        pass

    print("All validate_square_and_symmetric tests passed.")





def minmax( matrix, return_row_and_column_indices=False ):
    """
    Returns the minimum of the maxima of the rows of an :math:`m \\times n` matrix :math:`A`. In symbols:

    .. math::
    
        \min_{i}  (  \max_{j} A_{ij} )

    - If :math:`m = 0`, then the minimum of the maxima is :math:`\\infty`.
    - If :math:`m > 0` and :math:`n = 0`, then the minimum of the maxima is :math:`-\\infty`.

    Parameters
    ----------
    matrix : scipy.sparse.csr_matrix or numpy.ndarray
        A 2-dimensional matrix, either dense or sparse. If the matrix is ``scipy.sparse.csr_matrix``, then 
        
        - structural zero entries will be treated as positive infinity.
        - the matrix does **not** have to have sorted indices, as flagged by ``matrix.has_sorted_indices``.

    return_row_and_column_indices : bool, optional
        If True, returns a dictionary with the row and column indices of the minimum of the maxima,
        along with the minimum value. If False (default), returns only the minimum value.
    
    Returns
    -------
    minmax_value : float
        If ``return_row_and_column_indices`` is False, returns the minimum of the maxima.
    result : dict
        If ``return_row_and_column_indices`` is True, returns a dictionary with keys:

            - ``row``: index :math:`i` of the row where the minimum of the maxima occurs,
            - ``col``: index :math:`j` of the column where the maximum occurs,
            - ``minmax_value``: the minmax value, equal to :math:`A_{ij}`.
    Raises
    ------
    TypeError
        If the input is not a ``scipy.sparse.csr_matrix`` or a ``numpy.ndarray``.
    Exception
        If the input is not a 2-dimensional matrix.
    """

    
    if not ( isinstance( matrix, scipy.sparse.csr_matrix ) or isinstance( matrix, np.ndarray ) ):
        raise TypeError("The input to oat.matrix.minmax must be a ``scipy.sparse.csr_matrix`` or a ``numpy.ndarray``")
    
    # Check if matrix is 2-dimensional
    if not hasattr(matrix, "shape") or len(matrix.shape) != 2:
        raise Exception(f"Input must be a 2-dimensional matrix, but got shape {getattr(matrix, 'shape', None)}")    

    # Handle edge cases for empty matrices
    m,n = matrix.shape
    if m == 0:
        # if the matrix has no rows then we minimize over the empty set, resulting in inf
        if return_row_and_column_indices:
            return dict( row=None, col=None, minmax_value=inf)
        else:
            return inf
    elif n == 0:
        # if the matrix has at least one row but no columns, then we maximize over the emptyset, resulting in -inf
        if return_row_and_column_indices:
            return dict( row=None, col=None, minmax_value=-inf)
        else:
            return -inf
        
    # handle the case where the matrix is dense
    if isinstance( matrix, np.ndarray ):
        row             =   matrix.max(axis=1).argmin()
        col             =   matrix[row].argmax()
        minmax_value    =   matrix[row][col]

        if return_row_and_column_indices:
            return dict( row=row, col=col, minmax= minmax_value )
        else:
            return minmax_value   

    # handle the case where the matrix is sparse
    if isinstance( matrix, scipy.sparse.csr_matrix ):

        def argmaxima_iterator():
            """
            For each row, yields the index of the column where the maximum value occurs
            """
            for row_index in range(m):
                linear_indices =  range(matrix.indptr[row_index], matrix.indptr[row_index+1])
                if len(linear_indices) < n:
                    yield inf
                else:
                    index_max = max( linear_indices, key= lambda i: matrix.data[i] )
                    yield matrix.data[index_max]

        (row, minmax_value ) = min( enumerate(argmaxima_iterator()), key= lambda x: x[1] )

        col = matrix[row].argmax()

        if return_row_and_column_indices:
            return dict( row=row, col=col, minmax=minmax_value )
        else:
            return minmax_value     
        


def test_minmax():
    """
    Unit tests for the `minmax` function.

    This test suite covers the following cases:
    - Dense (NumPy ndarray) and sparse (scipy.sparse.csr_matrix) matrices.
    - Matrices with zero rows or zero columns.
    - Matrices where all entries are different.
    - Matrices where all entries are the same.
    - Sparse matrices with positive shape but no structural nonzero entries.
    - Sparse matrices with some rows or columns that have no structural nonzero entries.

    The tests check both the value returned by `minmax` and the correctness of the returned row and column indices
    when `return_row_and_column_indices=True`.
    """    

    # Dense: all entries different
    dense_diff = np.array([[1, 2], [3, 4]])
    assert minmax(dense_diff) == 2
    res = minmax(dense_diff, return_row_and_column_indices=True)
    assert res['minmax'] == 2
    assert dense_diff[res['row'], res['col']] == 2

    # Dense: all entries the same
    dense_same = np.ones((3, 3)) * 7
    assert minmax(dense_same) == 7
    res = minmax(dense_same, return_row_and_column_indices=True)
    assert res['minmax'] == 7
    assert dense_same[res['row'], res['col']] == 7

    # Dense: zero rows
    dense_zero_rows = np.empty((0, 5))
    assert minmax(dense_zero_rows) == np.inf

    # Dense: zero columns
    dense_zero_cols = np.empty((5, 0))
    assert minmax(dense_zero_cols) == -np.inf

    # Sparse: all entries different
    data = np.array([1, 2, 3, 4])
    row = np.array([0, 0, 1, 1])
    col = np.array([0, 1, 0, 1])
    sparse_diff = scipy.sparse.csr_matrix((data, (row, col)), shape=(2, 2))
    assert minmax(sparse_diff) == 2
    res = minmax(sparse_diff, return_row_and_column_indices=True)
    assert res['minmax'] == 2
    assert sparse_diff[res['row'], res['col']] == 2

    # Sparse: all entries the same
    data = np.ones(4) * 5
    row = np.array([0, 0, 1, 1])
    col = np.array([0, 1, 0, 1])
    sparse_same = scipy.sparse.csr_matrix((data, (row, col)), shape=(2, 2))
    assert minmax(sparse_same) == 5
    res = minmax(sparse_same, return_row_and_column_indices=True)
    assert res['minmax'] == 5
    assert sparse_same[res['row'], res['col']] == 5

    # Sparse: zero rows
    sparse_zero_rows = scipy.sparse.csr_matrix((0, 3))
    assert minmax(sparse_zero_rows) == np.inf

    # Sparse: zero columns
    sparse_zero_cols = scipy.sparse.csr_matrix((3, 0))
    assert minmax(sparse_zero_cols) == -np.inf

    # Sparse: positive shape, no structural nonzero entries
    sparse_empty = scipy.sparse.csr_matrix((3, 3))
    assert minmax(sparse_empty) == np.inf
    res = minmax(sparse_empty, return_row_and_column_indices=True)
    assert res['minmax'] == np.inf
    

    # Sparse: some rows/cols with no structural nonzero entries
    data = np.array([1,2,3])
    row = np.array([0,0,0])
    col = np.array([0,1,2])
    sparse_some_empty = scipy.sparse.csr_matrix((data, (row, col)), shape=(3, 3))
    assert minmax(sparse_some_empty) == 3
    res = minmax(sparse_some_empty, return_row_and_column_indices=True)
    assert res['minmax'] == 3
    assert res['row'] == 0
    assert res['col'] == 2    


    # Sparse: unsorted entries
    data = np.array([1,2,3])
    row = np.array([0,0,0])
    col = np.array([0,1,2])
    sparse_some_empty = scipy.sparse.csr_matrix((data, (row, col)), shape=(3, 3))
    sparse_some_empty.data = sparse_some_empty.data[[2,1,0]]
    sparse_some_empty.indices = sparse_some_empty.indices[[2,1,0]]      
    assert minmax(sparse_some_empty) == 3
    res = minmax(sparse_some_empty, return_row_and_column_indices=True)
    assert res['minmax'] == 3
    assert res['row'] == 0
    assert res['col'] == 2        


    print("All minmax tests passed.")


def is_structurally_zero(matrix, i, j):
    """
    Returns True if entry (i, j) is structurally zero in a scipy CSR matrix.
    """
    # Get the start and end pointers for row i
    start = matrix.indptr[i]
    end = matrix.indptr[i + 1]
    # The columns with stored values in row i
    row_indices = matrix.indices[start:end]
    # If j is not among the stored column indices, it's structurally zero
    return j not in row_indices






# def enclosing_radius_for_csr( matrix, argminmax=False ):
#     """
#     Calculates the :term:`enclosing radius` of an :math:`n \\times n` sparse dissimilarity matrix.

#     - :term:`Structural zero entries <sparse matrix>` are treated as ``inf``.
#     - If :math:`n = 0`, then ``inf`` is returned.




#     Caution
#     -------
#     This function **does not check** that the input matrix is symmetric.

#     Parameters
#     ----------
#     matrix : scipy.sparse.csr_matrix
#         An :math:`n \\times n` :term:`sparse dissimilarity matrix <sparse dissimilarity matrix>` in CSR format.
#     argminmax : bool, optional
#         If True, returns a dictionary with the enclosing radius and the row/column where it occurs.
#         If False (default), returns only the enclosing radius.

#     Returns
#     -------
#     enclosing_radius : float
#         If ``argminmax`` is False, returns the :term:`enclosing radius`.
#     result : dict
#         If ``argminmax`` is True, returns a dictionary with keys:
        
#             - ``row``: index of the row,
#             - ``col``: index of the column,
#             - ``enclosing_radius``: the :term:`enclosing radius` value, equal to ``matrix[row][col]``.   
#     """

#     if not isinstance( matrix, scipy.sparse.csr_matrix ):
#         raise TypeError("The input to oat.dissimilarity.enclosing_radius_for_csr must be a ``scipy.sparse.csr_matrix``")
    
#     # check that the input is square
#     m,n = matrix.shape
#     if m != n:
#         raise Exception(f"Input must be square, but the matrix provided has shape {matrix.shape}")
    
#     # check that the input is dense
#     if not matrix.nnz == m ** 2:


#     if m == 0:
#         if argminmax:
#             return dict( row=None, col=None, enclosing_radius=-inf)
#         else:
#             return inf
#     elif n == 0:
#         if argminmax:
#             return dict( row=None, col=None, enclosing_radius=-inf)
#         else:
#             return -inf

#     def argmaxima_iterator():
#         """
#         For each row, yields the index of the column where the maximum value occurs
#         """
#         for row_index in range(m):
#             linear_indices =  range(matrix.indptr[row_index], matrix.indptr[row_index+1])
#             if len(linear_indices) < n:
#                 yield inf
#             else:
#                 index_max = max( linear_indices, key= lambda i: matrix.data[i] )
#                 yield matrix.data[index_max]

#     (row, enclosing_radius ) = min( enumerate(argmaxima_iterator()), key= lambda x: x[1] )

#     col = matrix[row].argmax()

#     if argminmax:
#         return dict( row=row, col=col, enclosing_radius=enclosing_radius )
#     else:
#         return enclosing_radius
    


# def enclosing_radius_for_dense( matrix, argminmax=False ):
#     """
#     The :term:`enclosing radius` of a :term:`dense dissimilarity matrix <dissimilarity matrix>`.
    
#     **Caution (numerical error and asymmetry)**

#     Distance matrices produced in Python often suffer from numerical error; for example,

#     - ``sklearn.neighbors.radius_neighbors_graph`` often produces asymmetric matrices
#     - ``sklearn.metrics.pairwise_distances`` often produces asymmetric matrices

#     This function **does not check** that the input matrix is symmetric.

#     Parameters
#     -----------

#     matrix : scipy.sparse.csr_matrix
#         The :term:`dense dissimilarity matrix <dissimilarity matrix>`

#     Returns
#     -------
#     enclosing_radius : float
#         If ``argminmax`` is False, returns the :term:`enclosing radius`.

#     enclosing_radius_dict : dict
#         If ``argminmax`` is True, returns a dictionary with keys:
#             - ``row``: index of the row,
#             - ``col``: index of the column,
#             - ``enclosing_radius``: the :term:`enclosing radius` value, equal to ``matrix[row][col]``.    
#     """

#     validate_square_and_symmetric( matrix )

#     m = matrix.shape[0]

#     if m == 0:
#         if argminmax:
#             return dict( row=None, col=None, enclosing_radius=inf)
#         else:
#             return inf

#     row     =   matrix.max(axis=1).argmin()
#     col     =   matrix[row].argmax()
#     enclosing_radius = matrix[row][col]

#     if argminmax:
#         return dict( row=row, col=col, enclosing_radius=enclosing_radius )
#     else:
#         return enclosing_radius 