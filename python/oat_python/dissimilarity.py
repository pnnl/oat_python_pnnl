"""
Dissimilarity tools, including distance matrices.

Dissimilarity metrics (including distance, inverse correlation, etc) play a major role in
persistent homology calculations, and in particular in the construction of Vietoris Rips complexes.
However, working with dissimilarity matrices can be onerous, especially when the matrices are large.
This module provides a variety of tools to address some of the most common issues.
"""

#   SEE THE BOTTOM OF THIS FILE FOR UNIT TESTS

import copy
from cmath import inf
import networkx as nx
import numpy as np
import sklearn
import sklearn.metrics
from . import point_cloud
import scipy
from sklearn.neighbors import radius_neighbors_graph
import itertools
import warnings

import oat_python.matrix

# Ignore a specific warning
warnings.filterwarnings("ignore", category=scipy.sparse.SparseEfficiencyWarning)




# class FormattedDissimilarityMatrix:
#     def __init__(self, sparse_matrix):
#         """
#         Wraps a ``sparse_matrix`` in a ``FormattedDissimilarityMatrix``.

#         The constructor will check that the input is a symmetric (with zero margin of error) scipy sparse ``csr_matrix``.

#         It will also store explicit an explicit zero value in entry ``[p,p]``, for all ``p`` such that ``sparse_matrix[p,p]==0``.
#         """

#         print("\n\nDirect construction of FormattedDissimilarityMatrix often yields suboptimal results; consider the other constructors offered in the ``dissimilarity`` module. \n\n")

#         if not isinstance(sparse_matrix, scipy.sparse.csr_matrix):
#             raise TypeError("Input matrix must be a scipy.sparse.csr_matrix.")
#         transposed = sparse_matrix.transpose
#         if not all( [transposed.data == sparse_matrix.data, transposed.indptr == sparse_matrix.indptr, transposed.indices == sparse_matrix.indices ] ):
#             raise Exception("Input matrix must be symmetric")  
              
#         # ensure that all diagonal entries are stored explicitly
#         missing_diagonal_indices = [p for p in range(sparse_matrix.shape[0]) if sparse_matrix[p,p] == 0 ]
#         sparse_matrix[ missing_diagonal_indices, missing_diagonal_indices ] = 0

#         self._dissimilarity_matrix = sparse_matrix

#     def get_matrix(self):
#         """
#         Returns the internally stored sparse matrix
#         """
#         return self._distance_matrix        


# This method can be paired with ``oat.dissimilarity.sparse_matrix_for_cloud_slow`` to generate a
#     sparse dissimilarity matrix with the same Vietoris-Rips persistent homology as the dense
#     dissimilarity matrix of the point cloud.
#     See Also
#     --------
#     enclosing_radius_for_points : Performs the same calculation but can be much faster (at the cost of small numerical error).
#     Definition
#     The enclosing radius of a point cloud is obtained from its distance matrix ``D`` by:
#         (i) taking the maximum of each row of ``D``, then
#         (ii) taking the minimum of these values over all rows.
#     The enclosing radius produced by this function is guaranteed to be compatible with the distance matrix
#     produced by ``sparse_matrix_for_cloud_slow``, in the sense that
#     ``sparse_matrix_for_cloud_slow(points, max_dissimilarity=enclosing_radius_for_cloud_slow(points))``
#     has the same Vietoris-Rips persistent homology as
#     ``sparse_matrix_for_cloud_slow(points, max_dissimilarity=inf)``.
#     points : array-like
#         A point cloud represented as a list of points, e.g., a list of tuples or a numpy array of shape (num_points, dimension).
#         Each slice ``points[i]`` will be treated as a point.
#     argminmax : bool, optional
#         If True, returns a dictionary with the enclosing radius and the indices of the two points where it occurs.
#         If False (default), returns only the enclosing radius.
#     Returns
#     -------
#     enclosing_radius : float
#         If ``argminmax`` is False, returns the enclosing radius.
#     result : dict
#         If ``argminmax`` is True, returns a dictionary with keys:
#             - ``pointa``: index of the first point,
#             - ``pointb``: index of the second point,
#             - ``enclosing_radius``: the enclosing radius value.
#     Notes
#     -----
#     - If the points is empty, returns infinity.
#     - If the points lives in R^0, returns 0.

    # :return enclosing radius float: if ``argminmax=False``, returns the enclosuing radius
    # :return enclosing_radius dict: if ``argminmax=True``, returns  a dictionary with enclosing radius and the indices of the two points where it occurs

    # Returns
    # -------



#    Calculates the enclosing radius of a point cloud without calculating a dense distance matrix.

#     This method can be paired with ``oat.dissimilarity.sparse_matrix_for_cloud_slow`` in order to generate a
#     sparse dissimilarity matrix with the same Vietoris Rips persistent homology as the dense
#     dissimilarity matrix as the point cloud.


 

def enclosing_radius_for_cloud_slow( points, argminmax=False ):
    """
    Calculates the :term:`enclosing radius` of a point cloud in a memory efficient manner
    (without generating and storing a copy of the distance matrix).

    See also
    --------
    :py:class:`oat_python.dissimilarity.enclosing_radius_for_points`
        This performs the same calculation but can be much faster (at the cost of small numerical error).  

        
    .. admonition:: Exact compatibility with :func:`sparse_matrix_for_cloud_slow`
    
        Most enclosing radius calculations need a :term:`small buffer to account for numerical error <enclosing radius>`.        
        However, this function 
        is specifically designed to work together with the function :func:`sparse_matrix_for_cloud_slow`.
        In particular, the dissimilarity matrix
        ``sparse_matrix_for_cloud_slow( points, max_dissimilarity = enclosing_radius_for_cloud_slow(points) )``
        is gauranteed to have Vietoris Rips persistent homology isomorphic to the dissimilarity matrix ``sparse_matrix_for_cloud_slow( points, dissimilarity_masx = inf )``.
        

    Parameters
    ----------
    points : array-like
        A point cloud represented as a list of points, e.g., a list of tuples or a numpy array of shape (num_points, dimension).
        Each slice ``points[i]`` will be treated as a point.
    argminmax : bool, optional
        If True, returns a dictionary with the enclosing radius and the indices of the two points where it occurs.
        If False (default), returns only the enclosing radius.

    Returns
    -------
    enclosing_radius : float
        If ``argminmax`` is False, returns the enclosing radius.
    result : dict
        If ``argminmax`` is True, returns a dictionary with keys:
            - ``pointa``: index of one point,
            - ``pointb``: index of another point, which lies at distance ``enclosing_radius`` from ``point_a``
            - ``enclosing_radius``: the enclosing radius value.
    """

    # if the points is empty, then we're taking a minimum over an empty set, which is inf
    if np.shape(points)[0] == 0:
        return inf
    
    # if the points lives in R^0 then every point is distance 0 from every other point
    if np.shape(points)[1] == 0:
        return 0
    

    def argmaxima_iterator():
        """
        For each point, yields the index of the most distant point
        """
        for point in points:
            yield max( range(len(points)), key= lambda i: euclidean_distance( points[i], point ) )

    ( row, col ) = min( enumerate( argmaxima_iterator() ), key = lambda x : euclidean_distance( points[x[0]], points[x[1]] ) )

    enclosing_radius = euclidean_distance(points[row], points[col])

    if argminmax:
        return dict( pointa=row, pointb=col, enclosing_radius=euclidean_distance(points[row], points[col]))
    else:
        return enclosing_radius
    

def enclosing_radius_for_points( points, argminmax=False ):
    """
    Calculates the :term:`enclosing radius` of a point cloud without calculating a dense distance matrix.

    Caution
    -------
    This calculation is subject to numerical error.  See the glossary entry on :term:`enclosing radius` for details.

    Parameters
    ----------
    points : array-like
        Any format for a point cloud compatible with ``sklearn.neighbors.KDTree``.
    argminmax : bool, optional
        If True, returns a dictionary with the enclosing radius and the indices of the two points where it occurs.
        If False (default), returns only the enclosing radius.

    Returns
    -------
    enclosing_radius : float
        If ``argminmax`` is False, returns the enclosing radius.
    result : dict
        If ``argminmax`` is True, returns a dictionary with keys:
            - ``pointa``: index of one point,
            - ``pointb``: index of another point, which lies at distance ``enclosing_radius`` from ``pointa``,
            - ``enclosing_radius``: the enclosing radius value.
    """

    # if the points is empty, then we're taking a minimum over an empty set, which is inf
    if np.shape(points)[0] == 0:
        return inf
    
    # if the points lives in R^0 then every point is distance 0 from every other point
    if np.shape(points)[1] == 0:
        return 0

    n_points     =   np.shape(points)[0]
    tree        =   sklearn.neighbors.KDTree(points)
    minmax_val  =   np.inf
    minmax_row  =   0
    for row in range(points.shape[0]):
        (cols,vals) =   tree.query_radius(  
                            np.array(points[row]).reshape(1,-1), 
                            minmax_val + 0.0000000001, 
                            return_distance=True 
                        )
        cols    =   cols[0] # index into first (and only) row of this 2d numpy array
        vals    =   vals[0] # index into first (and only) row of this 2d numpy array
        if len(cols) == n_points: 
            col_rel         =   vals.argmax() 
            val             =   vals[col_rel]
            if val < minmax_val:
                minmax_val  =   val
                minmax_row  =   row
                minmax_col  =   cols[col_rel]

    if argminmax:
        cols        =   tree.query_radius(  
                                np.array(points[minmax_row]).reshape(1,-1), 
                                minmax_val + 0.0000000001, 
                                return_distance=True,  
                        )[-1] 
        max_ind     =   cols[1][0].argmax()
        minmax_col  =   cols[0][0][max_ind]
        # and take the last column in the sequence
     

        return dict( pointa=minmax_row, pointb=minmax_col, enclosing_radius=minmax_val)
    else:
        return minmax_val    
    


def enclosing_radius_for_matrix( 
        matrix, 
        validate_input=True,
        return_row_and_column_indices=False 
    ):
    """
    Calculates the :term:`enclosing radius` of a :term:`sparse dissimilarity matrix` or  :term:`dense dissimilarity matrix <dissimilarity matrix>`.

    This function is a wrapper for the following code:
    
    .. code-block:: python
    
        if validate_input:
            oat_python.matrix.validate_square_and_symmetric_matrix( matrix )

        return oat_python.matrix.minmax(
            matrix, 
            return_row_and_column_indices=return_row_and_column_indices,
        )

    
    Caution
    -------
    If ``validate_input`` is False and the input is not a valid
    :term:`sparse dissimilarity matrix` or a :term:`dense dissimilarity matrix <dissimilarity matrix>`
    (e.g. not square, not symmetric, or not a valid filter on the underlying simple graph),
    then this function will still return a value for the enclosing radius, but that
    value may not be meaningful in the context of persistent homology. Check the documentation
    for :term:`enclosing radius` if you want to work in this setting.



    Parameters
    ----------
    matrix : numpy.ndarray or scipy.sparse.csr_matrix
        A square symmetric :term:`sparse dissimilarity matrix` or  :term:`dense dissimilarity matrix <dissimilarity matrix>`
        in numpy array or scipy sparse CSR format.
    return_row_and_column_indices : bool, optional
        If True, returns a dictionary with the enclosing radius and a row/column where it occurs.
        If False (default), returns only the enclosing radius.
    validate_input : bool, optional
        If True (default), checks that the input is a valid
        :term:`sparse dissimilarity matrix` or a :term:`dense dissimilarity matrix <dissimilarity matrix>`.
        If False, skips this check and assumes the input is valid.
    

    Returns
    -------
    enclosing_radius : float
        If ``argminmax`` is False, returns the :term:`enclosing radius`.
    result : dict
        If ``argminmax`` is True, returns a dictionary with keys:

            - ``row``: index of the row,
            - ``col``: index of the column,
            - ``enclosing_radius``: the :term:`enclosing radius` value, equal to ``matrix[row][col]``.

    Raises
    ------
    TypeError
        If the input is not a numpy array or a scipy sparse CSR matrix.
    Exception
        If the input matrix
        (A) is not 2d, 
        (B) is not square, and ``validate_input`` is True, 
        (C) is not symmetric, and ``validate_input`` is True, or 
        (D) is not a valid filter on the underlying simple graph, and ``validate_input`` is True.
        This condition is described in the glossary entries for :term:`dissimilarity matrix` and :term:`dissimilarity matrix`.
    """

    # if desired, check that the input is square and symmetric
    if validate_input:
        oat_python.matrix.validate_square_and_symmetric_matrix( matrix )

    return oat_python.matrix.minmax(
        matrix, 
        return_row_and_column_indices=return_row_and_column_indices,
    )



def sparse_matrix_for_points( points, max_dissimilarity ):
    """
    Returns a sparse (Scipy CSR) Euclidean distance matrix, where all entries with value strictly greater than ``max_dissimilarity`` are dropped.

    **Assymetry**

    This function uses ``sklearn.neighbors.radius_neighbors_graph`` to construct a dissimilarity matrix, however
    the output of this function is **not symmetric** in general.  Therefore the construction process takes two steps:

    - construct a matrix ``A`` with ``sklearn.neighbors.radius_neighbors_graph``
    - replace ``A`` with the entrywise-maximum of ``A`` and the transpose of ``A``
    - store explicit zero values for all diagonal entries
    - if ``max_dissimilarity`` is nonnegative, then store explicit zero values for all diagonal entries;
      otherwise return an empty sparse matrix of appropriate size     

    Parameters
    ------------
    points : array-like
        Any format for a point cloud compatible with ``sklearn.neighbors.radius_neighbors_graph``.
    max_dissimilarity : float
        A non-negative real number; all distances with value above this threshold are dropped.

    Examples
    -----------
    .. code-block:: python

        import numpy as np
        import oat_python as oat

        points           =   np.random.rand(10,2)
        dissimilarity   =   oat.dissimilarity.sparse_matrix_for_cloud_slow(
                                points               =   points, 
                                max_dissimilarity   =   oat.dissimilarity.enclosing_radius(points) 
                            )
    """

    # if the points is empty then return a 0 x 0 matrix
    if np.shape(points)[0] == 0:
        return scipy.sparse.csr_matrix((0,0))

    if max_dissimilarity < 0:
        num_rows = np.shape(points)[0] # number of points in the points
        return scipy.sparse.csr_matrix((num_rows,num_rows)) # return an empty matrix of appropriate size

    A   =   radius_neighbors_graph( 
                points, 
                radius = max_dissimilarity, 
                mode='distance', 
                include_self=True
            )    
    A   =   A.maximum( A.T ) # ensure the matrix is symmetric
    
    A.setdiag(0) # set all diagonal entries to zero; otherwise the matrix will be empty, including along the diagonal, which is correct behavior
    A.sort_indices()
    
    return A

#   DEPRECATED; OK TO DELETE
# def sparse_distance_matrix(
#         points,
#         distance_max            =   None,
#         enclosing_tolerance     =   None,
#     ):
#     """
#     Returns a sparse, symmetric``scipy.sparse.csr`` matrix representing
#     the pairwise distance between points.  Distances over the provided
#     threshold are excluded.

#     If ``distance_max`` is ``None``, then the enclosing radius of the point
#     points will be used.
#     """

#     if distance_max is None:
#         if enclosing_tolerance is None:
#             A           =   scipy.sparse.csr_matrix( sklearn.metrics.pairwise_distances(points) )
#             enclosing   =   None
#         else:            
#             enclosing   =   enclosing_radius(points)
#             A           =   radius_neighbors_graph( 
#                                 points, 
#                                 enclosing["enclosing_radius"] * (1+enclosing_tolerance), 
#                                 mode='distance', 
#                                 include_self=True
#                             )  
#     else:
#         if not enclosing_tolerance is None:
#             Exception("Either ``eclosing_tolerance`` or ``distance_max`` must be ``None``")
#         else:
#             A           =   radius_neighbors_graph( 
#                                 points, 
#                                 distance_max, 
#                                 mode='distance', 
#                                 include_self=True
#                             )       
#             enclosing   =   None    
#     A   =   A.maximum( A.transpose() ) # ensures the matrix is symmetric
#     A[ range(len(points)), range(len(points)),  ] = 0
#     A.sort_indices()

#     return dict(distance_matrix = A, enclosing=enclosing)


def sparse_matrix_for_dense( dissimilarity_matrix, max_dissimilarity = np.inf ):
    """
    Converts a dense :term:`dissimilarity matrix` to a scipy sparse CSR matrix that meets
    the formatting requirements of the OAT persistent homology solver.
    
    In particular, this constructor checks that

    - the input is a square, symmetric numpy array
    - for each i, the entry (i,i) takes the smallest value of any entry in row i
    
    It then stores the input in a scipy sparse CSR matrix, where

      - all entries with value strictly greater than ``max_dissimilarity`` are removed
      - all entries with value less than or equal to ``max_dissimilarity`` are stored explicitly -- including entries with value 0

    Parameters
    ----------
    dissimilarity_matrix : numpy.ndarray
        A square symmetric :term:`dissimilarity matrix` in numpy array format.
    max_dissimilarity : float, optional
        A real number; all entries with value above this threshold will be dropped.

    Returns
    -------
    dissimilarity_matrix : scipy.sparse.csr_matrix
        A square symmetric :term:`sparse dissimilarity matrix` in scipy CSR format, with all entries greater than ``max_dissimilarity`` removed.
        This matrix is guaranteed to meet the formatting requirements of the OAT persistent homology solver.

    Raises
    ------
    TypeError
        If the input is not a numpy array.
    Exception
        - If the input matrix is not square
        - If the input matrix is not symmetric
        - There exists an :math:`i` such that row `i` of the input matrix contains a structural nonzero entry,
          but the diagonal entry `(i,i)` is structurally zero
    """

    if not isinstance( dissimilarity_matrix, np.ndarray ):
        raise TypeError("The input to ``oat.dissimilarity.sparse_matrix_for_dense`` must be a numpy array")

    if (len(dissimilarity_matrix.shape)!=2) or ( dissimilarity_matrix.shape[0] != dissimilarity_matrix.shape[1]):
        raise Exception(f"Input must be 2-dimensional square matrix, but the matrix provided has shape {dissimilarity_matrix.shape}")
    
    # if the matrix has size 0, then return the size-zero sparse matrix
    if dissimilarity_matrix.shape[0] == 0:
        return scipy.sparse.csr_matrix((0,0))
    
    for p in range(dissimilarity_matrix.shape[0]):
        if dissimilarity_matrix[p,p] != dissimilarity_matrix[p].min():
            raise Exception("The diagonal entry in each row must take the minimum value in that row.")
        for q in range(p, dissimilarity_matrix.shape[0]):
            if not dissimilarity_matrix[p,q] == dissimilarity_matrix[q,p]:
                raise Exception("Input must be symmetric (with zero margin of error)")
    
    data = []; row=[]; col=[];
    for ((i,j),val) in np.ndenumerate( dissimilarity_matrix ):
        if val <= max_dissimilarity:
            data.append(val)
            row.append(i)
            col.append(j)
    
    matrix = scipy.sparse.csr_matrix( 
                ( data, ( row, col ) ), 
                shape = dissimilarity_matrix.shape 
            )
    matrix.sort_indices()
    return matrix


def sparse_matrix_for_csr(
        dissimilarity_matrix, 
        max_dissimilarity               =   np.inf,
        clamp_diagonal_entries          =   True,
    ):
    """
    Validates and formats a scipy sparse CSR dissimilarity matrix for use with the OAT persistent homology solver.

    **Validation**

    This constructor checks that

    - the input is a square, symmetric scipy.sparse.csr_matrix :math:`D`

    - for each i, if row i of :math:`D` contains at least one :term:`structural nonzero entry<sparse matrix>`
      then the diagonal entry :math:`D_{i,i}` is structurally nonzero, and :math:`D_{i,i}` 
      is the minimum of all structural nonzeros stored in row i.

    **Formatting**

    In an out-of-place fashion, this constructor:

    - deletes all entries with value strictly greater than ``max_dissimilarity``
    - sorts the internal data structures that store the column indices and values of the structural nonzero entries
    
    Parameters
    ----------
    dissimilarity_matrix : scipy.sparse.csr_matrix
        A square symmetric :term:`sparse dissimilarity matrix` in scipy CSR format.
    max_dissimilarity : float, optional
        A real number; all entries with value above this threshold will be dropped.
    clamp_diagonal_entries : bool, optional
        If True (default), ensures that each diagonal entry is the smallest structural nonzero in its row.
        Diagonal entries in rows with no structural nonzero entries are left unchanged (as structural zeros).
        If False, does not modify diagonal entries, but throws an error if any diagonal entry violates the formatting requirements.

    Returns
    -------
    dissimilarity_matrix : scipy.sparse.csr_matrix
        A square symmetric :term:`sparse dissimilarity matrix` in scipy CSR format, with all entries greater than ``max_dissimilarity`` removed.
        This matrix is guaranteed to meet the formatting requirements of the OAT persistent homology solver.

    Raises
    ------
    TypeError
        If the input is not a ``scipy.sparse.csr_matrix``.
    Exception
        - If the input matrix is not square
        - If the input matrix is not symmetric
        - There exists an :math:`i` such that row `i` of the input matrix contains a structural nonzero entry,
          but the diagonal entry `(i,i)` is structurally zero
    """

    # error if input is not a scipy sparse csr matrix
    if not isinstance( dissimilarity_matrix, scipy.sparse.csr_matrix ):
        raise TypeError("The input to oat.dissimilarity.sparse_matrix_for_csr must be a ``scipy.sparse.csr_matrix``")    
    

    # error if input is not square
    if (len(dissimilarity_matrix.shape)!=2) or ( dissimilarity_matrix.shape[0] != dissimilarity_matrix.shape[1]):
        raise Exception(f"Input must be 2-dimensional square matrix, but the matrix provided has shape {dissimilarity_matrix.shape}")
    
    # error if input is not symmetric  
    for p in range(dissimilarity_matrix.shape[0]):
        row = dissimilarity_matrix.getrow(p)
        for q, v in zip( row.indices, row.data ): # row.indices is a list of column indices, row.data is a list of values
            if not v == dissimilarity_matrix[q,p]:
                raise Exception(f"Input must be symmetric, with zero margin of error. However, entry {(p,q)} is not equal to entry {(q,p)}")


    dissimilarity_matrix = copy.deepcopy(dissimilarity_matrix)


    # if the user doesn't permit modification of diagonal entries
    # then throw an error if any violates the formatting requirements
    # (specifically, each diagonal entry must be the smallest structural nonzero in its row)
    if not clamp_diagonal_entries:
        for p in range(dissimilarity_matrix.shape[0]):
            row = dissimilarity_matrix.getrow(p)
            if len(row.data) > 0:
                if not p in row.indices:
                    raise Exception(f"\n\nRow {p} contains at least one structural nonzero entry, but no "
                                    f"structural nonzero entry for column {p}. This violates OAT formatting requirements."
                                    f"\n\nIn most cases users want all diagonal "
                                    f"entries to equal zero, so you can fix this problem by calling ``matrix.setdiag(0)``."
                                    f"Alternatively, if you would like each diagonal entry to be the smallest structural nonzero entry in its row,"
                                    f"you can call ``sparse_matrix_for_csr`` with `clamp_diagonal_entries=True`.")
                if dissimilarity_matrix[p,p] > np.min( row.data ):
                    raise Exception(f"Entry {(p,p)} contains an explicit entry, but it's not the smallest explicit entry in row {p}"
                                    f"This violates OAT formatting requirements."
                                    f"\n\nIn most cases users want all diagonal "
                                    f"entries to equal zero, so you can fix this problem by calling ``matrix.setdiag(0)``."
                                    f"Alternatively, if you would like each diagonal entry to be the smallest structural nonzero entry in its row,"
                                    f"you can call ``sparse_matrix_for_csr`` with `clamp_diagonal_entries=True`.")                                    



    # remove entries over the max
    dissimilarity_matrix      =   scipy.sparse.coo_matrix(dissimilarity_matrix)
    I                         =   [ i for (i, val) in enumerate(dissimilarity_matrix.data) if val <= max_dissimilarity ]
    dissimilarity_matrix.row  =   dissimilarity_matrix.row[I]
    dissimilarity_matrix.col  =   dissimilarity_matrix.col[I]
    dissimilarity_matrix.data =   dissimilarity_matrix.data[I]
    dissimilarity_matrix      =   scipy.sparse.csr_matrix(dissimilarity_matrix)
    dissimilarity_matrix.sort_indices()

    # if user permits modification of diagonal entries, then ensure that each diagonal entry is the smallest structural nonzero in its row
    if clamp_diagonal_entries:
        diagonal_indices = []
        diagonal_values = []
        for p in range(dissimilarity_matrix.shape[0]):
            row = dissimilarity_matrix.getrow(p)
            if len(row.data) > 0:
                diagonal_indices.append(p)
                diagonal_values.append(np.min(row.data))
                if p in row.indices:
                    dissimilarity_matrix[p,p] = 0 # this ensures that we we append the minimum value to the end fo the triplets, we won't accidentally double the value
        dissimilarity_matrix        =   scipy.sparse.coo_matrix(dissimilarity_matrix)
        dissimilarity_matrix.row    =   np.append(dissimilarity_matrix.row,  diagonal_indices)
        dissimilarity_matrix.col    =   np.append(dissimilarity_matrix.col,  diagonal_indices)
        dissimilarity_matrix.data   =   np.append(dissimilarity_matrix.data, diagonal_values)
        dissimilarity_matrix        =   scipy.sparse.csr_matrix(dissimilarity_matrix)  
        dissimilarity_matrix.sort_indices()      
    
    return dissimilarity_matrix



def farthest_point_sampling(
        metric_space,
        stopping_condition = dict(epsilon=0),
    ):
    """
    Applies farthest point sampling to a point cloud or a dissimilarity matrix.

    **Input types**

    The input can be one of the following:

    - A point cloud with :math:`n` points.    
    - A metric space :math:`M = \{m_0, \ldots, m_{n-1}\}` represented by an
      :math:`n \\times n` square symmetric matrix :math:`D`, where :math:`D_{i,j}`
      is the distance between points :math:`m_i` and :math:`m_j`.
    - More generally, any square symmetric `n \\times n` matrix :math:`D`.

    No matter the input type, it's standard to regard the input informally as a metric space with points labled :math:`0, \ldots, n-1`,
    and regard each entry :math:`D_{i,j}` as the distance between points :math:`i` and :math:`j`.

    **Performance**

    If your data is a point cloud, then this function will typically perform much better
    if the points is passed as a list of points, e.g., a list of tuples or a numpy array of shape (num_points, dimension),
    rather than a distance matrix.
    This is because there is a significant memory cost to storing a large distance matrix in memory.
    The implementation of this function is carefully designed to avoid constructing a distance matrix
    at any time, so it can handle large point clouds efficiently.

    **Goal**

    The overall goal of this algorithm is select a collection of points :math:`X = \{i_0, \ldots, i_N\}`
    that samples evenly from the space as a whole. Specifically, we want to ensure that
    every point in the space is close to some point in :math:`X`, and no two points in `X` are too close together.

    **Algorithm**

    Farthest point sampling is an iterative procedure that builds a sequence of points as follows:

    1. Start by selecting an initial index :math:`i_0 = 0` (the first row/column).
       Let :math:`e_0 = \max \{ D_{i_0,i_k} : 0 < k < n \}` be the maximum distance from :math:`i_0` to any other point.

    2. At each step, choose :math:`i_m` as far as possible from :math:`i_0, .., i_{m-1}`.
       That is, we choose :math:`i_m ∉ \{i_0, \ldots, i_{m-1}\}` such that the scalar value
       :math:`e_m: = \min \{ D_{i_k,i_m} : k < m \}` is as large as possible.

    3. Repeat until some stopping condition is met (e.g. the sample has a desired size, or :math:`e_m` is sufficiently small).

    This produces:
      - a sequence of indices :math:`i_0,  \ldots, i_N` (the selected points, in order)
      - a sequence of values :math:`e_0, \ldots, e_N` (the covering radii at each step)

    If  :math:`D` is distance matrix of a true metric space,
    then the selected set forms an `epsilon net <https://en.wikipedia.org/wiki/Delone_set>`__ for epsilon = :math:`e_N`.
    Moreover, for each :math:`m`, the set :math:`\{i_0, ..., i_m\}` forms an epsilon net for epsilon = :math:`e_m`.

    If :math:`D` is not the distance matrix of a metric space then the output may not
    be an epsilon net, but it may still be a reasonable approximation of an "even" sample; for example,
    the output is still a set of points such that
    every point is within distance :math:`e_N` of the selected set (i.e., a covering of radius :math:`e_N`),
    but the set may not satisfy all properties of an epsilon net in a metric space.

    Parameters
    ----------
    metric_space : dict( str: array-like or scipy.sparse matrix )
        A dict containing a point cloud or a square symmetric matrix.

        The key should be either "point_cloud" or "dissimilarity_matrix", and the value should be one of the following:
        
          - If the key is "point_cloud", the value should be a point cloud represented as a list of points, e.g., a list of tuples or a numpy array of shape (num_points, dimension).
            Each slice ``points[i]`` will be treated as a point.

          - If the key is "dissimilarity_matrix", the value should be a square symmetric matrix represented as a numpy array or scipy sparse CSR format.
            This matrix does not have to satisfy the properties of a metric space (i.e. triangle inequality), but it should be square and symmetric.

    stopping_condition : dict, optional
        A dictionary specifying the stopping condition for the sampling process. This must be one of the following:

        - ``dict(epsilon=*)``: The sampling stops when :math:`e_m` is less than or equal to the given ``epsilon``.
        - ``dict(max_points=*)``: The sampling stops when the number of sampled points reaches the given maximum.

    Returns
    -------
    sampled_indices : list of int
        The list of indices :math:`[i_0, i_1, \ldots, i_N]` representing the sampled points.
    covering_radii : list of float
        The list of covering radii :math:`[e_0, e_1, \ldots, e_N]` corresponding to the sampled points.
    Notes
    -----
        - If the input is an empty point cloud or :math:`0 \\times 0` matrix, then `([], [])` is returned.
        - If the input is a point cloud with only one point, then `([0], [0])` is returned.
        - If the input is a square symmetric matrix with only one row/column, then `([0], [matrix[0,0]])` is returned.
        - If the input is a sparse matrix, then the function will treat all structural zero entries as infinity, not as zero.
    Raises
    ------
    TypeError
        If the input is not a numpy array or a scipy sparse CSR matrix, or if the input type is not recognized.
    Exception
        If the input matrix is not square or not symmetric, or if the stopping condition is not recognized.
    """


    def select_disjoint_argmax( disvec, greedyperm ):
        """
        Selects the index of the maximum value in disvec that is not in greedyperm.
        If all indices are in greedyperm, raises an exception.
        """
        maxpnt = np.argmax(disvec)

        if maxpnt not in greedyperm:
            # if the maximum is not in greedyperm, return it
            return maxpnt
        else:
            # otherwise, find an index not in greedyperm with the same maximum value
            maxval = disvec[maxpnt]
            for index, val in enumerate(disvec):
                if maxval == val:
                    if index not in greedyperm:
                        return index
            
        raise Exception("No available index in disvec with max value that is not already in greedyperm.")



    # EXCEPTION IF INPUT SPECIFICATION IS NOT RECOGNIZED
    # ==================================================
    if list(metric_space.keys()) not in [ ["point_cloud"], ["dissimilarity_matrix"] ]:
        raise Exception("Input `metric_space` must be either `dict(point_cloud=*)` or `dict(dissimilarity_matrix=*)`.")

    input_type, metric_space = list(metric_space.items())[0]
    

    # EXCEPTION IF STOPPING CONDITION IS NOT RECOGNIZED
    # ==================================================

    space_cardinality   =   len(metric_space) if input_type == "point_cloud" else metric_space.shape[0]

    if list(stopping_condition.keys()) not in [ ["epsilon"], ["max_points"] ]:
        raise Exception("Stopping condition must be either `dict(epsilon=*)` or `dict(max_points=*)`.")
    
    max_points          =   inf
    epsilon             =   inf
    if "epsilon" in stopping_condition:
        epsilon         =   stopping_condition["epsilon"]
    if "max_points" in stopping_condition:
        max_points      =   stopping_condition["max_points"]
    
    def stop_sampling( num_points, max_distance ):
        """
        Returns True if the sampling should stop, based on the stopping condition.
        """
        return num_points >= max_points or max_distance <= epsilon or num_points == space_cardinality   
        
    
    # POINT CLOUD
    # ==============
    if input_type == "point_cloud":

        metric_space = metric_space

        # edge cases
        if metric_space.shape[0] == 0:
            return [], []
        if metric_space.shape[0] == 1:
            return [0], [0]
        
        # entry j of this vector represents the min distance from point j to any other point in the point cloud    
        disvec      =   sklearn.metrics.pairwise_distances( metric_space[[0],:], metric_space )[0]
        maxpnt      =   np.argmax( disvec )
        maxdis      =   disvec[ maxpnt ]      
        maxdiscurve =   [maxdis] # maxdiscurve[p] = max distance from any point to the first p+1 points of the permutation
        greedyperm  =   [0, maxpnt]    

        # edge cases
        if max_points == 0:
            return [], []
        if max_points == 1:        
            return [0], [maxdis]        

        # run algorithm
        while not stop_sampling( len(greedyperm), maxdis ):
            disvec_marg =   sklearn.metrics.pairwise_distances( metric_space[[maxpnt],:], metric_space )[0]
            disvec      =   np.minimum( disvec, disvec_marg  ) # smallest distance to any point in our net
            maxpnt      =   select_disjoint_argmax( disvec, greedyperm )          
            maxdis      =   disvec[ maxpnt ]
            greedyperm.append(maxpnt)
            maxdiscurve.append(maxdis)

        # add the last covering radius to the maxdiscurve
        disvec_marg     =   sklearn.metrics.pairwise_distances( metric_space[[maxpnt],:], metric_space )[0]
        disvec          =   np.minimum( disvec, disvec_marg  ) # smallest distance to any point in our net        
        maxpnt          =   select_disjoint_argmax( disvec, greedyperm )
        maxdis          =   disvec[ maxpnt ] 
        maxdiscurve.append(maxdis)               

        return greedyperm, maxdiscurve    

    # DISTANCE MATRIX
    # ===============

    elif input_type == "matrix":

        dissimilarity_matrix = metric_space

        # edge cases
        if dissimilarity_matrix.shape[0] == 0:
            return [], []
        if dissimilarity_matrix.shape[0] == 1:
            return [0], [dissimilarity_matrix[0,0]]



        def get_disvec(dissimilarity_matrix,i):
            """
            If dissimilarity_matrix is dense then return dissimilarity_matrix[i].
            If dissimilarity_matrix is sparse then convert dissimilarity_matrix[i] to dense a dense vector disvec
            and replace disvec[j] with infinity whenever disvec[j]=0 (except for j=i)
            """
            if type(dissimilarity_matrix) == scipy.sparse._csr.csr_matrix:
                disvec = np.asarray(dissimilarity_matrix[i].todense()).reshape(-1)
                disvec[disvec==0] = np.inf
                disvec[i] = 0
                return disvec 
            else:
                return dissimilarity_matrix[i]

        disvec          =   get_disvec(dissimilarity_matrix,0)  # entry j of this vector represents the min distance from point j to any other point in the point cloud
        maxpnt          =   disvec.argmax()
        maxdis          =   disvec[ maxpnt ]
        maxdiscurve     =   [maxdis] # maxdiscurve[p] = max distance from any point to the first p+1 points of the permutation
        greedyperm      =   [0, maxpnt] 

        # edge cases
        if max_points == 0:
            return [], []
        if max_points == 1:        
            return [0], [maxdis]                 

        # run algorithm
        while not stop_sampling( len(greedyperm), maxdis ):
            disvec      =   np.minimum( disvec, get_disvec(dissimilarity_matrix, maxpnt)  ) # smallest distance to any point in our net
            maxpnt      =   select_disjoint_argmax( disvec, greedyperm )
            maxdis      =   disvec[ maxpnt ]
            greedyperm.append(maxpnt)
            maxdiscurve.append(maxdis)

        # add the last covering radius to the maxdiscurve
        disvec          =   np.minimum( disvec, get_disvec(dissimilarity_matrix, maxpnt)  ) # smallest distance to any point in our net            
        maxpnt          =   select_disjoint_argmax( disvec, greedyperm )
        maxdis          =   disvec[ maxpnt ] 
        maxdiscurve.append(maxdis)               

        return greedyperm, maxdiscurve

# ==================================================
# BEGIN SECTION: DEPRECATED, OK TO DELETE
# ==================================================

# def farthest_point_sampling_for_matrix( dissimilarity_matrix = np.zeros((0,0)), epsilon = 0 ):
#     """
#     Applies farthest point sampling to a sparse or dense symmetric `n \times n` matrix :math:`D`.
#     We typically think of the indices :math:`0, \ldots, n-1`
#     as points in a metric space, and regard each entry :math:`D_{i,j}` as the distance between points :math:`i` and :math:`j`.
#     However, this procedure can be applied to any square symmetric matrix :math:`D`.    

#     Farthest point sampling is an iterative procedure that builds a sequence of points as follows:

#     1. Start by selecting an initial index :math:`p_0 = 0` (the first row/column).
#     Let :math:`e_0 = \max \{ D_{p_0,p_i} : i > 0 \}` be the maximum distance from :math:`p_0` to any other point.

#     2. At each step, choose :math:`p_m` as far as possible from :math:`p_0, .., p_{m-1}`.
#        That is, we choose :math:`p_m ∉ \{p_0, \ldots, p_{m-1}\}` such that the scalar value
#        :math:`e_m: = \min \{ D_{p_i,p_m} : i < m \}` is as large as possible.

#     3. Repeat until :math:`e_m` is less than or equal to ``epsilon``.

#     This produces:
#       - a sequence of indices :math:`p_0, p_1, \ldots, p_N` (the selected points, in order)
#       - a sequence of values :math:`e_0, e_1, ..., e_N` (the covering radii at each step)

#     If the input matrix ``dissimilarity_matrix`` is a true distance matrix for a metric space,
#     then the selected set forms an `epsilon net <https://en.wikipedia.org/wiki/Delone_set>`__ for epsilon = :math:`e_N`.
#     Moreover, for each :math:`m`, the set :math:`[p_0, ..., p_m]` forms an epsilon net for epsilon = :math:`e_m`.

#     If ``dissimilarity_matrix`` is not a metric, then the output is still a set of points such that
#     every point is within distance :math:`e_N` of the selected set (i.e., a covering of radius :math:`e_N`),
#     but the set may not satisfy all properties of an epsilon net in a metric space.

#     Parameters
#     ----------
#     dissimilarity_matrix : array-like or scipy.sparse matrix
#         A sparse or dense square symmetric matrix; all structural zero entries will be treated as infinity.

#     epsilon : float
#         The covering radius threshold.

#     Returns
#     -------
#     greedyperm : list of int
#         The list of indices :math:`[p_0, p_1, \ldots, p_N]`.

#     maxdiscurve : list of float
#         The list of covering radii :math:`[e_0, e_1, \ldots, e_N]`

#     Notes
#     -----
#         - If the input matrix is empty, then `([], [])` is returned.
#         - If the input matrix has only one row/column, then `([0], [dissimilarity_matrix[0][0]])` is returned.    
#         - It is possible that :math:`e_N` will be greater than ``epsilon``, depending on the entries in the `dissimilarity_matrix``.
#           This happens only if the list :math:`[p_0, p_1, \ldots, p_N]` contains all the points in the metric space.

#     Raises
#     -------
#     Exception
#         - If the input matrix is not square or not symmetric, an exception is raised.
#         - If the input is not a numpy ndarray or a scipy sparse csr_matrix, an exception is raised.
#     """
#     print("This method of farthest point sampling may be slow.  For large point clouds, try ``farthest_point_sampling_for_points``, which has the added advantage that you don't have to compute a distance matrix.")
    
#     if isinstance(dissimilarity_matrix, np.ndarray):
#         assert dissimilarity_matrix.shape[0] == dissimilarity_matrix.shape[1], "Matrix must be square."
#         if not np.array_equal(dissimilarity_matrix, dissimilarity_matrix.T):
#             raise AssertionError("Dense dissimilarity matrix is not symmetric.")
#     elif isinstance(dissimilarity_matrix, scipy.sparse.csr_matrix):
#         assert dissimilarity_matrix.shape[0] == dissimilarity_matrix.shape[1], "Matrix must be square."
#         diff = (dissimilarity_matrix - dissimilarity_matrix.T).tocoo()
#         if diff.nnz > 0:
#             raise AssertionError("Sparse dissimilarity matrix is not symmetric.")
#     else:
#         raise TypeError("Input must be a numpy ndarray or a scipy.sparse.csr_matrix.")    


#     if dissimilarity_matrix.shape[0] == 0:
#         return [], []
#     if dissimilarity_matrix.shape[0] == 1:
#         return [0], [dissimilarity_matrix[0][0]]

#     def get_disvec(dissimilarity_matrix,i):
#         """
#         If dissimilarity_matrix is dense then return dissimilarity_matrix[i].
#         If dissimilarity_matrix is sparse then convert dissimilarity_matrix[i] to dense a dense vector disvec
#         and replace disvec[j] with infinity whenever disvec[j]=0 (except for j=i)
#         """
#         if type(dissimilarity_matrix) == scipy.sparse._csr.csr_matrix:
#             disvec = np.asarray(dissimilarity_matrix[i].todense()).reshape(-1)
#             disvec[disvec==0] = np.inf
#             disvec[i] = 0
#             return disvec 
#         else:
#             return dissimilarity_matrix[i]

#     disvec          =   get_disvec(dissimilarity_matrix,0)  # entry j of this vector represents the min distance from point j to any other point in the point cloud
#     maxpnt          =   disvec.argmax()
#     maxdis          =   disvec[ maxpnt ]
#     maxdiscurve     =   [maxdis] # maxdiscurve[p] = max distance from any point to the first p+1 points of the permutation
#     greedyperm      =   [0, maxpnt]      

#     while maxdis > epsilon:
#         disvec      =   np.minimum( disvec, get_disvec(dissimilarity_matrix, maxpnt)  ) # smallest distance to any point in our net
#         maxpnt      =   np.argmax( disvec )
#         maxdis      =   disvec[ maxpnt ]
#         greedyperm.append(maxpnt)
#         maxdiscurve.append(maxdis)

#     return greedyperm, maxdiscurve

# def farthest_point_sampling_for_cloud_fixed_cardinality( points, num_points_to_sample=1 ):
#     """
#     Applies farthest point sampling to a point cloud.   
    
#     Farthest point sampling is an iterative procedure that builds a sequence of points as follows.

#     1. Start by selecting an initial index :math:`p_0 = 0` (the first row/column).
#     Let :math:`e_0 = \max \{ D_{p_0,p_i} : i > 0 \}` be the maximum distance from point :math:`p_0` to any other point.

#     2. At each step, choose :math:`p_m` as far as possible from :math:`p_0, .., p_{m-1}`.
#        That is, we choose :math:`p_m ∉ \{p_0, \ldots, p_{m-1}\}` such that the scalar value
#        :math:`e_m: = \min \{ D_{p_i,p_m} : i < m \}` is as large as possible.

#     3. Repeat until `m = N`, where :math:`N+1` is the number of points to sample.

#     This produces:
#       - a sequence of indices :math:`p_0, p_1, \ldots, p_N` (the selected points, in order)
#       - a sequence of values :math:`e_0, e_1, ..., e_N` (the covering radii at each step)

#     The points indexed by :math:`p_0, p_1, \ldots, p_N` form an `epsilon net <https://en.wikipedia.org/wiki/Delone_set>`__ for epsilon = :math:`e_N`.
#     Moreover, for each :math:`m`, the points indexed by :math:`\{p_0, ..., p_m\}` form an epsilon net for epsilon = :math:`e_m`.


#     Parameters
#     ----------
#     points : array-like
#         A point cloud represented as a list of points, e.g., a list of tuples or a numpy array of shape (num_points, dimension).
#         Each slice ``points[i]`` will be treated as a point.
#     num_points_to_sample : int
#         The number of points to sample from the point cloud. This must be less than or equal to the number of points in the points.

#     Returns
#     -------
#     greedyperm : list of int
#         The list of indices :math:`[p_0, p_1, \ldots, p_N]`, where `N+1` is the number of points to sample.

#     maxdiscurve : list of float
#         The list of covering radii :math:`[e_0, e_1, \ldots, e_N]`, where `N+1` is the number of points to sample.

        
#     Notes
#     -----
#         - If the input points is empty, `([], [])` is returned.
#         - If the input points has only one point, `([0], [0])` is returned.    


#     Raises
#     -----------
#     Exception
#         If the number of points to sample exceeds the number of points in the points, an exception is raised.


#     See also
#     --------
#     :py:class:`oat_python.dissimilarity.farthest_point_sampling_for_points`
#         Similar to this function, but samples points until the covering radius :math:`e_m` is less than or equal to a given epsilon value.
#     :py:class:`oat_python.dissimilarity.farthest_point_sampling_for_matrix`
#         This function can be applied to the distance matrix :math:`D` of a point cloud to get the same (up to numerical error) result as :py:class:`oat_python.dissimilarity.farthest_point_sampling_for_points`.
#         However, if the number or points is large, then computing and storing :math:`D` in memory can be prohibitively slow and memory-intensive.
#     """
#     if points.shape[0] == 0:
#         return [], []
#     if points.shape[0] == 1:
#         return [0], [0]

#     if num_points_to_sample > points.shape[0]:
#         Exception("Error: net cardinality cannot exceed the number of points")


#     # entry j of this vector represents the min distance from point j to any other point in the point cloud    
#     disvec      =   sklearn.metrics.pairwise_distances( points[[0],:], points )
#     maxpnt      =   np.argmax( disvec )
#     maxdis      =   disvec[ maxpnt ]
#     maxdiscurve =   [maxdis] # maxdiscurve[p] = max distance from any point to the first p+1 points of the permutation
#     greedyperm  =   [0, maxpnt]    

#     for _ in range( num_points_to_sample-1 ):
#         disvec_marg =   sklearn.metrics.pairwise_distances( points[[maxpnt],:], points )
#         disvec      =   np.minimum( disvec, disvec_marg  ) # smallest distance to any point in our net
#         maxpnt      =   np.argmax( disvec )
#         maxdis      =   disvec[ maxpnt ]
#         greedyperm.append(maxpnt)
#         maxdiscurve.append(maxdis)

#     return greedyperm, maxdiscurve 


# def farthest_point_sampling_for_points( points, epsilon=0 ):
#     """
#     Applies farthest point sampling to a point cloud to obtain an :math:`\epsilon`-net.   
    
#     Farthest point sampling is an iterative procedure that builds a sequence of points as follows.

#     1. Start by selecting an initial index :math:`p_0 = 0` (the first row/column).
#     Let :math:`e_0 = \max \{ D_{p_0,p_i} : i > 0 \}` be the maximum distance from point :math:`p_0` to any other point.

#     2. At each step, choose :math:`p_m` as far as possible from :math:`p_0, .., p_{m-1}`.
#        That is, we choose :math:`p_m ∉ \{p_0, \ldots, p_{m-1}\}` such that the scalar value
#        :math:`e_m: = \min \{ D_{p_i,p_m} : i < m \}` is as large as possible.

#     3. Repeat until :math:`e_m` is less than or equal to :math:`\epsilon`.

#     This produces:
#       - a sequence of indices :math:`p_0, p_1, \ldots, p_N` (the selected points, in order)
#       - a sequence of values :math:`e_0, e_1, ..., e_N` (the covering radii at each step)

#     The points indexed by :math:`p_0, p_1, \ldots, p_N` form an :math:`\epsilon`-net (`see Wikipedia <https://en.wikipedia.org/wiki/Delone_set>`__).
#     Moreover, for each :math:`m`, the points indexed by :math:`\{p_0, ..., p_m\}` form an :math:`(e_m)`-net.


#     Parameters
#     ----------
#     points : array-like
#         A point cloud represented as a list of points, e.g., a list of tuples or a numpy array of shape (num_points, dimension).
#         Each slice ``points[i]`` will be treated as a point.
#     num_points_to_sample : int
#         The number of points to sample from the point cloud. This must be less than or equal to the number of points in the points.

#     Returns
#     -------
#     greedyperm : list of int
#         The list of indices :math:`[p_0, p_1, \ldots, p_N]`, where `N+1` is the number of points to sample.

#     maxdiscurve : list of float
#         The list of covering radii :math:`[e_0, e_1, \ldots, e_N]`, where `N+1` is the number of points to sample.

        
#     Notes
#     -----
#         - If the input points is empty, `([], [])` is returned.
#         - If the input points has only one point, `([0], [0])` is returned.    


#     Raises
#     -----------
#     Exception
#         If the number of points to sample exceeds the number of points in the points, an exception is raised.


#     See also
#     --------
#     :py:class:`oat_python.dissimilarity.farthest_point_sampling_for_points`
#         Similar to this function, but samples points until the covering radius :math:`e_m` is less than or equal to a given epsilon value.
#     :py:class:`oat_python.dissimilarity.farthest_point_sampling_for_matrix`
#         This function can be applied to the distance matrix :math:`D` of a point cloud to get the same (up to numerical error) result as :py:class:`oat_python.dissimilarity.farthest_point_sampling_for_points`.
#         However, if the number or points is large, then computing and storing :math:`D` in memory can be prohibitively slow and memory-intensive.
#     """
#     if points.shape[0] == 0:
#         return np.zeros((0,0))
#     if points.shape[0] == 1:
#         return [0], 0
    
#     # entry j of this vector represents the min distance from point j to any other point in the point cloud    
#     disvec      =   sklearn.metrics.pairwise_distances( points[[0],:], points )[0]
#     maxpnt      =   np.argmax( disvec )
#     maxdis      =   disvec[ maxpnt ]      
#     maxdiscurve =   [maxdis] # maxdiscurve[p] = max distance from any point to the first p+1 points of the permutation
#     greedyperm  =   [0, maxpnt]    

#     while maxdis > epsilon:
#         disvec_marg =   sklearn.metrics.pairwise_distances( points[[maxpnt],:], points )[0]
#         disvec      =   np.minimum( disvec, disvec_marg  ) # smallest distance to any point in our net
#         maxpnt      =   np.argmax( disvec )
#         maxdis      =   disvec[ maxpnt ]
#         greedyperm.append(maxpnt)
#         maxdiscurve.append(maxdis)

#     return greedyperm, maxdiscurve    

# ==================================================
# END SECTION: DEPRECATED, OK TO DELETE
# ==================================================




#   ==================================================
#   POINT CLOUD
#   ==================================================





def euclidean_distance_one_sided( pointa, pointb ):
    """
    Returns the Euclidean distance between two points, but reversing point order may yield different results due to numerical error.

    Parameters
    ----------
    pointa : array-like
        The first point.
    pointb : array-like
        The second point.

    Returns
    -------
    float
        The Euclidean distance from ``pointa`` to ``pointb``.
    """

    if np.prod(np.shape(pointa)) != np.prod(np.shape(pointb)):
        raise Exception("Error:points must have equal numbers of elements")

    sum_of_square_differences = 0
    for (p,q) in zip( np.nditer(pointa), np.nditer(pointb) ):
        sum_of_square_differences += ( p - q )**2
    
    return np.sqrt( sum_of_square_differences )  

def euclidean_distance( pointa, pointb ):
    """
    Returns the Euclidean distance between two points.

    To avoid some inconveniences that arise from numerical error, this function
    calculates ``euclidean_distance_one_sided( pointa, pointb )`` and ``euclidean_distance_one_sided( pointb, pointa )``,
    and returns the maximum of the two values.

    Parameters
    ----------
    pointa : array-like
        The first point.
    pointb : array-like
        The second point.

    Returns
    -------
    float
        The Euclidean distance from ``pointa`` to ``pointb``.
    """
    return max( euclidean_distance_one_sided(pointa,pointb), euclidean_distance_one_sided(pointb,pointa) )

def distance_iterator( points, reference_point ):
    """
    Iterates over the Euclidean distances from a reference point to every point in a point cloud.

    Parameters
    ----------
    points : array-like
        The point cloud, as a list of points or a numpy array of shape (num_points, dimension).
    reference_point : array-like
        The reference point from which distances are measured.

    Yields
    ------
    float
        The Euclidean distance from the reference point to each point in the points.
    """
    for point in points:
        if np.prod(np.shape(reference_point)) != np.prod(np.shape(points)[1:]):
            raise Exception("Error: a point in the point cloud does not have the same dimension as the reference point")        
        yield euclidean_distance( point, reference_point )  


#   DEVELOPERS; THIS WAS ORIGINALLY INTENDED TO CIRCUMVENT THE PROBLEM OF NUMERICAL ERROR IN sklearn's radius_neighbors
#               FUNCTION, BUT THAT FUNCTION USES SPECIAL METHODS TO ACCELERATE COMPUTATION, WHICH WE MAY NOT BE ABLE TO
#               MATCH; WE'LL HAVE TO FINISH DEFINING THIS FUNCTION, AND SEE
#   
def sparse_matrix_for_cloud_slow( points, max_dissimilarity ):
    """
    Returns a sparse symmetric Euclidean distance matrix, where entries strictly above ``max_dissimilarity`` are not explicitly stored.

    See also
    --------
    :py:func:`oat_python.dissimilarity.sparse_matrix_for_points`
        Returns a sparse Euclidean distance matrix for a point cloud using a fast nearest-neighbors approach.
        The output specifically to be compatible with the OAT persistent homology solver, and this function
        **typically executes much faster than :func:`sparse_matrix_for_cloud_slow`.**
        
        **However** the output of this function is typically slightly different from :func:`sparse_matrix_for_cloud_slow`,
        due to numerical error. Therefore calling :py:func:`oat_python.dissimilarity.sparse_matrix_for_points` with
        ``max_dissimilarity`` equal to the enclosing radius of the point cloud may yield a dissimilarity matrix whose
        persistent homology is **not equal** to that of the point cloud. This can be remedied by adding a tiny margin of
        error to the enclosing radius, e.g. ``max_dissimilarity = enclosing_radius + 0.00000001``.

    Parameters
    ----------
    points : array-like
        A point cloud represented as a list of points, e.g., a list of tuples or a numpy array of shape (num_points, dimension).
    max_dissimilarity : float
        A non-negative real number; all distances with value above this threshold are dropped from the (sparse) distance matrix.
        If ``max_dissimilarity`` is negative, then an empty sparse matrix of appropriate size is returned.

    Returns
    -------
    scipy.sparse.csr_matrix
        A sparse matrix of shape (num_points, num_points) representing the pairwise distances between points in the points.
        The matrix is strictly symmetric (it exactly equals its transpose), and all entries with value strictly greater than ``max_dissimilarity`` are not explicitly stored.
        If ``max_dissimilarity`` is negative, an empty sparse matrix of appropriate size is returned.
    """

    if np.ndim(points) ==0:
        Exception("The input to ``distance_matrix`` must be an array-like object of dimension at least 1")
    
    shape = np.shape( points )
    if max_dissimilarity < 0:
        return scipy.sparse.csr_matrix((shape[0],shape[0]))
    
    data    =   []
    row     =   []
    col     =   []
    for row_num in range(shape[0]):
        data.append(0); row.append(row_num); col.append(row_num);
        row_vec = points[row_num]
        for col_num in range(row_num+1, shape[0]):
            x = euclidean_distance( row_vec, points[col_num] )
            if x <= max_dissimilarity:
                data.append(x)  # add an entry
                row.append(row_num)
                col.append(col_num)
                data.append(x)  # add its transpose
                row.append(col_num)
                col.append(row_num)   
    matrix  =   scipy.sparse.coo_matrix( 
                    (
                        data, ( row, col, ),
                    ),
                    shape = (shape[0],shape[0]),                    
                )
    matrix  =   scipy.sparse.csr_matrix(matrix)
    matrix.sort_indices()
    return matrix
            



def hop_distance_for_networkx_graph(G):
    """
    Compute the hop (shortest path) distance matrix for a NetworkX graph.

    Parameters
    ----------
    G : networkx.Graph

    Returns
    -------
    D : numpy.ndarray
        A 2D NumPy array of shape (n_vertices, n_vertices), where D[i, j] is the hop distance (minimum number of edges)
        between vertex i and vertex j, using the order of `vertex_labels`.
    vertex_labels : list
        A list of vertex labels corresponding to the order of rows and columns in D.

    Notes
    -----
    - The hop distance between two vertices is the length of the shortest path connecting them.
    - The order of vertices in `vertex_labels` matches the row and column order in D.

    Example
    -------
    .. code-block:: python

        import networkx as nx
        from oat_python.plot import hop_distance_for_networkx_graph

        G = nx.path_graph([0, 1, 2, 3])
        D, labels = hop_distance_for_networkx_graph(G)
        print("Labels:", labels)
        print("Distance matrix:\\n", D)
    """
    hop_distances = dict(nx.shortest_path_length(G))
    n_vertices = G.number_of_nodes()
    vertex_labels = [x for x in hop_distances.keys() ]
    D  = np.full((n_vertices, n_vertices), np.inf)
    for vertex in range(n_vertices):
        for neighbor in range(vertex,n_vertices):
            D[vertex][neighbor] = hop_distances[ vertex_labels[vertex]  ][ vertex_labels[neighbor] ]
            D[neighbor,vertex] = D[vertex,neighbor]
    return D, vertex_labels




def hop_distance_for_simpices(simplices):
    """
    Compute the hop (shortest path) distance matrix for a collection of simplices.

    Parameters
    ----------
    simplices : iterable of iterables of integers
        A list of simplices, where each inner iterable contains vertex labels (hashable types).

    Returns
    -------
    D : numpy.ndarray
        A 2D NumPy array of shape (n_vertices, n_vertices), where D[i, j] is the hop distance (minimum number of edges)
        between vertex i and vertex j, using the order of `vertex_labels`.
    vertex_labels : list
        A list of vertex labels corresponding to the order of rows and columns in D.

    Notes
    -----
    - The hop distance between two vertices is the length of the shortest path connecting them.
      Two vertices are considered adjacent if they are both contained in one or more of the same simplices.
    - The order of vertices in `vertex_labels` matches the row and column order in D.

    Example
    -------
    .. code-block:: python

        import networkx as nx
        from oat_python.plot import hop_distance_for_networkx_graph

        G = nx.path_graph([0, 1, 2, 3])
        D, labels = hop_distance_for_networkx_graph(G)
        print("Labels:", labels)
        print("Distance matrix:\\n", D)
    """

    G = oat_python.simplex.networkx_graph_from_simplices(simplices)
    return hop_distance_for_networkx_graph(G)




















#   ===================




def validate_dissimilarity_matrix(matrix):
    """
    Validates that the input matrix is a valid :term:`sparse<sparse dissimilarity matrix>` or :term:`dense<dissimilarity matrix>`
    dissimilarity matrix.

    Parameters
    ----------
    matrix : numpy.ndarray or scipy.sparse.csr_matrix
        The matrix to validate.

    Raises
    ------
    Exception
        
        - If the matrix is not 2-dimensional, not square, or not symmetric.
        - If the matrix contains a row with one or more structural nonzero entries, 
          but the diagonal entry for that row is structurally zero.
    """

    oat_python.matrix.validate_square_and_symmetric_matrix( matrix )
    

    if isinstance(matrix, np.ndarray):
        for row_index in range(matrix.shape[0]):
            row = matrix[row_index]
            if row[ row_index ] != np.min(row):
                raise Exception(f"Entry ({row_index}, {row_index}) does not contain the minimum value in row {row_index}.")

    if isinstance(matrix, scipy.sparse.csr_matrix):
        for row_index in range(matrix.shape[0]):
            row = matrix[row_index]
            if len(row.data) > 0:
                if row_index not in row.indices:
                    raise Exception(f"Row {row_index} contains an explicitly stored entry, but no explicitly stored entry in column {row_index}.")
                if matrix[row_index, row_index] > np.min(row.data):
                    raise Exception(f"Entry ({row_index}, {row_index}) contains an explicit entry, but it's not the smallest explicit entry in row {row_index}.")




def test_validate_dissimilarity_matrix():
    """
    Unit tests for the `validate_dissimilarity_matrix` function.

    This test suite covers:
    - Dense and sparse matrices that are valid (square, symmetric, correct diagonal).
    - Matrices that are not square.
    - Matrices that are not symmetric.
    - Sparse matrices with a row that has structural nonzero entries but no diagonal entry.
    - Sparse matrices where the diagonal is not the minimum in its row.

    The test asserts that valid matrices pass and invalid matrices raise an Exception.
    """

    # Valid dense
    arr = np.array([[0, 1], [1, 0]])
    validate_dissimilarity_matrix(arr)

    # Valid sparse
    data = np.array([0, 1, 1, 0])
    row = np.array([0, 0, 1, 1])
    col = np.array([0, 1, 0, 1])
    sp = scipy.sparse.csr_matrix((data, (row, col)), shape=(2, 2))
    validate_dissimilarity_matrix(sp)

    # Not square
    arr2 = np.ones((2, 3))
    try:
        validate_dissimilarity_matrix(arr2)
        assert False, "Should have raised Exception for non-square"
    except Exception:
        pass

    # Not symmetric
    arr3 = np.array([[0, 1], [2, 0]])
    try:
        validate_dissimilarity_matrix(arr3)
        assert False, "Should have raised Exception for non-symmetric"
    except Exception:
        pass

    # Sparse: row with structural nonzero but no diagonal entry
    data = np.array([1,1])
    row = np.array([0,1])
    col = np.array([1,0])
    sp2 = scipy.sparse.csr_matrix((data, (row, col)), shape=(2, 2))
    try:
        validate_dissimilarity_matrix(sp2)
        assert False, "Should have raised Exception for missing diagonal"
    except Exception:
        pass

    # Sparse: diagonal not minimum in row
    data = np.array([5, 1, 1])
    row = np.array([0, 0, 1])
    col = np.array([0, 1, 0])
    sp3 = scipy.sparse.csr_matrix((data, (row, col)), shape=(2, 2))
    try:
        validate_dissimilarity_matrix(sp3)
        assert False, "Should have raised Exception for diagonal not minimum"
    except Exception:
        pass

    print("All validate_dissimilarity_matrix tests passed.")





def test_dissimilarity_matrix(max_grid_size):
    """
    Tests the correctness and consistency of various dissimilarity matrix formatting and radius calculation functions
    for grids of increasing size.

    For each grid size from 0 to max_grid_size (inclusive), this test:
        - Generates a 2D grid of points ("points").
        - Computes the pairwise distance matrix (dense and sparse formats).
        - Formats the data into cleaned CSR matrices using multiple methods:
            - From the points (slow and fast/NN implementations).
            - From the CSR matrix.
            - From the dense matrix.
        - Asserts that all formatted matrices are approximately equal.
        - Computes the enclosing radius using multiple methods and asserts their equivalence.
        - Tests sparsified versions of the matrices for a variety of thresholds, ensuring consistency across methods.

    Parameters
    ----------
    max_grid_size : int
        The maximum size of the grid (number of points along one axis) to test.

    Raises
    ------
    AssertionError
        If any of the formatted matrices or computed radii are not approximately equal across methods.
    """


    for grid_size in range(max_grid_size + 1):
        # generate an n x n gird of points
        x,y                         =   np.meshgrid( np.arange(grid_size), np.arange(grid_size) )
        points                      =   np.column_stack((x.ravel(), y.ravel()))

        # calculate the distance matrix, and format as a csr matrix; we regard these two matrices as "raw data"
        if grid_size > 0:
            dissimilarity_matrix_dense            =   sklearn.metrics.pairwise_distances(points)
            dissimilarity_matrix_dense            =   np.maximum( dissimilarity_matrix_dense, dissimilarity_matrix_dense.T) # ensure the matrix is symmetric
        else:
            dissimilarity_matrix_dense            =   np.zeros((0,0))
        dissimilarity_matrix_csr                  =   scipy.sparse.csr_matrix(dissimilarity_matrix_dense)
        dissimilarity_matrix_csr.setdiag(0)

        # format the raw data from the points, dense, and csr matrices into "cleaned" csr matrices
        formatted_for_points        =   sparse_matrix_for_cloud_slow(          points       , max_dissimilarity=inf    )    
        formatted_for_cloud_nn     =   sparse_matrix_for_points(   points       , max_dissimilarity=inf    )
        formatted_for_csr          =   sparse_matrix_for_csr(            dissimilarity_matrix_csr  , max_dissimilarity=inf    )    
        formatted_for_dense        =   sparse_matrix_for_dense(          dissimilarity_matrix_dense, max_dissimilarity=inf    )

        # check that these matrices are all approximately equal
        matrices = [formatted_for_points, formatted_for_cloud_nn, formatted_for_csr, formatted_for_dense ]
        for (inda, a), (indb, b) in itertools.product( enumerate(matrices), enumerate(matrices) ):
            oat_python.matrix.assert_almost_equal_csr( a, b, decimal=10, err_msg=f"inda: {inda}, indb: {indb}" )

        # generate a list of distance upper bounds, and check that they are similar to one another       

        radius_cloud_slow           =   enclosing_radius_for_cloud_slow(points)
        radius_points                =   enclosing_radius_for_points(points)
        radius_csr                  =   enclosing_radius_for_matrix(dissimilarity_matrix_csr)
        radius_dense                =   enclosing_radius_for_matrix(dissimilarity_matrix_dense)

        np.testing.assert_almost_equal( radius_cloud_slow, radius_points,            decimal=10 )
        np.testing.assert_almost_equal( radius_cloud_slow, radius_csr,              decimal=10 )
        np.testing.assert_almost_equal( radius_cloud_slow, radius_dense,            decimal=10 )    

        # generate some sparsified matrices, and check that they are similar
        thresholds      =\
        [ radius_cloud_slow, radius_points, radius_csr, radius_dense    ] +\
        [ 0,  1,  2,  3,  4,  5,  inf               ] +\
        [ 0, -1, -2, -3, -4, -5, -inf               ] +\
        list( np.unique(dissimilarity_matrix_dense)               ) +\
        list( dissimilarity_matrix_csr.data                       ) +\
        list( formatted_for_points.data             ) +\
        list( formatted_for_points.data             ) +\
        list( formatted_for_cloud_nn.data          ) +\
        list( formatted_for_csr.data               ) +\
        list( formatted_for_dense.data             )   

        for threshold in thresholds:
            formatted_for_points        =   sparse_matrix_for_cloud_slow(     points,                      max_dissimilarity=threshold    )    
            formatted_for_cloud_nn     =   sparse_matrix_for_points(          points,                      max_dissimilarity=threshold + 0.00000000001    )
            formatted_for_csr          =   sparse_matrix_for_csr(            dissimilarity_matrix_csr  , max_dissimilarity=threshold    )    
            formatted_for_dense        =   sparse_matrix_for_dense(          dissimilarity_matrix_dense, max_dissimilarity=threshold    )        

            matrices = [formatted_for_points, formatted_for_cloud_nn, formatted_for_csr, formatted_for_dense ]
            for (inda,a), (indb,b) in itertools.product(enumerate(matrices), enumerate(matrices)):
                oat_python.matrix.assert_almost_equal_csr( a, b, decimal=10, err_msg=f"inda: {inda}, indb: {indb}")

    print("test passed")




