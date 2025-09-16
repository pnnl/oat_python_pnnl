"""
Functions for simplices and their faces.
"""


import itertools
import numpy as np
import networkx as nx



def dimension_m_faces_for_simplices( simplices=[], m=0 ):
    """
    Return all dimension-``m`` faces of the simplices provided

    :param simplices: an iterable of iterables; each sub-iterable represents a simplex
    :param m: dimension of desired faces

    :return faces: a list of tuples representing the collection of dimension-``m`` faces. This list is sorted in lexicographic order.
    """
    faces = set()
    for simplex in simplices:
        for face in itertools.combinations( simplex, m+1 ):
            faces.add( face )    
    return sorted(faces)


def dimension_m_faces_as_rows_for_dimension_n_simplices_as_rows( dimension_n_simplices_as_rows: np.ndarray, m = 0, removeduplicatefaces=True ):
    """
    :param dimension_n_simplices_as_rows: a numpy array where each row represents a simplex 
    :param m: dimension of desired faces
    :param removeduplicatefaces: if True, remove duplicate faces from the output (default True)

    :return faces: a numpy array whose rows represent the dimension-``m`` faces of the simplices. These rows are NOT sorted in lexicographic order.
    """
    if not isinstance(dimension_n_simplices_as_rows, np.ndarray):
        raise TypeError("Input must be a NumPy array.")
    if dimension_n_simplices_as_rows.ndim != 2:
        raise ValueError("Array must have shape (k, n+1) for some k.")

    if len(dimension_n_simplices_as_rows) == 0:
        return []
    
    dimension_n_simplices_as_rows       =   np.array(dimension_n_simplices_as_rows)
    nvertices                           =   len(dimension_n_simplices_as_rows[0])
    subset_indices                      =   itertools.combinations( range(nvertices), m+1 )
    dimension_n_facesasrows             =   [dimension_n_simplices_as_rows[:,I] for I in subset_indices]
    dimension_n_facesasrows             =   np.concatenate(dimension_n_facesasrows, axis=0)

    if removeduplicatefaces:
        dimension_n_facesasrows         =   np.unique(dimension_n_facesasrows, axis=1)

    return dimension_n_facesasrows


def vertices_incident_to_simplices( simplices ):
    """
    Given a list of simplices, return a sorted list of the vertices incident to these simplices.

    :param simplices: an iterable of iterables; each sub-iterable represents a simplex

    :return vertices: a sorted list of the vertices incident to the simplices
    """
    vertices = set()
    for simplex in simplices:
        for vertex in simplex:
            vertices.add(vertex)
    return sorted(vertices)



def networkx_graph_for_simplices(simplices):
    """
    Generate a NetworkX graph representing the 1-skeleton of a simplicial complex.

    Parameters
    ----------
    simplices : list of list of int
        A list where each inner list represents a simplex (a collection of vertices).
        Each vertex should be hashable (typically an int).

    Returns
    -------
    G : networkx.Graph
        A NetworkX graph where nodes are vertices, and edges connect
        any pair of vertices that appear together in a simplex (i.e., the 1-skeleton).

    Example
    -------
    .. code-block:: python

        import networkx as nx
        from oat_python.plot import networkx_graph_for_simplices

        simplices = [[0, 1, 2], [2, 3]]
        G = networkx_graph_for_simplices(simplices)
        print(G.edges())
    """
    G=nx.Graph()
    
    for simplex in simplices:
        for v in simplex:
            G.add_node(v)
        for i,j in itertools.combinations(simplex, 2):
            G.add_edge(i,j)
    return G