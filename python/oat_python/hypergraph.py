"""
Tools for working with hypergraphs.
"""

import copy
from fractions import Fraction
import numpy as np
import networkx as nx
import scipy

from typing import Dict, List, Any







def unique_elements(L):
    """
    Returns a list of the unique elements of a list L.
    """
    unique_elements = []
    for l in L:
        if not l in unique_elements:
            unique_elements.append(l)
    return unique_elements


def unique_values(key_val_iter):
    """
    Returns a pair of lists U and P such that P[i] equals 
    the set of all keys with value U[i], and U has no 
    repeated entries.
    """
    unique_values = []
    key_partition = []
    for k, v in key_val_iter:
        duplicate_v = False
        for ind, val in enumerate(unique_values):
            if v == val:
                key_partition[ind].append(k)
                duplicate_v = True
                break
        if not duplicate_v:
            unique_values.append(v)
            key_partition.append([k])
    return unique_values, key_partition

def containment_map( list_of_collections ):
    """
    Given a list of colections with pairwise-empty intersections, denoted `list_of_collections`, return the dictionary `D` such that :math:`k \\in list_of_collections[ D[k] ]` for all `k`.
    """
    return { k: n for n, collection in enumerate(list_of_collections) for k in collection }

def transpose(L):
    """
    Transpose a list of lists of integers, L, producing a list of lists P
    such that L[i][j] = P[j][i]
    """
    transpose = []
    for lnum, l in enumerate(L):
        for v in l:
            while len(transpose) < v + 1:
                transpose.append([])
            transpose[v].append(lnum)
    return transpose



#   =========================================================
#   RELABELING
#   =========================================================


def relabel( hgdict: Dict[Any, List[Any]] ):
    """
    Relabel the nodes and edges of a hypergraph with consecutive integers.

    Given a hypergraph represented as a dictionary of lists, where each list is sorted in strictly ascending order,
    this function returns a copy of the hypergraph with nodes and edges relabeled with integers, and a dictionary
    containing the relabeling information.

    Parameters
    ----------
    hgdict : dict of list
        A hypergraph H represented by a dictionary of lists. Each list must be sorted in strictly ascending order.

    Returns
    -------
    H : list of list of int
        A copy of `hgdict` with nodes and edges relabeled with integers; formatted as a list of sorted lists.
    T : dict
        A dictionary containing relabeling information with the following keys:

            - 'new_edge_for_old_edge': maps original edge labels to new integer labels
            - 'new_node_for_old_node': maps original node labels to new integer labels
            - 'old_edge_for_new_edge': maps new integer edge labels to original edge labels
            - 'old_node_for_new_node': maps new integer node labels to original node labels

    Notes
    -----
    
    - `T["new_edge_for_old_edge"][old_edge]` gives the integer label of the edge in `H` corresponding to `old_edge`.
    - The other entries of `T` are defined similarly.
    """
    hgdict = copy.deepcopy(hgdict)

    # Check if the input is a dictionary
    if not isinstance(hgdict, dict):
        raise TypeError("Input must be a dictionary of sorted lists.")
    
    # Check if each value in the dictionary is a sorted list
    for edge_label, edge_vertices in hgdict.items():
        # Check if the value is a list
        if not isinstance(edge_vertices, list):
            raise TypeError(f"The value for key '{edge_label}' must be a list.")

        # Check if the list is sorted
        for i in range(len(edge_vertices) - 1):
            if not (edge_vertices[i] < edge_vertices[i + 1]):
                raise ValueError(f"The vertex list for hyperedge '{edge_label}' must be sorted in strictly ascending order, however edge {edge_label} has consecutive vertices {edge_vertices[i]} and {edge_vertices[i + 1]}.")        

    # relabel nodes by integers (this does note remove "parallel" vertices, meaning nodes that are in the exact same set of hyperedges)
    vertices    =   [x for l in hgdict.values() for x in l]
    nvl2ovl     =   sorted( list(set(vertices)) ) # new vertex label 2 old vertex label; sorting means that the map preserves order
    ovl2nvl     =   { key: ordinal for (ordinal,key) in enumerate(nvl2ovl) } # old vertex label 2 new vertex label

    nel2oel     =   list( hgdict.keys() )
    oel2nel     =   { k: p for p, k in enumerate( nel2oel ) }

    hypergraph  =   [ [ovl2nvl[v] for v in l ] for k, l in hgdict.items() ]
   
    translator  =   dict(
                            new_node_for_old_node    =   ovl2nvl,                            
                            new_edge_for_old_edge    =   oel2nel,
                            old_node_for_new_node    =   nvl2ovl,                             
                            old_edge_for_new_edge    =   nel2oel,
                        )
    
    return hypergraph, translator


#   =========================================================
#   REDUCTION
#   =========================================================


def reduce_hypergraph_with_labels( hgdict ):
    """
    Given a hypergraph H represented by a dictionary of lists D, 
    - regard D as a binary relation B, and let B' be the relation obtained by deleting duplicate rows and columns
    - relabel the rows and columns of B' with integers
    - let H be the corresponding hypergraph, represented as a list of lists of integers
    Every node (respectively, edge) of B maps to a unique row (respectively, edge) of B'.  Let T be a dictionary with keys 
    `new_edge_for_old_edge`
    `new_node_for_old_node`
    `old_edges_for_new_edge`
    `old_nodes_for_new_node`
    and require `T["new_edge_for_old_edge"][old_edge]` is the integer label of the edge in `H` corresponding to `old_edge`.  The other entries of T are defined similarly.
    
    Return H, T
    """
    hgdict = copy.deepcopy(hgdict)

    # remove duplicate entries in each edge
    for k in hgdict.keys():
        hgdict[k] = list(set(hgdict[k]))  

    # relabel nodes by integers (this does note remove "parallel" vertices, meaning nodes that are in the exact same set of hyperedges)
    vertices = [x for l in hgdict.values() for x in l]
    nvl2ovl = list(set(vertices)) # new vertex label 2 old vertex label
    ovl2nvl  = { key: ordinal for (ordinal,key) in enumerate(nvl2ovl) } # old vertex label 2 new vertex label     
    hg = { k: [ovl2nvl[v] for v in l ] for k, l in hgdict.items() }
    
    # sort the nodes in each hyperedge, and remove duplicate nodes (this is not the same as removing parallel nodes)
    for key,val in hg.items():
        val = list(set(val))
        val.sort()
        hg[key] = val
    
    # unique_hyperedge_slices = the set of all sets of vertices realized by a hyperedge (without repeats; by contrast, in theory there could be multiple hyperedges with the same vertex sets)
    # hyperedge_label_partition[i] = the set of all hyperedge labels with vertex set unique_hyperedge_slices[i]
    unique_hyperedge_slices, hyperedge_label_partition = unique_values(hg.items())
    
    unique_hypernode_slices, hypernode_label_partition \
        = unique_values( enumerate( transpose( unique_hyperedge_slices ) ) )
    
    reduced_hg = transpose( unique_hypernode_slices )
    
    # hypernode_label_partition = [ [nvl2ovl[i] for l in hypernode_label_partition for i in l] ]
    
    nvl2nnvl             = containment_map( hypernode_label_partition ) # new vertex labels to new new vertex labels
    new_node_for_old_node = { k: nvl2nnvl[v] for k, v in ovl2nvl.items() }    
    new_edge_for_old_edge = { k: n for n, partition in enumerate(hyperedge_label_partition) for k in partition }

    old_edges_for_new_edge   =   hyperedge_label_partition
    old_nodes_for_new_node   =   [ [nvl2ovl[x] for x in partition] for partition in hypernode_label_partition ]

    translator  =   dict(
                            new_node_for_old_node    =   new_node_for_old_node, 
                            new_edge_for_old_edge    =   new_edge_for_old_edge,
                            old_nodes_for_new_node   =   old_nodes_for_new_node,
                            old_edges_for_new_edge   =   old_edges_for_new_edge,
                        )
    
    return reduced_hg, translator
    
def reduce_hypergraph(hg):
    """
    Given a list of lists (regarded as the sparsity pattern of
    a sparse 0-1 matrix), returns a list of lists
    representing the sparse matrix obtained by deleting
    duplicate rows and columns
    """
    return transpose( unique_elements( transpose( np.unique( hg, axis=0 ) ) ) )


#   =========================================================
#   EDGE CONTAINMENT
#   =========================================================


def edge_containment_graph( L ):
    """
    Given a length-m list of lists L, returns a networkx **directed** graph G with
    vertex set 0, .., m-1, where there exists a directed edge i->j iff L[i] is a subset
    of L[j]
    """
    m = len(L)    
    S = [set(x) for x in L]
    G = nx.Digraph()
    for vertex in range(m): G.add_node(vertex)
    for row in range(m):
        for col in range(m):
            if S[row].issubset(S[col]):
                G.add_edge( row, col )
    return G

def edge_containment_graph_symmetrized( L ):
    """
    Given a length-m list of lists L, returns a networkx **undirected** graph G with
    vertex set 0, .., m-1, where i and j are adjacent iff L[i] contains L[j] or
    vice-versa
    """
    m = len(L)    
    S = [set(x) for x in L]
    G = nx.Graph()
    for vertex in range(m): G.add_node(vertex)    
    for row in range(m):
        for col in range(m):
            if S[row].issubset(S[col]):
                G.add_edge( row, col )
    return G

def edge_containment_relation( L ):
    """
    Given a length-m list of lists L, returns an m x m matrix adjacency matrix `adj` of
    type `scipy.sparse.csr_matrix` such that
    - adj[i][j] = 1 if L[j] contains L[i]
    - adj[i][j] = 0, otherwise
    """
    m = len(L)
    G = [set(x) for x in L]
    
    # initialized the sparse data of an identity matrix
    rows    =   list(range(m))
    cols    =   list(range(m))
    data    =   [1 for _ in range(m)]

    # fill the matrix
    for row in range(m):
        for col in range(m):
            if row == col:
                continue # we've already taken care of these entries
            if G[row].issubset(G[col]):
                rows.append(row)
                cols.append(col)
                data.append(1)
    return scipy.sparse.csr_matrix((data, (rows, cols)), shape=(m, m))

def edge_containment_relation_symmetrized( L ):
    """
    Given a length-m list of lists L, returns an m x m adjacency matrix `adj` of
    type `scipy.sparse.csr_matrix`, such that
    - adj[i][j] = 1 if L[i] contains L[j] or vice versa
    - adj[i][j] = 0, otherwise
    """
    m = len(L)
    G = [set(x) for x in L]

    # initialized the sparse data of an identity matrix
    rows    =   list(range(m))
    cols    =   list(range(m))
    data    =   [1 for _ in range(m)]

    # fill the matrix
    for row in range(m):
        for col in range(row+1,m):
            if G[row].issubset(G[col]):
                # add an entry
                rows.append(row)
                cols.append(col)
                data.append(1)
                # and its transpose
                rows.append(col)
                cols.append(row)
                data.append(1)                
    return scipy.sparse.csr_matrix((data, (rows, cols)), shape=(m, m))

def networkx_nerve_for_hypergraph( L ):
    """
    Given a length-m list of lists L, returns a networkx graph G with
    - m edges
    - an undirected edge {i,j} iff L[i] is a subset of L[j], or vice versa
    """
    m = len(L)
    S = [set(x) for x in L]    
    G = nx.Graph()   
    for i in range(m):
        G.add_node(i)
    for i in range(m):
        for j in range(i+1,m):
            if S[i].issubset(S[j]) or S[j].issubset(S[i]):
                G.add_edge(i,j)
    return G
