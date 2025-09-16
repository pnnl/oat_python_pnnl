"""
.. _dowker_rbs_homology_gallery:

Restricted Barycentric Subdivision (Vietoris-Rips)
===============================================================

The Restricted Barycentric Subdivision Homology of a hypergraph is the homology of the clique complex of the simple graph :math:`G` such that

- the vertices of :math:`G` are the hyperedges of the hypergraph
- the edges of :math:`G` are the pairs :math:`\{e,h\}` such that :math:`e \subseteq h` or :math:`h \subseteq e`

OAT doesn't offer a specialized function to compute this homology, but it does allow one to compute the *persistent homology* of any weighted graph. So, as a work around, we assign weight 0 to every edge of `G` and compute its persistent homology.

This notebook shows how to implement this strategy.
"""

# %%
import oat_python as oat

import copy
import plotly.graph_objects as go
import numpy as np
import networkx as nx
import hypernetx as hnx
import sklearn

# %% [markdown]
# Define a hypergraph
# -----------------------------------------

# %%
# The hypergraph has edges labeled "A", "B", "C", "D", with the following vertices:

E = { "A": ["x"], "B": ["y"], "C": ["x","y","z",], "D": ["x","y","w",], }

# %% [markdown]
# Plot the hypergraph using HyperNetX
# -----------------------------------------

# %%
hnxgraph = hnx.Hypergraph(E)
hnx.drawing.draw(hnxgraph)

# %% [markdown]
# Compute homology
# -----------------------------------------

# %%

# convert to a list of edges
G                               =   list( E.values() )
# graph representing the edge containment poset (forgetting direction)
containment                     =   oat.hypergraph.edge_containment_graph_symmetrized( G )
# graph whose edges form the set complement of the containment graph
anticontainment                 =   nx.complement( containment )
# adjacency matrix of the anticontainment graph
anti_adjacency                  =   nx.adjacency_matrix( anticontainment ).todense()
dissimilarity_matrix            =   oat.dissimilarity.sparse_matrix_for_dense(
                                        dissimilarity_matrix    =   anti_adjacency,
                                        max_dissimilarity       =   0.5
                                    )

# decomposition boundary matrix
decomposition                    =  oat.core.vietoris_rips.BoundaryMatrixDecomposition(
                                        dissimilarity_matrix    =   dissimilarity_matrix,
                                        max_homology_dimension  =   1, 
                                    )

# %%
# The persistent homology dataframe contains the persistent homology of the Vietoris-Rips complex.
# Because every edge in the underlying graph has weight 0, this is the same as the RBS homology of the hypergraph.
# The dataframe contains a cycle basis for the RBS homology, in which each cycle technically
# represets a homology class that is born at filtration parameter 0 and never dies.
persistent_homology             =   decomposition.persistent_homology_dataframe(
                                        return_cycle_representatives    =   True,
                                        return_bounding_chains          =   True,
                                    )
persistent_homology

# %% [markdown]
# Inspect a cycle representative
# -----------------------------------------

# %%
cycle           =   persistent_homology["cycle_representative"][1]
cycle

# %% [markdown]
# Relabel each vertex 
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Recall that a vertex in the Vietoris-Rips complex corresponds to a hyperedge in the initial hypergraph.
# These hyperedges have labels "A", "B", "C", "D", so lets relabel the vertices in the cycle representative accordingly.


# %% [markdown]
# Recall that each vertex in RBS homology represents an edge in the reduced hypergraph (which corresponds to a set of hyperedges in the initial hypergraph).  Here we relabel each vertex with **one** of the edges that maps to it.

# %%

original_edge_labels=   { p: k for p,k in enumerate(E.keys()) }

cycle               =   copy.deepcopy(persistent_homology["cycle_representative"][1])
cycle["simplex"]    =   [ [original_edge_labels[x] for x in simplex] for simplex in cycle["simplex"] ]

cycle

# %% [markdown]
# Plot a cycle representative
# -----------------------------------------
# 

# %%
edges                   =   persistent_homology["cycle_representative"][1]["simplex"] # the edges in the cycle
points                  =   oat.plot.vertex_embedding_for_simplices(
                                simplices           =   edges.tolist(),
                                dimension           =   3,
                            ) 

fig                     =   oat.plot.fig_3d_for_simplices( 
                                edges, 
                                points=points,
                                kwargs_points       =   dict(
                                                            mode="markers+text", 
                                                            text= [ val for val in original_edge_labels.values() ],
                                                            marker_size=8,
                                                            textfont_size=18,
                                                        ),                                
                            )
fig.update_layout(template="plotly_dark")



# %% [markdown]
# Suspend and repeat
# ---------------------

# %%
F = { "A": ["x"], "B": ["y"], "C": ["x","y","z",], "D": ["x","y","w",], "E":["x","y","z","w","a"], "F":["x","y","z","w","b"] }
hnxgraph = hnx.Hypergraph(F)
hnx.drawing.draw(hnxgraph)

# %%
# convert to a list of edges
G                               =   list( F.values() )
# graph representing the edge containment poset (forgetting direction)
containment                     =   oat.hypergraph.edge_containment_graph_symmetrized( G )
# graph whose edges form the set complement of the containment graph
anticontainment                 =   nx.complement( containment )
# adjacency matrix of the anticontainment graph
anti_adjacency                  =   nx.adjacency_matrix( anticontainment ).todense()
dissimilarity_matrix            =   oat.dissimilarity.sparse_matrix_for_dense(
                                        dissimilarity_matrix    =   anti_adjacency,
                                        max_dissimilarity       =   0.5
                                    )

# decomposition boundary matrix
decomposition                   =   oat.core.vietoris_rips.BoundaryMatrixDecomposition(
                                        dissimilarity_matrix    =   dissimilarity_matrix,
                                        max_homology_dimension  =   2, 
                                    )

# %%
# The persistent homology dataframe:
persistent_homology             =   decomposition.persistent_homology_dataframe(
                                        return_cycle_representatives    =   True,
                                        return_bounding_chains          =   True,
                                    )
persistent_homology

# %%
# A cycle representative:
cycle                       =   persistent_homology["cycle_representative"][1]
cycle

# %%
# A cycle with reverted vertex labels
original_edge_labels                       =   { p: k for p,k in enumerate(F.keys()) }
cycle                       =   copy.deepcopy(persistent_homology["cycle_representative"][1])
cycle["simplex"]            =   [ [original_edge_labels[x] for x in simplex] for simplex in cycle["simplex"] ]
cycle


# %%
# Plot the cycle representative
triangles               =   persistent_homology["cycle_representative"][1]["simplex"] # the edges in the cycle
points                  =   oat.plot.vertex_embedding_for_simplices(
                                simplices           =   triangles.tolist(),
                                dimension           =   3,
                            ) 

fig                     =   oat.plot.fig_3d_for_simplices( 
                                triangles, 
                                points=points,
                                kwargs_points       =   dict(
                                                            mode="markers+text", 
                                                            text= [ val for val in original_edge_labels.values() ],
                                                            marker_size=8,
                                                            textfont_size=18,
                                                        ),                                
                            )
fig.update_layout(template="plotly_dark")



# %% [markdown]
# Make triangles opaque and grey, for a different style.

# %%
# The trace for triangles is stored in fig.data[2]
fig.data[2].update( color = 'grey', opacity = 1.0)
fig


