"""
.. _vertex_embedding_gallery:

Vertex Embedding
========================================

This tutorial covers vertex embedding to visualize simplices and simplicial complexes.

Most plotting functions assume that the user can provide x-y-z coordinates for each vertex.
However, sometimes data doesn't come with x-y-z coordinates. For example, it might come from a
a graph or hypergraph. In this case, we need to find a way to generate these coordinates.

OAT provides several tools to help with this:

- :func:`oat_python.simplex.networkx_graph_from_simplices` converts a collection of simplices into a NetworkX graph.
  NetworkX has many built-in graph layout algorithms that can be used to generate x-y or x-y-z coordinates for the vertices.
- :func:`oat_python.dissimilarity.hop_distance_for_networkx_graph` computes the hop distance between vertices in a graph.
  The result is a distance matrix that can be used to generate a vertex embedding, e.g. using multidimensional scaling.
- :func:`oat_python.dissimilarity.hop_distance_for_simplices` computes the hop distance between vertices in a collection of simplices.
- :func:`oat.plot.vertex_embedding_for_simplices` Offers two methods, controlled by the keyword `method`:

  - ``spring``: this extracts the underlying graph (consisting of the vertices of the simplices, together with an edge (i,j) for every pair of vertices (i,j) that belong to one or more of the same simplices)
  - ``hop_mds``: this applies multidimensional scaling to the hop distance (i.e. shortest path distance) on the underlying graph.
"""




# %%
# Setup
# -----------------------------------------------

# %%
import oat_python as oat

import networkx as nx
import numpy as np
import plotly.graph_objects as go



# %%
# Define a collection of simplices, whose vertices are (i,j) pairs
N                       =   20
dowker_simplices        =   []
for i in range(N):
    for j in range(N):
        dowker_square   =   [(i,j), ( i+1,j), (i,j+1), ( i+1,j+1)]
        dowker_simplices.append(dowker_square)
dowker_simplices.append( [ (0,j) for j in range(N) ] )    # Join all vertices in the bottom row
dowker_simplices.append( [ (N,j) for j in range(N) ] )  # Join all vertices in the top row        

# %% 
# Relabel the (i,j) pairs as integers
xy_tuples               =   set()
for simplex in dowker_simplices:
    for xy_tuple in simplex:
        xy_tuples.add(xy_tuple)
xy_tuples               =   list(xy_tuples)
vertex_labels           =   {xy_tuple: k for k, xy_tuple in enumerate(xy_tuples)}
simplices               =   [
                                [vertex_labels[xy_tuple] for xy_tuple in simplex]
                                for simplex in dowker_simplices
                            ]


# %%
# Example 1
# -----------------------------------------------


# %%
# Generate a vertex embedding 
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
points                  =   oat.plot.vertex_embedding_for_simplices(
                                simplices,
                                dimension           =   3,
                                method              =   "hop_mds",
                                random_state        =   0,
                            )
points


# %%
# Plot the simplices using the vertex embedding
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
fig                     =   oat.plot.fig_3d_for_simplices(
                                simplices           =   simplices,
                                points              =   points,
                                kwargs_points       =   {"marker": {"size": 3, "color": "white"}},
                                kwargs_edges        =   {"line": {"color": "white", "width": 3}},
                            )
fig.update_layout(
    height=700,
)
fig




# %%
# Example 2
# -----------------------------------------------

# %%
# Modify the set of simplices
for j in range(N):
    dowker_square   =   [(0,j), ( N,j), (0,j+1), ( N,j+1)]
    dowker_square   =   [ vertex_labels[xy_tuple] for xy_tuple in dowker_square ]
    simplices.append(dowker_square)

# %%
# Generate a vertex embedding 
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
points                  =   oat.plot.vertex_embedding_for_simplices(
                                simplices,
                                dimension           =   3,
                                method              =   "hop_mds",
                                random_state        =   0,
                            )
points


# %%
# Plot the simplices using the vertex embedding
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
fig                     =   oat.plot.fig_3d_for_simplices(
                                simplices           =   simplices,
                                points              =   points,
                                kwargs_points       =   {"marker": {"size": 3, "color": "white"}},
                                kwargs_edges        =   {"line": {"color": "white", "width": 3}},
                            )
fig.update_layout(
    height=700,
)
fig


# %%
# Adjust the plot
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

oat.plot.set_background_color(fig, "black")
fig.update_layout(
    scene=dict(
        camera=dict(
            eye=dict(x=-0.3, y=-1.0, z=1.25),  # Adjust x, y, z for position and zoom
            up=dict(x=1, y=1, z=0),        # Z-axis points up
            center=dict(x=0, y=0, z=0)     # Camera looks at the origin
        )
    )
)