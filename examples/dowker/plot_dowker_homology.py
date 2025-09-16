"""
.. _dowker_homology_gallery:

Homology
=========================================

In this example we compute the homology of a Dowker complex.
"""

import oat_python as oat

# %%
# Define a dowker complex
# ------------------------
# Dowker complexes are represented as a list of sorted-lists of integers.
# Each sorted list represents a simplex, and the Dowker complex consists of all
# subsets of these simplices. 

dowker_simplices        =   [
                                [0,1,2,6],   # simplex 0
                                [0,1,2],     # simplex 1
                                [0,1,3],     # simplex 2
                                [0,2,3],     # simplex 3
                                [1,2,3],     # simplex 4
                                [3,4],       # simplex 5
                                [4,5],       # simplex 6
                                [3,5],       # simplex 7                                                                                                                         
                            ]

# %%
# Plot the Dowker complex
# ------------------------
#
# This step isn't necessary, but the complex is small and it will help to visualize it.

points                  =   oat.plot.vertex_embedding_for_simplices(
                                dowker_simplices,
                                dimension       =   3,      # 3D embedding   
                            )
fig                     =   oat.plot.fig_3d_for_simplices(
                                simplices       =   dowker_simplices,
                                points          =   points,
                                kwargs_points   =   dict(
                                                        mode="markers+text",  # indicates we want to plot text labels on points
                                                        text=[str(i) for i in range(len(points))] # text labels for each vertex
                                                    ),
                                kwargs_triangles=   dict(color="white") # color the triangles (transparent) black
                            )
fig.update_layout(template="plotly_dark") # dark theme
fig


# %%

# %%
# Compute homology by decomposing the boundary matrix
# -----------------------------------------------------
#
# Each row of the homology dataframe represents a homology class.
# Taken together, these homology classes form a basis for the homology groups of the Dowker complex
# in dimensions 0 .. max_homology_dimension.

decomposition           =   oat.core.dowker.BoundaryMatrixDecompositionDowker(
                                dowker_simplices        =   dowker_simplices,
                                max_homology_dimension  =   2  
                            )
homology                =   decomposition.homology()
homology

# %%
# Cycle representatives
# -----------------------------------------------------
#
# The "cycle representative" column in the homology dataframe contains a
# cycle representative for each homology class.
# 
homology["cycle_representative"][1]


# %%
# Plotting
# -----------------------------------------------------
#
# Let's plot some cycle representatives from the homology dataframe.
#
# See also
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# - :ref:`vertex_embedding_gallery`
# - :ref:`cycle_representative_strategies_gallery`
# - :ref:`styling_3d_gallery`

# %% [markdown]
# Plot a 1-cycle
# ^^^^^^^^^^^^^^^^^^^^

# %% [markdown]
# Pull the cycle from the homology dataframe.

# %%
one_cycle              =   homology["cycle_representative"][1]
one_cycle

# %%
edges               =   one_cycle.simplex
trace_1cycle        =   oat.plot.trace_3d_for_edges(
                            edges           =   edges,
                            points          =   points,
                            line            =   dict( width=10, color="red" ),
                            opacity         =   1.0,
                            showlegend      =   True,
                            name            =   "1-cycle"
                        )
fig.add_trace(trace_1cycle)
fig


# %% [markdown]
# Plot a 2-cycle
# ^^^^^^^^^^^^^^^^^^^^^^^^
# 

# %% [markdown]
# Pull the cycle from the homology dataframe.

# %%
two_cycle              =   homology["cycle_representative"][2]
two_cycle

# %% [markdown]
# Toggle off the one-cycle (but don't remove it from the figure).

# %%
fig.data[3].visible = "legendonly"

# %% [markdown]
# Add a trace for the 2-cycle, colored red.

# %%

triangles           =   two_cycle.simplex
trace_2cycle        =   oat.plot.trace_3d_for_triangles(
                            triangles       =   triangles,
                            points          =   points,
                            color           =   "red",
                            opacity         =   0.5,
                            showlegend      =   True,
                            name            =   "2-cycle"
                        )
fig.add_trace(trace_2cycle)
fig




# %% 
# Fundamental subspaces
# -----------------------------------------------------
#
# The ``.fundamental_subspace_dimensions()`` method returns a data frame with the dimensions of
# the spaces of chains, cycles, and boundaries.

decomposition.fundamental_subspace_dimensions()

# %%
# Betti numbers
# -----------------------------------------------------
#
# The ``.betti_numbers()`` method returns a list [b_0, b_1, ..., b_max_homology_dimension],
# where b_i is the dimension of the i-th homology group.

decomposition.betti_numbers()
