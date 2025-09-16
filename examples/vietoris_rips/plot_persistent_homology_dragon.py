"""
.. _vietoris_rips_dragon:

Persistent Homology: Stanford Dragon
========================================

This tutorial covers persistent homology of point clouds, and filtered Vietoris-Rips complexes more generally.
"""


# %%
import oat_python as oat
import plotly.graph_objects as go
import numpy as np


# %% 
# Load the point cloud
# -------------------------------------

# %%
# For this tutorial we will work with a point cloud of 1000 points, sampled from the Stanford Dragon. This cell will load the points
# and plot them.

# Load the point cloud
points  =   oat.point_cloud.stanford_dragon()

# Plot
trace   =   go.Scatter3d(
                x=points[:,0],
                y=points[:,1],
                z=points[:,2], 
                mode="markers", 
                marker = dict(opacity=1.0, size=3, color=points[:,1], colorscale="Peach")
            )
fig     =   go.Figure(data=trace)
fig.update_layout( 
    title=dict(text="Stanford dragon, 1000 points"), 
    template="plotly_dark",
    height=800,
    )
fig

# %%
# Compute persistent homology
# -----------------------------------
#
# We compute persistent homology by factoring the boundary matrix.
# The following cell generates a sparse distance matrix and feeds it to the persistent homology solver.
# The result is a boundary matrix `D` which has been factored into a U-match decomposition `TM = DS`.
# This decomposition contains everything we need for persistent homology.

# compute the minimum enclosing radius; all homology vanishes above this filtration parameter
enclosing               =   oat.dissimilarity.enclosing_radius_for_points(points)   

# construct a distance matrix where values over enclosing + 0.0000000001 are removed
dissimilarity_matrix    =   oat.dissimilarity.sparse_matrix_for_points(            
                                points                      =   points,
                                max_dissimilarity           =   enclosing + 0.0000000001, # adding 0.0000000001 avoids problems due to numerical error
                            )

# build and factor the boundary matrix
decomposition           =  oat.core.vietoris_rips.BoundaryMatrixDecomposition( 
                                dissimilarity_matrix        =   dissimilarity_matrix,
                                max_homology_dimension      =   1,
                                support_fast_column_lookup  =   True,
                            )


# %%
# Export a summary of the persistent homology to a data frame.
persistent_homology_dataframe            \
                        =   decomposition.persistent_homology_dataframe(
                                return_cycle_representatives    =   True,
                                return_bounding_chains          =   True,
                            )
persistent_homology_dataframe


# %%
# Plot the persistence diagram
# ------------------------------

fig_pd                  =   oat.plot.persistence_diagram(
                                persistent_homology_dataframe 
                            )
fig_pd

# %%
# Plot the barcode
# ------------------------------

# %%
fig_barcode             =   oat.plot.barcode(
                                persistent_homology_dataframe
                            )
fig_barcode

# %%
# Inspect homology and cycle representatives
# -----------------------------------------------
# 
# The `homology` object is a data frame

# %%
persistent_homology_dataframe

# %%
# Inspect a cycle representative and its bounding chain
# ---------------------------------------------------------
# 
# This dataframe is sorted by the values in the `filtration` column, with ties broken by lexicographic order on simplices.

# %%
persistent_homology_dataframe.cycle_representative[875]

# %%
persistent_homology_dataframe.bounding_chain[875]

# %%
# Plot a representative
# -----------------------------------------------
#
#
#
# .. note::
#
#   Check out the :ref:`Plotting tutorials <Tutorials>` for more resources on plotting, especially
#
#   - :ref:`styling_3d_gallery`
#   - :ref:`cycle_representative_strategies_gallery`
#   
#


# %%
# We'll plot a cycle representative for the following row of the persistent homology dataframe:
feature_number      =   1203

# %%

edges               =   persistent_homology_dataframe["cycle_representative"][feature_number]["simplex"].tolist() # the cycle
triangles           =   persistent_homology_dataframe["bounding_chain"][feature_number]["simplex"].tolist() # the chain that bounds the cycle
points              =   points

trace_edge          =   oat.plot.trace_3d_for_edges(
                            edges=edges,
                            points=points,
                            line=dict(color="white", width=10),
                            name="Cycle",                            
                        )
trace_triangle      =   oat.plot.trace_3d_for_triangles(
                            triangles=triangles,
                            points=points,
                            showlegend=True,
                            opacity=0.4,
                            color="white",
                            name="Bounding Chain",                            
                        ) 
trace_points        =   go.Scatter3d(
                            x=points[:,0],
                            y=points[:,1],
                            z=points[:,2], 
                            showlegend=True,
                            opacity=0.5,                            
                            mode="markers",                             
                            marker = dict(opacity=0.8, size=3, color=points[:,1], colorscale="Peach"),
                            name="Point Cloud",                            
                        )

fig = go.Figure(data= [ trace_edge, trace_triangle, trace_points ] )
fig.update_layout(
        title=dict(text="Cycle representative and bounding chain"),
        template="plotly_dark",
        height=800,
    )
fig.update_layout() 
fig

# %%
# Compare with an optimal cycle representative
# -----------------------------------------------
# 
# OAT offers tools to compute optimal cycle representatives. Let's compare with an optimized version of the same cycle.
#
# .. note::
# 
#   Check out the documentation for :func:`oat_python.plot.contrast_initial_and_optimal_cycles_in_3d` for details, and options to customize the plot.

# %%
fig = oat.plot.contrast_initial_and_optimal_cycles_in_3d(
    boundary_matrix_decomposition   =   decomposition,
    birth_simplex                   =   persistent_homology_dataframe["birth_simplex"][feature_number],
    points                          =   points,
)
fig.update_layout(
    margin=dict(l=20, r=10, t=150, b=10),
    width=None,  # set the width to automatic
)
fig

# %%
# Analyze an optimal representative
# -----------------------------------------------
# 
# Here's how to compute the optimal cycle, and inspect its data frame. First, compute the optimal cycle:

optimal_cycle_data     =    decomposition.optimize_cycle(
                                birth_simplex                   =   persistent_homology_dataframe["birth_simplex"][feature_number], 
                                problem_type                    =   "preserve PH basis",
                            )
optimal_cycle_data

# %% [markdown]
# The output is a dataframe that contains the solution, as well as several other pieces of information.
# 
# - ``initial_cycle``: the initial cycle representative returned by the ``decomposition``
# - ``optimal_cycle``: the optimal cycle representative. This cycle has form 
# 
#   .. math::
#     o = z + e + Dc
#
#   where
#
#   - :math:`o` is the optimal cycle,
#   - :math:`z` is the initial cycle,
#   - :math:`c` is a chain and :math:`Dc` is the boundary of :math:`c`,
#   - :math:`e` is a chain in the space spanned by essential cycles (that is, cycles which never become boundaries).
#
#     - Typically :math:`e` is zero, so :math:`o = z + Dc`. In fact this is will *always* true if :math:`z` is non-essential, i.e. if :math:`z` represents
#       persistent homology class with a finite death time. If this is true, then :math:`z` and `o` will eventually become homologous, since
#       they differ by a boundary. However, :math:`c` may have a birth time strictly later than :math:`z`, so :math:`z` and `o` may not be
#       homologous at the birth time of :math:`z`.
#
# - ``surface_between_cycles`` is a chain :math:`c`. You can think of this chain, informally,
#   as a surface whose boundary is the difference between the initial and optimal cycles. In particular, if :math:`e=0` then :math:`0 = z + Dc`.
# - ``difference_in_essential_cycles``: is the chain :math:`e` in the decomposition above.
# 
# Notice that the optimal cycle has just 21 nonzero coefficients, while the initial cycle has 75!

# %%
# Display the data frame for the optimal cycle
optimal_cycle_data["chain"]["optimal_cycle"]
