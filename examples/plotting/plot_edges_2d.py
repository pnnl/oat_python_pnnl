"""
.. _edges_2d_gallery:

Edges in 2D
========================================
"""

# %% [markdown]
# This tutorial show how to plot edges in 2D, using
#
# - :func:`oat_python.plot.trace_2d_for_edge`, which generates a trace for a single edge, and 
# - :func:`oat_python.plot.trace_2d_for_edges`, which generates a single trace for multiple edges.
#
#


# %%
# Setup
# ------------------------------------------------------

# %%
import oat_python as oat
import plotly
import plotly.graph_objects as go
import numpy as np

# %%
# We'll use a cycle graph on N vertices as an example. 
# Here we generate N evenly spaced points on a circle, which will be the vertices of the graph.
N               =   10
points          =   oat.point_cloud.circle( n_points=N, )
points


# %%
# Color scales 
# ------------------------------------------------------
#
#
# Plotly has no native functionality to plot multiple edges on a colorscale in a single trace.
# If you'd like to plot multiple edges on a colorscale, you can create a separate trace for each edge,
# and assign a color to each edge individually. Here's an example.
# 


# %%
# Get a Plotly colorscale object, which maps values in [0,1] to RGB colors.
# 
# - For a full list of Plotly colorscales, see: https://plotly.com/python/builtin-colorscales/
# - For another example of mapping scalar values to colors, see :ref:`color_mapping_gallery`.

colorscale      =   plotly.colors.get_colorscale('Rainbow') 

# %%
# Convert N evenly spaced values in [0,1] to colors using the colorscale. We'll use these colors for the edges of a cycle graph.
colors          =   plotly.colors.sample_colorscale(
                        colorscale, 
                        np.linspace(0, 1, N) 
                    )
colors

# %%
# Plot the edges of the graph, using :func:`oat_python.plot.trace_2d_for_edge`.

# Make one trace per edge, each with a different color. For an explanation of
# the keyword arguments, see the docstring for :func:`oat_python.plot.trace_2d_for_edge`.
traces          =   []
for p in range(N):
    trace       =   oat.plot.trace_2d_for_edge(
                       points          =   points,
                       edge            =   [p, (p+1) % N],
                       line            =   dict(
                                               color       =   colors[p],
                                               width       =   3,
                                           ),
                       name            =   f"Trace for edge {p}",
                    )

    traces.append(trace)

# Create a figure with all traces
fig = plotly.graph_objects.Figure(traces)

# Set the title, and make the aspect ratio equal
fig.update_layout(
    title="Using a colorscale by placing edges in separate traces",    
    yaxis=dict(
        scaleanchor="x",  # Anchor y-axis scaling to x-axis
        scaleratio=1.0    # Set the aspect ratio (1.0 for square aspect)
    ),
)

# Show the figure
fig


# %%
# Grouping legend entries
# ------------------------------------------------------
#
# Each trace appears separately in the legend by default. However,
# 
# - You can group multiple traces together in the legend using keyword arguments.
#   See the `Legend` section of :ref:`triangles_3d_gallery` for examples.
# - If you do not need to assign different colors to different edges, then
#   you can plot multiple edges in a single trace using :func:`oat_python.plot.trace_2d_for_edges`.
#   See the next section for an example.


# %%
# Grouping multiple edges in a single trace (may improve performance)
# ----------------------------------------------------------------------
#
#
# If you do not need to assign different colors to different edges, then
# you can plot multiple edges in a single trace using :func:`oat_python.plot.trace_2d_for_edges`.
# Here's an example.

# %%
# Make a single trace for all edges, with the same color for all edges.
# For an explanation of the keyword arguments, see the docstring for :func:`oat_python.plot.trace_2d_for_edges`.

trace       =   oat.plot.trace_2d_for_edges(
                    points          =   points,
                    edges           =   [ [p, (p+1) % N] for p in range(N) ],
                    line            =   dict(
                                            color       =   "crimson",
                                            width       =   3,
                                        ),
                    showlegend      =   True,
                    name            =   f"Edges of a cycle graph",
                )

# Create a figure with the trace
fig = plotly.graph_objects.Figure(trace)

# Set the title, and make the aspect ratio equal
fig.update_layout(
    title="Multiple edges in a single trace",    
    yaxis=dict(
        scaleanchor="x",  # Anchor y-axis scaling to x-axis
        scaleratio=1.0    # Set the aspect ratio (1.0 for square aspect)
    ),
)

# Show the figure
fig
