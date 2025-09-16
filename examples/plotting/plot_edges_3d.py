"""
.. _edges_3d_gallery:

Edges in 3D
========================================
"""

# %% [markdown]
# This tutorial show how to plot edges in 3D, using
#
# - :func:`oat_python.plot.trace_3d_for_edge`, which generates a trace for a single edge, and 
# - :func:`oat_python.plot.trace_3d_for_edges`, which generates a single trace for multiple edges.
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
N               =   60

# Generate N points in the xy-plane
points          =   oat.point_cloud.circle( n_points=N, )

# Add a z-coordinate to make the points 3D
points          =   np.hstack( (
                        points, 
                        np.sin(np.linspace(0, 6*np.pi, N)).reshape(-1, 1))
                    )  
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

# Initialize a list to hold the traces
traces          =   []

# Make a trace for the vertices, to help mark start and end points
vertex_trace    =   oat.plot.trace_3d_for_vertices(
                       points          =   points,
                       marker          =   dict(
                                               color       =   'white',
                                               size        =   5,
                                           ),
                       showlegend      =   True,
                       name            =   "Vertex Trace",
                    )
traces.append(vertex_trace)

# Make one trace per edge, each with a different color. For an explanation of
# the keyword arguments, see the docstring for :func:`oat_python.plot.trace_2d_for_edge`.
for p in range(N):
    edge_trace  =   oat.plot.trace_3d_for_edge(
                       points          =   points,
                       edge            =   [p, (p+1) % N],
                       line            =   dict(
                                               color       =   colors[p],
                                               width       =   6,
                                           ),
                       name            =   f"Trace for edge {p}",
                    )

    traces.append(edge_trace)

# Create a figure with all traces
fig = plotly.graph_objects.Figure(traces)

# Set the title, and make the aspect ratio equal
fig.update_layout(
    title="Using a colorscale by placing edges in separate traces",  
    scene = dict(
        aspectratio=go.layout.scene.Aspectratio(x=1, y=1, z=1), 
    )
)

# Show the figure
fig

# %%
# See the :ref:`styling_3d_gallery` for tips on styling in 3D.

# Apply a dark template
fig.update_layout(template="plotly_dark")
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

# Make a trace for the vertices, to help mark start and end points
vertex_trace    =   oat.plot.trace_3d_for_vertices(
                       points          =   points,
                       marker          =   dict(
                                               color       =   'white',
                                               size        =   5,
                                           ),
                       showlegend      =   True,
                       name            =   "Vertex Trace",
                    )

# %%
# Make a single trace for all edges, with the same color for all edges.
# For an explanation of the keyword arguments, see the docstring for :func:`oat_python.plot.trace_2d_for_edges`.

edge_trace       =   oat.plot.trace_3d_for_edges(
                    points          =   points,
                    edges           =   [ [p, (p+1) % N] for p in range(N) ],
                    line            =   dict(
                                            color       =   "white",
                                            width       =   3,
                                        ),
                    showlegend      =   True,
                    name            =   f"Edge Trace",
                )

# Create a figure with the trace
fig = plotly.graph_objects.Figure([ edge_trace, vertex_trace ])

# Set the title, make the aspect ratio equal, and apply a dark template
fig.update_layout(
    title="Multiple edges in a single trace",    
    template="plotly_dark",
    scene = dict(
        aspectratio=go.layout.scene.Aspectratio(x=1, y=1, z=1), 
    )
)

# Show the figure
fig
