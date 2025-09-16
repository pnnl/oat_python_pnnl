"""
.. _triangles_2d_gallery:

Triangles in 2D
========================================
"""

import oat_python as oat

import numpy as np
import plotly


# %%
# Setup
# ---------------------------------


# %%
N               =   6

circle          =   oat.point_cloud.circle(
                        n_points=N,
                    )
points          =   np.vstack( ( np.zeros((1,2)), circle ))
points


# %%
triangles       =   [ [0, i, i+1] for i in range(1, N) ] + [ [0, N, 1] ]
triangles


triangles_trace =   oat.plot.trace_2d_for_triangles(
                        triangles           =   triangles, 
                        points              =   points,
                        opacity             =   0.75,
                    )
triangles_trace


# %%
# Place the trace in a figure and display it.
fig             =   plotly.graph_objects.Figure(triangles_trace)
fig

# %% [markdown]
# Title & Layout
# --------------------------------
#
# The title and layout of a plot can be adjusted using `fig.update_layout()`
#


# %% [markdown]
# .. _see_also_triangles_2d_title_layout:
#
# See also
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
#   - `Full documentation for Plotly layout API <https://plotly.com/python/reference/layout/>`_
#
#
# Set title and adjust figure size / aspect ratio
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


fig.update_layout(
    title="Triangles",    
    yaxis=dict(           # Ensure an equal x-y aspect ratio by ..
        scaleanchor="x",  #   Anchoring y-axis scaling to x-axis
        scaleratio=1.0    #   Setting the aspect ratio (1.0 for square aspect)
    ),
)
fig

# %%
# Customize background
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# %%
# Plotly offers several `built-in templates <https://plotly.com/python/templates/>`_ for styling plots.

fig.update_layout(template='plotly_dark')
fig


# %%
# OAT also offers several convenience functions for updating the layout.
# The following removes all grids, zero lines, axes, axis titles, and tick marks.

oat.plot.blank_background(fig)
fig

# %% 
# This function can changes the background color. The color argument
# can be in any format that Plotly accepts, e.g., "white", "#FFFFFF", "rgb(255,255,255)", etc.
oat.plot.set_background_color(fig,"white")
fig


# %%
# Text
# --------------------------------
#
# Text can appear in multiple places in a Plotly 3D plot:
# 
# 
# - Title
#
#   - See the `Title/Layout` section
#
# - Legend
#
#   - See the `Legend` section.
#
# - Vertices
#
#   - The plots generated in this tutorial are plotly.graph_objects.Scatter traces, in which 
#     the ``fillcolor`` keyword argument is used to fill triangular regions defined by the vertices.
#     Plotly allows the user to place text annotations on the vertices of a Scatter trace using the ``text=..`` or ``hovertext=..`` keyword arguments.
#     However, due to the data formatting used internally by the function :func:`oat_python.plot.trace_2d_for_triangles`,
#     usage of these arguments is somewhat complicated. **The recommended alternative** is to add a separate plostly.graph_objects.Scatter
#     trace for the vertices, which can be annotated in a straightforward manner.
#
# - Edges and Triangles
#   
#  - Plotly offers no native functionality to place text annotations on edges or filled triangles.
#    The workaround is to create a separate Scatter trace, place a point at the center of each edge or triangle,
#    and use the ``text=..`` or ``hovertext=..`` keyword arguments to annotate these points.





# %%
# Edges
# ---------------------------------
#
# The edges of the triangles can be modified via the `line` attribute of the trace
# or the `line` keyword argument of the `trace_2d_for_triangles` function.


# %%
# Remove the lines
fig.data[0].update(mode="none")
fig.update_layout(title="No edges")
fig


# %%
# Replace the lines with thick white lines. Changing the line color may 
# reset the fill color, so we set it again to red.
fig.data[0].update(
    mode        =   'lines',
    line        =   dict(
                        color   =   'white', 
                        width   =   5
                    ),
    fillcolor   =   'red'
)
fig.update_layout(title="White edges")
fig





# %%
# Vertices
# --------------------------------
#
# Vertices can be added in a separate trace, using Plotly's ``Scatter3d``.
# The appearance of this scatter plot is highly customizable.
# See Plotly's documentation for details: https://plotly.com/python/reference/scatter3d/.

# %%
# Place x-y-z coordinates of each vertex in separate lists
x               =   [pt[0] for pt in points]
y               =   [pt[1] for pt in points]

# Create a trace for the vertices
vertices_trace  =   plotly.graph_objects.Scatter(
                        x=x,y=y,  # x, y coordinates
                        mode            =   "markers+text", # indicates we want some text to appear next to each marker
                        text            =   [f"Vertex {p}" for p in range( len(x) )], # the text we want to appear next to each point
                        textposition    =   "top left", # where we want the text positioned, relative to the marker
                        marker          =   dict(size=15, color="black"), # marker size and color
                        name            =   "Vertices",
                        showlegend      =   True, # indicate we want this simplex to appear in the legend
                    )
# Update the triangle trace, so that it appears in the legend
# The triangle trace is the first trace added to the figure, so it is stored in fig.data[0]
fig.data[0].update( 
                        name            =  "Triangles",
                        showlegend      =   True, # indicate we want this simplex to appear in the legend
                    )
# Add the vertices trace to the figure
fig.add_trace(vertices_trace)
# Update the title
fig.update_layout(title="Vertices added in a separate trace")
fig




# %%
#
#
# .. _styling_2d_gallery_color:
#
# Color
# --------------------------------
#
# %%
# Plotly offers native functionality to assign a single color to all triangles in a trace.
# Different colors for different triangles are not supported.
# If different colors are desired, then the workaround is to create a separate trace for each triangle,
# and assign a color to each triangle individually. See below for examples.

# %% [markdown]
#
# .. _see_also_triangles_2d_color:
#
# See also
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# - `Full documentation for the Plotly Scatter API <https://plotly.com/python/reference/scatter/>`_
# - The :ref:`triangles_2d_gallery_legend` for examples of grouping legend entries.
#
#

# %%
# Single fill color
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The fill color of the triangles in a single :func:`oat_python.plot.trace_2d_for_triangles` trace can be set using the ``fillcolor`` keyword argument,
# as we have already seen.

# %%
fig.data[0].fillcolor = "orange"
fig

# %%
# Automatic color rotation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
#
# Plotly automatically cycles through a sequence of colors when multiple traces are added to a figure.
#
traces          =   [
                        oat.plot.trace_2d_for_triangles(
                            triangles   =   [triangle], 
                            points      =   points,
                            opacity     =   1.0,
                            showlegend  =   True, # indicate we want this simplex to appear in the legend
                            name        =   f"Trace for triangle {triangle}", # label for the legend entry
                        ) 
                        for triangle in triangles
                    ]
fig             =   plotly.graph_objects.Figure(traces)
fig.update_layout(
    title="Plotly automatically cycles through distinct colors for different traces",    
    yaxis=dict(           # Ensure an equal x-y aspect ratio by ..
        scaleanchor="x",  #   Anchoring y-axis scaling to x-axis
        scaleratio=1.0    #   Setting the aspect ratio (1.0 for square aspect)
    ),
)
fig


# %%
# Manual color rotation (maximizes contrast)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Plotly offers several built-in color sequences that can be used to assign colors to traces.
# The purpose of these color sequences is to provide visually distinct colors for different traces.
# See the :ref:`color_mapping_gallery` gallery for color options and usage.

# %%
# Choose a color sequence
pastel_sequence =  plotly.express.colors.qualitative.Pastel
num_colors      =   len(pastel_sequence)

# %%
# Assign distinct colors to consecutive triangles
traces          =   [
                        oat.plot.trace_2d_for_triangles(
                            triangles   =   [triangle], 
                            points      =   points,
                            fillcolor   =   pastel_sequence[ counter % num_colors ], # assign a color from the color sequence
                            opacity     =   1.0,
                            line        =   dict(color="white", width=6),
                            showlegend  =   True, 
                            name        =   f"Trace for triangle {triangle}", 
                        ) 
                        for counter, triangle in enumerate(triangles)
                    ]
fig             =   plotly.graph_objects.Figure(traces)
fig.update_layout(
    title="Pastel color sequence (maximizes contrast)",    
    yaxis=dict(           # Ensure an equal x-y aspect ratio by ..
        scaleanchor="x",  #   Anchoring y-axis scaling to x-axis
        scaleratio=1.0    #   Setting the aspect ratio (1.0 for square aspect)
    ),
)
fig


# %%
# Continuous color scales (smooth gradients)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Plotly also offers continuous color scales that can be used to assign colors to simplices based on scalar values.
# See the :ref:`color_mapping_gallery` gallery for color options and usage.

# %%
# Choose a colormap
colormap            =   plotly.express.colors.sequential.Rainbow

# %%
# Assign a continuous color scale to the triangles. Outputs will be formatted as RGB strings.
num_triangles       =   len(triangles)
scalar_values       =   [  i / num_triangles for i in range(num_triangles) ] # scalar values evenly spaced in [0,1]
rainbow_sequence    =   plotly.colors.sample_colorscale(colormap, scalar_values)

# Assign a different color to each triangle
traces              =   [
                            oat.plot.trace_2d_for_triangles(
                                triangles   =   [triangle], 
                                points      =   points,
                                fillcolor   =   rainbow_sequence[ counter ], # assign a color from the color sequence
                                opacity     =   1.0,
                                line        =   dict(color="white", width=6),
                                showlegend  =   True, 
                                name        =   f"Trace for triangle {triangle}", 
                            ) 
                            for counter, triangle in enumerate(triangles)
                        ]
fig                 =   plotly.graph_objects.Figure(traces)
oat.plot.set_background_color(fig, "black")
fig.update_layout(
    title   =   "Rainbow colorscale",    
    yaxis   =   dict(                 # Ensure an equal x-y aspect ratio by ..
                    scaleanchor="x",  #   Anchoring y-axis scaling to x-axis
                    scaleratio=1.0    #   Setting the aspect ratio (1.0 for square aspect)
                ),
)
fig








# %%
#
# .. _triangles_2d_gallery_legend:
#
# Legend
# --------------------------------
# 
# Documentation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 
# See Ploty's documentation: https://plotly.com/python/legend/.
# 
# Show/hide legend entries
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 
# Each trace appears in the legend by default. This can be controlled with the ``showlegend`` keyword. For illustration, see below.
# 
# Toggle traces
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 
# You can toggle traces on or off by clicking their entries in the legend
#
# - Try this with any of the plots in this tutorial
# - The text in the legend entry for a trace is determined by the ``name="desired_legend_text"`` keyword
#
# Set default toggle to off
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Traces are toggled on by default. However, you can set a trace to be
# initially toggled off using the ``visible="legendonly"`` keyword.
# In the following example, we make every second triangle initially invisible.
traces          =   []
for counter, triangle in enumerate(triangles):
    trace       =   oat.plot.trace_2d_for_triangles(
                    triangles   =   [triangle], 
                    points      =   points,
                    fillcolor   =   "orange", 
                    line        =   dict(color="white", width=6),                    
                    opacity     =   1.0,
                    showlegend  =   True, # indicate we want this simplex to appear in the legend
                    visible     =   "legendonly" if counter in [0,2] else True, # make this simplex initially invisible
                    name        =   f"Simplex {triangle}", # label for the legend entry
                    text        =   f"Vertices: {triangle}", # text we want to appear when hovering the cursor over the simplex
                )
    traces.append(trace)

fig             =   plotly.graph_objects.Figure(traces)
fig.update_layout(
    title   =   "Some triangles are initially hidden",    
    yaxis   =   dict(                   # Ensure an equal x-y aspect ratio by ..
                    scaleanchor="x",    #   Anchoring y-axis scaling to x-axis
                    scaleratio=1.0      #   Setting the aspect ratio (1.0 for square aspect)
                ),
)
fig

# %%
# Toggle groups of traces
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 
# You can link multiple traces together, so that clicking a single legend entry toggles all traces in the group.
# To do this
#
#   - Assign legend groups by declaring a group name for each trae, with the ``legendgroup="group_name"`` keyword
#   - Optionally, assign a group title with keyword ``legendgrouptitle_text="title"``. the traces in this group will appear as bullets under ``"title"`` in the legend

# %%
# Generate a plotly trace for each triangle

traces          =   []
for counter, triangle in enumerate(triangles):
    legendgroup_num =   counter // 2
    legendgroup     =   f"Group {legendgroup_num}"
    fillcolor       =   pastel_sequence[legendgroup_num]
    trace           =   oat.plot.trace_2d_for_triangles(
                            triangles               =   [triangle], 
                            points                  =   points,
                            fillcolor               =   fillcolor, 
                            line                    =   dict(color="white", width=6),                    
                            opacity                 =   1.0,
                            showlegend              =   True, # indicate we want this simplex to appear in the legend
                            name                    =   f"Simplex {triangle}", # text we want to appear in the legend
                            text                    =   f"Vertices: {triangle}", # text we want to appear when hovering the cursor over the simplex
                            legendgroup             =   legendgroup,
                            legendgrouptitle_text   =   legendgroup,                        
                        )
    traces.append(trace)

#   Create plot and adjust the plot layout

fig             =   plotly.graph_objects.Figure(traces)
fig.update_layout(
    title   =   "Triangles are grouped (toggle on and off by group)",    
    yaxis   =   dict(                   # Ensure an equal x-y aspect ratio by ..
                    scaleanchor="x",    #   Anchoring y-axis scaling to x-axis
                    scaleratio=1.0      #   Setting the aspect ratio (1.0 for square aspect)
                ),
)
fig

# %%
# Group multiple traces under a single legend entry
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 
# Group multiple traces together under a single legend entry as follows
#
# - place desired traces in a group using the ``legendgroup="your_group_name"`` keyword
# - hide all but one of the traces using the ``showlegend=False`` keyword
# - assign the desired legend entry text to the unhidden trace, using ``name="desired_legend_text"``
# 
# **Now toggling a single legend entry toggles the entire group.**

# %%
# Generate a plotly trace for each triangle

traces              =   []
legend_counters     =   [0,0,0] # <-- create a counter for each legend group
for counter, triangle in enumerate(triangles):
    legendgroup_num =   counter // 2
    legendgroup     =   str(legendgroup_num)
    fillcolor       =   pastel_sequence[legendgroup_num]
    legend_counters[legendgroup_num] += 1
    trace           =   oat.plot.trace_2d_for_triangles(
                            triangles       =   [triangle], 
                            points          =   points,
                            fillcolor       =   fillcolor, 
                            line            =   dict(color="white", width=6),                    
                            opacity         =   1.0,
                            legendgroup     =   legendgroup,
                            showlegend      =   legend_counters[legendgroup_num] <= 1, # <-- show legend only for the first trace in each legend group
                            name            =   f"Group {legendgroup}", # <-- use the legend group name as the name for the legend entry
                            text            =   f"Vertices: {triangle}", 
                        )
    traces.append(trace)

# Create plot and adjust the plot layout

fig             =   plotly.graph_objects.Figure(traces)
fig.update_layout(
    title   =   "One legend entry per group",    
    yaxis   =   dict(                   # Ensure an equal x-y aspect ratio by ..
                    scaleanchor="x",    #   Anchoring y-axis scaling to x-axis
                    scaleratio=1.0      #   Setting the aspect ratio (1.0 for square aspect)
                ),
)
fig

