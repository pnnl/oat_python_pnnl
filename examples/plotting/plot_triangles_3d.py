"""
.. _triangles_3d_gallery:

Triangles in 3D
========================================

This tutorial covers commonly asked questions about ploting triangles in 3D using Plotly.
"""




# %%
# Setup
# -----------------------------------------------


# %%
# In this tutorial we plot eight triangles forming an octahedron.
# 
# Define the triangles:

import oat_python as oat
import plotly
import plotly.graph_objects as go
import copy

# Four triangles form the "upper hemisphere" of the octahedron
# with vertex 5 at the "north pole"
upper_hemisphere        =   [ 
                                [1, 2, 5],
                                [2, 3, 5],
                                [3, 4, 5],
                                [1, 4, 5],
                            ]

# Four triangles form the "lower hemisphere" of the octahedron
# with vertex 0 at the "south pole"
lower_hemisphere        =   [
                                [0, 1, 2],
                                [0, 2, 3],
                                [0, 3, 4],
                                [0, 1, 4],   
                            ]     

# Combine the two hemispheres to get all eight triangles
# in the octahedron
triangles               =   upper_hemisphere + lower_hemisphere

# %%
# Define x-y-z coordinates of the six vertices:

points                  =   {
                                0: (  0,  0, -1 ),
                                1: (  1,  0,  0 ),
                                2: (  0,  1,  0 ),
                                3: ( -1,  0,  0 ),
                                4: (  0, -1,  0 ),
                                5: (  0,  0,  1 ),
                            }

# %%
# Essentials
# --------------------------------
#
#
# **Basic workflow** 
#
# The plotting workflow explored in this tutorial has two main steps: 
#
# - Create one trace for each triangle
#   
#   - Each trace is a Plotly object of type `plotly.graph_objects.Mesh3D: API <https://plotly.com/python/reference/mesh3d/>`_.
#   - Traces are created with :func:`oat_python.plot.trace_3d_for_triangles`.
#   
# - Combine traces into a single figure
# 
#
# **Documentation**
#
# - :func:`oat_python.plot.trace_3d_for_triangles`
# - `plotly.graph_objects.Mesh3D: API <https://plotly.com/python/reference/mesh3d/>`_: 
# - `plotly.graph_objects.Mesh3D: Examples <https://plotly.com/python/3d-mesh/>`_: 
#
# 
# **Assigning colors and labels to vertices**
# 
# There are several keyword arguments that can be used to assign colors and labels to vertices.
# The usage for these keywords is a little different for :func:`oat_python.plot.trace_3d_for_triangles` than for
# `plotly.graph_objects.Mesh3D: API <https://plotly.com/python/reference/mesh3d/>`_. Most users
# won't need to worry about these differences, but if you do, see the documentation for :func:`oat_python.plot.trace_3d_for_triangles` for details.
#
# **Updating traces**
#
# Any keyword argument that can be used when creating a ``trace`` (such as color, opacity, name, text; see the `plotly.graph_objects.Mesh3D API <https://plotly.com/python/reference/mesh3d/>`_ for a complete list) can be updated later by one of the following methods
#     
# - ``trace.update(keyword1=value1, keyword2=value2, ..)``
# - ``trace.keyword = value``
#
# **Alternatives**
#
# It's relatively straightforward to create your own `plotly.graph_objects.Mesh3D: API <https://plotly.com/python/reference/mesh3d/>`_ traces from scratch!
#
#   - See documentation links above, for details.
#   - Feel free to copy the source code of :func:`oat_python.plot.trace_3d_for_triangles` as a starting point.
#
#
# **Example**


triangles_trace =   oat.plot.trace_3d_for_triangles(
                        triangles   =   triangles,
                        points      =   points,
                        color       =   "red",
                        opacity     =   0.6,
                    )

fig             =   go.Figure(triangles_trace)
fig


# %%
# Interactivity
# --------------------------------
# 
# Zoom/Pan/Rotate
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 
# The plot is interactive. You can zoom, pan, and rotate the figure using the mouse.
#
# Toggle
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# You can "toggle" simplices on and off by clicking legend entries. Try this out on some of the plots in this tutorial!
# See the `Legend` section for more details.
# 
# Hover text
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 
# You can make text appear when the mouse hovers over a vertex; see the `Text` section for details.


# %%
# Title/Layout
# --------------------------------
#
# The title and layout of a plot can be adjusted using `fig.update_layout()`
#
# .. _see_also_triangles_3d_title_layout:
#
# See also
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
#   - `Full documentation for Plotly layout API <https://plotly.com/python/reference/layout/>`_
#   - :ref:`styling_3d_gallery`
#
#
# Set title and adjust figure size / aspect ratio
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# %%
fig.update_layout(
    title="Title: Octahedron",
    height=700,     
    scene = dict(
        aspectratio=go.layout.scene.Aspectratio(x=1, y=1, z=1), # controls zoom
        xaxis = dict(range=[-1, 1],), # x axis limits
        yaxis = dict(range=[-1, 1],), # y axis limits
        zaxis = dict(range=[-1, 1],), # z axis limits
    ),    
)

# %%
# Customize background
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# %%
fig.update_layout(template="plotly_dark")
fig

# %%
oat.plot.set_background_color(fig, "black")
fig

# %%
oat.plot.set_background_color(fig, "white")
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
# - Vertices/Edges
#
#   - If you add separate traces for vertices and edges, they will have their own options
#     for text annotations. Sometimes the options are different from those used for triangles,
#     e.g. the ability to use "persistent" text labels, which remain visible even when the mouse
#     is not hovering over the vertex. See the relevant portions of
#     OAT's :ref:`tutorials` and Plotly documentation on Scatter and Scatter3d plots, for details.
#
# - Hover text
#
#   - It's possible to assign text to the vertices of a `plotly.graph_objects.Mesh3D: API <https://plotly.com/python/reference/mesh3d/>`_ trace using the ``text=..`` or ``hovertext=..`` keyword arguments.
#     See :func:`oat_python.plot.trace_3d_for_triangles` for details and usage. Limitations to be aware of:
#     
#     - Hover text can only be assigned to vertices, not to edges or faces.
#       
#       - If text on edges or faces is desired, one option is to add a separate ``Scatter3d`` trace for the edges or faces.
#       - Another option is to create a separate trace for each triangle,
#         and assign the same text label to each vertex using ``text="your_text"`` or ``hovertext="your_text"``.
#         This will cause ``"your_text"`` to appear when the mouse hovers over any vertex of the triangle.
#     
#     - Hover text is not persistent; it appears only when the mouse is hovering over a vertex. If persistent text is desired, add a separate ``Scatter3d`` trace for the vertices.
#   
#   - These arguments take a list ``[label0, label1, .., labeln-1]`` as input, where ``n`` is the number of vertices.
#     To use format this list correctly, you need to know which vertex corresponds to each label in the list.
#     See the :func:`oat_python.plot.trace_3d_for_triangles` documentation for details.


# %%
# Vertices
# --------------------------------
#
# Vertices can be added in a separate trace, using Plotly's ``Scatter3d``.
# The appearance of this scatter plot is highly customizable.
# See Plotly's documentation for details: https://plotly.com/python/reference/scatter3d/.

# %%
# Place x-y-z coordinates of each vertex in separate lists
x               =   [pt[0] for pt in points.values()]
y               =   [pt[1] for pt in points.values()]
z               =   [pt[2] for pt in points.values()]

# Create a trace for the vertices
vertices_trace  =   go.Scatter3d(
                        x=x,y=y,z=z,  # x, y, z coordinates
                        mode            =   "markers+text", # indicates we want some text to appear next to each marker
                        text            =   [f"Vertex {p}" for p in range( len(x) )], # the text we want to appear next to each point
                        textposition    =   "top center", # where we want the text positioned, relative to the marker
                        marker          =   dict(size=6, color="blue"), # marker size and color
                        name            =   "Vertices",
                    )
# Update the triangle trace, so that it appears in the legend
triangles_trace.update( 
                        name            =  "Triangles",
                        showlegend      =   True, # indicate we want this simplex to appear in the legend
                    )
fig             =   go.Figure([vertices_trace,triangles_trace])
fig

# %%
# Edges
# --------------------------------
#
# We aren't currently aware of ways to plot border lines around each simplex natively. Instead, one can plot the line segments separately.



# Get a list of all edges incident to the triangles
# -------------------------------------------------

edges           =   list(oat.simplex.dimension_m_faces_for_simplices(
                        simplices = triangles, 
                        m=1
                    ))
print(f"First five edges: {edges[:5]}")

# Make a trace for the edges
# ---------------------------

edges_trace     =   oat.plot.trace_3d_for_edges( 
                        edges,
                        points      =   points,
                        hoverinfo   =   'none',
                        line        =   dict(
                                            width   =   5, 
                                            color   =   "white",
                                        ),
                        name        =   "Edges",
                        showlegend  =   True,                    
                    )

# Plot the edges trace on top of previous figure
# ----------------------------------------------

fig = go.Figure([edges_trace, triangles_trace])
fig

# %%
#
#
# .. _styling_3d_gallery_color:
#
# Color
# --------------------------------
#
# 
# This tutorial focuses on triangles plotted using :func:`oat_python.plot.trace_3d_for_triangles`, which produces a `plotly.graph_objects.Mesh3D: API <https://plotly.com/python/reference/mesh3d/>`_.
# The color of each triangle can be set using several different keyword arguments, which can be passed to :func:`oat_python.plot.trace_3d_for_triangles`
# or can be updated later using ``trace.update(...)`` or ``trace.keyword = value``.
#
#   - ``color`` single color for all triangles
#   - ``vertexcolor`` list of colors for each vertex; can be scalar values between 0 and 1, or any list of colors in formats recognized by Plotly
#   - ``intensity`` list of values for each vertex, mapped to a colorscale
#   - ``facecolor`` list of colors for each face [overrides ``color`` and ``vertexcolor``]
#   - ``intensitymode`` either "vertex" or "cell" (default is "vertex"). If "vertex", the values in ``intensity`` are mapped to colors at each vertex, and colors on faces are interpolated from vertex colors. If "cell", the value in ``intensity`` applies to the whole triangle, and the triangle is colored uniformly.
#   - ``cmax`` Sets the upper bound of the color domain. Value should have the same units as `intensity` and if set, `cmin` must be set as well.
#   - ``cmid`` Sets the mid-point of the color domain by scaling `cmin` and/or `cmax` to be equidistant to this point. Value should have the same units as `intensity`. Has no effect when `cauto` is `False`.
#   - ``cmin`` Sets the lower bound of the color domain. Value should have the same units as `intensity` and if set, `cmax` must be set as well.
#
# .. _see_also_triangles_3d_color:
#
# See also
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# - `Full documentation for Plotly Mesh3D API <https://plotly.com/python/reference/mesh3d/>`_
# - :ref:`styling_3d_gallery` for examples and tips on styling
# - The :ref:`triangles_3d_gallery_legend` for examples of grouping legend entries.
#
#
#
#
# Automatic color rotation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
#
# Plotly automatically cycles through a sequence of colors when multiple traces are added to a figure.
#
traces          =   [
                        oat.plot.trace_3d_for_triangles(
                            triangles   =   [triangle], 
                            points      =   points,
                            opacity     =   0.5,
                            showlegend  =   True, # indicate we want this simplex to appear in the legend
                            name        =   f"Triangle {triangle}", # label for the legend entry
                        ) 
                        for triangle in triangles
                    ]
fig             =   go.Figure(traces)
fig


# %%
# Single solid color
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# A solid color can be assined to the set of all triangles using the ``color`` keyword argument.

triangles_trace =   oat.plot.trace_3d_for_triangles(
                        triangles   =   triangles, 
                        points      =   points,
                        color       =   "crimson", 
                        opacity     =   0.5,
                    )
fig             =   go.Figure(triangles_trace)
fig


# %%
# Colormap on vertices (smooth gradient)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 
# You can assign colors to vertices which will be smoothly interpolated across the surface of each triangle.
#
# - See :ref:`styling_3d_gallery_color` for a list of relevant keyword arguments.
# - **Caution** See :func:`oat_python.plot.trace_3d_for_triangles` for details on proper usage.


# %%

# Color each vertex according to its z-coordinate
# -----------------------------------------------------

triangles_trace =   oat.plot.trace_3d_for_triangles( 
                        triangles, 
                        points      =   points,
                        intensity   =   [point[2] for point in points.values()],  # assign a scalar color value to each point, equal to its z-coordinate
                        cmin        =   -1, 
                        cmax        =   1, 
                        colorscale  =   "Agsunset",
                        showscale   =   False,     
                        opacity     =   1.0,
                    ) 

# Create plot and adjust layout
# -----------------------------

fig             =   go.Figure(triangles_trace)
fig.update_layout(
    height  =   700,     
    scene   =   dict(
                    aspectratio =   go.layout.scene.Aspectratio(x=1, y=1, z=1), # controls zoom
                    xaxis       =   dict(range=[-1, 1],), # x axis limits
                    yaxis       =   dict(range=[-1, 1],), # y axis limits
                    zaxis       =   dict(range=[-1, 1],), # z axis limits
                ),    
)
fig


# %%
# Colormap on whole triangles (solid color per triangle)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 
# Whole triangles can also be colored on a colorscale:
#
# - The `facecolor` and `intensity` (with ``intensitymode="cell"``) keywords can be used to assign a solid color to each triangle.
#
#   - See :ref:`styling_3d_gallery_color` for a full list of relevant keyword arguments.
#   - These keywords take a list of length equal to the number of triangles. List entries can be either scalar values,
#     which are mapped to colors using a colorscale, or any list of colors in formats recognized by Plotly. The order of colors in the list
#     ``[c0, c1, .., cn-1]`` should match the order of triangles in the list ``[triangle for triangle in triangles]``.
# 

# %%

# Create a trace for each triangle
# --------------------------------

triangles_trace =   oat.plot.trace_3d_for_triangles( 
                        triangles, 
                        points          =   points,
                        intensity       =   [ (count / len(triangles)) for count in range(len(triangles)) ], # assign a solid color to each triangle
                        intensitymode   =   'cell', # assign intensity values to triangles, not vertices
                        cmin            =   -1, 
                        cmax            =   1, 
                        colorscale      =   "YlGnBu",
                        showscale       =   False,     
                        opacity         =   1.0,
                    ) 



# Create plot and adjust layout
# -----------------------------

fig             =   go.Figure(triangles_trace)
fig.update_layout(
    height  =   700,     
    scene   =   dict(
                    aspectratio =   go.layout.scene.Aspectratio(x=1, y=1, z=1), # controls zoom
                    xaxis       =   dict(range=[-1, 1],), # x axis limits
                    yaxis       =   dict(range=[-1, 1],), # y axis limits
                    zaxis       =   dict(range=[-1, 1],), # z axis limits
                ),    
)
fig



# %%
# Custom RGB colors
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Plotly offers native color maps for vertex and triangle coloring, as shown above, so there is
# little need to map scalar values to RGB colors manually. However, if you want to do this,
# check out the :ref:`color_mapping_gallery` gallery for examples and tips.



# %%
#
# .. _triangles_3d_gallery_legend:
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
    trace       =   oat.plot.trace_3d_for_triangles(
                    triangles   =   [triangle], 
                    points      =   points,
                    color       =   "red", 
                    opacity     =   0.5,
                    showlegend  =   True, # indicate we want this simplex to appear in the legend
                    visible     =   "legendonly" if counter % 2 == 0 else True, # make this simplex initially invisible
                    name        =    f"Simplex {triangle}", # label for the legend entry
                    text        =   f"Vertices: {triangle}", # text we want to appear when hovering the cursor over the simplex
                )
    traces.append(trace)

fig             =   go.Figure(traces)
fig.update_layout(
    height  =   700,     
    scene   =   dict(
                    aspectratio =   go.layout.scene.Aspectratio(x=1, y=1, z=1), # controls zoom
                    xaxis       =   dict(range=[-1, 1],), # x axis limits
                    yaxis       =   dict(range=[-1, 1],), # y axis limits
                    zaxis       =   dict(range=[-1, 1],), # z axis limits
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
for triangle in triangles:
    legendgroup =   "Lower Hemisphere" if 0 in triangle else "Upper Hemisphere"
    color       =   "blue" if 0 in triangle else "red"
    trace       =   oat.plot.trace_3d_for_triangles(
                        triangles               =   [triangle], 
                        points                  =   points,
                        color                   =   color, 
                        opacity                 =   0.5,
                        showlegend              =   True, # indicate we want this simplex to appear in the legend
                        name                    =   f"Simplex {triangle}", # text we want to appear in the legend
                        text                    =   f"Vertices: {triangle}", # text we want to appear when hovering the cursor over the simplex
                        legendgroup             =   legendgroup,
                        legendgrouptitle_text   =   legendgroup,                        
                    )
    traces.append(trace)

#   Create plot and adjust the plot layout

fig             =   go.Figure(traces)
fig.update_layout(
    height  =   700,     
    scene   =   dict(
                    aspectratio =   go.layout.scene.Aspectratio(x=1, y=1, z=1), # controls zoom
                    xaxis       =   dict(range=[-1, 1],), # x axis limits
                    yaxis       =   dict(range=[-1, 1],), # y axis limits
                    zaxis       =   dict(range=[-1, 1],), # z axis limits
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

traces          =   []
legend_counter  =   {"Upper Hemisphere": 0, "Lower Hemisphere": 0} # <-- create a counter for each legend group
for triangle in triangles:
    legendgroup =   "Lower Hemisphere" if 0 in triangle else "Upper Hemisphere"
    color       =   "blue" if 0 in triangle else "red"
    legend_counter[legendgroup] += 1
    trace       =   oat.plot.trace_3d_for_triangles(
                        triangles       =   [triangle], 
                        points          =   points,
                        color           =   color, 
                        opacity         =   0.5,
                        legendgroup     =   legendgroup,
                        showlegend      =   legend_counter[legendgroup] <= 1, # <-- show legend only for the first trace in each legend group
                        name            =   legendgroup, # <-- use the legend group name as the name for the legend entry
                        text            =   f"Vertices: {triangle}", 
                    )
    traces.append(trace)

# Create plot and adjust the plot layout

fig             =   go.Figure(traces)
fig.update_layout(
    height  =   700,     
    scene   =   dict(
                    aspectratio =   go.layout.scene.Aspectratio(x=1, y=1, z=1), # controls zoom
                    xaxis       =   dict(range=[-1, 1],), # x axis limits
                    yaxis       =   dict(range=[-1, 1],), # y axis limits
                    zaxis       =   dict(range=[-1, 1],), # z axis limits
                ),    
)
fig

