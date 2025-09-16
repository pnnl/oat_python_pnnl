"""
.. _styling_3d_gallery:

Styling in 3D
=================

This tutorial shows how to adjust the layout of 3D plots in Plotly. For more advanced usage,
see the Plotly documentation: https://plotly.com/python/reference/layout/.
"""


# %%

import oat_python as oat

import networkx as nx
import numpy as np
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# %%
# Setup
# ---------------------------------------
#
# Let's generate a random simplicial complex to work with.
# We'll sample random points in a 30-dimensional ambient space,
# then construct a Vietoris-Rips complex.


# Sample points

np.random.seed(0)           # Set random seed for reproducibility
n_points                =   10
ambient_dimension       =   30
points_array            =   np.random.rand(n_points, ambient_dimension)

# Generate a Vietoris-Rips complex on these points

epsilon                 =     2.2
dissimilarity_matrix    =     oat.dissimilarity.sparse_matrix_for_points(
                               points_array,
                               max_dissimilarity   =   2.2,
                            )   
vietoris_rips_complex   =   oat.core.vietoris_rips.VietorisRipsComplex(
                               dissimilarity_matrix,
                            )

# List the simplices in the complex

simplices               =   vietoris_rips_complex.simplices_for_dimensions([0,1,2])
simplices[30:35]

# %%
# Generate 3D coordinates for the vertices of the complex, based on its combinatorial structure

points                 =    oat.plot.vertex_embedding_for_simplices(
                                simplices.simplex,
                                dimension                 =   3,    
                                iterations                =   1000,  
                                seed                      =   11,
                                method                    =   "spring",
                            )

# %%
# Generate an initial plot, using :func:`oat_python.plot.fig_3d_for_simplices`

fig                 =   oat.plot.fig_3d_for_simplices(
                            simplices       =   simplices.simplex,
                            points          =   points,
                        )
fig

# %%
# Canvas size
# ---------------------------------------

# %%
# Set a fixed width and height (in pixels)
fig.update_layout(
    width               =   300,
    height              =   300,
)

# %%
# Reset the width to automatic sizing
fig.update_layout(
    width               =   None,
)

# %%
# Increase the height
fig.update_layout(
    height             =   600,
)

# %%
# Aspect ratio
# ---------------------------------------
#
# Enforce a 1:1:1 aspect ratio (compare with the plot above):

fig.update_layout(
    scene = dict(
        aspectratio=go.layout.scene.Aspectratio(x=1, y=1, z=1), 
    )
)
fig


# %%
# Axis limits
# ---------------------------------------
#
# Set explicit axis limits.

fig.update_layout(
    scene = dict(
        xaxis = dict(range=[-1, 1],), # x axis limits
        yaxis = dict(range=[-1.5, 1],), # y axis limits
        zaxis = dict(range=[-1, 1],), # z axis limits
    ),    
)
fig

# %%
# Grid lines
# ---------------------------------------
#
# Grid lines can be removed with ``oat_python.plot.blank_background``.

oat.plot.blank_background(fig)
fig
# %%
# Zoom
# ---------------------------------------
#
# The camera position can be initialized with greater/lesser zoom by adjusting the `eye` parameter.

fig.update_layout(scene_camera=dict(
    eye=dict(x=0.6, y=0.6, z=0.6)
))
fig

# %%
# Background color
# ---------------------------------------
#
# Use a solid white background.

oat.plot.set_background_color(fig, 'white')
fig



# %%
# Color
# ---------------------------------------
#
# Colors for vertices, edges, and triangles can be adjusted either by modifying the traces in ``fig.data``, or 
# by passing keyword arguments to the relevant constructor functions.
# The traces for vertices, edges, and triangles are stored in ``fig.data[0]``, ``fig.data[1]``, and ``fig.data[2]``, respectively.
# These plots are generated, internally, using the functions
#
# - :func:`plotly.graph_objects.Scatter3d` (for vertices)
# - :func:`oat_python.plot.trace_3d_for_edges` (for edges)
# - :func:`oat_python.plot.trace_3d_for_triangles` (for triangles)
#
# Check out the documentation for these functions to see what parameters can be adjusted.


# %% [markdown]
# .. _see_also_style_3d_color:
#
# See also
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# - Color lists
# 
#   - `The list of built-in Plotly color sequences (high contrast) and color scales (smooth gradients) <color_mapping_gallery>`_.
#   - `The named CSS colors are listed here <https://htmlcolorcodes.com/color-names/>`_.
#
# - The :ref:`triangles_2d_gallery` gallery shows how to color triangles in 2D.
# - The :ref:`edges_2d_gallery` and :ref:`edges_3d_gallery` galleries show how to color edges in 2D and 3D.
# - The :ref:`triangles_3d_gallery` gallery shows how to color triangles in 3D based on scalar values assigned to vertices or faces.


# %%
# Choosing good triangle colors
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# %%
# You can use any color format that Plotly understands to update colors,
# including named colors ("white","red","brown"), hex codes (#RRGGBB), and RGB/RGBA values (rgb(255, 0, 0)).

# Change simplex color
fig.data[2].update(color="red", opacity=0.7)           # triangles
fig.data[1].update(line=dict(color="black", width=4))   # edges
fig.data[0].update(marker=dict(size=7, color="black"))  # vertices


# Re-set the background color
oat.plot.set_background_color(fig, 'white')
fig


# %%
# Dark colors often fade away against a black background
oat.plot.set_background_color(fig, 'black')
fig

# %%
# White edges and vertices contrast better

fig.data[0].update(marker=dict(size=7, color="white"))  # vertices
fig.data[1].update(line=dict(color="white", width=6))   # edges

oat.plot.set_background_color(fig, 'black')
fig


# %%
# Shades of grey
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#

# %%
# Shades of grey can give a clean look

# Color all simplices white
fig.data[2].update(color="grey", opacity=1.0)            # triangles
fig.data[1].update(line=dict(color="white", width=8))    # edges
fig.data[0].update(marker=dict(size=10, color='white'))  # vertices


# Re-set the background color
oat.plot.set_background_color(fig, 'black')
fig


# %%
# With transparency, the effect is luminous

fig.data[2].update( 
    color="white", 
    facecolor=None, # we previously assigned an explicit value to facecolor, which takes precedence over color; so to use the color parameter, we need to reset facecolor to None
    opacity=0.4,
)          

oat.plot.set_background_color(fig, 'black')
fig


# %%
# Mosaic 
# ---------------------------------------
# 
# Plotly provides predefined color sequences that can be used to assign different colors to different simplices.
# These "discrete color sequences" are designed to create visual contrast between objects. See
# the :ref:`color_mapping_gallery` gallery for a complete list of color pallettes.


# %% 
# Load a Plotly sequence of high-contrast pastel colors
pastel_sequence     =   px.colors.qualitative.Plotly 
pastel_sequence

# %%
# Shrink/expand this sequence to match the number of triangles triangles (wrapping around cyclicly, if necessary)
num_triangles       =   len( [s for s in simplices.simplex if len(s) == 3] ) # count triangles in the plot
triangle_colors     =   [pastel_sequence[i % len(pastel_sequence)] for i in range(num_triangles)] # match the length of the sequence to the number of triangles

# %%
# Assign these colors to the triangles in the plot
fig.data[2].update( facecolor=triangle_colors )
fig

# %%
# Smooth color gradients
# ---------------------------------------

# %%
# Color gradients can be used to convey additional information about simplices.
# Here we color triangles based on the distance of their vertices from the origin.
# 
# - See the documentation for :func:`oat_python.plot.trace_3d_for_triangles` for usage instructions.
# - See the `Plotly documentation on colorscales <https://plotly.com/python/colorscales/>`_ for a list of built-in colorscales.

triangles               =   [s for s in simplices.simplex if len(s) == 3] 
vertices                =   oat.simplex.vertices_incident_to_simplices(triangles)
pastel_sequence                  =   [np.linalg.norm(points[v]) for v in vertices] # distance of each vertex from the origin

fig.data[2].update( 
    intensity           =   pastel_sequence,     # assign scalar color values to vertices
    colorscale          =   'Jet',      # choose a colorscale
    intensitymode       =   'vertex',   # specify that colors are assigned to vertices (not faces)
    showscale           =   False,      # hide the colorbar
)

fig


# %% 
# Color values for vertices can also be passed directly to the :func:`oat_python.plot.trace_3d_for_triangles` function.
# Here we color triangles based on their z-coordinate.
# Again, see the documentation for :func:`oat_python.plot.trace_3d_for_triangles` for usage instructions.

trace           =   oat.plot.trace_3d_for_triangles(
                        triangles           =   triangles, 
                        points              =   points,
                        opacity             =   1.0,
                        intensity           =   [ point[2] for point in points.values() ],  # color triangles based on the z-coordinate of their vertices
                        colorscale          =   'Pinkyl',
                        intensitymode       =   'vertex',
                    )
triangles_fig   =   go.Figure(trace)
triangles_fig



# %%
# Opacity
# ---------------------------------------
#

# %%
# Opaque simplices can make make 3d structure easier to perceive.

fig.data[2].update(opacity=1.0)          # triangles
fig









# %%
# Multiple static views (alternative to video)
# -----------------------------------------------
#

# %%
# If you're unable to make a video of a rotating object, you can
# often convey the same depth information by showing multiple
# static views of the object from different angles.

# Create a new figure with 3 3D subplots
subfig = make_subplots(
    rows=3, cols=1,
    specs=[[{'type': 'scene'}], [{'type': 'scene'}], [{'type': 'scene'}]],
    subplot_titles=["View 1", "View 2", "View 3"],
    vertical_spacing=0.05 # Adjust this value to control spacing
)



# Add the same traces to each subplot
for trace in fig.data:
    subfig.add_trace(trace, row=1, col=1)
    subfig.add_trace(trace, row=2, col=1)
    subfig.add_trace(trace, row=3, col=1)

# Set different camera angles for each subplot
subfig.update_layout(
    scene=dict(camera=dict(eye=dict(x=0.8, y=0.8, z=0.8))),
    scene2=dict(camera=dict(eye=dict(x=0., y=0., z=1.6))),
    scene3=dict(camera=dict(eye=dict(x=-0.8, y=-0.8, z=0.8))),
    height=400, width=1200
)

subfig.update_layout( 
    showlegend = False,
    height = 1400, 
    width = None,
    template = "plotly_dark",    
)  

subfig