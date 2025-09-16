"""
.. _color_mapping_gallery:

Color Mapping
=================

It's often useful to plot collections of objects (points, edges, triangles, tetrahedra, etc.) on a colorscale. In some cases
Plotly allows this natively

    - `plotly.graph_objects.Scatter3d <https://plotly.com/python-api-reference/generated/plotly.graph_objects.Scatter3d.html>`_
      and `plotly.graph_objects.Scatter <https://plotly.com/python-api-reference/generated/plotly.graph_objects.Scatter.html>`_:
      these constructors take keyword arguments for ``marker``, which in turn takes keyword arguments for
      ``color``, ``colorscale``, ``cmin``, ``cmax``, and ``colorbar``, which can plot vertices on a colorscale.
    - `plotly.graph_objects.Mesh3D <https://plotly.com/python-api-reference/generated/plotly.graph_objects.Mesh3d.html>`_:
      This constructor takes keyword arguments for ``intensity``, ``colorscale``, ``cmin``, ``cmax``, and ``colorbar``, which can
      plot both vertices and faces on a colorscale.

However, several other types of objects can't be plotted natively on a colorscale:

    - filled triangles in 2D
    - edges in 2D and 3D

In these cases, the workaround is to create a separate trace for each object, and assign a color to each object individually. 
**The key challenge is mapping scalar values to color values that Plotly understands. This tutorial shows how to do this.**


Complete guides to coloring edges and triangles
---------------------------------------------------

Here are tutorials for the colormapping use cases that aren't covered natively by Plotly:

- :ref:`edges_2d_gallery` (edges in 2D)
- :ref:`edges_3d_gallery` (edges in 3D)
- :ref:`triangles_2d_gallery` (triangles in 2D)

"""



# %%
# Mapping scalars to a continuous colorscale (smooth gradients)
# --------------------------------------------------------------
#
# Plotly offers colorscale objects that map scalar values to RGB colors automatically.
# Here's how to use them.

# %%
import plotly
import plotly.express as px

# %%
# Get a 'Rainbow' colorscale object.
colorscale      =   plotly.colors.get_colorscale('Rainbow') 

# %%
# Convert scalars [0.2, 0.6, 0.8] into RGB colors.
rgb_values      =   plotly.colors.sample_colorscale(
                        colorscale, 
                        [0.2, 0.6, 0.8]
                    )
rgb_values


# %%
# Mapping integers to a discrete colorscale (high contrast)
# --------------------------------------------------------------
#
#
# The point of a continuous colorscale is to represent a continuous range of scalar values.
# But sometimes what you want instead is a sequence of colors that are visually distinct.
# Plotly provides several predefined discrete color sequences that can be used for this purpose.


# %%
# Here's one sequence. Colors are formatted as CSS color strings.
color_sequence  =   px.colors.qualitative.Plotly

#%% 
# Here's another sequence.
color_sequence  =   px.colors.sequential.Viridis

# %% 
# Here's an example of how to use a Plotly qualitative color sequence to assign N colors (possibly with repetition) to a list of N objects.
N = 10  # number of distinct colors needed
num_colors      =   len(color_sequence)
color_values    =   [color_sequence[i % num_colors] for i in range(N)] # match the length of the sequence to N
color_values




# %%
# List of discrete color sequences
# ---------------------------------------
#
# Here's a complete list of Plotly qualitative color sequences.
fig = px.colors.qualitative.swatches()
fig


# %%
fig = px.colors.sequential.swatches()
fig

# %%
fig = px.colors.diverging.swatches()
fig


# %% 
# List of continuous color sequences
# ---------------------------------------
#
# The continuous color sequences provided by Plotly fall into two categories: sequential and diverging.
#
# Here are the sequential color sequences.
fig = px.colors.sequential.swatches_continuous()
fig


# %% 
# Here are the diverging color sequences.
fig = px.colors.diverging.swatches_continuous()
fig