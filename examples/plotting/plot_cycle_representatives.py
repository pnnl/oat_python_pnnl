"""
.. _cycle_representative_strategies_gallery:

Strategies for Plotting Cycle Representatives
==============================================

This tutorial contains tips that we found helpful when plotting cycle representatives.


**Generate x-y-z coordinates for non-Euclidean data**


Sometimes data doesn't come with x-y-z coordinates. For example, it might
be a graph or hypergraph. Check out the :ref:`vertex_embedding_gallery` gallery
for strategies to generate this data.



- :ref:`styling_3d_gallery`
- use opaque instead of transparent colors

**Multiple still frames (different angles, useful for papers and slide presentations)**


If you're trying to convey a complex 3D structure to an audience, you might not have
the option to use a video. In this case, consider using multiple still frames from different angles.

- See :ref:`styling_3d_gallery` for examples on generating multiple subplots in Plotly.

  - Or for a simpler approach, just rotate the camera manually and take screenshots.

  
**Opaque colors**

Opaque colors can make the 3d structure of an object more apparent.

- This approach works well with multiple still frames. If you're using
  multiple frames, then you don't need to see "through" objects as much.

- Try varying the colors of your triangles, too, to help distinguish different parts of the object.


**Wire frame**


Even when plotting 2-dimensional homology classes, sometimes it's as good or better to plot
a wire frame instead (meaning just the edges incident to the triangles):

- saves computational resources (especially for large complexes)
- can be easier to visualize

Try toggling the triangles on and off in the plots below to see the difference.
"""

# %%
# Example
# -----------------
#
# Here's an example to illustrate some of these strategies.

# %%
from networkx import triangles
import oat_python as oat
import plotly.graph_objects as go
import numpy as np



# %%
# Generate a point cloud with the Stanford dragon and a sphere.

points                  =   oat.point_cloud.stanford_dragon()


# %% 
# Compute the persistent homology of the Vietoris-Rips complex.


# compute the minimum enclosing radius; all homology vanishes above this filtration parameter
enclosing               =   oat.dissimilarity.enclosing_radius_for_points(points)   

# construct a distance matrix where values over enclosing + 0.0000000001 are removed
dissimilarity_matrix    =   oat.dissimilarity.sparse_matrix_for_points(            
                                points                      =   points,
                                max_dissimilarity           =   enclosing + 0.0000000001, # adding 0.0000000001 avoids problems due to numerical error
                            )

# # format the input to the persistent homology solver
# dissimilarity_matrix    =   oat.dissimilarity.sparse_matrix_for_points(            
#                                 points                          =   points,
#                                 max_dissimilarity               =   0.3,
#                             )

# build and factor the boundary matrix
decomposition           =  oat.core.vietoris_rips.BoundaryMatrixDecomposition( 
                                dissimilarity_matrix            =   dissimilarity_matrix,
                                max_homology_dimension          =   1,
                                support_fast_column_lookup      =   True,
                            )

# extract the persistent homology dataframe, including cycle representatives and bounding chains
persistent_homology_dataframe            \
                        =   decomposition.persistent_homology_dataframe(
                                return_cycle_representatives    =   True,
                                return_bounding_chains          =   True,
                            )

# %%
# Inspect the largest cycle representatives

persistent_homology_dataframe.nlargest(10, 'num_cycle_simplices')


# %%
# The largest cycle representative lies in row 1331 of the persistent homology dataframe.
# Extract the list of triangles in its bounding chain.

triangles           =   persistent_homology_dataframe["bounding_chain"][1203]["simplex"].tolist()

# %%
# Plot the triangles

fig                 =   oat.plot.fig_3d_for_simplices(
                            simplices       =   triangles,
                            points          =   points,
                        )
fig.update_layout(template="plotly_dark", height=800)
fig

# %%
# Color and shrink the vertices, to help differentiate portions of the object
fig.data[0].marker  =   dict(color = -points[:,2], colorscale="Peach", size=3)
fig


# %%
# Make the triangles opaque, and color them by height.
fig                 =   oat.plot.fig_3d_for_simplices(
                            simplices         =   triangles,
                            points            =   points,
                            kwargs_points     =   dict(marker  =   dict( size=3, color=-points[:,2], colorscale="Peach")),
                            kwargs_triangles  =   dict(
                                                        intensity=[point[2] for point in points],                                                        
                                                        intensitymode="vertex",
                                                        colorscale="Peach",
                                                        opacity=1.0,
                                                        showscale=False,
                                                    )
                        )
fig.update_layout(template="plotly_dark", height=800)
fig

# %%
# Generate multiple views from different angles

from plotly.subplots import make_subplots

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
    scene=dict( camera=dict(eye=dict(x=0.0, y=1.0, z=1.0))),
    scene2=dict(camera=dict(eye=dict(x=0.9, y=1.2, z=1.2))),
    scene3=dict(camera=dict(eye=dict(x=0.7, y=-1.2, z=1.0))),
    height=700, 
)

subfig.update_layout(
    # showlegend = False,
    height = 1800,
    width = None,
    template = "plotly_dark",
)

subfig