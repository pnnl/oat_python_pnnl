"""
Vector Index Tool
========================================
"""

# %%
import oat_python as oat

from fractions import Fraction
import numpy as np
import pandas as pd

# %%
# Create a Vietoris-Rips complex from a points of points
points                      =   np.random.rand(5, 3)
dissimilarity_matrix        =   oat.dissimilarity.sparse_matrix_for_points(points, max_dissimilarity=np.inf)
vietoris_rips_complex       =   oat.core.vietoris_rips.VietorisRipsComplex(dissimilarity_matrix)
boundary_matrix             =   vietoris_rips_complex.boundary_matrix_oracle()

# %%
# Generate a VectorIndexTool
vector_index_tool           =   vietoris_rips_complex.vector_index_tool(dimensions=[0,1])
vector_index_tool

# %%
# Convert a dataframe to a dense 1d numpy.ndarray
simplices_in_index_tool     =   vector_index_tool.simplices()
vector_as_dataframe         =   boundary_matrix.column_for_simplex([0,1,2])
vector_as_ndarray           =   vector_index_tool.dense_array_for_dataframe(vector_as_dataframe)

html                        =   oat.plot.display_dataframes_side_by_side(
                                    vector_as_dataframe, pd.DataFrame(dict(coefficient=vector_as_ndarray)), simplices_in_index_tool,
                                    titles=['Vector as DataFrame', 'Vector as dense array', 'Simplices in index tool']
                                )
html


# %%
# Convert the dense 1d numpy.ndarray back to a dataframe
vector_as_dataframe.equals( 
    vector_index_tool.dataframe_for_dense_array(vector_as_ndarray) 
)

# %%
# Look up the index for a given simplex
vector_index_tool.index_number_for_simplex((1,3))

# %%
# Errors for invalid simplices
try:
    vector_index_tool.index_number_for_simplex((3,2,1)) # not strictly sorted
except Exception as e:
    print(f"Caught error: {type(e).__name__}: {e}")

try:
    vector_index_tool.index_number_for_simplex((10,20)) # not in the Vietoris-Rips complex
except Exception as e:
    print(f"Caught error: {type(e).__name__}: {e}")    

# %%
# Check that a simplex is in the index tool
vector_index_tool.contains_simplex((1,4))

# %%
# Check that a simplex is not in the index tool
vector_index_tool.contains_simplex((1,5))