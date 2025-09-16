"""
Laplacian
==========

The Laplacian of a simplicial complex `K` with boundary matrix `D` is the matrix `L = D^T D + D D^T`.

OAT provides a matrix oracle that computes the entries of `L` in a lazy fashion, without storing them permanently in memory.
"""


# %%
import oat_python as oat

from fractions import Fraction
import numpy as np
import pandas as pd
import plotly
import plotly.subplots
import plotly.graph_objects as go
import scipy.sparse as sparse
from scipy.sparse import diags
from scipy.sparse.linalg import LinearOperator, eigsh
import tqdm


# %% [markdown]
# Generate a Vietoris-Rips complex
# ------------------------------------------

# %% [markdown]
# Fix a random seed for reproducibility

# %%
np.random.seed(0)

# %% [markdown]
# Generate a point cloud

# %%
points           =   oat.point_cloud.circle( n_points=30, radius=1.0 )
points           =   points + np.random.normal( scale=0.1, size=points.shape )  # Add some noise

# %% [markdown]
# Initialize a Vietoris-Rips complex

# %%
dissimilarity_matrix        =   oat.dissimilarity.sparse_matrix_for_points(
                                    points               =   points, 
                                    max_dissimilarity   =   1.0
                                )


vietoris_rips_complex       =   oat.core.vietoris_rips.VietorisRipsComplex(
                                    dissimilarity_matrix=dissimilarity_matrix,
                                )

# %% [markdown]
# Plot the 1-skeleton of the Vietoris-Rips complex

# %%

# initialize an empty list of traces
traces              =   []

# add a trace for the points
traces.append(
    go.Scatter(
        x=points[:,0], 
        y=points[:,1], 
        mode='markers', 
        name='Points'
    )
)

# plot the edges
edges                   =   vietoris_rips_complex \
                                .simplices_for_dimensions([1]) \
                                .simplex # this pulls out the "simplex" clolumn of the dataframe
for edge in edges:
    trace            =   oat.plot.trace_2d_for_edge( edge=edge, points = points )
    trace.update( line = dict( width=1, color='blue' ) )
    traces.append(trace)

# plot the traces
fig                  =   go.Figure(data=traces)
fig.update_layout(
    title='1-Skeleton of Vietoris-Rips Complex',
    xaxis=dict(title='X'),
    yaxis=dict(title='Y'),
    width=600,
    height=600,
)
fig

# %% [markdown]
# Access the Laplacian
# ------------------------------------------

# %% [markdown]
# Initialize a Laplacian matrix oracle

# %%
laplacian_matrix_oracle     =   vietoris_rips_complex.laplacian_matrix_oracle()

# %% [markdown]
# Look up a row

# %%
vector_dataframe            =   laplacian_matrix_oracle.row_for_simplex((0,))
vector_dataframe

# %% [markdown]
# Multiply the Laplacian with a vector

# %%
product                     =   laplacian_matrix_oracle \
                                    .product_with_vector( vector_dataframe )
product

# %% [markdown]
# Compute an eigenvector
# ------------------------------------------
# 
# To do this we will use the power method built into Scipy.
# 
# First define a function which takes a 1d numpy.ndarray as input and returns the Laplacian operator applied to it.

# %%
def laplacian_operator( v ):
    """
    Evaluate the Laplacian operator on a 0-chain `v` represented
    as a 1-dimensional numpy.ndarray
    """
    vector_dataframe = vietoris_rips_complex.simplices_for_dimensions(dimensions=[0])
    vector_dataframe["coefficient"] = v
    product = laplacian_matrix_oracle.product_with_vector(vector_dataframe)
    return product.coefficient.to_numpy()




# %% [markdown]
# Obtain eigenvalues and eigenvectors of the Laplacian operator using scipy's eigsh function.

# %%
import scipy.sparse as sparse
import scipy.sparse.linalg
from scipy.sparse.linalg import LinearOperator, eigsh

node_dataframe          =   vietoris_rips_complex.simplices_for_dimensions( [0] )
n_nodes                 =   node_dataframe.shape[0]

scipy_operator          =   LinearOperator(
                                shape           =   (n_nodes, n_nodes),
                                matvec          =   laplacian_operator,
                            )
eigvals, eigvecs        =   eigsh(
                                scipy_operator,     # the linear operator
                                k=6,                # first k eigenvalues
                                which='SM',         # 'SM' = smallest magnitude
                            )  

# %% [markdown]
# Plot the eigenvector.

# %%

node_list                       =   [ simplex[0] for simplex in node_dataframe.simplex ]
trace                           =   go.Scatter(
                                        x                               =   points[ node_list, [0] ],
                                        y                               =   points[ node_list, [1] ],
                                        mode                            =   'markers',
                                        marker                          =   dict(
                                            size                        =   10,
                                            color                       =   eigvecs[:, 1],
                                            colorscale                  =   'Jet',
                                            colorbar                    =   dict(title='Eigenvector Coefficient')
                                        ),
                                        name                            =   'Eigenvector Coefficient'
                                    )
fig = go.Figure(data=[trace])
fig.update_layout(
    title='Second Eigenvector of the Laplacian Operator',
    xaxis=dict(title='X'),
    yaxis=dict(title='Y'),
    width=700,
    height=600,
)
fig


# %% [markdown]
# Validate the oracle
# ------------------------------------------

# %% [markdown]
# We can also compute the Laplacian in a non-lazy fashion.  First compute the dimension-1 boundary matrix.

# %%

d1                          =   vietoris_rips_complex \
                                    .boundary_matrix_oracle() \
                                    .write_submatrix_to_csr(
                                        vietoris_rips_complex.submatrix_index_tool(
                                            row_dimensions      =   [0],
                                            column_dimensions   =   [1]
                                        )
                                    )

d1


# %% [markdown]
# Compute the Laplacian matrix for 0-chains:

# %%
scipy_zero_laplacian         =   d1 @ d1.T

# %% [markdown]
# Verify that each row of the Laplacian oracle matches the corresponding row of the Scipy sparse matrix

# %%
vector_index_tool            =   vietoris_rips_complex.vector_index_tool(dimensions=[0])


for node_number in range(n_nodes):
        simplex                         =   vector_index_tool \
                                                .simplex_for_index_number(node_number) \
                                                .simplex()
        row_dataframe                   =   laplacian_matrix_oracle \
                                                .row_for_simplex(simplex)
        row_dataframe["coefficient"]    =   row_dataframe["coefficient"].apply(lambda x: Fraction(x))
        oracle_row                      =   vector_index_tool \
                                                .dense_array_for_dataframe(row_dataframe) \
                                                .astype(float)
        scipy_row            =   scipy_zero_laplacian[node_number].todense()

        assert np.allclose(oracle_row, scipy_row, rtol=1e-5, atol=1e-8), \
            f"Mismatch in row {node_number}: {oracle_row} vs {scipy_row}"


