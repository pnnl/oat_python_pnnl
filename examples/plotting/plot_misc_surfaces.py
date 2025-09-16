"""
.. _plot_misc_surfaces_gallery:

Miscellaneous Surfaces
================================

This gallery contains fun miscellaneous plots.
"""

# %%
import oat_python as oat
import numpy as np
import plotly.graph_objects as go
import itertools

# %% [markdown]
# Orthographic projection (dropping perspective effect)
# --------------------------------------------------------------------

# %%
trace_octahedron = oat.plot.surface_octahedron()
trace_octahedron.update(opacity=0.1)

points = np.array( # coordinate oracle
    [
        # first four columns "walk around the equator along adjacent vertices"
        # the final two columns represent the north/south poles
        [ -1,  0, 1,  0,  0,  0 ], # x
        [  0, -1, 0,  1,  0,  0 ], # y
        [  0,  0, 0,  0, -1,  1 ], # z
    ]
).T

circle      =   [ [0,1], [1,2], [2,3], [3,0] ]
triangles   =   np.array( [ edge + [pole] for edge in circle for pole in [4,5] ] )
edges       =   oat.simplex.dimension_m_faces_for_simplices( triangles, m=1 )

trace       =   oat.plot.trace_3d_for_edges( edges=edges, points=points )
trace.update(mode="lines+markers", line=dict(color="white", width=4), marker=dict(color="cornflowerblue", size=15), )
fig         =   go.Figure( [trace_octahedron, trace])

fig.update_layout(
    template="simple_white",
    height=700,     
    scene = dict(
        aspectratio=go.layout.scene.Aspectratio(x=1, y=1, z=1.5), # controls zoom
        xaxis = dict(range=[-1.2, 1.2],), # x axis limits
        yaxis = dict(range=[-1.2, 1.2],), # y axis limits
        zaxis = dict(range=[-1.2, 1.2],), # z axis limits
        camera=dict(
            projection=dict(
                type='orthographic'
            )
        )        
    ),    
)
fig

# %% [markdown]
# A wire sphere
# --------------------------------------------------------------------

# %%
data            =   oat.plot.wire_sphere_3d(0,0,0,1, nlattitude=5, nlongitude=4)
for n, trace in enumerate(data):
    trace.update( legendgroup="1", showlegend = (n==0), line=dict(color="white", width=3) )

fig = go.Figure(data=data)
fig.update_layout( title=dict(text="Wireframe sphere"), height=700, template="plotly_dark",)

fig.update_layout(scene = dict(xaxis = dict(showgrid = False,showticklabels = False, ),
                                   yaxis = dict(showgrid = False,showticklabels = False),
                                   zaxis = dict(showgrid = False,showticklabels = False)
             ))

fig

# %% [markdown]
# Simplices
# --------------------------------------------------------------------

# %%
ppoints      =   np.random.rand(10,3)
simplices   =   [list(x) for x in itertools.combinations( range(5),4)]
triangles   =   oat.simplex.dimension_m_faces_for_simplices( simplices, m=2 )
trace       =   oat.plot.trace_3d_for_triangles( triangles, points=ppoints )

trace.update(intensity=np.random.rand(10))
data = [trace]
fig = go.Figure(data)
fig

# %% [markdown]
# Rectangle
# --------------------------------------------------------------------

# %%
trace, x, y, z = oat.plot.surface_rectangle( -1,1, -2,2, -4, 4 )
trace.update( surfacecolor = x * y * z )


fig = go.Figure( data=[trace] )
fig.update_layout( height=800 )
fig.update_layout(
    title = f"Rectangle",
    scene = dict(
        # aspectmode = "cube",
        aspectratio=go.layout.scene.Aspectratio(x=2, y=2, z=2),
        xaxis = dict(range=[-5, 5],),
        yaxis = dict(range=[-5, 5],),
        zaxis = dict(range=[-5, 5],),                
    )
)
fig

# %% [markdown]
# Cubes
# --------------------------------------------------------------------

# %%
#   PLOT THE POINT CLOUD
data = []
trace = go.Scatter3d(x=points[:,0],y=points[:,1],z=points[:,2], mode="markers", marker=dict(symbol="circle-open", color=points[:,0], colorscale="rainbow"), name="Point cloud")
data.append(trace)

#   PLOT THE HYPEREDGES

for counter, point in enumerate(points):
    trace, x, y, z = oat.plot.surface_cube( point[0],point[1],point[2], width=1, anchor="center")
    trace.update(opacity=1, showscale=False, showlegend=True, name=f"Edge {counter}" )
    trace.update(surfacecolor = x, cmin=-2, cmax=2, colorscale="greys")
    
    # hide some cubes (but you can toggle them back on)
    if counter %2 ==0 : trace.update(visible='legendonly',) 
    data.append(trace)

fig = go.Figure(data)
fig.update_layout( height=800 )
fig.update_layout(
    title = f"Toggle the cubes",
    scene = dict(
        # aspectmode = "cube",
        aspectratio=go.layout.scene.Aspectratio(x=2, y=2, z=2),
        xaxis = dict(range=[-2.5,2.5],),
        yaxis = dict(range=[-2.5,2.5],),
        zaxis = dict(range=[-2.5,2.5],),                
    )
)
fig


