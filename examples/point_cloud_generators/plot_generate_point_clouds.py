"""
Point Cloud Generators
=========================================

This gallery shows how to generate several different types of point clouds.
"""


import oat_python as oat
import plotly.graph_objects as go


# %%
# Stanford Dragon
# ----------------------------------------

points              =   oat.point_cloud.stanford_dragon()

trace               =   go.Scatter3d(
                            x=points[:,0],
                            y=points[:,1],
                            z=points[:,2], 
                            mode="markers", 
                            marker=dict(opacity=1, size=4, color=points[:,2], colorscale="Aggrnyl")
                        )
fig                 =   go.Figure(data=trace)
fig.update_layout(
    height=500,
) 
fig


# %%
# Fibonacci sphere
# ----------------------------------------

# %% 
# Whole sphere:
points              =   oat.point_cloud.sphere_or_slice(
                            n_points=300, 
                        )

trace               =   go.Scatter3d(
                            x=points[:,0],
                            y=points[:,1],
                            z=points[:,2], 
                            mode="markers", 
                            marker=dict(opacity=1, size=4, color=points[:,2], colorscale="Aggrnyl")
                        )
fig                 =   go.Figure(data=trace)
fig.update_layout(
    height=500,
) 
fig

# %% 
# Slice of sphere:
# ----------------------------------------

points              =   oat.point_cloud.sphere_or_slice(
                            n_points=300, 
                            xmin=-0.5,
                            xmax=0.5,
                        )

trace               =   go.Scatter3d(
                            x=points[:,0],
                            y=points[:,1],
                            z=points[:,2], 
                            mode="markers", 
                            marker=dict(opacity=1, size=4, color=points[:,2], colorscale="Aggrnyl")
                        )
fig                 =   go.Figure(data=trace)
fig.update_layout(
    height=500,
) 
fig


# %%
# Spiral sphere
# ----------------------------------------

points              =   oat.point_cloud.sphere_or_slice_spiral(
                            n_points=300, 
                            noise_scale=0.07, 
                            random_seed=0
                        )

trace               =   go.Scatter3d(
                            x=points[:,0],
                            y=points[:,1],
                            z=points[:,2], 
                            mode="markers", 
                            marker=dict(opacity=1, size=4, color=points[:,2], colorscale="Aggrnyl")
                        )
fig                 =   go.Figure(data=trace)
fig.update_layout(
    height=500
) 
fig

# %%
# Torus
# ----------------------------------------


points              =   oat.point_cloud.torus(
                            radius_outer = 4, 
                            radius_inner = 1, 
                            n_points_outer = 60, 
                            n_points_inner = 20, 
                            repeat_last = True,
                        )
trace               =   go.Scatter3d(
                            x=points[:,0],
                            y=points[:,1],
                            z=points[:,2], 
                            mode="markers", 
                            marker=dict(opacity=1, size=4, color=points[:,2], colorscale="Aggrnyl")
                        )
fig                 =   go.Figure(data=trace)
fig.update_layout(
    height=500,
    scene = dict(
        aspectratio=go.layout.scene.Aspectratio(x=1, y=1, z=1),
        xaxis = dict(range=[-5,5],), # x axis limits
        yaxis = dict(range=[-5,5],), # y axis limits
        zaxis = dict(range=[-5,5],), # z axis limits        
    ),    
) 
fig



# %%
# Torus curve
# ----------------------------------------

points              =   oat.point_cloud.torus_curve(
                            n_points=300,
                            inner_radius=0.2,
                            outer_radius=1.0,
                            angle_initial=0,
                            nturns=10,
                        )

trace               =   go.Scatter3d(
                            x=points[:,0],
                            y=points[:,1],
                            z=points[:,2], 
                            mode="markers", 
                            marker=dict(opacity=1, size=4, color=points[:,2], colorscale="Aggrnyl")
                        )
fig                 =   go.Figure(data=trace)
fig.update_layout(
    height=500,
    scene = dict(
        aspectratio=go.layout.scene.Aspectratio(x=1, y=1, z=1),
        xaxis = dict(range=[-1.5,1.5],), # x axis limits
        yaxis = dict(range=[-1.5,1.5],), # y axis limits
        zaxis = dict(range=[-1.5,1.5],), # z axis limits        
    ),    
) 
fig


# %%
# Annulus
# ----------------------------------------


points              =   oat.point_cloud.annulus(
                            n_points=300,
                            inner_radius=0.5,
                            outer_radius=1.0,
                            random_seed=0
                        )

trace               =   go.Scatter(
                            x=points[:,0],
                            y=points[:,1],
                            mode="markers", 
                            marker=dict(opacity=1, size=4, color=points[:,1], colorscale="Aggrnyl")
                        )
fig                 =   go.Figure(data=trace)
fig.update_layout(
    yaxis=dict(           # Ensure an equal x-y aspect ratio by ..
        scaleanchor="x",  #   Anchoring y-axis scaling to x-axis
        scaleratio=1.0    #   Setting the aspect ratio (1.0 for square aspect)
    ),
)
fig


# %%
# Circle
# ----------------------------------------

# %%
# Uniform spacing:
points              =   oat.point_cloud.circle(
                            n_points=80,
                            radius=1.0,
                            mode="uniform",
                        )

trace               =   go.Scatter(
                            x=points[:,0],
                            y=points[:,1],
                            mode="markers", 
                            marker=dict(opacity=1, size=4, color=points[:,1], colorscale="Aggrnyl")
                        )
fig                 =   go.Figure(data=trace)
fig.update_layout(
    yaxis=dict(           # Ensure an equal x-y aspect ratio by ..
        scaleanchor="x",  #   Anchoring y-axis scaling to x-axis
        scaleratio=1.0    #   Setting the aspect ratio (1.0 for square aspect)
    ),
)
fig

# %%
# Random spacing:
points              =   oat.point_cloud.circle(
                            n_points=80,
                            radius=1.0,
                            mode="random",
                            random_seed=0
                        )

trace               =   go.Scatter(
                            x=points[:,0],
                            y=points[:,1],
                            mode="markers", 
                            marker=dict(opacity=1, size=4, color=points[:,1], colorscale="Aggrnyl")
                        )
fig                 =   go.Figure(data=trace)
fig.update_layout(
    yaxis=dict(           # Ensure an equal x-y aspect ratio by ..
        scaleanchor="x",  #   Anchoring y-axis scaling to x-axis
        scaleratio=1.0    #   Setting the aspect ratio (1.0 for square aspect)
    ),
)
fig



# %%
# Disk
# ----------------------------------------

points              =   oat.point_cloud.disk(
                            n_points=200,
                            radius=1.0,
                            random_seed=0,
                        )

trace               =   go.Scatter(
                            x=points[:,0],
                            y=points[:,1],
                            mode="markers", 
                            marker=dict(opacity=1, size=4, color=points[:,1], colorscale="Aggrnyl")
                        )
fig                 =   go.Figure(data=trace)
fig.update_layout(
    yaxis=dict(           # Ensure an equal x-y aspect ratio by ..
        scaleanchor="x",  #   Anchoring y-axis scaling to x-axis
        scaleratio=1.0    #   Setting the aspect ratio (1.0 for square aspect)
    ),
)
fig



# %%
# Two circles
# ----------------------------------------

points              =   oat.point_cloud.two_circles()

trace               =   go.Scatter(
                            x=points[:,0],
                            y=points[:,1],
                            mode="markers", 
                            marker=dict(opacity=1, size=4, color=points[:,1], colorscale="Aggrnyl")
                        )
fig                 =   go.Figure(data=trace)
fig.update_layout(
    yaxis=dict(           # Ensure an equal x-y aspect ratio by ..
        scaleanchor="x",  #   Anchoring y-axis scaling to x-axis
        scaleratio=1.0    #   Setting the aspect ratio (1.0 for square aspect)
    ),
)
fig