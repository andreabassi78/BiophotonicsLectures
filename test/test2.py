import plotly.graph_objects as go

# Create figure
fig = go.Figure()

# Constants
img_width = 532
img_height = 532
scale_factor = 0.5

# Add invisible scatter trace.
# This trace is added to help the autoresize logic work.
# fig.add_trace(
#     go.Scatter(
#         x=[0, img_width * scale_factor],
#         y=[0, img_height * scale_factor],
#         mode="markers",
#         marker_opacity=0
#     )
# )

# Configure axes
fig.update_xaxes(
    visible=False,
    range=[0, img_width * scale_factor]
)

fig.update_yaxes(
    visible=False,
    range=[0, img_height * scale_factor],
    # the scaleanchor attribute ensures that the aspect ratio stays constant
    scaleanchor="x"
)

# Add image
fig.add_layout_image(
    dict(
        x=0,
        sizex=img_width * scale_factor,
        y=img_height * scale_factor,
        sizey=img_height * scale_factor,
        xref="x",
        yref="y",
        opacity=1.0,
        layer="below",
        sizing="stretch",
        source="nyc.jpg")
)

# Configure other layout
fig.update_layout(
    width=img_width * scale_factor,
    height=img_height * scale_factor,
    margin={"l": 0, "r": 0, "t": 0, "b": 0},
)

# Disable the autosize on double click because it adds unwanted margins around the image
# More detail: https://plotly.com/python/configuration-options/
fig.show(config={'doubleClick': 'reset'})