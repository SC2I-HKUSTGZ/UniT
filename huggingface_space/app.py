import gradio as gr
import numpy as np
from PIL import Image
import plotly.graph_objects as go


def create_pointcloud_figure(points, colors):
    """Create a Plotly 3D scatter plot for point cloud visualization."""
    fig = go.Figure(data=[go.Scatter3d(
        x=points[:, 0].tolist(),
        y=points[:, 1].tolist(),
        z=points[:, 2].tolist(),
        mode='markers',
        marker=dict(
            size=2,
            color=colors,
            opacity=0.8
        )
    )])
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(showgrid=False, showticklabels=False, title=''),
            yaxis=dict(showgrid=False, showticklabels=False, title=''),
            zaxis=dict(showgrid=False, showticklabels=False, title=''),
            aspectmode='data',
            bgcolor='white'
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='white',
        height=500
    )
    
    return fig


def create_placeholder_figure():
    """Create a placeholder figure when no image is uploaded."""
    fig = go.Figure()
    fig.add_annotation(
        text="📷 Upload an image and click Generate<br>to create 3D point cloud",
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16, color="#666"),
        align="center"
    )
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor='#f5f7fa',
        paper_bgcolor='#f5f7fa',
        height=500,
        margin=dict(l=0, r=0, t=0, b=0)
    )
    return fig


def generate_demo_pointcloud(image):
    """Generate a demo point cloud from image (placeholder)."""
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    # Sample points from image
    num_points = min(25000, h * w)
    indices = np.random.choice(h * w, num_points, replace=False)
    ys, xs = np.unravel_index(indices, (h, w))
    
    # Get brightness for depth
    if len(img_array.shape) == 3:
        brightness = np.mean(img_array, axis=2)
    else:
        brightness = img_array
    
    depths = brightness[ys, xs] / 255.0 * 5
    
    # Project to 3D
    fx, fy = w, h
    cx, cy = w / 2, h / 2
    
    points = np.zeros((num_points, 3))
    points[:, 0] = (xs - cx) / fx * depths
    points[:, 1] = (ys - cy) / fy * depths
    points[:, 2] = depths
    
    # Get colors
    if len(img_array.shape) == 3:
        colors = img_array[ys, xs, :]
    else:
        colors = np.stack([img_array[ys, xs]] * 3, axis=1)
    
    color_strings = [f'rgb({r},{g},{b})' for r, g, b in colors]
    
    return points, color_strings


def process_image(image):
    """Process image and return point cloud visualization as Plotly figure."""
    if image is None:
        return create_placeholder_figure()
    
    try:
        points, colors = generate_demo_pointcloud(image)
        fig = create_pointcloud_figure(points, colors)
        return fig
    except Exception as e:
        # Return error as a figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="#c00"),
            align="center"
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            plot_bgcolor='#fff0f0',
            paper_bgcolor='#fff0f0',
            height=500,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        return fig


# Use gr.Blocks for more control over the interface
with gr.Blocks(title="UniT: Unified Geometry Learner", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎯 UniT: Unified Geometry Learner
    
    Upload an image to generate a 3D point cloud reconstruction.
    
    > ⚠️ **Note**: This is a demo with placeholder outputs. The actual model will be integrated soon.
    
    **Links**: [Project Page](https://sc2i-hkustgz.github.io/UniT/) | Paper | Code
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Upload Image")
            with gr.Row():
                clear_btn = gr.ClearButton([input_image], value="🗑️ Clear")
                submit_btn = gr.Button("🚀 Generate 3D", variant="primary")
        
        with gr.Column(scale=2):
            output_plot = gr.Plot(label="3D Point Cloud (drag to rotate, scroll to zoom)", value=create_placeholder_figure())
    
    submit_btn.click(fn=process_image, inputs=input_image, outputs=output_plot)
    input_image.clear(fn=lambda: create_placeholder_figure(), outputs=output_plot)

if __name__ == "__main__":
    demo.launch()
