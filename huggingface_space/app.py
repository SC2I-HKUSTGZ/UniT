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


# Initial placeholder HTML
PLACEHOLDER_HTML = """
<div style="height: 450px; display: flex; align-items: center; justify-content: center; 
            background: linear-gradient(135deg, #f5f7fa 0%, #e4e8eb 100%); 
            border-radius: 12px; color: #666; font-size: 16px; text-align: center;">
    <div>
        <div style="font-size: 48px; margin-bottom: 16px;">📷</div>
        <div>Upload an image and click Submit<br>to generate 3D point cloud</div>
    </div>
</div>
"""


def process_image(image):
    """Process image and return point cloud visualization as HTML."""
    if image is None:
        return PLACEHOLDER_HTML
    
    try:
        points, colors = generate_demo_pointcloud(image)
        fig = create_pointcloud_figure(points, colors)
        return fig.to_html(full_html=False, include_plotlyjs='cdn')
    except Exception as e:
        return f"""
        <div style="height: 450px; display: flex; align-items: center; justify-content: center; 
                    background: #fff0f0; border-radius: 12px; color: #c00; font-size: 16px;">
            Error processing image: {str(e)}
        </div>
        """


# Use simple gr.Interface
demo = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=gr.HTML(label="3D Point Cloud (drag to rotate, scroll to zoom)", value=PLACEHOLDER_HTML),
    title="🎯 UniT: Unified Geometry Learner",
    description="""
Upload an image to generate a 3D point cloud reconstruction.

> ⚠️ **Note**: This is a demo with placeholder outputs. The actual model will be integrated soon.

**Links**: [Project Page](https://sc2i-hkustgz.github.io/UniT/) | Paper | Code
""",
    allow_flagging="never",
    submit_btn="🚀 Generate 3D",
    clear_btn="🗑️ Clear"
)

if __name__ == "__main__":
    demo.launch()
