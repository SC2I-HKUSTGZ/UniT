"""
UniT: Unified Geometry Learner - Hugging Face Space Demo
A clean, minimal UI for 3D reconstruction from images/videos.
"""

import os
import cv2
import numpy as np
import gradio as gr
import shutil
import tempfile
from datetime import datetime
from PIL import Image

# For 3D visualization
import trimesh
from scipy.spatial.transform import Rotation


# =============================================================================
# Utility Functions
# =============================================================================

def create_placeholder_glb():
    """Create a simple placeholder 3D scene."""
    # Create a simple colored cube as placeholder
    mesh = trimesh.creation.box(extents=[1, 1, 1])
    
    # Add some color variation
    colors = np.array([
        [100, 149, 237, 255],  # Cornflower blue
    ] * len(mesh.faces))
    mesh.visual.face_colors = colors
    
    # Create scene
    scene = trimesh.Scene()
    scene.add_geometry(mesh)
    
    return scene


def generate_demo_pointcloud(images):
    """
    Generate a demo point cloud from images (placeholder algorithm).
    In production, this would be replaced with actual model inference.
    """
    all_points = []
    all_colors = []
    
    for idx, img in enumerate(images):
        if isinstance(img, str):
            img = Image.open(img)
        if not isinstance(img, Image.Image):
            continue
            
        img = img.convert('RGB')
        img_array = np.array(img)
        h, w = img_array.shape[:2]
        
        # Sample points from image
        num_points = min(5000, h * w)
        indices = np.random.choice(h * w, num_points, replace=False)
        ys, xs = np.unravel_index(indices, (h, w))
        
        # Get brightness for depth simulation
        brightness = np.mean(img_array, axis=2)
        depths = brightness[ys, xs] / 255.0 * 3 + idx * 0.5
        
        # Project to 3D
        fx, fy = w, h
        cx, cy = w / 2, h / 2
        
        points = np.zeros((num_points, 3))
        points[:, 0] = (xs - cx) / fx * depths
        points[:, 1] = -(ys - cy) / fy * depths
        points[:, 2] = depths
        
        colors = img_array[ys, xs, :]
        
        all_points.append(points)
        all_colors.append(colors)
    
    if not all_points:
        # Return default placeholder
        return np.array([[0, 0, 0]]), np.array([[128, 128, 128]])
    
    return np.vstack(all_points), np.vstack(all_colors)


def predictions_to_glb(points, colors, conf_thres=0.0, show_cam=True, camera_poses=None):
    """Convert point cloud predictions to a GLB scene."""
    
    # Filter by confidence if needed (placeholder - no actual confidence here)
    vertices_3d = points
    colors_rgb = colors
    
    if len(vertices_3d) == 0:
        vertices_3d = np.array([[0, 0, 0]])
        colors_rgb = np.array([[128, 128, 128]])
    
    # Create scene
    scene = trimesh.Scene()
    
    # Add point cloud
    point_cloud = trimesh.PointCloud(
        vertices=vertices_3d,
        colors=np.hstack([colors_rgb, np.full((len(colors_rgb), 1), 255)])
    )
    scene.add_geometry(point_cloud)
    
    # Add camera visualizations if requested
    if show_cam and camera_poses is not None:
        for i, pose in enumerate(camera_poses):
            add_camera_to_scene(scene, pose, i, len(camera_poses))
    
    # Apply rotation for better initial view
    align_rotation = np.eye(4)
    align_rotation[:3, :3] = Rotation.from_euler("x", 180, degrees=True).as_matrix()
    scene.apply_transform(align_rotation)
    
    return scene


def add_camera_to_scene(scene, pose, idx, total):
    """Add a camera visualization to the scene."""
    import matplotlib.cm as cm
    
    colormap = cm.get_cmap("rainbow")
    rgba = colormap(idx / max(total - 1, 1))
    color = tuple(int(255 * c) for c in rgba[:3])
    
    # Create small camera cone
    cam_size = 0.1
    cone = trimesh.creation.cone(radius=cam_size * 0.5, height=cam_size, sections=4)
    
    # Transform to camera position
    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler("z", 45, degrees=True).as_matrix()
    rot[2, 3] = -cam_size
    
    transform = pose @ rot
    cone.apply_transform(transform)
    cone.visual.face_colors = [color + (255,)] * len(cone.faces)
    
    scene.add_geometry(cone)


def write_ply(points, colors, path):
    """Write point cloud to PLY file."""
    with open(path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        for i in range(len(points)):
            x, y, z = points[i]
            r, g, b = colors[i].astype(int)
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")


# =============================================================================
# Upload Handling
# =============================================================================

def handle_uploads(input_video, input_images, interval=None):
    """Process uploaded video/images and return target directory with image paths."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target_dir = os.path.join(tempfile.gettempdir(), f"unit_demo_{timestamp}")
    images_dir = os.path.join(target_dir, "images")
    
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(images_dir, exist_ok=True)
    
    image_paths = []
    
    # Handle image uploads
    if input_images is not None:
        img_interval = max(1, int(interval)) if interval and interval > 0 else 1
        for i, file_data in enumerate(input_images[::img_interval]):
            if isinstance(file_data, dict) and "name" in file_data:
                file_path = file_data["name"]
            else:
                file_path = file_data
            
            dst_path = os.path.join(images_dir, f"{i:06d}.png")
            shutil.copy(file_path, dst_path)
            image_paths.append(dst_path)
    
    # Handle video upload
    if input_video is not None:
        if isinstance(input_video, dict) and "name" in input_video:
            video_path = input_video["name"]
        else:
            video_path = input_video
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(interval)) if interval and interval > 0 else max(1, int(fps))
        
        count = 0
        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            count += 1
            if count % frame_interval == 0:
                path = os.path.join(images_dir, f"{frame_num:06d}.png")
                cv2.imwrite(path, frame)
                image_paths.append(path)
                frame_num += 1
        cap.release()
    
    return target_dir, sorted(image_paths)


def update_gallery_on_upload(input_video, input_images, interval):
    """Update gallery when files are uploaded."""
    if not input_video and not input_images:
        return None, None, "Upload media to begin."
    
    target_dir, image_paths = handle_uploads(input_video, input_images, interval)
    return target_dir, image_paths, f"Loaded {len(image_paths)} images. Click 'Reconstruct' to generate 3D model."


# =============================================================================
# Reconstruction Pipeline
# =============================================================================

def reconstruct(target_dir, conf_thres=20.0, show_cam=True):
    """
    Perform 3D reconstruction from uploaded images.
    Currently uses placeholder algorithm - will be replaced with actual model.
    """
    if not target_dir or target_dir == "None" or not os.path.isdir(target_dir):
        return None, None, "No valid upload found. Please upload media first."
    
    images_dir = os.path.join(target_dir, "images")
    if not os.path.isdir(images_dir):
        return None, None, "No images found. Please upload media first."
    
    # Load images
    image_files = sorted([
        os.path.join(images_dir, f) 
        for f in os.listdir(images_dir) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    
    if len(image_files) == 0:
        return None, None, "No images found in upload."
    
    # Generate point cloud (placeholder)
    points, colors = generate_demo_pointcloud(image_files)
    
    # Create fake camera poses for visualization
    num_frames = len(image_files)
    camera_poses = []
    for i in range(num_frames):
        pose = np.eye(4)
        angle = i / max(num_frames - 1, 1) * np.pi * 0.5
        pose[:3, :3] = Rotation.from_euler("y", np.degrees(angle), degrees=True).as_matrix()
        pose[2, 3] = -2  # Move cameras back
        camera_poses.append(pose)
    
    # Convert to GLB
    scene = predictions_to_glb(
        points, colors,
        conf_thres=conf_thres / 100.0,
        show_cam=show_cam,
        camera_poses=camera_poses
    )
    
    # Save GLB file
    glb_path = os.path.join(target_dir, "reconstruction.glb")
    scene.export(file_obj=glb_path)
    
    # Save PLY file
    ply_path = os.path.join(target_dir, "reconstruction.ply")
    write_ply(points, colors, ply_path)
    
    return glb_path, ply_path, f"Reconstruction complete! Generated {len(points):,} points from {len(image_files)} images."


def update_visualization(target_dir, conf_thres, show_cam):
    """Update visualization with new parameters."""
    if not target_dir or target_dir == "None" or not os.path.isdir(target_dir):
        return None, None, "No reconstruction available."
    
    # Check for saved predictions
    images_dir = os.path.join(target_dir, "images")
    if not os.path.isdir(images_dir):
        return None, None, "No reconstruction available."
    
    # Re-run reconstruction with new parameters
    return reconstruct(target_dir, conf_thres, show_cam)


def clear_all():
    """Clear all outputs."""
    return None, None, None, None, "Ready for new upload."


# =============================================================================
# Gradio UI
# =============================================================================

# Clean, minimal CSS (white theme)
CUSTOM_CSS = """
/* Clean white theme */
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: #fafafa;
}

/* Header styling */
.header-title {
    text-align: center;
    font-size: 2rem;
    font-weight: 700;
    color: #1a1a1a;
    margin-bottom: 0.5rem;
}

.header-subtitle {
    text-align: center;
    color: #666;
    font-size: 1rem;
    margin-bottom: 1.5rem;
}

.header-links {
    text-align: center;
    margin-bottom: 1rem;
}

.header-links a {
    color: #2563eb;
    text-decoration: none;
    margin: 0 0.75rem;
    font-weight: 500;
}

.header-links a:hover {
    text-decoration: underline;
}

/* Status message */
.status-msg {
    text-align: center;
    padding: 1rem;
    font-size: 1.1rem;
    color: #333;
    background: #f0f9ff;
    border-radius: 8px;
    border: 1px solid #bae6fd;
}

/* Section headers */
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #374151;
    margin-bottom: 0.75rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #e5e7eb;
}

/* Note box */
.note-box {
    background: #fffbeb;
    border: 1px solid #fcd34d;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    font-size: 0.9rem;
    color: #92400e;
    margin-top: 1rem;
}
"""

# Build the Gradio interface
with gr.Blocks(
    theme=gr.themes.Soft(),
    css=CUSTOM_CSS,
    title="UniT - 3D Reconstruction"
) as demo:
    
    # Hidden state
    target_dir_state = gr.Textbox(visible=False, value="None")
    
    # Header
    gr.HTML("""
        <div class="header-title">UniT: Unified Geometry Learner</div>
        <div class="header-subtitle">Transform images and videos into interactive 3D point clouds</div>
        <div class="header-links">
            <a href="https://sc2i-hkustgz.github.io/UniT/" target="_blank">Project Page</a>
            <a href="#" target="_blank">Paper</a>
            <a href="#" target="_blank">Code</a>
        </div>
    """)
    
    # Status message
    status_output = gr.Markdown("Upload images or video to begin.", elem_classes=["status-msg"])
    
    with gr.Row():
        # Left column: Input
        with gr.Column(scale=1):
            gr.Markdown("### Upload Media", elem_classes=["section-header"])
            
            input_video = gr.Video(label="Upload Video", interactive=True)
            input_images = gr.File(
                file_count="multiple",
                label="Or Upload Images",
                file_types=["image"],
                interactive=True
            )
            interval_input = gr.Number(
                value=None,
                label="Frame/Image Interval",
                info="Sampling interval. Video default: 1 FPS. Images default: all.",
                minimum=1,
                step=1
            )
            
            # Image preview gallery
            image_gallery = gr.Gallery(
                label="Preview",
                columns=4,
                height=200,
                object_fit="contain",
                show_download_button=False
            )
        
        # Right column: Output
        with gr.Column(scale=2):
            gr.Markdown("### 3D Reconstruction", elem_classes=["section-header"])
            
            # 3D model viewer
            model_output = gr.Model3D(
                label="Interactive 3D View (drag to rotate, scroll to zoom)",
                height=450,
                clear_color=[1.0, 1.0, 1.0, 1.0]  # White background
            )
            
            # Download option
            ply_download = gr.File(label="Download PLY", interactive=False)
    
    # Control buttons
    with gr.Row():
        reconstruct_btn = gr.Button("Reconstruct", variant="primary", scale=3)
        clear_btn = gr.Button("Clear", scale=1)
    
    # Visualization controls
    with gr.Accordion("Visualization Settings", open=False):
        with gr.Row():
            conf_slider = gr.Slider(
                minimum=0,
                maximum=100,
                value=20,
                step=1,
                label="Confidence Threshold (%)",
                info="Filter low-confidence points"
            )
            show_cam_checkbox = gr.Checkbox(
                label="Show Cameras",
                value=True,
                info="Display camera positions in 3D view"
            )
    
    # Note
    gr.HTML("""
        <div class="note-box">
            <strong>Note:</strong> This is a demo with placeholder outputs. 
            The actual UniT model will be integrated soon for real 3D reconstruction.
        </div>
    """)
    
    # ==========================================================================
    # Event Handlers
    # ==========================================================================
    
    # Auto-update gallery on upload
    input_video.change(
        fn=update_gallery_on_upload,
        inputs=[input_video, input_images, interval_input],
        outputs=[target_dir_state, image_gallery, status_output]
    )
    
    input_images.change(
        fn=update_gallery_on_upload,
        inputs=[input_video, input_images, interval_input],
        outputs=[target_dir_state, image_gallery, status_output]
    )
    
    # Reconstruct button
    reconstruct_btn.click(
        fn=lambda: "Processing...",
        outputs=[status_output]
    ).then(
        fn=reconstruct,
        inputs=[target_dir_state, conf_slider, show_cam_checkbox],
        outputs=[model_output, ply_download, status_output]
    )
    
    # Clear button
    clear_btn.click(
        fn=clear_all,
        outputs=[model_output, ply_download, image_gallery, target_dir_state, status_output]
    )
    
    # Real-time visualization updates
    conf_slider.change(
        fn=update_visualization,
        inputs=[target_dir_state, conf_slider, show_cam_checkbox],
        outputs=[model_output, ply_download, status_output]
    )
    
    show_cam_checkbox.change(
        fn=update_visualization,
        inputs=[target_dir_state, conf_slider, show_cam_checkbox],
        outputs=[model_output, ply_download, status_output]
    )


# Launch
if __name__ == "__main__":
    demo.queue(max_size=10).launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
