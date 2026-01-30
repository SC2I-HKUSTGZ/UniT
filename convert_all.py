#!/usr/bin/env python3
"""Convert PLY files to optimized splat format for web display."""
import struct
import numpy as np
import os

def parse_ply(filepath):
    """Parse a binary PLY file and extract vertices with positions and colors."""
    with open(filepath, 'rb') as f:
        header = b''
        while True:
            line = f.readline()
            header += line
            if b'end_header' in line:
                break
        
        header_str = header.decode('utf-8')
        vertex_count = 0
        for line in header_str.split('\n'):
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[-1])
                break
        
        print(f"  Found {vertex_count:,} vertices")
        
        # Determine format - check for properties
        has_nx = 'property float nx' in header_str or 'property float normal_x' in header_str
        has_color = 'red' in header_str or 'diffuse_red' in header_str
        
        # Count float and uchar properties to determine struct format
        props = []
        for line in header_str.split('\n'):
            if line.startswith('property float'):
                props.append('f')
            elif line.startswith('property uchar') or line.startswith('property uint8'):
                props.append('B')
            elif line.startswith('property double'):
                props.append('d')
        
        # Build struct format
        struct_format = '<' + ''.join(props)
        struct_size = struct.calcsize(struct_format)
        
        vertices = []
        for i in range(vertex_count):
            data = f.read(struct_size)
            if len(data) < struct_size:
                break
            values = struct.unpack(struct_format, data)
            
            # First 3 floats are always x, y, z
            x, y, z = values[0], values[1], values[2]
            
            # Find color values (usually last 3 or 4 uchars)
            uchars = [v for v in values if isinstance(v, int) and 0 <= v <= 255]
            if len(uchars) >= 3:
                r, g, b = uchars[0], uchars[1], uchars[2]
            else:
                r, g, b = 128, 128, 128
            
            vertices.append((x, y, z, r, g, b))
        
        return np.array(vertices, dtype=np.float32)

def float32_to_float16_bytes(val):
    """Convert float32 to float16 bytes."""
    return struct.pack('<e', val)

def convert_to_splat(vertices, output_path, sample_ratio=0.6):
    """Convert vertices to optimized splat format."""
    # Downsample by ratio
    target_points = int(len(vertices) * sample_ratio)
    if target_points < len(vertices):
        indices = np.random.choice(len(vertices), target_points, replace=False)
        vertices = vertices[indices]
        print(f"  Downsampled to {target_points:,} points ({sample_ratio*100:.0f}%)")
    
    # Keep original scale, only center the point cloud
    positions = vertices[:, :3]
    center = np.mean(positions, axis=0)
    positions = positions - center  # Center only, no scaling
    
    max_extent = np.max(np.abs(positions))
    print(f"  Max extent from center: {max_extent:.3f} (original units preserved)")
    
    # Write binary splat format: [x_f16, y_f16, z_f16, r_u8, g_u8, b_u8, a_u8] = 10 bytes per point
    with open(output_path, 'wb') as f:
        # Write point count as uint32
        f.write(struct.pack('<I', len(vertices)))
        
        for i in range(len(vertices)):
            x, y, z = positions[i]
            r, g, b = int(vertices[i, 3]), int(vertices[i, 4]), int(vertices[i, 5])
            
            # Write position as float16
            f.write(float32_to_float16_bytes(x))
            f.write(float32_to_float16_bytes(y))
            f.write(float32_to_float16_bytes(z))
            
            # Write color as uint8
            f.write(struct.pack('BBBB', r, g, b, 255))
    
    file_size = os.path.getsize(output_path)
    print(f"  Saved to {output_path} ({file_size / 1024 / 1024:.2f} MB)")

def main():
    source_dir = '/Users/bronya/Desktop/webpage/source_assets'
    output_dir = '/Users/bronya/Desktop/webpage/assets'
    
    # Files to convert (name -> (ply_file, preview_file, scene_name))
    files = [
        ('HKUST_redbird', 'HKUST_redbird.ply', 'HKUST_redbird.png', 'scene1'),
        ('HKUST_INTR', 'HKUST_INTR.ply', 'HKUST_INTR.png', 'scene2'),
        ('HKUST_toy', 'HKUST_toy.ply', 'HKUST_toy.png', 'scene3'),
        ('cartoon', 'cartoon.ply', 'cartoon.jpg', 'scene4'),
        ('room', 'room.ply', 'room.png', 'scene5'),
    ]
    
    os.makedirs(f'{output_dir}/samples', exist_ok=True)
    
    for name, ply_file, preview_file, scene_name in files:
        ply_path = f'{source_dir}/{ply_file}'
        splat_path = f'{output_dir}/{scene_name}.splat'
        
        print(f"\nProcessing {name}...")
        
        if os.path.exists(ply_path):
            try:
                vertices = parse_ply(ply_path)
                convert_to_splat(vertices, splat_path)
            except Exception as e:
                print(f"  Error: {e}")
        else:
            print(f"  PLY file not found: {ply_path}")
        
        # Copy preview image
        preview_src = f'{source_dir}/{preview_file}'
        # Use appropriate extension
        ext = preview_file.split('.')[-1]
        preview_dst = f'{output_dir}/samples/{scene_name}.{ext}'
        if os.path.exists(preview_src):
            import shutil
            shutil.copy(preview_src, preview_dst)
            print(f"  Copied preview to {preview_dst}")
    
    print("\nDone!")

if __name__ == '__main__':
    main()
