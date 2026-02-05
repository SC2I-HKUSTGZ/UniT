---
title: UniT - Unified Geometry Learner
emoji: 🎯
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.19.2
python_version: 3.11
app_file: app.py
pinned: false
license: apache-2.0
short_description: 3D reconstruction from images and videos using UniT
---

# UniT: Group Autoregressive Transformer As Unified Geometry Learner

Transform images and videos into interactive 3D point clouds.

## Features

- **Video Reconstruction**: Upload a video to extract frames and generate 3D model
- **Multi-Image Reconstruction**: Upload multiple images for 3D reconstruction  
- **Interactive 3D Viewer**: Rotate, zoom, and pan the GLB model
- **Adjustable Parameters**: Confidence threshold and camera visualization
- **Download Results**: Export as PLY file for further processing

## Usage

1. Upload a video or multiple images
2. Set frame/image sampling interval (optional)
3. Click "Reconstruct" to generate 3D model
4. Interact with the 3D visualization
5. Download PLY file if needed

## Links

- [Project Page](https://sc2i-hkustgz.github.io/UniT/)
- [Paper](#)
- [Code](#)

## Citation

```bibtex
@article{wang2026unit,
  title={UniT: Group Autoregressive Transformer As Unified Geometry Learner},
  author={Wang, Haotian and Huang, Yusong and Zheng, Xinhu},
  journal={arXiv preprint arXiv:2601.00000},
  year={2026}
}
```
