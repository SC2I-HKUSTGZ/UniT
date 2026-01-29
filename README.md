# Academic Paper Project Page Template

A modern, responsive template for creating academic paper project pages on GitHub Pages. Inspired by top-tier conference paper websites like [D4RT](https://d4rt-paper.github.io/), [MV-DUSt3R+](https://mv-dust3rp.github.io/), and [MapAnything](https://map-anything.github.io/).

## Features

- **Modern Dark Theme**: Clean, professional design with customizable color scheme
- **Responsive Layout**: Works on desktop, tablet, and mobile devices
- **Interactive 3D Viewer**: Three.js-powered point cloud/mesh visualization
- **Video Support**: Teaser videos and result video galleries
- **Data Visualization**: Chart.js integration for performance comparisons
- **Citation Copy**: One-click BibTeX citation copying
- **Smooth Animations**: Scroll reveal and hover effects

## Quick Start

### 1. Clone or Download

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```

### 2. Customize Content

Edit `index.html` to update:

- **Title & Authors**: Update paper title, author names, and affiliations
- **Conference Badge**: Change to your target venue (CVPR, ICCV, NeurIPS, etc.)
- **Links**: Add links to PDF, arXiv, code repository, video, dataset
- **Abstract**: Replace with your paper abstract
- **Method Section**: Update method description and figures
- **Results**: Update quantitative results table and charts
- **Comparison**: Add your comparison images/videos
- **Citation**: Update BibTeX entry

### 3. Add Assets

Place your files in the `assets/` directory:

```
assets/
├── teaser.mp4              # Teaser video
├── method.png              # Method overview figure
├── comparison/
│   ├── input.png           # Input images
│   ├── baseline1.png       # Baseline comparisons
│   ├── baseline2.png
│   └── ours.png            # Your results
└── videos/
    ├── result1.mp4         # Result videos
    ├── result2.mp4
    └── result3.mp4
```

### 4. Deploy to GitHub Pages

```bash
# Initialize git repository (if not already)
git init
git add .
git commit -m "Initial commit"

# Push to GitHub
git remote add origin git@github.com:YOUR_USERNAME/YOUR_REPO.git
git branch -M main
git push -u origin main
```

Then enable GitHub Pages in your repository settings:
1. Go to Settings → Pages
2. Select "Deploy from a branch"
3. Choose "main" branch and "/ (root)" folder
4. Click Save

Your site will be available at `https://YOUR_USERNAME.github.io/YOUR_REPO/`

## Customization

### Colors

Edit CSS variables in `style.css`:

```css
:root {
    --color-primary: #6366f1;        /* Primary accent color */
    --color-primary-dark: #4f46e5;
    --color-primary-light: #818cf8;
    --color-secondary: #10b981;       /* Secondary color */
    --color-bg: #0f0f0f;              /* Background color */
    /* ... more variables */
}
```

### 3D Viewer

The template includes a Three.js-based 3D viewer that supports:

- Point clouds (demo data included)
- Meshes
- Gaussian splat visualization (simplified)

To load your own PLY point cloud:

```javascript
// In script.js or inline
const geometry = await window.PaperTemplate.loadPLYPointCloud('path/to/your/pointcloud.ply');
```

### Charts

Update chart data in `script.js`:

```javascript
// Performance comparison chart
data: {
    labels: ['Method1', 'Method2', 'Ours'],
    datasets: [{
        data: [26.5, 28.7, 31.7],
        // ...
    }]
}
```

## File Structure

```
webpage/
├── index.html          # Main HTML file
├── style.css           # Styles
├── script.js           # JavaScript (3D viewer, charts, interactions)
├── README.md           # This file
└── assets/
    ├── teaser.mp4
    ├── method.png
    ├── comparison/
    └── videos/
```

## Dependencies (CDN)

- [Inter Font](https://fonts.google.com/specimen/Inter) - Typography
- [JetBrains Mono](https://fonts.google.com/specimen/JetBrains+Mono) - Code font
- [Font Awesome 6](https://fontawesome.com/) - Icons
- [Three.js r128](https://threejs.org/) - 3D visualization
- [Chart.js](https://www.chartjs.org/) - Data charts

## Browser Support

- Chrome (recommended)
- Firefox
- Safari
- Edge

## License

MIT License - Feel free to use this template for your academic papers!

## Acknowledgements

Template design inspired by:
- [D4RT](https://d4rt-paper.github.io/) - Google DeepMind
- [MV-DUSt3R+](https://mv-dust3rp.github.io/) - Meta Reality Labs
- [MapAnything](https://map-anything.github.io/) - Meta Reality Labs

---

Made with ❤️ for the research community
