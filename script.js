/* ========================================
   UniT project page — interactive examples
   Chapter nav + PLY point-cloud viewer + video+cover sync
   ======================================== */

document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initDemo();
});

// ========================================
// Navigation (Chapters)
// ========================================
function initNavigation() {
    const chapterBtns = document.querySelectorAll('.chapters button');
    chapterBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            chapterBtns.forEach(b => b.classList.remove('selected'));
            btn.classList.add('selected');
            const sectionId = btn.dataset.section;
            const target = document.getElementById(sectionId);
            if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        });
    });
}

// ========================================
// Per-demo configuration
// pointSize tuned to the density / extent of each cloud.
// flipY=true matches trimesh's export of the UniT outputs.
// camera is the (unit) offset from bbox center, scaled by bounding sphere radius.
// ========================================
const DEMO_CONFIGS = {
    hkust_intr: {
        title: 'HKUST INTR',
        ply: 'assets/demos/hkust_intr/scene.ply',
        video: 'assets/demos/hkust_intr/reconstructed.mp4',
        pointSize: 0.05,
        flipY: true,
        camera: { x: -0.6, y: 0.3, z: -1.4 }
    },
    hkust_toy: {
        title: 'HKUST Toy',
        ply: 'assets/demos/hkust_toy/scene.ply',
        video: 'assets/demos/hkust_toy/reconstructed.mp4',
        pointSize: 0.012,
        flipY: true,
        camera: { x: -0.4, y: 0.2, z: -1.5 }
    },
    hkust_redbird: {
        title: 'HKUST Red Bird',
        ply: 'assets/demos/hkust_redbird/scene.ply',
        video: 'assets/demos/hkust_redbird/reconstructed.mp4',
        pointSize: 0.14,
        flipY: true,
        camera: { x: -0.5, y: 0.35, z: -1.6 }
    },
    drift: {
        title: 'Drift',
        ply: 'assets/demos/drift/scene.ply',
        video: 'assets/demos/drift/reconstructed.mp4',
        pointSize: 0.3,
        flipY: true,
        camera: { x: -0.5, y: 0.25, z: -1.2 }
    },
    gta_sfm: {
        title: 'GTA SfM',
        ply: 'assets/demos/gta_sfm/scene.ply',
        video: 'assets/demos/gta_sfm/reconstructed.mp4',
        pointSize: 0.25,
        flipY: true,
        camera: { x: -0.4, y: 0.2, z: -1.4 }
    },
    kitti: {
        title: 'KITTI',
        ply: 'assets/demos/kitti/scene.ply',
        video: 'assets/demos/kitti/reconstructed.mp4',
        pointSize: 0.9,
        flipY: true,
        camera: { x: -0.3, y: 0.25, z: -0.9 }
    }
};

// ========================================
// Interactive Examples
// ========================================
function initDemo() {
    const canvas = document.getElementById('demo-canvas');
    if (!canvas) return;

    const viewer = new PlyViewer(canvas, document.getElementById('demo-message'));
    const video = document.getElementById('demo-video');
    const cover = document.getElementById('demo-cover');
    const coverImg = document.getElementById('demo-cover-img');
    const thumbs = document.querySelectorAll('.demo-thumb');

    let currentDemo = null;

    function showCoverFor(key) {
        const cfg = DEMO_CONFIGS[key];
        if (!cfg) return;
        coverImg.src = `assets/demos/${key}/cover.jpg`;
        video.pause();
        video.removeAttribute('src');
        video.load();
        cover.classList.remove('hidden');
    }

    function activate(key) {
        currentDemo = key;
        thumbs.forEach(t => t.classList.toggle('selected', t.dataset.demo === key));
        showCoverFor(key);
    }

    function playDemo() {
        if (!currentDemo) return;
        const cfg = DEMO_CONFIGS[currentDemo];
        cover.classList.add('hidden');
        video.src = cfg.video;
        video.load();
        video.play().catch(() => { /* autoplay blocked — user already clicked, should be fine */ });
        viewer.loadPly(cfg);
    }

    // Thumbnail click: switch preview + reset viewer
    thumbs.forEach(t => {
        t.addEventListener('click', () => {
            const key = t.dataset.demo;
            if (key === currentDemo) return;
            activate(key);
            viewer.clear();
        });
    });

    // Cover click: play video + load point cloud
    cover.addEventListener('click', playDemo);

    // Default selection
    activate('hkust_intr');
}

// ========================================
// PLY point cloud viewer (THREE.js PLYLoader)
// ========================================
class PlyViewer {
    constructor(canvas, messageEl) {
        this.canvas = canvas;
        this.messageEl = messageEl;
        this.container = canvas.parentElement;
        this.pointCloud = null;
        this.abortController = null;
        this.cache = {};
        this.currentKey = null;

        this.init();
        this.animate();
        window.addEventListener('resize', () => this.onResize());
    }

    init() {
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;

        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0xffffff);

        this.camera = new THREE.PerspectiveCamera(55, width / height, 0.01, 2000);
        this.camera.position.set(2, 1.5, 2);

        this.renderer = new THREE.WebGLRenderer({
            canvas: this.canvas,
            antialias: false,
            powerPreference: 'high-performance'
        });
        this.renderer.setSize(width, height);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));

        this.controls = new THREE.OrbitControls(this.camera, this.canvas);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.08;
        this.controls.minDistance = 0.05;
        this.controls.maxDistance = 1000;
    }

    setMessage(text) {
        if (!this.messageEl) return;
        this.messageEl.textContent = text;
        this.messageEl.style.display = text ? 'flex' : 'none';
        this.messageEl.style.opacity = text ? '1' : '0';
    }

    clear() {
        if (this.pointCloud) {
            this.scene.remove(this.pointCloud);
            this.pointCloud.geometry.dispose();
            this.pointCloud.material.dispose();
            this.pointCloud = null;
        }
        if (this.abortController) {
            this.abortController.abort();
            this.abortController = null;
        }
        this.setMessage('Click the play button to start');
    }

    async loadPly(cfg) {
        const key = cfg.ply;
        if (this.currentKey === key && this.pointCloud) return;
        this.currentKey = key;

        // Cancel any pending request
        if (this.abortController) this.abortController.abort();
        this.abortController = new AbortController();

        this.setMessage('Loading 0%');

        try {
            let buffer = this.cache[key];
            if (!buffer) {
                const response = await fetch(key, { signal: this.abortController.signal });
                if (!response.ok) throw new Error(`Failed to load: ${response.status}`);
                const total = parseInt(response.headers.get('content-length') || '0', 10);
                const reader = response.body.getReader();
                const chunks = [];
                let received = 0;
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    chunks.push(value);
                    received += value.length;
                    if (total > 0) {
                        const percent = Math.min(100, Math.round((received / total) * 100));
                        this.setMessage(`Loading ${percent}%`);
                    }
                }
                buffer = concatChunks(chunks).buffer;
                this.cache[key] = buffer;
            }

            if (this.currentKey !== key) return; // user switched while loading

            const geometry = parsePly(buffer);
            this.buildPointCloud(geometry, cfg);
            this.setMessage('');
        } catch (err) {
            if (err.name === 'AbortError') return;
            console.error('Error loading PLY:', err);
            this.setMessage('Failed to load point cloud');
        }
    }

    buildPointCloud(geometry, cfg) {
        if (this.pointCloud) {
            this.scene.remove(this.pointCloud);
            this.pointCloud.geometry.dispose();
            this.pointCloud.material.dispose();
        }

        if (cfg.flipY) {
            const pos = geometry.getAttribute('position');
            for (let i = 0; i < pos.count; i++) pos.setY(i, -pos.getY(i));
            pos.needsUpdate = true;
        }

        geometry.computeBoundingBox();
        const center = new THREE.Vector3();
        geometry.boundingBox.getCenter(center);
        geometry.translate(-center.x, -center.y, -center.z);
        geometry.computeBoundingSphere();
        const radius = geometry.boundingSphere.radius || 1;

        const material = new THREE.PointsMaterial({
            size: cfg.pointSize,
            vertexColors: geometry.hasAttribute('color'),
            color: geometry.hasAttribute('color') ? 0xffffff : 0x4477aa,
            sizeAttenuation: true
        });

        this.pointCloud = new THREE.Points(geometry, material);
        this.scene.add(this.pointCloud);

        const cam = cfg.camera || { x: -0.5, y: 0.3, z: -1.5 };
        this.camera.position.set(radius * cam.x, radius * cam.y, radius * cam.z);
        this.controls.target.set(0, 0, 0);
        this.controls.update();
    }

    onResize() {
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }
}

// ========================================
// Binary little-endian PLY parser
// Handles: x,y,z floats + red,green,blue[,alpha] uchars
// (matches trimesh export used by UniT)
// ========================================
function parsePly(buffer) {
    const bytes = new Uint8Array(buffer);
    const headerEndMarker = 'end_header\n';
    let headerEnd = -1;
    for (let i = 0; i < bytes.length - headerEndMarker.length; i++) {
        let match = true;
        for (let j = 0; j < headerEndMarker.length; j++) {
            if (bytes[i + j] !== headerEndMarker.charCodeAt(j)) { match = false; break; }
        }
        if (match) { headerEnd = i + headerEndMarker.length; break; }
    }
    if (headerEnd === -1) throw new Error('PLY header not terminated');

    const header = new TextDecoder('ascii').decode(bytes.subarray(0, headerEnd));
    const lines = header.split('\n');
    let vertexCount = 0;
    const props = [];
    for (const line of lines) {
        const parts = line.trim().split(/\s+/);
        if (parts[0] === 'element' && parts[1] === 'vertex') {
            vertexCount = parseInt(parts[2], 10);
        } else if (parts[0] === 'property' && parts.length >= 3) {
            props.push({ type: parts[1], name: parts[2] });
        }
    }

    const typeSizes = { char: 1, uchar: 1, short: 2, ushort: 2, int: 4, uint: 4, float: 4, double: 8 };
    let stride = 0;
    for (const p of props) stride += typeSizes[p.type] || 0;

    const view = new DataView(buffer, headerEnd);
    const positions = new Float32Array(vertexCount * 3);
    const hasColor = props.some(p => p.name === 'red' || p.name === 'r');
    const colors = hasColor ? new Float32Array(vertexCount * 3) : null;

    for (let i = 0; i < vertexCount; i++) {
        let off = i * stride;
        let px = 0, py = 0, pz = 0, r = 1, g = 1, b = 1;
        for (const p of props) {
            let val = 0;
            switch (p.type) {
                case 'float': val = view.getFloat32(off, true); off += 4; break;
                case 'double': val = view.getFloat64(off, true); off += 8; break;
                case 'uchar': case 'char': val = view.getUint8(off); off += 1; break;
                case 'ushort': val = view.getUint16(off, true); off += 2; break;
                case 'short': val = view.getInt16(off, true); off += 2; break;
                case 'uint': val = view.getUint32(off, true); off += 4; break;
                case 'int': val = view.getInt32(off, true); off += 4; break;
                default: throw new Error(`Unsupported PLY type: ${p.type}`);
            }
            if (p.name === 'x') px = val;
            else if (p.name === 'y') py = val;
            else if (p.name === 'z') pz = val;
            else if (p.name === 'red' || p.name === 'r') r = val / 255;
            else if (p.name === 'green' || p.name === 'g') g = val / 255;
            else if (p.name === 'blue' || p.name === 'b') b = val / 255;
        }
        positions[i * 3] = px;
        positions[i * 3 + 1] = py;
        positions[i * 3 + 2] = pz;
        if (colors) {
            colors[i * 3] = r;
            colors[i * 3 + 1] = g;
            colors[i * 3 + 2] = b;
        }
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    if (colors) geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    return geometry;
}

function concatChunks(chunks) {
    const total = chunks.reduce((s, a) => s + a.length, 0);
    const out = new Uint8Array(total);
    let off = 0;
    for (const c of chunks) { out.set(c, off); off += c.length; }
    return out;
}
