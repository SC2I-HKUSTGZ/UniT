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
        cloud: 'assets/demos/hkust_intr/scene.pnt',
        video: 'assets/demos/hkust_intr/reconstructed.mp4',
        pointSize: 0.05,
        flipY: true,
        camera: { x: -0.6, y: 0.3, z: -1.4 }
    },
    hkust_toy: {
        title: 'HKUST Toy',
        cloud: 'assets/demos/hkust_toy/scene.pnt',
        video: 'assets/demos/hkust_toy/reconstructed.mp4',
        pointSize: 0.012,
        flipY: true,
        camera: { x: -0.4, y: 0.2, z: -1.5 }
    },
    hkust_redbird: {
        title: 'HKUST Red Bird',
        cloud: 'assets/demos/hkust_redbird/scene.pnt',
        video: 'assets/demos/hkust_redbird/reconstructed.mp4',
        pointSize: 0.14,
        flipY: true,
        camera: { x: -0.5, y: 0.35, z: -1.6 }
    },
    drift: {
        title: 'Drift',
        cloud: 'assets/demos/drift/scene.pnt',
        video: 'assets/demos/drift/reconstructed.mp4',
        pointSize: 0.3,
        flipY: true,
        camera: { x: -0.5, y: 0.25, z: -1.2 }
    },
    gta_sfm: {
        title: 'GTA SfM',
        cloud: 'assets/demos/gta_sfm/scene.pnt',
        video: 'assets/demos/gta_sfm/reconstructed.mp4',
        pointSize: 0.25,
        flipY: true,
        camera: { x: -0.4, y: 0.2, z: -1.4 }
    },
    kitti: {
        title: 'KITTI',
        cloud: 'assets/demos/kitti/scene.pnt',
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

    const viewer = new PointCloudViewer(canvas, document.getElementById('demo-message'));
    const video = document.getElementById('demo-video');
    const cover = document.getElementById('demo-cover');
    const coverImg = document.getElementById('demo-cover-img');
    const thumbs = document.querySelectorAll('.demo-thumb');

    let currentDemo = null;

    function activate(key) {
        const cfg = DEMO_CONFIGS[key];
        if (!cfg) return;
        currentDemo = key;
        thumbs.forEach(t => t.classList.toggle('selected', t.dataset.demo === key));
        coverImg.src = `assets/demos/${key}/cover.jpg`;
        video.pause();
        video.removeAttribute('src');
        video.load();
        cover.classList.remove('hidden');

        // Best practice: kick off network fetch as soon as the user hints at
        // interest (thumbnail click / page load), so by the time they press
        // play the bytes are already on-device or mid-flight.
        viewer.prefetch(cfg);
    }

    function playDemo() {
        if (!currentDemo) return;
        const cfg = DEMO_CONFIGS[currentDemo];
        cover.classList.add('hidden');
        video.src = cfg.video;
        video.load();
        video.play().catch(() => {}); // autoplay blocked — user already clicked
        viewer.show(cfg);
    }

    thumbs.forEach(t => {
        t.addEventListener('click', () => {
            const key = t.dataset.demo;
            if (key === currentDemo) return;
            activate(key);
            viewer.clear();
        });
    });

    cover.addEventListener('click', playDemo);

    activate('hkust_intr');
}

// ========================================
// Point cloud viewer
//
// Loads .pnt files with zero-copy typed-array views (basically free vs. the
// old per-vertex DataView loop).  prefetch() kicks off the fetch on
// thumbnail select; show() awaits it + builds the Points mesh.
// ========================================
class PointCloudViewer {
    constructor(canvas, messageEl) {
        this.canvas = canvas;
        this.messageEl = messageEl;
        this.container = canvas.parentElement;
        this.pointCloud = null;
        this.currentKey = null;
        this.pending = new Map(); // key → { promise, abort, progress }

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
        this.currentKey = null;
        this.setMessage('Click the play button to start');
    }

    prefetch(cfg) {
        const url = cfg.cloud;
        if (this.pending.has(url)) return this.pending.get(url).promise;

        const abort = new AbortController();
        const state = { abort, progress: 0, buffer: null };
        const promise = this._fetchBuffer(url, abort.signal, pct => {
            state.progress = pct;
            if (this.currentKey === url) this.setMessage(`Loading ${pct}%`);
        }).then(buf => {
            state.buffer = buf;
            return buf;
        }).catch(err => {
            if (err.name !== 'AbortError') {
                console.error('Prefetch failed:', err);
                this.pending.delete(url);
            }
            throw err;
        });
        state.promise = promise;
        this.pending.set(url, state);
        return promise;
    }

    async show(cfg) {
        const url = cfg.cloud;
        if (this.currentKey === url && this.pointCloud) return;
        this.currentKey = url;

        const state = this.pending.get(url) || { promise: this.prefetch(cfg) };
        this.setMessage(state.buffer ? '' : `Loading ${state.progress || 0}%`);

        try {
            const buffer = await state.promise;
            if (this.currentKey !== url) return; // user switched while loading

            const t0 = performance.now();
            const geometry = parsePnt(buffer);
            this.buildPointCloud(geometry, cfg);
            console.debug(`PNT parse+build: ${(performance.now() - t0).toFixed(1)}ms`);
            this.setMessage('');
        } catch (err) {
            if (err.name === 'AbortError') return;
            console.error('Error loading point cloud:', err);
            this.setMessage('Failed to load point cloud');
        }
    }

    async _fetchBuffer(url, signal, onProgress) {
        const response = await fetch(url, { signal });
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
            if (total > 0 && onProgress) {
                onProgress(Math.min(100, Math.round((received / total) * 100)));
            }
        }

        // Concatenate into a single ArrayBuffer, so the Float32/Uint8 views
        // below can alias the contents with zero copy.
        const out = new Uint8Array(received);
        let off = 0;
        for (const c of chunks) { out.set(c, off); off += c.length; }
        return out.buffer;
    }

    buildPointCloud(geometry, cfg) {
        if (this.pointCloud) {
            this.scene.remove(this.pointCloud);
            this.pointCloud.geometry.dispose();
            this.pointCloud.material.dispose();
        }

        if (cfg.flipY) {
            // In-place scale on a zero-copy Float32Array view
            const pos = geometry.getAttribute('position').array;
            for (let i = 1; i < pos.length; i += 3) pos[i] = -pos[i];
        }

        geometry.computeBoundingBox();
        const center = new THREE.Vector3();
        geometry.boundingBox.getCenter(center);
        geometry.translate(-center.x, -center.y, -center.z);
        geometry.computeBoundingSphere();
        const radius = geometry.boundingSphere.radius || 1;

        const material = new THREE.PointsMaterial({
            size: cfg.pointSize,
            vertexColors: true,
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
// .pnt parser — zero-copy typed-array aliases
//
// Format:
//   [0..4)    magic "UNPT"
//   [4..8)    version (uint32 LE, currently 1)
//   [8..12)   vertex count (uint32 LE)
//   [12 .. 12+n*12)            float32 xyz
//   [12+n*12 .. 12+n*15)       uint8   rgb
// ========================================
function parsePnt(buffer) {
    const header = new DataView(buffer, 0, 12);
    const magic =
        String.fromCharCode(header.getUint8(0)) +
        String.fromCharCode(header.getUint8(1)) +
        String.fromCharCode(header.getUint8(2)) +
        String.fromCharCode(header.getUint8(3));
    if (magic !== 'UNPT') throw new Error(`Not a UNPT file (got "${magic}")`);
    const version = header.getUint32(4, true);
    if (version !== 1) throw new Error(`Unsupported UNPT version ${version}`);
    const count = header.getUint32(8, true);

    // Zero-copy: these views alias directly into the downloaded buffer.
    const positions = new Float32Array(buffer, 12, count * 3);
    const colors = new Uint8Array(buffer, 12 + count * 12, count * 3);

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    // normalized=true tells WebGL to divide u8 by 255 in the shader,
    // so we ship 1 byte/channel instead of 4.
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3, true));
    return geometry;
}
