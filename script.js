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
// Per-demo visual tuning.
// density: how many points the .pnt was sampled to (just for reference — the
//   file itself determines the actual count).  Paired with `pointSize` so
//   rendered density looks balanced across very different scene scales.
// pointSize: world-unit size of each splat (sizeAttenuation is on, so
//   pixel size shrinks with distance).  Tuned visually per scene.
// camera: unit-offset from the cloud's bounding-sphere center, scaled by
//   the sphere radius at load time.
const DEMO_CONFIGS = {
    hkust_intr: {
        title: 'HKUST INTR',
        cloud: 'assets/demos/hkust_intr/scene.pnt',
        density: 550000,
        pointSize: 0.035,
        flipY: true,
        camera: { x: -0.6, y: 0.3, z: -1.4 }
    },
    hkust_toy: {
        title: 'HKUST Toy',
        cloud: 'assets/demos/hkust_toy/scene.pnt',
        density: 280000,
        pointSize: 0.011,
        flipY: true,
        camera: { x: -0.4, y: 0.2, z: -1.5 }
    },
    hkust_redbird: {
        title: 'HKUST Red Bird',
        cloud: 'assets/demos/hkust_redbird/scene.pnt',
        density: 400000,
        pointSize: 0.11,
        flipY: true,
        camera: { x: -0.5, y: 0.35, z: -1.6 }
    },
    drift: {
        title: 'Drift',
        cloud: 'assets/demos/drift/scene.pnt',
        density: 320000,
        pointSize: 0.28,
        flipY: true,
        camera: { x: -0.25, y: 0.25, z: -0.75 }
    },
    gta_sfm: {
        title: 'GTA SfM',
        cloud: 'assets/demos/gta_sfm/scene.pnt',
        density: 520000,
        pointSize: 0.2,
        flipY: true,
        camera: { x: -0.4, y: 0.2, z: -1.4 }
    },
    kitti: {
        title: 'KITTI',
        cloud: 'assets/demos/kitti/scene.pnt',
        density: 450000,
        pointSize: 0.65,
        flipY: true,
        camera: { x: -0.15, y: 0.22, z: -0.5 }
    }
};

// ========================================
// Interactive Examples
//
// Top panel: single 3D point cloud for the currently selected demo.
// Bottom row: auto-looping reconstruction videos as "live thumbnails";
// clicking one swaps the 3D cloud up top.
// ========================================
function initDemo() {
    const canvas = document.getElementById('demo-canvas');
    if (!canvas) return;

    const viewer = new PointCloudViewer(canvas, document.getElementById('demo-message'));
    const thumbs = document.querySelectorAll('.demo-thumb');

    let currentDemo = null;
    function select(key) {
        const cfg = DEMO_CONFIGS[key];
        if (!cfg || key === currentDemo) return;
        currentDemo = key;
        thumbs.forEach(t => {
            const isSelected = t.dataset.demo === key;
            t.classList.toggle('selected', isSelected);
            const v = t.querySelector('video');
            if (!v) return;
            if (isSelected) {
                v.currentTime = 0;
                v.play().catch(() => {});
            } else {
                v.pause();
                v.currentTime = 0;
            }
        });
        viewer.show(cfg);
    }

    thumbs.forEach(t => {
        t.addEventListener('click', () => select(t.dataset.demo));
    });

    select('hkust_intr');
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

        // Measurement state
        this.measureMode = false;
        this.measurePoints = [];
        this.measureMarkers = [];
        this.measureLine = null;
        this.measureLabel = null;
        this.raycaster = new THREE.Raycaster();
        this.raycaster.params.Points.threshold = 0.1;  // reset per scene at load
        this.mouse = new THREE.Vector2();

        this.init();
        this.initMeasureUI();
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

    initMeasureUI() {
        this.measureBtn = document.getElementById('measure-btn');
        this.measureHintEl = document.getElementById('measure-hint');
        if (this.measureBtn) {
            this.measureBtn.addEventListener('click', () => this.toggleMeasureMode());
        }
        // Use pointerdown so we don't conflict with OrbitControls drag —
        // only treat it as a measurement click if no drag happens.
        let downAt = null;
        this.canvas.addEventListener('pointerdown', e => {
            downAt = { x: e.clientX, y: e.clientY };
        });
        this.canvas.addEventListener('pointerup', e => {
            if (!downAt) return;
            const dx = e.clientX - downAt.x;
            const dy = e.clientY - downAt.y;
            downAt = null;
            if (Math.hypot(dx, dy) > 5) return; // was a drag, ignore
            this.onCanvasClick(e);
        });
    }

    toggleMeasureMode() {
        this.measureMode = !this.measureMode;
        if (this.measureBtn) {
            this.measureBtn.classList.toggle('active', this.measureMode);
            this.measureBtn.title = this.measureMode
                ? 'Click to disable distance measurement'
                : 'Click two points to measure distance';
        }
        if (this.measureHintEl) {
            this.measureHintEl.classList.toggle('visible', this.measureMode);
            this.measureHintEl.textContent = 'Click two points to measure distance';
        }
        if (!this.measureMode) this.clearMeasurement();
        this.canvas.style.cursor = this.measureMode ? 'crosshair' : '';
    }

    onCanvasClick(event) {
        if (!this.measureMode || !this.pointCloud) return;
        const rect = this.canvas.getBoundingClientRect();
        this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
        this.raycaster.setFromCamera(this.mouse, this.camera);
        const hits = this.raycaster.intersectObject(this.pointCloud);
        if (hits.length > 0) this.addMeasurePoint(hits[0].point.clone());
    }

    addMeasurePoint(point) {
        if (this.measurePoints.length >= 2) this.clearMeasurement();
        this.measurePoints.push(point);

        const markerSize = Math.max(0.005, (this._sceneRadius || 1) * 0.012);
        const geom = new THREE.SphereGeometry(markerSize, 16, 16);
        const mat = new THREE.MeshBasicMaterial({
            color: this.measurePoints.length === 1 ? 0x00c853 : 0xff3d00,
            depthTest: false
        });
        const marker = new THREE.Mesh(geom, mat);
        marker.renderOrder = 999;
        marker.position.copy(point);
        this.scene.add(marker);
        this.measureMarkers.push(marker);

        if (this.measurePoints.length === 2) {
            this.drawMeasureLine();
            this.showDistance();
        }
    }

    drawMeasureLine() {
        if (this.measureLine) {
            this.scene.remove(this.measureLine);
            this.measureLine.geometry.dispose();
            this.measureLine.material.dispose();
        }
        const geometry = new THREE.BufferGeometry().setFromPoints(this.measurePoints);
        const material = new THREE.LineBasicMaterial({
            color: 0xffb300, linewidth: 2, depthTest: false
        });
        this.measureLine = new THREE.Line(geometry, material);
        this.measureLine.renderOrder = 999;
        this.scene.add(this.measureLine);
    }

    showDistance() {
        const d = this.measurePoints[0].distanceTo(this.measurePoints[1]);
        if (this.measureHintEl) {
            this.measureHintEl.textContent = `Distance: ${d.toFixed(2)} (scene units)`;
        }

        if (this.measureLabel) {
            this.scene.remove(this.measureLabel);
            this.measureLabel.material.map?.dispose();
            this.measureLabel.material.dispose();
        }
        const canvas = document.createElement('canvas');
        canvas.width = 256; canvas.height = 64;
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#fff';
        ctx.font = 'bold 28px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(d.toFixed(2), canvas.width / 2, canvas.height / 2);

        const tex = new THREE.CanvasTexture(canvas);
        const mat = new THREE.SpriteMaterial({ map: tex, depthTest: false });
        this.measureLabel = new THREE.Sprite(mat);
        const mid = new THREE.Vector3().addVectors(this.measurePoints[0], this.measurePoints[1]).multiplyScalar(0.5);
        mid.y += (this._sceneRadius || 1) * 0.05;
        this.measureLabel.position.copy(mid);
        const labelScale = (this._sceneRadius || 1) * 0.22;
        this.measureLabel.scale.set(labelScale, labelScale * 0.25, 1);
        this.measureLabel.renderOrder = 1000;
        this.scene.add(this.measureLabel);
    }

    clearMeasurement() {
        this.measureMarkers.forEach(m => {
            this.scene.remove(m);
            m.geometry.dispose();
            m.material.dispose();
        });
        this.measureMarkers = [];
        if (this.measureLine) {
            this.scene.remove(this.measureLine);
            this.measureLine.geometry.dispose();
            this.measureLine.material.dispose();
            this.measureLine = null;
        }
        if (this.measureLabel) {
            this.scene.remove(this.measureLabel);
            this.measureLabel.material.map?.dispose();
            this.measureLabel.material.dispose();
            this.measureLabel = null;
        }
        this.measurePoints = [];
        if (this.measureHintEl && this.measureMode) {
            this.measureHintEl.textContent = 'Click two points to measure distance';
        }
    }

    setMessage(text) {
        if (!this.messageEl) return;
        this.messageEl.textContent = text;
        this.messageEl.style.display = text ? 'flex' : 'none';
        this.messageEl.style.opacity = text ? '1' : '0';
    }

    clear() {
        this.clearMeasurement();
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
        this.clearMeasurement();
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
        this._sceneRadius = radius;

        // Raycast pick tolerance scales with point size, so measurement
        // clicks on sparse outdoor scenes (KITTI, Drift) are as forgiving
        // as clicks on dense indoor ones (Toy, INTR).
        this.raycaster.params.Points.threshold = cfg.pointSize * 1.5;

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
