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
        viewer.show(cfg).then(() => scheduleBackgroundPrefetch(key));
    }

    // Background prefetch of the *other* demos, one at a time, kicked off
    // only after the currently-selected cloud has finished loading.  On
    // slow links this prevents six ~3 MB fetches from fighting over the
    // same pipe and making every single one feel slow.
    let prefetchTimer = null;
    function scheduleBackgroundPrefetch(priorityKey) {
        if (prefetchTimer) { clearTimeout(prefetchTimer); prefetchTimer = null; }

        // Skip on known-slow connections.
        const conn = navigator.connection;
        if (conn) {
            if (conn.saveData) return;
            if (['slow-2g', '2g'].includes(conn.effectiveType)) return;
        }

        const queue = Object.keys(DEMO_CONFIGS).filter(k => k !== priorityKey);

        const runNext = () => {
            if (queue.length === 0 || currentDemo !== priorityKey) return;
            const key = queue.shift();
            const cfg = DEMO_CONFIGS[key];
            viewer.prefetch(cfg)
                .catch(() => {}) // ignore cancels/failures — user-initiated picks win
                .finally(() => {
                    if (currentDemo !== priorityKey) return;
                    prefetchTimer = setTimeout(runNext, 500);
                });
        };
        // Small initial delay so the selected demo finishes parsing /
        // first paint before we start pulling neighbours.
        prefetchTimer = setTimeout(runNext, 1500);
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

        // Both markers use the same compact two-layer style: a small
        // white core ringed by a blue halo.  Less visually noisy than
        // green/red "start/end" spheres; drawn with depthTest off so
        // they stay visible against the point cloud.
        const r = Math.max(0.004, (this._sceneRadius || 1) * 0.010);
        const core = new THREE.Mesh(
            new THREE.SphereGeometry(r * 0.55, 16, 16),
            new THREE.MeshBasicMaterial({ color: 0xffffff, depthTest: false })
        );
        const halo = new THREE.Mesh(
            new THREE.SphereGeometry(r, 16, 16),
            new THREE.MeshBasicMaterial({ color: 0x1a73e8, transparent: true, opacity: 0.55, depthTest: false })
        );
        core.position.copy(point);
        halo.position.copy(point);
        core.renderOrder = 1001;
        halo.renderOrder = 1000;
        this.scene.add(halo);
        this.scene.add(core);
        this.measureMarkers.push(halo, core);

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
            color: 0x1a73e8, linewidth: 2, depthTest: false,
            transparent: true, opacity: 0.85
        });
        this.measureLine = new THREE.Line(geometry, material);
        this.measureLine.renderOrder = 999;
        this.scene.add(this.measureLine);
    }

    showDistance() {
        // Scene units in UniT reconstructions are metric (meters).
        const d = this.measurePoints[0].distanceTo(this.measurePoints[1]);
        const formatted = d >= 10 ? d.toFixed(1) : d.toFixed(2);
        if (this.measureHintEl) {
            this.measureHintEl.textContent = `Distance: ${formatted} m`;
        }

        if (this.measureLabel) {
            this.scene.remove(this.measureLabel);
            this.measureLabel.material.map?.dispose();
            this.measureLabel.material.dispose();
        }

        // Rounded-pill label drawn onto a 2× canvas, used as a Sprite.
        const DPR = 2;
        const canvas = document.createElement('canvas');
        canvas.width = 320 * DPR;
        canvas.height = 96 * DPR;
        const ctx = canvas.getContext('2d');
        ctx.scale(DPR, DPR);
        const w = 320, h = 96;
        const pad = 14, radius = 20;
        const boxW = w - pad * 2, boxH = h - pad * 2;
        ctx.fillStyle = 'rgba(26, 115, 232, 0.95)';
        ctx.beginPath();
        ctx.roundRect(pad, pad, boxW, boxH, radius);
        ctx.fill();
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.55)';
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.fillStyle = '#ffffff';
        ctx.font = '600 42px "Google Sans", Arial, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(`${formatted} m`, w / 2, h / 2 + 2);

        const tex = new THREE.CanvasTexture(canvas);
        tex.minFilter = THREE.LinearFilter;
        const mat = new THREE.SpriteMaterial({ map: tex, depthTest: false });
        this.measureLabel = new THREE.Sprite(mat);
        const mid = new THREE.Vector3().addVectors(this.measurePoints[0], this.measurePoints[1]).multiplyScalar(0.5);
        mid.y += (this._sceneRadius || 1) * 0.06;
        this.measureLabel.position.copy(mid);
        // Label size scales with the scene so it reads at roughly the
        // same apparent size across indoor toys and KITTI streets.
        const s = (this._sceneRadius || 1) * 0.18;
        this.measureLabel.scale.set(s, s * (96 / 320), 1);
        this.measureLabel.renderOrder = 1002;
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
        if (this.pending.has(url)) {
            const existing = this.pending.get(url);
            if (existing.buffer || !existing.aborted) return existing.promise;
        }

        const abort = new AbortController();
        const state = { abort, progress: 0, buffer: null, aborted: false };
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
            } else {
                state.aborted = true;
            }
            throw err;
        });
        state.promise = promise;
        this.pending.set(url, state);
        return promise;
    }

    // Abort any in-flight prefetches for URLs other than `keepUrl`.
    // Called when the user picks a new demo so on slow connections the
    // newly-selected cloud gets the full pipe immediately instead of
    // competing with leftover background fetches.
    cancelOtherFetches(keepUrl) {
        for (const [url, state] of this.pending.entries()) {
            if (url === keepUrl) continue;
            if (state.buffer) continue;          // already done, keep the cached bytes
            state.abort.abort();
            this.pending.delete(url);
        }
    }

    async show(cfg) {
        const url = cfg.cloud;
        if (this.currentKey === url && this.pointCloud) return;
        this.currentKey = url;
        this.cancelOtherFetches(url);

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
// .pnt parser
//
// v1 (legacy, still supported for safety):
//   [0..4)    magic "UNPT"
//   [4..8)    version = 1
//   [8..12)   count (uint32 LE)
//   [12 .. 12+n*12)      float32 xyz
//   [12+n*12 .. 12+n*15) uint8   rgb
//
// v2 (current, ~40% smaller — quantized int16 positions):
//   [0..4)    magic "UNPT"
//   [4..8)    version = 2
//   [8..12)   count (uint32 LE)
//   [12..24)  min_xyz   (3 × float32)
//   [24..36)  scale_xyz (3 × float32)
//   [36 .. 36+n*6)       uint16 quantized xyz
//   [36+n*6 .. 36+n*9)   uint8  rgb
// ========================================
function parsePnt(buffer) {
    const view = new DataView(buffer);
    const magic =
        String.fromCharCode(view.getUint8(0)) +
        String.fromCharCode(view.getUint8(1)) +
        String.fromCharCode(view.getUint8(2)) +
        String.fromCharCode(view.getUint8(3));
    if (magic !== 'UNPT') throw new Error(`Not a UNPT file (got "${magic}")`);
    const version = view.getUint32(4, true);
    const count = view.getUint32(8, true);

    let positions, colors;
    if (version === 2) {
        const minX = view.getFloat32(12, true);
        const minY = view.getFloat32(16, true);
        const minZ = view.getFloat32(20, true);
        const sx = view.getFloat32(24, true);
        const sy = view.getFloat32(28, true);
        const sz = view.getFloat32(32, true);
        const q = new Uint16Array(buffer, 36, count * 3);
        positions = new Float32Array(count * 3);
        // Tight decode loop — ~10 ms for 500k verts on M1.
        for (let i = 0; i < count; i++) {
            const j = i * 3;
            positions[j]     = minX + q[j]     * sx;
            positions[j + 1] = minY + q[j + 1] * sy;
            positions[j + 2] = minZ + q[j + 2] * sz;
        }
        colors = new Uint8Array(buffer, 36 + count * 6, count * 3);
    } else if (version === 1) {
        // Zero-copy views directly into the downloaded buffer.
        positions = new Float32Array(buffer, 12, count * 3);
        colors = new Uint8Array(buffer, 12 + count * 12, count * 3);
    } else {
        throw new Error(`Unsupported UNPT version ${version}`);
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    // normalized=true → shader divides u8 by 255 so we ship 1 byte/channel.
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3, true));
    return geometry;
}
