/* ========================================
   UniT project page — interactive examples
   Chapter nav + PNT v3 point-cloud viewer + video+cover sync

   Loading strategy (progressive + parallel + cached)
   -----------------------------------------------------
   Each scene is served as a gzipped .pnt.gz v3 file whose payload has
   two sections:

     (1) A small coarse prefix (<=80 k Morton-sorted points).  As soon
         as the prefix bytes arrive we dequantise them, build a full-
         sized BufferGeometry, and render the coarse cloud.  On a
         typical connection this takes well under a second — users
         never stare at a blank canvas.

     (2) The fine remainder.  Those bytes continue streaming through
         the same `DecompressionStream` while the coarse cloud is
         already interactive.  When the last byte lands, we fill the
         remaining slots of the same geometry and update `drawRange`
         to expose the full cloud.

   Other wins layered on top:
     - `Cache API` under the name "unit-pnt-v3": first visit pays the
       network cost, every subsequent visit is an instant memory read.
     - After the current scene's coarse render, we kick off *parallel*
       prefetches of the other five scenes so subsequent clicks are
       usually cache hits.  HTTP/2 multiplexing keeps this cheap.
     - `<link rel="preload">` in protected.html starts the default
       scene's fetch before any JS has executed.
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
//
// `density` is purely informational — the actual point count is
// determined by the header inside the .pnt.gz file.  `pointSize` is
// tuned visually per scene so scale & density read consistently
// across very different extents.  `camera` is the unit offset from
// the bounding-sphere centre, scaled by the sphere radius on load.
// ========================================
const DEMO_CONFIGS = {
    hkust_intr: {
        title: 'HKUST INTR',
        cloud: 'assets/demos/hkust_intr/scene.pnt.gz',
        density: 3350000,
        pointSize: 0.014,
        flipY: true,
        camera: { x: -0.6, y: 0.3, z: -1.4 }
    },
    hkust_toy: {
        title: 'HKUST Toy',
        cloud: 'assets/demos/hkust_toy/scene.pnt.gz',
        density: 2080000,
        pointSize: 0.0047,
        flipY: true,
        camera: { x: -0.4, y: 0.2, z: -1.5 }
    },
    hkust_redbird: {
        title: 'HKUST Red Bird',
        cloud: 'assets/demos/hkust_redbird/scene.pnt.gz',
        density: 2150000,
        pointSize: 0.047,
        flipY: true,
        camera: { x: -0.5, y: 0.35, z: -1.6 }
    },
    drift: {
        title: 'Drift',
        cloud: 'assets/demos/drift/scene.pnt.gz',
        density: 490000,
        pointSize: 0.155,
        flipY: true,
        camera: { x: -0.25, y: 0.25, z: -0.75 }
    },
    gta_sfm: {
        title: 'GTA SfM',
        cloud: 'assets/demos/gta_sfm/scene.pnt.gz',
        density: 3200000,
        pointSize: 0.094,
        flipY: true,
        camera: { x: -0.4, y: 0.2, z: -1.4 }
    },
    kitti: {
        title: 'KITTI',
        cloud: 'assets/demos/kitti/scene.pnt.gz',
        density: 2570000,
        pointSize: 0.31,
        flipY: true,
        camera: { x: -0.15, y: 0.22, z: -0.5 }
    }
};

const CACHE_NAME = 'unit-pnt-v3';

// ========================================
// Interactive Examples
// ========================================
function initDemo() {
    const canvas = document.getElementById('demo-canvas');
    if (!canvas) return;

    const loader = new PntLoader(CACHE_NAME);
    const viewer = new PointCloudViewer(canvas, document.getElementById('demo-message'), loader);
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
        viewer.show(key, cfg).then(() => schedulePrefetch(key));
    }

    // Fire up to 5 parallel prefetches for the non-selected scenes after
    // the current one's coarse render has landed.  HTTP/2 handles the
    // multiplexing; skipping the old 500 ms serial gate means neighbours
    // are usually warm by the time the user clicks them.
    function schedulePrefetch(priorityKey) {
        const conn = navigator.connection;
        if (conn && (conn.saveData || ['slow-2g', '2g'].includes(conn.effectiveType))) {
            return; // respect data-saver mode
        }
        // Wait a tick so first-demo fine-stream bytes get the pipe first,
        // then fan out in parallel.
        setTimeout(() => {
            if (currentDemo !== priorityKey) return;
            Object.entries(DEMO_CONFIGS).forEach(([k, cfg]) => {
                if (k === priorityKey) return;
                loader.prefetch(cfg.cloud).catch(() => {});
            });
        }, 600);
    }

    thumbs.forEach(t => {
        t.addEventListener('click', () => select(t.dataset.demo));
    });

    select('hkust_intr');
}


// ========================================
// PntLoader
//
// Responsibilities:
//   1. Fetch  .pnt.gz  (Cache API first, network fallback).
//   2. Pipe the response body through DecompressionStream('gzip').
//   3. Surface three milestones to the caller:
//        - onHeader(header)                 (40 bytes in)
//        - onCoarse(header, coarseBytes)    (header + coarse section in)
//        - resolve with {header, bytes}     (whole payload in)
//      Each milestone is paired with a typed-array view into a single
//      grow-doubling Uint8Array, so there is no per-chunk copy.
// ========================================
class PntLoader {
    constructor(cacheName) {
        this.cacheName = cacheName;
        this.inflight = new Map();  // url -> Promise<Uint8Array>
    }

    async _openCache() {
        if (!('caches' in self)) return null;
        try { return await caches.open(this.cacheName); }
        catch { return null; }
    }

    // Fetch compressed bytes (cache-then-network).  The `<link rel="prefetch">`
    // tags in protected.html populate the browser's HTTP disk cache before
    // we ever reach this function, so first-click fetches land straight from
    // local disk.  Options are left at defaults — HTTP-cache hits work for
    // any mode/credentials, unlike the preload cache which is finicky about
    // matching.
    async _fetchCompressed(url, signal) {
        const cache = await this._openCache();
        if (cache) {
            const hit = await cache.match(url);
            if (hit) return hit;
        }
        const resp = await fetch(url, { signal });
        if (!resp.ok) throw new Error(`HTTP ${resp.status} for ${url}`);
        // Clone BEFORE the body is consumed; the clone goes into the
        // cache, the original is returned for decompression.
        if (cache) cache.put(url, resp.clone()).catch(() => {});
        return resp;
    }

    // Background prefetch: store the compressed bytes in Cache API, skip
    // decompression.  Subsequent .load() calls hit the cache.
    async prefetch(url) {
        if (this.inflight.has(url)) return this.inflight.get(url);
        const cache = await this._openCache();
        if (cache) {
            const hit = await cache.match(url);
            if (hit) return;
        }
        const p = (async () => {
            const resp = await fetch(url, { priority: 'low' });
            if (!resp.ok) return;
            if (cache) await cache.put(url, resp.clone()).catch(() => {});
            // Consume the body so the connection is released even if we
            // never `load()` this URL.
            await resp.arrayBuffer().catch(() => {});
        })();
        this.inflight.set(url, p);
        p.finally(() => this.inflight.delete(url));
        return p;
    }

    // Streaming load with milestone callbacks.
    async load(url, { signal, onHeader, onCoarse, onProgress } = {}) {
        const resp = await this._fetchCompressed(url, signal);

        // Progress bar uses compressed-byte count when we can see it;
        // otherwise (Cache API responses don't always expose content-
        // length after decompression) we fall back to a spinner-ish
        // message upstream.
        const totalCompressed = parseInt(resp.headers.get('content-length') || '0', 10);

        // DecompressionStream yields the *uncompressed* bytes.  Chain
        // tees for progress measurement on the compressed side so we
        // don't double-count.
        let compressedReceived = 0;
        const progressStream = new TransformStream({
            transform(chunk, controller) {
                compressedReceived += chunk.byteLength;
                if (onProgress && totalCompressed > 0) {
                    onProgress(Math.min(99, Math.round(compressedReceived / totalCompressed * 100)));
                }
                controller.enqueue(chunk);
            }
        });

        const stream = resp.body
            .pipeThrough(progressStream)
            .pipeThrough(new DecompressionStream('gzip'));
        const reader = stream.getReader();

        let buf = new Uint8Array(1 << 16);   // grow-doubling scratch
        let len = 0;
        let header = null;
        let coarseFired = false;

        const ensureCap = (needed) => {
            if (buf.length >= needed) return;
            let cap = buf.length;
            while (cap < needed) cap *= 2;
            const next = new Uint8Array(cap);
            next.set(buf.subarray(0, len));
            buf = next;
        };

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            ensureCap(len + value.byteLength);
            buf.set(value, len);
            len += value.byteLength;

            // Header parse: 40 bytes.
            if (!header && len >= 40) {
                header = parsePntV3Header(buf.buffer, 0);
                if (onHeader) onHeader(header);
            }
            // Coarse fire: header + coarse section (9 bytes/point).
            if (header && !coarseFired) {
                const need = 40 + header.coarseCount * 9;
                if (len >= need) {
                    coarseFired = true;
                    if (onCoarse) onCoarse(header, buf, len);
                }
            }
        }

        // Shrink to exact length so downstream typed-array views don't
        // read past the real data.
        const tight = buf.subarray(0, len);
        if (onProgress) onProgress(100);
        return { header: header ?? parsePntV3Header(tight.buffer, 0), bytes: tight };
    }
}


// ========================================
// PNT v3 parser
// ========================================
function parsePntV3Header(buffer, offset = 0) {
    const view = new DataView(buffer, offset, 40);
    const magic =
        String.fromCharCode(view.getUint8(0)) +
        String.fromCharCode(view.getUint8(1)) +
        String.fromCharCode(view.getUint8(2)) +
        String.fromCharCode(view.getUint8(3));
    if (magic === 'UNPT') {
        // v1 / v2 fall-through: signal to caller to use the legacy path.
        return { magic, version: view.getUint32(4, true), legacy: true };
    }
    if (magic !== 'UNP3') throw new Error(`Not a UNP3 file (got "${magic}")`);
    const version = view.getUint32(4, true);
    const count = view.getUint32(8, true);
    const coarseCount = view.getUint32(12, true);
    const minX = view.getFloat32(16, true);
    const minY = view.getFloat32(20, true);
    const minZ = view.getFloat32(24, true);
    const sx = view.getFloat32(28, true);
    const sy = view.getFloat32(32, true);
    const sz = view.getFloat32(36, true);
    return {
        magic, version, count, coarseCount,
        min: [minX, minY, minZ],
        scale: [sx, sy, sz],
        legacy: false
    };
}

// Translate (+ optional flipY) a contiguous slice of `positions`
// in-place.  The centre is computed on pre-flip coordinates, so after
// flipping the Y axis we just negate each translated Y value rather
// than re-deriving the centre.  (Algebraic equivalence: applying
// `y' = -(y - cy)` yields the same layout as first flipping then
// centring on the flipped Y values.)
//
// Used twice per scene: once on the coarse slots as soon as they
// decode, then on the fine slots when the full stream lands.  Both
// calls share the SAME `xform` so the two sections end up in the same
// coordinate frame (origin-centred, optional flipY).
function applyTransform(positions, start, count, xform) {
    const { cx, cy, cz, flipY } = xform;
    const end = start + count;
    for (let i = start; i < end; i++) {
        const k = i * 3;
        positions[k]     -= cx;
        positions[k + 1] -= cy;
        positions[k + 2] -= cz;
    }
    if (flipY) {
        for (let i = start; i < end; i++) {
            positions[i * 3 + 1] = -positions[i * 3 + 1];
        }
    }
}

// Compute bounding box on `positions[start..start+count)`, derive
// centre + bounding-sphere radius, then call `applyTransform` so the
// slice is centred at the origin and (if `cfg.flipY`) Y-flipped.
// Returns the `xform` so the fine pass can reuse it.
function computeBboxAndTransform(positions, start, count, cfg) {
    let xmin = Infinity, xmax = -Infinity;
    let ymin = Infinity, ymax = -Infinity;
    let zmin = Infinity, zmax = -Infinity;
    const end = start + count;
    for (let i = start; i < end; i++) {
        const k = i * 3;
        const x = positions[k], y = positions[k + 1], z = positions[k + 2];
        if (x < xmin) xmin = x; if (x > xmax) xmax = x;
        if (y < ymin) ymin = y; if (y > ymax) ymax = y;
        if (z < zmin) zmin = z; if (z > zmax) zmax = z;
    }
    const cx = (xmin + xmax) * 0.5;
    const cy = (ymin + ymax) * 0.5;
    const cz = (zmin + zmax) * 0.5;
    const hx = (xmax - xmin) * 0.5;
    const hy = (ymax - ymin) * 0.5;
    const hz = (zmax - zmin) * 0.5;
    const radius = Math.sqrt(hx * hx + hy * hy + hz * hz) || 1;
    const xform = { cx, cy, cz, flipY: !!cfg.flipY, radius };
    applyTransform(positions, start, count, xform);
    return xform;
}

// Decode one byte-plane SoA section (coarse or fine) into preallocated
// Float32 position + Uint8 color slots.  `bytes` is a Uint8Array that
// already contains the section; `byteOffset` is where the section
// starts inside it.  `writeOffset` is the destination slot index
// (in points) inside `positions`/`colors`.
//
// Returns the byte offset immediately after the section.
function decodeSection(bytes, byteOffset, n, positions, colors, writeOffset, header) {
    const base = bytes.byteOffset + byteOffset;
    const buf = bytes.buffer;
    const [mx, my, mz] = header.min;
    const [sx, sy, sz] = header.scale;

    const xLo = new Uint8Array(buf, base,             n);
    const xHi = new Uint8Array(buf, base + n,         n);
    const yLo = new Uint8Array(buf, base + n * 2,     n);
    const yHi = new Uint8Array(buf, base + n * 3,     n);
    const zLo = new Uint8Array(buf, base + n * 4,     n);
    const zHi = new Uint8Array(buf, base + n * 5,     n);
    const r   = new Uint8Array(buf, base + n * 6,     n);
    const g   = new Uint8Array(buf, base + n * 7,     n);
    const b   = new Uint8Array(buf, base + n * 8,     n);

    // Tight SoA dequant loop — ~8 ms per million points on M1.
    const posBase = writeOffset * 3;
    const colBase = writeOffset * 3;
    for (let i = 0; i < n; i++) {
        const qx = xLo[i] | (xHi[i] << 8);
        const qy = yLo[i] | (yHi[i] << 8);
        const qz = zLo[i] | (zHi[i] << 8);
        const k = i * 3;
        positions[posBase + k]     = mx + qx * sx;
        positions[posBase + k + 1] = my + qy * sy;
        positions[posBase + k + 2] = mz + qz * sz;
        colors[colBase + k]     = r[i];
        colors[colBase + k + 1] = g[i];
        colors[colBase + k + 2] = b[i];
    }
    return byteOffset + n * 9;
}


// ========================================
// Point cloud viewer
// ========================================
class PointCloudViewer {
    constructor(canvas, messageEl, loader) {
        this.canvas = canvas;
        this.messageEl = messageEl;
        this.container = canvas.parentElement;
        this.loader = loader;
        this.pointCloud = null;
        this.activeKey = null;
        this.activeAbort = null;

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

    clearCloud() {
        this.clearMeasurement();
        if (this.pointCloud) {
            this.scene.remove(this.pointCloud);
            this.pointCloud.geometry.dispose();
            this.pointCloud.material.dispose();
            this.pointCloud = null;
        }
    }

    // --- Progressive load + render ---------------------------------------
    //
    // `show` runs *two* render passes for a single click:
    //
    //    pass 1 (coarse):  builds the full-sized geometry, fills just
    //                      the coarse prefix slots, sets drawRange to
    //                      `coarseCount`, renders.  Happens as soon as
    //                      the prefix bytes arrive (~hundreds of ms).
    //
    //    pass 2 (fine):    fills the remaining slots *in place* and
    //                      bumps drawRange to the full count.  Happens
    //                      when the full gzip stream has drained.
    //
    // If the user switches scenes mid-load we abort via AbortController
    // and move on.  The point cloud built for the previous scene is
    // disposed inside `clearCloud()` so GPU memory never leaks.
    // ---------------------------------------------------------------------
    async show(key, cfg) {
        if (this.activeKey === key && this.pointCloud) return;
        if (this.activeAbort) this.activeAbort.abort();
        const abort = new AbortController();
        this.activeAbort = abort;
        this.activeKey = key;

        this.setMessage('Loading…');

        // Per-scene state captured in closure so a later `show` call
        // can't accidentally rewrite the earlier scene's buffers.
        let positions = null;
        let colors = null;
        let builtCoarse = false;
        let xform = null;           // {cx, cy, cz, flipY, radius}

        try {
            const onHeader = (header) => {
                if (abort.signal.aborted) return;
                if (header.legacy) {
                    throw new Error(`Server returned legacy ${header.magic} v${header.version}; expected UNP3. Did the .pnt.gz redirect to .pnt?`);
                }
                positions = new Float32Array(header.count * 3);
                colors    = new Uint8Array(header.count * 3);
            };

            // When coarse bytes land, decode them, compute the scene's
            // centre/flipY/radius from *just the coarse slots*, apply
            // those transforms in-place to the same slots, and install
            // the geometry.  The unused fine slots stay at (0, 0, 0)
            // but are outside `drawRange`, so they never render.  We
            // stash `xform` so the fine pass can apply the same
            // translation + flipY — otherwise the two sections end up
            // in different coordinate frames and you see ghost images.
            const onCoarse = (header, bytes) => {
                if (abort.signal.aborted) return;
                decodeSection(bytes, 40, header.coarseCount, positions, colors, 0, header);
                xform = computeBboxAndTransform(positions, 0, header.coarseCount, cfg);
                this._installGeometry(positions, colors, header.count,
                                       header.coarseCount, cfg, xform.radius);
                builtCoarse = true;
                this.setMessage('Refining…');
            };

            const onProgress = (pct) => {
                if (abort.signal.aborted) return;
                if (!builtCoarse) this.setMessage(`Loading ${pct}%`);
            };

            const { header, bytes } = await this.loader.load(cfg.cloud, {
                signal: abort.signal, onHeader, onCoarse, onProgress
            });
            if (abort.signal.aborted || this.activeKey !== key) return;

            // Pass 2: fine section, right after the coarse block in memory.
            const fineOffset = 40 + header.coarseCount * 9;
            const fineCount = header.count - header.coarseCount;
            if (fineCount > 0) {
                decodeSection(bytes, fineOffset, fineCount, positions, colors,
                              header.coarseCount, header);
                if (xform) {
                    // Fine slots are still in raw file coordinates; push them
                    // through the coarse pass's transform so both sections
                    // share one coordinate frame.  (Skipping this is what
                    // caused the "ghost cloud" overlay bug.)
                    applyTransform(positions, header.coarseCount, fineCount, xform);
                }
            }
            if (!builtCoarse) {
                // Coarse milestone never fired — e.g. a tiny file where
                // the whole payload arrived before any intermediate read
                // boundary.  Transform everything at once and render.
                const x = computeBboxAndTransform(positions, 0, header.count, cfg);
                this._installGeometry(positions, colors, header.count,
                                       header.count, cfg, x.radius);
            } else {
                this._extendDrawRange(header.count);
            }
            this.setMessage('');
        } catch (err) {
            if (err.name === 'AbortError' || abort.signal.aborted) return;
            console.error('Error loading point cloud:', err);
            this.setMessage('Failed to load point cloud');
        }
    }

    // Replace the current cloud with a freshly-built one.  Positions
    // are assumed to be *already* centred + flipped by
    // `computeBboxAndTransform` — we never call `geometry.translate()`
    // or `computeBoundingBox()` here because those touch the unused
    // zero slots and would corrupt fine points written later.
    _installGeometry(positions, colors, totalCount, drawCount, cfg, radius) {
        this.clearCloud();

        const geometry = new THREE.BufferGeometry();
        const posAttr = new THREE.BufferAttribute(positions, 3);
        posAttr.setUsage(THREE.DynamicDrawUsage);
        geometry.setAttribute('position', posAttr);
        const colAttr = new THREE.BufferAttribute(colors, 3, true);
        colAttr.setUsage(THREE.DynamicDrawUsage);
        geometry.setAttribute('color', colAttr);

        geometry.setDrawRange(0, drawCount);
        // Bounding sphere set explicitly so three.js doesn't recompute
        // it from all positions (which include the zero-padded fine
        // slots); frustum culling + raycasting both use this.
        geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0, 0), radius);

        this._sceneRadius = radius;
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

    // Called after the fine section has been written *and transformed*
    // into the existing geometry's typed arrays.  We bump the draw
    // range and tell three.js to re-upload the position + color
    // attributes.
    _extendDrawRange(total) {
        if (!this.pointCloud) return;
        const geo = this.pointCloud.geometry;
        geo.setDrawRange(0, total);
        geo.getAttribute('position').needsUpdate = true;
        geo.getAttribute('color').needsUpdate = true;
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
