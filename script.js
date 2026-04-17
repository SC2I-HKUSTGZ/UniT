/* ========================================
   UniT project page — interactive examples
   Chapter nav + PNT v4 point-cloud viewer + video+cover sync

   Loading strategy (true streaming + parallel + cached)
   -----------------------------------------------------
   Each scene is served as a gzipped .pnt.gz v4 file.  The payload is
   split into fixed-size blocks of 16 384 points; the entire cloud was
   randomly shuffled before being blocked, so **any file prefix is a
   uniform random subsample of the whole scene**.  That means:

     - The first block rendered already spans the full bounding box —
       the cloud looks like a sparse sketch of the real scene, not a
       spatial corner.
     - Each subsequent block adds more points, uniformly distributed.
       The cloud visibly densifies in real-time as bytes arrive, all
       the way up to full detail, with no discrete "now it refines"
       step.
     - If the user cancels or switches scenes mid-load, whatever
       arrived so far is still a usable visualisation.

   Implementation:
     - `DecompressionStream('gzip')` feeds a grow-doubling scratch
       buffer.  A parser walks it block-by-block, emitting `onBlock`
       callbacks the instant each block has fully arrived.
     - The viewer preallocates the final-size position / colour arrays
       on header, installs an empty geometry (drawRange=0), then
       appends each block in place.  `BufferAttribute.updateRange`
       batches GPU sub-uploads to just the newly-added slice per
       animation frame — total upload over a load is ~1×N points,
       not quadratic.
     - The scene's centre + bounding sphere come from the header's
       min/scale (the full-scene quantisation bbox), so the transform
       is exact from byte 44 onward — no re-centring as more points
       land.

   Other wins layered on top:
     - `Cache API` under the name "unit-pnt-v4": first visit pays the
       network cost, every subsequent visit is an instant memory read.
     - After the current scene's first block renders, we kick off
       *parallel* prefetches of the other five scenes so subsequent
       clicks are usually cache hits.  HTTP/2 multiplexing keeps this
       cheap.
     - `<link rel="prefetch">` tags in protected.html warm the HTTP
       cache before any thumbnail is clicked.
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
        density: 3810000,
        pointSize: 0.011,
        flipY: true,
        camera: { x: -0.6, y: 0.3, z: -1.4 }
    },
    hkust_toy: {
        title: 'HKUST Toy',
        cloud: 'assets/demos/hkust_toy/scene.pnt.gz',
        density: 2680000,
        pointSize: 0.0042,
        flipY: true,
        camera: { x: -0.4, y: 0.2, z: -1.5 }
    },
    hkust_redbird: {
        title: 'HKUST Red Bird',
        cloud: 'assets/demos/hkust_redbird/scene.pnt.gz',
        density: 2280000,
        pointSize: 0.044,
        flipY: true,
        camera: { x: -0.5, y: 0.35, z: -1.6 }
    },
    drift: {
        title: 'Drift',
        cloud: 'assets/demos/drift/scene.pnt.gz',
        density: 556000,
        pointSize: 0.145,
        flipY: true,
        camera: { x: -0.25, y: 0.25, z: -0.75 }
    },
    gta_sfm: {
        title: 'GTA SfM',
        cloud: 'assets/demos/gta_sfm/scene.pnt.gz',
        density: 4070000,
        pointSize: 0.082,
        flipY: true,
        camera: { x: -0.4, y: 0.2, z: -1.4 }
    },
    kitti: {
        title: 'KITTI',
        cloud: 'assets/demos/kitti/scene.pnt.gz',
        density: 2890000,
        pointSize: 0.30,
        flipY: true,
        camera: { x: -0.15, y: 0.22, z: -0.5 }
    }
};

const CACHE_NAME = 'unit-pnt-v4';

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

    // Fire parallel prefetches for the non-selected scenes shortly
    // after the current scene's load resolves.  HTTP/2 handles the
    // multiplexing; the short delay gives the current load's trailing
    // blocks the pipe first, then neighbours warm up in the background
    // so later clicks hit the HTTP cache.
    function schedulePrefetch(priorityKey) {
        const conn = navigator.connection;
        if (conn && (conn.saveData || ['slow-2g', '2g'].includes(conn.effectiveType))) {
            return; // respect data-saver mode
        }
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
//   3. Walk the decompressed byte stream block-by-block, firing:
//        - onHeader(header)                                    (44 bytes in)
//        - onBlock(header, bytes, byteOffset, blockCount, blockIdx)
//                                                              (every 9*bc bytes)
//      Each callback receives a typed-array view into a single grow-
//      doubling Uint8Array, so there is no per-chunk copy.
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

    // Streaming load with per-block callbacks.
    async load(url, { signal, onHeader, onBlock, onProgress } = {}) {
        const resp = await this._fetchCompressed(url, signal);

        // Progress bar uses compressed-byte count when we can see it;
        // otherwise (Cache API responses don't always expose content-
        // length after decompression) we fall back to a spinner-ish
        // message upstream.
        const totalCompressed = parseInt(resp.headers.get('content-length') || '0', 10);

        // DecompressionStream yields the *uncompressed* bytes.  Chain
        // a passthrough TransformStream on the compressed side so we
        // can count bytes for progress without double-reading.
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
        // Byte cursor: where the next unparsed block starts.  Starts
        // after the 44-byte header once `header` has been parsed.
        let cursor = 0;
        let blocksDecoded = 0;

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

            // Parse header once (44 bytes).
            if (!header && len >= 44) {
                header = parsePntV4Header(buf.buffer, 0);
                cursor = 44;
                if (onHeader) onHeader(header);
            }

            // Drain as many complete blocks as have arrived since last iter.
            while (header && blocksDecoded < header.numBlocks) {
                const bc = (blocksDecoded < header.numBlocks - 1)
                    ? header.blockSize
                    : (header.count - blocksDecoded * header.blockSize);
                const blockBytes = bc * 9;                 // SoA: 6 bytes pos + 3 bytes colour per point
                if (len - cursor < blockBytes) break;      // block still in-flight
                if (onBlock) onBlock(header, buf, cursor, bc, blocksDecoded);
                cursor += blockBytes;
                blocksDecoded++;
            }
        }

        const tight = buf.subarray(0, len);
        if (onProgress) onProgress(100);
        return { header: header ?? parsePntV4Header(tight.buffer, 0), bytes: tight };
    }
}


// ========================================
// PNT v4 parser + helpers
// ========================================
function parsePntV4Header(buffer, offset = 0) {
    const view = new DataView(buffer, offset, 44);
    const magic =
        String.fromCharCode(view.getUint8(0)) +
        String.fromCharCode(view.getUint8(1)) +
        String.fromCharCode(view.getUint8(2)) +
        String.fromCharCode(view.getUint8(3));
    if (magic !== 'UNP4') {
        throw new Error(`Not a UNP4 file (got "${magic}") — refresh to clear old cache?`);
    }
    const version    = view.getUint32(4,  true);
    const count      = view.getUint32(8,  true);
    const blockSize  = view.getUint32(12, true);
    const numBlocks  = view.getUint32(16, true);
    const minX = view.getFloat32(20, true);
    const minY = view.getFloat32(24, true);
    const minZ = view.getFloat32(28, true);
    const sx   = view.getFloat32(32, true);
    const sy   = view.getFloat32(36, true);
    const sz   = view.getFloat32(40, true);
    return {
        magic, version, count, blockSize, numBlocks,
        min:   [minX, minY, minZ],
        scale: [sx, sy, sz],
    };
}

// Derive the scene's centre-at-origin transform directly from the file
// header.  The header's (min, scale) pair describes the exact quantised
// bbox of the *full* cloud, so we know the centre and bounding-sphere
// radius the moment 44 bytes arrive — no need to sample any points,
// no "sampling error" as more blocks land.
function computeXformFromHeader(header, cfg) {
    const [mx, my, mz] = header.min;
    const [sx, sy, sz] = header.scale;
    const HALF = 32767.5;                // midpoint of the uint16 grid
    const cx = mx + HALF * sx;
    const cy = my + HALF * sy;
    const cz = mz + HALF * sz;
    const hx = HALF * sx;
    const hy = HALF * sy;
    const hz = HALF * sz;
    const radius = Math.sqrt(hx * hx + hy * hy + hz * hz) || 1;
    return { cx, cy, cz, flipY: !!cfg.flipY, radius };
}

// Translate (+ optional flipY) a contiguous slice of `positions`
// in-place.  Algebraic identity: `(-y - (-cy_raw))` == `-(y - cy_raw)`,
// so we can translate first (all axes) then flip Y, and still end up
// with the same layout as "flip first, then centre on the flipped Y".
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

// Decode one block (byte-plane SoA) into preallocated Float32 position
// + Uint8 colour slots.  `bytes` is the grow-doubling scratch Uint8Array
// the loader reads into; `byteOffset` is where the block starts in that
// buffer; `writeOffset` is the *point index* where the block's points
// should land in the geometry.
function decodeBlock(bytes, byteOffset, n, positions, colors, writeOffset, header) {
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

    // --- True streaming load + render ------------------------------------
    //
    // Every block the loader emits goes straight into the geometry:
    //
    //    1. `onHeader`  installs an empty full-sized geometry.  We can
    //       do this because `computeXformFromHeader` derives the scene's
    //       centre and bounding-sphere radius from the file's (min, scale)
    //       directly — no points needed.
    //
    //    2. `onBlock`   decodes into the next `blockCount` slots of the
    //       pre-allocated typed arrays, applies the same transform, and
    //       marks the new slice dirty.  An rAF-batched flush commits
    //       the dirty slice to the GPU with `BufferAttribute.updateRange`,
    //       so each block pays a ~200 KB sub-upload instead of forcing
    //       a whole-buffer re-upload (which would be O(N²) over N blocks).
    //
    // Net effect: the cloud visibly densifies in real time as bytes
    // arrive.  Mid-load cancellation via `AbortController` leaves the
    // partially-rendered scene displayed until the next `show` call.
    // ---------------------------------------------------------------------
    async show(key, cfg) {
        if (this.activeKey === key && this.pointCloud) return;
        if (this.activeAbort) this.activeAbort.abort();
        const abort = new AbortController();
        this.activeAbort = abort;
        this.activeKey = key;

        this.setMessage('Loading…');

        // Per-scene state captured in closure so a later `show` call
        // can't race-write the earlier scene's buffers.
        let positions = null;
        let colors = null;
        let xform = null;           // {cx, cy, cz, flipY, radius}
        let writeOffset = 0;        // next unfilled point slot
        let flushedUpTo = 0;        // slots already committed to GPU
        let flushScheduled = false;
        let firstBlockRendered = false;
        const geomRef = { current: null };   // the geometry we're filling

        const scheduleFlush = () => {
            if (flushScheduled || abort.signal.aborted) return;
            flushScheduled = true;
            requestAnimationFrame(() => {
                flushScheduled = false;
                if (abort.signal.aborted || this.activeKey !== key) return;
                const geo = geomRef.current;
                if (!geo) return;
                const from = flushedUpTo;
                const to = writeOffset;
                if (from >= to) return;
                // Sub-upload just the newly-filled slice.  updateRange is
                // in *scalar* units (3 floats per point for position,
                // 3 bytes per point for colour).
                const posAttr = geo.getAttribute('position');
                const colAttr = geo.getAttribute('color');
                posAttr.updateRange.offset = from * 3;
                posAttr.updateRange.count  = (to - from) * 3;
                posAttr.needsUpdate = true;
                colAttr.updateRange.offset = from * 3;
                colAttr.updateRange.count  = (to - from) * 3;
                colAttr.needsUpdate = true;
                geo.setDrawRange(0, to);
                flushedUpTo = to;
            });
        };

        try {
            const onHeader = (header) => {
                if (abort.signal.aborted) return;
                positions = new Float32Array(header.count * 3);
                colors    = new Uint8Array(header.count * 3);
                xform = computeXformFromHeader(header, cfg);
                // Install an empty geometry (drawRange=0); blocks fill it.
                geomRef.current = this._installGeometry(
                    positions, colors, header.count, 0, cfg, xform.radius
                );
            };

            const onBlock = (header, bytes, byteOffset, blockCount, blockIdx) => {
                if (abort.signal.aborted) return;
                decodeBlock(bytes, byteOffset, blockCount, positions, colors,
                            writeOffset, header);
                applyTransform(positions, writeOffset, blockCount, xform);
                writeOffset += blockCount;
                if (!firstBlockRendered) {
                    firstBlockRendered = true;
                    this.setMessage('');
                }
                scheduleFlush();
            };

            const onProgress = (pct) => {
                if (abort.signal.aborted || firstBlockRendered) return;
                this.setMessage(`Loading ${pct}%`);
            };

            await this.loader.load(cfg.cloud, {
                signal: abort.signal, onHeader, onBlock, onProgress
            });
            if (abort.signal.aborted || this.activeKey !== key) return;

            // Commit any residual slots that hadn't flushed yet.
            if (writeOffset > flushedUpTo) scheduleFlush();
            this.setMessage('');
        } catch (err) {
            if (err.name === 'AbortError' || abort.signal.aborted) return;
            console.error('Error loading point cloud:', err);
            this.setMessage('Failed to load point cloud');
        }
    }

    // Install an (initially empty) full-sized geometry.  Positions &
    // colours are the pre-allocated typed arrays that blocks will fill
    // in place.  We set `boundingSphere` explicitly from the header's
    // known full-scene radius so three.js never re-derives it from the
    // (still-zero) unfilled slots.  Returns the geometry so the caller
    // can retain a reference for later updateRange plumbing.
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
        return geometry;
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
