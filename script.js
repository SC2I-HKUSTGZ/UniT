/* ========================================
   UniT project page — interactive examples
   Chapter nav + PNT v4 point-cloud viewer + video+cover sync

   Loading strategy (lazy init + true streaming + bounded memory)
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
     - `DecompressionStream('gzip')` feeds a rolling scratch buffer.  A
       parser walks it block-by-block, emitting `onBlock` callbacks the
       instant each block has fully arrived, then drops consumed bytes.
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
     - The whole viewer is lazy-started only when the user explicitly
       opens or interacts with the Examples section.  This keeps the
       project page and Results section cheap to open.
     - Scene switches abort the active fetch/decompress/read loop and
       stale GPU flushes so old loads cannot build up behind the current
       thumbnail.
   ======================================== */

document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initIntroVideo();   // lazy-loads intro.mp4 only when it scrolls into view
    evictStaleCaches(); // removes stale unit-pnt-* caches from older viewer builds
    initDeferredDemo(); // starts Three.js only after explicit Examples intent
});

// ========================================
// Intro video — lazy load
//
// protected.html ships the intro video with `preload="none"` so the
// 18 MB mp4 doesn't race the 55 MB default point cloud at page load.
// Flip it to `auto` + call play() only once the element is within the
// viewport.  In practice the intro is above the fold, so this fires
// within the first few hundred ms of page load — but *after* the
// viewer has already started streaming its cloud.
// ========================================
function initIntroVideo() {
    const v = document.getElementById('intro-video');
    if (!v) return;
    // Once the video enters the viewport, nudge preload to "auto" so
    // browsers that treat `preload="none" autoplay` as a no-op (Safari)
    // still start playback.  We intentionally do NOT call v.load() — that
    // resets currentTime and forces a re-fetch even if Chrome has already
    // begun loading via the intrinsic autoplay path.  Just flipping
    // preload + calling play() is the minimum work needed.
    const arm = () => {
        if (v.preload !== 'auto') v.preload = 'auto';
        if (v.paused) {
            const p = v.play();
            if (p && typeof p.catch === 'function') p.catch(() => {});
        }
    };
    if ('IntersectionObserver' in window) {
        const io = new IntersectionObserver((entries, obs) => {
            for (const e of entries) {
                if (!e.isIntersecting) continue;
                obs.disconnect();
                arm();
            }
        }, { threshold: 0.1 });
        io.observe(v);
    } else {
        // Older browser — just arm it after the viewer has a head start.
        setTimeout(arm, 800);
    }
}

// ========================================
// Cache eviction
//
// The Cache API name embeds a version number (unit-pnt-vN) that we bump
// whenever .pnt.gz on disk changes format or geometry.  Without explicit
// cleanup, every bump leaks up to ~175 MB of stale cache per user (the
// old entries still live under the old cache name forever).  On startup,
// delete every unit-pnt-* cache that isn't the current one.
// ========================================
function evictStaleCaches() {
    if (!('caches' in self)) return;
    caches.keys().then(names => {
        for (const n of names) {
            if (n.startsWith('unit-pnt-') && n !== CACHE_NAME) {
                caches.delete(n).catch(() => {});
            }
        }
    }).catch(() => {});
}

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
// Deferred Examples bootstrap
//
// The point-cloud viewer is the only heavyweight part of the page: it
// creates a WebGL renderer, downloads/decompresses a 55 MB default scene,
// allocates large typed arrays, and renders while active. Keep all of that
// behind explicit user intent so opening the project page or jumping past
// Examples to Results cannot saturate the GPU.
// ========================================
function initDeferredDemo() {
    const demo = document.getElementById('demo');
    if (!demo) return;

    let started = false;
    const start = () => {
        if (started) return;
        started = true;
        initDemo().catch(err => {
            console.error('Failed to initialize demo:', err);
            started = false;
        });
    };

    const demoButton = document.querySelector('.chapters button[data-section="demo"]');
    if (demoButton) {
        demoButton.addEventListener('click', () => requestAnimationFrame(start), { once: true });
    }

    demo.addEventListener('click', start, { once: true });
}

// ========================================
// Per-demo configuration
//
// Fields baked at build time (drive the initial look of each scene):
//   cloud         — .pnt.gz path
//   pointSize     — splat size in world units, tuned so different
//                   scene extents read consistently
//   flipY         — negate Y after centring (handy when the source
//                   uses a +Y-down convention)
//   camera        — unit offset from the bounding-sphere centre,
//                   scaled by the sphere radius on load
//   samplingRate  — fraction of points drawn [0..1].  Because the
//                   .pnt.gz stream is randomly shuffled, any prefix
//                   is a uniform spatial sample — so this is just a
//                   drawRange cutoff, no re-sampling cost.
//   brightness    — scalar multiplier applied to vertex colours in
//                   the fragment shader (values > 1 are allowed;
//                   they brighten under-exposed scans).
//   background    — CSS colour for the canvas clear colour.
//   rotation      — extra Euler angles (degrees, XYZ) applied to
//                   the point cloud object; use this to straighten
//                   scenes whose native axes aren't world-aligned.
//
// All of these are exposed in the live controls panel; the copy-
// config button in that panel prints a block that can be pasted
// back in here verbatim.
// ========================================
// ----- Previous tuned defaults — kept commented for quick restoration. -----
// const DEMO_CONFIGS = {
//     hkust_intr: {
//         title: 'HKUST (GZ) INTR',
//         cloud: 'assets/demos/hkust_intr/scene.pnt.gz',
//         pointSize: 0.011,
//         flipY: true,
//         camera: { x: -0.6, y: 0.3, z: -1.4 },
//         samplingRate: 1.0,
//         brightness: 1.0,
//         background: '#ffffff',
//         rotation: { x: 0, y: 0, z: 0 }
//     },
//     hkust_toy: {
//         title: 'HKUST (GZ) Toy',
//         cloud: 'assets/demos/hkust_toy/scene.pnt.gz',
//         pointSize: 0.0042,
//         flipY: true,
//         camera: { x: -0.4, y: 0.2, z: -1.5 },
//         samplingRate: 1.0,
//         brightness: 1.0,
//         background: '#ffffff',
//         rotation: { x: 0, y: 0, z: 0 }
//     },
//     hkust_redbird: {
//         title: 'HKUST (GZ) Red Bird',
//         cloud: 'assets/demos/hkust_redbird/scene.pnt.gz',
//         pointSize: 0.044,
//         flipY: true,
//         camera: { x: -0.5, y: 0.35, z: -1.6 },
//         samplingRate: 1.0,
//         brightness: 1.0,
//         background: '#ffffff',
//         rotation: { x: 0, y: 0, z: 0 }
//     },
//     drift: {
//         title: 'Drift',
//         cloud: 'assets/demos/drift/scene.pnt.gz',
//         pointSize: 0.145,
//         flipY: true,
//         camera: { x: -0.25, y: 0.25, z: -0.75 },
//         samplingRate: 1.0,
//         brightness: 1.0,
//         background: '#ffffff',
//         rotation: { x: 0, y: 0, z: 0 }
//     },
//     gta_sfm: {
//         title: 'GTA SfM',
//         cloud: 'assets/demos/gta_sfm/scene.pnt.gz',
//         pointSize: 0.082,
//         flipY: true,
//         camera: { x: -0.4, y: 0.2, z: -1.4 },
//         samplingRate: 1.0,
//         brightness: 1.0,
//         background: '#ffffff',
//         rotation: { x: 0, y: 0, z: 0 }
//     },
//     kitti: {
//         title: 'KITTI',
//         cloud: 'assets/demos/kitti/scene.pnt.gz',
//         pointSize: 0.30,
//         flipY: true,
//         camera: { x: -0.15, y: 0.22, z: -0.5 },
//         samplingRate: 1.0,
//         brightness: 1.0,
//         background: '#ffffff',
//         rotation: { x: 0, y: 0, z: 0 }
//     }
// };

// Hard-coded from ply_config.txt — initial viewer settings per scene.
const DEMO_CONFIGS = {
    hkust_intr: {
        title: 'HKUST (GZ) INTR',
        cloud: 'assets/demos/hkust_intr/scene.pnt.gz',
        preview: 'assets/demos/hkust_intr/preview.pnt.gz',
        pointSize: 0.012007,
        flipY: true,
        camera: { x: -0.489, y: 0.244, z: -1.140 },
        samplingRate: 1.000,
        brightness: 1.00,
        background: '#ffffff',
        rotation: { x: 3, y: 25, z: 0 }
    },
    hkust_toy: {
        title: 'HKUST (GZ) Toy',
        cloud: 'assets/demos/hkust_toy/scene.pnt.gz',
        preview: 'assets/demos/hkust_toy/preview.pnt.gz',
        pointSize: 0.000001,
        flipY: true,
        camera: { x: -0.400, y: 0.200, z: -1.500 },
        samplingRate: 1.000,
        brightness: 1.00,
        background: '#ffffff',
        rotation: { x: 10, y: 12, z: 0 }
    },
    hkust_redbird: {
        title: 'HKUST (GZ) Red Bird',
        cloud: 'assets/demos/hkust_redbird/scene.pnt.gz',
        preview: 'assets/demos/hkust_redbird/preview.pnt.gz',
        pointSize: 0.0000,
        flipY: true,
        camera: { x: -0.349, y: 0.244, z: -1.117 },
        samplingRate: 0.800,
        brightness: 1.00,
        background: '#ffffff',
        rotation: { x: -3, y: 43, z: 0 }
    },
    drift: {
        title: 'Drift',
        cloud: 'assets/demos/drift/scene.pnt.gz',
        preview: 'assets/demos/drift/preview.pnt.gz',
        pointSize: 0.000001,
        flipY: true,
        camera: { x: -0.250, y: 0.250, z: -0.750 },
        samplingRate: 0.800,
        brightness: 1.00,
        background: '#ffffff',
        rotation: { x: -16, y: -46, z: 0 }
    },
    gta_sfm: {
        title: 'GTA SfM',
        cloud: 'assets/demos/gta_sfm/scene.pnt.gz',
        preview: 'assets/demos/gta_sfm/preview.pnt.gz',
        pointSize: 0.000001,
        flipY: true,
        camera: { x: -0.361, y: 0.180, z: -1.263 },
        samplingRate: 0.800,
        brightness: 1.00,
        background: '#ffffff',
        rotation: { x: -23, y: 4, z: 0 }
    },
    kitti: {
        title: 'KITTI',
        cloud: 'assets/demos/kitti/scene.pnt.gz',
        preview: 'assets/demos/kitti/preview.pnt.gz',
        pointSize: 0.103814,
        flipY: true,
        camera: { x: -0.308, y: 0.451, z: -1.025 },
        samplingRate: 0.980,
        brightness: 1.00,
        background: '#ffffff',
        rotation: { x: 23, y: 10, z: 0 }
    }
};

// Bumped to v2 so existing per-user overrides from the old baked
// defaults are discarded in favour of the new DEMO_CONFIGS values.
const CONTROLS_STORAGE_KEY = 'unit-demo-controls-v2';

const CACHE_NAME = 'unit-pnt-v11';

const LOAD_POINT_BUDGET_FACTOR = 0.65;
const INTERACTION_POINT_BUDGET_FACTOR = 0.45;
const INTERACTION_BUDGET_SETTLE_MS = 450;
const PNT_COMPACT_THRESHOLD = 1 << 20;

window.__UNIT_DEMO_PERF = window.__UNIT_DEMO_PERF || {
    activeScene: null,
    activeLoadId: 0,
    activeUrl: null,
    loading: false,
    abortedLoads: 0,
    completedLoads: 0,
    livePointClouds: 0,
    activeThumbTimers: 0,
    visiblePoints: 0,
    loadedPoints: 0,
    totalPoints: 0,
    status: 'idle'
};

const DEMO_DEPENDENCY_SCRIPTS = [
    'https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js',
    'https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js'
];

let demoDependenciesPromise = null;

function loadScriptOnce(src) {
    return new Promise((resolve, reject) => {
        const absoluteSrc = new URL(src, document.baseURI).href;
        const existing = Array.from(document.scripts).find(s => s.src === absoluteSrc);
        if (existing) {
            if (existing.dataset.loaded === 'true') {
                resolve();
                return;
            }
            existing.addEventListener('load', () => resolve(), { once: true });
            existing.addEventListener('error', () => reject(new Error(`Failed to load ${src}`)), { once: true });
            return;
        }
        const script = document.createElement('script');
        script.src = src;
        script.async = false;
        script.onload = () => {
            script.dataset.loaded = 'true';
            resolve();
        };
        script.onerror = () => reject(new Error(`Failed to load ${src}`));
        document.head.appendChild(script);
    });
}

function ensureDemoDependencies() {
    if (window.THREE && window.THREE.OrbitControls) return Promise.resolve();
    if (!demoDependenciesPromise) {
        demoDependenciesPromise = DEMO_DEPENDENCY_SCRIPTS
            .reduce((p, src) => p.then(() => loadScriptOnce(src)), Promise.resolve())
            .then(() => {
                if (!window.THREE || !window.THREE.OrbitControls) {
                    throw new Error('Three.js demo dependencies did not initialize');
                }
            });
    }
    return demoDependenciesPromise;
}

// Small Cloudflare Worker (in webpage/worker/) that stores the current
// per-scene initial view as a single JSON blob in Workers KV.  Source of
// truth for the baked defaults once deployed — every page load fetches
// this on top of DEMO_CONFIGS, and admins (URL ?setting=<secret>) can
// POST new values from the "Save as Initial View" button.
const CONFIG_WORKER_URL = 'https://unit-demo-config.enceladus-huang.workers.dev';

// Fetch the remote config and mutate DEMO_CONFIGS in place so every
// later baked-config read sees the live values.  Returns the list of
// keys that were actually patched so callers can re-apply them to the
// live viewer without touching unrelated scenes.  Failures fall through
// silently — the page still renders with the hard-coded defaults.
async function applyRemoteConfig() {
    try {
        const resp = await fetch(CONFIG_WORKER_URL + '/config', { cache: 'no-store' });
        if (!resp.ok) return [];
        const remote = await resp.json();
        if (!remote || typeof remote !== 'object') return [];
        const patched = [];
        for (const [key, patch] of Object.entries(remote)) {
            if (!DEMO_CONFIGS[key] || !patch || typeof patch !== 'object') continue;
            const baked = DEMO_CONFIGS[key];
            DEMO_CONFIGS[key] = {
                ...baked,
                ...patch,
                camera:   { ...(baked.camera   || {}), ...(patch.camera   || {}) },
                rotation: { ...(baked.rotation || {}), ...(patch.rotation || {}) },
            };
            patched.push(key);
        }
        return patched;
    } catch { /* network error — continue with baked defaults */ return []; }
}

// ========================================
// Interactive Examples
// ========================================
async function initDemo() {
    const canvas = document.getElementById('demo-canvas');
    if (!canvas) return;
    const messageEl = document.getElementById('demo-message');
    if (messageEl) {
        messageEl.textContent = 'Preparing 3D viewer...';
        messageEl.style.display = 'flex';
        messageEl.style.opacity = '1';
    }
    await ensureDemoDependencies();

    // Remote config is fetched in parallel with the first PLY download.
    // It used to block the whole viewer for up to 3 s (Promise.race with a
    // 3 s timeout), which served no one: if the Worker is fast, the race
    // paid a round-trip for a patch that's almost always empty; if it's
    // slow, the user stared at "Loading…" for 3 s before any bytes moved.
    // Now we start the default scene with baked DEMO_CONFIGS instantly,
    // and re-apply the live config to the current scene when/if it lands.
    const configPromise = applyRemoteConfig();

    const loader = new PntLoader(CACHE_NAME);
    const viewer = new PointCloudViewer(canvas, messageEl, loader);
    const thumbs = Array.from(document.querySelectorAll('.demo-thumb'));
    const controls = new ViewerControls(viewer);

    let currentDemo = null;
    let activeThumbVideo = null;
    let activeThumbToken = 0;
    let activeThumbLoopTimer = null;

    // When the remote config finally arrives, re-bind the currently-viewed
    // scene if it was patched.  Other scenes pick up the live values on
    // their next select() — no need to rerun the render pipeline.
    configPromise.then((patchedKeys) => {
        if (!patchedKeys || !patchedKeys.length) return;
        if (currentDemo && patchedKeys.includes(currentDemo)) {
            const cfg = DEMO_CONFIGS[currentDemo];
            const resolved = controls.bind(currentDemo, cfg);
            // Update live viewer state without re-loading the cloud.
            viewer.setPointSize(resolved.pointSize);
            if (resolved.samplingRate != null) viewer.setSamplingRate(resolved.samplingRate);
            if (resolved.brightness   != null) viewer.setBrightness(resolved.brightness);
            if (resolved.background)           viewer.setBackground(resolved.background);
            if (resolved.rotation)             viewer.setRotation(resolved.rotation);
            if (resolved.camera)               viewer.setCameraOffset(resolved.camera);
        }
    });

    function select(key) {
        const cfg = DEMO_CONFIGS[key];
        if (!cfg) return;
        if (key === currentDemo) {
            const selectedVideo = document.querySelector(`.demo-thumb[data-demo="${key}"] video`);
            if (selectedVideo) focusThumbVideo(selectedVideo);
            return;
        }
        currentDemo = key;
        let selectedVideo = null;
        thumbs.forEach(t => {
            const isSelected = t.dataset.demo === key;
            t.classList.toggle('selected', isSelected);
            const v = t.querySelector('video');
            if (!v) return;
            if (isSelected) selectedVideo = v;
            else blurThumbVideo(v);
        });
        if (selectedVideo) focusThumbVideo(selectedVideo);
        const resolved = controls.bind(key, cfg);
        viewer.show(key, resolved);
    }

    // Start (or resume) autoplay on the currently-focused thumbnail.
    // Three independent things can defer playback here:
    //   1. `preload="metadata"` keeps the off-screen thumbs light, so
    //      the first play() call can fire before a single frame has
    //      been decoded — retry on loadeddata/canplay covers that.
    //   2. Chrome's muted-autoplay policy suspends playback whenever
    //      the element is outside the viewport, even if play() was
    //      previously invoked.  The thumbnails live well below the
    //      fold, so on first load the select() call lands before the
    //      user has scrolled to them.  An IntersectionObserver kicks
    //      play() again each time the selected thumb becomes visible.
    //   3. Chrome also pauses media when the tab is hidden.  When the
    //      user comes back (visibilitychange → "visible"), re-arm the
    //      currently-selected thumb so it resumes looping.
    const isActiveThumbVideo = (v, token = activeThumbToken) =>
        v && v === activeThumbVideo && token === activeThumbToken;

    const pauseInactiveThumbVideos = () => {
        thumbs.forEach(t => {
            const v = t.querySelector('video');
            if (!v || v === activeThumbVideo) return;
            if (!v.paused) v.pause();
        });
    };

    const rewindIfAtEnd = (v, force = false) => {
        const duration = Number.isFinite(v.duration) ? v.duration : 0;
        const atEnd = v.ended || (duration > 0 && v.currentTime >= duration - 0.08);
        if (!force && !atEnd) return;
        try { v.currentTime = 0; } catch {}
    };

    const tryPlay = (v, token = activeThumbToken) => {
        if (!isActiveThumbVideo(v, token)) return;
        pauseInactiveThumbVideos();
        rewindIfAtEnd(v);
        if (!v.paused) return;
        const p = v.play();
        if (p && typeof p.catch === 'function') {
            p.catch(() => {
                if (isActiveThumbVideo(v, token)) pauseInactiveThumbVideos();
            });
        }
    };

    const maintainActiveThumbLoop = (v, token = activeThumbToken) => {
        if (!isActiveThumbVideo(v, token)) return;
        const thumb = v.closest('.demo-thumb');
        if (!thumb || !thumb.classList.contains('selected')) return;

        v.muted = true;
        v.loop = true;
        v.playsInline = true;
        if (v.playbackRate === 0) v.playbackRate = 1;

        const duration = Number.isFinite(v.duration) ? v.duration : 0;
        if (v.ended || (duration > 0 && v.currentTime >= duration - 0.08)) {
            rewindIfAtEnd(v, true);
        }
        tryPlay(v, token);
    };

    const armActiveThumbLoop = (v, token) => {
        if (activeThumbLoopTimer) clearInterval(activeThumbLoopTimer);
        activeThumbLoopTimer = setInterval(() => {
            maintainActiveThumbLoop(v, token);
        }, 250);
        window.__UNIT_DEMO_PERF.activeThumbTimers = 1;
    };

    const getSelectedVideo = () => {
        const sel = document.querySelector('.demo-thumb.selected video');
        return sel || null;
    };
    const thumbVisibility = new IntersectionObserver((entries) => {
        for (const entry of entries) {
            if (!entry.isIntersecting) continue;
            const v = entry.target;
            const thumb = v.closest('.demo-thumb');
            if (thumb && thumb.classList.contains('selected')) tryPlay(v);
        }
    }, { threshold: 0.1 });
    document.addEventListener('visibilitychange', () => {
        if (document.visibilityState === 'visible') maintainActiveThumbLoop(getSelectedVideo());
    });

    function focusThumbVideo(v) {
        activeThumbVideo = v;
        activeThumbToken += 1;
        const token = activeThumbToken;
        pauseInactiveThumbVideos();

        v.muted = true;
        v.loop = true;
        v.playsInline = true;
        if (v.preload !== 'auto') v.preload = 'auto';
        rewindIfAtEnd(v, true);
        armActiveThumbLoop(v, token);
        const start = () => {
            if (!isActiveThumbVideo(v, token)) return;
            pauseInactiveThumbVideos();
            maintainActiveThumbLoop(v, token);
        };
        start();                                            // kick off loading + play
        thumbVisibility.observe(v);
    }

    function blurThumbVideo(v) {
        thumbVisibility.unobserve(v);
        v.pause();
        try { v.currentTime = 0; } catch {}
    }

    thumbs.forEach(t => {
        const v = t.querySelector('video');
        if (v) {
            const enforceSingleActivePlayback = () => {
                if (v === activeThumbVideo) {
                    pauseInactiveThumbVideos();
                } else {
                    v.pause();
                    try { v.currentTime = 0; } catch {}
                }
            };
            v.addEventListener('play', enforceSingleActivePlayback);
            v.addEventListener('playing', enforceSingleActivePlayback);
            v.addEventListener('ended', () => maintainActiveThumbLoop(v));
            v.addEventListener('pause', () => {
                if (v === activeThumbVideo) {
                    setTimeout(() => maintainActiveThumbLoop(v), 0);
                }
            });
            v.addEventListener('loadeddata', () => maintainActiveThumbLoop(v));
            v.addEventListener('canplay', () => maintainActiveThumbLoop(v));
            v.addEventListener('stalled', () => maintainActiveThumbLoop(v));
            v.addEventListener('suspend', () => maintainActiveThumbLoop(v));
            v.addEventListener('timeupdate', () => {
                if (v === activeThumbVideo) rewindIfAtEnd(v);
            });
        }
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
//      Each callback receives a typed-array view into a rolling buffer;
//      consumed bytes are compacted away so the full decompressed file is
//      never retained alongside the final GPU buffers.
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

    // Fetch compressed bytes (cache-then-network). Foreground loads do
    // not clone responses into Cache API: a clone keeps downloading even
    // after the visible load is aborted, which is exactly the hidden work
    // that makes rapid scene switching accumulate resources. Browser HTTP
    // cache still handles repeat visits; explicit prefetch() remains
    // available for future non-interactive warming.
    async _fetchCompressed(url, signal) {
        if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');
        const cache = await this._openCache();
        if (cache) {
            const hit = await cache.match(url);
            if (hit) return hit;
        }
        // `priority: 'high'` marks the foreground load. Safe fallback:
        // browsers that don't support it just ignore the hint.
        const resp = await fetch(url, { signal, priority: 'high' });
        if (!resp.ok) throw new Error(`HTTP ${resp.status} for ${url}`);
        return resp;
    }

    // Background prefetch: store the compressed bytes in Cache API, skip
    // decompression.  Subsequent .load() calls hit the cache.
    //
    // We stream-drain the response body into a WritableStream sink rather
    // than calling resp.arrayBuffer().  arrayBuffer() allocates a
    // contiguous buffer sized for the whole payload (up to 55 MB here),
    // which sits on the heap until the caller's scope ends.  The sink
    // version lets each chunk be GC'd as soon as cache.put() has consumed
    // its clone, so prefetch holds only a few KB of transient memory
    // regardless of file size.
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
            // Drain the body without materialising an ArrayBuffer.
            if (resp.body) {
                try { await resp.body.pipeTo(new WritableStream()); } catch {}
            }
        })();
        this.inflight.set(url, p);
        p.finally(() => this.inflight.delete(url));
        return p;
    }

    // Streaming load with per-block callbacks.
    async load(url, { signal, onHeader, onBlock, onProgress } = {}) {
        const resp = await this._fetchCompressed(url, signal);
        if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');

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
        const abortReader = () => reader.cancel().catch(() => {});
        if (signal) {
            if (signal.aborted) {
                abortReader();
                throw new DOMException('Aborted', 'AbortError');
            }
            signal.addEventListener('abort', abortReader, { once: true });
        }

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

        try {
            while (true) {
                if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');
                const { done, value } = await reader.read();
                if (done) break;
                if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');
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
                    if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');
                    const bc = (blocksDecoded < header.numBlocks - 1)
                        ? header.blockSize
                        : (header.count - blocksDecoded * header.blockSize);
                    const blockBytes = bc * 9;                 // SoA: 6 bytes pos + 3 bytes colour per point
                    if (len - cursor < blockBytes) break;      // block still in-flight
                    if (onBlock) onBlock(header, buf, cursor, bc, blocksDecoded);
                    cursor += blockBytes;
                    blocksDecoded++;
                }

                // Drop bytes that have already been parsed.  This keeps
                // transient decompressed memory bounded by incoming chunks
                // plus at most one incomplete block.
                if (cursor > 0 && (cursor >= PNT_COMPACT_THRESHOLD || cursor > buf.length / 2)) {
                    buf.copyWithin(0, cursor, len);
                    len -= cursor;
                    cursor = 0;
                }
            }
        } finally {
            if (signal) signal.removeEventListener('abort', abortReader);
        }

        if (onProgress) onProgress(100);
        return { header: header ?? parsePntV4Header(buf.buffer, 0), blocksDecoded };
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

// Translate (+ optional flipY, + unconditional flipX) a contiguous slice
// of `positions` in-place.  Algebraic identity: `(-x - (-cx_raw))` ==
// `-(x - cx_raw)`, so we can translate first (all axes) then flip the
// signs, and still end up with the same layout as "flip first, then
// centre on the flipped axis".
//
// The X negation is applied unconditionally because every reconstructed
// cloud comes out horizontally mirrored relative to its reference video
// under this pipeline's coordinate convention; flipping X after centring
// corrects that once for every scene.
function applyTransform(positions, start, count, xform) {
    const { cx, cy, cz, flipY } = xform;
    const end = start + count;
    for (let i = start; i < end; i++) {
        const k = i * 3;
        positions[k]     = -(positions[k]     - cx);
        positions[k + 1] -=   cy;
        positions[k + 2] -=   cz;
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
        this.activeLoadId = 0;
        this._pendingFlushRaf = null;
        this._interactionBudgetTimer = null;
        this._loadingBudget = false;
        this._interactionBudget = false;
        this._baseSamplingRate = 1;
        this._loadedPoints = 0;

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
            powerPreference: 'low-power'
        });
        this.renderer.setSize(width, height);
        this.renderer.setPixelRatio(1);

        this.controls = new THREE.OrbitControls(this.camera, this.canvas);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.08;
        this.controls.minDistance = 0.05;
        this.controls.maxDistance = 1000;
        // Orbiting the camera is disabled: dragging rotates the point
        // cloud itself (see _installRotateDrag), so the X/Y/Z rotation
        // values shown in the controls panel always match the on-screen
        // orientation.  Zoom (wheel) and pan (right-click) stay on
        // OrbitControls since they don't affect that invariant.
        this.controls.enableRotate = false;
        this.controls.addEventListener('start', () => this._beginInteractionBudget());
        this.controls.addEventListener('end', () => this._endInteractionBudgetSoon());
        this.canvas.addEventListener('wheel', () => this._markInteractionActive(), { passive: true });

        // Callback invoked whenever drag-rotation updates the cloud's
        // orientation; ViewerControls wires this up to keep its sliders
        // in sync.  Signature: ({x, y, z}) in degrees, each wrapped to
        // [-180, 180] and rounded to match the slider step.
        this.onRotationChanged = null;

        // Current scene's effective config — mutated by the controls panel.
        this._effective = null;
        this._totalPoints = 0;

        this._installRotateDrag();
        this._installRenderVisibilityGate();
    }

    // Custom drag-to-rotate: left-button drag over the canvas rotates
    // the point cloud directly, so the effective orientation is always
    // reflected back in the Pitch/Yaw/Roll sliders without a quaternion
    // round-trip.  Horizontal drag adds to the Y (yaw) angle, vertical
    // drag adds to the X (pitch) angle; Z (roll) is only touchable via
    // the slider.  Euler angles compose in the object's local frame,
    // which matches how setRotation() from the sliders drives the scene
    // so the two controls stay in exact lockstep.
    _installRotateDrag() {
        let dragging = false;
        let pid = null;
        let lastX = 0, lastY = 0;
        let startX = 0, startY = 0;
        let moved = false;
        const THRESH = 5; // px before drag becomes a rotation

        const d2r = Math.PI / 180;
        const r2d = 180 / Math.PI;
        const wrap = (deg) => Math.round(((deg + 180) % 360 + 360) % 360 - 180);

        this.canvas.addEventListener('pointerdown', (e) => {
            if (e.button !== 0) return;          // left / primary only
            if (!this.pointCloud) return;
            dragging = true;
            pid = e.pointerId;
            startX = lastX = e.clientX;
            startY = lastY = e.clientY;
            moved = false;
            this._beginInteractionBudget();
            try { this.canvas.setPointerCapture(pid); } catch {}
        });

        this.canvas.addEventListener('pointermove', (e) => {
            if (!dragging || e.pointerId !== pid || !this.pointCloud) return;
            if (!moved) {
                if (Math.hypot(e.clientX - startX, e.clientY - startY) < THRESH) return;
                moved = true;
            }
            const dx = e.clientX - lastX;
            const dy = e.clientY - lastY;
            lastX = e.clientX;
            lastY = e.clientY;
            this._markInteractionActive();

            const rect = this.canvas.getBoundingClientRect();
            // Full-width drag ≈ one full yaw turn; full-height ≈ one
            // pitch turn.  Matches OrbitControls' default sensitivity
            // closely enough that the interaction feels identical.
            //
            // Pitch sign: the baked cameras all sit at -Z looking at the
            // origin, so a positive world-X rotation tilts the cloud's
            // top AWAY from the viewer — the opposite of the standard
            // "drag down ⇒ see more of the top" convention used by
            // OrbitControls.  Negate dy so the drag direction matches
            // what the user expects.  Yaw already matches because the
            // rotation axis and angular direction in world space are
            // unchanged by applyTransform's X mirror, so a positive
            // world-Y rotation still reads as "scene rotated right" on
            // screen.
            const yawDeg   =  (dx / rect.width)  * 360;
            const pitchDeg = -(dy / rect.height) * 360;

            const rot = this.pointCloud.rotation;
            rot.y += yawDeg   * d2r;
            rot.x += pitchDeg * d2r;

            if (this.onRotationChanged) {
                this.onRotationChanged({
                    x: wrap(rot.x * r2d),
                    y: wrap(rot.y * r2d),
                    z: wrap(rot.z * r2d)
                });
            }
        });

        const end = (e) => {
            if (!dragging || e.pointerId !== pid) return;
            dragging = false;
            this._endInteractionBudgetSoon();
            try { this.canvas.releasePointerCapture(pid); } catch {}
            pid = null;
        };
        this.canvas.addEventListener('pointerup', end);
        this.canvas.addEventListener('pointercancel', end);
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
                ? 'Measuring — click two points on the cloud (click here to exit)'
                : 'Toggle distance measurement — click two points on the cloud to measure';
            const labelEl = this.measureBtn.querySelector('.measure-label');
            if (labelEl) {
                labelEl.textContent = this.measureMode ? 'Measuring…' : 'Measure distance';
            }
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
        // Force matrices up-to-date before raycasting: the drag handler
        // mutates pointCloud.rotation outside the render loop, and a click
        // can land between frames (before renderer.render() refreshes
        // matrixWorld).  Without this the raycaster silently samples a
        // stale transform and misses every point.
        this.camera.updateMatrixWorld();
        this.pointCloud.updateMatrixWorld();
        // Widen the raycast threshold for the click path so aim-precision
        // doesn't have to match the rendered pixel size.  The render-side
        // threshold stays at pointSize × 1.5 (set in _installGeometry /
        // setPointSize).  ~1% of the scene radius gives the user a few
        // CSS pixels of click slack on every scene scale.
        const prevThreshold = this.raycaster.params.Points.threshold;
        this.raycaster.params.Points.threshold = Math.max(
            prevThreshold, (this._sceneRadius || 1) * 0.01
        );
        this.raycaster.setFromCamera(this.mouse, this.camera);
        const hits = this.raycaster.intersectObject(this.pointCloud);
        this.raycaster.params.Points.threshold = prevThreshold;
        // Raycaster returns the hit in world space. Convert to the point
        // cloud's local frame so markers parented to `this.pointCloud`
        // track the cloud when it rotates — otherwise a later drag leaves
        // the markers floating at their original world position while the
        // points they were anchored to move away.
        if (hits.length > 0) {
            const local = this.pointCloud.worldToLocal(hits[0].point.clone());
            this.addMeasurePoint(local);
        }
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
        this.pointCloud.add(halo);
        this.pointCloud.add(core);
        this.measureMarkers.push(halo, core);

        if (this.measurePoints.length === 2) {
            this.drawMeasureLine();
            this.showDistance();
        }
    }

    drawMeasureLine() {
        if (this.measureLine) {
            this.measureLine.parent?.remove(this.measureLine);
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
        this.pointCloud.add(this.measureLine);
    }

    showDistance() {
        const d = this.measurePoints[0].distanceTo(this.measurePoints[1]);
        const formatted = d >= 10 ? d.toFixed(1) : d.toFixed(2);
        if (this.measureHintEl) {
            this.measureHintEl.textContent = `Distance: ${formatted} m`;
        }

        if (this.measureLabel) {
            this.measureLabel.parent?.remove(this.measureLabel);
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
        this.pointCloud.add(this.measureLabel);
    }

    clearMeasurement() {
        this.measureMarkers.forEach(m => {
            m.parent?.remove(m);
            m.geometry.dispose();
            m.material.dispose();
        });
        this.measureMarkers = [];
        if (this.measureLine) {
            this.measureLine.parent?.remove(this.measureLine);
            this.measureLine.geometry.dispose();
            this.measureLine.material.dispose();
            this.measureLine = null;
        }
        if (this.measureLabel) {
            this.measureLabel.parent?.remove(this.measureLabel);
            this.measureLabel.material.map?.dispose();
            this.measureLabel.material.dispose();
            this.measureLabel = null;
        }
        this.measurePoints = [];
        if (this.measureHintEl && this.measureMode) {
            this.measureHintEl.textContent = 'Click two points to measure distance';
        }
    }

    _updatePerf(partial) {
        Object.assign(window.__UNIT_DEMO_PERF, partial);
    }

    _visiblePointCap() {
        const total = this._totalPoints || 0;
        const loaded = this._loadedPoints || 0;
        if (!total || !loaded) return 0;
        let factor = this._baseSamplingRate;
        if (this._loadingBudget) factor *= LOAD_POINT_BUDGET_FACTOR;
        if (this._interactionBudget) factor *= INTERACTION_POINT_BUDGET_FACTOR;
        const cap = Math.max(1, Math.floor(total * Math.max(0, Math.min(1, factor))));
        return Math.min(loaded, cap);
    }

    _applyDrawBudget() {
        if (!this.pointCloud) return;
        const visible = this._visiblePointCap();
        this.pointCloud.geometry.setDrawRange(0, visible);
        this._updatePerf({
            visiblePoints: visible,
            loadedPoints: this._loadedPoints,
            totalPoints: this._totalPoints,
            loading: this._loadingBudget,
        });
    }

    _beginInteractionBudget() {
        if (!this.pointCloud) return;
        if (this._interactionBudgetTimer) clearTimeout(this._interactionBudgetTimer);
        if (!this._interactionBudget) {
            this._interactionBudget = true;
            this._applyDrawBudget();
        }
    }

    _endInteractionBudgetSoon(delay = INTERACTION_BUDGET_SETTLE_MS) {
        if (this._interactionBudgetTimer) clearTimeout(this._interactionBudgetTimer);
        this._interactionBudgetTimer = setTimeout(() => {
            this._interactionBudget = false;
            this._applyDrawBudget();
        }, delay);
    }

    _markInteractionActive() {
        this._beginInteractionBudget();
        this._endInteractionBudgetSoon();
    }

    _cancelPendingFlush() {
        if (this._pendingFlushRaf) {
            cancelAnimationFrame(this._pendingFlushRaf);
            this._pendingFlushRaf = null;
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
        this._cancelPendingFlush();
        if (this.pointCloud) {
            this.scene.remove(this.pointCloud);
            this.pointCloud.geometry.dispose();
            this.pointCloud.material.dispose();
            this.pointCloud = null;
        }
        this._loadedPoints = 0;
        this._totalPoints = 0;
        this._updatePerf({
            livePointClouds: 0,
            visiblePoints: 0,
            loadedPoints: 0,
            totalPoints: 0,
        });
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
        this._cancelPendingFlush();
        const abort = new AbortController();
        this.activeAbort = abort;
        this.activeKey = key;
        const loadId = ++this.activeLoadId;
        this._baseSamplingRate = Math.max(0, Math.min(1, cfg.samplingRate != null ? cfg.samplingRate : 1));
        this._loadingBudget = true;
        this._interactionBudget = false;

        this.setMessage('Loading…');
        this._updatePerf({
            activeScene: key,
            activeLoadId: loadId,
            activeUrl: cfg.cloud,
            loading: true,
            status: 'loading',
        });

        const isStale = () => abort.signal.aborted || this.activeKey !== key || this.activeLoadId !== loadId;

        const loadIntoViewer = async (url, phase) => {
            if (isStale()) return false;
            const isPreview = phase === 'preview';
            this._loadingBudget = !isPreview;
            this._updatePerf({ activeUrl: url, status: phase });

            // Per-load state captured in closure so a later show() call
            // cannot race-write this scene's buffers.
            let positions = null;
            let colors = null;
            let xform = null;           // {cx, cy, cz, flipY, radius}
            let writeOffset = 0;        // next unfilled point slot
            let flushedUpTo = 0;        // slots already committed to GPU
            let flushScheduled = false;
            let firstBlockRendered = false;
            const geomRef = { current: null };   // the geometry we're filling

            const scheduleFlush = () => {
                if (flushScheduled || isStale()) return;
                flushScheduled = true;
                this._pendingFlushRaf = requestAnimationFrame(() => {
                    this._pendingFlushRaf = null;
                    flushScheduled = false;
                    if (isStale()) return;
                    const geo = geomRef.current;
                    if (!geo || geo !== this.pointCloud?.geometry) return;
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
                    flushedUpTo = to;
                    this._applyDrawBudget();
                });
            };

            const onHeader = (header) => {
                if (isStale()) return;
                this._cancelPendingFlush();
                positions = new Float32Array(header.count * 3);
                colors    = new Uint8Array(header.count * 3);
                xform = computeXformFromHeader(header, cfg);
                // Install an empty geometry (drawRange=0); blocks fill it.
                geomRef.current = this._installGeometry(
                    positions, colors, header.count, 0, cfg, xform.radius
                );
                this._totalPoints = header.count;
                this._loadedPoints = 0;
                this._updatePerf({
                    totalPoints: header.count,
                    loadedPoints: 0,
                    visiblePoints: 0,
                    livePointClouds: this.pointCloud ? 1 : 0,
                });
            };

            const onBlock = (header, bytes, byteOffset, blockCount) => {
                if (isStale()) return;
                decodeBlock(bytes, byteOffset, blockCount, positions, colors,
                            writeOffset, header);
                applyTransform(positions, writeOffset, blockCount, xform);
                writeOffset += blockCount;
                this._loadedPoints = writeOffset;
                if (!firstBlockRendered) {
                    firstBlockRendered = true;
                    this.setMessage(isPreview ? '' : 'Refining…');
                }
                scheduleFlush();
            };

            const onProgress = (pct) => {
                if (isStale() || firstBlockRendered) return;
                this.setMessage(isPreview ? `Preview ${pct}%` : `Loading ${pct}%`);
            };

            await this.loader.load(url, {
                signal: abort.signal, onHeader, onBlock, onProgress
            });
            if (isStale()) return false;

            // Commit any residual slots that had not flushed yet.
            if (writeOffset > flushedUpTo) scheduleFlush();
            return true;
        };

        try {
            if (cfg.preview && cfg.preview !== cfg.cloud) {
                const previewOk = await loadIntoViewer(cfg.preview, 'preview');
                if (!previewOk) return;
                await new Promise(resolve => requestAnimationFrame(resolve));
                if (isStale()) return;
                this.setMessage('Refining…');
            }

            const fullOk = await loadIntoViewer(cfg.cloud, 'full');
            if (!fullOk) return;
            await new Promise(resolve => requestAnimationFrame(resolve));
            if (isStale()) return;

            this._loadingBudget = false;
            this._interactionBudget = false;
            this._applyDrawBudget();
            this.setMessage('');
            this._updatePerf({
                loading: false,
                status: 'ready',
                completedLoads: window.__UNIT_DEMO_PERF.completedLoads + 1,
            });
            // Emit a one-shot "ready" event so external controllers (e.g.
            // the settings panel) can sync their UI to the just-loaded
            // scene's values.
            if (this.onSceneReady) this.onSceneReady(key, cfg);
        } catch (err) {
            if (err.name === 'AbortError' || abort.signal.aborted) {
                this._cancelPendingFlush();
                this._updatePerf({
                    abortedLoads: window.__UNIT_DEMO_PERF.abortedLoads + 1,
                    status: this.activeLoadId === loadId ? 'aborted' : window.__UNIT_DEMO_PERF.status,
                    loading: this.activeLoadId === loadId ? false : window.__UNIT_DEMO_PERF.loading,
                });
                return;
            }
            console.error('Error loading point cloud:', err);
            this.setMessage('Failed to load point cloud');
            this._updatePerf({ loading: false, status: 'error' });
        } finally {
            if (this.activeLoadId === loadId) {
                this.activeAbort = null;
            }
        }
    }

    // Runtime setters used by the view controls panel.  All of them no-op
    // gracefully when there is no point cloud loaded yet (the panel can
    // still write values; the next scene load will pick them up via cfg).
    setPointSize(size) {
        if (!this.pointCloud) return;
        this.pointCloud.material.size = size;
        this.raycaster.params.Points.threshold = size * 1.5;
    }

    setSamplingRate(rate) {
        this._baseSamplingRate = Math.max(0, Math.min(1, rate));
        this._applyDrawBudget();
    }

    setBrightness(b) {
        if (!this.pointCloud) return;
        const shader = this.pointCloud.material.userData?.shader;
        if (shader && shader.uniforms.uBrightness) {
            shader.uniforms.uBrightness.value = b;
        }
    }

    setBackground(color) {
        this.scene.background = new THREE.Color(color);
    }

    setRotation(rotDegrees) {
        if (!this.pointCloud) return;
        const d2r = Math.PI / 180;
        this.pointCloud.rotation.set(
            (rotDegrees.x || 0) * d2r,
            (rotDegrees.y || 0) * d2r,
            (rotDegrees.z || 0) * d2r
        );
    }

    // Move the camera without re-installing the scene.  `cam` is a unit
    // offset multiplied by the scene's bounding-sphere radius, same as
    // the one baked into DEMO_CONFIGS.  The orbit target is reset to the
    // origin so spin-around works as expected after the snap.
    setCameraOffset(cam) {
        const r = this._sceneRadius || 1;
        this.camera.position.set(r * cam.x, r * cam.y, r * cam.z);
        this.controls.target.set(0, 0, 0);
        this.controls.update();
    }

    // Capture the current camera as unit offsets ({x, y, z} divided by
    // radius), suitable for pasting back into DEMO_CONFIGS.camera.
    getCameraOffset() {
        const r = this._sceneRadius || 1;
        return {
            x: this.camera.position.x / r,
            y: this.camera.position.y / r,
            z: this.camera.position.z / r
        };
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

        // Inject a brightness uniform into the fragment shader.  We keep
        // a reference to the compiled Shader object so the controls panel
        // can update `uBrightness` live without a full material rebuild.
        // Values > 1 are allowed — they map to over-exposure, which is
        // exactly what under-lit scans need to read well on a white page.
        //
        // Anchor the injection at `#include <tonemapping_fragment>` rather
        // than `#include <output_fragment>`: r128's points_frag writes
        // `gl_FragColor` inline (no output_fragment chunk exists), so the
        // older anchor silently no-ops.  Splicing in right before the
        // tonemapping pass also means the multiplier is applied in linear
        // space, which is the correct order for exposure-style scaling.
        const initialBrightness = (cfg.brightness != null) ? cfg.brightness : 1.0;
        material.userData.uBrightness = { value: initialBrightness };
        material.onBeforeCompile = (shader) => {
            shader.uniforms.uBrightness = material.userData.uBrightness;
            shader.fragmentShader = shader.fragmentShader
                .replace(
                    'void main() {',
                    'uniform float uBrightness;\nvoid main() {'
                )
                .replace(
                    '#include <tonemapping_fragment>',
                    'gl_FragColor.rgb *= uBrightness;\n\t#include <tonemapping_fragment>'
                );
            material.userData.shader = shader;
        };

        this.pointCloud = new THREE.Points(geometry, material);

        // Apply scene rotation (straightens crooked captures).  Stored in
        // degrees in the config for human-readability; Three.js wants rad.
        const rot = cfg.rotation || { x: 0, y: 0, z: 0 };
        const d2r = Math.PI / 180;
        this.pointCloud.rotation.set(
            (rot.x || 0) * d2r, (rot.y || 0) * d2r, (rot.z || 0) * d2r
        );

        // Points' raycast() iterates the full positions buffer regardless
        // of drawRange, which means measurements snap to hidden points
        // when sampling < 100%.  Wrap it to respect drawRange.count.
        const originalRaycast = this.pointCloud.raycast.bind(this.pointCloud);
        this.pointCloud.raycast = (raycaster, intersects) => {
            const drawCount = geometry.drawRange.count;
            if (drawCount === Infinity || drawCount >= totalCount) {
                return originalRaycast(raycaster, intersects);
            }
            const before = intersects.length;
            originalRaycast(raycaster, intersects);
            // Filter out hits beyond drawRange — original is out-of-order
            // (by distance), so we keep only intersections whose index is
            // inside the drawn prefix.
            for (let i = intersects.length - 1; i >= before; i--) {
                if (intersects[i].index >= drawCount) intersects.splice(i, 1);
            }
        };

        this.scene.add(this.pointCloud);

        // Apply background override if provided (default: white).
        if (cfg.background) {
            this.scene.background = new THREE.Color(cfg.background);
        }

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

    _installRenderVisibilityGate() {
        this._inViewport = true;
        this._docVisible = document.visibilityState === 'visible';
        this._rafId = null;

        const sync = () => this._syncRenderLoop();
        document.addEventListener('visibilitychange', () => {
            this._docVisible = document.visibilityState === 'visible';
            sync();
        });

        if ('IntersectionObserver' in window) {
            this._inViewport = false;
            this._renderObserver = new IntersectionObserver((entries) => {
                this._inViewport = entries.some(e => e.isIntersecting);
                sync();
            }, { threshold: 0.01 });
            this._renderObserver.observe(this.container);
        }

        sync();
    }

    _syncRenderLoop() {
        const shouldRender = this._inViewport && this._docVisible;
        if (shouldRender && !this._rafId) {
            this.animate();
        } else if (!shouldRender && this._rafId) {
            cancelAnimationFrame(this._rafId);
            this._rafId = null;
        }
    }

    animate() {
        if (!this._inViewport || !this._docVisible) {
            this._rafId = null;
            return;
        }
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
        this._rafId = requestAnimationFrame(() => this.animate());
    }
}


// ========================================
// ViewerControls — MeshLab-style settings panel
//
// Owns the UI for tweaking point size, sampling rate, brightness,
// background, scene rotation, and the initial camera offset.  All
// values are persisted per-scene in localStorage so a user's tweaks
// survive reloads.  The "Copy config" button dumps the current state
// as a JSON block that can be pasted straight back into DEMO_CONFIGS.
// ========================================
class ViewerControls {
    constructor(viewer) {
        this.viewer = viewer;
        this.currentKey = null;
        this.currentCfg = null;
        this.overrides = this._loadAll();

        this.panel  = document.getElementById('demo-controls-panel');
        this.toggle = document.getElementById('controls-toggle-btn');

        // Bail silently if the panel markup isn't present (e.g. on an old
        // cached protected.html); the viewer still runs, just without the
        // settings UI.
        if (!this.panel || !this.toggle) return;

        // Gate the settings UI behind a ?setting query param.  Regular
        // visitors never see the gear icon; admins tweaking initial views
        // append ?setting (or ?setting=1) to the URL to unlock the panel.
        if (!new URLSearchParams(window.location.search).has('setting')) {
            this.toggle.style.display = 'none';
            return;
        }

        this._bindInputs();
        this._bindButtons();

        // Keep the Pitch/Yaw/Roll sliders in lockstep with mouse-drag
        // rotation: whenever the viewer rotates the cloud via drag, we
        // write the new degrees back into the inputs, their labels, and
        // localStorage so Copy Config and the next page load both see
        // the drag-imparted orientation.
        this.viewer.onRotationChanged = (rot) => this._onViewerRotated(rot);

        this.toggle.addEventListener('click', () => this._togglePanel());
        const closeBtn = document.getElementById('controls-close');
        if (closeBtn) closeBtn.addEventListener('click', () => this._togglePanel(false));
    }

    _onViewerRotated(rot) {
        if (!this.currentCfg) return;
        const set = (id, valId, value) => {
            const el = this._$(id); if (el) el.value = value;
            const lab = this._$(valId); if (lab) lab.textContent = value + '°';
        };
        set('ctrl-rot-x', 'ctrl-rot-x-val', rot.x);
        set('ctrl-rot-y', 'ctrl-rot-y-val', rot.y);
        set('ctrl-rot-z', 'ctrl-rot-z-val', rot.z);
        this._persist({ rotation: { x: rot.x, y: rot.y, z: rot.z } });
    }

    // Called by initDemo before `viewer.show()`.  Merges baked config
    // with any saved per-scene overrides and returns the effective cfg
    // that the viewer should render with.  Also refreshes the panel so
    // its inputs reflect the resolved values.
    bind(key, cfg) {
        this.currentKey = key;
        const effective = this._merge(cfg, this.overrides[key]);
        this.currentCfg = effective;
        this._syncInputs(effective);
        return effective;
    }

    _merge(base, override) {
        if (!override) return { ...base };
        const merged = { ...base, ...override };
        // Deep-merge the two nested objects so partial overrides work.
        merged.camera   = { ...(base.camera   || {}), ...(override.camera   || {}) };
        merged.rotation = { ...(base.rotation || {}), ...(override.rotation || {}) };
        return merged;
    }

    _loadAll() {
        try {
            const raw = localStorage.getItem(CONTROLS_STORAGE_KEY);
            return raw ? JSON.parse(raw) : {};
        } catch { return {}; }
    }

    _saveAll() {
        try { localStorage.setItem(CONTROLS_STORAGE_KEY, JSON.stringify(this.overrides)); }
        catch {}
    }

    // Record a per-scene tweak and persist.  The override layers on top
    // of the baked cfg so future page loads of that scene re-apply it.
    _persist(patch) {
        if (!this.currentKey) return;
        const prev = this.overrides[this.currentKey] || {};
        this.overrides[this.currentKey] = { ...prev, ...patch };
        this._saveAll();
        // Also update the live effective cfg so getCurrentConfig() is
        // coherent for the "Copy config" button.
        Object.assign(this.currentCfg, patch);
    }

    _togglePanel(force) {
        const want = (force != null) ? force : !this.panel.classList.contains('open');
        this.panel.classList.toggle('open', want);
        this.toggle.classList.toggle('active', want);
    }

    _$(id) { return document.getElementById(id); }

    _bindInputs() {
        const bindRange = (id, labelId, onChange, formatter) => {
            const el = this._$(id);
            const lab = this._$(labelId);
            if (!el) return;
            el.addEventListener('input', () => {
                const v = parseFloat(el.value);
                if (lab) lab.textContent = formatter(v);
                onChange(v);
            });
        };

        bindRange('ctrl-pointsize', 'ctrl-pointsize-val', (v) => {
            this.viewer.setPointSize(v);
            this._persist({ pointSize: v });
        }, (v) => v.toFixed(6));

        bindRange('ctrl-sampling', 'ctrl-sampling-val', (v) => {
            this.viewer.setSamplingRate(v / 100);
            this._persist({ samplingRate: v / 100 });
        }, (v) => Math.round(v) + '%');

        bindRange('ctrl-brightness', 'ctrl-brightness-val', (v) => {
            this.viewer.setBrightness(v);
            this._persist({ brightness: v });
        }, (v) => v.toFixed(2) + '×');

        const bindRot = (axis) => {
            const el = this._$(`ctrl-rot-${axis}`);
            const lab = this._$(`ctrl-rot-${axis}-val`);
            if (!el) return;
            el.addEventListener('input', () => {
                const v = parseFloat(el.value);
                if (lab) lab.textContent = v + '°';
                const rot = { ...(this.currentCfg?.rotation || { x: 0, y: 0, z: 0 }) };
                rot[axis] = v;
                this.viewer.setRotation(rot);
                this._persist({ rotation: rot });
            });
        };
        bindRot('x'); bindRot('y'); bindRot('z');

        const bgRow = this._$('ctrl-bg-swatches');
        if (bgRow) {
            bgRow.querySelectorAll('.bg-swatch').forEach(btn => {
                btn.addEventListener('click', () => {
                    const bg = btn.dataset.bg;
                    bgRow.querySelectorAll('.bg-swatch').forEach(b => b.classList.toggle('selected', b === btn));
                    this.viewer.setBackground(bg);
                    this._persist({ background: bg });
                });
            });
        }
    }

    _bindButtons() {
        const reset = this._$('ctrl-reset');
        if (reset) reset.addEventListener('click', () => this._resetCurrent());

        const capture = this._$('ctrl-capture-cam');
        if (capture) capture.addEventListener('click', () => this._saveInitialView(capture));

        const copy = this._$('ctrl-copy');
        if (copy) copy.addEventListener('click', () => this._copyConfig(copy));
    }

    // Snapshot the effective config for every scene and POST it to the
    // Worker.  The ?setting=<secret> query param doubles as the admin
    // key: it both unlocks the settings panel and authenticates the
    // write.  A scene's effective config is DEMO_CONFIGS[key] deep-
    // merged with this.overrides[key], plus the live camera for the
    // currently-viewed scene (which the sliders don't touch).
    async _saveInitialView(btn) {
        const secret = new URLSearchParams(window.location.search).get('setting');
        if (!secret) {
            // Panel wouldn't be visible without ?setting, but guard anyway.
            btn.textContent = 'Admin mode required';
            setTimeout(() => { btn.textContent = 'Save as Initial View'; }, 1800);
            return;
        }
        // Fold the live camera into the current scene's override slot so
        // it's picked up by the payload builder below.
        if (this.currentKey && this.viewer && this.viewer.pointCloud) {
            const cam = this.viewer.getCameraOffset();
            const prev = this.overrides[this.currentKey] || {};
            this.overrides[this.currentKey] = {
                ...prev,
                camera: { ...(prev.camera || {}), ...cam },
            };
            this._saveAll();
            Object.assign(this.currentCfg.camera, cam);
        }
        const payload = {};
        for (const [key, baked] of Object.entries(DEMO_CONFIGS)) {
            const override = this.overrides[key] || {};
            payload[key] = {
                pointSize:    override.pointSize    != null ? override.pointSize    : baked.pointSize,
                samplingRate: override.samplingRate != null ? override.samplingRate : baked.samplingRate,
                brightness:   override.brightness   != null ? override.brightness   : baked.brightness,
                background:   override.background   != null ? override.background   : baked.background,
                camera:   { ...(baked.camera   || {}), ...(override.camera   || {}) },
                rotation: { ...(baked.rotation || {}), ...(override.rotation || {}) },
            };
        }
        btn.textContent = 'Saving…';
        btn.disabled = true;
        let status = 0;
        try {
            const resp = await fetch(CONFIG_WORKER_URL + '/config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'X-Admin-Key': secret },
                body: JSON.stringify(payload),
            });
            status = resp.status;
            if (!resp.ok) throw new Error('HTTP ' + status);
            btn.textContent = 'Initial view saved ✓';
        } catch (e) {
            btn.textContent = status === 401 ? 'Bad secret' : 'Save failed';
        }
        btn.disabled = false;
        setTimeout(() => { btn.textContent = 'Save as Initial View'; }, 1800);
    }

    _syncInputs(cfg) {
        const set = (id, valId, value, formatter) => {
            const el = this._$(id); if (!el) return;
            el.value = value;
            const lab = this._$(valId);
            if (lab && formatter) lab.textContent = formatter(value);
        };
        set('ctrl-pointsize', 'ctrl-pointsize-val', cfg.pointSize, (v) => (+v).toFixed(6));
        const samp = (cfg.samplingRate != null ? cfg.samplingRate : 1) * 100;
        set('ctrl-sampling', 'ctrl-sampling-val', samp, (v) => Math.round(v) + '%');
        const bright = (cfg.brightness != null ? cfg.brightness : 1);
        set('ctrl-brightness', 'ctrl-brightness-val', bright, (v) => (+v).toFixed(2) + '×');
        const rot = cfg.rotation || { x: 0, y: 0, z: 0 };
        set('ctrl-rot-x', 'ctrl-rot-x-val', rot.x || 0, (v) => v + '°');
        set('ctrl-rot-y', 'ctrl-rot-y-val', rot.y || 0, (v) => v + '°');
        set('ctrl-rot-z', 'ctrl-rot-z-val', rot.z || 0, (v) => v + '°');

        const bgRow = this._$('ctrl-bg-swatches');
        if (bgRow) {
            const current = (cfg.background || '#ffffff').toLowerCase();
            bgRow.querySelectorAll('.bg-swatch').forEach(b => {
                b.classList.toggle('selected', b.dataset.bg.toLowerCase() === current);
            });
        }
    }

    _resetCurrent() {
        if (!this.currentKey) return;
        delete this.overrides[this.currentKey];
        this._saveAll();
        const baked = DEMO_CONFIGS[this.currentKey];
        if (!baked) return;
        this.currentCfg = { ...baked,
            camera:   { ...baked.camera },
            rotation: { ...baked.rotation }
        };
        this._syncInputs(this.currentCfg);
        // Push all values back into the viewer so the scene snaps to its
        // baked defaults in one action.
        this.viewer.setPointSize(baked.pointSize);
        this.viewer.setSamplingRate(baked.samplingRate != null ? baked.samplingRate : 1);
        this.viewer.setBrightness(baked.brightness != null ? baked.brightness : 1);
        this.viewer.setBackground(baked.background || '#ffffff');
        this.viewer.setRotation(baked.rotation || { x: 0, y: 0, z: 0 });
        this.viewer.setCameraOffset(baked.camera);
    }

    _copyConfig(btn) {
        if (!this.currentCfg || !this.currentKey) return;
        const cfg = this.currentCfg;
        const cam = this.viewer.getCameraOffset();
        const lines = [
            `${this.currentKey}: {`,
            `    title: ${JSON.stringify(cfg.title || this.currentKey)},`,
            `    cloud: ${JSON.stringify(cfg.cloud)},`,
            cfg.preview ? `    preview: ${JSON.stringify(cfg.preview)},` : null,
            `    pointSize: ${(+cfg.pointSize).toFixed(4)},`,
            `    flipY: ${!!cfg.flipY},`,
            `    camera: { x: ${cam.x.toFixed(3)}, y: ${cam.y.toFixed(3)}, z: ${cam.z.toFixed(3)} },`,
            `    samplingRate: ${(+(cfg.samplingRate ?? 1)).toFixed(3)},`,
            `    brightness: ${(+(cfg.brightness ?? 1)).toFixed(2)},`,
            `    background: ${JSON.stringify(cfg.background || '#ffffff')},`,
            `    rotation: { x: ${+(cfg.rotation?.x || 0)}, y: ${+(cfg.rotation?.y || 0)}, z: ${+(cfg.rotation?.z || 0)} }`,
            `}`
        ].filter(Boolean);
        const text = lines.join('\n');
        navigator.clipboard.writeText(text).then(() => {
            btn.textContent = 'Copied ✓';
            setTimeout(() => { btn.textContent = 'Copy Config'; }, 1500);
        }).catch(() => {
            // Clipboard API can fail in non-secure contexts; show inline.
            const box = this._$('ctrl-copy-output');
            if (box) { box.textContent = text; box.style.display = 'block'; }
        });
    }
}
