/* ========================================
   D4RT Style - JavaScript
   4D Point Cloud Viewer with Splat Loading
   Performance Optimized Version
   ======================================== */

document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initViewerLazy(); // Use lazy loading
    initSampleSelector();
    initVideoGallery();
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
            const targetSection = document.getElementById(sectionId);
            if (targetSection) {
                targetSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    });
}

// ========================================
// Sample Selector (3D Viewer)
// ========================================
function initSampleSelector() {
    const modeBtns = document.querySelectorAll('.sample-type-button');
    modeBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            modeBtns.forEach(b => b.classList.remove('selected'));
            btn.classList.add('selected');
            if (window.viewer4D) {
                window.viewer4D.switchMode(btn.dataset.mode);
            }
        });
    });
    
    const sampleImgs = document.querySelectorAll('.sample-img[data-scene]');
    sampleImgs.forEach(img => {
        img.addEventListener('click', () => {
            sampleImgs.forEach(i => i.classList.remove('selected'));
            img.classList.add('selected');
            if (window.viewer4D) {
                window.viewer4D.loadScene(img.dataset.scene);
            }
        });
    });
}

// ========================================
// Video Gallery
// ========================================
function initVideoGallery() {
    const videoThumbs = document.querySelectorAll('.video-thumbs .sample-img');
    const galleryVideo = document.getElementById('gallery-video');
    
    if (!galleryVideo || videoThumbs.length === 0) return;
    
    videoThumbs.forEach(thumb => {
        thumb.addEventListener('click', () => {
            videoThumbs.forEach(t => t.classList.remove('selected'));
            thumb.classList.add('selected');
            
            const videoSrc = thumb.dataset.video;
            if (videoSrc) {
                galleryVideo.src = videoSrc;
                galleryVideo.load();
                galleryVideo.play().catch(() => {});
            }
        });
    });
}

// ========================================
// Lazy Loading with Intersection Observer
// ========================================
function initViewerLazy() {
    const canvas = document.getElementById('canvas');
    if (!canvas) return;
    
    const container = canvas.parentElement.parentElement; // visualization-container
    
    // Create observer for lazy loading
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting && !window.viewer4D) {
                // Initialize viewer when section comes into view
                window.viewer4D = new Viewer4D('canvas');
                observer.disconnect();
            }
        });
    }, {
        rootMargin: '200px' // Start loading 200px before visible
    });
    
    observer.observe(container);
    
    // Also init immediately if already visible (fallback)
    const rect = container.getBoundingClientRect();
    if (rect.top < window.innerHeight + 200) {
        window.viewer4D = new Viewer4D('canvas');
        observer.disconnect();
    }
}

// ========================================
// Web Worker for Parsing (Inline Blob)
// ========================================
const parserWorkerCode = `
    // Float16 lookup table
    const float16Table = new Float32Array(65536);
    for (let i = 0; i < 65536; i++) {
        const s = (i & 0x8000) >> 15;
        const e = (i & 0x7C00) >> 10;
        const f = i & 0x03FF;
        if (e === 0) {
            float16Table[i] = (s ? -1 : 1) * (f / 1024) * (1 / 16384);
        } else if (e === 31) {
            float16Table[i] = f ? NaN : ((s ? -1 : 1) * Infinity);
        } else {
            float16Table[i] = (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
        }
    }
    
    self.onmessage = function(e) {
        const buffer = e.data.buffer;
        const view = new DataView(buffer);
        let offset = 0;
        
        const pointCount = view.getUint32(offset, true);
        offset = 4;
        
        const positions = new Float32Array(pointCount * 3);
        const colors = new Float32Array(pointCount * 3);
        
        for (let i = 0; i < pointCount; i++) {
            const x = float16Table[view.getUint16(offset, true)];
            const y = float16Table[view.getUint16(offset + 2, true)];
            const z = float16Table[view.getUint16(offset + 4, true)];
            offset += 6;
            
            positions[i * 3] = x;
            positions[i * 3 + 1] = -y; // Flip Y
            positions[i * 3 + 2] = z;
            
            colors[i * 3] = view.getUint8(offset) / 255;
            colors[i * 3 + 1] = view.getUint8(offset + 1) / 255;
            colors[i * 3 + 2] = view.getUint8(offset + 2) / 255;
            offset += 4;
        }
        
        self.postMessage({
            positions: positions,
            colors: colors,
            count: pointCount
        }, [positions.buffer, colors.buffer]);
    };
`;

// ========================================
// 4D Point Cloud Viewer (Three.js)
// ========================================
class Viewer4D {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) return;
        
        this.container = this.canvas.parentElement;
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.pointCloud = null;
        this.message = document.getElementById('message');
        
        // Animation state
        this.isPlaying = true;
        this.currentFrame = 0;
        this.totalFrames = 60;
        this.rotationAngle = 0;
        
        // Splat data
        this.splatData = null;
        this.currentScene = 'scene1';
        this.abortController = null;
        this.sceneCache = {};
        this.pointCloudCache = {};
        this.userInteractedDuringLoad = false;
        this.isLoading = false;
        this.preloadQueue = [];
        this.isPreloading = false;
        
        // Web Worker
        this.parserWorker = null;
        this.initWorker();
        
        // UI elements
        this.playPauseBtn = document.getElementById('play-pause');
        this.frameSlider = document.getElementById('frame-slider');
        this.frameCounter = document.getElementById('frame-counter');
        
        this.init();
        // Load preview first for fast initial display, then full version
        this.loadInitialScene();
        this.setupControls();
        this.animate();
        
        this.controls.addEventListener('start', () => {
            if (this.isLoading) {
                this.userInteractedDuringLoad = true;
            }
            this.hideMessage();
        });
    }
    
    initWorker() {
        try {
            const blob = new Blob([parserWorkerCode], { type: 'application/javascript' });
            this.parserWorker = new Worker(URL.createObjectURL(blob));
        } catch (e) {
            console.warn('Web Worker not supported, using main thread parsing');
            this.parserWorker = null;
        }
    }
    
    init() {
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0xffffff);
        
        this.camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 1000);
        this.camera.position.set(3, 2, 3);
        
        this.renderer = new THREE.WebGLRenderer({ 
            canvas: this.canvas,
            antialias: false,
            powerPreference: 'high-performance'
        });
        this.renderer.setSize(width, height);
        this.renderer.setPixelRatio(1);
        
        this.controls = new THREE.OrbitControls(this.camera, this.canvas);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.minDistance = 0.1;
        this.controls.maxDistance = 100;
        this.controls.autoRotate = false;
        
        window.addEventListener('resize', () => this.onResize());
    }
    
    // Load initial scene with preview-first strategy
    async loadInitialScene() {
        this.currentScene = 'scene1';
        this.showMessage('Loading...');
        this.isLoading = true;
        
        try {
            // Step 1: Load preview (small file, fast)
            const previewResponse = await fetch('assets/scene1_preview.splat');
            if (previewResponse.ok) {
                const previewBuffer = await previewResponse.arrayBuffer();
                this.splatData = await this.parseWithWorker(previewBuffer);
                this.createPointCloudFromSplat(true);
                this.hideMessage();
            }
            
            // Step 2: Load full version in background (don't show loading UI)
            this.isLoading = false;
            const fullResponse = await fetch('assets/scene1.splat');
            if (fullResponse.ok) {
                const fullBuffer = await fullResponse.arrayBuffer();
                const fullData = await this.parseWithWorker(fullBuffer);
                
                // Replace with full version (preserve camera)
                this.splatData = fullData;
                this.sceneCache['scene1'] = fullData;
                delete this.pointCloudCache['scene1']; // Clear old cache
                this.createPointCloudFromSplat(false); // Keep camera position
                console.log('Upgraded to full quality:', fullData.count.toLocaleString(), 'points');
            }
            
            // Step 3: Start preloading other scenes
            this.schedulePreload();
            
        } catch (error) {
            console.error('Initial load error:', error);
            this.isLoading = false;
            this.hideMessage();
            this.createDemoScene();
        }
    }
    
    // Schedule preloading of other scenes
    schedulePreload() {
        // Preload order: smaller files first
        const preloadOrder = ['scene3', 'scene5', 'scene4', 'scene2'];
        
        const scheduleNext = () => {
            if (preloadOrder.length === 0) return;
            
            const nextScene = preloadOrder.shift();
            if (!this.sceneCache[nextScene] && !this.pointCloudCache[nextScene]) {
                this.preloadScene(nextScene).then(() => {
                    // Use requestIdleCallback for next preload
                    if ('requestIdleCallback' in window) {
                        requestIdleCallback(() => scheduleNext(), { timeout: 5000 });
                    } else {
                        setTimeout(scheduleNext, 1000);
                    }
                });
            } else {
                scheduleNext();
            }
        };
        
        // Start preloading after initial scene loads
        setTimeout(() => {
            if ('requestIdleCallback' in window) {
                requestIdleCallback(() => scheduleNext(), { timeout: 3000 });
            } else {
                setTimeout(scheduleNext, 2000);
            }
        }, 1000);
    }
    
    // Preload a scene in the background
    async preloadScene(sceneName) {
        if (this.sceneCache[sceneName] || this.isLoading) return;
        
        const sceneFiles = {
            'scene1': 'assets/scene1.splat',
            'scene2': 'assets/scene2.splat',
            'scene3': 'assets/scene3.splat',
            'scene4': 'assets/scene4.splat',
            'scene5': 'assets/scene5.splat'
        };
        
        const url = sceneFiles[sceneName];
        if (!url) return;
        
        try {
            const response = await fetch(url);
            if (!response.ok) return;
            
            const buffer = await response.arrayBuffer();
            const data = await this.parseWithWorker(buffer);
            
            this.sceneCache[sceneName] = data;
            console.log(`Preloaded ${sceneName}`);
        } catch (e) {
            // Silent fail for preloading
        }
    }
    
    // Parse using Web Worker
    parseWithWorker(buffer) {
        return new Promise((resolve, reject) => {
            if (!this.parserWorker) {
                // Fallback to main thread
                resolve(this.parseSplatDataSync(buffer));
                return;
            }
            
            const handler = (e) => {
                this.parserWorker.removeEventListener('message', handler);
                resolve(e.data);
            };
            
            this.parserWorker.addEventListener('message', handler);
            this.parserWorker.postMessage({ buffer }, [buffer]);
        });
    }
    
    // Synchronous parsing (fallback)
    parseSplatDataSync(buffer) {
        const view = new DataView(buffer);
        let offset = 0;
        
        const pointCount = view.getUint32(offset, true);
        offset = 4;
        
        const positions = new Float32Array(pointCount * 3);
        const colors = new Float32Array(pointCount * 3);
        
        // Build float16 table inline
        const float16Table = new Float32Array(65536);
        for (let i = 0; i < 65536; i++) {
            const s = (i & 0x8000) >> 15;
            const e = (i & 0x7C00) >> 10;
            const f = i & 0x03FF;
            if (e === 0) {
                float16Table[i] = (s ? -1 : 1) * (f / 1024) * (1 / 16384);
            } else if (e === 31) {
                float16Table[i] = f ? NaN : ((s ? -1 : 1) * Infinity);
            } else {
                float16Table[i] = (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
            }
        }
        
        for (let i = 0; i < pointCount; i++) {
            positions[i * 3] = float16Table[view.getUint16(offset, true)];
            positions[i * 3 + 1] = -float16Table[view.getUint16(offset + 2, true)];
            positions[i * 3 + 2] = float16Table[view.getUint16(offset + 4, true)];
            offset += 6;
            
            colors[i * 3] = view.getUint8(offset) / 255;
            colors[i * 3 + 1] = view.getUint8(offset + 1) / 255;
            colors[i * 3 + 2] = view.getUint8(offset + 2) / 255;
            offset += 4;
        }
        
        return { positions, colors, count: pointCount };
    }
    
    async loadSplatFile(url, sceneName = 'scene1') {
        if (this.abortController) {
            this.abortController.abort();
        }
        this.abortController = new AbortController();
        
        this.currentScene = sceneName;
        
        // Check if Three.js object is cached - instant switch
        if (this.pointCloudCache[sceneName]) {
            this.createPointCloudFromSplat(false);
            return;
        }
        
        // Check if data is cached
        if (this.sceneCache[sceneName]) {
            this.splatData = this.sceneCache[sceneName];
            this.createPointCloudFromSplat(false);
            return;
        }
        
        this.showMessage('Loading 0%');
        this.isLoading = true;
        this.userInteractedDuringLoad = false;
        
        try {
            const response = await fetch(url, { signal: this.abortController.signal });
            if (!response.ok) throw new Error('Failed to load');
            
            const contentLength = response.headers.get('content-length');
            const total = contentLength ? parseInt(contentLength, 10) : 0;
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
                    this.showMessage(`Loading ${percent}%`);
                }
            }
            
            // Concat buffers
            const totalLength = chunks.reduce((sum, arr) => sum + arr.length, 0);
            const result = new Uint8Array(totalLength);
            let offset = 0;
            for (const arr of chunks) {
                result.set(arr, offset);
                offset += arr.length;
            }
            
            // Parse with Web Worker
            this.splatData = await this.parseWithWorker(result.buffer);
            console.log(`Loading ${this.splatData.count.toLocaleString()} points`);
            
            const shouldResetCamera = !this.userInteractedDuringLoad;
            this.createPointCloudFromSplat(shouldResetCamera);
            
            // Cache data
            this.sceneCache[sceneName] = {
                positions: this.splatData.positions,
                colors: this.splatData.colors,
                count: this.splatData.count
            };
            
            this.isLoading = false;
            this.hideMessage();
            this.abortController = null;
            
        } catch (error) {
            this.isLoading = false;
            if (error.name === 'AbortError') return;
            console.error('Error loading:', error);
            this.createDemoScene();
            this.hideMessage();
        }
    }
    
    createPointCloudFromSplat(resetCamera = true) {
        if (!this.splatData && !this.pointCloudCache[this.currentScene]) return;
        
        if (this.pointCloud) {
            this.scene.remove(this.pointCloud);
        }
        
        if (this.pointCloudCache[this.currentScene]) {
            this.pointCloud = this.pointCloudCache[this.currentScene];
            this.scene.add(this.pointCloud);
        } else {
            const geometry = new THREE.BufferGeometry();
            geometry.setAttribute('position', new THREE.BufferAttribute(this.splatData.positions, 3));
            geometry.setAttribute('color', new THREE.BufferAttribute(this.splatData.colors, 3));
            
            geometry.computeBoundingBox();
            const center = new THREE.Vector3();
            geometry.boundingBox.getCenter(center);
            geometry.translate(-center.x, -center.y, -center.z);
            
            const material = new THREE.PointsMaterial({
                size: 0.035,
                vertexColors: true,
                sizeAttenuation: true
            });
            
            this.pointCloud = new THREE.Points(geometry, material);
            this.scene.add(this.pointCloud);
            this.pointCloudCache[this.currentScene] = this.pointCloud;
        }
        
        if (resetCamera) {
            this.pointCloud.geometry.computeBoundingSphere();
            const radius = this.pointCloud.geometry.boundingSphere.radius;
            
            const cameraSettings = {
                'scene1': { x: -0.4, y: 0.5, z: -1.7 },
                'scene2': { x: -0.4, y: 0.5, z: -1.7 },
                'scene3': { x: -0.4, y: 0.15, z: -1.7 },
                'scene4': { x: -0.4, y: 0.15, z: -1.7 },
                'scene5': { x: -0.3, y: 0.3, z: -1.2 }
            };
            
            const settings = cameraSettings[this.currentScene] || cameraSettings['scene1'];
            this.camera.position.set(
                radius * settings.x,
                radius * settings.y,
                radius * settings.z
            );
            this.controls.target.set(0, 0, 0);
            this.controls.update();
        }
    }
    
    createDemoScene() {
        const pointCount = 50000;
        const positions = new Float32Array(pointCount * 3);
        const colors = new Float32Array(pointCount * 3);
        
        for (let i = 0; i < pointCount; i++) {
            const i3 = i * 3;
            positions[i3] = (Math.random() - 0.5) * 4;
            positions[i3 + 1] = Math.random() * 2;
            positions[i3 + 2] = (Math.random() - 0.5) * 4;
            
            const shade = 0.4 + Math.random() * 0.3;
            colors[i3] = shade;
            colors[i3 + 1] = shade - 0.05;
            colors[i3 + 2] = shade - 0.1;
        }
        
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        
        const material = new THREE.PointsMaterial({
            size: 0.015,
            vertexColors: true,
            sizeAttenuation: true
        });
        
        this.pointCloud = new THREE.Points(geometry, material);
        this.scene.add(this.pointCloud);
    }
    
    showMessage(text) {
        if (this.message) {
            this.message.textContent = text;
            this.message.style.display = 'flex';
            this.message.style.opacity = '1';
        }
    }
    
    hideMessage() {
        if (this.message && this.message.style.display !== 'none') {
            this.message.style.opacity = '0';
            setTimeout(() => {
                this.message.style.display = 'none';
            }, 300);
        }
    }
    
    setupControls() {
        if (this.playPauseBtn) {
            this.playPauseBtn.addEventListener('click', () => this.togglePlayback());
        }
        
        if (this.frameSlider) {
            this.frameSlider.max = this.totalFrames - 1;
            this.frameSlider.addEventListener('input', () => {
                this.currentFrame = parseInt(this.frameSlider.value);
                this.rotationAngle = (this.currentFrame / this.totalFrames) * Math.PI * 2;
                this.updateUI();
            });
        }
        
        this.updateUI();
    }
    
    togglePlayback() {
        this.isPlaying = !this.isPlaying;
        this.updatePlayPauseButton();
        this.hideMessage();
    }
    
    updatePlayPauseButton() {
        if (this.playPauseBtn) {
            this.playPauseBtn.querySelector('span').textContent = this.isPlaying ? '⏸' : '▶';
        }
    }
    
    updateUI() {
        if (this.frameSlider) {
            this.frameSlider.value = this.currentFrame;
        }
        if (this.frameCounter) {
            this.frameCounter.textContent = `${this.currentFrame} / ${this.totalFrames - 1}`;
        }
    }
    
    switchMode(mode) {
        if (this.pointCloud) {
            this.pointCloud.material.size = mode === 'pointcloud' ? 0.02 : 0.015;
        }
    }
    
    loadScene(sceneName) {
        const sceneFiles = {
            'scene1': 'assets/scene1.splat',
            'scene2': 'assets/scene2.splat',
            'scene3': 'assets/scene3.splat',
            'scene4': 'assets/scene4.splat',
            'scene5': 'assets/scene5.splat'
        };
        
        const file = sceneFiles[sceneName] || `assets/${sceneName}.splat`;
        this.loadSplatFile(file, sceneName);
        
        this.currentFrame = 0;
        this.rotationAngle = 0;
        this.isPlaying = true;
        this.updatePlayPauseButton();
        this.updateUI();
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
