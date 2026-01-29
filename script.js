/* ========================================
   D4RT Style - JavaScript
   4D Point Cloud Viewer with Splat Loading
   ======================================== */

document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initViewer();
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
    
    // Only select sample images that have data-scene (3D viewer samples)
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
            // Update selected state
            videoThumbs.forEach(t => t.classList.remove('selected'));
            thumb.classList.add('selected');
            
            // Switch video source
            const videoSrc = thumb.dataset.video;
            if (videoSrc) {
                galleryVideo.src = videoSrc;
                galleryVideo.load();
                galleryVideo.play().catch(() => {}); // Auto-play if allowed
            }
        });
    });
}

// ========================================
// 4D Point Cloud Viewer (Three.js)
// Supports .splat file loading
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
        this.lastFrameTime = 0;
        this.frameInterval = 33;
        this.rotationAngle = 0;
        
        // Splat data
        this.splatData = null;
        this.splatCenter = [0, 0, 0];
        this.splatScale = 1;
        this.currentScene = 'scene1';
        
        // UI elements
        this.playPauseBtn = document.getElementById('play-pause');
        this.frameSlider = document.getElementById('frame-slider');
        this.frameCounter = document.getElementById('frame-counter');
        
        this.init();
        this.loadSplatFile('assets/scene1.splat', 'scene1');
        this.setupControls();
        this.animate();
        
        this.controls.addEventListener('start', () => {
            this.hideMessage();
        });
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
            antialias: false,  // Disable for better performance
            powerPreference: 'high-performance'
        });
        this.renderer.setSize(width, height);
        this.renderer.setPixelRatio(1);  // Lower pixel ratio for performance
        
        this.controls = new THREE.OrbitControls(this.camera, this.canvas);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.minDistance = 0.1;
        this.controls.maxDistance = 100;
        this.controls.autoRotate = false;
        
        window.addEventListener('resize', () => this.onResize());
    }
    
    async loadSplatFile(url, sceneName = 'scene1') {
        this.showMessage('Loading 0%');
        this.currentScene = sceneName;
        
        try {
            const response = await fetch(url);
            if (!response.ok) {
                throw new Error('Failed to load splat file');
            }
            
            const contentLength = response.headers.get('content-length');
            const total = contentLength ? parseInt(contentLength, 10) : 0;
            const reader = response.body.getReader();
            
            const chunks = [];
            let received = 0;
            let lastUpdate = 0;
            
            // Stream and progressively display
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                chunks.push(value);
                received += value.length;
                
                // Update progress
                if (total > 0) {
                    const percent = Math.round((received / total) * 100);
                    this.showMessage(`Loading ${percent}%`);
                }
                
                // Update display every 10% or 2MB
                if (received - lastUpdate > Math.max(total * 0.1, 2000000)) {
                    lastUpdate = received;
                    const partialBuffer = this.concatArrayBuffers(chunks);
                    this.parseSplatDataPartial(partialBuffer);
                    this.createPointCloudFromSplat();
                }
            }
            
            // Final full render
            const buffer = this.concatArrayBuffers(chunks);
            this.parseSplatData(buffer);
            this.createPointCloudFromSplat();
            this.hideMessage();
            
        } catch (error) {
            console.error('Error loading splat:', error);
            this.createDemoScene();
            this.hideMessage();
        }
    }
    
    concatArrayBuffers(arrays) {
        const totalLength = arrays.reduce((sum, arr) => sum + arr.length, 0);
        const result = new Uint8Array(totalLength);
        let offset = 0;
        for (const arr of arrays) {
            result.set(arr, offset);
            offset += arr.length;
        }
        return result.buffer;
    }
    
    parseSplatDataPartial(buffer) {
        // Parse as much as we can from partial data
        const view = new DataView(buffer);
        if (buffer.byteLength < 4) return;
        
        const totalPoints = view.getUint32(0, true);
        const bytesPerPoint = 10; // 6 bytes position + 4 bytes color
        const availableBytes = buffer.byteLength - 4;
        const availablePoints = Math.floor(availableBytes / bytesPerPoint);
        const pointCount = Math.min(totalPoints, availablePoints);
        
        if (pointCount < 1000) return; // Wait for more data
        
        let offset = 4;
        const positions = new Float32Array(pointCount * 3);
        const colors = new Float32Array(pointCount * 3);
        
        for (let i = 0; i < pointCount; i++) {
            positions[i * 3] = this.float16ToFloat32(view.getUint16(offset, true));
            positions[i * 3 + 1] = this.float16ToFloat32(view.getUint16(offset + 2, true));
            positions[i * 3 + 2] = this.float16ToFloat32(view.getUint16(offset + 4, true));
            offset += 6;
            
            colors[i * 3] = view.getUint8(offset) / 255;
            colors[i * 3 + 1] = view.getUint8(offset + 1) / 255;
            colors[i * 3 + 2] = view.getUint8(offset + 2) / 255;
            offset += 4;
        }
        
        this.splatData = { positions, colors, count: pointCount };
    }
    
    parseSplatData(buffer) {
        const view = new DataView(buffer);
        let offset = 0;
        
        // Read point count (first 4 bytes as uint32 little-endian)
        const pointCount = view.getUint32(offset, true);
        offset = 4;
        
        console.log(`Loading ${pointCount.toLocaleString()} points`);
        
        // Read point data: [x_f16, y_f16, z_f16, r_u8, g_u8, b_u8, a_u8] = 10 bytes per point
        const positions = new Float32Array(pointCount * 3);
        const colors = new Float32Array(pointCount * 3);
        
        for (let i = 0; i < pointCount; i++) {
            // Read float16 positions and convert to float32
            const x = this.float16ToFloat32(view.getUint16(offset, true));
            const y = this.float16ToFloat32(view.getUint16(offset + 2, true));
            const z = this.float16ToFloat32(view.getUint16(offset + 4, true));
            offset += 6;
            
            // Positions are already normalized in [-10, 10] range
            positions[i * 3] = x;
            positions[i * 3 + 1] = y;
            positions[i * 3 + 2] = z;
            
            // Read colors (RGBA, but we only use RGB)
            colors[i * 3] = view.getUint8(offset) / 255;
            colors[i * 3 + 1] = view.getUint8(offset + 1) / 255;
            colors[i * 3 + 2] = view.getUint8(offset + 2) / 255;
            offset += 4;
        }
        
        this.splatData = { positions, colors, count: pointCount };
    }
    
    float16ToFloat32(h) {
        const s = (h & 0x8000) >> 15;
        const e = (h & 0x7C00) >> 10;
        const f = h & 0x03FF;
        
        if (e === 0) {
            return (s ? -1 : 1) * Math.pow(2, -14) * (f / Math.pow(2, 10));
        } else if (e === 0x1F) {
            return f ? NaN : ((s ? -1 : 1) * Infinity);
        }
        
        return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / Math.pow(2, 10));
    }
    
    createPointCloudFromSplat() {
        if (!this.splatData) return;
        
        if (this.pointCloud) {
            this.scene.remove(this.pointCloud);
            this.pointCloud.geometry.dispose();
            this.pointCloud.material.dispose();
        }
        
        // Flip Y axis to correct orientation
        const positions = this.splatData.positions.slice();
        for (let i = 0; i < positions.length; i += 3) {
            positions[i + 1] = -positions[i + 1]; // Flip Y
        }
        
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(this.splatData.colors, 3));
        
        // Center the point cloud
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
        
        // Adjust camera to fit based on scene
        geometry.computeBoundingSphere();
        const radius = geometry.boundingSphere.radius;
        
        // Scene-specific camera positions
        const cameraSettings = {
            'scene1': { x: -0.4, y: 0.5, z: -1.7 },   // Default front view
            'scene2': { x: -0.4, y: 0.5, z: -1.7 },   // Default front view
            'scene3': { x: -0.4, y: 0.15, z: -1.7 },  // Lower Y for less top-down
            'scene4': { x: -0.4, y: 0.15, z: -1.7 },  // Lower Y for less top-down
            'scene5': { x: -0.3, y: 0.3, z: -1.2 }    // Closer zoom
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
    
    createDemoScene() {
        // Fallback demo scene if splat loading fails
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
            this.playPauseBtn.addEventListener('click', () => {
                this.togglePlayback();
            });
        }
        
        if (this.frameSlider) {
            this.frameSlider.max = this.totalFrames - 1;
            
            this.frameSlider.addEventListener('input', () => {
                this.currentFrame = parseInt(this.frameSlider.value);
                this.rotationAngle = (this.currentFrame / this.totalFrames) * Math.PI * 2;
                this.updateUI();
            });
            
            this.frameSlider.addEventListener('mousedown', () => {
                this.wasPlaying = this.isPlaying;
                this.isPlaying = false;
                this.updatePlayPauseButton();
            });
            
            this.frameSlider.addEventListener('mouseup', () => {
                if (this.wasPlaying) {
                    this.isPlaying = true;
                    this.updatePlayPauseButton();
                }
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
        
        // Static point cloud - no auto rotation
        // User can freely rotate with mouse/touch controls
        
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }
}

function initViewer() {
    window.viewer4D = new Viewer4D('canvas');
}
