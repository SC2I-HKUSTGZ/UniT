/* ============================================================
   UniT Offline Point Cloud Viewer
   -----------------------------------------------------------
   Standalone MeshLab-style viewer for .ply / .pnt.gz files.
   Loads local files via file input or drag-drop, parses them
   client-side with a zero-copy DataView walker, and renders
   with the same Three.js setup as the online site so the
   parameters match one-to-one.

   Config produced by "Copy Config" is a drop-in snippet for
   the DEMO_CONFIGS map in ../script.js — use that to round-
   trip tweaks from this viewer into the published site.
   ============================================================ */

(() => {
    // --- State -----------------------------------------------------------
    const files = [];              // [{ name, size, data: {positions, colors, ...} }]
    let currentIndex = -1;
    let three = null;              // { scene, camera, renderer, controls, pointCloud, ... }

    const defaultParams = () => ({
        pointSize: 0.05,
        samplingRate: 1.0,
        brightness: 1.0,
        ambient: 0.10,
        opacity: 1.0,
        background: '#ffffff',
        flipY: false,
        rotation: { x: 0, y: 0, z: 0 },
        camera: { x: -0.5, y: 0.3, z: -1.5 },
        fov: 55
    });

    // --- DOM -------------------------------------------------------------
    const el = {
        canvas:        document.getElementById('viewer-canvas'),
        canvasWrap:    document.getElementById('canvas-wrap'),
        hint:          document.getElementById('viewer-hint'),
        status:        document.getElementById('viewer-status'),
        fileInput:     document.getElementById('file-input'),
        clearBtn:      document.getElementById('clear-btn'),
        sceneInfo:     document.getElementById('scene-info'),
        fileList:      document.getElementById('file-list'),

        pointSize:     document.getElementById('ctrl-pointsize'),
        sampling:      document.getElementById('ctrl-sampling'),
        brightness:    document.getElementById('ctrl-brightness'),
        ambient:       document.getElementById('ctrl-ambient'),
        opacity:       document.getElementById('ctrl-opacity'),
        rotX:          document.getElementById('ctrl-rot-x'),
        rotY:          document.getElementById('ctrl-rot-y'),
        rotZ:          document.getElementById('ctrl-rot-z'),
        flipY:         document.getElementById('ctrl-flipy'),
        bgSwatches:    document.getElementById('ctrl-bg-swatches'),
        camX:          document.getElementById('ctrl-cam-x'),
        camY:          document.getElementById('ctrl-cam-y'),
        camZ:          document.getElementById('ctrl-cam-z'),
        fov:           document.getElementById('ctrl-fov'),
        applyCam:      document.getElementById('ctrl-apply-cam'),
        captureCam:    document.getElementById('ctrl-capture-cam'),
        reset:         document.getElementById('ctrl-reset'),
        copy:          document.getElementById('ctrl-copy'),
        copyOut:       document.getElementById('ctrl-copy-output'),

        lbl: {
            pointSize:  document.getElementById('ctrl-pointsize-val'),
            sampling:   document.getElementById('ctrl-sampling-val'),
            brightness: document.getElementById('ctrl-brightness-val'),
            ambient:    document.getElementById('ctrl-ambient-val'),
            opacity:    document.getElementById('ctrl-opacity-val'),
            rotX:       document.getElementById('ctrl-rot-x-val'),
            rotY:       document.getElementById('ctrl-rot-y-val'),
            rotZ:       document.getElementById('ctrl-rot-z-val'),
            fov:        document.getElementById('ctrl-fov-val')
        }
    };

    // --- PLY parser ------------------------------------------------------
    // Binary little-endian PLY: the header is ASCII, the vertex data is
    // packed according to `property` declarations.  We only materialise
    // XYZ (+ optional RGB) into typed arrays.  Anything else in the
    // payload is skipped by advancing a cursor.
    function parsePly(arrayBuffer, onProgress) {
        const bytes = new Uint8Array(arrayBuffer);
        // Header ends with "end_header\n".  Scan forward until we hit it.
        let headerEnd = -1;
        const needle = 'end_header\n';
        for (let i = 0; i < bytes.length - needle.length; i++) {
            let match = true;
            for (let j = 0; j < needle.length; j++) {
                if (bytes[i + j] !== needle.charCodeAt(j)) { match = false; break; }
            }
            if (match) { headerEnd = i + needle.length; break; }
        }
        if (headerEnd < 0) throw new Error('Not a valid PLY file (no end_header)');

        const headerText = new TextDecoder('ascii').decode(bytes.subarray(0, headerEnd));
        const lines = headerText.split(/\r?\n/);

        let format = null;
        let vertexCount = 0;
        let inVertexElement = false;
        const props = [];  // [{ name, type }]
        for (const line of lines) {
            const parts = line.trim().split(/\s+/);
            if (!parts.length) continue;
            if (parts[0] === 'format') format = parts[1];
            else if (parts[0] === 'element' && parts[1] === 'vertex') {
                vertexCount = parseInt(parts[2], 10);
                inVertexElement = true;
            } else if (parts[0] === 'element') inVertexElement = false;
            else if (parts[0] === 'property' && inVertexElement) {
                // Handle both `property float x` and `property list uchar int …`
                // (we don't support list properties inside vertex element; skip).
                if (parts[1] === 'list') {
                    props.push({ name: parts[parts.length - 1], type: 'list' });
                } else {
                    props.push({ name: parts[parts.length - 1], type: parts[1] });
                }
            }
        }
        if (format !== 'binary_little_endian') {
            throw new Error(`Only binary_little_endian PLY is supported (got "${format}")`);
        }

        // Figure out per-vertex byte size, and offsets to xyz + rgb.
        const typeSize = { float: 4, float32: 4, double: 8, uchar: 1, uint8: 1,
                           char: 1, int8: 1, ushort: 2, uint16: 2, short: 2,
                           int16: 2, uint: 4, uint32: 4, int: 4, int32: 4 };
        const typeReader = (dv, off, t) => {
            switch (t) {
                case 'float': case 'float32':   return dv.getFloat32(off, true);
                case 'double':                  return dv.getFloat64(off, true);
                case 'uchar': case 'uint8':     return dv.getUint8(off);
                case 'char':  case 'int8':      return dv.getInt8(off);
                case 'ushort': case 'uint16':   return dv.getUint16(off, true);
                case 'short':  case 'int16':    return dv.getInt16(off, true);
                case 'uint':   case 'uint32':   return dv.getUint32(off, true);
                case 'int':    case 'int32':    return dv.getInt32(off, true);
                default:                        return 0;
            }
        };

        let stride = 0;
        const offsets = {};
        for (const p of props) {
            if (p.type === 'list') {
                throw new Error('PLY list properties on vertex element are not supported');
            }
            const s = typeSize[p.type];
            if (s == null) throw new Error(`Unsupported PLY property type: ${p.type}`);
            offsets[p.name] = { offset: stride, type: p.type };
            stride += s;
        }

        const has = (n) => offsets[n] != null;
        const pickColor = (channel) => {
            if (has(channel)) return offsets[channel];
            if (has('diffuse_' + channel)) return offsets['diffuse_' + channel];
            return null;
        };

        if (!has('x') || !has('y') || !has('z')) {
            throw new Error('PLY vertex element must have x, y, z');
        }
        const xOff = offsets.x, yOff = offsets.y, zOff = offsets.z;
        const rOff = pickColor('red');
        const gOff = pickColor('green');
        const bOff = pickColor('blue');
        const hasColor = rOff && gOff && bOff;

        // Build typed arrays.
        const positions = new Float32Array(vertexCount * 3);
        const colors    = new Uint8Array(vertexCount * 3);

        const dv = new DataView(arrayBuffer, headerEnd);
        // Reporting every 64k verts keeps the UI responsive on big clouds.
        const reportEvery = Math.max(65536, Math.floor(vertexCount / 50));
        for (let i = 0; i < vertexCount; i++) {
            const base = i * stride;
            positions[i * 3]     = typeReader(dv, base + xOff.offset, xOff.type);
            positions[i * 3 + 1] = typeReader(dv, base + yOff.offset, yOff.type);
            positions[i * 3 + 2] = typeReader(dv, base + zOff.offset, zOff.type);
            if (hasColor) {
                colors[i * 3]     = typeReader(dv, base + rOff.offset, rOff.type);
                colors[i * 3 + 1] = typeReader(dv, base + gOff.offset, gOff.type);
                colors[i * 3 + 2] = typeReader(dv, base + bOff.offset, bOff.type);
            } else {
                colors[i * 3] = colors[i * 3 + 1] = colors[i * 3 + 2] = 200;
            }
            if (onProgress && i % reportEvery === 0) onProgress(i / vertexCount);
        }
        if (onProgress) onProgress(1);

        return { positions, colors, vertexCount, hasColor };
    }

    // --- .pnt.gz parser --------------------------------------------------
    // Mirrors the v4 format in ../script.js: 44-byte header, then
    // blocks of SoA byte-planes.  We decompress the whole file first
    // (DecompressionStream) and then walk it in one pass.
    async function parsePntGz(arrayBuffer, onProgress) {
        // Decompress via DecompressionStream if available.
        let decompressed;
        if (typeof DecompressionStream !== 'undefined') {
            const stream = new Blob([arrayBuffer]).stream()
                .pipeThrough(new DecompressionStream('gzip'));
            const parts = [];
            const reader = stream.getReader();
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                parts.push(value);
            }
            let total = 0;
            for (const p of parts) total += p.byteLength;
            decompressed = new Uint8Array(total);
            let off = 0;
            for (const p of parts) { decompressed.set(p, off); off += p.byteLength; }
        } else {
            throw new Error('Browser lacks DecompressionStream support for .pnt.gz');
        }

        const dv = new DataView(decompressed.buffer);
        const magic =
            String.fromCharCode(dv.getUint8(0)) +
            String.fromCharCode(dv.getUint8(1)) +
            String.fromCharCode(dv.getUint8(2)) +
            String.fromCharCode(dv.getUint8(3));
        if (magic !== 'UNP4') throw new Error(`Not a UNP4 file (got "${magic}")`);
        const count     = dv.getUint32(8, true);
        const blockSize = dv.getUint32(12, true);
        const numBlocks = dv.getUint32(16, true);
        const minX = dv.getFloat32(20, true);
        const minY = dv.getFloat32(24, true);
        const minZ = dv.getFloat32(28, true);
        const sx   = dv.getFloat32(32, true);
        const sy   = dv.getFloat32(36, true);
        const sz   = dv.getFloat32(40, true);

        const positions = new Float32Array(count * 3);
        const colors    = new Uint8Array(count * 3);

        let cursor = 44, write = 0;
        for (let b = 0; b < numBlocks; b++) {
            const bc = (b < numBlocks - 1) ? blockSize : (count - b * blockSize);
            const base = cursor;
            const xLo = base,           xHi = base + bc,
                  yLo = base + bc * 2,  yHi = base + bc * 3,
                  zLo = base + bc * 4,  zHi = base + bc * 5,
                  rB  = base + bc * 6,  gB  = base + bc * 7, bB = base + bc * 8;
            for (let i = 0; i < bc; i++) {
                const qx = decompressed[xLo + i] | (decompressed[xHi + i] << 8);
                const qy = decompressed[yLo + i] | (decompressed[yHi + i] << 8);
                const qz = decompressed[zLo + i] | (decompressed[zHi + i] << 8);
                const k = (write + i) * 3;
                positions[k]     = minX + qx * sx;
                positions[k + 1] = minY + qy * sy;
                positions[k + 2] = minZ + qz * sz;
                colors[k]     = decompressed[rB + i];
                colors[k + 1] = decompressed[gB + i];
                colors[k + 2] = decompressed[bB + i];
            }
            cursor += bc * 9;
            write += bc;
            if (onProgress) onProgress(write / count);
        }

        return { positions, colors, vertexCount: count, hasColor: true };
    }

    // --- Three.js viewer -------------------------------------------------
    function initThree() {
        if (three) return three;
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0xffffff);

        const rect = el.canvasWrap.getBoundingClientRect();
        const camera = new THREE.PerspectiveCamera(55, rect.width / rect.height, 0.01, 5000);
        camera.position.set(2, 1.5, 2);

        const renderer = new THREE.WebGLRenderer({
            canvas: el.canvas, antialias: true, powerPreference: 'high-performance'
        });
        renderer.setSize(rect.width, rect.height);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.75));

        const controls = new THREE.OrbitControls(camera, el.canvas);
        controls.enableDamping = true;
        controls.dampingFactor = 0.08;
        controls.minDistance = 0.01;
        controls.maxDistance = 5000;

        function animate() {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }
        animate();

        window.addEventListener('resize', () => onResize());

        three = { scene, camera, renderer, controls,
                  pointCloud: null, material: null, geometry: null,
                  radius: 1, totalPoints: 0 };
        return three;
    }

    function onResize() {
        if (!three) return;
        const r = el.canvasWrap.getBoundingClientRect();
        three.camera.aspect = r.width / r.height;
        three.camera.updateProjectionMatrix();
        three.renderer.setSize(r.width, r.height);
    }

    function clearPointCloud() {
        if (!three || !three.pointCloud) return;
        three.scene.remove(three.pointCloud);
        three.geometry.dispose();
        three.material.dispose();
        three.pointCloud = null;
        three.geometry = null;
        three.material = null;
        three.totalPoints = 0;
    }

    // Install a fresh point cloud from decoded positions/colors, centre
    // on bounding-box midpoint and size the camera offset by the box's
    // half-diagonal (matches the online viewer's radius convention).
    function installCloud(decoded, params) {
        initThree();
        clearPointCloud();

        const { positions, colors, vertexCount } = decoded;

        // Centre the cloud.  We compute the bbox directly rather than
        // trusting `geometry.computeBoundingBox` because the positions
        // buffer is shared with the typed array we want to mutate.
        let minX = Infinity, minY = Infinity, minZ = Infinity;
        let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
        for (let i = 0; i < vertexCount; i++) {
            const x = positions[i * 3], y = positions[i * 3 + 1], z = positions[i * 3 + 2];
            if (x < minX) minX = x; if (x > maxX) maxX = x;
            if (y < minY) minY = y; if (y > maxY) maxY = y;
            if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
        }
        const cx = (minX + maxX) * 0.5;
        const cy = (minY + maxY) * 0.5;
        const cz = (minZ + maxZ) * 0.5;
        const hx = (maxX - minX) * 0.5;
        const hy = (maxY - minY) * 0.5;
        const hz = (maxZ - minZ) * 0.5;
        const radius = Math.sqrt(hx * hx + hy * hy + hz * hz) || 1;
        for (let i = 0; i < vertexCount; i++) {
            positions[i * 3]     -= cx;
            positions[i * 3 + 1] -= cy;
            positions[i * 3 + 2] -= cz;
        }

        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3, true));
        geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0, 0), radius);

        const material = new THREE.PointsMaterial({
            size: params.pointSize,
            vertexColors: true,
            sizeAttenuation: true,
            transparent: params.opacity < 1,
            opacity: params.opacity,
            depthWrite: params.opacity >= 1
        });

        // Shader tweak: uniform brightness + ambient boost (biases dark
        // pixels toward a user-set floor, like turning on an ambient
        // lamp without needing normals).  Anchor the injection at
        // `<tonemapping_fragment>` rather than `<output_fragment>` —
        // r128's points_frag writes `gl_FragColor` inline, so the older
        // anchor would silently no-op.  Splicing before tonemapping
        // keeps the scaling in linear space, which is the right order
        // for exposure-like adjustments.
        material.userData.uBrightness = { value: params.brightness };
        material.userData.uAmbient = { value: params.ambient };
        material.onBeforeCompile = (shader) => {
            shader.uniforms.uBrightness = material.userData.uBrightness;
            shader.uniforms.uAmbient    = material.userData.uAmbient;
            shader.fragmentShader = shader.fragmentShader
                .replace(
                    'void main() {',
                    'uniform float uBrightness;\nuniform float uAmbient;\nvoid main() {'
                )
                .replace(
                    '#include <tonemapping_fragment>',
                    'vec3 col = gl_FragColor.rgb * uBrightness;\n\t' +
                    'col = col + (vec3(1.0) - col) * uAmbient;\n\t' +
                    'gl_FragColor.rgb = col;\n\t' +
                    '#include <tonemapping_fragment>'
                );
            material.userData.shader = shader;
        };

        const pointCloud = new THREE.Points(geometry, material);
        three.scene.add(pointCloud);

        three.pointCloud = pointCloud;
        three.geometry = geometry;
        three.material = material;
        three.radius = radius;
        three.totalPoints = vertexCount;

        applyParams(params);
        setCamera(params.camera, params.fov);

        el.hint.classList.add('hidden');
    }

    // --- Params ---------------------------------------------------------
    let params = defaultParams();

    function applyParams(p) {
        if (!three || !three.pointCloud) return;
        three.material.size = p.pointSize;
        three.material.opacity = p.opacity;
        three.material.transparent = p.opacity < 1;
        three.material.depthWrite = p.opacity >= 1;
        three.material.needsUpdate = true;
        if (three.material.userData.shader) {
            three.material.userData.shader.uniforms.uBrightness.value = p.brightness;
            three.material.userData.shader.uniforms.uAmbient.value    = p.ambient;
        }

        const count = Math.max(1, Math.floor(three.totalPoints * p.samplingRate));
        three.geometry.setDrawRange(0, count);

        const d2r = Math.PI / 180;
        const pitch = (p.rotation.x || 0) * d2r;
        const yaw   = (p.rotation.y || 0) * d2r;
        const roll  = (p.rotation.z || 0) * d2r;
        const euler = new THREE.Euler(pitch, yaw, roll, 'XYZ');
        three.pointCloud.rotation.copy(euler);
        three.pointCloud.scale.y = p.flipY ? -1 : 1;

        three.scene.background = new THREE.Color(p.background);
    }

    function setCamera(cam, fov) {
        if (!three) return;
        const r = three.radius || 1;
        three.camera.position.set(r * cam.x, r * cam.y, r * cam.z);
        three.camera.fov = fov;
        three.camera.updateProjectionMatrix();
        three.controls.target.set(0, 0, 0);
        three.controls.update();
    }

    function getCameraOffset() {
        if (!three) return { x: 0, y: 0, z: 0 };
        const r = three.radius || 1;
        return {
            x: +(three.camera.position.x / r).toFixed(4),
            y: +(three.camera.position.y / r).toFixed(4),
            z: +(three.camera.position.z / r).toFixed(4)
        };
    }

    // --- UI wiring ------------------------------------------------------
    function wireUI() {
        const bindRange = (input, label, key, fmt) => {
            input.addEventListener('input', () => {
                const v = parseFloat(input.value);
                params[key] = v;
                if (label) label.textContent = fmt(v);
                applyParams(params);
            });
        };

        bindRange(el.pointSize,  el.lbl.pointSize,  'pointSize',  v => v.toFixed(3));
        bindRange(el.brightness, el.lbl.brightness, 'brightness', v => v.toFixed(2) + '×');
        bindRange(el.ambient,    el.lbl.ambient,    'ambient',    v => v.toFixed(2));
        bindRange(el.opacity,    el.lbl.opacity,    'opacity',    v => v.toFixed(2));
        el.sampling.addEventListener('input', () => {
            const v = parseInt(el.sampling.value, 10);
            params.samplingRate = v / 100;
            el.lbl.sampling.textContent = v + '%';
            applyParams(params);
        });

        const bindRot = (input, label, axis) => {
            input.addEventListener('input', () => {
                const v = parseFloat(input.value);
                params.rotation[axis] = v;
                if (label) label.textContent = v + '°';
                applyParams(params);
            });
        };
        bindRot(el.rotX, el.lbl.rotX, 'x');
        bindRot(el.rotY, el.lbl.rotY, 'y');
        bindRot(el.rotZ, el.lbl.rotZ, 'z');

        el.flipY.addEventListener('change', () => {
            params.flipY = el.flipY.checked;
            applyParams(params);
        });

        el.bgSwatches.querySelectorAll('.bg-swatch').forEach(btn => {
            btn.addEventListener('click', () => {
                el.bgSwatches.querySelectorAll('.bg-swatch').forEach(b =>
                    b.classList.toggle('selected', b === btn)
                );
                params.background = btn.dataset.bg;
                applyParams(params);
            });
        });

        el.fov.addEventListener('input', () => {
            const v = parseInt(el.fov.value, 10);
            params.fov = v;
            el.lbl.fov.textContent = v + '°';
            if (three) {
                three.camera.fov = v;
                three.camera.updateProjectionMatrix();
            }
        });

        el.applyCam.addEventListener('click', () => {
            const cam = {
                x: parseFloat(el.camX.value) || 0,
                y: parseFloat(el.camY.value) || 0,
                z: parseFloat(el.camZ.value) || 0
            };
            params.camera = cam;
            setCamera(cam, params.fov);
        });

        el.captureCam.addEventListener('click', () => {
            const cam = getCameraOffset();
            params.camera = cam;
            el.camX.value = cam.x;
            el.camY.value = cam.y;
            el.camZ.value = cam.z;
            el.captureCam.textContent = 'Captured ✓';
            setTimeout(() => { el.captureCam.textContent = 'Use current'; }, 1500);
        });

        el.reset.addEventListener('click', () => {
            params = defaultParams();
            syncInputs();
            applyParams(params);
            setCamera(params.camera, params.fov);
        });

        el.copy.addEventListener('click', () => copyConfig());

        // File input + drag/drop.
        el.fileInput.addEventListener('change', (e) => handleFiles(e.target.files));
        el.clearBtn.addEventListener('click', () => clearAll());

        const cw = el.canvasWrap;
        ['dragenter', 'dragover'].forEach(evt =>
            cw.addEventListener(evt, (e) => {
                e.preventDefault();
                e.dataTransfer.dropEffect = 'copy';
                cw.classList.add('dragging');
            }));
        ['dragleave', 'drop'].forEach(evt =>
            cw.addEventListener(evt, (e) => {
                e.preventDefault();
                if (evt !== 'drop' && e.target !== cw) return;
                cw.classList.remove('dragging');
            }));
        cw.addEventListener('drop', (e) => {
            if (!e.dataTransfer) return;
            handleFiles(e.dataTransfer.files);
        });
    }

    function syncInputs() {
        el.pointSize.value  = params.pointSize;   el.lbl.pointSize.textContent  = params.pointSize.toFixed(3);
        el.sampling.value   = Math.round(params.samplingRate * 100);
        el.lbl.sampling.textContent = Math.round(params.samplingRate * 100) + '%';
        el.brightness.value = params.brightness;  el.lbl.brightness.textContent = params.brightness.toFixed(2) + '×';
        el.ambient.value    = params.ambient;     el.lbl.ambient.textContent    = params.ambient.toFixed(2);
        el.opacity.value    = params.opacity;     el.lbl.opacity.textContent    = params.opacity.toFixed(2);
        el.rotX.value       = params.rotation.x;  el.lbl.rotX.textContent       = (params.rotation.x || 0) + '°';
        el.rotY.value       = params.rotation.y;  el.lbl.rotY.textContent       = (params.rotation.y || 0) + '°';
        el.rotZ.value       = params.rotation.z;  el.lbl.rotZ.textContent       = (params.rotation.z || 0) + '°';
        el.flipY.checked    = !!params.flipY;
        el.fov.value        = params.fov;         el.lbl.fov.textContent        = params.fov + '°';
        el.camX.value       = params.camera.x;
        el.camY.value       = params.camera.y;
        el.camZ.value       = params.camera.z;
        el.bgSwatches.querySelectorAll('.bg-swatch').forEach(b => {
            b.classList.toggle('selected', b.dataset.bg.toLowerCase() === params.background.toLowerCase());
        });
    }

    function copyConfig() {
        const f = files[currentIndex];
        const keyHint = f ? f.name.replace(/\.(ply|pnt|gz)$/gi, '').replace(/[^A-Za-z0-9_]/g, '_').toLowerCase() : 'scene';
        const cam = getCameraOffset();
        const lines = [
            `${keyHint}: {`,
            `    title: ${JSON.stringify(f ? f.name : 'scene')},`,
            `    cloud: "assets/demos/${keyHint}/scene.pnt.gz",`,
            `    pointSize: ${params.pointSize.toFixed(4)},`,
            `    flipY: ${!!params.flipY},`,
            `    camera: { x: ${cam.x}, y: ${cam.y}, z: ${cam.z} },`,
            `    fov: ${params.fov},`,
            `    samplingRate: ${params.samplingRate.toFixed(3)},`,
            `    brightness: ${params.brightness.toFixed(2)},`,
            `    ambient: ${params.ambient.toFixed(2)},`,
            `    opacity: ${params.opacity.toFixed(2)},`,
            `    background: ${JSON.stringify(params.background)},`,
            `    rotation: { x: ${params.rotation.x || 0}, y: ${params.rotation.y || 0}, z: ${params.rotation.z || 0} }`,
            `}`
        ];
        const text = lines.join('\n');
        el.copyOut.textContent = text;
        navigator.clipboard?.writeText(text).then(() => {
            el.copy.textContent = 'Copied ✓';
            setTimeout(() => { el.copy.textContent = 'Copy Config'; }, 1500);
        }).catch(() => {
            el.copy.textContent = 'See box ↓';
            setTimeout(() => { el.copy.textContent = 'Copy Config'; }, 1500);
        });
    }

    // --- File handling --------------------------------------------------
    function setStatus(text) {
        if (!text) { el.status.classList.remove('visible'); el.status.textContent = ''; return; }
        el.status.textContent = text;
        el.status.classList.add('visible');
    }

    function updateFileListUI() {
        el.fileList.innerHTML = '';
        files.forEach((f, idx) => {
            const li = document.createElement('li');
            li.className = idx === currentIndex ? 'selected' : '';
            const name = document.createElement('span');
            name.textContent = f.name;
            name.style.overflow = 'hidden';
            name.style.textOverflow = 'ellipsis';
            name.style.whiteSpace = 'nowrap';
            const meta = document.createElement('span');
            meta.className = 'file-meta';
            meta.textContent = f.data
                ? formatPoints(f.data.vertexCount)
                : (f.error ? 'error' : 'loading…');
            li.append(name, meta);
            li.addEventListener('click', () => {
                if (f.data) selectFile(idx);
            });
            el.fileList.appendChild(li);
        });
        updateSceneInfo();
    }

    function updateSceneInfo() {
        const f = files[currentIndex];
        if (!f || !f.data) {
            el.sceneInfo.innerHTML = '<p class="hint">Drop a <code>.ply</code> file anywhere on the viewer, or click <strong>Open PLY…</strong>.</p>';
            return;
        }
        el.sceneInfo.innerHTML = `
            <div><strong>${escapeHtml(f.name)}</strong></div>
            <div style="margin-top:6px; font-size:0.82em; color:#5f6368">
                ${formatPoints(f.data.vertexCount)} points · ${(f.size / 1e6).toFixed(2)} MB on disk · radius ${three?.radius.toFixed(2) ?? '—'}
            </div>`;
    }

    function escapeHtml(s) {
        return s.replace(/[&<>"']/g, (c) => ({
            '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
        }[c]));
    }

    function formatPoints(n) {
        if (n >= 1e6) return (n / 1e6).toFixed(2) + 'M';
        if (n >= 1e3) return (n / 1e3).toFixed(0) + 'K';
        return String(n);
    }

    function clearAll() {
        files.length = 0;
        currentIndex = -1;
        updateFileListUI();
        clearPointCloud();
        el.hint.classList.remove('hidden');
        setStatus('');
    }

    async function handleFiles(fileList) {
        if (!fileList || !fileList.length) return;
        for (const f of fileList) {
            const entry = { name: f.name, size: f.size, data: null, error: null };
            files.push(entry);
            updateFileListUI();
            try {
                setStatus(`Reading ${f.name}…`);
                const buf = await f.arrayBuffer();
                const lower = f.name.toLowerCase();
                let decoded;
                if (lower.endsWith('.ply')) {
                    decoded = parsePly(buf, (p) => setStatus(`Parsing ${f.name} — ${(p * 100).toFixed(0)}%`));
                } else if (lower.endsWith('.pnt.gz') || lower.endsWith('.gz') || lower.endsWith('.pnt')) {
                    decoded = await parsePntGz(buf, (p) => setStatus(`Decoding ${f.name} — ${(p * 100).toFixed(0)}%`));
                } else {
                    throw new Error(`Unsupported file type: ${f.name}`);
                }
                entry.data = decoded;
                setStatus(`${f.name} ready — ${formatPoints(decoded.vertexCount)} pts`);
                setTimeout(() => setStatus(''), 2000);
                if (currentIndex === -1) selectFile(files.length - 1);
                else updateFileListUI();
            } catch (err) {
                console.error(err);
                entry.error = err.message;
                setStatus(`Error: ${err.message}`);
                setTimeout(() => setStatus(''), 4000);
                updateFileListUI();
            }
        }
        // Reset the input so re-selecting the same file re-loads it.
        el.fileInput.value = '';
    }

    function selectFile(idx) {
        if (idx < 0 || idx >= files.length) return;
        const f = files[idx];
        if (!f.data) return;
        currentIndex = idx;
        updateFileListUI();
        installCloud(f.data, params);
        updateSceneInfo();
    }

    // --- Boot -----------------------------------------------------------
    wireUI();
    syncInputs();
    // Pre-initialise Three.js so the canvas is ready before the first
    // file drops in.
    initThree();
})();
