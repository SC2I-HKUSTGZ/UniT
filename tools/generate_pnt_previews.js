#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const zlib = require('zlib');

const ROOT = path.resolve(__dirname, '..');
const DEMO_DIR = path.join(ROOT, 'assets', 'demos');
const TARGET_POINTS = 393216; // 24 blocks when blockSize is 16384.

function parseHeader(buf) {
    return {
        magic: buf.subarray(0, 4).toString('ascii'),
        version: buf.readUInt32LE(4),
        count: buf.readUInt32LE(8),
        blockSize: buf.readUInt32LE(12),
        numBlocks: buf.readUInt32LE(16),
    };
}

function patchHeader(buf, count, numBlocks) {
    const out = Buffer.from(buf.subarray(0, 44));
    out.writeUInt32LE(count, 8);
    out.writeUInt32LE(numBlocks, 16);
    return out;
}

function generatePreview(sceneDir) {
    const input = path.join(sceneDir, 'scene.pnt.gz');
    const output = path.join(sceneDir, 'preview.pnt.gz');
    if (!fs.existsSync(input)) return null;

    const decoded = zlib.gunzipSync(fs.readFileSync(input));
    const header = parseHeader(decoded);
    if (header.magic !== 'UNP4') {
        throw new Error(`${input} is not a UNP4 file`);
    }

    const previewBlocks = Math.max(1, Math.min(
        header.numBlocks,
        Math.floor(TARGET_POINTS / header.blockSize)
    ));
    const previewCount = Math.min(header.count, previewBlocks * header.blockSize);
    const previewBytes = 44 + previewCount * 9;
    const patchedHeader = patchHeader(decoded, previewCount, previewBlocks);
    const payload = decoded.subarray(44, previewBytes);
    const preview = Buffer.concat([patchedHeader, payload]);
    fs.writeFileSync(output, zlib.gzipSync(preview, { level: 9 }));

    return {
        scene: path.basename(sceneDir),
        points: previewCount,
        blocks: previewBlocks,
        mb: +(fs.statSync(output).size / 1048576).toFixed(2),
    };
}

const summaries = fs.readdirSync(DEMO_DIR, { withFileTypes: true })
    .filter(d => d.isDirectory())
    .map(d => generatePreview(path.join(DEMO_DIR, d.name)))
    .filter(Boolean);

for (const summary of summaries) {
    console.log(JSON.stringify(summary));
}
