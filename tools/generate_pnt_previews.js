#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const zlib = require('zlib');

const ROOT = path.resolve(__dirname, '..');
const DEMO_DIR = path.join(ROOT, 'assets', 'demos');
const LEVELS = [
    ['preview.pnt.gz', 393216],  // 24 blocks when blockSize is 16384.
    ['lod-1.pnt.gz', 1572864],   // 96 blocks: dense enough to refine before full.
];

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

function writeLevel(decoded, header, sceneDir, fileName, targetPoints) {
    const output = path.join(sceneDir, fileName);
    const blocks = Math.max(1, Math.min(
        header.numBlocks,
        Math.floor(targetPoints / header.blockSize)
    ));
    const count = Math.min(header.count, blocks * header.blockSize);
    const bytes = 44 + count * 9;
    const patchedHeader = patchHeader(decoded, count, blocks);
    const payload = decoded.subarray(44, bytes);
    const level = Buffer.concat([patchedHeader, payload]);
    fs.writeFileSync(output, zlib.gzipSync(level, { level: 9 }));
    return {
        file: fileName,
        points: count,
        blocks,
        mb: +(fs.statSync(output).size / 1048576).toFixed(2),
    };
}

function generatePreview(sceneDir) {
    const input = path.join(sceneDir, 'scene.pnt.gz');
    if (!fs.existsSync(input)) return null;

    const decoded = zlib.gunzipSync(fs.readFileSync(input));
    const header = parseHeader(decoded);
    if (header.magic !== 'UNP4') {
        throw new Error(`${input} is not a UNP4 file`);
    }

    return {
        scene: path.basename(sceneDir),
        levels: LEVELS.map(([fileName, targetPoints]) =>
            writeLevel(decoded, header, sceneDir, fileName, targetPoints)
        ),
    };
}

const summaries = fs.readdirSync(DEMO_DIR, { withFileTypes: true })
    .filter(d => d.isDirectory())
    .map(d => generatePreview(path.join(DEMO_DIR, d.name)))
    .filter(Boolean);

for (const summary of summaries) {
    console.log(JSON.stringify(summary));
}
