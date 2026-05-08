#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const zlib = require('zlib');

const ROOT = path.resolve(__dirname, '..');
const DEMO_DIR = path.join(ROOT, 'assets', 'demos');
const POINTS_PER_CHUNK = 65_536;
const HEADER_SIZE = 64;
const INDEX_ENTRY_SIZE = 24;

function parsePntHeader(buf) {
    return {
        magic: buf.subarray(0, 4).toString('ascii'),
        version: buf.readUInt32LE(4),
        count: buf.readUInt32LE(8),
        blockSize: buf.readUInt32LE(12),
        numBlocks: buf.readUInt32LE(16),
        min: [buf.readFloatLE(20), buf.readFloatLE(24), buf.readFloatLE(28)],
        scale: [buf.readFloatLE(32), buf.readFloatLE(36), buf.readFloatLE(40)],
    };
}

function writeHeader(header, chunkCount) {
    const out = Buffer.alloc(HEADER_SIZE);
    out.write('PNTS', 0, 'ascii');
    out.writeUInt32LE(1, 4);
    out.writeUInt32LE(header.count, 8);
    out.writeUInt32LE(header.blockSize, 12);
    out.writeUInt32LE(POINTS_PER_CHUNK, 16);
    out.writeUInt32LE(chunkCount, 20);
    out.writeUInt32LE(INDEX_ENTRY_SIZE, 24);
    out.writeUInt32LE(HEADER_SIZE, 28);
    out.writeFloatLE(header.min[0], 32);
    out.writeFloatLE(header.min[1], 36);
    out.writeFloatLE(header.min[2], 40);
    out.writeFloatLE(header.scale[0], 44);
    out.writeFloatLE(header.scale[1], 48);
    out.writeFloatLE(header.scale[2], 52);
    return out;
}

function blockOffset(header, blockIndex) {
    return 44 + blockIndex * header.blockSize * 9;
}

function buildScene(sceneDir) {
    const input = path.join(sceneDir, 'scene.pnt.gz');
    if (!fs.existsSync(input)) return null;

    const decoded = zlib.gunzipSync(fs.readFileSync(input));
    const header = parsePntHeader(decoded);
    if (header.magic !== 'UNP4' || header.version !== 4) {
        throw new Error(`${input} is not a UNP4 v4 file`);
    }
    if (POINTS_PER_CHUNK % header.blockSize !== 0) {
        throw new Error(`${input}: POINTS_PER_CHUNK must be a multiple of blockSize`);
    }

    const blocksPerChunk = POINTS_PER_CHUNK / header.blockSize;
    const chunkCount = Math.ceil(header.numBlocks / blocksPerChunk);
    const chunks = [];

    for (let chunkIndex = 0; chunkIndex < chunkCount; chunkIndex += 1) {
        const firstBlock = chunkIndex * blocksPerChunk;
        const lastBlock = Math.min(firstBlock + blocksPerChunk, header.numBlocks);
        const firstPoint = firstBlock * header.blockSize;
        const pointCount = Math.min(header.count - firstPoint, POINTS_PER_CHUNK);
        const rawParts = [];

        for (let blockIndex = firstBlock; blockIndex < lastBlock; blockIndex += 1) {
            const bc = (blockIndex < header.numBlocks - 1)
                ? header.blockSize
                : (header.count - blockIndex * header.blockSize);
            const start = blockOffset(header, blockIndex);
            rawParts.push(decoded.subarray(start, start + bc * 9));
        }

        const raw = Buffer.concat(rawParts);
        const compressed = zlib.gzipSync(raw, { level: 9 });
        chunks.push({
            firstPoint,
            pointCount,
            blockCount: lastBlock - firstBlock,
            rawSize: raw.length,
            compressed,
        });
    }

    const index = Buffer.alloc(chunks.length * INDEX_ENTRY_SIZE);
    let payloadOffset = HEADER_SIZE + index.length;
    chunks.forEach((chunk, i) => {
        const base = i * INDEX_ENTRY_SIZE;
        index.writeUInt32LE(chunk.firstPoint, base);
        index.writeUInt32LE(chunk.pointCount, base + 4);
        index.writeUInt32LE(chunk.blockCount, base + 8);
        index.writeUInt32LE(chunk.compressed.length, base + 12);
        index.writeUInt32LE(chunk.rawSize, base + 16);
        index.writeUInt32LE(payloadOffset, base + 20);
        payloadOffset += chunk.compressed.length;
    });

    const output = path.join(sceneDir, 'scene.pnts');
    fs.writeFileSync(output, Buffer.concat([
        writeHeader(header, chunks.length),
        index,
        ...chunks.map(c => c.compressed),
    ]));

    const originalSize = fs.statSync(input).size;
    const streamSize = fs.statSync(output).size;
    return {
        scene: path.basename(sceneDir),
        points: header.count,
        chunks: chunks.length,
        originalMB: +(originalSize / 1048576).toFixed(2),
        streamMB: +(streamSize / 1048576).toFixed(2),
        ratio: +(streamSize / originalSize).toFixed(3),
    };
}

const summaries = fs.readdirSync(DEMO_DIR, { withFileTypes: true })
    .filter(d => d.isDirectory())
    .map(d => buildScene(path.join(DEMO_DIR, d.name)))
    .filter(Boolean);

for (const summary of summaries) {
    console.log(JSON.stringify(summary));
}
