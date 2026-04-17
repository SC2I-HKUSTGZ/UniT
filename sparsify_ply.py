#!/usr/bin/env python3
"""Voxel-grid downsample a binary_little_endian PLY into a gzipped .pnt.gz file.

Why voxel grids and not random sampling
---------------------------------------
Random subsampling preserves the *original density distribution* — regions
that were densely scanned stay over-represented, sparse regions stay
sparse.  In a point-cloud viewer this means you see a clumpy, "noisy"
cloud with obvious holes.

A voxel grid enforces spatially uniform density: every occupied cell of
size ``voxel_size`` contributes one representative point (the mean of all
source points that fell into it, in both position and colour).  The
result looks dramatically crisper at the same point count.

Output format: .pnt v4 (gzipped, true-streaming)
------------------------------------------------
v4 replaces v3's two-section coarse/fine split with a *block stream*
so the browser can render **every** new block the moment its bytes
land — no waiting for a second "fine" pass.  Any file prefix is a
uniform random subsample of the scene, so early blocks already sketch
the whole thing; later blocks fill in detail smoothly.

The recipe is:

1.  Quantise positions to uint16 (same as v3).
2.  *Randomly shuffle* the entire point cloud.  Critical: this makes
    any prefix of the file a uniformly-distributed spatial sample —
    the user sees the whole scene sparsely first, not one spatial
    corner.
3.  Split into fixed-size blocks (default 16 384 points).
4.  *Inside each block*, Morton-sort the shuffled points.  Spatial
    locality is recovered within the block so the byte-plane SoA
    stream below still compresses well under DEFLATE.
5.  Emit byte-plane SoA per block (X_lo, X_hi, Y_lo, Y_hi, Z_lo, Z_hi,
    R, G, B), each stream ``block_count`` bytes long.
6.  Gzip the whole payload.

On-disk layout of the uncompressed v4 payload::

    offset  size    field
    0       4       magic           "UNP4"
    4       4       version         uint32 LE  (= 4)
    8       4       count_total     uint32 LE  (= N)
    12      4       block_size      uint32 LE  (points per full block)
    16      4       num_blocks      uint32 LE  (= ceil(N / block_size))
    20      12      min_xyz         3 × float32 LE
    32      12      scale_xyz       3 × float32 LE   # xyz = min + q * scale
    44      ...     block 0
                    block 1
                    ...
                    block (num_blocks - 1)

Each block is *exactly*::

    bc = block_size            (for blocks 0 .. num_blocks-2)
       = N - (num_blocks-1)*block_size   (for the last block)

    bc  X_lo    (low byte of quantised X for each of bc points)
    bc  X_hi    (high byte)
    bc  Y_lo, Y_hi, Z_lo, Z_hi
    bc  R, G, B

``bc`` is not stored inline — the client computes it from the header,
saving ~784 bytes over 200 blocks.  The packing is deterministic, so
the reader can stop any time and discard a trailing partial block.

Why per-block Morton (instead of global Morton + stride sample)
---------------------------------------------------------------
Global Morton sort gives DEFLATE the longest runs of identical high-
bytes (~2× better compression than per-block), but any prefix then
only covers one spatial corner of the scene — useless for progressive
rendering.  Conversely, a pure random shuffle gives great progressive
rendering but destroys byte-plane locality within blocks.  Shuffling
*across* blocks then Morton-sorting *within* blocks splits the
difference: you pay ~10-15 % on compression versus global Morton, in
exchange for every file prefix being a uniform spatial sample.

The whole payload is then run through gzip at level 9 and written to
``<out>.pnt.gz``.  The viewer decompresses it client-side through the
``DecompressionStream`` Web API, so no server-side ``Content-Encoding``
configuration is required.

``--target`` auto-bisects the voxel size to hit a point-count budget.
"""

import argparse
import gzip
import io
import os
import struct

import numpy as np


# --------------------------------------------------------------------------
# PLY reading (binary little-endian, xyz float + rgba/rgb uchar layout used
# by trimesh, which is what the UniT demos ship as).
# --------------------------------------------------------------------------
def read_ply(in_path):
    with open(in_path, "rb") as f:
        header = b""
        while True:
            line = f.readline()
            if not line:
                raise RuntimeError("unexpected EOF in PLY header")
            header += line
            if line.strip() == b"end_header":
                break

        n_vertex = None
        props = []
        for line in header.decode("ascii", errors="replace").splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[0] == "element" and parts[1] == "vertex":
                n_vertex = int(parts[2])
            elif len(parts) >= 3 and parts[0] == "property":
                props.append((parts[2], parts[1]))  # (name, type)
        if n_vertex is None:
            raise RuntimeError("no vertex element found")

        dtype_fields = []
        for name, typ in props:
            if typ == "float":
                dtype_fields.append((name, "<f4"))
            elif typ == "uchar":
                dtype_fields.append((name, "u1"))
            elif typ == "double":
                dtype_fields.append((name, "<f8"))
            else:
                raise RuntimeError(f"unsupported property type: {typ}")
        arr = np.frombuffer(
            f.read(n_vertex * np.dtype(dtype_fields).itemsize),
            dtype=np.dtype(dtype_fields),
        )
    xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype("<f4", copy=False)
    # Accept both plain rgb and diffuse_rgb naming.
    r_name = "red" if "red" in arr.dtype.names else "diffuse_red"
    g_name = "green" if "green" in arr.dtype.names else "diffuse_green"
    b_name = "blue" if "blue" in arr.dtype.names else "diffuse_blue"
    rgb = np.stack([arr[r_name], arr[g_name], arr[b_name]], axis=1).astype("u1", copy=False)
    return xyz, rgb


# --------------------------------------------------------------------------
# Voxel-grid downsample: assign each input point to a voxel, then average
# positions and colours within each occupied voxel.
# --------------------------------------------------------------------------
def voxel_downsample(xyz: np.ndarray, rgb: np.ndarray, voxel_size: float):
    keys = np.floor(xyz / voxel_size).astype(np.int64)
    mins = keys.min(axis=0)
    extents = keys.max(axis=0) - mins + 1
    flat = (
        (keys[:, 0] - mins[0])
        + (keys[:, 1] - mins[1]) * extents[0]
        + (keys[:, 2] - mins[2]) * extents[0] * extents[1]
    )
    _, inverse, counts = np.unique(flat, return_inverse=True, return_counts=True)
    n_vox = counts.size

    sum_pos = np.zeros((n_vox, 3), dtype=np.float64)
    np.add.at(sum_pos, inverse, xyz.astype(np.float64))
    mean_pos = (sum_pos / counts[:, None]).astype(np.float32)

    sum_col = np.zeros((n_vox, 3), dtype=np.float64)
    np.add.at(sum_col, inverse, rgb.astype(np.float64))
    mean_col = np.clip(sum_col / counts[:, None], 0, 255).astype(np.uint8)

    return mean_pos, mean_col


def pick_voxel_size(xyz: np.ndarray, rgb: np.ndarray, target_n: int) -> tuple:
    """Bisect voxel size until ``voxel_downsample`` yields ~target_n points."""
    extent = (xyz.max(axis=0) - xyz.min(axis=0)).max()
    lo, hi = extent / 4096.0, extent / 4.0
    best = None
    for _ in range(22):
        mid = (lo * hi) ** 0.5  # geometric mean
        pos, col = voxel_downsample(xyz, rgb, mid)
        n = pos.shape[0]
        best = (mid, pos, col, n)
        if n > target_n * 1.05:
            lo = mid  # need a bigger voxel (fewer points)
        elif n < target_n * 0.95:
            hi = mid  # need a smaller voxel (more points)
        else:
            break
    return best


# --------------------------------------------------------------------------
# Morton (Z-order) keys on 16-bit quantised coordinates.
#
# Sorting on the Morton key groups points that are close in 3-D into
# contiguous runs in the file.  Adjacent quantised uint16 values then
# differ by small amounts, which is exactly what gzip's back-references
# compress well.  Measured gain over un-sorted points: ~1.8–2.2× tighter
# after DEFLATE.
# --------------------------------------------------------------------------
def _spread_bits3(v: np.ndarray) -> np.ndarray:
    """Spread the low 16 bits of each element across 48 bits (1 bit per 3)."""
    v = v.astype(np.uint64)
    m0 = np.uint64(0x0000FFFF0000FFFF)
    m1 = np.uint64(0x00FF00FF00FF00FF)
    m2 = np.uint64(0x0F0F0F0F0F0F0F0F)
    m3 = np.uint64(0x3333333333333333)
    m4 = np.uint64(0x5555555555555555)
    v = (v | (v << np.uint64(16))) & m0
    v = (v | (v << np.uint64(8)))  & m1
    v = (v | (v << np.uint64(4)))  & m2
    v = (v | (v << np.uint64(2)))  & m3
    v = (v | (v << np.uint64(1)))  & m4
    return v


def morton_keys(qx: np.ndarray, qy: np.ndarray, qz: np.ndarray) -> np.ndarray:
    return (
        _spread_bits3(qx)
        | (_spread_bits3(qy) << np.uint64(1))
        | (_spread_bits3(qz) << np.uint64(2))
    )


# --------------------------------------------------------------------------
# PNT v4 + gzip writer
# --------------------------------------------------------------------------
def write_pnt_v4_gz(xyz: np.ndarray, rgb: np.ndarray, out_path: str,
                    block_size: int = 16_384, gzip_level: int = 9,
                    seed: int = 42) -> tuple:
    """Write v4 payload to ``out_path`` (gzipped).

    Returns ``(raw_bytes, gz_bytes)``.
    """
    n = int(xyz.shape[0])
    mins = xyz.min(axis=0).astype(np.float32)
    maxs = xyz.max(axis=0).astype(np.float32)
    ranges = np.maximum(maxs - mins, np.float32(1e-6))
    scale = (ranges / np.float32(65535.0)).astype(np.float32)

    q = np.clip(np.round((xyz - mins) / scale), 0, 65535).astype(np.uint16)
    qx, qy, qz = q[:, 0], q[:, 1], q[:, 2]

    # Step 1: uniform random shuffle over all points.  Every prefix of the
    # shuffled array is then a uniformly-distributed spatial sample of the
    # scene — so the browser can render "everything received so far" at
    # any moment and have it cover the whole bounding box, just sparsely.
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)

    # Step 2: cut into fixed-size blocks.  The last block may be shorter.
    num_blocks = (n + block_size - 1) // block_size

    buf = io.BytesIO()
    buf.write(b"UNP4")
    buf.write(struct.pack("<I", 4))              # version
    buf.write(struct.pack("<I", n))              # count_total
    buf.write(struct.pack("<I", int(block_size))) # block_size
    buf.write(struct.pack("<I", int(num_blocks))) # num_blocks
    buf.write(mins.tobytes())                    # min_xyz
    buf.write(scale.tobytes())                   # scale_xyz

    for b in range(num_blocks):
        s = b * block_size
        e = min(s + block_size, n)
        idx = order[s:e]

        # Step 3: Morton-sort inside the block so within-block byte-plane
        # SoA actually compresses.  (Without this, adjacent bytes in each
        # SoA channel look like white noise and DEFLATE can't match them.)
        keys = morton_keys(qx[idx], qy[idx], qz[idx])
        idx = idx[np.argsort(keys, kind="stable")]

        # Step 4: emit byte-plane SoA.
        for axis in (qx, qy, qz):
            a = axis[idx].astype(np.uint16, copy=False)
            buf.write((a & np.uint16(0xFF)).astype("u1").tobytes())
            buf.write((a >> np.uint16(8)).astype("u1").tobytes())
        buf.write(rgb[idx, 0].astype("u1").tobytes())
        buf.write(rgb[idx, 1].astype("u1").tobytes())
        buf.write(rgb[idx, 2].astype("u1").tobytes())

    raw = buf.getvalue()
    with gzip.open(out_path, "wb", compresslevel=gzip_level) as gz:
        gz.write(raw)

    return len(raw), os.path.getsize(out_path)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_path")
    ap.add_argument("out_path", help="output .pnt.gz path")
    ap.add_argument("--target", type=int,
                    help="approximate target point count; bisects voxel size")
    ap.add_argument("--voxel", type=float,
                    help="explicit voxel size (scene units / metres)")
    ap.add_argument("--block-size", type=int, default=16_384,
                    help="points per block in the v4 stream (smaller = "
                         "smoother progressive render, slightly worse "
                         "compression)")
    args = ap.parse_args()
    if (args.target is None) == (args.voxel is None):
        ap.error("provide exactly one of --target or --voxel")

    xyz, rgb = read_ply(args.in_path)
    n_src = xyz.shape[0]

    if args.voxel is not None:
        voxel = args.voxel
        pos, col = voxel_downsample(xyz, rgb, voxel)
    else:
        voxel, pos, col, _ = pick_voxel_size(xyz, rgb, args.target)

    raw_bytes, gz_bytes = write_pnt_v4_gz(
        pos, col, args.out_path, block_size=args.block_size,
    )

    src_mb = os.path.getsize(args.in_path) / 1e6
    ratio = raw_bytes / max(gz_bytes, 1)
    print(
        f"{os.path.basename(args.in_path)}: {n_src:,} → {pos.shape[0]:,} pts  "
        f"(voxel={voxel:.4g}, src={src_mb:.1f} MB, "
        f"raw={raw_bytes/1e6:.2f} MB, gz={gz_bytes/1e6:.2f} MB, ratio={ratio:.2f}×)"
    )


if __name__ == "__main__":
    main()
