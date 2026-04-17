#!/usr/bin/env python3
"""Voxel-grid downsample a binary_little_endian PLY into a .pnt v2 file.

Why voxel grids and not random sampling
---------------------------------------
Random subsampling preserves the *original density distribution* — regions
that were densely scanned stay over-represented, sparse regions stay
sparse.  In a point-cloud viewer this means you see a clumpy, "noisy"
cloud with obvious holes.

A voxel grid enforces spatially uniform density: every occupied cell of
size `voxel_size` contributes one representative point (the mean of all
source points that fell into it, in both position and colour).  The
result looks dramatically crisper at the same point count, so for a
given bandwidth budget we can ship a *better-looking* cloud.

The output file format (.pnt v2) matches what the viewer expects:

    [0..4)    magic   "UNPT"
    [4..8)    version uint32 LE = 2
    [8..12)   count   uint32 LE
    [12..24)  min_xyz   (3 × float32)
    [24..36)  scale_xyz (3 × float32)      # decode: xyz = min + q * scale
    [36 .. 36+n*6)       pos_q  (uint16 × 3)
    [36+n*6 .. 36+n*9)   rgb    (uint8  × 3)

--target auto-bisects the voxel size to hit a point-count budget.
"""

import argparse
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
            else:
                raise RuntimeError(f"unsupported property type: {typ}")
        arr = np.frombuffer(
            f.read(n_vertex * np.dtype(dtype_fields).itemsize),
            dtype=np.dtype(dtype_fields),
        )
    xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype("<f4", copy=False)
    rgb = np.stack([arr["red"], arr["green"], arr["blue"]], axis=1).astype("u1", copy=False)
    return xyz, rgb


# --------------------------------------------------------------------------
# Voxel-grid downsample: assign each input point to a voxel, then average
# positions and colours within each occupied voxel.
# --------------------------------------------------------------------------
def voxel_downsample(xyz: np.ndarray, rgb: np.ndarray, voxel_size: float):
    keys = np.floor(xyz / voxel_size).astype(np.int64)
    mins = keys.min(axis=0)
    extents = keys.max(axis=0) - mins + 1
    # Pack (i, j, k) into a single int64 key so np.unique can group them.
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
    lo, hi = extent / 2048.0, extent / 4.0
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
# PNT v2 writer
# --------------------------------------------------------------------------
def write_pnt_v2(xyz: np.ndarray, rgb: np.ndarray, out_path: str):
    n = xyz.shape[0]
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    ranges = np.maximum(maxs - mins, 1e-6)
    scale = ranges / 65535.0
    q = np.clip(np.round((xyz - mins) / scale), 0, 65535).astype("<u2")

    with open(out_path, "wb") as out:
        out.write(b"UNPT")
        out.write(struct.pack("<I", 2))            # version
        out.write(struct.pack("<I", n))            # vertex count
        out.write(mins.astype("<f4").tobytes())    # min_xyz
        out.write(scale.astype("<f4").tobytes())   # scale_xyz
        out.write(q.tobytes())                     # positions  (uint16 × 3)
        out.write(rgb.astype("u1").tobytes())      # colours    (uint8  × 3)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_path")
    ap.add_argument("out_path")
    ap.add_argument("--target", type=int,
                    help="approximate target point count; bisects voxel size")
    ap.add_argument("--voxel", type=float,
                    help="explicit voxel size (scene units / metres)")
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

    write_pnt_v2(pos, col, args.out_path)

    src_mb = os.path.getsize(args.in_path) / 1e6
    dst_mb = os.path.getsize(args.out_path) / 1e6
    print(
        f"{os.path.basename(args.in_path)}: {n_src:,} → {pos.shape[0]:,} pts  "
        f"(voxel={voxel:.4g}, {src_mb:.1f} MB → {dst_mb:.2f} MB)"
    )


if __name__ == "__main__":
    main()
