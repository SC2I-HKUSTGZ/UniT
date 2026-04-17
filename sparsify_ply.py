#!/usr/bin/env python3
"""Random-downsample binary_little_endian PLY files to .pnt for web viewing.

.pnt v2 layout (quantized, 9 bytes/vertex vs 15 in v1):

    [0..4)   magic   "UNPT"                       4 B
    [4..8)   version uint32 LE = 2                4 B
    [8..12)  count   uint32 LE                    4 B
    [12..24) min_xyz 3 × float32                 12 B
    [24..36) scale_xyz 3 × float32               12 B
    [36 .. 36+n*6)    pos_q  count × uint16[3]   6n B
    [36+n*6 .. 36+n*9) rgb    count × uint8[3]   3n B

Positions are reconstructed on load as  pos = min + q * scale  (per axis).
uint16 gives >65k steps per axis; even on a 180 m-long KITTI clip the
effective precision is ~2.7 mm, more than enough for visualization.
The +36 start keeps the uint16 position block 4-byte aligned.
"""

import argparse
import os
import struct

import numpy as np


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
    return arr


def write_pnt_v2(arr, out_path):
    n = len(arr)
    xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype("<f4", copy=False)
    rgb = np.stack([arr["red"], arr["green"], arr["blue"]], axis=1).astype("u1", copy=False)

    # Per-axis min / scale for int16 quantization.
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
        out.write(q.tobytes())                     # positions (uint16 × 3)
        out.write(rgb.tobytes())                   # colors (uint8 × 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_path")
    ap.add_argument("out_path")
    ap.add_argument("--target", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    arr = read_ply(args.in_path)
    n = len(arr)
    if args.target >= n:
        idx = np.arange(n)
    else:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(n, size=args.target, replace=False)
        idx.sort()
    sampled = arr[idx]
    write_pnt_v2(sampled, args.out_path)

    src_mb = os.path.getsize(args.in_path) / 1e6
    dst_mb = os.path.getsize(args.out_path) / 1e6
    print(f"{os.path.basename(args.in_path)}: {n} → {len(sampled)} pts  ({src_mb:.1f} MB → {dst_mb:.2f} MB)")


if __name__ == "__main__":
    main()
