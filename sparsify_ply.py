#!/usr/bin/env python3
"""Random-downsample binary_little_endian PLY files to .pnt (zero-copy web format).

.pnt layout:
    [0..4)   magic "UNPT" (ASCII)
    [4..8)   format version uint32 LE (=1)
    [8..12)  vertex count uint32 LE
    [12 .. 12 + n*12)                 float32 xyz  (3 * n * 4 bytes)
    [12 + n*12 .. 12 + n*15)          uint8   rgb  (3 * n bytes)

The position block starts at offset 12 (4-byte aligned), so the loader can
alias it directly as a Float32Array with zero copy.  Colors are a plain
Uint8Array with `normalized = true` on the THREE.BufferAttribute.
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
                props.append((parts[2], parts[1]))
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
        arr = np.frombuffer(f.read(n_vertex * np.dtype(dtype_fields).itemsize),
                            dtype=np.dtype(dtype_fields))
    return arr


def write_pnt(arr, out_path):
    n = len(arr)
    xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype("<f4", copy=False)
    rgb = np.stack([arr["red"], arr["green"], arr["blue"]], axis=1).astype("u1", copy=False)
    with open(out_path, "wb") as out:
        out.write(b"UNPT")
        out.write(struct.pack("<I", 1))  # version
        out.write(struct.pack("<I", n))  # vertex count
        out.write(xyz.tobytes())
        out.write(rgb.tobytes())


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
    write_pnt(sampled, args.out_path)

    src_mb = os.path.getsize(args.in_path) / 1e6
    dst_mb = os.path.getsize(args.out_path) / 1e6
    print(f"{os.path.basename(args.in_path)}: {n} → {len(sampled)} pts  ({src_mb:.1f} MB → {dst_mb:.1f} MB)")


if __name__ == "__main__":
    main()
