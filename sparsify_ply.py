#!/usr/bin/env python3
"""Random-downsample binary_little_endian PLY files (xyz + rgba) for web viewing."""

import argparse
import os
import struct
import sys

import numpy as np


HEADER_TEMPLATE = (
    "ply\n"
    "format binary_little_endian 1.0\n"
    "comment sparsified for web viewer\n"
    "element vertex {n}\n"
    "property float x\n"
    "property float y\n"
    "property float z\n"
    "property uchar red\n"
    "property uchar green\n"
    "property uchar blue\n"
    "end_header\n"
)


def read_header(f):
    header = b""
    while True:
        line = f.readline()
        if not line:
            raise RuntimeError("unexpected EOF in header")
        header += line
        if line.strip() == b"end_header":
            break
    text = header.decode("ascii", errors="replace")
    n_vertex = None
    props = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "element" and parts[1] == "vertex":
            n_vertex = int(parts[2])
        elif len(parts) >= 3 and parts[0] == "property":
            props.append((parts[2], parts[1]))  # (name, type)
    if n_vertex is None:
        raise RuntimeError("no vertex element found")
    return n_vertex, props


def sparsify(in_path, out_path, target_n, seed=0):
    with open(in_path, "rb") as f:
        n_vertex, props = read_header(f)
        # Expected: xyz floats (12 bytes) + r g b [a] uchars (3 or 4)
        dtype_fields = []
        size = 0
        for name, typ in props:
            if typ == "float":
                dtype_fields.append((name, "<f4"))
                size += 4
            elif typ == "uchar":
                dtype_fields.append((name, "u1"))
                size += 1
            else:
                raise RuntimeError(f"unsupported property type: {typ}")
        arr = np.frombuffer(f.read(n_vertex * size), dtype=np.dtype(dtype_fields))

    n = len(arr)
    if target_n >= n:
        idx = np.arange(n)
    else:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=target_n, replace=False)
        idx.sort()
    sampled = arr[idx]

    out_n = len(sampled)
    xyz = np.stack([sampled["x"], sampled["y"], sampled["z"]], axis=1).astype("<f4")
    r = sampled["red"].astype("u1")
    g = sampled["green"].astype("u1")
    b = sampled["blue"].astype("u1")

    with open(out_path, "wb") as out:
        out.write(HEADER_TEMPLATE.format(n=out_n).encode("ascii"))
        buf = np.empty(out_n, dtype=np.dtype([
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ]))
        buf["x"] = xyz[:, 0]
        buf["y"] = xyz[:, 1]
        buf["z"] = xyz[:, 2]
        buf["red"] = r
        buf["green"] = g
        buf["blue"] = b
        out.write(buf.tobytes())

    return n, out_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_path")
    ap.add_argument("out_path")
    ap.add_argument("--target", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    n_before, n_after = sparsify(args.in_path, args.out_path, args.target, args.seed)
    src_mb = os.path.getsize(args.in_path) / 1e6
    dst_mb = os.path.getsize(args.out_path) / 1e6
    print(f"{os.path.basename(args.in_path)}: {n_before} → {n_after} pts  ({src_mb:.1f} MB → {dst_mb:.1f} MB)")


if __name__ == "__main__":
    main()
