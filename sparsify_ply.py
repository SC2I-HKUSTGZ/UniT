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

Output format: .pnt v3 (gzipped)
--------------------------------
v3 is designed for *fast progressive web delivery*, not disk footprint.
Three things change vs. v2:

1.  **Struct-of-Arrays layout.**  All X's, then all Y's, then Z's, then
    R, G, B.  Uniform per-channel byte streams compress much better
    under DEFLATE than the interleaved xyzxyz… of v2.
2.  **Morton-ordered points.**  Points are pre-sorted on a 3-D Morton
    curve so spatial neighbours are file-adjacent.  Adjacent uint16
    values in the quantised streams are then nearly identical, which
    is exactly what gzip's LZ77 thrives on — compression typically
    doubles compared to un-sorted data.
3.  **Two-section progressive layout.**  A small (<=80 k point)
    *coarse* subset is placed at the front of the file; the remainder
    follows.  The viewer renders the coarse subset as soon as it is
    decoded (usually well under a second), then streams the fine
    subset into the same geometry buffer over the rest of the download.

On-disk layout of the uncompressed v3 payload::

    offset  size       field
    0       4          magic           "UNP3"
    4       4          version         uint32 LE  (= 3)
    8       4          count_total     uint32 LE  (= N)
    12      4          count_coarse    uint32 LE  (= K <= N)
    16      12         min_xyz         3 × float32 LE
    28      12         scale_xyz       3 × float32 LE    # xyz = min + q * scale
    40            — Section A (coarse prefix, K points, byte-plane SoA) —
                K    X_lo      (low byte  of each quantised X)
                K    X_hi      (high byte of each quantised X)
                K    Y_lo
                K    Y_hi
                K    Z_lo
                K    Z_hi
                K    R
                K    G
                K    B
    40+9K         — Section B (fine remainder, N-K points, same byte-plane SoA) —
                N-K  X_lo, X_hi, Y_lo, Y_hi, Z_lo, Z_hi, R, G, B

Splitting each uint16 into two byte planes (low and high) is a standard
trick for columnar numeric compression: after Morton sorting the high
byte drifts slowly and the low byte is almost white-noise, so DEFLATE
can match long runs in the high-byte stream while the low-byte stream
costs roughly its raw size.  Net effect over v2 (interleaved little-
endian uint16): typically 2–3× tighter after gzip.

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
# PNT v3 + gzip writer
# --------------------------------------------------------------------------
def write_pnt_v3_gz(xyz: np.ndarray, rgb: np.ndarray, out_path: str,
                    coarse_frac: float = 0.05, max_coarse: int = 80_000,
                    min_coarse: int = 20_000, gzip_level: int = 9,
                    seed: int = 42) -> tuple:
    """Write v3 payload to ``out_path`` (gzipped).  Returns (raw_bytes, gz_bytes)."""
    n = int(xyz.shape[0])
    mins = xyz.min(axis=0).astype(np.float32)
    maxs = xyz.max(axis=0).astype(np.float32)
    ranges = np.maximum(maxs - mins, np.float32(1e-6))
    scale = (ranges / np.float32(65535.0)).astype(np.float32)

    q = np.clip(np.round((xyz - mins) / scale), 0, 65535).astype(np.uint16)
    qx, qy, qz = q[:, 0], q[:, 1], q[:, 2]

    # Coarse prefix: a uniform random subset (so early render isn't spatially
    # clumpy), but Morton-sorted *within* the subset so its bytes compress.
    coarse_count = min(int(round(n * coarse_frac)), max_coarse)
    coarse_count = max(min(n, min_coarse), coarse_count) if n >= min_coarse else n
    coarse_count = min(coarse_count, n)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    coarse_idx = perm[:coarse_count]
    fine_idx = perm[coarse_count:]

    def morton_sort(indices):
        if indices.size == 0:
            return indices
        keys = morton_keys(qx[indices], qy[indices], qz[indices])
        return indices[np.argsort(keys, kind="stable")]

    coarse_idx = morton_sort(coarse_idx)
    fine_idx = morton_sort(fine_idx)

    buf = io.BytesIO()
    buf.write(b"UNP3")
    buf.write(struct.pack("<I", 3))                         # version
    buf.write(struct.pack("<I", n))                         # total count
    buf.write(struct.pack("<I", int(coarse_count)))         # coarse count
    buf.write(mins.tobytes())                               # min_xyz
    buf.write(scale.tobytes())                              # scale_xyz

    def write_section(indices):
        if indices.size == 0:
            return
        # Byte-plane SoA: low byte then high byte, per axis.  Morton order
        # keeps the high-byte streams very slowly varying (huge DEFLATE
        # wins), while the low-byte streams look pseudo-random (DEFLATE
        # basically passes them through).
        for axis in (qx, qy, qz):
            a = axis[indices].astype(np.uint16, copy=False)
            buf.write((a & np.uint16(0xFF)).astype("u1").tobytes())
            buf.write((a >> np.uint16(8)).astype("u1").tobytes())
        buf.write(rgb[indices, 0].astype("u1").tobytes())
        buf.write(rgb[indices, 1].astype("u1").tobytes())
        buf.write(rgb[indices, 2].astype("u1").tobytes())

    write_section(coarse_idx)
    write_section(fine_idx)

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
    ap.add_argument("--coarse-frac", type=float, default=0.05,
                    help="fraction of points to place in the coarse prefix")
    ap.add_argument("--max-coarse", type=int, default=80_000,
                    help="cap on coarse-prefix point count")
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

    raw_bytes, gz_bytes = write_pnt_v3_gz(
        pos, col, args.out_path,
        coarse_frac=args.coarse_frac, max_coarse=args.max_coarse,
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
