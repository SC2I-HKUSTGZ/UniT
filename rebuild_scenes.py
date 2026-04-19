#!/usr/bin/env python3
"""Rebuild every demo's scene.pnt.gz from the canonical PLY set.

Reads from ../canonical_plys/ (see canonical_plys/MANIFEST.md), writes
full-fidelity (no voxel downsample) v4 streams back into
assets/demos/<key>/scene.pnt.gz.

Run from the webpage/ directory:

    python3 rebuild_scenes.py

After the rebuild: commit the changed .pnt.gz files, bump CACHE_NAME in
script.js (e.g. unit-pnt-v10 → v11), and push.  The CACHE_NAME bump is
load-bearing — the client keeps its v<N> Cache API entry keyed to the
URL, so without the bump returning visitors keep the old bytes.
"""
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)  # for sparsify_ply.py
from sparsify_ply import read_ply, write_pnt_v4_gz  # noqa: E402

CANONICAL = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "canonical_plys"))
OUT_ROOT  = os.path.join(SCRIPT_DIR, "assets", "demos")

SCENES = {
    "hkust_intr":    "1.HKUST_INTR/HKUST_INTR.ply",
    "hkust_toy":     "2.HKUST_Toy/HKUST_Toy.ply",
    "hkust_redbird": "3.HKUST_RedBird/HKUST_Redbird_video.ply",
    "drift":         "4.drift-straight/drift-straight_conf30_maxdepth12_stride6.ply",
    "gta_sfm":       "6.gta_sfm/gta_sfm_conf20_maxdepth30_stride3.ply",
    "kitti":         "7.kitti/kitti_conf20_maxdepth40_stride2.ply",
}


def main() -> None:
    for key, rel in SCENES.items():
        src = os.path.join(CANONICAL, rel)
        dst = os.path.join(OUT_ROOT, key, "scene.pnt.gz")
        if not os.path.exists(src):
            print(f"SKIP  {key}: missing source {src}")
            continue
        xyz, rgb = read_ply(src)
        raw, gz = write_pnt_v4_gz(xyz, rgb, dst)
        print(f"OK    {key}: {xyz.shape[0]:,} pts, gz {gz/1e6:.2f} MB")


if __name__ == "__main__":
    main()
