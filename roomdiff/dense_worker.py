"""Subprocess entry: dense depth without importing pycolmap (avoids OpenMP/torch crash).

Usage: python -m roomdiff.dense_worker <work_dir>
Expects work_dir/frames_meta.npz (object array of frame dicts) and frame JPEGs on disk.
Optional work_dir/sparse_obs.npz with cams_A/pts_A/cams_B/pts_B for outlier filtering.
Writes work_dir/dense_obs.npz with cams_A, pts_A, cams_B, pts_B.
"""
from __future__ import annotations

import os
import pathlib
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TORCH_NUM_THREADS", "1")


def main(argv=None):
    argv = list(argv or sys.argv[1:])
    if len(argv) < 1:
        print("usage: python -m roomdiff.dense_worker <work_dir> [stride]", file=sys.stderr)
        return 2
    work = pathlib.Path(argv[0])
    stride = int(argv[1]) if len(argv) > 1 else 8
    import numpy as np
    from .depth import dense_observations_from_frames, filter_near_sparse

    packed = list(np.load(work / "frames_meta.npz", allow_pickle=True)["frames"])
    result = dense_observations_from_frames(packed, work / "frames", stride=stride)
    meta = result.pop("_meta", {})
    sparse_path = work / "sparse_obs.npz"
    if sparse_path.is_file():
        sparse = np.load(sparse_path)
        all_sparse = np.vstack([sparse["pts_A"], sparse["pts_B"]])
        diag = float(np.linalg.norm(all_sparse.max(0) - all_sparse.min(0)))
        max_dist = max(0.06, 0.015 * diag)
        for tag in ("A", "B"):
            if tag not in result:
                continue
            s_pts = sparse[f"pts_{tag}"]
            cams, pts = filter_near_sparse(result[tag]["cams"], result[tag]["pts"], s_pts, max_dist)
            result[tag] = dict(
                cams=np.vstack([sparse[f"cams_{tag}"], cams]) if len(cams) else sparse[f"cams_{tag}"],
                pts=np.vstack([s_pts, pts]) if len(pts) else s_pts,
            )
            meta[f"kept_dense_{tag}"] = int(len(cams))
        meta["max_dist"] = max_dist
    out = work / "dense_obs.npz"
    payload = {}
    for tag, obs in result.items():
        if tag.startswith("_"):
            continue
        payload[f"cams_{tag}"] = obs["cams"]
        payload[f"pts_{tag}"] = obs["pts"]
    payload["meta_skipped"] = np.array([meta.get("skipped_frames", 0)])
    np.savez_compressed(out, **payload)
    print(f"wrote {out} tags={[k for k in result if not k.startswith('_')]} meta={meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
