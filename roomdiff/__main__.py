"""Compare two recordings of the same room: python -m roomdiff demo | videos --before A --after B"""
import argparse
import os
import pathlib
import subprocess
import sys
import tempfile
import warnings

import numpy as np

# pycolmap and torch both ship OpenMP; allow both on macOS. Torch still runs in a
# subprocess for depth/CLIP so the two native stacks never share one process.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from . import synth
from .pipeline import analyze


def main(argv=None):
    ap = argparse.ArgumentParser(prog="roomdiff", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="synthetic two-pass room with known ground truth (no videos needed)")
    d.add_argument("--out", default="examples/demo")
    d.add_argument("--voxel", type=float, default=0.06)

    s_ = sub.add_parser("sample", help="render two sample phone-style videos with the same changes")
    s_.add_argument("--out", default="examples/sample_videos")

    v = sub.add_parser("videos", help="two phone videos of the same room")
    v.add_argument("--before", required=True, help="first recording (mp4/mov)")
    v.add_argument("--after", required=True, help="second recording (mp4/mov)")
    v.add_argument("--out", default="output/videos")
    v.add_argument("--fps", type=float, default=2.0)
    v.add_argument("--voxel", type=float, default=None,
                   help="voxel size in reconstruction units (default: 1.5%% of the scene's robust diagonal)")
    v.add_argument("--scale", type=float, default=1.0, help="multiply coordinates to get metres")
    v.add_argument("--sparse-only", action="store_true",
                   help="skip Depth Anything V2; use sparse COLMAP points only")
    v.add_argument("--stride", type=int, default=8, help="pixel stride when unprojecting dense depth")
    v.add_argument("--min-voxels", type=int, default=30, help="minimum cluster size for a change object")
    v.add_argument("--no-clip", action="store_true", help="use shape-based labels instead of CLIP")
    args = ap.parse_args(argv)

    if args.cmd == "sample":
        print("wrote", synth.write_videos(args.out))
        return 0

    if args.cmd == "demo":
        before, after, gt = synth.make_demo()
        rep = analyze(before, after, args.out, voxel=args.voxel, gt=gt, title="Synthetic room: what changed?")
        print(rep["stats"])
        for c in rep["changes"]:
            print("-", c["text"])
        return 0

    from . import colmap_frontend as cf
    work = tempfile.mkdtemp(prefix="roomdiff_")
    frames_dir = f"{work}/frames"
    cf.extract_frames(args.before, frames_dir, "A", args.fps)
    cf.extract_frames(args.after, frames_dir, "B", args.fps)
    rec = cf.reconstruct(frames_dir, f"{work}/colmap")
    frame_context = cf.frame_index(rec, frames_dir)
    _save_frames_meta(work, frame_context)

    sparse = cf.observations(rec)
    np.savez_compressed(
        pathlib.Path(work) / "sparse_obs.npz",
        cams_A=sparse["A"]["cams"], pts_A=sparse["A"]["pts"],
        cams_B=sparse["B"]["cams"], pts_B=sparse["B"]["pts"],
    )
    if args.sparse_only:
        obs = sparse
        warnings.warn("Using sparse COLMAP points only (--sparse-only); detections may be empty.")
    else:
        obs = _dense_obs_subprocess(work, args.stride)
        for tag in ("A", "B"):
            if tag not in obs or len(obs[tag]["pts"]) == 0:
                warnings.warn(f"Dense depth missing tag {tag}; using sparse COLMAP points.")
                obs[tag] = sparse[tag]

    for o in obs.values():
        o["cams"] = o["cams"] * args.scale
        o["pts"] = o["pts"] * args.scale
    for fr in frame_context:
        fr["center"] = np.asarray(fr["center"], dtype=np.float64) * args.scale
        fr["cam_from_world"] = np.asarray(fr["cam_from_world"], dtype=np.float64).copy()
        fr["cam_from_world"][:3, 3] *= args.scale
        fr["anchors_z"] = np.asarray(fr["anchors_z"], dtype=np.float64) * args.scale
    _save_frames_meta(work, frame_context)

    P = np.vstack([obs["A"]["pts"], obs["B"]["pts"]])
    lo, hi = np.percentile(P, 2, axis=0), np.percentile(P, 98, axis=0)
    for o in obs.values():
        keep = np.all((o["pts"] >= lo) & (o["pts"] <= hi), axis=1)
        o["pts"], o["cams"] = o["pts"][keep], o["cams"][keep]
    diag = float(np.linalg.norm(hi - lo))
    # Dense depth fills the grid; a finer voxel than the old sparse default is needed.
    voxel = args.voxel or max(0.05, min(0.10, 0.01 * diag))
    labels = "shape" if args.no_clip else "clip"
    rep = analyze(
        obs["A"], obs["B"], args.out, voxel=voxel, bounds=(lo - voxel, hi + voxel),
        unit="m" if args.scale != 1.0 else "units", title="What changed in the room?",
        frame_context=frame_context, min_voxels=args.min_voxels, labels=labels,
        vision_work=work, min_votes=2, max_move=5.0, max_shape_diff=0.75, edge=2,
    )
    print(rep["stats"])
    for c in rep["changes"]:
        print("-", c["text"])
    return 0


def _save_frames_meta(work, frames):
    path = pathlib.Path(work) / "frames_meta.npz"
    np.savez_compressed(path, frames=np.array(frames, dtype=object))


def _dense_obs_subprocess(work, stride):
    subprocess.run(
        [sys.executable, "-m", "roomdiff.dense_worker", str(work), str(stride)],
        check=True,
    )
    data = np.load(pathlib.Path(work) / "dense_obs.npz")
    obs = {}
    for tag in ("A", "B"):
        ck, pk = f"cams_{tag}", f"pts_{tag}"
        if ck in data and pk in data and len(data[pk]):
            obs[tag] = dict(cams=data[ck], pts=data[pk])
    if not obs:
        raise RuntimeError("Dense depth worker produced no observations.")
    return obs


if __name__ == "__main__":
    sys.exit(main())
