"""Compare two recordings of the same room: python -m roomdiff demo | videos --before A --after B"""
import argparse
import sys
import tempfile

import numpy as np

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
    cf.extract_frames(args.before, f"{work}/frames", "A", args.fps)
    cf.extract_frames(args.after, f"{work}/frames", "B", args.fps)
    rec = cf.reconstruct(f"{work}/frames", f"{work}/colmap")
    obs = cf.observations(rec)
    for o in obs.values():
        o["cams"] = o["cams"] * args.scale
        o["pts"] = o["pts"] * args.scale
    # COLMAP's scale is arbitrary and a few wild points stretch the box, so trim to the 2-98th
    # percentile of the points and size voxels relative to the scene.
    P = np.vstack([obs["A"]["pts"], obs["B"]["pts"]])
    lo, hi = np.percentile(P, 2, axis=0), np.percentile(P, 98, axis=0)
    for o in obs.values():
        keep = np.all((o["pts"] >= lo) & (o["pts"] <= hi), axis=1)
        o["pts"], o["cams"] = o["pts"][keep], o["cams"][keep]
    voxel = args.voxel or 0.015 * float(np.linalg.norm(hi - lo))
    rep = analyze(obs["A"], obs["B"], args.out, voxel=voxel, bounds=(lo - voxel, hi + voxel),
                  unit="m" if args.scale != 1.0 else "units", title="What changed in the room?")
    print(rep["stats"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
