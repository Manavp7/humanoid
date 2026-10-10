"""Subprocess entry: CLIP-label changes without pycolmap loaded.

Usage: python -m roomdiff.clip_worker <work_dir>
Reads work_dir/changes_for_clip.json and work_dir/frames_meta.npz.
Writes work_dir/clip_labels.json: list of label strings aligned with changes.
"""
from __future__ import annotations

import json
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
        print("usage: python -m roomdiff.clip_worker <work_dir>", file=sys.stderr)
        return 2
    work = pathlib.Path(argv[0])
    import numpy as np

    from .clip_label import make_label_fn

    frames = list(np.load(work / "frames_meta.npz", allow_pickle=True)["frames"])
    # Reconstruct arrays that may have been stored as object
    for fr in frames:
        for key in ("K", "cam_from_world", "center", "anchors_uv", "anchors_z"):
            fr[key] = np.asarray(fr[key])
    changes = json.loads((work / "changes_for_clip.json").read_text())
    for c in changes:
        for key in ("size", "at", "lo", "hi", "to", "to_lo", "to_hi"):
            if key in c:
                c[key] = np.asarray(c[key], dtype=np.float64)
    label_fn = make_label_fn(frames)
    labels = [label_fn(c) for c in changes]
    (work / "clip_labels.json").write_text(json.dumps(labels))
    print("labels", labels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
