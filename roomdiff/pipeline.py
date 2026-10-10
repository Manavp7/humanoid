"""End-to-end analysis: two observation sets in, a change report and viewer out."""
import json
import pathlib
import subprocess
import sys

import numpy as np

from .changes import find_changes
from .describe import describe, evaluate, label_for
from .viewer import write_outputs
from .voxels import OCC, UNKNOWN, Grid


def analyze(before, after, out_dir, voxel=0.06, min_votes=1, unit="m", gt=None, title="Room change report",
            bounds=None, frame_context=None, min_voxels=12, labels="shape", vision_work=None,
            max_move=3.0, max_shape_diff=0.4, edge=1):
    """before/after: dicts with 'cams' (N,3) and 'pts' (N,3), in the same coordinate frame.

    frame_context / vision_work: when labels='clip', CLIP runs in a subprocess under vision_work
    (avoids loading torch in the same process as pycolmap). Demo path leaves these None.
    """
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    allp = np.vstack([np.vstack([o["pts"], o["cams"]]) for o in (before, after)])
    lo, hi = allp.min(0) - 2 * voxel, allp.max(0) + 2 * voxel
    if bounds is not None:
        lo, hi = bounds

    grids = []
    for obs in (before, after):
        g = Grid(lo, hi, voxel)
        g.add(obs["cams"], obs["pts"])
        grids.append(g)
    s_b, s_a = grids[0].states(min_votes), grids[1].states(min_votes)

    changes = find_changes(s_b, s_a, grids[0], grids[1], min_voxels=min_voxels, max_move=max_move,
                           max_shape_diff=max_shape_diff, edge=edge)

    label_fn = None
    label_source = "shape"
    if labels == "clip" and (frame_context or vision_work):
        label_fn = _clip_labels_via_subprocess(changes, vision_work)
        if label_fn is not None:
            label_source = "clip"
    lines = describe(changes, unit, label_fn=label_fn)

    static = grids[0].centers(np.argwhere((s_b == OCC) & (s_a == OCC)))
    # Space that is hidden in one recording but visible in the other: never reported as a change.
    unverified = int(np.sum(((s_b == OCC) & (s_a == UNKNOWN)) | ((s_b == UNKNOWN) & (s_a == OCC))))
    observed_both = int(np.sum((s_b != UNKNOWN) & (s_a != UNKNOWN)))
    total = int(s_b.size)
    stats = dict(
        voxel=voxel, unit=unit, grid_shape=list(s_b.shape), voxels_total=total,
        voxels_seen_in_both=observed_both, voxels_seen_in_both_pct=round(100 * observed_both / total, 1),
        voxels_unverified=unverified, static_occupied=int(len(static)),
        changes=len(changes), kinds={k: sum(c["kind"] == k for c in changes) for k in ("moved", "added", "removed")},
        labels=label_source,
    )
    if gt is not None:
        stats["evaluation"] = evaluate(changes, gt)

    skip = {"text", "idx"}
    report = dict(title=title, stats=stats, changes=[
        {k: (np.round(v, 3).tolist() if isinstance(v, np.ndarray) else (round(v, 3) if isinstance(v, float) else v))
         for k, v in c.items() if k not in skip} | {"text": c["text"]} for c in changes])
    (out / "changes.json").write_text(json.dumps(report, indent=2))
    write_outputs(out, changes, static, lo, hi, stats, lines, title, unit)
    return report


def _clip_labels_via_subprocess(changes, vision_work):
    """Return a label_fn that maps each change to a CLIP label, or None on failure."""
    if not changes:
        return lambda c: label_for(c["size"])
    if vision_work is None:
        return None
    work = pathlib.Path(vision_work)
    serializable = []
    for c in changes:
        item = {}
        for k, v in c.items():
            item[k] = v.tolist() if isinstance(v, np.ndarray) else (float(v) if isinstance(v, (np.floating,)) else v)
        serializable.append(item)
    (work / "changes_for_clip.json").write_text(json.dumps(serializable))
    try:
        subprocess.run([sys.executable, "-m", "roomdiff.clip_worker", str(work)], check=True)
        labels = json.loads((work / "clip_labels.json").read_text())
    except Exception as exc:
        print(f"CLIP labeling failed ({exc}); falling back to shape labels.", file=sys.stderr)
        return None

    def label_fn(c):
        # describe() iterates in order; match by identity via index
        idx = label_fn._i
        label_fn._i += 1
        if 0 <= idx < len(labels):
            return labels[idx]
        return label_for(c["size"])

    label_fn._i = 0
    return label_fn
