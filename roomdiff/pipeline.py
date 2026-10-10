"""End-to-end analysis: two observation sets in, a change report and viewer out."""
import json
import pathlib

import numpy as np

from .changes import find_changes
from .describe import describe, evaluate
from .viewer import write_outputs
from .voxels import FREE, OCC, UNKNOWN, Grid


def analyze(before, after, out_dir, voxel=0.06, min_votes=1, unit="m", gt=None, title="Room change report",
            bounds=None):
    """before/after: dicts with 'cams' (N,3) and 'pts' (N,3), in the same coordinate frame."""
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

    changes = find_changes(s_b, s_a, grids[0])
    lines = describe(changes, unit)

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
    )
    if gt is not None:
        stats["evaluation"] = evaluate(changes, gt)

    report = dict(title=title, stats=stats, changes=[
        {k: (np.round(v, 3).tolist() if isinstance(v, np.ndarray) else (round(v, 3) if isinstance(v, float) else v))
         for k, v in c.items() if k not in ("text",)} | {"text": c["text"]} for c in changes])
    (out / "changes.json").write_text(json.dumps(report, indent=2))
    write_outputs(out, changes, static, lo, hi, stats, lines, title, unit)
    return report
