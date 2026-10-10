"""Turn two occupancy grids into a list of moved / added / removed objects."""
import numpy as np
from scipy import ndimage

from .voxels import FREE, OCC

_NEIGHBOURHOOD = np.ones((3, 3, 3), bool)


def find_changes(s_before, s_after, grid, min_voxels=12, max_move=3.0, max_shape_diff=0.4):
    # Only report where BOTH recordings looked: occupied in one and seen empty in the other.
    # UNKNOWN on either side is never a change. A voxel right next to a surface seen in the
    # other recording is excluded too: surface positions jitter by about a voxel between passes,
    # so those voxels are edge noise rather than missing objects.
    occ_b, occ_a = s_before == OCC, s_after == OCC
    near_b = ndimage.binary_dilation(occ_b, structure=_NEIGHBOURHOOD)
    near_a = ndimage.binary_dilation(occ_a, structure=_NEIGHBOURHOOD)
    removed = _clusters(occ_b & (s_after == FREE) & ~near_a, grid, min_voxels)
    added = _clusters(occ_a & (s_before == FREE) & ~near_b, grid, min_voxels)
    return _pair(removed, added, max_move, max_shape_diff)


def _clusters(mask, grid, min_voxels):
    lab, _ = ndimage.label(mask, structure=np.ones((3, 3, 3)))
    found = []
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        idx = np.argwhere(lab[sl] == i) + np.array([s.start for s in sl])
        if len(idx) < min_voxels:
            continue
        pts = grid.centers(idx)
        lo, hi = pts.min(0) - grid.voxel / 2, pts.max(0) + grid.voxel / 2
        found.append(dict(centroid=pts.mean(0), lo=lo, hi=hi, size=hi - lo, n_voxels=len(idx)))
    return found


def _pair(removed, added, max_move, max_shape_diff):
    # Greedy nearest-first matching: a removed and an added cluster with the same extent (x, y, z
    # within a log-ratio budget) and close enough together are the same object that moved.
    # Shape is checked before distance, so a nearby but differently sized object is not a match.
    cands = []
    for i, r in enumerate(removed):
        for j, a in enumerate(added):
            dist = float(np.linalg.norm(a["centroid"] - r["centroid"]))
            shape = float(np.abs(np.log(a["size"] / r["size"])).sum())
            if dist <= max_move and shape <= max_shape_diff:
                cands.append((dist, i, j))
    used_r, used_a, out = set(), set(), []
    for dist, i, j in sorted(cands):
        if i in used_r or j in used_a:
            continue
        used_r.add(i)
        used_a.add(j)
        r, a = removed[i], added[j]
        out.append(dict(kind="moved", at=r["centroid"], to=a["centroid"], distance=dist,
                        size=r["size"], lo=r["lo"], hi=r["hi"], to_lo=a["lo"], to_hi=a["hi"]))
    for i, r in enumerate(removed):
        if i not in used_r:
            out.append(dict(kind="removed", at=r["centroid"], size=r["size"], lo=r["lo"], hi=r["hi"]))
    for j, a in enumerate(added):
        if j not in used_a:
            out.append(dict(kind="added", at=a["centroid"], size=a["size"], lo=a["lo"], hi=a["hi"]))
    return out
