"""Turn two occupancy grids into a list of moved / added / removed objects.

Visibility rule: a change is only occupied in one recording and *seen empty* in the other.
UNKNOWN never counts. Moves additionally require a free-space path between the two locations
so an object cannot "teleport" through unobserved or occupied space.
"""
from collections import deque

import numpy as np
from scipy import ndimage

from .voxels import FREE, OCC

_NEIGHBOURHOOD = np.ones((3, 3, 3), bool)
_FACE = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


def find_changes(s_before, s_after, grid_before, grid_after=None, min_voxels=12, max_move=3.0,
                 max_shape_diff=0.4, edge=1):
    # Only report where BOTH recordings looked: occupied in one and seen empty in the other.
    # UNKNOWN on either side is never a change. A voxel right next to a surface seen in the
    # other recording is excluded too: surface positions jitter by about a voxel between passes,
    # so those voxels are edge noise rather than missing objects.
    grid_after = grid_before if grid_after is None else grid_after
    occ_b, occ_a = s_before == OCC, s_after == OCC
    struct = np.ones((2 * edge + 1,) * 3, bool) if edge != 1 else _NEIGHBOURHOOD
    near_b = ndimage.binary_dilation(occ_b, structure=struct)
    near_a = ndimage.binary_dilation(occ_a, structure=struct)
    removed = _clusters(occ_b & (s_after == FREE) & ~near_a, grid_before, min_voxels)
    added = _clusters(occ_a & (s_before == FREE) & ~near_b, grid_before, min_voxels)

    for c in removed:
        c["confidence"] = _cluster_confidence(c["idx"], grid_before.occ, grid_after.free)
    for c in added:
        c["confidence"] = _cluster_confidence(c["idx"], grid_after.occ, grid_before.free)

    walkable = (s_before == FREE) | (s_after == FREE)
    return _pair(removed, added, max_move, max_shape_diff, walkable, grid_before)


def _soft_conf(mean_votes):
    return float(1.0 - np.exp(-max(mean_votes, 0.0) / 3.0))


def _cluster_confidence(idx, occ_votes, free_votes):
    """Ray-agreement: occupancy evidence on the present side + free-space on the empty side."""
    i = idx[:, 0], idx[:, 1], idx[:, 2]
    mean_occ = float(np.mean(occ_votes[i])) if len(idx) else 0.0
    mean_free = float(np.mean(free_votes[i])) if len(idx) else 0.0
    return _soft_conf(0.5 * (mean_occ + mean_free))


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
        found.append(dict(
            centroid=pts.mean(0), lo=lo, hi=hi, size=hi - lo, n_voxels=len(idx), idx=idx,
        ))
    return found


def _world_to_idx(pt, grid):
    idx = np.floor((np.asarray(pt, float) - grid.lo) / grid.voxel).astype(int)
    return tuple(idx.tolist())


def _free_path(start_pt, end_pt, walkable, grid, max_move):
    """6-connected BFS on walkable voxels; refuse paths longer than max_move."""
    shape = walkable.shape
    start = _world_to_idx(start_pt, grid)
    end = _world_to_idx(end_pt, grid)
    if not (0 <= start[0] < shape[0] and 0 <= start[1] < shape[1] and 0 <= start[2] < shape[2]):
        return False
    if not (0 <= end[0] < shape[0] and 0 <= end[1] < shape[1] and 0 <= end[2] < shape[2]):
        return False
    # Allow the endpoints even if marked OCC (cluster centroids sit on surfaces).
    max_steps = max(1, int(np.ceil(max_move / grid.voxel)) + 2)
    q = deque([(start, 0)])
    seen = {start}
    while q:
        (x, y, z), dist = q.popleft()
        if (x, y, z) == end:
            return True
        if dist >= max_steps:
            continue
        for dx, dy, dz in _FACE:
            n = (x + dx, y + dy, z + dz)
            if n in seen:
                continue
            if not (0 <= n[0] < shape[0] and 0 <= n[1] < shape[1] and 0 <= n[2] < shape[2]):
                continue
            if n != end and not walkable[n]:
                continue
            seen.add(n)
            q.append((n, dist + 1))
    return False


def _pair(removed, added, max_move, max_shape_diff, walkable, grid):
    # Greedy nearest-first matching: same extent, close enough, and a free-space path between
    # the two locations. Shape is checked before distance; path blocks teleporting through walls.
    cands = []
    for i, r in enumerate(removed):
        for j, a in enumerate(added):
            dist = float(np.linalg.norm(a["centroid"] - r["centroid"]))
            shape = float(np.abs(np.log(np.maximum(a["size"], 1e-6) / np.maximum(r["size"], 1e-6))).sum())
            if dist <= max_move and shape <= max_shape_diff:
                cands.append((dist, i, j))
    used_r, used_a, out = set(), set(), []
    for dist, i, j in sorted(cands):
        if i in used_r or j in used_a:
            continue
        r, a = removed[i], added[j]
        if not _free_path(r["centroid"], a["centroid"], walkable, grid, max_move):
            continue
        used_r.add(i)
        used_a.add(j)
        conf = float(min(r["confidence"], a["confidence"]))
        out.append(dict(
            kind="moved", at=r["centroid"], to=a["centroid"], distance=dist,
            size=r["size"], lo=r["lo"], hi=r["hi"], to_lo=a["lo"], to_hi=a["hi"],
            confidence=conf, path_ok=True,
        ))
    for i, r in enumerate(removed):
        if i not in used_r:
            out.append(dict(
                kind="removed", at=r["centroid"], size=r["size"], lo=r["lo"], hi=r["hi"],
                confidence=r["confidence"],
            ))
    for j, a in enumerate(added):
        if j not in used_a:
            out.append(dict(
                kind="added", at=a["centroid"], size=a["size"], lo=a["lo"], hi=a["hi"],
                confidence=a["confidence"],
            ))
    return out
