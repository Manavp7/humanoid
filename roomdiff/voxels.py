"""Voxel occupancy with free-space carving from camera rays.

Every observation is a camera centre plus a 3D surface point seen along the ray between them.
Voxels on the ray (before the surface) are evidence of free space; the voxel holding the
surface point is evidence of occupancy. A voxel nobody looked through stays UNKNOWN.
"""
import numpy as np

UNKNOWN, FREE, OCC = 0, 1, 2


class Grid:
    def __init__(self, lo, hi, voxel):
        self.lo = np.asarray(lo, float)
        self.voxel = float(voxel)
        self.shape = tuple(np.ceil((np.asarray(hi, float) - self.lo) / self.voxel).astype(int))
        self.occ = np.zeros(self.shape, np.int32)   # observations that saw a surface here
        self.free = np.zeros(self.shape, np.int32)  # observations whose ray passed through here

    def centers(self, idx):
        return self.lo + (np.asarray(idx) + 0.5) * self.voxel

    def add(self, cams, pts, chunk=4000):
        cams = np.asarray(cams, float)
        pts = np.asarray(pts, float)
        dist = np.linalg.norm(pts - cams, axis=1)
        steps = int(np.ceil(dist.max() / (self.voxel * 0.5))) + 1
        t = np.linspace(0.0, 1.0, steps)
        for s in range(0, len(cams), chunk):
            c, p, d = cams[s:s + chunk], pts[s:s + chunk], dist[s:s + chunk]
            along = t[None, :] * d[:, None]
            valid = along < (d[:, None] - self.voxel)      # stop one voxel short of the surface
            samples = c[:, None, :] + t[None, :, None] * (p - c)[:, None, :]
            self._scatter(samples[valid], self.free)
            self._scatter(p, self.occ)

    def _scatter(self, pts, target):
        idx = np.floor((pts - self.lo) / self.voxel).astype(int)
        ok = np.all((idx >= 0) & (idx < self.shape), axis=1)
        flat = np.ravel_multi_index(idx[ok].T, self.shape)
        target += np.bincount(flat, minlength=target.size).reshape(target.shape)

    def states(self, min_votes=2):
        s = np.full(self.shape, UNKNOWN, np.uint8)
        s[(self.occ >= min_votes) & (self.occ > self.free)] = OCC
        s[(self.free >= min_votes) & (self.free > self.occ)] = FREE
        return s
