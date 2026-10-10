"""Synthetic two-pass recording of a small room, with known ground truth.

Camera rays are cast analytically against the room box and axis-aligned object boxes, then
sensor-like noise is added. Each pass uses different camera poses, so viewpoints differ.
"""
import pathlib

import numpy as np

ROOM = np.array([6.0, 5.0, 2.6])
FOCAL, W, H = 70.0, 96, 72

STATIC = {
    "cabinet": ([0.1, 3.0, 0.0], [0.6, 4.5, 1.9]),
    "table": ([1.0, 1.0, 0.0], [2.2, 1.8, 0.75]),
    "floor box": ([2.8, 0.3, 0.0], [3.3, 0.8, 0.5]),
}
BEFORE = {"chair": ([1.5, 2.6, 0.0], [2.0, 3.1, 0.9]), "backpack": ([4.6, 0.5, 0.0], [4.95, 0.75, 0.45])}
AFTER = {"chair": ([3.9, 3.4, 0.0], [4.4, 3.9, 0.9]), "bin": ([2.3, 4.0, 0.0], [2.7, 4.4, 0.7])}


def _boxes(objs):
    return [(np.array(lo, float), np.array(hi, float)) for lo, hi in list(STATIC.values()) + list(objs.values())]


def _slab(o, d, lo, hi):
    inv = 1.0 / np.where(np.abs(d) < 1e-9, 1e-9, d)
    t1, t2 = (lo - o) * inv, (hi - o) * inv
    return np.minimum(t1, t2).max(1), np.maximum(t1, t2).min(1)


def _cast(o, d, boxes):
    _, t_hit = _slab(o, d, np.zeros(3), ROOM)       # camera is inside the room: exit = wall/floor/ceiling
    for lo, hi in boxes:
        tmin, tmax = _slab(o, d, lo, hi)
        ok = (tmin > 0) & (tmin <= tmax) & (tmin < t_hit)
        t_hit[ok] = tmin[ok]
    return t_hit


def _camera_poses(rng, n):
    poses = []
    for _ in range(n):
        pos = rng.uniform([0.6, 0.6, 1.3], [ROOM[0] - 0.6, ROOM[1] - 0.6, 1.6])
        target = rng.uniform([1.0, 1.0, 0.4], [ROOM[0] - 1.0, ROOM[1] - 1.0, 1.2])
        poses.append((pos, target))
    return poses


def _render(pos, target, boxes, step=1):
    fwd = target - pos
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    uu, vv = np.meshgrid(np.arange(0, W, step) + 0.5, np.arange(0, H, step) + 0.5)
    x = (uu.ravel() - W / 2) / FOCAL
    y = -(vv.ravel() - H / 2) / FOCAL
    d = fwd[None] + x[:, None] * right[None] + y[:, None] * up[None]
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    t = _cast(pos, d, boxes)
    return pos + t[:, None] * d


def record(objs, seed, n_cams=24):
    rng = np.random.default_rng(seed)
    boxes = _boxes(objs)
    cams, pts = [], []
    for pos, target in _camera_poses(rng, n_cams):
        p = _render(pos, target, boxes)
        p = p + rng.normal(0, 0.01, p.shape)
        pts.append(p)
        cams.append(np.repeat(pos[None], len(p), 0))
    return dict(cams=np.vstack(cams), pts=np.vstack(pts))


def make_demo():
    before = record({**BEFORE}, seed=1)
    after = record({**AFTER}, seed=2)
    gt = [
        dict(kind="moved", at=_center(BEFORE["chair"]), to=_center(AFTER["chair"])),
        dict(kind="removed", at=_center(BEFORE["backpack"])),
        dict(kind="added", at=_center(AFTER["bin"])),
    ]
    return before, after, gt


def _center(box):
    lo, hi = box
    return (np.array(lo) + np.array(hi)) / 2


def _texture(p, block=0.03):
    """Random per-block colour on every surface, so the rendered frames have features for COLMAP."""
    cell = np.floor(p / block)
    h = np.sin(cell[:, 0] * 127.1 + cell[:, 1] * 311.7 + cell[:, 2] * 74.7) * 43758.5453
    rgb = h[:, None] * np.array([1.0, 1.7, 2.3])[None]
    return np.clip(rgb - np.floor(rgb), 0, 1)


def _trajectory(center, radius, z, phase, n, target_z):
    ang = phase + np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = np.stack([center[0] + radius * np.cos(ang), center[1] + radius * np.sin(ang),
                    np.full(n, z)], 1)
    target = np.repeat(np.array([center[0], center[1], target_z])[None], n, 0)
    return pos, target


def write_videos(out_dir, fps=10, n_frames=80, size=(640, 480), focal=500.0):
    """Render two phone-like walks around the room, with the object changes between them."""
    import cv2
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    w, h = size
    walks = {
        "before": (BEFORE, _trajectory([3.0, 2.5, 0], 1.9, 1.5, 0.0, n_frames, 0.6)),
        "after": (AFTER, _trajectory([3.0, 2.5, 0], 1.6, 1.3, 1.1, n_frames, 0.9)),
    }
    for name, (objs, (pos, target)) in walks.items():
        boxes = _boxes(objs)
        writer = cv2.VideoWriter(str(out / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        uu, vv = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
        for k in range(n_frames):
            fwd = target[k] - pos[k]
            fwd /= np.linalg.norm(fwd)
            right = np.cross(fwd, [0, 0, 1.0])
            right /= np.linalg.norm(right)
            up = np.cross(right, fwd)
            x = (uu.ravel() - w / 2) / focal
            y = -(vv.ravel() - h / 2) / focal
            d = fwd[None] + x[:, None] * right[None] + y[:, None] * up[None]
            d /= np.linalg.norm(d, axis=1, keepdims=True)
            hit = pos[k] + _cast(pos[k], d, boxes)[:, None] * d
            rgb = (_texture(hit) * 255).astype(np.uint8).reshape(h, w, 3)
            writer.write(rgb[:, :, ::-1])
        writer.release()
    return out
