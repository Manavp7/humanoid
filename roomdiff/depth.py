"""Dense per-frame depth via Depth Anything V2, scaled to COLMAP sparse points.

Produces the same observation schema as sparse COLMAP tracks: (camera centre, surface point)
pairs in the reconstruction frame, so the voxel free-space carver needs no changes.
"""
from __future__ import annotations

import pathlib

import numpy as np

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
_MIN_ANCHORS = 20


def _device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model(device):
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID)
    model.to(device)
    model.eval()
    return processor, model


def predict_disparity(image_rgb, processor, model, device):
    """Return a float32 HxW map where larger values mean closer (relative inverse depth)."""
    import torch
    from PIL import Image

    pil = Image.fromarray(image_rgb)
    inputs = processor(images=pil, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        pred = model(**inputs).predicted_depth
    pred = torch.nn.functional.interpolate(
        pred.unsqueeze(1), size=pil.size[::-1], mode="bicubic", align_corners=False
    )
    return pred.squeeze().float().cpu().numpy()


def fit_scale_shift(disp, uv, z_colmap, min_anchors=_MIN_ANCHORS):
    """Fit s, t so s * disp(u,v) + t ≈ 1/z_colmap. Returns (s, t) or None."""
    h, w = disp.shape
    u = np.round(uv[:, 0]).astype(int)
    v = np.round(uv[:, 1]).astype(int)
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h) & (z_colmap > 1e-4)
    if ok.sum() < min_anchors:
        return None
    d = disp[v[ok], u[ok]].astype(np.float64)
    inv_z = (1.0 / z_colmap[ok]).astype(np.float64)
    # Robust-ish: drop extreme disparity percentiles before least squares.
    lo, hi = np.percentile(d, [5, 95])
    keep = (d >= lo) & (d <= hi)
    if keep.sum() < min_anchors:
        return None
    A = np.column_stack([d[keep], np.ones(keep.sum())])
    sol, _, _, _ = np.linalg.lstsq(A, inv_z[keep], rcond=None)
    s, t = float(sol[0]), float(sol[1])
    if not np.isfinite(s) or abs(s) < 1e-12:
        return None
    return s, t


def depth_from_disparity(disp, s, t):
    inv = s * disp.astype(np.float64) + t
    z = np.full(disp.shape, np.nan, np.float64)
    good = inv > 1e-6
    z[good] = 1.0 / inv[good]
    return z


def unproject_grid(z, K, cam_from_world, stride=8, z_max=None):
    """Unproject a strided pixel grid with depths z into world points; return (N,3)."""
    h, w = z.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    vs = np.arange(0, h, stride)
    us = np.arange(0, w, stride)
    uu, vv = np.meshgrid(us, vs)
    uu = uu.ravel()
    vv = vv.ravel()
    zz = z[vv, uu]
    ok = np.isfinite(zz) & (zz > 1e-3)
    if z_max is not None:
        ok &= zz < z_max
    if not np.any(ok):
        return np.zeros((0, 3))
    uu, vv, zz = uu[ok].astype(np.float64), vv[ok].astype(np.float64), zz[ok]
    x = (uu - cx) / fx * zz
    y = (vv - cy) / fy * zz
    cam_pts = np.column_stack([x, y, zz])  # COLMAP camera frame
    # world = R^T * (x_cam - t) for cam_from_world: x_cam = R x_world + t
    R = np.asarray(cam_from_world[:3, :3], dtype=np.float64)
    t = np.asarray(cam_from_world[:3, 3], dtype=np.float64)
    world = (R.T @ (cam_pts - t).T).T
    return world


def dense_observations_from_frames(frames, image_dir, stride=8, min_anchors=_MIN_ANCHORS):
    """frames: list of dicts from colmap_frontend.frame_index(rec).

    Each frame needs: name, tag, path (or resolvable via image_dir), K (3x3),
    cam_from_world (3x4 or 4x4), center (3,), anchors_uv (M,2), anchors_z (M,).
    """
    import cv2

    device = _device()
    processor, model = _load_model(device)
    image_dir = pathlib.Path(image_dir)
    per = {}
    skipped = 0
    for fr in frames:
        path = pathlib.Path(fr.get("path") or (image_dir / fr["name"]))
        if not path.is_file():
            skipped += 1
            continue
        bgr = cv2.imread(str(path))
        if bgr is None:
            skipped += 1
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        disp = predict_disparity(rgb, processor, model, device)
        fit = fit_scale_shift(disp, fr["anchors_uv"], fr["anchors_z"], min_anchors=min_anchors)
        if fit is None:
            skipped += 1
            continue
        s, t = fit
        z = depth_from_disparity(disp, s, t)
        z_max = float(np.percentile(fr["anchors_z"], 95) * 2.5) if len(fr["anchors_z"]) else None
        pts = unproject_grid(z, fr["K"], fr["cam_from_world"], stride=stride, z_max=z_max)
        if len(pts) == 0:
            skipped += 1
            continue
        center = np.asarray(fr["center"], dtype=np.float64)
        cams = np.repeat(center[None, :], len(pts), axis=0)
        tag = fr["tag"]
        bucket = per.setdefault(tag, {"cams": [], "pts": []})
        bucket["cams"].append(cams)
        bucket["pts"].append(pts)

    if not per:
        raise RuntimeError(
            "Dense depth produced no observations; try --sparse-only or more overlapping frames."
        )
    out = {}
    for tag, v in per.items():
        out[tag] = dict(cams=np.vstack(v["cams"]), pts=np.vstack(v["pts"]))
    out["_meta"] = dict(skipped_frames=skipped, used_frames=sum(1 for t in per for _ in [1]))
    return out


def filter_near_sparse(dense_cams, dense_pts, sparse_pts, max_dist):
    """Keep dense samples whose surface point is near the sparse cloud (drops depth outliers)."""
    from scipy.spatial import cKDTree
    if len(dense_pts) == 0 or len(sparse_pts) == 0:
        return dense_cams[:0], dense_pts[:0]
    tree = cKDTree(sparse_pts)
    dist, _ = tree.query(dense_pts, k=1)
    keep = dist <= max_dist
    return dense_cams[keep], dense_pts[keep]
