"""Real-video front end: phone videos -> frames -> one joint COLMAP reconstruction -> observations.

Both videos go into the same reconstruction, so the static walls and furniture put them in one
coordinate frame. Each observation is (camera centre of a frame, triangulated 3D point) and is
tagged with the recording the frame came from. The scale is arbitrary unless --scale is given.

Dense depth (Depth Anything V2) fills the same (cams, pts) schema after scale-aligning each
frame's relative depth to the sparse COLMAP points visible in that frame.
"""
from __future__ import annotations

import pathlib
import subprocess

import numpy as np


def extract_frames(video, out_dir, tag, fps=2.0):
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(video), "-vf", f"fps={fps}",
                    "-q:v", "2", str(out / f"{tag}_%05d.jpg")], check=True)


def reconstruct(image_dir, work_dir):
    import pycolmap

    work = pathlib.Path(work_dir)
    sparse = work / "sparse"
    sparse.mkdir(parents=True, exist_ok=True)
    db = work / "database.db"
    # Shared intrinsics help short phone clips of the same room; SIFT on CPU is enough here.
    reader = pycolmap.ImageReaderOptions()
    # Prefer one camera model for both recordings when FOV is similar.
    pycolmap.extract_features(db, image_dir, camera_mode=pycolmap.CameraMode.SINGLE,
                              reader_options=reader)
    pycolmap.match_exhaustive(db)
    opts = pycolmap.IncrementalPipelineOptions()
    opts.min_model_size = 3
    opts.init_num_trials = 200
    opts.mapper.init_min_num_inliers = 30
    opts.mapper.init_min_tri_angle = 4.0
    opts.mapper.abs_pose_min_num_inliers = 15
    opts.mapper.filter_min_tri_angle = 0.5
    maps = pycolmap.incremental_mapping(db, image_dir, sparse, options=opts)
    if not maps:
        raise RuntimeError("COLMAP could not register any frames; try more overlap or slower motion.")
    rec = max(maps.values(), key=lambda r: r.num_reg_images())
    if rec.num_reg_images() < 4:
        raise RuntimeError(
            f"COLMAP only registered {rec.num_reg_images()} frames; "
            "try a higher --fps, slower motion, or more textured surfaces."
        )
    return rec


def observations(rec, tags=("A", "B"), min_track=3):
    per = {t: dict(cams=[], pts=[]) for t in tags}
    for p3 in rec.points3D.values():
        if len(p3.track.elements) < min_track:   # short tracks are the usual source of wild points
            continue
        for el in p3.track.elements:
            img = rec.images[el.image_id]
            tag = img.name.split("_")[0]
            if tag in per:
                per[tag]["cams"].append(np.asarray(img.projection_center()))
                per[tag]["pts"].append(np.asarray(p3.xyz))
    missing = [t for t in tags if not per[t]["pts"]]
    if missing:
        raise RuntimeError(f"No frames from recording(s) {missing} were registered in the reconstruction.")
    return {t: dict(cams=np.array(v["cams"]), pts=np.array(v["pts"])) for t, v in per.items()}


def _cam_from_world_matrix(img):
    """3x4 world-to-camera matrix [R|t] from a pycolmap Image."""
    cfw = img.cam_from_world() if callable(getattr(img, "cam_from_world", None)) else img.cam_from_world
    mat = getattr(cfw, "matrix", None)
    if mat is not None:
        M = np.asarray(mat() if callable(mat) else mat, dtype=np.float64)
        if M.shape == (3, 4):
            return M
        if M.shape == (4, 4):
            return M[:3, :]
    rot = cfw.rotation
    R = np.asarray(rot.matrix() if callable(getattr(rot, "matrix", None)) else rot.matrix(), dtype=np.float64)
    t = np.asarray(cfw.translation, dtype=np.float64).reshape(3)
    return np.hstack([R, t[:, None]])


def _camera_K(cam):
    """3x3 pinhole matrix from a pycolmap Camera (ignores distortion for unprojection)."""
    if hasattr(cam, "calibration_matrix"):
        return np.asarray(cam.calibration_matrix(), dtype=np.float64)
    params = np.asarray(cam.params, dtype=np.float64)
    model = str(cam.model).upper() if hasattr(cam, "model") else ""
    # SIMPLE_PINHOLE: f, cx, cy | PINHOLE: fx, fy, cx, cy | SIMPLE_RADIAL: f, cx, cy, k
    if "SIMPLE" in model or len(params) == 3 or (len(params) == 4 and "RADIAL" in model and "OPENCV" not in model):
        f, cx, cy = params[0], params[1], params[2]
        return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
    fx, fy, cx, cy = params[0], params[1], params[2], params[3]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def _world_to_cam(xyz, cam_from_world):
    R = cam_from_world[:3, :3]
    t = cam_from_world[:3, 3]
    return (R @ np.asarray(xyz, dtype=np.float64).T).T + t


def _project(xyz_cam, K):
    z = xyz_cam[:, 2]
    u = K[0, 0] * xyz_cam[:, 0] / z + K[0, 2]
    v = K[1, 1] * xyz_cam[:, 1] / z + K[1, 2]
    return np.column_stack([u, v]), z


def frame_index(rec, image_dir, tags=("A", "B"), min_track=3):
    """Build per-registered-image metadata for dense depth and CLIP crops."""
    image_dir = pathlib.Path(image_dir)
    # Collect sparse 3D points observed in each image (for depth scale alignment).
    by_image = {}
    for p3 in rec.points3D.values():
        if len(p3.track.elements) < min_track:
            continue
        xyz = np.asarray(p3.xyz, dtype=np.float64)
        for el in p3.track.elements:
            by_image.setdefault(el.image_id, []).append(xyz)

    frames = []
    for image_id, img in rec.images.items():
        if hasattr(img, "has_pose") and not img.has_pose:
            continue
        tag = img.name.split("_")[0]
        if tag not in tags:
            continue
        cam = rec.cameras[img.camera_id]
        K = _camera_K(cam)
        cfw = _cam_from_world_matrix(img)
        anchors = by_image.get(image_id, [])
        if not anchors:
            continue
        xyz = np.asarray(anchors, dtype=np.float64)
        xyz_cam = _world_to_cam(xyz, cfw)
        uv, z = _project(xyz_cam, K)
        in_front = z > 1e-3
        uv, z = uv[in_front], z[in_front]
        w, h = int(cam.width), int(cam.height)
        in_img = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        uv, z = uv[in_img], z[in_img]
        frames.append(dict(
            image_id=image_id,
            name=img.name,
            tag=tag,
            path=str(image_dir / img.name),
            width=w,
            height=h,
            K=K,
            cam_from_world=cfw,
            center=np.asarray(img.projection_center(), dtype=np.float64),
            anchors_uv=uv,
            anchors_z=z,
        ))
    return frames


def dense_observations(rec, image_dir, tags=("A", "B"), stride=8, min_track=3):
    """Dense (cams, pts) from Depth Anything V2, scale-aligned to COLMAP per frame."""
    from .depth import dense_observations_from_frames

    frames = frame_index(rec, image_dir, tags=tags, min_track=min_track)
    if not frames:
        raise RuntimeError("No registered frames with sparse anchors for dense depth.")
    result = dense_observations_from_frames(frames, image_dir, stride=stride)
    meta = result.pop("_meta", {})
    missing = [t for t in tags if t not in result or len(result[t]["pts"]) == 0]
    if missing:
        # Fall back: merge sparse for missing tags so analyze can still run.
        sparse = observations(rec, tags=tags, min_track=min_track)
        for t in missing:
            result[t] = sparse[t]
        meta["sparse_fallback_tags"] = missing
    result["_meta"] = meta
    result["_frames"] = frames
    return result


def project_box_to_crop(box_lo, box_hi, frames, tag=None, padding=0.15, min_side=16):
    """Pick the frame (optionally filtered by tag) where the AABB projects largest; return a crop.

    Returns (PIL.Image, frame_dict) or (None, None).
    """
    from PIL import Image

    lo = np.asarray(box_lo, dtype=np.float64)
    hi = np.asarray(box_hi, dtype=np.float64)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])

    best = None
    for fr in frames:
        if tag is not None and fr["tag"] != tag:
            continue
        xyz_cam = _world_to_cam(corners, fr["cam_from_world"])
        if np.any(xyz_cam[:, 2] <= 1e-3):
            continue
        uv, _ = _project(xyz_cam, fr["K"])
        u0, v0 = uv[:, 0].min(), uv[:, 1].min()
        u1, v1 = uv[:, 0].max(), uv[:, 1].max()
        # Clip to image
        u0c, v0c = max(0.0, u0), max(0.0, v0)
        u1c, v1c = min(fr["width"] - 1.0, u1), min(fr["height"] - 1.0, v1)
        if u1c - u0c < min_side or v1c - v0c < min_side:
            continue
        area = (u1c - u0c) * (v1c - v0c)
        # Prefer boxes mostly inside the frame
        full_area = max((u1 - u0) * (v1 - v0), 1.0)
        score = area * (area / full_area)
        if best is None or score > best[0]:
            pad_u = padding * (u1c - u0c)
            pad_v = padding * (v1c - v0c)
            best = (score, fr, (
                max(0, int(u0c - pad_u)),
                max(0, int(v0c - pad_v)),
                min(fr["width"], int(u1c + pad_u)),
                min(fr["height"], int(v1c + pad_v)),
            ))
    if best is None:
        return None, None
    _, fr, (x0, y0, x1, y1) = best
    if x1 - x0 < min_side or y1 - y0 < min_side:
        return None, None
    img = Image.open(fr["path"]).convert("RGB")
    return img.crop((x0, y0, x1, y1)), fr
