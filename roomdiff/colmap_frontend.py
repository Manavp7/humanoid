"""Real-video front end: phone videos -> frames -> one joint COLMAP reconstruction -> observations.

Both videos go into the same reconstruction, so the static walls and furniture put them in one
coordinate frame. Each observation is (camera centre of a frame, triangulated 3D point) and is
tagged with the recording the frame came from. The scale is arbitrary unless --scale is given.
"""
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
    pycolmap.extract_features(db, image_dir)
    pycolmap.match_exhaustive(db)
    maps = pycolmap.incremental_mapping(db, image_dir, sparse)
    if not maps:
        raise RuntimeError("COLMAP could not register any frames; try more overlap or slower motion.")
    return max(maps.values(), key=lambda r: r.num_reg_images())


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
