# What Changed in the Room?

Compare two phone videos of the same indoor area and report what moved, appeared or disappeared,
in 3D, with a plain-English description of each change.

> **Honest status.** The change logic is validated on a generated synthetic room with known ground
> truth (precision and recall 1.00). The real-video path runs joint COLMAP, dense Depth Anything V2
> (scale-aligned to sparse points, filtered to the sparse cloud), and CLIP labels on change crops.
> On generated sample videos it produces detections end to end; scores are not as clean as the
> synthetic demo because monocular depth on those renders is noisy. No real phone videos were
> available while building this.

## How it works

1. **Frames.** Both videos are sampled at a few frames per second (ffmpeg).
2. **One shared 3D frame.** All frames from both recordings go into a single COLMAP reconstruction, so
   static walls and furniture put both recordings in the same coordinate system.
3. **Dense depth.** Depth Anything V2 Small predicts per-frame relative depth. Each frame is
   scale-and-shift aligned to the sparse COLMAP points in that view, unprojected, then kept only
   near the sparse cloud (outliers dropped). Torch runs in a subprocess so it does not share a
   process with pycolmap. Use `--sparse-only` to skip dense depth.
4. **Free space, not just points.** Every observation is a camera centre plus a 3D surface point. Voxels
   along the camera-to-surface ray count as *seen empty*. The surface voxel counts as *occupied*. Voxels
   nobody looked through stay *unknown*.
5. **Changes only where both recordings looked.** A change is a voxel that is occupied in one recording
   and seen empty in the other. Unknown never counts as a change, so an object hidden behind another one,
   or a corner one video never filmed, is reported as unverified instead of as removed.
6. **Edge noise is excluded.** Surface positions jitter by about a voxel between passes. Voxels next to a
   surface in the other recording are ignored.
7. **Objects and moves.** Changed voxels are grouped into objects. A removed and an added object of the
   same extent that are close together are paired as *moved*, with a distance and direction.
8. **Descriptions.** Each change is labelled with CLIP on a crop of the best registered frame (shape-based
   guesses if `--no-clip` or the crop fails). Outputs are a 3D web viewer, a top-down PNG and a
   Markdown report.

## Run it

Requires Python 3.10+ and ffmpeg for the video path. The first `videos` run downloads Depth Anything V2
and CLIP weights from Hugging Face (needs network once).

```bash
python -m venv venv && . venv/bin/activate
pip install -r requirements.txt

# 1. No videos / no HF models needed: synthetic room with known changes, scored against ground truth
python -m roomdiff demo --out examples/demo

# 2. Sample phone-style videos (generated), then the dense-depth + CLIP video path
python -m roomdiff sample --out examples/sample_videos
python -m roomdiff videos --before examples/sample_videos/before.mp4 \
                          --after examples/sample_videos/after.mp4 --out examples/videos_dense

# Sparse COLMAP only (debug; usually too thin to detect changes)
python -m roomdiff videos --before ... --after ... --out output/sparse --sparse-only --no-clip

# Your own phone videos (use --scale if you know a real length, e.g. a 0.9 m table, to get metres)
python -m roomdiff videos --before my_before.mov --after my_after.mov --out output/mine --fps 3
```

Open `viewer.html` in a browser. three.js is copied next to it in `vendor/`, so no internet access is needed. Browsers block module imports from `file://`, so serve the folder instead, e.g. `python -m http.server -d examples/demo` and open `http://localhost:8000/viewer.html`.

## Example outputs

Synthetic demo (`examples/demo/`):

```
- The chair-sized object moved 2.31 m (+2.15 m in x and +0.83 m in y). It was at (1.8, 2.8) and is now at (4.0, 3.6).
- A small object (bag or box) that was in the first recording is gone from (4.7, 0.7).
- A new small object (bag or box) appeared at (2.5, 4.1), about 0.60 m tall.
```

Evaluation against the ground truth in the generated scene: 3 of 3 changes found, 0 false positives. Files: `report.md`, `changes.json`, `topdown.png`, `viewer.html`.

Sample inputs: `examples/sample_videos/before.mp4` and `after.mp4`. Dense-depth outputs land in
`examples/videos_dense/` after you run the `videos` command above.

## Design choices and tradeoffs

- **Geometry first, labels second.** Where something changed is decided from 3D occupancy and visibility.
  A language or vision model only names what changed. That keeps the detector honest about viewpoint and
  occlusion. A YOLO-style detector only knows fixed classes and works on single 2D frames, so it cannot
  tell a moved object from one seen from a new angle on its own.
- **Visibility is the core idea.** The "unknown" state is what separates a real change from "I could not
  see it". This matters most in rooms that were filmed from different sides.
- **Shape before distance when pairing moves.** The nearest object is not always the same object. An
  added bin near the old chair position was wrongly paired with the chair until matching required similar
  extent first.
- **Dense depth feeds the same carver.** Depth Anything V2 is relative; each frame is fit to COLMAP with a
  scale and shift in disparity space. Points far from the sparse cloud are dropped so monocular outliers
  do not flood the grid.
- **Torch isolated from pycolmap.** On macOS the two native stacks can abort if loaded together, so depth
  and CLIP run as `python -m roomdiff.dense_worker` / `clip_worker` subprocesses.
- **Parameters were tuned on the synthetic scene.** Voxel size, minimum cluster size, the shape budget
  and the move radius were chosen there; the video path uses slightly looser pairing and a finer voxel
  relative to the scene. Expect to retune for real rooms.

## Limitations

- **Sample-video detections are imperfect.** Dense depth on the generated clips finds real occupancy
  differences but is not scored against ground truth the way the synthetic demo is.
- **CPU runtime.** Dense depth + CLIP on CPU takes on the order of a minute per sample-video pair after
  weights are cached. CUDA is used when available inside the workers.
- **Scale.** COLMAP's scale is arbitrary. Without `--scale` distances are in reconstruction units.
- **Not tested here:** real phone recordings. The 3D viewer was checked in headless Chromium, with three.js
  vendored (r158, MIT licence in `roomdiff/vendor/`).

## Layout

```
roomdiff/
  __main__.py          CLI: demo | sample | videos
  synth.py             synthetic room, ground truth, sample video rendering
  colmap_frontend.py   video -> frames -> joint COLMAP -> observations + crops
  depth.py             Depth Anything V2 + COLMAP scale-shift + unprojection
  dense_worker.py      subprocess entry for dense depth (no pycolmap)
  clip_label.py        CLIP labels on change crops
  clip_worker.py       subprocess entry for CLIP (no pycolmap)
  voxels.py            occupancy grid with free-space carving
  changes.py           unknown / free / occupied diff, clustering, moved pairing
  describe.py          labels, sentences, ground-truth evaluation
  pipeline.py          analysis entry point and stats
  viewer.py            HTML viewer, top-down PNG, Markdown report
  vendor/              three.js r158 (module build and OrbitControls) copied into each output
```
