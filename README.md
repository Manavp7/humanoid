# What Changed in the Room?

Compare two phone videos of the same indoor area and report what moved, appeared or disappeared,
in 3D, with a plain-English description of each change.

> **Honest status.** The change logic is validated on a generated synthetic room with known ground
> truth (precision and recall 1.00). The real-video path runs end to end on generated sample videos,
> but on those videos the sparse COLMAP point cloud is too thin to produce detections yet (see
> *Limitations*). No real phone videos were available while building this.

## How it works

1. **Frames.** Both videos are sampled at a few frames per second (ffmpeg).
2. **One shared 3D frame.** All frames from both recordings go into a single COLMAP reconstruction, so
   static walls and furniture put both recordings in the same coordinate system.
3. **Free space, not just points.** Every observation is a camera centre plus a 3D surface point. Voxels
   along the camera-to-surface ray count as *seen empty*. The surface voxel counts as *occupied*. Voxels
   nobody looked through stay *unknown*.
4. **Changes only where both recordings looked.** A change is a voxel that is occupied in one recording
   and seen empty in the other. Unknown never counts as a change, so an object hidden behind another one,
   or a corner one video never filmed, is reported as unverified instead of as removed.
5. **Edge noise is excluded.** Surface positions jitter by about a voxel between passes. Voxels next to a
   surface in the other recording are ignored.
6. **Objects and moves.** Changed voxels are grouped into objects. A removed and an added object of the
   same extent that are close together are paired as *moved*, with a distance and direction.
7. **Descriptions.** Each change gets a label and a sentence. Outputs are a 3D web viewer, a top-down
   PNG and a Markdown report.

## Run it

Requires Python 3.10+ and ffmpeg for the video path.

```bash
python -m venv venv && . venv/bin/activate
pip install -r requirements.txt

# 1. No videos needed: synthetic room with known changes, scored against ground truth
python -m roomdiff demo --out examples/demo

# 2. Sample phone-style videos (generated), then the real-video path
python -m roomdiff sample --out examples/sample_videos
python -m roomdiff videos --before examples/sample_videos/before.mp4 \
                          --after examples/sample_videos/after.mp4 --out output/videos

# Your own phone videos (use --scale if you know a real length, e.g. a 0.9 m table, to get metres)
python -m roomdiff videos --before my_before.mov --after my_after.mov --out output/mine --fps 3
```

Open `viewer.html` in a browser. It loads three.js from a CDN, so it needs internet access.

## Example outputs

Synthetic demo (`examples/demo/`):

```
- The chair-sized object moved 2.31 m (+2.15 m in x and +0.83 m in y). It was at (1.8, 2.8) and is now at (4.0, 3.6).
- A small object (bag or box) that was in the first recording is gone from (4.7, 0.7).
- A new small object (bag or box) appeared at (2.5, 4.1), about 0.60 m tall.
```

Evaluation against the ground truth in the generated scene: 3 of 3 changes found, 0 false positives.
Files: `report.md`, `changes.json`, `topdown.png`, `viewer.html`.

Sample inputs: `examples/sample_videos/before.mp4` and `after.mp4`. They were rendered from a synthetic room
with the same changes (the chair moves, the backpack is removed, a bin is added), with different camera walks.

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
- **Simple and CPU-only.** numpy, scipy and OpenCV, plus pycolmap for the video path. No GPU required.
- **Parameters were tuned on the synthetic scene.** Voxel size, minimum cluster size, the 0.4 shape budget
  and the 3 m move radius were chosen on that scene. Expect to retune them for real rooms.

## Limitations

- **Sparse reconstruction.** The video path uses COLMAP's sparse points, which are far fewer than the
  surface samples in the synthetic demo. On the generated sample videos the pipeline runs, but finds no
  changes, because there are too few occupied voxels. The next step is dense per-frame depth
  (Depth Anything V2 on CPU, scaled to the COLMAP points), which feeds the same occupancy code.
- **Scale.** COLMAP's scale is arbitrary. Without `--scale` distances are in reconstruction units.
- **Labels are shape guesses.** `roomdiff/describe.py` labels objects from footprint and height. CLIP
  on the frame crops would give real names; it is not wired in yet.
- **Not tested here:** the 3D viewer in a browser (the CDN was unreachable from the build sandbox), and
  real phone recordings.

## Layout

```
roomdiff/
  __main__.py          CLI: demo | sample | videos
  synth.py             synthetic room, ground truth, sample video rendering
  colmap_frontend.py   video -> frames -> joint COLMAP reconstruction -> observations
  voxels.py            occupancy grid with free-space carving
  changes.py           unknown / free / occupied diff, clustering, moved pairing
  describe.py          labels, sentences, ground-truth evaluation
  pipeline.py          analysis entry point and stats
  viewer.py            HTML viewer, top-down PNG, Markdown report
```
