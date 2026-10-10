# What Changed in the Room?

**Change only where both recordings proved free space. Unknown is never “removed.”**

That is the whole idea. Two phone videos of the same indoor area are put in one 3D frame; each voxel is occupied, seen empty, or never seen. A change is occupied in one recording and *seen empty* in the other. Occluded or unfilmed space stays **unverified** instead of becoming a false removal. Labels (shape or CLIP) only name what geometry already found.

> **Honest status.** Synthetic demo: precision and recall 1.00. Occlusion ablation: naive differencing makes **10** false-removal clusters; visibility-first makes **0**. Real-video path (COLMAP + Depth Anything V2 + CLIP) runs on sample clips and a VSCD env2 pair. No personal phone videos were filmed for this build.

## Why not YOLO / point differencing?

- **YOLO** knows fixed classes on single 2D frames. It cannot tell a moved object from a new viewpoint, and it has no “I never saw that corner” state.
- **Naive 3D differencing** (occupied before, missing after) treats unknown as gone — so furniture behind a sofa looks “removed.” See [`examples/ablation/`](examples/ablation/).
- **This system** decides *where* with occupancy + free-space rays; language only names crops. Moves need similar shape **and** a free-space path between locations (no teleporting through walls).

## How it works

1. **Frames.** Sample both videos (ffmpeg).
2. **Shared 3D frame.** Joint COLMAP reconstruction.
3. **Dense depth (videos).** Depth Anything V2, scale-aligned to COLMAP, filtered to the sparse cloud (torch in a subprocess).
4. **Free-space carving.** Camera→surface rays mark FREE; surfaces mark OCC; the rest stays UNKNOWN.
5. **Visibility-first diff.** Change ⇔ OCC vs FREE. UNKNOWN never counts.
6. **Ray-agreement confidence.** Each object gets a score from how many rays agreed on occ/free.
7. **Objects and moves.** Cluster voxels; pair removed+added by shape, distance, and a FREE-space path.
8. **Descriptions.** Shape labels or CLIP on crops. Outputs: 3D viewer, top-down PNG, Markdown report.

## Run it

Requires Python 3.10+ and ffmpeg for the video path. First `videos` run downloads HF weights.

```bash
python -m venv venv && . venv/bin/activate
pip install -r requirements.txt

# Synthetic room + ground truth (no HF models)
python -m roomdiff demo --out examples/demo

# Occlusion counterexample: naive false removals vs ours
python -m roomdiff ablation --out examples/ablation

# Sample phone-style videos, then dense + CLIP path
python -m roomdiff sample --out examples/sample_videos
python -m roomdiff videos --before examples/sample_videos/before.mp4 \
                          --after examples/sample_videos/after.mp4 --out examples/videos_dense

# Your own videos (--scale if you know a real length)
python -m roomdiff videos --before my_before.mov --after my_after.mov --out output/mine --fps 3
```

Serve a viewer folder (browsers block `file://` modules), e.g. `python -m http.server -d examples/demo` → `http://localhost:8000/viewer.html`.

## Example outputs

**Synthetic demo** (`examples/demo/`): chair moved, backpack removed, bin added — 3/3 vs GT.

**Ablation** (`examples/ablation/`): `naive_vs_ours.png` + `report.md` — naive **10** false-removal clusters, ours **0**.

**Sample videos** → `examples/videos_dense/` after you run `videos`.

**Real video (VSCD env2)** → `examples/real_vscd/output/` (source MP4s from [Hugging Face VSCD](https://huggingface.co/datasets/jiae1234/vscd); not vendored).

## Design choices

- **Geometry first, labels second.** Visibility separates real change from “could not see it.”
- **Shape before distance** when pairing moves (a nearby bin is not the chair).
- **Free-space path** required for “moved.”
- **Ray-agreement confidence** grades evidence without changing the decision rule.
- **Torch isolated from pycolmap** (subprocess workers) to avoid OpenMP crashes on macOS.
- Parameters tuned on the synthetic scene; retune for real rooms.

## Limitations

- Sample / real-video detections are noisier than the synthetic demo.
- COLMAP scale is arbitrary without `--scale`.
- Monocular depth bias remains after scale-shift alignment.
- CPU runtime for dense depth + CLIP is on the order of a minute per short pair once weights are cached.

## Layout

```
roomdiff/
  __main__.py          CLI: demo | sample | ablation | videos
  ablation.py          naive vs visibility-first occlusion counterexample
  synth.py             synthetic room, ground truth, sample videos
  colmap_frontend.py   frames → joint COLMAP → observations + crops
  depth.py / dense_worker.py   Depth Anything V2 (subprocess)
  clip_label.py / clip_worker.py   CLIP labels (subprocess)
  voxels.py            occupancy + free-space carving
  changes.py           visibility diff, confidence, free-space move paths
  describe.py          sentences + GT eval
  pipeline.py          analysis entry
  viewer.py            HTML / PNG / Markdown
  vendor/              three.js r158
```
