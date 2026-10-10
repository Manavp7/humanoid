"""Occlusion ablation: naive occupancy differencing vs visibility-first changes.

Naive baseline: occupied-before and not-occupied-after (UNKNOWN counts as "gone").
Ours: occupied-before and *seen empty* after. Unknown stays unverified.

This is the counterexample that sells the thesis — without new foundation models.
"""
from __future__ import annotations

import json
import pathlib

import cv2
import numpy as np
from scipy import ndimage

from . import synth
from .changes import find_changes
from .voxels import FREE, OCC, UNKNOWN, Grid


def _build_grids(before, after, voxel=0.06):
    allp = np.vstack([np.vstack([o["pts"], o["cams"]]) for o in (before, after)])
    lo, hi = allp.min(0) - 2 * voxel, allp.max(0) + 2 * voxel
    grids = []
    states = []
    for obs in (before, after):
        g = Grid(lo, hi, voxel)
        g.add(obs["cams"], obs["pts"])
        grids.append(g)
        states.append(g.states(min_votes=1))
    return grids, states, lo, hi


def naive_removed_mask(s_before, s_after):
    """Occupied before, anything-but-occupied after — treats unknown as removal."""
    return (s_before == OCC) & (s_after != OCC)


def ours_removed_mask(s_before, s_after):
    return (s_before == OCC) & (s_after == FREE)


def _cluster_count(mask, min_voxels=12):
    lab, n = ndimage.label(mask, structure=np.ones((3, 3, 3)))
    keep = 0
    for i in range(1, n + 1):
        if int(np.sum(lab == i)) >= min_voxels:
            keep += 1
    return keep, int(mask.sum())


def run_ablation(out_dir, voxel=0.06, min_voxels=12):
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    before, after, gt = synth.make_demo()
    grids, (s_b, s_a), lo, hi = _build_grids(before, after, voxel=voxel)

    naive_mask = naive_removed_mask(s_b, s_a)
    ours_mask = ours_removed_mask(s_b, s_a)
    # Edge-dilate like find_changes so the comparison is fair on surface jitter.
    near_a = ndimage.binary_dilation(s_a == OCC, structure=np.ones((3, 3, 3)))
    naive_mask = naive_mask & ~near_a
    ours_mask = ours_mask & ~near_a

    naive_clusters, naive_voxels = _cluster_count(naive_mask, min_voxels)
    ours_voxels = int(ours_mask.sum())
    unverified = int(np.sum(((s_b == OCC) & (s_a == UNKNOWN)) | ((s_b == UNKNOWN) & (s_a == OCC))))

    changes = find_changes(s_b, s_a, grids[0], grids[1], min_voxels=min_voxels)
    # Naive: every removal-like cluster beyond GT removals is a false removal (incl. unknown).
    # Ours: count reported "removed" objects after move-pairing vs GT.
    true_removed = sum(1 for g in gt if g["kind"] == "removed")
    naive_false = max(0, naive_clusters - true_removed)
    ours_removed = sum(c["kind"] == "removed" for c in changes)
    ours_false = max(0, ours_removed - true_removed)
    ours_clusters = ours_removed

    summary = dict(
        thesis="Change only where both recordings proved free space; unknown is never removed.",
        naive=dict(
            rule="occupied_before AND not_occupied_after (UNKNOWN counts as gone)",
            removed_clusters=naive_clusters,
            removed_voxels=naive_voxels,
            false_removal_clusters=naive_false,
        ),
        ours=dict(
            rule="occupied_before AND seen_empty_after (UNKNOWN unverified)",
            removed_clusters=ours_clusters,
            removed_voxels=ours_voxels,
            false_removal_clusters=ours_false,
            reported_changes=len(changes),
            kinds={k: sum(c["kind"] == k for c in changes) for k in ("moved", "added", "removed")},
        ),
        voxels_unverified=unverified,
        ground_truth_removed=true_removed,
    )
    (out / "naive_vs_ours.json").write_text(json.dumps(summary, indent=2))

    _topdown_pair(out / "naive_vs_ours.png", s_b, s_a, naive_mask, ours_mask, grids[0], lo, hi)
    _report(out / "report.md", summary)

    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")
    return summary


def _project_mask(mask, grid, lo, hi, px=40):
    """Collapse a 3D bool mask to a top-down occupancy count image."""
    w = max(1, int((hi[0] - lo[0]) * px))
    h = max(1, int((hi[1] - lo[1]) * px))
    img = np.zeros((h, w), np.float32)
    idx = np.argwhere(mask)
    if len(idx) == 0:
        return img
    pts = grid.centers(idx)
    for p in pts:
        x = int((p[0] - lo[0]) * px)
        y = int(h - (p[1] - lo[1]) * px)
        if 0 <= x < w and 0 <= y < h:
            img[y, x] += 1
    return img


def _topdown_pair(path, s_b, s_a, naive_mask, ours_mask, grid, lo, hi, px=40):
    static = (s_b == OCC) & (s_a == OCC)
    unverified = ((s_b == OCC) & (s_a == UNKNOWN)) | ((s_b == UNKNOWN) & (s_a == OCC))
    panels = []
    for title, rem in (("Naive: unknown = removed", naive_mask), ("Ours: unknown ≠ removed", ours_mask)):
        base = _project_mask(static, grid, lo, hi, px)
        rem_i = _project_mask(rem, grid, lo, hi, px)
        unk_i = _project_mask(unverified, grid, lo, hi, px)
        rgb = np.full((*base.shape, 3), 255, np.uint8)
        rgb[base > 0] = (180, 180, 180)
        rgb[unk_i > 0] = (220, 200, 120)   # unverified = amber
        rgb[rem_i > 0] = (70, 70, 220)     # "removed" = red (BGR)
        cv2.putText(rgb, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
        panels.append(rgb)
    gap = np.full((panels[0].shape[0], 12, 3), 245, np.uint8)
    canvas = np.concatenate([panels[0], gap, panels[1]], axis=1)
    cv2.imwrite(str(path), canvas)


def _report(path, summary):
    n, o = summary["naive"], summary["ours"]
    lines = [
        "# Occlusion ablation: why unknown ≠ removed",
        "",
        summary["thesis"],
        "",
        "## Naive occupancy differencing",
        f"- Rule: `{n['rule']}`.",
        f"- Removal clusters: **{n['removed_clusters']}** ({n['removed_voxels']} voxels).",
        f"- False-removal clusters (beyond ground-truth removals): **{n['false_removal_clusters']}**.",
        "",
        "## Ours (visibility-first)",
        f"- Rule: `{o['rule']}`.",
        f"- Removal clusters: **{o['removed_clusters']}** ({o['removed_voxels']} voxels).",
        f"- False-removal clusters: **{o['false_removal_clusters']}**.",
        f"- Reported changes: {o['reported_changes']} "
        f"(moved {o['kinds']['moved']}, added {o['kinds']['added']}, removed {o['kinds']['removed']}).",
        "",
        f"Unverified voxels (occupied on one side, unseen on the other): {summary['voxels_unverified']}.",
        "",
        "See `naive_vs_ours.png` (left = naive false removals in red; right = ours; amber = unverified).",
        "",
    ]
    path.write_text("\n".join(lines))
