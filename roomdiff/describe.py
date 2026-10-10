"""Plain-language labels and descriptions for detected changes.

Default labels are shape-based guesses (footprint and height). The video path can pass a
CLIP-backed label_fn that names crops of the registered frames instead.
"""
import numpy as np


def label_for(size):
    footprint, height = float(max(size[0], size[1])), float(size[2])
    if footprint > 1.0:
        return "large object (table- or sofa-sized)"
    if height > 1.2:
        return "tall object (cabinet- or shelf-sized)"
    if footprint < 0.45 and height < 0.6:
        return "small object (bag or box)"
    if footprint < 0.45:
        return "narrow object (bin or stand)"
    if height >= 0.6:
        return "chair-sized object"
    return "low box"


def _article(label):
    return "An" if label[:1].lower() in "aeiou" else "A"


def _shift(a, b):
    d = np.asarray(b) - np.asarray(a)
    return f"{d[0]:+.2f} m in x and {d[1]:+.2f} m in y"


def describe(changes, unit="m", label_fn=None):
    lines = []
    for c in changes:
        label = label_fn(c) if label_fn is not None else label_for(c["size"])
        c["label"] = label
        at = c["at"]
        if c["kind"] == "moved":
            c["text"] = (f"The {label} moved {c['distance']:.2f} {unit} ({_shift(at, c['to'])}). "
                         f"It was at ({at[0]:.1f}, {at[1]:.1f}) and is now at "
                         f"({c['to'][0]:.1f}, {c['to'][1]:.1f}).")
        elif c["kind"] == "added":
            c["text"] = (f"A new {label} appeared at ({at[0]:.1f}, {at[1]:.1f}), "
                         f"about {c['size'][2]:.2f} {unit} tall.")
        else:
            c["text"] = (f"{_article(label)} {label} that was in the first recording is gone from "
                         f"({at[0]:.1f}, {at[1]:.1f}).")
        lines.append(c["text"])
    return lines


def evaluate(changes, gt, tol=0.5):
    """Match detections to ground truth by kind and position. gt items: kind, at, optional to."""
    matched, tp = set(), 0
    for g in gt:
        for k, c in enumerate(changes):
            if k in matched or c["kind"] != g["kind"]:
                continue
            if np.linalg.norm(np.subtract(c["at"], g["at"])) > tol:
                continue
            if g["kind"] == "moved" and np.linalg.norm(np.subtract(c["to"], g["to"])) > tol:
                continue
            matched.add(k)
            tp += 1
            break
    fp = len(changes) - tp
    fn = len(gt) - tp
    return dict(true_positive=tp, false_positive=fp, false_negative=fn,
                precision=tp / max(tp + fp, 1), recall=tp / max(len(gt), 1), tolerance=tol)
