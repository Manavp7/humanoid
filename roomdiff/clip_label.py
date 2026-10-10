"""Open-vocabulary labels for change crops via CLIP.

Geometry still decides *where* something changed; CLIP only names the crop.
"""
from __future__ import annotations

PROMPTS = [
    "a photo of a chair in a room",
    "a photo of a backpack on the floor",
    "a photo of a bag on the floor",
    "a photo of a cardboard box",
    "a photo of a storage bin",
    "a photo of a trash bin",
    "a photo of a table",
    "a photo of a lamp",
    "a photo of a plant pot",
    "a photo of a sofa",
    "a photo of a stool",
    "a photo of miscellaneous clutter",
]

# Shorter names for report sentences
_SHORT = {
    "a photo of a chair in a room": "chair",
    "a photo of a backpack on the floor": "backpack",
    "a photo of a bag on the floor": "bag",
    "a photo of a cardboard box": "box",
    "a photo of a storage bin": "bin",
    "a photo of a trash bin": "bin",
    "a photo of a table": "table",
    "a photo of a lamp": "lamp",
    "a photo of a plant pot": "plant",
    "a photo of a sofa": "sofa",
    "a photo of a stool": "stool",
    "a photo of miscellaneous clutter": "object",
}

_MODEL_ID = "openai/clip-vit-base-patch32"
_STATE = None


def _device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load():
    global _STATE
    if _STATE is not None:
        return _STATE
    import torch
    from transformers import CLIPModel, CLIPProcessor

    device = _device()
    processor = CLIPProcessor.from_pretrained(_MODEL_ID)
    model = CLIPModel.from_pretrained(_MODEL_ID)
    model.to(device)
    model.eval()
    _STATE = (processor, model, device)
    return _STATE


def label_crop(image, prompts=None):
    """Return a short noun phrase for a PIL RGB crop."""
    import torch

    prompts = prompts or PROMPTS
    processor, model, device = _load()
    inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
        probs = out.logits_per_image.softmax(dim=1)[0]
        idx = int(torch.argmax(probs).item())
    return _SHORT.get(prompts[idx], prompts[idx].removeprefix("a ").removeprefix("an "))


def make_label_fn(frames):
    """Build a label_fn(change_dict) -> str using AABB crops from registered frames."""
    from .colmap_frontend import project_box_to_crop
    from .describe import label_for

    def label_fn(c):
        kind = c["kind"]
        if kind == "removed":
            crop, _ = project_box_to_crop(c["lo"], c["hi"], frames, tag="A")
        elif kind == "added":
            crop, _ = project_box_to_crop(c["lo"], c["hi"], frames, tag="B")
        else:  # moved — prefer after location
            crop, _ = project_box_to_crop(c["to_lo"], c["to_hi"], frames, tag="B")
            if crop is None:
                crop, _ = project_box_to_crop(c["lo"], c["hi"], frames, tag="A")
        if crop is None:
            return label_for(c["size"])
        try:
            return label_crop(crop)
        except Exception:
            return label_for(c["size"])

    return label_fn
