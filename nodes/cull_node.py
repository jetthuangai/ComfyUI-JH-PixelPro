"""ComfyUI wrapper for the culling score (pick the keepers from a batch)."""

from __future__ import annotations

import json

import torch

from ..core.cull import cull_scores, select


class JHPixelProCullScore:
    """Score a batch for sharpness, exposure, closed eyes and duplicates; keep the best."""

    CATEGORY = "ComfyUI-JH-PixelPro/qa"
    RETURN_TYPES = ("IMAGE", "STRING", "INT")
    RETURN_NAMES = ("selected", "report", "kept")
    OUTPUT_TOOLTIPS = (
        "The kept photos, best first (at least one).",
        "JSON: one entry per input photo with score, sharpness, exposure, eyes_closed, "
        "faces and duplicate_of.",
        "How many photos were kept.",
    )
    FUNCTION = "cull"

    @classmethod
    def INPUT_TYPES(cls: type) -> dict:  # noqa: N802
        return {
            "required": {
                "images": (
                    "IMAGE",
                    {"tooltip": "A batch of photos of the same size (e.g. Load Image Batch)."},
                ),
                "top_k": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 1000,
                        "step": 1,
                        "tooltip": "Keep at most this many; 0 = every photo that passes.",
                    },
                ),
                "min_score": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Drop photos scoring below this (0..1).",
                    },
                ),
                "check_eyes": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Penalise closed eyes (needs MediaPipe; skipped if missing).",
                    },
                ),
            },
        }

    def cull(
        self, images: torch.Tensor, top_k: int, min_score: float, check_eyes: bool
    ) -> tuple[torch.Tensor, str, int]:
        with torch.no_grad():
            rows = cull_scores(images, check_eyes=bool(check_eyes))
        keep = select(rows, int(top_k), float(min_score))
        if not keep:  # never return an empty batch: fall back to the best photo
            keep = [max(rows, key=lambda r: r["score"])["index"]]
        for r in rows:
            r["keep"] = r["index"] in keep
        selected = images[torch.tensor(keep, dtype=torch.long)]
        return selected, json.dumps(rows, indent=1), len(keep)
