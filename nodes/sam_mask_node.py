"""ComfyUI wrapper for ``core.sam_mask.sam_mask`` (Segment Anything one-click mask)."""

from __future__ import annotations

import torch

from ..core import sam_mask as sam_core
from ..core.mask_preview import mask_preview
from ..core.sam_mask import MODEL_TYPES, sam_mask


class JHPixelProSAMMask:
    """Segment an object with Segment Anything from points, a face box or a rough mask."""

    CATEGORY = "ComfyUI-JH-PixelPro/mask"
    RETURN_TYPES = ("MASK", "FLOAT", "IMAGE")
    RETURN_NAMES = ("mask", "score", "preview")
    FUNCTION = "segment"

    @classmethod
    def INPUT_TYPES(cls: type) -> dict:  # noqa: N802 — ComfyUI node contract.
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image to cut the object out of."}),
                "model": (
                    list(MODEL_TYPES),
                    {
                        "default": "vit_b",
                        "tooltip": (
                            "SAM checkpoint from ComfyUI/models/sams/. vit_b (375 MB, ~3 GB VRAM) "
                            "is the balanced default, mobile_sam (39 MB, ~0.4 GB) is the fastest, "
                            "vit_l (1.2 GB) and vit_h (2.5 GB) are the most accurate. A missing "
                            "file is downloaded on first use. Only one model stays in VRAM."
                        ),
                    },
                ),
                "points_json": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": (
                            'Positive "clicks" on the object: [[x,y],[x,y]]. Pixels (0..size-1), '
                            "or 0-1 fractions when every value is within 0-1 and at least one is "
                            "a fraction such as 0.5 ([[1,1]] is pixel 1,1). Empty with no other "
                            "prompt = image centre."
                        ),
                    },
                ),
                "negative_points_json": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Points that must NOT be in the mask, same format.",
                    },
                ),
                "multimask": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "ON: SAM proposes 3 masks and the best predicted IoU is kept "
                            "(good for a single click). OFF: one mask, better with a box or "
                            "several points."
                        ),
                    },
                ),
                "mask_threshold": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -20.0,
                        "max": 20.0,
                        "step": 0.1,
                        "tooltip": (
                            "Logit cut-off. 0 = SAM default. Lower grows the mask, higher "
                            "shrinks it."
                        ),
                    },
                ),
            },
            "optional": {
                "bbox_json": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "forceInput": True,
                        "tooltip": (
                            "bbox_json from Face Detect / Face Detect Lite. Uses the entry "
                            "matching the batch index (or the first one)."
                        ),
                    },
                ),
                "mask_hint": (
                    "MASK",
                    {
                        "tooltip": (
                            "Rough mask from any node: its bounding box and a point inside it "
                            "become the prompt."
                        )
                    },
                ),
                "keep_model_loaded": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "ON: keep the SAM model in VRAM for the next run (faster). "
                            "OFF: free it right after this run (ComfyUI's Free memory "
                            "button cannot see it)."
                        ),
                    },
                ),
            },
        }

    def segment(
        self,
        image: torch.Tensor,
        model: str,
        points_json: str,
        negative_points_json: str,
        multimask: bool,
        mask_threshold: float,
        bbox_json: str = "",
        mask_hint: torch.Tensor | None = None,
        keep_model_loaded: bool = True,
    ) -> tuple[torch.Tensor, float, torch.Tensor]:
        with torch.no_grad():
            mask, scores = sam_mask(
                image,
                model_type=model,
                points_json=points_json,
                negative_points_json=negative_points_json,
                bbox_json=bbox_json or "",
                mask_hint=mask_hint,
                multimask=bool(multimask),
                mask_threshold=float(mask_threshold),
            )
            preview = mask_preview(image.clamp(0.0, 1.0), mask, mode="overlay", show_edges=True)
        if not keep_model_loaded:
            sam_core.unload()
        return (mask, float(scores.mean()), preview)
