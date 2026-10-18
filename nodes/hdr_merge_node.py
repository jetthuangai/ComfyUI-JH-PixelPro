"""ComfyUI wrapper for HDR merge (exposure fusion of a bracketed batch)."""

from __future__ import annotations

import torch

from ..core.hdr_merge import hdr_merge


class JHPixelProHDRMerge:
    """Fuse bracketed exposures (a batch) into one image with detail in shadows and sky."""

    CATEGORY = "ComfyUI-JH-PixelPro/color"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "merge"

    @classmethod
    def INPUT_TYPES(cls: type) -> dict:  # noqa: N802
        return {
            "required": {
                "images": (
                    "IMAGE",
                    {
                        "tooltip": "The bracketed shots as one batch (2 or more, any order). "
                        "Use Batch Images or Load Image Batch to combine them."
                    },
                ),
                "align": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Line up hand-held shots (small shifts)."},
                ),
                "deghost": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Where something moved between shots (people, leaves), "
                        "use the middle exposure so nothing appears twice.",
                    },
                ),
            },
        }

    def merge(self, images: torch.Tensor, align: bool, deghost: bool) -> tuple[torch.Tensor]:
        with torch.no_grad():
            out = hdr_merge(images, align=bool(align), deghost=bool(deghost))
        return (out.to(images.device),)
