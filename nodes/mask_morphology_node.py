"""ComfyUI wrapper for classical MASK morphology operations."""

from __future__ import annotations

import torch

from ..core.mask_morphology import MORPHOLOGY_OPERATIONS, mask_morphology


class JHPixelProMaskMorphology:
    """Grow, shrink, clean, or extract MASK edges using morphology kernels."""

    CATEGORY = "ComfyUI-JH-PixelPro/mask"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "apply"

    @classmethod
    def INPUT_TYPES(cls: type) -> dict:  # noqa: N802
        return {
            "required": {
                "mask": ("MASK",),
                "operation": (MORPHOLOGY_OPERATIONS, {"default": "dilate"}),
                "radius": ("INT", {"default": 3, "min": 1, "max": 64, "step": 1}),
                "iterations": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
            },
        }

    def apply(
        self,
        mask: torch.Tensor,
        operation: str,
        radius: int,
        iterations: int,
    ) -> tuple[torch.Tensor]:
        with torch.no_grad():
            out = mask_morphology(
                mask,
                operation=operation,
                radius=radius,
                iterations=iterations,
            )
        return (out,)
