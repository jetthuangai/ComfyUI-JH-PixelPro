"""ComfyUI wrapper for combining two MASK tensors."""

from __future__ import annotations

import torch

from ..core.mask_combine import (
    MASK_COMBINE_BLEND_MODES,
    MASK_COMBINE_OPERATIONS,
    combine_masks,
)


class JHPixelProMaskCombine:
    """Combine two masks with add/subtract/intersect/union/difference/xor/multiply."""

    CATEGORY = "ComfyUI-JH-PixelPro/mask"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "combine"

    @classmethod
    def INPUT_TYPES(cls: type) -> dict:  # noqa: N802
        return {
            "required": {
                "mask_a": ("MASK",),
                "mask_b": ("MASK",),
                "operation": (MASK_COMBINE_OPERATIONS, {"default": "union"}),
                "blend_mode": (MASK_COMBINE_BLEND_MODES, {"default": "hard"}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "feather_sigma": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 64.0, "step": 0.1},
                ),
            },
        }

    def combine(
        self,
        mask_a: torch.Tensor,
        mask_b: torch.Tensor,
        operation: str,
        blend_mode: str,
        opacity: float,
        feather_sigma: float,
    ) -> tuple[torch.Tensor]:
        with torch.no_grad():
            out = combine_masks(
                mask_a,
                mask_b,
                operation=operation,
                blend_mode=blend_mode,
                opacity=opacity,
                feather_sigma=feather_sigma,
            )
        return (out,)
