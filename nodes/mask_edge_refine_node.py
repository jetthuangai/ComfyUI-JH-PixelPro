"""ComfyUI wrapper for guided-filter edge-aware mask refinement."""

from __future__ import annotations

import torch

from ..core.mask_edge_refine import edge_aware_refine


class JHPixelProEdgeAwareMaskRefiner:
    """Refine a MASK using an IMAGE guide so mask edges follow image detail."""

    CATEGORY = "ComfyUI-JH-PixelPro/mask"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("refined_mask",)
    FUNCTION = "refine"

    @classmethod
    def INPUT_TYPES(cls: type) -> dict:  # noqa: N802
        return {
            "required": {
                "mask": ("MASK",),
                "guide": ("IMAGE",),
                "radius": ("INT", {"default": 8, "min": 1, "max": 32, "step": 1}),
                "eps": (
                    "FLOAT",
                    {"default": 0.001, "min": 0.000001, "max": 1.0, "step": 0.0001},
                ),
                "feather_sigma": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 64.0, "step": 0.1},
                ),
            },
        }

    def refine(
        self,
        mask: torch.Tensor,
        guide: torch.Tensor,
        radius: int,
        eps: float,
        feather_sigma: float,
    ) -> tuple[torch.Tensor]:
        with torch.no_grad():
            refined = edge_aware_refine(
                mask,
                guide,
                radius=radius,
                eps=eps,
                feather_sigma=feather_sigma,
            )
        return (refined,)
