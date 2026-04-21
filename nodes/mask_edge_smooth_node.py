"""ComfyUI wrapper for bilateral MASK edge smoothing."""

from __future__ import annotations

import torch

from ..core.mask_edge_smooth import mask_edge_smooth


class JHPixelProMaskEdgeSmoother:
    """Smooth mask edges with optional image-guided joint bilateral filtering."""

    CATEGORY = "ComfyUI-JH-PixelPro/mask"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "smooth"

    @classmethod
    def INPUT_TYPES(cls: type) -> dict:  # noqa: N802
        return {
            "required": {
                "mask": ("MASK",),
                "sigma_spatial": (
                    "FLOAT",
                    {"default": 4.0, "min": 0.1, "max": 64.0, "step": 0.1},
                ),
                "sigma_range": (
                    "FLOAT",
                    {"default": 0.1, "min": 0.001, "max": 1.0, "step": 0.001},
                ),
                "iterations": ("INT", {"default": 1, "min": 1, "max": 5, "step": 1}),
            },
            "optional": {
                "guide": ("IMAGE",),
            },
        }

    def smooth(
        self,
        mask: torch.Tensor,
        sigma_spatial: float,
        sigma_range: float,
        iterations: int,
        guide: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor]:
        with torch.no_grad():
            out = mask_edge_smooth(
                mask,
                guide,
                sigma_spatial=sigma_spatial,
                sigma_range=sigma_range,
                iterations=iterations,
            )
        return (out,)
