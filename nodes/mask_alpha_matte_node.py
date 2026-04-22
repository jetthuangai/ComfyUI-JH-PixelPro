"""ComfyUI wrapper for Levin 2008 closed-form alpha matte extraction."""

from __future__ import annotations

import torch

from ..core.mask_alpha_matte import alpha_matte_extract


class JHPixelProAlphaMatteExtractor:
    """Extract soft alpha from a 3-value trimap via Levin 2008 matting."""

    CATEGORY = "ComfyUI-JH-PixelPro/mask"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("alpha",)
    FUNCTION = "extract"

    @classmethod
    def INPUT_TYPES(cls: type) -> dict:  # noqa: N802
        return {
            "required": {
                "trimap": (
                    "MASK",
                    {
                        "tooltip": (
                            "3-value trimap MASK: 0.0 background, 0.5 unknown, "
                            "1.0 foreground. Tolerance ±0.05."
                        )
                    },
                ),
                "guide": ("IMAGE",),
                "epsilon": (
                    "FLOAT",
                    {"default": 0.0000001, "min": 0.00000001, "max": 0.01, "step": 0.0000001},
                ),
                "window_radius": ("INT", {"default": 1, "min": 1, "max": 3, "step": 1}),
                "lambda_constraint": (
                    "FLOAT",
                    {"default": 100.0, "min": 1.0, "max": 10000.0, "step": 1.0},
                ),
            },
        }

    def extract(
        self,
        trimap: torch.Tensor,
        guide: torch.Tensor,
        epsilon: float,
        window_radius: int,
        lambda_constraint: float = 100.0,
    ) -> tuple[torch.Tensor]:
        with torch.no_grad():
            alpha = alpha_matte_extract(
                trimap,
                guide,
                epsilon=epsilon,
                window_radius=window_radius,
                lambda_constraint=lambda_constraint,
            )
        return (alpha,)
