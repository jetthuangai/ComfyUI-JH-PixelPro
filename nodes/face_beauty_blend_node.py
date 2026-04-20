"""ComfyUI wrapper for mask-aware beauty blending."""

from __future__ import annotations

import torch

from ..core.face_pipeline import beauty_blend


class JHPixelProFaceBeautyBlend:
    CATEGORY = "ComfyUI-JH-PixelPro/face"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("blended",)
    FUNCTION = "blend"

    @classmethod
    def INPUT_TYPES(cls) -> dict:  # noqa: N802
        return {
            "required": {
                "base": ("IMAGE",),
                "retouched": ("IMAGE",),
                "mask": ("MASK",),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Overall blend strength.",
                    },
                ),
                "feather": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 64,
                        "step": 1,
                        "tooltip": "Gaussian feather radius in pixels. 0 = no feather.",
                    },
                ),
            },
        }

    def blend(
        self,
        base: torch.Tensor,
        retouched: torch.Tensor,
        mask: torch.Tensor,
        strength: float,
        feather: int,
    ) -> tuple[torch.Tensor]:
        with torch.no_grad():
            output = beauty_blend(
                base,
                retouched,
                mask,
                strength=strength,
                feather=feather,
                device=base.device,
            )
        return (output,)
