"""ComfyUI wrapper for saturation-range mask extraction."""

from __future__ import annotations

import torch

from ..core.selective_color import saturation_range_mask


class JHPixelProSaturationMask:
    """Build a soft mask for pixels inside a saturation range."""

    CATEGORY = "ComfyUI-JH-PixelPro/color"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "build"

    @classmethod
    def INPUT_TYPES(cls: type) -> dict:  # noqa: N802
        return {
            "required": {
                "image": ("IMAGE",),
                "sat_min": (
                    "FLOAT",
                    {
                        "default": 0.3,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Minimum saturation included in the output mask.",
                    },
                ),
                "sat_max": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Maximum saturation included in the output mask.",
                    },
                ),
                "feather": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Soft edge width for the saturation thresholds.",
                    },
                ),
            },
        }

    def build(
        self,
        image: torch.Tensor,
        sat_min: float,
        sat_max: float,
        feather: float,
    ) -> tuple[torch.Tensor]:
        with torch.no_grad():
            mask = saturation_range_mask(
                image,
                sat_min,
                sat_max,
                feather=feather,
                device=image.device,
            )
        return (mask,)
