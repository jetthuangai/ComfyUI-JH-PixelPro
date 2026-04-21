"""ComfyUI wrapper for selective hue / saturation adjustment within a hue band."""

from __future__ import annotations

import torch

from ..core.selective_color import apply_hue_sat_shift, hue_range_mask


class JHPixelProHueSaturationRange:
    """Adjust hue and saturation inside a selected hue band."""

    CATEGORY = "ComfyUI-JH-PixelPro/color"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply"

    @classmethod
    def INPUT_TYPES(cls: type) -> dict:  # noqa: N802
        return {
            "required": {
                "image": ("IMAGE",),
                "hue_center": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 360.0,
                        "step": 1.0,
                        "tooltip": (
                            "Hue center in degrees (0=red, 60=yellow, 120=green, "
                            "180=cyan, 240=blue, 300=magenta)."
                        ),
                    },
                ),
                "band_width": (
                    "FLOAT",
                    {
                        "default": 30.0,
                        "min": 0.0,
                        "max": 180.0,
                        "step": 1.0,
                        "tooltip": (
                            "Band half-width in degrees. 30° = narrow band, "
                            "60° = medium, 180° = full wheel."
                        ),
                    },
                ),
                "hue_shift": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -180.0,
                        "max": 180.0,
                        "step": 1.0,
                        "tooltip": "Shift hue inside the selected band by this amount (degrees).",
                    },
                ),
                "sat_mult": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": (
                            "Multiply saturation inside the band. 1.0 = unchanged, "
                            "0 = desaturate, 2 = double saturation."
                        ),
                    },
                ),
                "sat_add": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -1.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Additive saturation offset applied after multiplication.",
                    },
                ),
            },
        }

    def apply(
        self,
        image: torch.Tensor,
        hue_center: float,
        band_width: float,
        hue_shift: float,
        sat_mult: float,
        sat_add: float,
    ) -> tuple[torch.Tensor]:
        device = image.device
        with torch.no_grad():
            mask = hue_range_mask(image, hue_center, band_width, device=device)
            output = apply_hue_sat_shift(
                image,
                mask,
                hue_shift,
                sat_mult,
                sat_add,
                device=device,
            )
        return (output,)
