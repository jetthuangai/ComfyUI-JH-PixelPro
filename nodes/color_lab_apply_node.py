"""ComfyUI wrapper: apply a ColorLab ``params_json`` preset with one strength slider."""

from __future__ import annotations

import torch

from ..core.color_lab import apply_preset


class JHPixelProColorLabApply:
    """Re-apply ColorLab settings (from a ColorLab or Lightroom Preset node) to any image."""

    CATEGORY = "ComfyUI-JH-PixelPro/color"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "params_json")
    OUTPUT_TOOLTIPS = ("Image with the preset applied.", "The preset at the chosen strength.")
    FUNCTION = "apply"

    @classmethod
    def INPUT_TYPES(cls: type) -> dict:  # noqa: N802
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image to grade."}),
                "params_json": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Preset from a ColorLab or Lightroom Preset node.",
                    },
                ),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Scales every amount (grading hues are kept). 0 = original.",
                    },
                ),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Optional mask limiting where the preset applies."}),
            },
        }

    def apply(
        self,
        image: torch.Tensor,
        params_json: str,
        strength: float,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, str]:
        out, preset = apply_preset(image, params_json, float(strength))
        if float(strength) == 0.0:
            return image, preset
        if mask is not None:
            if mask.ndim != 3 or mask.shape[1:] != image.shape[1:3]:
                raise ValueError("mask must have shape (B,H,W) matching the image.")
            weight = mask.to(device=out.device, dtype=out.dtype).clamp(0.0, 1.0).unsqueeze(-1)
            out = image + (out - image) * weight
        return out, preset
