"""ComfyUI wrapper for bundled LUT presets."""

from __future__ import annotations

import torch

from ..core.lut import apply_lut_3d
from ..core.lut_preset import list_presets, load_preset

PRESET_OPTIONS = list_presets()


class JHPixelProLUTPreset:
    """Apply one of the pack-bundled Adobe Cube LUT presets."""

    CATEGORY = "ComfyUI-JH-PixelPro/color"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply"

    @classmethod
    def INPUT_TYPES(cls: type) -> dict:  # noqa: N802 — ComfyUI node contract mandates UPPER_CASE.
        presets = list_presets()
        default = "neutral-identity" if "neutral-identity" in presets else presets[0]
        return {
            "required": {
                "image": ("IMAGE",),
                "preset": (
                    presets,
                    {
                        "default": default,
                        "tooltip": "Bundled generic .cube preset from the pack presets/ directory.",
                    },
                ),
                "intensity": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Blend factor: 0 = original image, 1 = full LUT preset.",
                    },
                ),
            }
        }

    def apply(self, image: torch.Tensor, preset: str, intensity: float) -> tuple[torch.Tensor]:
        parsed = load_preset(preset)
        with torch.no_grad():
            out = apply_lut_3d(
                image,
                parsed["lut"].to(image.device),
                strength=float(intensity),
                domain_min=parsed["domain_min"].to(image.device),
                domain_max=parsed["domain_max"].to(image.device),
            )
        return (out,)
