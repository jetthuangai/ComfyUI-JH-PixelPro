"""ComfyUI wrapper for ``core.lut.identity_hald``."""

from __future__ import annotations

import torch

from ..core.lut import identity_hald

_LEVEL_CHOICES = ["4", "6", "8", "10", "12"]


class JHPixelProHALDIdentity:
    """Generate an identity HALD image to feed into a color-grade chain + LUT export."""

    CATEGORY = "ComfyUI-JH-PixelPro/color"
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("image", "level")
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls: type) -> dict:  # noqa: N802 — ComfyUI node contract mandates UPPER_CASE.
        return {
            "required": {
                "level": (
                    _LEVEL_CHOICES,
                    {
                        "default": "8",
                        "tooltip": (
                            "HALD level L. Cube N=L² (L=8→N=64 industry "
                            "standard). Image side = L³ pixels: L=4→64×64, "
                            "L=8→512×512, L=12→1728×1728. Higher L = finer "
                            "LUT but larger intermediate image."
                        ),
                    },
                ),
            },
        }

    def generate(self, level: str) -> tuple[torch.Tensor, int]:
        level_int = int(level)
        image = identity_hald(level_int)
        return (image, level_int)
