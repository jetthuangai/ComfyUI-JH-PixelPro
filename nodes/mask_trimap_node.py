"""ComfyUI wrapper for 3-value trimap construction."""

from __future__ import annotations

import torch

from ..core.mask_trimap import build_trimap


class JHPixelProTrimapBuilder:
    """Build a 3-value MASK trimap: 0.0 BG / 0.5 Unknown / 1.0 FG (±0.05)."""

    CATEGORY = "ComfyUI-JH-PixelPro/mask"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("trimap",)
    FUNCTION = "build"

    @classmethod
    def INPUT_TYPES(cls: type) -> dict:  # noqa: N802
        return {
            "required": {
                "mask": (
                    "MASK",
                    {"tooltip": "Binary or soft MASK used to derive a 3-value trimap."},
                ),
                "fg_radius": ("INT", {"default": 4, "min": 1, "max": 64, "step": 1}),
                "bg_radius": ("INT", {"default": 8, "min": 1, "max": 64, "step": 1}),
                "smoothing": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 16.0, "step": 0.1}),
            },
        }

    def build(
        self,
        mask: torch.Tensor,
        fg_radius: int,
        bg_radius: int,
        smoothing: float,
    ) -> tuple[torch.Tensor]:
        with torch.no_grad():
            trimap = build_trimap(
                mask,
                fg_radius=fg_radius,
                bg_radius=bg_radius,
                smoothing=smoothing,
            )
        return (trimap,)
